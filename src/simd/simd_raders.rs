//! The body of the SIMD Rader's implementations, shared by every SIMD backend.
//!
//! Everything here is generic over `SimdVector`, from `simd_vector.rs`. A backend implements that
//! trait once per vector type and gets the algorithm, so this is the only copy of it.
//!
//! Only the pairwise multiply between the two inner FFTs is SIMD, through `SimdVector`'s
//! `mul_complex_conjugated_slice*` methods so that it lands inside the backend's target feature;
//! see `SimdVector::cross_layer_radixn` for why that matters. The input and output reorderings
//! stay scalar: both are permuted copies, and without a gather instruction there is nothing for a
//! vector to do.

use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::Complex;
use num_integer::Integer;
use num_traits::Zero;
use primal_check::miller_rabin;
use strength_reduce::StrengthReducedU64;

use crate::array_utils::{workaround_transmute, workaround_transmute_mut};
use crate::common::FftNum;
use crate::math_utils;
use crate::{twiddles, Direction, Fft, FftDirection, Length};

use super::simd_vector::SimdVector;

/// Implementation of Rader's Algorithm, SIMD accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub struct SimdRaders<V: SimdVector, T> {
    inner_fft: Arc<dyn Fft<T>>,

    /// The FFT of the reordered twiddles, conjugated. The conjugation is what turns the
    /// `(x * twiddle).conj()` of the scalar version into a single `mul_complex_conjugated`.
    inner_fft_data: Box<[Complex<T>]>,

    // Buffer offsets for the input reordering: inner FFT input k reads buffer position
    // permutation[k], which is g^(k+1) mod len, less one. The sequence depends only on the
    // primitive root and the length, so building it once here keeps a serial chain of
    // strength-reduced modular multiplies out of the hot loops. avx_raders.rs precomputes its
    // output mapping for the same reason. This costs 4 * (len - 1) bytes, the same order as the
    // twiddles avx_raders.rs and Bluestein's already store at this size.
    permutation: Box<[u32]>,

    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,

    direction: FftDirection,

    _phantom: PhantomData<V>,
}

impl<V: SimdVector, T: FftNum> SimdRaders<V, T> {
    /// Creates a FFT instance which will process inputs/outputs of size `inner_fft.len() + 1`.
    ///
    /// # Panics
    /// Panics if `inner_fft.len() + 1` is not a prime number.
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        // Internal sanity check: Make sure that the vector's scalar type is T.
        // This struct has two generic parameters V and T, but T must always be V's scalar type,
        // and they are only kept separate to help work around the lack of specialization.
        assert_eq!(TypeId::of::<V::ScalarType>(), TypeId::of::<T>());

        let inner_fft_len = inner_fft.len();
        let len = inner_fft_len + 1;
        assert!(miller_rabin(len as u64), "For raders algorithm, inner_fft.len() + 1 must be prime. Expected prime number, got {} + 1 = {}", inner_fft_len, len);

        let direction = inner_fft.fft_direction();
        let reduced_len = StrengthReducedU64::new(len as u64);

        // compute the primitive root and its inverse for this size
        let primitive_root = math_utils::primitive_root(len as u64).unwrap();

        // compute the multiplicative inverse of primative_root mod len and vice versa.
        // i64::extended_gcd will compute both the inverse of left mod right, and the inverse of right mod left, but we're only goingto use one of them
        // the primtive root inverse might be negative, if o make it positive by wrapping
        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(len as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + len as i64
        } as u64;

        // precompute the coefficients to use inside the process method
        let inner_fft_scale = T::one() / T::from_usize(inner_fft_len).unwrap();
        let mut inner_fft_input = vec![Complex::zero(); inner_fft_len];

        // primitive_root() above only returns Some for len > 2, so len is an odd prime here and
        // inner_fft_len = len - 1 is even. That makes the multiplicative group mod len cyclic of
        // even order, and in a cyclic group of even order, g^(order/2) is the unique element of
        // order 2, which mod a prime is -1. That holds for primitive_root_inverse just as much
        // as for primitive_root, so the twiddle_input sequence e_p = primitive_root_inverse^p
        // mod len satisfies e_{p + (len-1)/2} == len - e_p. Since compute_twiddle(len - x, ..)
        // is the complex conjugate of compute_twiddle(x, ..), the second half of this array is
        // just the conjugate of the first half: only the first half needs an actual
        // compute_twiddle (trig) call and a step through the modular-multiply chain. Idea from
        // https://github.com/ejmahler/RustFFT/pull/178#discussion_r3995689422
        let (first_half, second_half) = inner_fft_input.split_at_mut(inner_fft_len / 2);
        let mut twiddle_input = 1;
        for (input_cell, conjugate_input_cell) in first_half.iter_mut().zip(second_half) {
            let twiddle = twiddles::compute_twiddle(twiddle_input, len, direction);
            *input_cell = twiddle * inner_fft_scale;
            *conjugate_input_cell = input_cell.conj();

            twiddle_input =
                ((twiddle_input as u64 * primitive_root_inverse) % reduced_len) as usize;
        }

        // Precompute the input reordering. Stored already offset by one so the hot loops are a
        // plain indexed load with no arithmetic.
        let mut permutation = Vec::with_capacity(inner_fft_len);
        let mut input_index = 1u64;
        for _ in 0..inner_fft_len {
            input_index = (input_index * primitive_root) % reduced_len;
            permutation.push(u32::try_from(input_index - 1).unwrap());
        }

        let required_inner_scratch = inner_fft.get_inplace_scratch_len();
        let extra_inner_scratch = if required_inner_scratch <= inner_fft_len {
            0
        } else {
            required_inner_scratch
        };
        let inplace_scratch_len = inner_fft_len + extra_inner_scratch;
        let immut_scratch_len = inner_fft_len + required_inner_scratch;

        //precompute a FFT of our reordered twiddle factors
        let mut inner_fft_scratch = vec![Zero::zero(); required_inner_scratch];
        inner_fft.process_with_scratch(&mut inner_fft_input, &mut inner_fft_scratch);

        // When computing the FFT we want this array conjugated, so conjugate it now
        for twiddle in inner_fft_input.iter_mut() {
            *twiddle = twiddle.conj();
        }

        Self {
            inner_fft,
            inner_fft_data: inner_fft_input.into_boxed_slice(),

            permutation: permutation.into_boxed_slice(),

            len,
            inplace_scratch_len,
            outofplace_scratch_len: extra_inner_scratch,
            immut_scratch_len,
            direction,

            _phantom: PhantomData,
        }
    }

    /// Copies `src` into `dest`, applying the input reordering.
    #[inline]
    fn gather_input(&self, src: &[Complex<T>], dest: &mut [Complex<T>]) {
        for (dest_element, &input_index) in dest.iter_mut().zip(self.permutation.iter()) {
            *dest_element = src[input_index as usize];
        }
    }

    /// Copies `src` into `dest`, conjugating and applying the output reordering.
    ///
    /// The output reordering walks `g^-1` where the input reordering walks `g`. Since
    /// `g^(len - 1) == 1` we have `g^-k == g^(len - 1 - k)`, so the inverse sequence is the
    /// forward table read backwards and rotated by one. Peeling the rotated element off keeps
    /// the loop a plain zip over two slices instead of a chained iterator.
    #[inline]
    fn scatter_output(&self, src: &[Complex<T>], dest: &mut [Complex<T>]) {
        let (&last_index, head) = self.permutation.split_last().unwrap();
        let (last_element, src_head) = src.split_last().unwrap();

        for (src_element, &output_index) in src_head.iter().zip(head.iter().rev()) {
            dest[output_index as usize] = src_element.conj();
        }
        dest[last_index as usize] = last_element.conj();
    }

    /// Multiply the inner result by the cached setup data, conjugating as we go to set up for an
    /// inverse FFT. The cached data is stored pre-conjugated, so one conjugating multiply does
    /// both.
    #[inline(always)]
    fn multiply_inner_inplace(&self, buffer: &mut [Complex<T>]) {
        unsafe {
            V::mul_complex_conjugated_slice_inplace(
                workaround_transmute_mut(buffer),
                workaround_transmute(&self.inner_fft_data),
            );
        }
    }

    /// `multiply_inner_inplace`, writing the result somewhere else.
    #[inline(always)]
    fn multiply_inner(&self, src: &[Complex<T>], dest: &mut [Complex<T>]) {
        unsafe {
            V::mul_complex_conjugated_slice(
                workaround_transmute(src),
                workaround_transmute_mut(dest),
                workaround_transmute(&self.inner_fft_data),
            );
        }
    }

    /// Everything after the input reordering, shared by the immutable and in-place paths: the two
    /// inner FFTs with the multiply in between, then the output reordering.
    ///
    /// `inner_scratch` is `None` when the caller has none to spare, which the in-place path hits
    /// whenever the inner FFT needs no more than `len - 1`. The output is only written by the final
    /// scatter, so it stands in until then.
    #[inline(always)]
    fn transform_and_scatter(
        &self,
        first_input: Complex<T>,
        scratch: &mut [Complex<T>],
        inner_scratch: Option<&mut [Complex<T>]>,
        output_first: &mut Complex<T>,
        output: &mut [Complex<T>],
    ) {
        let inner_scratch = match inner_scratch {
            Some(inner_scratch) => inner_scratch,
            None => &mut *output,
        };

        // perform the first of two inner FFTs
        self.inner_fft
            .process_with_scratch(scratch, &mut inner_scratch[..]);

        // scratch[0] now contains the sum of elements 1..len. We need the sum of all elements, so all we have to do is add the first input
        *output_first = first_input + scratch[0];

        self.multiply_inner_inplace(scratch);

        // We need to add the first input value to all output values. We can accomplish this by adding it to the DC input of our inner ifft.
        // Of course, we have to conjugate it, just like we conjugated the complex multiplied above
        scratch[0] = scratch[0] + first_input.conj();

        // execute the second FFT
        self.inner_fft
            .process_with_scratch(scratch, &mut inner_scratch[..]);

        // copy the final values into the output, reordering as we go
        self.scatter_output(scratch, output);
    }

    fn perform_fft_immut(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        // The first output element is just the sum of all the input elements, and we need to store off the first input value
        let (output_first, output) = output.split_first_mut().unwrap();
        let (input_first, input) = input.split_first().unwrap();
        let (scratch, extra_scratch) = scratch.split_at_mut(self.len - 1);

        // copy the input into the scratch space, reordering as we go
        self.gather_input(input, scratch);

        self.transform_and_scatter(
            *input_first,
            scratch,
            Some(extra_scratch),
            output_first,
            output,
        );
    }

    fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        // The first output element is just the sum of all the input elements, and we need to store off the first input value
        let (output_first, output) = output.split_first_mut().unwrap();
        let (input_first, input) = input.split_first_mut().unwrap();

        // copy the input into the output, reordering as we go
        self.gather_input(input, output);

        // perform the first of two inner FFTs
        let inner_scratch = if scratch.len() > 0 {
            &mut scratch[..]
        } else {
            &mut input[..]
        };
        self.inner_fft.process_with_scratch(output, inner_scratch);

        // output[0] now contains the sum of elements 1..len. We need the sum of all elements, so all we have to do is add the first input
        *output_first = *input_first + output[0];

        // The inner FFT's output is in `output` and the second inner FFT runs out of `input`, so
        // this multiply crosses from one to the other.
        self.multiply_inner(output, input);

        // We need to add the first input value to all output values. We can accomplish this by adding it to the DC input of our inner ifft.
        // Of course, we have to conjugate it, just like we conjugated the complex multiplied above
        input[0] = input[0] + input_first.conj();

        // execute the second FFT
        let inner_scratch = if scratch.len() > 0 {
            scratch
        } else {
            &mut output[..]
        };
        self.inner_fft.process_with_scratch(input, inner_scratch);

        // copy the final values into the output, reordering as we go
        self.scatter_output(input, output);
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // The first output element is just the sum of all the input elements, and we need to store off the first input value
        let (buffer_first, buffer) = buffer.split_first_mut().unwrap();
        let buffer_first_val = *buffer_first;

        let (scratch, extra_scratch) = scratch.split_at_mut(self.len - 1);

        // copy the buffer into the scratch, reordering as we go
        self.gather_input(buffer, scratch);

        // the buffer is free until the final scatter, so it can stand in as the inner FFT's
        // scratch when we weren't given any of our own
        let inner_scratch = if extra_scratch.is_empty() {
            None
        } else {
            Some(extra_scratch)
        };
        self.transform_and_scatter(
            buffer_first_val,
            scratch,
            inner_scratch,
            buffer_first,
            buffer,
        );
    }
}

boilerplate_simd_fft!(
    SimdRaders,
    |this: &SimdRaders<_, _>| this.len,
    |this: &SimdRaders<_, _>| this.inplace_scratch_len,
    |this: &SimdRaders<_, _>| this.outofplace_scratch_len,
    |this: &SimdRaders<_, _>| this.immut_scratch_len
);

/// The test bodies, shared the same way the algorithm is. Every backend runs all of them, from a
/// thin test function per body that names its own vector types.
#[cfg(test)]
pub mod test_bodies {
    use super::*;
    use crate::algorithm::Dft;
    use crate::test_utils::check_fft_algorithm;
    use num_traits::Float;
    use rand::distr::uniform::SampleUniform;

    /// Every prime length up to 100, both directions. The inner length is one less than a prime, so
    /// it is always even and never exercises the partial load and store in the multiply loop; the
    /// Bluestein's tests cover that path.
    pub fn prime_lengths<V>()
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        for len in 3..100 {
            if miller_rabin(len as u64) {
                check::<V>(len, FftDirection::Forward);
                check::<V>(len, FftDirection::Inverse);
            }
        }
    }

    fn check<V>(len: usize, direction: FftDirection)
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        let inner_fft = Arc::new(Dft::new(len - 1, direction));
        let fft: SimdRaders<V, V::ScalarType> = SimdRaders::new(inner_fft);

        check_fft_algorithm::<V::ScalarType>(&fft, len, direction);
    }
}
