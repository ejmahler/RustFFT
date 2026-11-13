//! The body of the SIMD Rader's implementations, shared by every SIMD backend.
//!
//! Everything here is generic over `SimdVector`, from `simd_vector.rs`. A backend implements that
//! trait once per vector type and gets the algorithm, so this is the only copy of it.
//!
//! Only the pairwise multiply between the two inner FFTs is SIMD, through `mul_complex_slice`. The
//! input and output reorderings stay scalar: both are permuted copies, and without a gather
//! instruction there is nothing for a vector to do.

use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::Complex;
use num_integer::Integer;
use num_traits::Zero;
use primal_check::miller_rabin;
use strength_reduce::StrengthReducedU64;

use crate::array_utils::{DoubleBuf, workaround_transmute, workaround_transmute_mut};
use crate::common::FftNum;
use crate::math_utils;
use crate::simd::simd_mul_slice::mul_complex_slice;
use crate::simd::simd_vector::SimdVector;
use crate::{Fft, FftDirection, twiddles};

/// The Rader's input reordering table, stored in the narrowest integer type that holds every
/// offset. The table is read once per pass alongside the data, so a narrower type is less memory
/// traffic, and a smaller cache footprint for the data to share.
enum Permutation {
    U16(Box<[u16]>),
    U32(Box<[u32]>),
    Usize(Box<[usize]>),
}

impl Permutation {
    /// Picks the narrowest type for offsets `0..=max_index`, and collects `indices` into it.
    fn new(max_index: usize, indices: impl Iterator<Item = usize>) -> Self {
        if u16::try_from(max_index).is_ok() {
            Self::U16(indices.map(|index| index as u16).collect())
        } else if u32::try_from(max_index).is_ok() {
            Self::U32(indices.map(|index| index as u32).collect())
        } else {
            Self::Usize(indices.collect())
        }
    }
}

/// An element type for a `Permutation` table.
trait PermutationIndex: Copy {
    fn to_usize(self) -> usize;
}
impl PermutationIndex for u16 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}
impl PermutationIndex for u32 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}
impl PermutationIndex for usize {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self
    }
}

/// The body of `SimdRaders::gather_input`, once per index type.
#[inline(always)]
fn gather<T: Copy, I: PermutationIndex>(permutation: &[I], src: &[T], dest: &mut [T]) {
    for (dest_element, &input_index) in dest.iter_mut().zip(permutation.iter()) {
        *dest_element = src[input_index.to_usize()];
    }
}

/// The body of `SimdRaders::scatter_output`, once per index type.
#[inline(always)]
fn scatter<T: FftNum, I: PermutationIndex>(
    permutation: &[I],
    src: &[Complex<T>],
    dest: &mut [Complex<T>],
) {
    let (&last_index, head) = permutation.split_last().unwrap();
    let (last_element, src_head) = src.split_last().unwrap();

    for (src_element, &output_index) in src_head.iter().zip(head.iter().rev()) {
        dest[output_index.to_usize()] = src_element.conj();
    }
    dest[last_index.to_usize()] = last_element.conj();
}

/// Rader's Algorithm, generic over SIMD instruction set.
/// This is designed to be embedded into a feature-specific struct (such as `SseRaders` or
/// `NeonRaders`), and not created directly.
pub struct SimdRaders<V: SimdVector, T> {
    inner_fft: Arc<dyn Fft<T>>,

    /// The FFT of the reordered twiddles, conjugated. The conjugation is what turns the
    /// `(x * twiddle).conj()` between the two inner FFTs into a single `mul_complex_conjugated`.
    inner_fft_data: Box<[Complex<T>]>,

    // Buffer offsets for the input reordering: inner FFT input k reads buffer position
    // permutation[k], which is g^(k+1) mod len, less one. The sequence depends only on the
    // primitive root and the length, so building it once here keeps a serial chain of
    // strength-reduced modular multiplies out of the hot loops. avx_raders.rs precomputes its
    // output mapping for the same reason.
    permutation: Permutation,

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
        assert!(
            miller_rabin(len as u64),
            "For raders algorithm, inner_fft.len() + 1 must be prime. Expected prime number, got {} + 1 = {}",
            inner_fft_len,
            len
        );

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
        // plain indexed load with no arithmetic, which puts the offsets in 0..inner_fft_len.
        let input_indices = (0..inner_fft_len).scan(1u64, |input_index, _| {
            *input_index = (*input_index * primitive_root) % reduced_len;
            Some((*input_index - 1) as usize)
        });
        let permutation = Permutation::new(inner_fft_len - 1, input_indices);

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

            permutation,

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
        match &self.permutation {
            Permutation::U16(permutation) => gather(permutation, src, dest),
            Permutation::U32(permutation) => gather(permutation, src, dest),
            Permutation::Usize(permutation) => gather(permutation, src, dest),
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
        match &self.permutation {
            Permutation::U16(permutation) => scatter(permutation, src, dest),
            Permutation::U32(permutation) => scatter(permutation, src, dest),
            Permutation::Usize(permutation) => scatter(permutation, src, dest),
        }
    }

    /// Multiply the inner result by the cached setup data, conjugating as we go to set up for an
    /// inverse FFT. The cached data is stored pre-conjugated, so one conjugating multiply does
    /// both.
    #[inline(always)]
    fn multiply_inner_inplace(&self, buffer: &mut [Complex<T>]) {
        unsafe {
            mul_complex_slice::<V, &mut [Complex<V::ScalarType>], true>(
                workaround_transmute_mut(buffer),
                workaround_transmute(&self.inner_fft_data),
            );
        }
    }

    /// `multiply_inner_inplace`, writing the result somewhere else.
    #[inline(always)]
    fn multiply_inner(&self, src: &[Complex<T>], dest: &mut [Complex<T>]) {
        unsafe {
            mul_complex_slice::<V, _, true>(
                DoubleBuf {
                    input: workaround_transmute(src),
                    output: workaround_transmute_mut(dest),
                },
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

    #[inline(always)]
    pub fn process_single_immutable(
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

    #[inline(always)]
    pub fn process_single_outofplace(
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
        let inner_scratch = if !scratch.is_empty() {
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
        let inner_scratch = if !scratch.is_empty() {
            scratch
        } else {
            &mut output[..]
        };
        self.inner_fft.process_with_scratch(input, inner_scratch);

        // copy the final values into the output, reordering as we go
        self.scatter_output(input, output);
    }

    #[inline(always)]
    pub fn process_single(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
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

    #[inline(always)]
    pub fn get_inplace_scratch_len(&self) -> usize {
        self.inplace_scratch_len
    }
    #[inline(always)]
    pub fn get_outofplace_scratch_len(&self) -> usize {
        self.outofplace_scratch_len
    }
    #[inline(always)]
    pub fn get_immutable_scratch_len(&self) -> usize {
        self.immut_scratch_len
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    pub fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::FftPlannerScalar;
    use crate::algorithm::BluesteinsAlgorithm;
    use crate::test_utils::{compare_vectors, random_signal};

    /// Both sides of the u16 to u32 switch in `Permutation`: 65537 is the largest prime whose
    /// offsets all fit in a u16, and 65539 is the next prime. A DFT reference is too slow at these
    /// lengths, so the reference is Bluestein's. Only the reordering depends on the index type, so
    /// the scalar vector type covers every backend.
    #[test]
    fn index_type_boundary() {
        let direction = FftDirection::Forward;
        let mut planner = FftPlannerScalar::<f64>::new();
        for (len, expect_u16) in [(65537, true), (65539, false)] {
            let raders = SimdRaders::<Complex<f64>, f64>::new(planner.plan_fft(len - 1, direction));
            match raders.permutation {
                Permutation::U16(_) => assert!(expect_u16, "len {len} should not use u16"),
                Permutation::U32(_) => assert!(!expect_u16, "len {len} should use u16"),
                Permutation::Usize(_) => panic!("len {len} should fit in u32"),
            }

            let bluesteins = BluesteinsAlgorithm::new(len, planner.plan_fft(1 << 18, direction));
            let mut expected = random_signal::<f64>(len);
            let mut actual = expected.clone();
            bluesteins.process(&mut expected);

            let mut scratch = vec![Complex::zero(); raders.get_inplace_scratch_len()];
            raders.process_single(&mut actual, &mut scratch);
            assert!(compare_vectors(&expected, &actual), "len {len} is wrong");
        }
    }
}

/// The test bodies, shared the same way the algorithm is. Every backend runs all of them, from a
/// thin test function per body that names its own vector types.
#[cfg(test)]
pub mod test_bodies {
    use super::*;
    use crate::algorithm::Dft;
    use crate::test_utils::check_fft_algorithm;
    use num_traits::Float;
    use rand::distr::uniform::SampleUniform;

    type NewFn<T> = dyn Fn(Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>>;

    /// Every prime length up to 100, both directions. The inner length is one less than a prime, so
    /// it is always even and never exercises the partial load and store in the multiply loop; the
    /// Bluestein's tests cover that path.
    pub fn prime_lengths<V>(new_fn: &NewFn<V::ScalarType>)
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        for len in 3..100 {
            if miller_rabin(len as u64) {
                for direction in [FftDirection::Forward, FftDirection::Inverse] {
                    let inner_fft = Arc::new(Dft::new(len - 1, direction));
                    let fft = new_fn(inner_fft);
                    check_fft_algorithm::<V::ScalarType>(fft.as_ref(), len, direction);
                }
            }
        }
    }
}
