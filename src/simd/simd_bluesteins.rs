//! The body of the SIMD Bluestein's implementations, shared by every SIMD backend.
//!
//! Everything here is generic over `SimdVector`, from `simd_vector.rs`. A backend implements that
//! trait once per vector type and gets the algorithm, so this is the only copy of it.
//!
//! The algorithm is the same as the scalar `BluesteinsAlgorithm`: two inner FFTs with a pairwise
//! complex multiply before, between and after them. Only those three multiply loops are SIMD, and
//! they run through `SimdVector`'s `mul_complex_*_slice` methods so that they land inside the
//! backend's target feature; see `SimdVector::cross_layer_radixn` for why that matters.
//!
//! The twiddles and the precomputed inner FFT are stored as plain complex numbers, loaded a vector
//! at a time by the multiply loops. Neither length has to be a multiple of `COMPLEX_PER_VECTOR`:
//! since no backend holds more than two complex numbers in a vector, an odd length leaves exactly
//! one element over, and the multiply loops handle it with a partial load and store.

use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use crate::array_utils::{workaround_transmute, workaround_transmute_mut};
use crate::common::FftNum;
use crate::{twiddles, Direction, Fft, FftDirection, Length};

use super::simd_vector::SimdVector;

/// Implementation of Bluestein's Algorithm, SIMD accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub struct SimdBluesteins<V: SimdVector, T> {
    inner_fft: Arc<dyn Fft<T>>,

    /// The FFT of the Bluestein's twiddles, conjugated. The conjugation is what turns the
    /// `(x * multiplier).conj()` of the scalar version into a single `mul_complex_conjugated`.
    inner_fft_multiplier: Box<[Complex<T>]>,
    /// The twiddles applied on the way in and on the way out, not conjugated.
    twiddles: Box<[Complex<T>]>,

    inner_fft_len: usize,
    len: usize,
    direction: FftDirection,

    scratch_len: usize,

    _phantom: PhantomData<V>,
}

impl<V: SimdVector, T: FftNum> SimdBluesteins<V, T> {
    /// Creates a FFT instance which will process inputs/outputs of size `len`.
    /// `inner_fft.len()` must be >= `len * 2 - 1`.
    ///
    /// # Panics
    /// Panics if `inner_fft.len() < len * 2 - 1`.
    pub fn new(len: usize, inner_fft: Arc<dyn Fft<T>>) -> Self {
        // Internal sanity check: Make sure that the vector's scalar type is T.
        // This struct has two generic parameters V and T, but T must always be V's scalar type,
        // and they are only kept separate to help work around the lack of specialization.
        assert_eq!(TypeId::of::<V::ScalarType>(), TypeId::of::<T>());

        let inner_fft_len = inner_fft.len();
        assert!(len * 2 - 1 <= inner_fft_len, "Bluestein's algorithm requires inner_fft.len() >= self.len() * 2 - 1. Expected >= {}, got {}", len * 2 - 1, inner_fft_len);

        let direction = inner_fft.fft_direction();

        // when computing FFTs, we're going to run our inner multiply pairwise by some precomputed
        // data, then run an inverse inner FFT. We need to precompute that inner data here
        let inner_fft_scale = T::one() / T::from_usize(inner_fft_len).unwrap();

        // Compute twiddle factors that we'll run our inner FFT on
        let mut inner_fft_input = vec![Complex::zero(); inner_fft_len];
        twiddles::fill_bluesteins_twiddles(
            &mut inner_fft_input[..len],
            direction.opposite_direction(),
        );

        // Scale the computed twiddles and copy them to the end of the array
        inner_fft_input[0] = inner_fft_input[0] * inner_fft_scale;
        for i in 1..len {
            let twiddle = inner_fft_input[i] * inner_fft_scale;
            inner_fft_input[i] = twiddle;
            inner_fft_input[inner_fft_len - i] = twiddle;
        }

        //Compute the inner fft
        let mut inner_fft_scratch = vec![Complex::zero(); inner_fft.get_inplace_scratch_len()];
        inner_fft.process_with_scratch(&mut inner_fft_input, &mut inner_fft_scratch);

        // When computing the FFT we want this array conjugated, so conjugate it now
        for multiplier in inner_fft_input.iter_mut() {
            *multiplier = multiplier.conj();
        }

        // also compute some more mundane twiddle factors to start and end with
        let mut twiddles = vec![Complex::zero(); len];
        twiddles::fill_bluesteins_twiddles(&mut twiddles, direction);

        Self {
            inner_fft,

            inner_fft_multiplier: inner_fft_input.into_boxed_slice(),
            twiddles: twiddles.into_boxed_slice(),

            inner_fft_len,
            len,
            direction,

            scratch_len: inner_fft_len + inner_fft_scratch.len(),

            _phantom: PhantomData,
        }
    }

    /// Copy the input into the inner FFT's buffer, applying the twiddles and zeroing the rest.
    #[inline(always)]
    fn prepare(&self, input: &[Complex<T>], inner_input: &mut [Complex<T>]) {
        let (twiddled, zeroes) = inner_input.split_at_mut(self.len);
        unsafe {
            V::mul_complex_slice(
                workaround_transmute(input),
                workaround_transmute_mut(twiddled),
                workaround_transmute(&self.twiddles),
            );
        }
        // The buffer only fills part of the inner FFT input, so zero fill the rest. This is half
        // the inner length or more, so it's left as a plain fill for the memset.
        zeroes.fill(Complex::zero());
    }

    /// Copy the inner FFT's output back out, conjugating it to complete the inverse FFT and
    /// applying the twiddles again.
    #[inline(always)]
    fn finalize(&self, inner_input: &[Complex<T>], output: &mut [Complex<T>]) {
        unsafe {
            V::mul_complex_conjugated_slice(
                workaround_transmute(inner_input),
                workaround_transmute_mut(output),
                workaround_transmute(&self.twiddles),
            );
        }
    }

    /// The two inner FFTs, with the multiply by the precomputed data in between.
    #[inline(always)]
    fn transform_inner(&self, inner_input: &mut [Complex<T>], inner_scratch: &mut [Complex<T>]) {
        // run our inner forward FFT
        self.inner_fft
            .process_with_scratch(inner_input, inner_scratch);

        // Multiply by the precomputed data, and conjugate the result to set up for an inverse FFT.
        // The multiplier is stored pre-conjugated, so one conjugating multiply does both.
        unsafe {
            V::mul_complex_conjugated_slice_inplace(
                workaround_transmute_mut(inner_input),
                workaround_transmute(&self.inner_fft_multiplier),
            );
        }

        // inverse FFT. we're computing a forward but we're massaging it into an inverse by
        // conjugating the inputs and outputs
        self.inner_fft
            .process_with_scratch(inner_input, inner_scratch);
    }

    #[inline(always)]
    fn perform_fft_immut(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_len);

        self.prepare(input, inner_input);
        self.transform_inner(inner_input, inner_scratch);
        self.finalize(inner_input, output);
    }

    #[inline(always)]
    fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.perform_fft_immut(input, output, scratch);
    }

    /// The in-place case is the out-of-place one with the input and the output being the same
    /// buffer: the inner FFT runs entirely within the scratch, `prepare` only reads the buffer and
    /// `finalize` only writes it, with both inner FFTs in between.
    #[inline(always)]
    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_len);

        self.prepare(buffer, inner_input);
        self.transform_inner(inner_input, inner_scratch);
        self.finalize(inner_input, buffer);
    }
}

boilerplate_simd_fft!(
    SimdBluesteins,
    |this: &SimdBluesteins<_, _>| this.len,
    |this: &SimdBluesteins<_, _>| this.scratch_len,
    |this: &SimdBluesteins<_, _>| this.scratch_len,
    |this: &SimdBluesteins<_, _>| this.scratch_len
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

    /// Every inner length from the smallest legal one up past the next power of two, for a range of
    /// FFT lengths. Covers both parities of the inner length, so for the f32 vectors it covers the
    /// partial load and store in the multiply loops.
    pub fn inner_lengths<V>()
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        for len in 2..16 {
            let min_inner: usize = len * 2 - 1;
            let max_inner = min_inner.checked_next_power_of_two().unwrap();

            for inner_len in min_inner..=max_inner {
                check::<V>(len, inner_len, FftDirection::Forward);
                check::<V>(len, inner_len, FftDirection::Inverse);
            }
        }
    }

    fn check<V>(len: usize, inner_len: usize, direction: FftDirection)
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        let inner_fft = Arc::new(Dft::new(inner_len, direction));
        let fft: SimdBluesteins<V, V::ScalarType> = SimdBluesteins::new(len, inner_fft);

        check_fft_algorithm::<V::ScalarType>(&fft, len, direction);
    }
}
