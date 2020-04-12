use std::sync::Arc;
use std::arch::x86_64::*;

use num_complex::Complex;
use num_traits::Zero;

use common::FFTnum;

use ::{Length, IsInverse, Fft};

use super::avx32_utils::{AvxComplexArrayf32, AvxComplexArrayMutf32};
use super::avx32_utils;

/// Implementation of Bluestein's Algorithm
///
/// This algorithm computes an arbitrary-sized FFT in O(nlogn) time. It does this by converting this size n FFT into a
/// size M where M >= 2N - 1. M is usually a power of two, although that isn't a requirement.
///
/// It requires a large scratch space, so it's probably inconvenient to use as an inner FFT to other algorithms.
///
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Bluestein's Algorithm
/// use rustfft::algorithm::RadersAlgorithm;
/// use rustfft::{FFTplanner, Fft};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1201];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1201];
///
/// // plan a FFT of size n - 1 = 1200
/// let mut planner = FFTplanner::new(false);
/// let inner_fft = planner.plan_fft(1200);
///
/// let fft = RadersAlgorithm::new(1201, inner_fft);
/// fft.process(&mut input, &mut output);
/// ~~~
///
/// Bluestein's Algorithm is relatively expensive compared to other FFT algorithms. Benchmarking shows that it is up to
/// an order of magnitude slower than similar composite sizes. In the example size above of 1201, benchmarking shows
/// that it takes 2.5x more time to compute than a FFT of size 1200.

pub struct BluesteinsAvx<T> {
    inner_fft: Arc<Fft<T>>,

    inner_fft_multiplier: Box<[__m256]>,
    twiddles: Box<[__m256]>,

    len: usize,
    remainder_count: usize,
    inverse: bool,
}

impl BluesteinsAvx<f32> {
    fn compute_bluesteins_twiddle(index: usize, len: usize, inverse: bool) -> Complex<f32> {
        let index_float = index as f64;
        let index_squared = index_float * index_float;

        f32::generate_twiddle_factor_floatindex(index_squared, len*2, !inverse)
    }

    /// Creates a FFT instance which will process inputs/outputs of size `len`. `inner_fft.len()` must be >= `len * 2 - 1`
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must run a FFT of size inner_fft.len() within the
    /// constructor. This further underlines the fact that Bluesteins Algorithm is more expensive to run than other
    /// FFT algorithms
    ///
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(len: usize, inner_fft: Arc<Fft<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(len, inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(len: usize, inner_fft: Arc<Fft<f32>>) -> Self {
        let inner_fft_len = inner_fft.len();
        assert!(len * 2 - 1 <= inner_fft_len, "Bluestein's algorithm requires inner_fft.len() >= self.len() * 2 - 1. Expected >= {}, got {}", len * 2 - 1, inner_fft_len);
        assert!(inner_fft_len % 4 == 0, "BluesteinsAvx<f32> requires its inner_fft.len() to be a multiple of 4. inner_fft.len() = {}", inner_fft_len);

        // when computing FFTs, we're going to run our inner multiply pairwise by some precomputed data, then run an inverse inner FFT. We need to precompute that inner data here
        let inner_len_float = inner_fft_len as f32;
        let inverse = inner_fft.is_inverse();

        // Compute twiddle factors that we'll run our inner FFT on
        let mut inner_fft_input = vec![Complex::zero(); inner_fft_len];
        for i in 0..len {
            inner_fft_input[i] = Self::compute_bluesteins_twiddle(i, len, inverse) / inner_len_float;
        }
        for i in 1..len {
            inner_fft_input[inner_fft_len - i] = inner_fft_input[i];
        }

        //Compute the inner fft
        let mut inner_fft_scratch = vec![Complex::zero(); inner_fft.get_inplace_scratch_len()];
        inner_fft.process_inplace_with_scratch(&mut inner_fft_input, &mut inner_fft_scratch);

        // Pre-conjugate the result of the inner FFT. when computing the FFT later, we want it to be conjugated
        let conjugation_mask = avx32_utils::broadcast_complex_f32(Complex::new(0.0, -0.0));
        let inner_fft_multiplier = inner_fft_input.chunks_exact(4).map(|chunk| {
            let chunk_vector = chunk.load_complex_f32(0);
            _mm256_xor_ps(chunk_vector, conjugation_mask) // compute our conjugation by xoring ourr data with a precomputed mask
        }).collect::<Vec<_>>().into_boxed_slice();

        // also compute some more mundane twiddle factors to start and end with.
        // these will be the "main" twiddles. we will also hve "remainder" twiddles down below. We support remainders of 0 and 4, and there's less work if the remainder is 4
        // So if len is divisible by 4, compute one less twiddle herelet mut twiddles = Vec::with_capacity(num_twiddle_columns * 7);
        let (main_chunks, remainder) = avx32_utils::compute_chunk_count_complex_f32(len);
        let twiddles : Vec<_> = (0..main_chunks+1).map(|x| {
            let chunk_size = if x == main_chunks { remainder } else { 4 };

            let mut twiddle_chunk = [Complex::zero();4];
            for i in 0..chunk_size {
                twiddle_chunk[i] = Self::compute_bluesteins_twiddle(x*4+i, len, !inverse);
            }
            twiddle_chunk.load_complex_f32(0)
        }).collect();

        Self {
            inner_fft: inner_fft,

            inner_fft_multiplier,
            twiddles: twiddles.into_boxed_slice(),

            len,
            remainder_count: remainder,
            inverse,
        }
    }

    // Do the necessary setup for bluestein's algorithm: copy the data to the inner buffers, apply some twiddle factors, zero out the rest of the inner buffer
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn prepare_bluesteins(&self, input: &[Complex<f32>], inner_fft_buffer: &mut [Complex<f32>]) {
        let (main_chunks, _remainder) = avx32_utils::compute_chunk_count_complex_f32(self.len());

        // Copy the buffer into our inner FFT input, applying twiddle factors as we go. the buffer will only fill part of the FFT input, so zero fill the rest
        for (i, twiddle) in self.twiddles.iter().enumerate().take(main_chunks)  {
            let input_vector = input.load_complex_f32(i*4);
            let product_vector = avx32_utils::fma::complex_multiply_f32(input_vector, *twiddle);
            inner_fft_buffer.store_complex_f32(i*4, product_vector);
        }

        // the buffer will almost certainly have a remainder. it's so likely, in fact, that we're just going to apply a remainder unconditionally
        // it uses a couple more instructions in the rare case when our FFT size is a multiple of 4, but wastes instructions when it's not
        {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(self.remainder_count);
            let input_vector = input.load_complex_remainder_f32(remainder_mask, main_chunks * 4);
            let product_vector = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(main_chunks), input_vector);
            inner_fft_buffer.store_complex_f32(main_chunks * 4, product_vector);
        }

        // zero fill the rest of the `inner` array
        let zerofill_start = main_chunks + 1;
        for i in zerofill_start..(inner_fft_buffer.len()/4) {
            inner_fft_buffer.store_complex_f32(i*4, _mm256_setzero_ps());
        }
    }

    // Do the necessary finalization for bluestein's algorithm: Conjugate the inner FFT buffer, apply some twiddle factors, zero out the rest of the inner buffer
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn finalize_bluesteins(&self, inner_fft_buffer: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let (main_chunks, _remainder) = avx32_utils::compute_chunk_count_complex_f32(self.len());

        // copy our data to the output, applying twiddle factors again as we go. Also conjugate inner_input to complete the inverse FFT
        for i in 0..main_chunks  {
            let inner_vector = inner_fft_buffer.load_complex_f32(i*4);
            let product_vector = avx32_utils::fma::complex_conjugated_multiply_f32(inner_vector, *self.twiddles.get_unchecked(i));
            output.store_complex_f32(i*4, product_vector);
        }

        // again, unconditionally apply a remainder
        {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(self.remainder_count);

            let inner_vector = inner_fft_buffer.load_complex_f32(main_chunks * 4);
            let product_vector = avx32_utils::fma::complex_conjugated_multiply_f32(inner_vector, *self.twiddles.get_unchecked(main_chunks));
            output.store_complex_remainder_f32(remainder_mask, product_vector, main_chunks * 4);
        }
    }

    fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_multiplier.len()*4);

        // do the necessary setup for bluestein's algorithm: copy the data to the inner buffers, apply some twiddle factors, zero out the rest of the inner buffer
        unsafe { self.prepare_bluesteins(buffer, inner_input) };

        // run our inner forward FFT
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT. 
        // We can conjugate the result of multiplication by conjugating both inputs. We pre-conjugated the multiplier array, so we just need to conjugate inner_input
        unsafe { avx32_utils::fma::pairwise_complex_multiply_conjugated(inner_input, &self.inner_fft_multiplier) };

        // inverse FFT. we're computing a forward but we're massaging it into an inverse by conjugating the inputs and outputs
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // conjugate the inner FFT again to finalize the 
        unsafe { self.finalize_bluesteins(inner_input, buffer) };
    }

    fn perform_fft_out_of_place_f32(&self, input: &[Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_multiplier.len()*4);

        // do the necessary setup for bluestein's algorithm: copy the data to the inner buffers, apply some twiddle factors, zero out the rest of the inner buffer
        unsafe { self.prepare_bluesteins(input, inner_input) };

        // run our inner forward FFT
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT. 
        // We can conjugate the result of multiplication by conjugating both inputs. We pre-conjugated the multiplier array, so we just need to conjugate inner_input
        unsafe { avx32_utils::fma::pairwise_complex_multiply_conjugated(inner_input, &self.inner_fft_multiplier) };

        // inverse FFT. we're computing a forward but we're massaging it into an inverse by conjugating the inputs and outputs
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // conjugate the inner FFT again to finalize the 
        unsafe { self.finalize_bluesteins(inner_input, output) };
    }
}
boilerplate_fft_simd_f32!(BluesteinsAvx, 
    |this: &BluesteinsAvx<_>| this.len, // FFT len
    |this: &BluesteinsAvx<_>| this.inner_fft_multiplier.len()*4 + this.inner_fft.get_inplace_scratch_len(), // in-place scratch len
    |this: &BluesteinsAvx<_>| this.inner_fft_multiplier.len()*4 + this.inner_fft.get_inplace_scratch_len() // out of place scratch len
);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use test_utils::check_fft_algorithm;
    use algorithm::DFT;

    #[test]
    fn test_bluesteins_avx() {
        for len in 2..20 {
            // for this len, compute the range of inner FFT lengths we'll use.
            // Bluesteins AVX requires a multiple of 4 for the inner FFT, so we need to go up to the next multiple of 4 from the minimum
            let minimum_inner : usize = len * 2 - 1;
            let remainder = minimum_inner % 4;

            // remainder will never be 0, because "n * 2 - 1" is guaranteed to be odd. so we can just subtract the remainder and add 4.
            let next_multiple_of_4 = minimum_inner - remainder + 4;
            let maximum_inner = minimum_inner.checked_next_power_of_two().unwrap() + 1;

            // start at the next multiple of 4, and increment by 4 unti lwe get to the next power of 2.
            for inner_len in (next_multiple_of_4..maximum_inner).step_by(4) {
                test_bluesteins_avx_with_length(len, inner_len, false);
                test_bluesteins_avx_with_length(len, inner_len, true);
            }
        }
    }

    fn test_bluesteins_avx_with_length(len: usize, inner_len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(inner_len, inverse));
        let fft = BluesteinsAvx::new(len, inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_fft_algorithm(&fft, len, inverse);
    }
}
