use std::sync::Arc;
use std::arch::x86_64::*;

use num_complex::Complex;
use num_traits::Zero;

use common::FFTnum;

use ::{Length, IsInverse, Fft};

use super::avx_utils::AvxComplexArrayf32;
use super::avx_utils;

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

        // also compute some more mundane twiddle factors to start and end with.
        // these will be the "main" twiddles. we will also hve "remainder" twiddles down below. We support remainders of 0 and 4, and there's less work if the remainder is 4
        // So if len is divisible by 4, compute one less twiddle herelet mut twiddles = Vec::with_capacity(num_twiddle_columns * 7);
        let (main_chunks, remainder) = avx_utils::compute_chunk_count_complex_f32(len);
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

            inner_fft_multiplier: inner_fft_input.chunks_exact(4).map(|chunk| chunk.load_complex_f32(0)).collect(),
            twiddles: twiddles.into_boxed_slice(),

            len,
            remainder_count: remainder,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_multiplier.len()*4);
        
        let (main_chunks, _remainder) = avx_utils::compute_chunk_count_complex_f32(self.len());

        // Copy the buffer into our inner FFT input, applying twiddle factors as we go. the buffer will only fill part of the FFT input, so zero fill the rest
        for i in 0..main_chunks  {
            let buffer_vector = buffer.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_multiply_fma_f32(buffer_vector, *self.twiddles.get_unchecked(i));
            inner_input.store_complex_f32(i*4, product_vector);
        }

        // the buffer will almost certainly have a remainder. it's so likely, in fact, that we're just going to apply a remainder unconditionally
        // it uses a couple more instructions in the rare case when our FFT size is a multiple of 4, but wastes instructions when it's not
        {
            let remainder_mask = avx_utils::RemainderMask::new_f32(self.remainder_count);

            let buffer_vector = buffer.load_complex_remainder_f32(remainder_mask, main_chunks * 4);
            let product_vector = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(main_chunks), buffer_vector);
            inner_input.store_complex_f32(main_chunks * 4, product_vector);
        }

        // zero fill the rest of the `inner` array
        let zerofill_start = main_chunks + 1;
        for i in zerofill_start..(inner_input.len()/4) {
            inner_input.store_complex_f32(i*4, _mm256_setzero_ps());
        }

        // run our inner forward FFT
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT
        let conjugation_mask = avx_utils::broadcast_complex_f32(Complex::new(0.0, -0.0));
        for (i, twiddle) in self.inner_fft_multiplier.iter().enumerate() {
            let inner_vector = inner_input.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_multiply_fma_f32(inner_vector, *twiddle);
            let conjugated_vector = _mm256_xor_ps(product_vector, conjugation_mask);

            inner_input.store_complex_f32(i*4, conjugated_vector);
        }

        // inverse FFT. we're computing a forward but we're massaging it into an inverse by conjugating the inputs and outputs
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // copy our data back to the buffer, applying twiddle factors again as we go. Also conjugate inner_input to complete the inverse FFT
        for i in 0..main_chunks  {
            let inner_vector = inner_input.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_conjugated_multiply_fma_f32(inner_vector, *self.twiddles.get_unchecked(i));
            buffer.store_complex_f32(i*4, product_vector);
        }

        // again, unconditionally apply a remainder
        {
            let remainder_mask = avx_utils::RemainderMask::new_f32(self.remainder_count);

            let inner_vector = inner_input.load_complex_f32(main_chunks * 4);
            let product_vector = avx_utils::complex_conjugated_multiply_fma_f32(inner_vector, *self.twiddles.get_unchecked(main_chunks));
            buffer.store_complex_remainder_f32(remainder_mask, product_vector, main_chunks * 4);
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_out_of_place_f32(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_multiplier.len()*4);
        
        let (main_chunks, _remainder) = avx_utils::compute_chunk_count_complex_f32(self.len());

        // Copy the buffer into our inner FFT input, applying twiddle factors as we go. the buffer will only fill part of the FFT input, so zero fill the rest
        for i in 0..main_chunks  {
            let buffer_vector = input.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_multiply_fma_f32(buffer_vector, *self.twiddles.get_unchecked(i));
            inner_input.store_complex_f32(i*4, product_vector);
        }

        // the buffer will almost certainly have a remainder. it's so likely, in fact, that we're just going to apply a remainder unconditionally
        // it uses a couple more instructions in the rare case when our FFT size is a multiple of 4, but wastes instructions when it's not
        {
            let remainder_mask = avx_utils::RemainderMask::new_f32(self.remainder_count);

            let buffer_vector = input.load_complex_remainder_f32(remainder_mask, main_chunks * 4);
            let product_vector = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(main_chunks), buffer_vector);
            inner_input.store_complex_f32(main_chunks * 4, product_vector);
        }

        // zero fill the rest of the `inner` array
        let zerofill_start = main_chunks + 1;
        for i in zerofill_start..(inner_input.len()/4) {
            inner_input.store_complex_f32(i*4, _mm256_setzero_ps());
        }

        // run our inner forward FFT
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT
        let conjugation_mask = avx_utils::broadcast_complex_f32(Complex::new(0.0, -0.0));
        for (i, twiddle) in self.inner_fft_multiplier.iter().enumerate() {
            let inner_vector = inner_input.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_multiply_fma_f32(inner_vector, *twiddle);
            let conjugated_vector = _mm256_xor_ps(product_vector, conjugation_mask);

            inner_input.store_complex_f32(i*4, conjugated_vector);
        }

        // inverse FFT. we're computing a forward but we're massaging it into an inverse by conjugating the inputs and outputs
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // copy our data back to the buffer, applying twiddle factors again as we go. Also conjugate inner_input to complete the inverse FFT
        for i in 0..main_chunks  {
            let inner_vector = inner_input.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_conjugated_multiply_fma_f32(inner_vector, *self.twiddles.get_unchecked(i));
            output.store_complex_f32(i*4, product_vector);
        }

        // again, unconditionally apply a remainder
        {
            let remainder_mask = avx_utils::RemainderMask::new_f32(self.remainder_count);

            let inner_vector = inner_input.load_complex_f32(main_chunks * 4);
            let product_vector = avx_utils::complex_conjugated_multiply_fma_f32(inner_vector, *self.twiddles.get_unchecked(main_chunks));
            output.store_complex_remainder_f32(remainder_mask, product_vector, main_chunks * 4);
        }
    }
}
boilerplate_fft_simd_unsafe!(BluesteinsAvx, 
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
        for &len in &[3,5,7,11,13,16] {
            test_bluesteins_avx_with_length(len, false);
            test_bluesteins_avx_with_length(len, true);
        }
    }

    fn test_bluesteins_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new((len *2 - 1).checked_next_power_of_two().unwrap(), inverse));
        let fft = BluesteinsAvx::new(len, inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_fft_algorithm(&fft, len, inverse);
    }
}
