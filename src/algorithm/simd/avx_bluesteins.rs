use std::sync::Arc;
use std::arch::x86_64::*;

use num_complex::Complex;
use num_traits::Zero;

use common::{FFTnum, verify_length_inline, verify_length_minimum};

use ::{Length, IsInverse, FftInline};

use algorithm::simd::avx_utils::AvxComplexArrayf32;
use super::avx_utils;

/// Implementation of Bluestein's Algorithm
///
/// This algorithm computes an arbitrary-sized FFT in O(nlogn) time. It does this by converting this size n FFT into a
/// size M FFT, where M >= 2N - 1. M is usually a power of two, although that isn't a requirement.
///
/// It requires a large scratch space, so it's probably inconvenient to use as an inner FFT to other algorithms.
///
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Bluestein's Algorithm
/// use rustfft::algorithm::RadersAlgorithm;
/// use rustfft::{FFT, FFTplanner};
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
    conjugation_mask: __m256,
    remainder_twiddles: __m256,
    remainder_mask: avx_utils::RemainderMask,

    inner_fft: Arc<FftInline<T>>,

    inner_fft_multiplier: Box<[__m256]>,
    twiddles: Box<[__m256]>,

    len: usize,
    remainder_count: usize,
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
    pub fn new(len: usize, inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
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
    unsafe fn new_with_avx(len: usize, inner_fft: Arc<FftInline<f32>>) -> Self {
        let inner_fft_len = inner_fft.len();
        assert!(len * 2 - 1 <= inner_fft_len, "Bluestein's algorithm requires inner_fft.len() >= self.len() * 2 - 1. Expected >= {}, got {}", len * 2 - 1, inner_fft_len);

        // when computing FFTs, we're going to run our inner FFT, multiply pairwise by some precomputed data, then run an inverse inner FFT. We need to precompute that inner data here
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
        let mut inner_fft_scratch = vec![Complex::zero(); inner_fft.get_required_scratch_len()];
        inner_fft.process_inline(&mut inner_fft_input, &mut inner_fft_scratch);

        // also compute some more mundane twiddle factors to start and end with.
        // these will be the "main" twiddles. we will also hve "remainder" twiddles down below. We support remainders of 0 and 4, and there's less work if the remainder is 4
        // So if len is divisible by 4, compute one less twiddle here
        let remainder_count = {
            let modulo = len % 4;
            if modulo == 0 { 4 } else { modulo }
        };
        let remainder_index = len - remainder_count;
        let twiddles : Vec<_> = (0..remainder_index/4).map(|i| {
            let twiddle_chunk = [
                Self::compute_bluesteins_twiddle(i*4, len, !inverse),
                Self::compute_bluesteins_twiddle(i*4+1, len, !inverse),
                Self::compute_bluesteins_twiddle(i*4+2, len, !inverse),
                Self::compute_bluesteins_twiddle(i*4+3, len, !inverse),
            ];
            twiddle_chunk.load_complex_f32(0)
        }).collect();

        // Handle the remainder twiddles, by populating the relevant ones, and leaving a zero for the irrelevant ones
        let mut remainder_chunk = [Complex::zero(); 4];
        for i in 0..remainder_count {
            remainder_chunk[i] = Self::compute_bluesteins_twiddle(remainder_index + i, len, !inverse);
        }

        Self {
            conjugation_mask: avx_utils::broadcast_complex_f32(Complex::new(0.0, -0.0)),

            remainder_twiddles: remainder_chunk.load_complex_f32(0),
            remainder_mask: avx_utils::RemainderMask::new_f32(remainder_count),

            inner_fft: inner_fft,

            inner_fft_multiplier: inner_fft_input.chunks_exact(4).map(|chunk| chunk.load_complex_f32(0)).collect(),
            twiddles: twiddles.into_boxed_slice(),

            len,
            remainder_count,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_multiplier.len()*4);
        let quarter_len = (buffer.len() - self.remainder_count) / 4;

        // Copy the buffer into our inner FFT input, applying twiddle factors as we go. the buffer will only fill part of the FFT input, so zero fill the rest
        for i in 0..quarter_len  {
            let buffer_vector = buffer.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_multiply_fma_f32(buffer_vector, *self.twiddles.get_unchecked(i));
            inner_input.store_complex_f32(i*4, product_vector);
        }

        // the buffer will almost certainly have a remainder. it's so likely, in fact, that we're just going to apply a remainder unconditionally
        // it uses a couple more instructions in the rare case when our FFT size is a multiple of 4, but wastes instructions when it's not
        let buffer_vector = buffer.load_complex_remainder_f32(quarter_len * 4, self.remainder_mask);
        let product_vector = avx_utils::complex_multiply_fma_f32(self.remainder_twiddles, buffer_vector);
        inner_input.store_complex_f32(quarter_len * 4, product_vector);

        // zero fill the rest of the `inner` array
        let zerofill_start = quarter_len + 1;
        for i in zerofill_start..(inner_input.len()/4) {
            inner_input.store_complex_f32(i*4, _mm256_setzero_ps());
        }

        // run our inner forward FFT
        self.inner_fft.process_inline(inner_input, inner_scratch);

        // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT
        for (i, twiddle) in self.inner_fft_multiplier.iter().enumerate() {
            let inner_vector = inner_input.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_multiply_fma_f32(inner_vector, *twiddle);
            let conjugated_vector = _mm256_xor_ps(product_vector, self.conjugation_mask);

            inner_input.store_complex_f32(i*4, conjugated_vector);
        }

        // inverse FFT. we're computing a forward FFT, but we're massaging it into an inverse by conjugating the inputs and outputs
        self.inner_fft.process_inline(inner_input, inner_scratch);

        // copy our data back to the buffer, applying twiddle factors again as we go. Also conjugate inner_input to complete the inverse FFT
        for i in 0..quarter_len  {
            let inner_vector = inner_input.load_complex_f32(i*4);
            let product_vector = avx_utils::complex_conjugated_multiply_fma_f32(inner_vector, *self.twiddles.get_unchecked(i));
            buffer.store_complex_f32(i*4, product_vector);
        }

        // again, unconditionally apply a remainder
        let inner_vector = inner_input.load_complex_f32(quarter_len * 4);
        let product_vector = avx_utils::complex_conjugated_multiply_fma_f32(inner_vector, self.remainder_twiddles);
        buffer.store_complex_remainder_f32(quarter_len * 4, product_vector, self.remainder_mask);
    }
}

default impl<T: FFTnum> FftInline<T> for BluesteinsAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for BluesteinsAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    fn get_required_scratch_len(&self) -> usize {
        self.inner_fft_multiplier.len()*4 + self.inner_fft.get_required_scratch_len()
    }
}
impl<T> Length for BluesteinsAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for BluesteinsAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inner_fft.is_inverse()
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use test_utils::check_inline_fft_algorithm;
    use algorithm::DFT;
    use algorithm::BluesteinsAlgorithm;

    #[test]
    fn test_bluesteins_avx() {
        for &len in &[3,5,7,11,13,16] {
            test_bluesteins_avx_with_length(len, false);
            test_bluesteins_avx_with_length(len, true);
        }
    }

    fn test_bluesteins_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new((len *2 - 1).checked_next_power_of_two().unwrap(), inverse)) as Arc<dyn FftInline<f32>>;

        let control = BluesteinsAlgorithm::new(len, Arc::clone(&inner_fft));
        let fft = BluesteinsAvx::new(len, inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&control, len, inverse);
        check_inline_fft_algorithm(&fft, len, inverse);
    }
}
