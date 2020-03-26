use std::sync::Arc;
use std::arch::x86_64::*;

use num_complex::Complex;

use common::{FFTnum, verify_length_inline, verify_length_minimum};

use ::{Length, IsInverse, FftInline};

use super::avx_utils::AvxComplexArrayf32;
use super::avx_utils;

/// FFT algorithm optimized for power-of-two sizes, using AVX instructions
///
/// ~~~
/// // Computes a forward FFT of size 4096
/// use std::sync::Arc;
/// use rustfft::algorithm::{SplitRadixAvx, SplitRadix, DFT};
/// use rustfft::FftInline;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let len = 4096;
///
/// let inner_fft_quarter = Arc::new(DFT::new(len/4, false)) as Arc<dyn FftInline<f32>>;
/// let inner_fft_half = Arc::new(DFT::new(len/2, false)) as Arc<dyn FftInline<f32>>;
///
/// let mut buffer:  Vec<Complex<f32>> = vec![Zero::zero(); len];
///
/// // The SplitRadixAvx algorithm requries the "avx" and "fma" instruction sets. The constructor will return Err() if this machine is missing either instruction set
/// if let Ok(avx_algorithm) = SplitRadixAvx::new(Arc::clone(&inner_fft_half), Arc::clone(&inner_fft_quarter)) {
///     let mut scratch: Vec<Complex<f32>> = vec![Zero::zero(); avx_algorithm.get_required_scratch_len()];
///
///     avx_algorithm.process_inline(&mut buffer, &mut scratch);
/// } else {
///     let scalar_algorithm = SplitRadix::new(inner_fft_half, inner_fft_quarter);
///     let mut scratch: Vec<Complex<f32>> = vec![Zero::zero(); scalar_algorithm.get_required_scratch_len()];
///
///     scalar_algorithm.process_inline(&mut buffer, &mut scratch);
/// }
/// ~~~
pub struct SplitRadixAvx<T> {
    twiddles: Box<[__m256]>,
    fft_half: Arc<FftInline<T>>,
    fft_quarter: Arc<FftInline<T>>,
    len: usize,
    twiddle_config: avx_utils::Rotate90Config,
    inverse: bool,
}

impl SplitRadixAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(fft_half: Arc<FftInline<f32>>, fft_quarter: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(fft_half, fft_quarter) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(fft_half: Arc<FftInline<f32>>, fft_quarter: Arc<FftInline<f32>>) -> Self {
        assert_eq!(
            fft_half.is_inverse(), fft_quarter.is_inverse(), 
            "fft_half and fft_quarter must both be inverse, or neither. got fft_half inverse={}, fft_quarter inverse={}",
            fft_half.is_inverse(), fft_quarter.is_inverse());

        assert_eq!(
            fft_half.len(), fft_quarter.len() * 2, 
            "fft_half must be 2x the len of fft_quarter. got fft_half len={}, fft_quarter len={}",
            fft_half.len(), fft_quarter.len());

        let inverse = fft_quarter.is_inverse();
        let quarter_len = fft_quarter.len();
        let len = quarter_len * 4;

        assert_eq!(len % 16, 0, "SplitRadixAvx requires its FFT length to be a multiple of 16. Got {}", len);

        let sixteenth_len = quarter_len / 4;
        let twiddles : Vec<_> = (0..sixteenth_len).map(|i| {
            let twiddle_chunk = [
                f32::generate_twiddle_factor(i*4, len, inverse),
                f32::generate_twiddle_factor(i*4+1, len, inverse),
                f32::generate_twiddle_factor(i*4+2, len, inverse),
                f32::generate_twiddle_factor(i*4+3, len, inverse),
            ];
            twiddle_chunk.load_complex_f32(0)
        }).collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
            fft_half,
            fft_quarter,
            len,
            twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let half_len = self.len / 2;
        let quarter_len = self.len / 4;
        let three_quarter_len = half_len + quarter_len;
        let sixteenth_len = self.len / 16;

        let (scratch_quarter1, scratch_quarter3) = scratch.split_at_mut(quarter_len);

        for i in 0..sixteenth_len {
            let chunk0 = buffer.load_complex_f32(i*16);
            let chunk1 = buffer.load_complex_f32(i*16 + 4);
            let (even0, odd0) = avx_utils::split_evens_odds_f32(chunk0, chunk1);

            let chunk2 = buffer.load_complex_f32(i*16 + 8);
            let chunk3 = buffer.load_complex_f32(i*16 + 12);
            let (even1, odd1) = avx_utils::split_evens_odds_f32(chunk2, chunk3);

            let (quarter1, quarter3) = avx_utils::split_evens_odds_f32(odd0, odd1);

            buffer.store_complex_f32(i*8, even0);
            buffer.store_complex_f32(i*8 + 4, even1);
            scratch_quarter1.store_complex_f32(i*4, quarter1);

            // We need to rotate every entry in quarter3 downwards one, and wrap the last entry back to the first
            // We'll accomplish the shift here by adding 1 to the index, and complete the rotation after the loop
           scratch_quarter3.store_complex_f32(i*4+1, quarter3);
        }

        // complete the rotate of scratch_quarter3 by copying the last element to the first. then, slice off the last element
        *scratch_quarter3.get_unchecked_mut(0) = *scratch_quarter3.get_unchecked(quarter_len);
        let scratch_quarter3 = &mut scratch_quarter3[..quarter_len];

        // Split up the input buffer. The first half contains the even-sized inner FFT data, and the second half will contain scratch space for our inner FFTs
        let (inner_buffer, inner_scratch) = buffer.split_at_mut(half_len);

        // Execute the inner FFTs
        self.fft_half.process_inline(inner_buffer, inner_scratch);
        self.fft_quarter.process_inline(scratch_quarter1, inner_scratch);
        self.fft_quarter.process_inline(scratch_quarter3, inner_scratch);

        // Recombine into a single buffer
        for i in 0..sixteenth_len {
            let inner_even0_entry = buffer.load_complex_f32(i * 4);
            let inner_even1_entry = buffer.load_complex_f32(quarter_len + i * 4);
            let inner_quarter1_entry = scratch_quarter1.load_complex_f32(i * 4);
            let inner_quarter3_entry = scratch_quarter3.load_complex_f32(i * 4);

            let twiddle = *self.twiddles.get_unchecked(i);

            let twiddled_quarter1 = avx_utils::complex_multiply_fma_f32(twiddle, inner_quarter1_entry);
            let twiddled_quarter3 = avx_utils::complex_conjugated_multiply_fma_f32(twiddle, inner_quarter3_entry);
            let (quarter_sum, quarter_diff) = avx_utils::column_butterfly2_f32(twiddled_quarter1, twiddled_quarter3);

            let (output_i, output_i_half) = avx_utils::column_butterfly2_f32(inner_even0_entry, quarter_sum);

            // compute the twiddle for quarter diff by rotating it
            let quarter_diff_rotated = avx_utils::rotate90_f32(quarter_diff, self.twiddle_config);

            let (output_quarter1, output_quarter3) = avx_utils::column_butterfly2_f32(inner_even1_entry, quarter_diff_rotated);

            buffer.store_complex_f32(i * 4, output_i);
            buffer.store_complex_f32(i * 4 + quarter_len, output_quarter1);
            buffer.store_complex_f32(i * 4 + half_len, output_i_half);
            buffer.store_complex_f32(i * 4 + three_quarter_len, output_quarter3);
        }
    }
}
default impl<T: FFTnum> FftInline<T> for SplitRadixAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for SplitRadixAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    fn get_required_scratch_len(&self) -> usize {
        self.len / 2 + 1
    }
}

impl<T> Length for SplitRadixAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for SplitRadixAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_inline_fft_algorithm;
    use std::sync::Arc;
    use algorithm::*;

    #[test]
    fn test_splitradix_avx() {
        for pow in 4..8 {
            let len = 1 << pow;
            test_splitradix_avx_with_length(len, false);
            test_splitradix_avx_with_length(len, true);
        }
    }

    fn test_splitradix_avx_with_length(len: usize, inverse: bool) {
        let quarter = Arc::new(DFT::new(len / 4, inverse)) as Arc<dyn FftInline<f32>>;
        let half = Arc::new(DFT::new(len / 2, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = SplitRadixAvx::new(half, quarter).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }
}