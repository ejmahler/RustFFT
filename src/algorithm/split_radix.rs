use std::sync::Arc;

use num_complex::Complex;

use common::{FFTnum, verify_length_inline, verify_length_minimum};

use ::{Length, IsInverse, FftInline};
use twiddles;

/// FFT algorithm optimized for multiple-of-4 FFT sizes.
///
/// This algorithm is quite slow, and is only used for testing SIMD split-radix algorithms.
///
/// ~~~
/// // Computes a forward FFT of size 4096
/// use std::sync::Arc;
/// use rustfft::algorithm::{SplitRadix, DFT};
/// use rustfft::FftInline;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let len = 4096;
///
/// let inner_fft_quarter = Arc::new(DFT::new(len/4, false));
/// let inner_fft_half = Arc::new(DFT::new(len/2, false));
///
/// let fft = SplitRadix::new(inner_fft_half, inner_fft_quarter);
///
/// let mut buffer:  Vec<Complex<f32>> = vec![Zero::zero(); len];
/// let mut scratch: Vec<Complex<f32>> = vec![Zero::zero(); fft.get_required_scratch_len()];
///
/// fft.process_inline(&mut buffer, &mut scratch);
/// ~~~
pub struct SplitRadix<T> {
    twiddles: Box<[Complex<T>]>,
    fft_half: Arc<FftInline<T>>,
    fft_quarter: Arc<FftInline<T>>,
    len: usize,
    inverse: bool,
}

impl<T: FFTnum> SplitRadix<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(fft_half: Arc<FftInline<T>>, fft_quarter: Arc<FftInline<T>>) -> Self {
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

        let twiddles : Vec<_> = (0..quarter_len).map(|i| twiddles::single_twiddle(i, len, inverse)).collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
            fft_half,
            fft_quarter,
            len,
            inverse,
        }
    }

    unsafe fn perform_fft(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let half_len = self.len() / 2;
        let quarter_len = self.len() / 4;

        // Split our scratch up into a section for our quarter1 FFT, and one for our quarter3 FFT
        let (scratch_quarter1, scratch_quarter3) = scratch.split_at_mut(quarter_len);

        // consolidate the evens int othe first half of the input buffer, and divide the odds up into the scratch
        *scratch_quarter1.get_unchecked_mut(0)  = *buffer.get_unchecked(1);
        *buffer.get_unchecked_mut(1)            = *buffer.get_unchecked(2);
        for i in 1..quarter_len {
            *scratch_quarter3.get_unchecked_mut(i)  = *buffer.get_unchecked(i * 4 - 1);
            *buffer.get_unchecked_mut(i * 2)        = *buffer.get_unchecked(i * 4);
            *scratch_quarter1.get_unchecked_mut(i)  = *buffer.get_unchecked(i * 4 + 1);
            *buffer.get_unchecked_mut(i * 2 + 1)    = *buffer.get_unchecked(i * 4 + 2);
        }
        *scratch_quarter3.get_unchecked_mut(0) = *buffer.get_unchecked(buffer.len() - 1);

        // Split up the input buffer. The first half contains the even-sized inner FFT data, and the second half will contain scratch space for our inner FFTs
        let (inner_buffer, inner_scratch) = buffer.split_at_mut(half_len);

        // Execute the inner FFTs
        self.fft_half.process_inline(inner_buffer, inner_scratch);
        self.fft_quarter.process_inline(scratch_quarter1, inner_scratch);
        self.fft_quarter.process_inline(scratch_quarter3, inner_scratch);

        // Recombine into a single buffer
        for i in 0..quarter_len {
            let twiddle = *self.twiddles.get_unchecked(i);

            let half0_result = *buffer.get_unchecked(i);
            let half1_result = *buffer.get_unchecked(i + quarter_len);

            let twiddled_quarter1 = twiddle * scratch_quarter1.get_unchecked(i);
            let twiddled_quarter3 = twiddle.conj() * scratch_quarter3.get_unchecked(i);

            let quarter_sum  = twiddled_quarter1 + twiddled_quarter3;
            let quarter_diff = twiddled_quarter1 - twiddled_quarter3;
            let rotated_quarter_diff = twiddles::rotate_90(quarter_diff, self.is_inverse());

            *buffer.get_unchecked_mut(i)            = half0_result + quarter_sum;
            *buffer.get_unchecked_mut(i + half_len) = half0_result - quarter_sum;

            *buffer.get_unchecked_mut(i + quarter_len)     = half1_result + rotated_quarter_diff;
            *buffer.get_unchecked_mut(i + quarter_len * 3) = half1_result - rotated_quarter_diff;
        }
    }
}

impl<T: FFTnum> FftInline<T> for SplitRadix<T> {
    fn process_inline(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        // The caller might have passed in more scratch than we need. if so, cap it off
        let scratch = if scratch.len() > self.get_required_scratch_len() {
            &mut scratch[..self.get_required_scratch_len()]
        } else {
            scratch
        };

        unsafe { self.perform_fft(buffer, scratch) };
    }
    fn get_required_scratch_len(&self) -> usize {
        self.len / 2
    }
}
impl<T> Length for SplitRadix<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for SplitRadix<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::{check_inline_fft_algorithm};
    use std::sync::Arc;
    use algorithm::*;

    #[test]
    fn test_splitradix_scalar() {
        for pow in 3..8 {
            let len = 1 << pow;
            test_splitradix_with_length(len, false);
            test_splitradix_with_length(len, true);
        }
    }

    fn test_splitradix_with_length(len: usize, inverse: bool) {
        let quarter = Arc::new(DFT::new(len / 4, inverse));
        let half = Arc::new(DFT::new(len / 2, inverse));
        let fft = SplitRadix::new(half, quarter);

        check_inline_fft_algorithm(&fft, len, inverse);
    }
}
