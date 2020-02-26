use std::sync::Arc;

use num_complex::Complex;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, IsInverse, FFT};
use twiddles;

/// FFT algorithm optimized for power-of-two sizes
///
/// ~~~
/// // Computes a forward FFT of size 4096
/// use rustfft::algorithm::SplitRadix;
/// use rustfft::FFT;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 4096];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 4096];
///
/// let fft = SplitRadix::new(4096, false);
/// fft.process(&mut input, &mut output);
/// ~~~

pub struct SplitRadix<T> {
    twiddles: Box<[Complex<T>]>,
    fft_half: Arc<FFT<T>>,
    fft_quarter: Arc<FFT<T>>,
    len: usize,
    inverse: bool,
}

impl<T: FFTnum> SplitRadix<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(fft_half: Arc<FFT<T>>, fft_quarter: Arc<FFT<T>>) -> Self {
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

    unsafe fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        let half_len = self.len / 2;
        let quarter_len = self.len / 4;

        // Split the output into the buffers we'll use for the sub-FFT input
        let (inner_half_input, inner_quarter_input) = output.split_at_mut(half_len);
        let (inner_quarter1_input, inner_quarter3_input) = inner_quarter_input.split_at_mut(quarter_len);

        *inner_half_input.get_unchecked_mut(0)     = *input.get_unchecked(0);
        *inner_quarter1_input.get_unchecked_mut(0) = *input.get_unchecked(1);
        *inner_half_input.get_unchecked_mut(1)     = *input.get_unchecked(2);
        for i in 1..quarter_len {
            *inner_quarter3_input.get_unchecked_mut(i)     = *input.get_unchecked(i * 4 - 1);
            *inner_half_input.get_unchecked_mut(i * 2)     = *input.get_unchecked(i * 4);
            *inner_quarter1_input.get_unchecked_mut(i)     = *input.get_unchecked(i * 4 + 1);
            *inner_half_input.get_unchecked_mut(i * 2 + 1) = *input.get_unchecked(i * 4 + 2);
        }
        *inner_quarter3_input.get_unchecked_mut(0) = *input.get_unchecked(input.len() - 1);

        // Split the input into the buffers we'll use for the sub-FFT output
        let (inner_half_output, inner_quarter_output) = input.split_at_mut(half_len);
        let (inner_quarter1_output, inner_quarter3_output) = inner_quarter_output.split_at_mut(quarter_len);

        // Execute the inner FFTs
        self.fft_half.process(inner_half_input, inner_half_output);
        self.fft_quarter.process(inner_quarter1_input, inner_quarter1_output);
        self.fft_quarter.process(inner_quarter3_input, inner_quarter3_output);

        // Recombine into a single buffer
        for i in 0..quarter_len {
            let twiddle = *self.twiddles.get_unchecked(i);

            let twiddled_quarter1 = twiddle * inner_quarter1_output[i];
            let twiddled_quarter3 = twiddle.conj() * inner_quarter3_output[i];
            let quarter_sum  = twiddled_quarter1 + twiddled_quarter3;
            let quarter_diff = twiddled_quarter1 - twiddled_quarter3;

            *output.get_unchecked_mut(i)            = *inner_half_output.get_unchecked(i) + quarter_sum;
            *output.get_unchecked_mut(i + half_len) = *inner_half_output.get_unchecked(i) - quarter_sum;
            if self.is_inverse() {
                *output.get_unchecked_mut(i + quarter_len)     = *inner_half_output.get_unchecked(i + quarter_len) + twiddles::rotate_90(quarter_diff, true);
                *output.get_unchecked_mut(i + quarter_len * 3) = *inner_half_output.get_unchecked(i + quarter_len) + twiddles::rotate_90(quarter_diff, false);
            } else {
                *output.get_unchecked_mut(i + quarter_len)     = *inner_half_output.get_unchecked(i + quarter_len) + twiddles::rotate_90(quarter_diff, false);
                *output.get_unchecked_mut(i + quarter_len * 3) = *inner_half_output.get_unchecked(i + quarter_len) + twiddles::rotate_90(quarter_diff, true);
            }
        }
    }
}

impl<T: FFTnum> FFT<T> for SplitRadix<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

         unsafe {self.perform_fft(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            unsafe {self.perform_fft(in_chunk, out_chunk) };
        }
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
    use test_utils::check_fft_algorithm;
    use std::sync::Arc;
    use algorithm::DFT;

    #[test]
    fn test_splitradix() {
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

        check_fft_algorithm(&fft, len, inverse);
    }
}
