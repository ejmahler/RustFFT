use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, IsInverse, FFT};

/// Implementation of Bluestein's Algorithm
///
/// This algorithm computes a FFT of any size by using converting it to a convolution, where a longer power-of-two length FFT is used. 
/// The primary use is for prime-sized FFTs of lengths where Rader's alorithm is slow.
///
/// This implementation is based on the one in the [fourier library](https://github.com/calebzulawski/fourier) by Caleb Zulawski.
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Bluesteins's Algorithm
/// use rustfft::algorithm::Bluesteins;
/// use rustfft::{FFT, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1201];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1201];
///
/// // plan a forward FFT for the nearest power of two larger or equal to 2 * 1201 - 1 -> 4096
/// let mut planner = FFTplanner::new(false);
/// let inner_fft_fw = planner.plan_fft(4096);
///
/// // plan an inverse FFT for the nearest power of two larger or equal to 2 * 1201 - 1 -> 4096
/// let mut planner = FFTplanner::new(true);
/// let inner_fft_inv = planner.plan_fft(4096);
///
/// let fft = Bluesteins::new(1201, inner_fft_fw, inner_fft_inv, false);
/// fft.process(&mut input, &mut output);
/// ~~~
///
/// Bluestein's algorithm is mainly useful for prime-length FFTs, and in particular for the lengths where
/// Rader's algorithm is slow due to hitting a Cunningham Chain. 

pub struct Bluesteins<T> {
    len: usize,
    inner_fft_fw: Arc<FFT<T>>,
    inner_fft_inv: Arc<FFT<T>>,
    w_twiddles: Box<[Complex<T>]>,
    x_twiddles: Box<[Complex<T>]>,
    inverse: bool,
}

fn calculate_twiddle<T: FFTnum>(index: f64, len: usize) -> Complex<T> {
    let theta = index * core::f64::consts::PI / len as f64;
    Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    )
}



/// Calculate the "w" twiddles.
fn calculate_w_twiddles<T: FFTnum>(
    len: usize,
    fft: &Arc<FFT<T>>,
    twiddles: &mut [Complex<T>],
    inverse: bool,
) {
    let mut scratch = vec![Complex::zero(); fft.len()];
    let scale = T::one() / T::from_usize(fft.len()).unwrap();
    for (i, tw) in scratch.iter_mut().enumerate() {
        if let Some(index) = {
            if i < len {
                Some((i as f64).powi(2))
            } else if i > fft.len() - len {
                Some(((i as f64) - (fft.len() as f64)).powi(2))
            } else {
                None
            }
        } {
            let twiddle = calculate_twiddle(index, len)*scale;
            if inverse {
                *tw = twiddle;
            }
            else {
                *tw = twiddle.conj();
            }
        } 
    }
    fft.process(&mut scratch, &mut twiddles[..]);
}


/// Calculate the "x" twiddles.
fn calculate_x_twiddles<T: FFTnum>(
    len: usize,
    twiddles: &mut [Complex<T>],
    inverse: bool,
) {
    if inverse {
        for (i, tw) in twiddles.iter_mut().enumerate() {
            *tw = calculate_twiddle(-(i as f64).powi(2), len);
        }
    }
    else {
        for (i, tw) in twiddles.iter_mut().enumerate() {
            *tw = calculate_twiddle(-(i as f64).powi(2), len).conj();
        }
    }
}


impl<T: FFTnum > Bluesteins<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `len`. `inner_fft_fw.len()` and `inner_fft_inv.len()` must be the nearest 
    /// power of two that is equal to or larger than `2 * len - 1`.
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must run a FFT within the
    /// constructor. 
    pub fn new(len: usize, inner_fft_fw: Arc<FFT<T>>, inner_fft_inv: Arc<FFT<T>>, inverse: bool) -> Self {
        let inner_fft_len = (2 * len - 1).checked_next_power_of_two().unwrap();
        assert_eq!(inner_fft_len, inner_fft_fw.len(), "For Bluesteins algorithm, inner_fft_fw.len() must be a power of to larger than or equal to 2*self.len() - 1. Expected {}, got {}", inner_fft_len, inner_fft_fw.len());
        assert_eq!(inner_fft_len, inner_fft_inv.len(), "For Bluesteins algorithm, inner_fft_inv.len() must be a power of to larger than or equal to 2*self.len() - 1. Expected {}, got {}", inner_fft_len, inner_fft_inv.len());

        let mut w_twiddles = vec![Complex::zero(); inner_fft_len];
        let mut x_twiddles = vec![Complex::zero(); len];
        calculate_w_twiddles(len, &inner_fft_fw, &mut w_twiddles, inverse);
        calculate_x_twiddles(len, &mut x_twiddles, inverse);
        Self {
            len,
            inner_fft_fw,
            inner_fft_inv,
            w_twiddles: w_twiddles.into_boxed_slice(),
            x_twiddles: x_twiddles.into_boxed_slice(),
            inverse,
        }
    }

    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        assert_eq!(self.len(), input.len());

        let mut scratch_a = vec![Complex::zero(); self.inner_fft_fw.len()];
        let mut scratch_b = vec![Complex::zero(); self.inner_fft_fw.len()];
        for (w, (x, i)) in scratch_a.iter_mut().zip(self.x_twiddles.iter().zip(input.iter())) {
            *w = x * i;
        }
        self.inner_fft_fw.process(&mut scratch_a, &mut scratch_b);
        for (w, wi) in scratch_b.iter_mut().zip(self.w_twiddles.iter()) {
            *w = *w * wi;
        }
        self.inner_fft_inv.process(&mut scratch_b, &mut scratch_a);
        for (i, (w, xi)) in output.iter_mut().zip(scratch_a.iter().zip(self.x_twiddles.iter())) {
            *i = w * xi;
        }
    }

   
}

impl<T: FFTnum> FFT<T> for Bluesteins<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

        self.perform_fft(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            self.perform_fft(in_chunk, out_chunk);
        }
    }
}
impl<T> Length for Bluesteins<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for Bluesteins<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use test_utils::check_fft_algorithm;
    use algorithm::DFT;

    #[test]
    fn test_bluestein() {
        for &len in &[3,5,7,11,13] {
            test_bluestein_with_length(len, false);
            test_bluestein_with_length(len, true);
        }
    }

    fn test_bluestein_with_length(len: usize, inverse: bool) {
        let inner_fft_len = (2 * len - 1).checked_next_power_of_two().unwrap();
        let inner_fft_fw = Arc::new(DFT::new(inner_fft_len, false));
        let inner_fft_inv = Arc::new(DFT::new(inner_fft_len, true));
        let fft = Bluesteins::new(len, inner_fft_fw, inner_fft_inv, inverse);

        check_fft_algorithm(&fft, len, inverse);
    }
}
