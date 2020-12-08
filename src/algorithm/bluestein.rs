use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, IsInverse, FFT};

/// Implementation of Bluestein's Algorithm
///
/// This algorithm computes a FFT of any size by using converting it to a convolution, where a longer FFT of an "easy" length is used. 
/// The primary use is for prime-sized FFTs of lengths where Rader's alorithm is slow.
/// When performing a FFT the inner FFT is run twice. These FFTs make up nearly all the processing time. 
/// There are also simple O(n) processing steps before, in between and after the inner FFTs. 
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
/// // Plan a forward FFT for the nearest power of two larger or equal to 2 * 1201 - 1 -> 4096
/// // Note that the inner fft determines if the Bluestein fft is a forward or inverse transform.
/// let mut planner = FFTplanner::new(false);
/// let inner_fft = planner.plan_fft(4096);
///
/// let fft = Bluesteins::new(1201, inner_fft);
/// fft.process(&mut input, &mut output);
/// ~~~
///
/// Bluestein's algorithm is mainly useful for prime-length FFTs, and in particular for the lengths where
/// Rader's algorithm is slow due to hitting a Cunningham Chain. 

pub struct Bluesteins<T> {
    len: usize,
    inner_fft: Arc<FFT<T>>,
    w_twiddles: Box<[Complex<T>]>,
    x_twiddles: Box<[Complex<T>]>,
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
            *tw = calculate_twiddle(index, len).conj()*scale;
        }
    }
    fft.process(&mut scratch, &mut twiddles[..]);
    if fft.is_inverse() {
        for tw in twiddles.iter_mut() {
            *tw = tw.conj();
        }
    }
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
    /// Creates a FFT instance which will process inputs/outputs of size `len`. The inner FFT can be any length that is equal to or larger than `2 * len - 1`.
    /// The inner FFT determines if this becomes a forward or inverse transform.
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must run a FFT within the
    /// constructor. 
    pub fn new(len: usize, inner_fft: Arc<FFT<T>> ) -> Self {
        let min_inner_fft_len = 2 * len - 1;
        assert!(inner_fft.len() >= min_inner_fft_len, "For Bluesteins algorithm, inner_fft.len() must be equal to or larger than 2*self.len() - 1. Expected at least {}, got {}", min_inner_fft_len, inner_fft.len());

        let mut w_twiddles = vec![Complex::zero(); inner_fft.len()];
        let mut x_twiddles = vec![Complex::zero(); len];
        calculate_w_twiddles(len, &inner_fft, &mut w_twiddles);
        calculate_x_twiddles(len, &mut x_twiddles, inner_fft.is_inverse());
        Self {
            len,
            inner_fft,
            w_twiddles: w_twiddles.into_boxed_slice(),
            x_twiddles: x_twiddles.into_boxed_slice(),
        }
    }

    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        // The steps of this algorithm are:
        // - Multiply inputs with the X twiddles, and store in the first `len` elements of longer vector.
        // - Perform a forward FFT of this vector to get a "spectrum".
        // - Multipy the spectrum with the W twiddles.
        // - Inverse transform the modified spectrum.
        // - Multiply first `len` elements of the iFFT output with the X-twiddles, and store in the output.
        //
        // By using the fact that iFFT(X) = FFT(X*)* and FFT(X) = iFFT(X*)*, we can use a single inner
        // FFT for both forward and inverse transform, by conjugating the inputs and outputs as needed.
        
        assert_eq!(self.len(), input.len());

        let mut scratch = vec![Complex::zero(); 2*self.inner_fft.len()];
        let (mut scratch_a, mut scratch_b) = scratch.split_at_mut(self.inner_fft.len());
        // Copy input data to scratch vector, and multiply with X twiddles.
        // If the inner FFT is inverse, then we conjugate the input here to make the first FFT step a forward transform
        if self.inner_fft.is_inverse() {
            for (w, (x, i)) in scratch_a.iter_mut().zip(self.x_twiddles.iter().zip(input.iter())) {
                *w = (x * i).conj();
            }
        }
        else {
            for (w, (x, i)) in scratch_a.iter_mut().zip(self.x_twiddles.iter().zip(input.iter())) {
                *w = x * i;
            }
        }
        
        // Perform forward FFT (either directly or using an inverse FFT with conjugated inputs and outputs).
        self.inner_fft.process(&mut scratch_a, &mut scratch_b);

        // Multiply spectrum with W twiddles.
        // If the inner FFT is inverse, then we conjugate the results from the first FFT step to make it a forward transform.
        if self.inner_fft.is_inverse() {
            for (w, wi) in scratch_b.iter_mut().zip(self.w_twiddles.iter()) {
                *w = w.conj() * wi;
            }
        }
        // If the inner FFT is forward, then we conjugate the input to the second FFT step here to make it an inverse transform
        else {
            for (w, wi) in scratch_b.iter_mut().zip(self.w_twiddles.iter()) {
                *w = (*w * wi).conj();
            }
        }

        // Perform inverse FFT (either directly or using a forward FFT with conjugated inputs and outputs).
        self.inner_fft.process(&mut scratch_b, &mut scratch_a);

        // Multiply with X twiddles and store in output.
        // If the inner FFT is inverse, use the output directly.
        if self.inner_fft.is_inverse() {
            for (i, (w, xi)) in output.iter_mut().zip(scratch_a.iter().zip(self.x_twiddles.iter())) {
                *i = w * xi;
            }
        }
        // If the inner FFT is forward, then we conjugate the output from the second FFT step to make it an inverse transform.
        else {
            for (i, (w, xi)) in output.iter_mut().zip(scratch_a.iter().zip(self.x_twiddles.iter())) {
                *i = w.conj() * xi;
            }
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
        self.inner_fft.is_inverse()
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
        for &len in &[3,5,7,11,13,123] {
            test_bluestein_with_length(len, false);
            test_bluestein_with_length(len, true);
        }
    }

    fn test_bluestein_with_length(len: usize, inverse: bool) {
        let inner_fft_len = (2 * len - 1).checked_next_power_of_two().unwrap();
        let inner_fft = Arc::new(DFT::new(inner_fft_len, inverse));
        let fft = Bluesteins::new(len, inner_fft);

        check_fft_algorithm(&fft, len, inverse);
    }
}
