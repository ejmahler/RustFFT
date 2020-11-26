use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, IsInverse, FFT};

/// Implementation of Rader's Algorithm
///
/// This algorithm computes a prime-sized FFT in O(nlogn) time. It does this by converting this size n FFT into a
/// size (n - 1) FFT, which is guaranteed to be composite.
///
/// The worst case for this algorithm is when (n - 1) is 2 * prime, resulting in a
/// [Cunningham Chain](https://en.wikipedia.org/wiki/Cunningham_chain)
///
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Rader's Algorithm
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
/// Rader's Algorithm is relatively expensive compared to other FFT algorithms. Benchmarking shows that it is up to
/// an order of magnitude slower than similar composite sizes. In the example size above of 1201, benchmarking shows
/// that it takes 2.5x more time to compute than a FFT of size 1200.

pub struct Bluesteins<T> {
    len: usize,
    inner_fft_fw: Arc<FFT<T>>,
    inner_fft_inv: Arc<FFT<T>>,
    w_forward: Box<[Complex<T>]>,
    w_inverse: Box<[Complex<T>]>,
    x_forward: Box<[Complex<T>]>,
    x_inverse: Box<[Complex<T>]>,
    //scratch_a: Box<[Complex<T>]>,
    //scratch_b: Box<[Complex<T>]>,
    inverse: bool,
}

fn compute_half_twiddle<T: FFTnum>(index: f64, size: usize) -> Complex<T> {
    let theta = index * core::f64::consts::PI / size as f64;
    Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    )
}

/// Initialize the "w" twiddles.
fn initialize_w_twiddles<T: FFTnum>(
    len: usize,
    fft: &Arc<FFT<T>>,
    forward_twiddles: &mut [Complex<T>],
    inverse_twiddles: &mut [Complex<T>],
) {
    let mut forward_twiddles_temp = vec![Complex::zero(); fft.len()];
    let mut inverse_twiddles_temp = vec![Complex::zero(); fft.len()];
    for i in 0..fft.len() {
        if let Some(index) = {
            if i < len {
                Some((i as f64).powi(2))
            } else if i > fft.len() - len {
                Some(((i as f64) - (fft.len() as f64)).powi(2))
            } else {
                None
            }
        } {
            let twiddle = compute_half_twiddle(index, len);
            forward_twiddles_temp[i] = twiddle.conj();
            inverse_twiddles_temp[i] = twiddle;
        } else {
            forward_twiddles_temp[i] = Complex::zero();
            inverse_twiddles_temp[i] = Complex::zero();
        }
    }
    fft.process(&mut forward_twiddles_temp, &mut forward_twiddles[..]);
    fft.process(&mut inverse_twiddles_temp, &mut inverse_twiddles[..]);
}

/// Initialize the "x" twiddles.
fn initialize_x_twiddles<T: FFTnum>(
    len: usize,
    forward_twiddles: &mut [Complex<T>],
    inverse_twiddles: &mut [Complex<T>],
) {
    for i in 0..len {
        let twiddle = compute_half_twiddle(-(i as f64).powi(2), len);
        forward_twiddles[i] = twiddle.conj();
        inverse_twiddles[i] = twiddle;
    }
}

impl<T: FFTnum > Bluesteins<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `len`. `inner_fft.len()` must be `len - 1`
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must run a FFT of size n - 1 within the
    /// constructor. This further underlines the fact that Rader's Algorithm is more expensive to run than other
    /// FFT algorithms
    ///
    /// Note also that if `len` is not prime, this algorithm may silently produce garbage output
    pub fn new(len: usize, inner_fft_fw: Arc<FFT<T>>, inner_fft_inv: Arc<FFT<T>>, inverse: bool) -> Self {
        let inner_fft_len = (2 * len - 1).checked_next_power_of_two().unwrap();
        assert_eq!(inner_fft_len, inner_fft_fw.len(), "For Bluesteins algorithm, inner_fft.len() must be a power of to larger than or equal to 2*self.len() - 1. Expected {}, got {}", inner_fft_len, inner_fft_fw.len());

        let mut w_forward = vec![Complex::zero(); inner_fft_len];
        let mut w_inverse = vec![Complex::zero(); inner_fft_len];
        let mut x_forward = vec![Complex::zero(); len];
        let mut x_inverse = vec![Complex::zero(); len];
        //let mut scratch_a = vec![Complex::zero(); inner_fft_len];
        //let mut scratch_b = vec![Complex::zero(); inner_fft_len];
        initialize_w_twiddles(len, &inner_fft_fw, &mut w_forward, &mut w_inverse);
        initialize_x_twiddles(len, &mut x_forward, &mut x_inverse);
        //println!("w_forward");
        //for tw in w_forward.iter() {
        //    println!("{:?}, {:?}", tw.re, tw.im);
        //}
        //println!("w_inverse");
        //for tw in w_inverse.iter() {
        //    println!("{:?}, {:?}", tw.re, tw.im);
        //}
        //println!("x_forward");
        //for tw in x_forward.iter() {
        //    println!("{:?}, {:?}", tw.re, tw.im);
        //}
        //println!("x_inverse");
        //for tw in x_inverse.iter() {
        //    println!("{:?}, {:?}", tw.re, tw.im);
        //}
        Self {
            len,
            inner_fft_fw: inner_fft_fw,
            inner_fft_inv: inner_fft_inv,
            w_forward: w_forward.into_boxed_slice(),
            w_inverse: w_inverse.into_boxed_slice(),
            x_forward: x_forward.into_boxed_slice(),
            x_inverse: x_inverse.into_boxed_slice(),
            //scratch_a: scratch_a.into_boxed_slice(),
            //scratch_b: scratch_b.into_boxed_slice(),
            inverse,
        }
    }

    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        //assert_eq!(self.x_forward.len(), input.len());
        
        let size = input.len();
        let mut scratch_a = vec![Complex::zero(); self.inner_fft_fw.len()];
        let mut scratch_b = vec![Complex::zero(); self.inner_fft_fw.len()];
        if !self.inverse {
            for (w, (x, i)) in scratch_a.iter_mut().zip(self.x_forward.iter().zip(input.iter())) {
                *w = x * i;
            }
            //for w in scratch_a[size..].iter_mut() {
            //    *w = Complex::zero();
            //}
            self.inner_fft_fw.process(&mut scratch_a, &mut scratch_b);
            for (w, wi) in scratch_b.iter_mut().zip(self.w_forward.iter()) {
                *w = *w * wi;
            }
            self.inner_fft_inv.process(&mut scratch_b, &mut scratch_a);
            let scale = T::one() / T::from_usize(self.inner_fft_inv.len()).unwrap();
            for (i, (w, xi)) in output.iter_mut().zip(scratch_a.iter().zip(self.x_forward.iter())) {
                *i = w * xi * scale;
            }
        }
        else {
            for (w, (x, i)) in scratch_a.iter_mut().zip(self.x_inverse.iter().zip(input.iter())) {
                *w = x * i;
            }
            //for w in scratch_a[size..].iter_mut() {
            //    *w = Complex::zero();
            //}
            self.inner_fft_fw.process(&mut scratch_a, &mut scratch_b);
            for (w, wi) in scratch_b.iter_mut().zip(self.w_inverse.iter()) {
                *w = *w * wi;
            }
            self.inner_fft_inv.process(&mut scratch_b, &mut scratch_a);
            let scale = T::one() / T::from_usize(self.inner_fft_inv.len()).unwrap();
            for (i, (w, xi)) in output.iter_mut().zip(scratch_a.iter().zip(self.x_inverse.iter())) {
                *i = w * xi * scale;
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
