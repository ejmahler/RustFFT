use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use common::{FFTnum, verify_length_inline, verify_length_minimum};

use ::{Length, IsInverse, FftInline};

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
/// Rader's Algorithm is relatively expensive compared to other FFT algorithms. Benchmarking shows that it is up to
/// an order of magnitude slower than similar composite sizes. In the example size above of 1201, benchmarking shows
/// that it takes 2.5x more time to compute than a FFT of size 1200.

pub struct BluesteinsAlgorithm<T> {
    inner_fft: Arc<FftInline<T>>,

    inner_fft_multiplier: Box<[Complex<T>]>,
    twiddles: Box<[Complex<T>]>,

    len: usize,
}

impl<T: FFTnum> BluesteinsAlgorithm<T> {
    fn compute_bluesteins_twiddle(index: usize, size: usize, inverse: bool) -> Complex<T> {
        let index_multiplier = core::f64::consts::PI / size as f64;

        let index_float = index as f64;
        let index_squared = index_float * index_float;

        let theta = index_squared * index_multiplier;
        let result = Complex::new(
            T::from_f64(theta.cos()).unwrap(),
            T::from_f64(theta.sin()).unwrap(),
        );

        if inverse {
            result.conj()
        } else {
            result
        }
    }


    /// Creates a FFT instance which will process inputs/outputs of size `len`. `inner_fft.len()` must be >= `len * 2 - 1`
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must run a FFT of size inner_fft.len() within the
    /// constructor. This further underlines the fact that Bluesteins Algorithm is more expensive to run than other
    /// FFT algorithms
    pub fn new(len: usize, inner_fft: Arc<FftInline<T>>) -> Self {
        let inner_fft_len = inner_fft.len();
        assert!(len * 2 - 1 <= inner_fft_len, "Bluestein's algorithm requires inner_fft.len() >= self.len() * 2 - 1. Expected >= {}, got {}", len * 2 - 1, inner_fft_len);

        // when computing FFTs, we're going to run our inner FFT, multiply pairise by some precomputed data, then run an inverse inner FFT. We need to precompute that inner data here
        let inner_len_float = T::from_usize(inner_fft_len).unwrap();
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

        // also compute some more mundane twiddle factors to start and end with
        let twiddles : Vec<_> = (0..len).map(|i| Self::compute_bluesteins_twiddle(i, len, !inverse)).collect();

        Self {
            inner_fft: inner_fft,

            inner_fft_multiplier: inner_fft_input.into_boxed_slice(),
            twiddles: twiddles.into_boxed_slice(),

            len,
        }
    }

    fn perform_bluestein_fft(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_multiplier.len());

        // Copy the buffer into our inner FFT input. the buffer will only fill part of the FFT input, so zero fill the rest
        for ((buffer_entry, inner_entry), twiddle) in buffer.iter().zip(inner_input.iter_mut()).zip(self.twiddles.iter()) {
            *inner_entry = *buffer_entry * *twiddle ;
        }
        for inner in inner_input.iter_mut().skip(buffer.len()) {
            *inner = Complex::zero();
        }

        // run our inner forward FFT
        self.inner_fft.process_inline(inner_input, inner_scratch);

        // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT
        for (inner, multiplier) in inner_input.iter_mut().zip(self.inner_fft_multiplier.iter()) {
            *inner = (*inner * *multiplier).conj();
        }

        // inverse FFT. we're computing a forward FFT, but we're massaging it into an inverse by conjugating the inputs and outputs
        self.inner_fft.process_inline(inner_input, inner_scratch);

        // copy our data back to the buffer, applying twiddle factors again as we go. Also conjugate inner_input to complete the inverse FFT
        for ((buffer_entry, inner_entry), twiddle) in buffer.iter_mut().zip(inner_input.iter()).zip(self.twiddles.iter()) {
            *buffer_entry = inner_entry.conj() * twiddle;
        }
    }
}

impl<T: FFTnum> FftInline<T> for BluesteinsAlgorithm<T> {
    fn process_inline(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        self.perform_bluestein_fft(buffer, scratch);
    }
    fn get_required_scratch_len(&self) -> usize {
        self.inner_fft_multiplier.len() + self.inner_fft.get_required_scratch_len()
    }
}
impl<T> Length for BluesteinsAlgorithm<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for BluesteinsAlgorithm<T> {
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

    #[test]
    fn test_bluesteins() {
        for &len in &[3,5,7,11,13] {
            test_bluesteins_with_length(len, false);
            test_bluesteins_with_length(len, true);
        }
    }

    fn test_bluesteins_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new((len *2 - 1).checked_next_power_of_two().unwrap(), inverse));
        let fft = BluesteinsAlgorithm::new(len, inner_fft);

        check_inline_fft_algorithm(&fft, len, inverse);
    }
}
