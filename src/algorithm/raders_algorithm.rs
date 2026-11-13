use std::sync::Arc;

use num_complex::Complex;

use crate::{Fft, FftNum, simd::simd_raders::SimdRaders};

/// Implementation of Rader's Algorithm
///
/// This algorithm computes a prime-sized FFT in O(nlogn) time. It does this by converting this size-N FFT into a
/// size-(N - 1) FFT, which is guaranteed to be composite.
///
/// The worst case for this algorithm is when (N - 1) is 2 * prime, resulting in a
/// [Cunningham Chain](https://en.wikipedia.org/wiki/Cunningham_chain)
///
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Rader's Algorithm
/// use rustfft::algorithm::RadersAlgorithm;
/// use rustfft::{Fft, FftPlanner};
/// use rustfft::num_complex::Complex;
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1201];
///
/// // plan a FFT of size n - 1 = 1200
/// let mut planner = FftPlanner::new();
/// let inner_fft = planner.plan_fft_forward(1200);
///
/// let fft = RadersAlgorithm::new(inner_fft);
/// fft.process(&mut buffer);
/// ~~~
///
/// Rader's Algorithm is relatively expensive compared to other FFT algorithms. Benchmarking shows that it is up to
/// an order of magnitude slower than similar composite sizes. In the example size above of 1201, benchmarking shows
/// that it takes 2.5x more time to compute than a FFT of size 1200.
pub struct RadersAlgorithm<T: FftNum>(SimdRaders<Complex<T>, T>);

impl<T: FftNum> RadersAlgorithm<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `inner_fft.len() + 1`.
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must compute a FFT using `inner_fft` within the
    /// constructor. This further underlines the fact that Rader's Algorithm is more expensive to run than other
    /// FFT algorithms
    ///
    /// # Panics
    /// Panics if `inner_fft.len() + 1` is not a prime number.
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRaders::new(inner_fft))
    }
}
boilerplate_scalar_fft!(RadersAlgorithm, #[inline(always)]);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::FftPlanner;
    use crate::simd::simd_raders::test_bodies;

    fn new_fn<T: FftNum>(inner_fft: Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
        Arc::new(RadersAlgorithm::new(inner_fft))
    }

    #[test]
    fn test_scalar_raders_f32() {
        test_bodies::prime_lengths::<Complex<f32>>(&new_fn);
    }

    #[test]
    fn test_scalar_raders_f64() {
        test_bodies::prime_lengths::<Complex<f64>>(&new_fn);
    }

    #[test]
    fn test_raders_32bit_overflow() {
        // Construct and use Raders instances for a few large primes
        // that could panic due to overflow errors on 32-bit builds.
        let mut planner = FftPlanner::<f32>::new();
        for len in [112501, 216569, 417623] {
            let inner_fft = planner.plan_fft_forward(len - 1);
            let fft: RadersAlgorithm<f32> = RadersAlgorithm::new(inner_fft);
            let mut data = vec![Complex::new(0.0, 0.0); len];
            fft.process(&mut data);
        }
    }
}
