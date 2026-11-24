use std::sync::Arc;

use num_complex::Complex;

use crate::{Fft, FftNum, simd::simd_bluesteins::SimdBluesteins};

/// Implementation of Bluestein's Algorithm
///
/// This algorithm computes an arbitrary-sized FFT in O(nlogn) time. It does this by converting this size-N FFT into a
/// size-M FFT where M >= 2N - 1.
///
/// The choice of M is very important for the performance of Bluestein's Algorithm. The most obvious choice is the next-largest
/// power of two -- but if there's a smaller/faster FFT size that satisfies the `>= 2N - 1` requirement, that will significantly
/// improve this algorithm's overall performance.
///
/// ~~~
/// // Computes a forward FFT of size 1201, using Bluestein's Algorithm
/// use rustfft::algorithm::BluesteinsAlgorithm;
/// use rustfft::{Fft, FftPlanner};
/// use rustfft::num_complex::Complex;
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1201];
///
/// // We need to find an inner FFT whose size is greater than 1201*2 - 1.
/// // The size 2401 (7^4) satisfies this requirement, while also being relatively fast.
/// let mut planner = FftPlanner::new();
/// let inner_fft = planner.plan_fft_forward(2401);
///
/// let fft = BluesteinsAlgorithm::new(1201, inner_fft);
/// fft.process(&mut buffer);
/// ~~~
///
/// Bluesteins's Algorithm is relatively expensive compared to other FFT algorithms. Benchmarking shows that it is up to
/// an order of magnitude slower than similar composite sizes. In the example size above of 1201, benchmarking shows
/// that it takes 5x more time to compute than computing a FFT of size 1200 via a step of MixedRadix.
pub struct BluesteinsAlgorithm<T: FftNum>(SimdBluesteins<Complex<T>, T>);

impl<T: FftNum> BluesteinsAlgorithm<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `len`. `inner_fft.len()` must be >= `len * 2 - 1`
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must compute a FFT using `inner_fft` within the
    /// constructor. This further underlines the fact that Bluesteins Algorithm is more expensive to run than other
    /// FFT algorithms
    ///
    /// # Panics
    /// Panics if `inner_fft.len() < len * 2 - 1`.
    pub fn new(len: usize, inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdBluesteins::new(len, inner_fft))
    }
}
boilerplate_scalar_fft!(BluesteinsAlgorithm, #[inline(always)]);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::simd::simd_bluesteins::test_bodies;

    fn new_fn<T: FftNum>(len: usize, inner_fft: Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
        Arc::new(BluesteinsAlgorithm::new(len, inner_fft))
    }

    #[test]
    fn test_scalar_bluesteins_f32() {
        test_bodies::inner_lengths::<Complex<f32>>(&new_fn);
    }

    #[test]
    fn test_scalar_bluesteins_f64() {
        test_bodies::inner_lengths::<Complex<f64>>(&new_fn);
    }
}
