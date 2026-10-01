use num_complex::Complex;

use crate::simd::simd_bluesteins::SimdBluesteins;

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
///
/// This is the shared `SimdBluesteins` with a single complex number per vector, which is the scalar
/// case. Its pairwise multiply loops then compile to the same plain loops this algorithm had before
/// they were shared, so there is nothing left here but the name.
pub type BluesteinsAlgorithm<T> = SimdBluesteins<Complex<T>, T>;

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::algorithm::Dft;
    use crate::simd::simd_bluesteins::test_bodies;
    use crate::test_utils::check_fft_algorithm;
    use crate::FftDirection;
    use std::sync::Arc;

    #[test]
    fn test_scalar_bluesteins_f32() {
        test_bodies::inner_lengths::<Complex<f32>>();
    }

    #[test]
    fn test_scalar_bluesteins_f64() {
        test_bodies::inner_lengths::<Complex<f64>>();
    }

    // The above go through SimdBluesteins directly, so also cover the public name.
    #[test]
    fn test_scalar_bluesteins() {
        for &len in &[3, 5, 7, 11, 13] {
            test_bluesteins_with_length(len, FftDirection::Forward);
            test_bluesteins_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_bluesteins_with_length(len: usize, direction: FftDirection) {
        let inner_fft = Arc::new(Dft::new(
            (len * 2 - 1).checked_next_power_of_two().unwrap(),
            direction,
        ));
        let fft = BluesteinsAlgorithm::new(len, inner_fft);

        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
