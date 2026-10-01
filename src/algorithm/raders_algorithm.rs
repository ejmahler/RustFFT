use num_complex::Complex;

use crate::simd::simd_raders::SimdRaders;

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
///
/// This is the shared `SimdRaders` with a single complex number per vector, which is the scalar
/// case. Its pairwise multiply loop then compiles to the same plain loop this algorithm had before
/// it was shared, so there is nothing left here but the name.
pub type RadersAlgorithm<T> = SimdRaders<Complex<T>, T>;

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::algorithm::Dft;
    use crate::simd::simd_raders::test_bodies;
    use crate::test_utils::check_fft_algorithm;
    use crate::{Fft, FftDirection, FftPlanner};
    use primal_check::miller_rabin;
    use std::sync::Arc;

    #[test]
    fn test_scalar_raders_f32() {
        test_bodies::prime_lengths::<Complex<f32>>();
    }

    #[test]
    fn test_scalar_raders_f64() {
        test_bodies::prime_lengths::<Complex<f64>>();
    }

    // The above go through SimdRaders directly, so also cover the public name.
    #[test]
    fn test_scalar_raders() {
        for len in 3..100 {
            if miller_rabin(len as u64) {
                test_raders_with_length(len, FftDirection::Forward);
                test_raders_with_length(len, FftDirection::Inverse);
            }
        }
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

    fn test_raders_with_length(len: usize, direction: FftDirection) {
        let inner_fft = Arc::new(Dft::new(len - 1, direction));
        let fft = RadersAlgorithm::new(inner_fft);

        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
