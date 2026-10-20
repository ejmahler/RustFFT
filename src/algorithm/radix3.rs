use std::sync::Arc;

use num_complex::Complex;

use crate::Fft;
use crate::algorithm::butterflies::{Butterfly1, Butterfly3, Butterfly9, Butterfly27};
use crate::array_utils::compute_logarithm;
use crate::common::RadixFactor;
use crate::simd::simd_radixn::SimdRadixN;
use crate::{FftDirection, common::FftNum};

/// FFT algorithm optimized for power-of-three sizes
///
/// ~~~
/// // Computes a forward FFT of size 2187
/// use rustfft::algorithm::Radix3;
/// use rustfft::{Fft, FftDirection};
/// use rustfft::num_complex::Complex;
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 2187];
///
/// let fft = Radix3::new(2187, FftDirection::Forward);
/// fft.process(&mut buffer);
/// ~~~

pub struct Radix3<T: FftNum>(SimdRadixN<Complex<T>, T>);

impl<T: FftNum> Radix3<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-three FFT
    pub fn new(len: usize, direction: FftDirection) -> Self {
        // Compute the total power of 3 for this length. IE, len = 3^exponent
        let exponent = compute_logarithm::<3>(len).unwrap_or_else(|| {
            panic!(
                "Radix3 algorithm requires a power-of-three input size. Got {}",
                len
            )
        });

        // figure out which base length we're going to use
        let (base_exponent, base_fft) = match exponent {
            0 => (0, Arc::new(Butterfly1::new(direction)) as Arc<dyn Fft<T>>),
            1 => (1, Arc::new(Butterfly3::new(direction)) as Arc<dyn Fft<T>>),
            2 => (2, Arc::new(Butterfly9::new(direction)) as Arc<dyn Fft<T>>),
            _ => (3, Arc::new(Butterfly27::new(direction)) as Arc<dyn Fft<T>>),
        };

        Self::new_with_base(exponent - base_exponent, base_fft)
    }

    /// Constructs a Radix3 instance which computes FFTs of length `3^k * base_fft.len()`
    fn new_with_base(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        let factors = std::iter::repeat_n(RadixFactor::Factor3, k as usize).collect::<Vec<_>>();
        Self(SimdRadixN::new(&factors, base_fft))
    }
}
boilerplate_scalar_fft!(Radix3, #[inline(always)]);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::{check_fft_algorithm, construct_base};

    #[test]
    fn test_scalar_radix3_with_length() {
        for pow in 0..5 {
            let len = 3usize.pow(pow);

            let forward_fft = Radix3::new(len, FftDirection::Forward);
            check_fft_algorithm::<f32>(&forward_fft, len, FftDirection::Forward);

            let inverse_fft = Radix3::new(len, FftDirection::Inverse);
            check_fft_algorithm::<f32>(&inverse_fft, len, FftDirection::Inverse);
        }
    }

    #[test]
    fn test_scalar_radix3_with_base() {
        for base in 1..=9 {
            let base_forward = construct_base(base, FftDirection::Forward);
            let base_inverse = construct_base(base, FftDirection::Inverse);

            for k in 0..4 {
                test_radix3(k, Arc::clone(&base_forward));
                test_radix3(k, Arc::clone(&base_inverse));
            }
        }
    }

    fn test_radix3(k: u32, base_fft: Arc<dyn Fft<f32>>) {
        let len = base_fft.len() * 3usize.pow(k as u32);
        let direction = base_fft.fft_direction();
        let fft = Radix3::new_with_base(k, base_fft);

        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
