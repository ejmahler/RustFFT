use std::sync::Arc;

use num_complex::Complex;

use crate::{
    algorithm::butterflies::{
        Butterfly1, Butterfly16, Butterfly2, Butterfly32, Butterfly4, Butterfly8,
    },
    simd::simd_radix4::SimdRadix4,
    Fft, FftDirection, FftNum,
};

/// FFT algorithm optimized for power-of-two sizes
///
/// ~~~
/// // Computes a forward FFT of size 4096
/// use rustfft::algorithm::Radix4;
/// use rustfft::{Fft, FftDirection};
/// use rustfft::num_complex::Complex;
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 4096];
///
/// let fft = Radix4::new(4096, FftDirection::Forward);
/// fft.process(&mut buffer);
/// ~~~
pub struct Radix4<T: FftNum>(SimdRadix4<Complex<T>, T>);
impl<T: FftNum> Radix4<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(len: usize, direction: FftDirection) -> Self {
        assert!(
            len.is_power_of_two(),
            "Radix4 algorithm requires a power-of-two input size. Got {}",
            len
        );

        // figure out which base length we're going to use
        let exponent = len.trailing_zeros();
        let (base_exponent, base_fft) = match exponent {
            0 => (0, Arc::new(Butterfly1::new(direction)) as Arc<dyn Fft<T>>),
            1 => (1, Arc::new(Butterfly2::new(direction)) as Arc<dyn Fft<T>>),
            2 => (2, Arc::new(Butterfly4::new(direction)) as Arc<dyn Fft<T>>),
            3 => (3, Arc::new(Butterfly8::new(direction)) as Arc<dyn Fft<T>>),
            _ => {
                if exponent % 2 == 1 {
                    (5, Arc::new(Butterfly32::new(direction)) as Arc<dyn Fft<T>>)
                } else {
                    (4, Arc::new(Butterfly16::new(direction)) as Arc<dyn Fft<T>>)
                }
            }
        };

        Self::new_with_base((exponent - base_exponent) / 2, base_fft)
    }

    /// Constructs a Radix4 instance which computes FFTs of length `4^k * base_fft.len()`
    pub fn new_with_base(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadix4::new(k, base_fft))
    }
}
boilerplate_scalar_fft!(Radix4, #[inline(always)]);

#[cfg(test)]
mod unit_tests {
    use std::sync::Arc;

    use num_complex::Complex;

    use crate::{
        algorithm::Radix4,
        simd::simd_radix4::test_bodies,
        test_utils::{check_fft_algorithm, construct_base},
        Fft, FftDirection, FftNum,
    };

    fn new_fn<T: FftNum>(k: u32, base_fft: Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
        Arc::new(Radix4::new_with_base(k, base_fft))
    }

    #[test]
    fn test_scalar_radix4_f64() {
        // a scalar vector is a single complex number, so every base length is legal
        test_bodies::factor_pairs::<Complex<f64>>(&[1, 2, 3, 4, 5, 6], &new_fn);
    }

    #[test]
    fn test_scalar_radix4_f32() {
        // the scalar vector is one complex number whatever the precision, so f32 is no different
        test_bodies::factor_pairs::<Complex<f32>>(&[1, 2, 3, 4, 5, 6], &new_fn);
    }

    #[test]
    fn test_scalar_radix4_composite_base() {
        test_bodies::composite_base::<Complex<f32>, Complex<f64>>(&new_fn, &new_fn);
    }

    // The above tests get coverage of SimdRadixN itself - but since we're a newtype with wrapper functions etc, we also want coverage on the wrapper code
    #[test]
    fn test_scalar_radix4_with_length() {
        for pow in 0..8 {
            let len = 1 << pow;

            let forward_fft = Radix4::new(len, FftDirection::Forward);
            check_fft_algorithm::<f32>(&forward_fft, len, FftDirection::Forward);

            let inverse_fft = Radix4::new(len, FftDirection::Inverse);
            check_fft_algorithm::<f32>(&inverse_fft, len, FftDirection::Inverse);
        }
    }

    #[test]
    fn test_scalar_radix4_with_base() {
        for base in 1..=9 {
            let base_forward = construct_base(base, FftDirection::Forward);
            let base_inverse = construct_base(base, FftDirection::Inverse);

            for k in 0..4 {
                test_radix4(k, Arc::clone(&base_forward));
                test_radix4(k, Arc::clone(&base_inverse));
            }
        }
    }

    fn test_radix4(k: u32, base_fft: Arc<dyn Fft<f64>>) {
        let len = base_fft.len() * 4usize.pow(k as u32);
        let direction = base_fft.fft_direction();
        let fft = Radix4::new_with_base(k, base_fft);

        check_fft_algorithm::<f64>(&fft, len, direction);
    }
}
