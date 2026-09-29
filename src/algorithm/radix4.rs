use std::sync::Arc;

use num_complex::Complex;

use crate::{
    algorithm::butterflies::{
        Butterfly1, Butterfly16, Butterfly2, Butterfly32, Butterfly4, Butterfly8,
    },
    simd::simd_radix4::SimdRadix4,
    Direction, Fft, FftDirection, FftNum, Length,
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
impl<T: FftNum> Length for Radix4<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<T: FftNum> Direction for Radix4<T> {
    fn fft_direction(&self) -> FftDirection {
        self.0.fft_direction()
    }
}
impl<T: FftNum> Fft<T> for Radix4<T> {
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        self.0.process_with_scratch(buffer, scratch);
    }

    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.0
            .process_outofplace_with_scratch(input, output, scratch);
    }

    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.0
            .process_immutable_with_scratch(input, output, scratch);
    }

    fn get_inplace_scratch_len(&self) -> usize {
        self.0.get_inplace_scratch_len()
    }

    fn get_outofplace_scratch_len(&self) -> usize {
        self.0.get_outofplace_scratch_len()
    }

    fn get_immutable_scratch_len(&self) -> usize {
        self.0.get_immutable_scratch_len()
    }
}

#[cfg(test)]
mod unit_tests {
    use std::sync::Arc;

    use num_complex::Complex;

    use crate::{
        algorithm::Radix4,
        simd::simd_radix4::test_bodies,
        test_utils::{check_fft_algorithm, construct_base},
        Fft, FftDirection,
    };

    #[test]
    fn test_scalar_radix4_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<Complex<f64>>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_scalar_radix4_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<Complex<f32>>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_scalar_radix4_composite_base() {
        test_bodies::composite_base::<Complex<f32>, Complex<f64>>();
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
