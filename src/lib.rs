#![cfg_attr(all(feature = "bench", test), feature(test))]

extern crate num;

mod algorithm;
mod butterflies;
mod math_utils;

use num::{Complex, Zero, One, Float, FromPrimitive, Signed};
use num::traits::cast;
use std::f32;

use algorithm::{FFTAlgorithm, CooleyTukey, Radix4, RadersAlgorithm};

pub struct FFT<T> {
    len: usize,
    algorithm: Box<FFTAlgorithm<T>>,
}

impl<T> FFT<T>
    where T: Signed + FromPrimitive + Copy + 'static
{
    /// Creates a new FFT context that will process signal of length
    /// `len`. If `inverse` is `true`, then this struct will run inverse
    /// FFTs. This implementation of the FFT doesn't do any scaling on both
    /// the forward and backward transforms, so doing a forward then backward
    /// FFT on a signal will scale the signal by its length.
    pub fn new(len: usize, inverse: bool) -> Self {
        let algorithm = if len < 2 {
            Box::new(NoopAlgorithm {}) as Box<FFTAlgorithm<T>>
        } else if len.is_power_of_two() {
            Box::new(Radix4::new(len, inverse)) as Box<FFTAlgorithm<T>>
        } else {
            let factors = factor(len);

            // benchmarking shows that raders algorithm isn't faster than the
            // naive o(n^2) algorithm below around 100
            if factors.len() == 1 && len > 100 {
                // there is only one factor, meaning the input has a prime size
                Box::new(RadersAlgorithm::new(len, inverse)) as Box<FFTAlgorithm<T>>
            } else {
                Box::new(CooleyTukey::new(len, factors, inverse)) as Box<FFTAlgorithm<T>>
            }
        };

        FFT {
            len: len,
            algorithm: algorithm,
        }
    }

    /// Runs the FFT on the input `signal` buffer, and places the output in the
    /// `spectrum` buffer.
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        assert!(signal.len() == spectrum.len());
        assert!(signal.len() == self.len);

        self.algorithm.process(signal, spectrum);
    }
}

struct NoopAlgorithm {}
impl<T> FFTAlgorithm<T> for NoopAlgorithm
    where T: Signed + FromPrimitive + Copy
{
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        spectrum.copy_from_slice(signal);
    }
}

pub fn dft<T: Float>(signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
    for (k, spec_bin) in spectrum.iter_mut().enumerate() {
        let mut sum = Zero::zero();
        for (i, &x) in signal.iter().enumerate() {
            let angle = cast::<_, T>(-1 * (i * k) as isize).unwrap() *
                        cast(2.0 * f32::consts::PI).unwrap() /
                        cast(signal.len()).unwrap();
            let twiddle = Complex::from_polar(&One::one(), &angle);
            sum = sum + twiddle * x;
        }
        *spec_bin = sum;
    }
}

/// Factors an integer into its prime factors.
fn factor(n: usize) -> Vec<(usize, usize)> {
    let mut factors = Vec::new();
    let mut next = n;
    while next > 1 {
        for div in 2..next + 1 {
            if next % div == 0 {
                next = next / div;
                factors.push((div, next));
                break;
            }
        }
    }
    return factors;
}
