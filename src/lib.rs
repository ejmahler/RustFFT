#![cfg_attr(all(feature = "bench", test), feature(test))]

extern crate num;

mod butterflies;
mod mixed_radix;
mod radix4;

use num::{Complex, Zero, One, Float, FromPrimitive, Signed};
use num::traits::cast;
use std::f32;

use mixed_radix::cooley_tukey;
use radix4::process_radix4;

enum Algorithm<T> {
    MixedRadix(Vec<(usize, usize)>, Vec<Complex<T>>),
    Radix4,
    Noop,
}

pub struct FFT<T> {
    algorithm: Algorithm<T>,
    twiddles: Vec<Complex<T>>,
    inverse: bool,
}

impl<T> FFT<T> where T: Signed + FromPrimitive + Copy {
    /// Creates a new FFT context that will process signal of length
    /// `len`. If `inverse` is `true`, then this struct will run inverse
    /// FFTs. This implementation of the FFT doesn't do any scaling on both
    /// the forward and backward transforms, so doing a forward then backward
    /// FFT on a signal will scale the signal by its length.
    pub fn new(len: usize, inverse: bool) -> Self {
        let dir = if inverse { 1 } else { -1 };

        let algorithm = if len < 2 {
            Algorithm::Noop
        } else if is_power_of_two(len) {
            Algorithm::Radix4
        } else {
            let factors = factor(len);
            let max_fft_len = factors.iter().map(|&(a, _)| a).max();
            let scratch = match max_fft_len {
                None | Some(0...5) => vec![Zero::zero(); 0],
                Some(l) => vec![Zero::zero(); l],
            };

            Algorithm::MixedRadix(factors, scratch)
        };

        FFT {
            algorithm: algorithm,
            twiddles: (0..len)
                .map(|i| dir as f32 * i as f32 * 2.0 * f32::consts::PI / len as f32)
                .map(|phase| Complex::from_polar(&1.0, &phase))
                .map(|c| {
                    Complex {
                        re: FromPrimitive::from_f32(c.re).unwrap(),
                        im: FromPrimitive::from_f32(c.im).unwrap(),
                    }
                })
                .collect(),
            inverse: inverse,
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
        assert!(signal.len() == self.twiddles.len());

        match self.algorithm {
            Algorithm::Radix4 => {
                process_radix4(signal.len(),
                        signal,
                        spectrum,
                        1,
                        &self.twiddles[..],
                        self.inverse);
            }
            Algorithm::MixedRadix(ref factors, ref mut scratch) => {
                cooley_tukey(signal,
                             spectrum,
                             1,
                             &self.twiddles[..],
                             factors,
                             scratch,
                             self.inverse)
            }
            Algorithm::Noop => {
                for (source, destination) in signal.iter().zip(spectrum.iter_mut()) {
                    *destination = *source;
                }
            },
        }
    }
}

pub fn dft<T: Float>(signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
    for (k, spec_bin) in spectrum.iter_mut().enumerate() {
        let mut sum = Zero::zero();
        for (i, &x) in signal.iter().enumerate() {
            let angle = cast::<_, T>(-1 * (i * k) as isize).unwrap()
                * cast(2.0 * f32::consts::PI).unwrap()
                / cast(signal.len()).unwrap();
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

// returns true if n is a power of 2, false otherwise
fn is_power_of_two(n: usize) -> bool {
    return n & n - 1 == 0;
}
