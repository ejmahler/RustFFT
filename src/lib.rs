#![cfg_attr(all(feature = "bench", test), feature(test))]

extern crate num;

mod butterflies;

use num::{Complex, Zero, One, Float, Num, FromPrimitive, Signed};
use num::traits::cast;
use std::f32;

use butterflies::{butterfly_2, butterfly_3, butterfly_4, butterfly_5};

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
                copy_data_for_radix4(signal.len(), signal, spectrum, 1, 0, 0);
                radix_4(signal.len(),
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
            Algorithm::Noop => copy_data(signal, spectrum, 1),
        }
    }
}

fn cooley_tukey<T>(signal: &[Complex<T>],
                   spectrum: &mut [Complex<T>],
                   stride: usize,
                   twiddles: &[Complex<T>],
                   factors: &[(usize, usize)],
                   scratch: &mut [Complex<T>],
                   inverse: bool) where T: Signed + FromPrimitive + Copy {
    if let Some(&(n1, n2)) = factors.first() {
        if n2 == 1 {
            // we theoretically need to compute n1 ffts of size n2
            // but n2 is 1, and a fft of size 1 is just a copy
            copy_data(signal, spectrum, stride);
        } else {
            // Recursive call to perform n1 ffts of length n2
            for i in 0..n1 {
                cooley_tukey(&signal[i * stride..],
                             &mut spectrum[i * n2..],
                             stride * n1, twiddles, &factors[1..],
                             scratch, inverse);
            }
        }

        match n1 {
            5 => unsafe { butterfly_5(spectrum, stride, twiddles, n2) },
            4 => unsafe { butterfly_4(spectrum, stride, twiddles, n2, inverse) },
            3 => unsafe { butterfly_3(spectrum, stride, twiddles, n2) },
            2 => unsafe { butterfly_2(spectrum, stride, twiddles, n2) },
            _ => butterfly(spectrum, stride, twiddles, n2, n1, &mut scratch[..n1]),
        }
    }
}

fn radix_4<T>(size: usize,
              spectrum: &mut [Complex<T>],
              stride: usize,
              twiddles: &[Complex<T>],
              inverse: bool) where T: Signed + FromPrimitive + Copy {
    match size {
        4 => unsafe { butterfly_4(spectrum, stride, twiddles, 1, inverse) },
        2 => unsafe { butterfly_2(spectrum, stride, twiddles, 1) },
        _ => {
            for i in 0..4 {
                radix_4(size / 4,
                        &mut spectrum[i * (size / 4)..],
                        stride * 4,
                        twiddles,
                        inverse);
            }
            unsafe { butterfly_4(spectrum, stride, twiddles, size / 4, inverse) };
        }
    }
}

fn butterfly<T: Num + Copy>(data: &mut [Complex<T>],
                            stride: usize,
                            twiddles: &[Complex<T>],
                            num_ffts: usize,
                            fft_len: usize,
                            scratch: &mut [Complex<T>]) {
    // for each fft we have to perform...
    for fft_idx in 0..num_ffts {

        // copy over data into scratch space
        let mut data_idx = fft_idx;
        for s in scratch.iter_mut() {
            *s = unsafe { *data.get_unchecked(data_idx) };
            data_idx += num_ffts;
        }

        // perfom the butterfly from the scratch space into the original buffer
        let mut data_idx = fft_idx;
        while data_idx < fft_len * num_ffts {
            let out_sample = unsafe { data.get_unchecked_mut(data_idx) };
            *out_sample = Zero::zero();
            let mut twiddle_idx = 0usize;
            for in_sample in scratch.iter() {
                let twiddle = unsafe { twiddles.get_unchecked(twiddle_idx) };
                *out_sample = *out_sample + in_sample * twiddle;
                twiddle_idx += stride * data_idx;
                if twiddle_idx >= twiddles.len() { twiddle_idx -= twiddles.len() }
            }
            data_idx += num_ffts;
        }

    }
}

fn copy_data<T: Copy>(signal: &[Complex<T>], spectrum: &mut [Complex<T>], stride: usize)
{
    let mut spectrum_idx = 0usize;
    let mut signal_idx = 0usize;
    while signal_idx < signal.len() {
        unsafe {
            *spectrum.get_unchecked_mut(spectrum_idx) = *signal.get_unchecked(signal_idx);
        }
        spectrum_idx += 1;
        signal_idx += stride;
    }
}

// TODO: we should be able to do this in a simple loop, rather than recursively, using bit reversal
// the algorithm will be a little more complicated though due
// to us potentially using radix 2 for one of the teps
fn copy_data_for_radix4<T: Copy>(size: usize,
                           signal: &[Complex<T>],
                           spectrum: &mut [Complex<T>],
                           stride: usize,
                           signal_offset: usize,
                           spectrum_offset: usize)
{
    match size {
        4 => unsafe {
            for i in 0..4 {
                *spectrum.get_unchecked_mut(spectrum_offset + i) =
                    *signal.get_unchecked(signal_offset + i * stride);
            }
        },
        2 => unsafe {
            for i in 0..2 {
                *spectrum.get_unchecked_mut(spectrum_offset + i) =
                    *signal.get_unchecked(signal_offset + i * stride);
            }
        },
        _ => {
            for i in 0..4 {
                copy_data_for_radix4(size / 4,
                                     signal,
                                     spectrum,
                                     stride * 4,
                                     signal_offset + i * stride,
                                     spectrum_offset + i * (size / 4));
            }
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
