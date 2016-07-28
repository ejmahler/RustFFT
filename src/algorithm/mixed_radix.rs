use std::f32;

use num::{Complex, FromPrimitive, Signed, Zero};

use algorithm::FFTAlgorithm;
use butterflies::{butterfly_2, butterfly_3, butterfly_4, butterfly_5, butterfly};

pub struct CooleyTukey<T> {
    twiddles: Vec<Complex<T>>,
    factors: Vec<usize>,
    scratch: Vec<Complex<T>>,
    inverse: bool,
}

impl<T> CooleyTukey<T>
    where T: Signed + FromPrimitive + Copy
{
    pub fn new(len: usize, factors: &[(usize, usize)], inverse: bool) -> Self {
        let dir = if inverse {
            1
        } else {
            -1
        };
        let max_fft_len = factors.iter().map(|&(a, _)| a).max();
        let scratch = match max_fft_len {
            None | Some(0...5) => vec![Zero::zero(); 0],
            Some(l) => vec![Zero::zero(); l],
        };

        CooleyTukey {
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
            factors: CooleyTukey::<T>::expand_factors(factors),
            scratch: scratch,
            inverse: inverse,
        }
    }

    fn expand_factors(factors: &[(usize, usize)]) -> Vec<usize> {
        let count = factors.iter().map(|&(_, count)| count).fold(0, |acc, x| acc + x);

        let mut expanded_factors = Vec::with_capacity(count);
        for &(factor, count) in factors {
            for _ in 0..count {
                expanded_factors.push(factor);
            }
        }

        expanded_factors
    }
}

impl<T> FFTAlgorithm<T> for CooleyTukey<T>
    where T: Signed + FromPrimitive + Copy
{
    /// Runs the FFT on the input `signal` array, placing the output in the 'spectrum' array
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        cooley_tukey(signal.len(), signal,
                     spectrum,
                     1,
                     self.twiddles.as_slice(),
                     self.factors.as_slice(),
                     self.scratch.as_mut_slice(),
                     self.inverse);
    }
}

fn cooley_tukey<T>(size: usize,
                   signal: &[Complex<T>],
                   spectrum: &mut [Complex<T>],
                   stride: usize,
                   twiddles: &[Complex<T>],
                   factors: &[usize],
                   scratch: &mut [Complex<T>],
                   inverse: bool)
    where T: Signed + FromPrimitive + Copy
{
    if let Some(&n1) = factors.first() {
        let n2 = size / n1;
        if n2 == 1 {
            // we theoretically need to compute n1 ffts of size n2
            // but n2 is 1, and a fft of size 1 is just a copy
            copy_data(signal, spectrum, stride);
        } else {
            // Recursive call to perform n1 ffts of length n2
            for i in 0..n1 {
                cooley_tukey(n2,
                             &signal[i * stride..],
                             &mut spectrum[i * n2..],
                             stride * n1,
                             twiddles,
                             &factors[1..],
                             scratch,
                             inverse);
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

fn copy_data<T: Copy>(signal: &[Complex<T>], spectrum: &mut [Complex<T>], stride: usize) {
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
