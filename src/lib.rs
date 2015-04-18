#![cfg_attr(test, feature(test))]

extern crate num;

mod butterflies;

use num::{Complex, Zero, Float};
use num::traits::cast;
use std::iter::repeat;
use std::f32;

use butterflies::{butterfly_2, butterfly_3, butterfly_4, butterfly_5};

pub struct FFT<T> {
    factors: Vec<(usize, usize)>,
    twiddles: Vec<Complex<T>>,
    inverse: bool,
}

impl<T: Float> FFT<T> {
    pub fn new(len: usize, inverse: bool) -> Self {
        let dir = if inverse { 1 } else { -1 };
        FFT::<T> {
            factors: factor(len),
            twiddles: (0..len)
                      .map(|i| cast::<_, T>(dir * i as isize).unwrap()
                               * cast(2.0 * f32::consts::PI).unwrap()
                               / cast(len).unwrap())
                      .map(|phase| Complex::<T>::from_polar(&cast(1).unwrap(), &phase))
                      .collect(),
            inverse: inverse,
        }
    }

    pub fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        debug_assert!(signal.len() == spectrum.len());
        debug_assert!(signal.len() == self.twiddles.len());
        cooley_tukey(signal, spectrum, 1, &self.twiddles[..], &self.factors[..], self.inverse);
    }
}

fn cooley_tukey<T: Float>(signal: &[Complex<T>],
                          spectrum: &mut [Complex<T>],
                          stride: usize,
                          twiddles: &[Complex<T>],
                          factors: &[(usize, usize)],
                          inverse: bool) {
    if let Some(&(n1, n2)) = factors.first() {
        if n2 == 1 {
            // An FFT of length 1 is just the identity operator
            let mut spectrum_idx = 0usize;
            let mut signal_idx = 0usize;
            while signal_idx < signal.len() {
                unsafe { *spectrum.get_unchecked_mut(spectrum_idx) =
                    *signal.get_unchecked(signal_idx); }
                spectrum_idx += 1;
                signal_idx += stride;
            }
        } else {
            // Recursive call to perform n1 ffts of length n2
            for i in (0..n1) {
                cooley_tukey(&signal[i * stride..],
                             &mut spectrum[i * n2..],
                             stride * n1, twiddles, &factors[1..],
                             inverse);
            }
        }

        match n1 {
            5 => unsafe { butterfly_5(spectrum, stride, twiddles, n2) },
            4 => unsafe { butterfly_4(spectrum, stride, twiddles, n2, inverse) },
            3 => unsafe { butterfly_3(spectrum, stride, twiddles, n2) },
            2 => unsafe { butterfly_2(spectrum, stride, twiddles, n2) },
            _ => butterfly(spectrum, stride, twiddles, n2, n1),
        }
    }
}

fn butterfly<T: Float>(data: &mut [Complex<T>], stride: usize,
                       twiddles: &[Complex<T>], num_ffts: usize, fft_len: usize) {

    // TODO pre-allocate this space at FFT initialization
    let mut scratch: Vec<Complex<T>> = repeat(Zero::zero()).take(fft_len).collect();

    // for each fft we have to perform...
    for fft_idx in (0..num_ffts) {

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

pub fn dft<'a, 'b, I, O>(signal: I, spectrum: O)
where I: Iterator<Item=&'a Complex<f32>> + ExactSizeIterator + Clone,
      O: Iterator<Item=&'b mut Complex<f32>>
{
    for (k, spec_bin) in spectrum.enumerate()
    {
        let signal_iter = signal.clone();
        let mut sum: Complex<f32> = Zero::zero();
        for (i, &x) in signal_iter.enumerate() {
            let angle = -1. * ((i * k) as f32) * 2.0 * f32::consts::PI / (signal.len() as f32);
            let twiddle: Complex<f32> = Complex::from_polar(&1f32, &angle);
            sum = sum + twiddle * x;
        }
        *spec_bin = sum;
    }
}

fn factor(n: usize) -> Vec<(usize, usize)>
{
    let mut factors = Vec::new();
    let mut next = n;
    while next > 1 {
        for div in (2..next + 1) {
            if next % div == 0 {
                next = next / div;
                factors.push((div, next));
                break;
            }
        }
    }
    return factors;
}
