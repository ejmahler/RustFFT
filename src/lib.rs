#![feature(test, core, slice_patterns, step_by)]
extern crate num;
extern crate test;

mod hardcoded_butterflies;

use num::{Complex, Zero};
use std::iter::{repeat, range_step_inclusive};
use std::f32;

use hardcoded_butterflies::{butterfly_2, butterfly_3, butterfly_4, butterfly_5};

pub struct FFT {
    factors: Vec<(usize, usize)>,
    twiddles: Vec<Complex<f32>>,
}

impl FFT {
    pub fn new(len: usize) -> Self {
        FFT {
            factors: factor(len),
            twiddles: (0..len)
                      .map(|i| -1. * (i as f32) * f32::consts::PI_2 / (len as f32))
                      .map(|phase| Complex::from_polar(&1., &phase))
                      .collect(),
        }
    }

    pub fn process(&mut self, signal: &[Complex<f32>], spectrum: &mut [Complex<f32>]) {
        cooley_tukey(signal, spectrum, 1, &self.twiddles[..], &self.factors[..]);
    }
}

fn cooley_tukey(signal: &[Complex<f32>],
                spectrum: &mut [Complex<f32>],
                stride: usize,
                twiddles: &[Complex<f32>],
                factors: &[(usize, usize)]) {
    if let [(n1, n2), other_factors..] = factors {
        if n2 == 1 {
            // An FFT of length 1 is just the identity operator
            let mut spectrum_idx = 0usize;
            for i in (0..signal.len()).step_by(stride) {
                unsafe { *spectrum.get_unchecked_mut(spectrum_idx) = *signal.get_unchecked(i); }
                spectrum_idx += 1;
            }
        } else {
            // Recursive call to perform n1 ffts of length n2
            for i in (0..n1) {
                cooley_tukey(&signal[i * stride..],
                             &mut spectrum[i * n2..],
                             stride * n1, twiddles, other_factors);
            }
        }

        match n1 {
            5 => butterfly_5(spectrum, stride, twiddles, n2),
            4 => butterfly_4(spectrum, stride, twiddles, n2),
            3 => butterfly_3(spectrum, stride, twiddles, n2),
            2 => butterfly_2(spectrum, stride, twiddles, n2),
            _ => butterfly(spectrum, stride, twiddles, n2, n1),
        }
    }
}

fn butterfly(data: &mut [Complex<f32>], stride: usize,
             twiddles: &[Complex<f32>], num_ffts: usize, fft_len: usize) {

    // TODO pre-allocate this space at FFT initialization
    let mut scratch: Vec<Complex<f32>> = repeat(Zero::zero()).take(fft_len).collect();

    // for each fft we have to perform...
    for fft_idx in (0..num_ffts) {

        // copy over data into scratch space
        let mut data_idx = fft_idx;
        for s in scratch.iter_mut() {
            *s = unsafe { *data.get_unchecked(data_idx) };
            data_idx += num_ffts;
        }

        // perfom the butterfly from the scratch space into the original buffer
        for data_idx in (fft_idx..fft_len * num_ffts).step_by(num_ffts) {
            let out_sample = unsafe { data.get_unchecked_mut(data_idx) };
            *out_sample = Zero::zero();
            let mut twiddle_idx = 0usize;
            for in_sample in scratch.iter() {
                let twiddle = unsafe { twiddles.get_unchecked(twiddle_idx) };
                *out_sample = *out_sample + in_sample * twiddle;
                twiddle_idx += stride * data_idx;
                if twiddle_idx >= twiddles.len() { twiddle_idx -= twiddles.len() }
            }
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
            let angle = -1. * ((i * k) as f32) * f32::consts::PI_2 / (signal.len() as f32);
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
        for div in Some(2usize).into_iter().chain(range_step_inclusive(3usize, next, 2)) {
            if next % div == 0 {
                next = next / div;
                factors.push((div, next));
                break;
            }
        }
    }
    return factors;
}
