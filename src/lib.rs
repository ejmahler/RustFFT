extern crate num;

use num::{Complex, Zero};
use std::iter::{repeat, range_step_inclusive, range_step};
use std::f32;
use stride::{Stride, StrideMut};

mod stride;

pub struct FFT {
    scratch: Vec<Complex<f32>>,
    factors: Vec<usize>,
}

impl FFT {
    pub fn new(len: usize) -> Self {
        FFT {
            scratch: repeat(Zero::zero()).take(len).collect(),
            factors: factor(len),
        }
    }

    pub fn process(&mut self, signal: &[Complex<f32>], spectrum: &mut [Complex<f32>]) {
        cooley_tukey(Stride::from_slice(signal),
                     StrideMut::from_slice(spectrum),
                     StrideMut::from_slice(self.scratch.as_mut_slice()),
                     self.factors.as_slice());
    }
}

fn cooley_tukey<'a>(signal: Stride<'a, Complex<f32>>,
                    spectrum: StrideMut<'a, Complex<f32>>,
                    scratch: StrideMut<'a, Complex<f32>>,
                    factors: &[usize]) {
    match factors {
        [_] | [] => dft(signal, spectrum),
        [n1, other_factors..] => {
            let n2 = signal.len() / n1;
            for i in range(0, n1)
            {
                cooley_tukey(signal.skip_some(i).stride(n1),
                             scratch.skip_some(i).stride(n1),
                             spectrum.skip_some(i).stride(n1),
                             other_factors);
            }

            multiply_by_twiddles(scratch.clone(), n1, n2);

            for (i, offset) in range_step(0us, scratch.len(), n1).enumerate() {
                let row = scratch.skip_some(offset).take_some(n1);
                dft_mut(row, spectrum.skip_some(i).stride(n2));
            }
        }
    }
}

fn multiply_by_twiddles<'a, I>(xs: I, n1: usize, n2: usize) where I: Iterator<Item=&'a mut Complex<f32>>
{
    for (i, elt) in xs.enumerate() {
        let k2 = i / n1;
        let k1 = i % n1;
        let angle: f32 = (-1is as f32) * ((k2 * k1) as f32) * f32::consts::PI_2 / ((n1 * n2) as f32);
        let twiddle: Complex<f32> = Complex::from_polar(&1f32, &angle);
        *elt = *elt * twiddle;
    }
}

//TODO this suffers from accumulation of error in calcualtion of `angle`
pub fn dft<'a, 'b, I, O>(signal: I, spectrum: O)
where I: Iterator<Item=&'a Complex<f32>> + ExactSizeIterator + Clone,
      O: Iterator<Item=&'b mut Complex<f32>>
{
    for (k, spec_bin) in spectrum.enumerate()
    {
        let mut signal_iter = signal.clone();
        let mut sum: Complex<f32> = Zero::zero();
        let mut angle = 0f32;
        let rad_per_sample = (k as f32) * f32::consts::PI_2 / (signal.len() as f32);
        for &x in signal_iter
        {
            let twiddle: Complex<f32> = Complex::from_polar(&1f32, &angle);
            sum = sum + twiddle * x;
            angle = angle - rad_per_sample;
        }
        *spec_bin = sum;
    }
}

//TODO this suffers from accumulation of error in calcualtion of `angle`
//TODO how can we avoid this duplication?
pub fn dft_mut<'a, 'b, I, O>(signal: I, spectrum: O)
where I: Iterator<Item=&'a mut Complex<f32>> + ExactSizeIterator + Clone,
      O: Iterator<Item=&'b mut Complex<f32>>
{
    for (k, spec_bin) in spectrum.enumerate()
    {
        let mut signal_iter = signal.clone();
        let mut sum: Complex<f32> = Zero::zero();
        let mut angle = 0f32;
        let rad_per_sample = (k as f32) * f32::consts::PI_2 / (signal.len() as f32);
        for &mut x in signal_iter
        {
            let twiddle: Complex<f32> = Complex::from_polar(&1f32, &angle);
            sum = sum + twiddle * x;
            angle = angle - rad_per_sample;
        }
        *spec_bin = sum;
    }
}

fn factor(n: usize) -> Vec<usize>
{
    let mut factors: Vec<usize> = Vec::new();
    let mut next = n;
    while next > 1
    {
        for div in Some(2us).into_iter().chain(range_step_inclusive(3us, next, 2))
        {
            if next % div == 0
            {
                next = next / div;
                factors.push(div);
                break;
            }
        }
    }
    return factors;
}
