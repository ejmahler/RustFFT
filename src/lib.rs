extern crate num;

use num::{Complex, Zero};
use std::iter::{repeat, range_step_inclusive};
use std::f32;
use iter_util::stride;

mod iter_util;

pub struct FFT {
    scratch: Vec<Complex<f32>>,
    factors: Vec<uint>,
}

impl FFT {
    pub fn new(len: uint) -> Self {
        FFT {
            scratch: repeat(Zero::zero()).take(len).collect(),
            factors: factor(len),
        }
    }

    pub fn process<'a, 'b, I, O>(&mut self, signal: I, spectrum: O)
    where I: ExactSizeIterator<&'a Complex<f32>> + Clone, O: Iterator<&'b mut Complex<f32>> {
        cooley_tukey(signal, spectrum, self.scratch.iter_mut(), self.factors.as_slice())
    }
}

fn cooley_tukey<'a, 'b, 'c, I, O, S>(signal: I, spectrum: O, scratch: S, factors: &[uint])
where I: ExactSizeIterator<&'a Complex<f32>> + Clone,
      O: Iterator<&'b mut Complex<f32>>,
      S: Iterator<&'c mut Complex<f32>> {
    match factors {
        [1] | [] => dft(signal, spectrum),
        [n1, other_factors..] => (), //TODO add recursive calls
    }
}

fn multiply_by_twiddles<'a, I: Iterator<&'a mut Complex<f32>>>(xs: I, n1: uint, n2: uint)
{
    for (i, elt) in xs.enumerate() {
        let k2 = i / n1;
        let k1 = i % n1;
        let angle: f32 = (-1i as f32) * ((k2 * k1) as f32) * f32::consts::PI_2 / ((n1 * n2) as f32);
        let twiddle: Complex<f32> = Complex::from_polar(&1f32, &angle);
        *elt = *elt * twiddle;
    }
}

//TODO this suffers from accumulation of error in calcualtion of `angle`
pub fn dft<'a, 'b, I, O>(signal: I, spectrum: O)
where I: ExactSizeIterator<&'a Complex<f32>> + Clone, O: Iterator<&'b mut Complex<f32>>
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

fn factor(n: uint) -> Vec<uint>
{
    let mut factors: Vec<uint> = Vec::new();
    let mut next = n;
    while next > 1
    {
        for div in Some(2u).into_iter().chain(range_step_inclusive(3u, next, 2))
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
