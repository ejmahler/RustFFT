#![feature(associated_types)]

extern crate num;

use num::{Complex, Zero};
use std::iter::{repeat, range_step_inclusive, range_step};
use std::f32;
use std::ops::{Index, IndexMut};
use stride::{Stride, StrideMut};

mod stride;

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

    //pub fn process<I, IM>(&self, signal: &I, spectrum: &mut IM)
        //where I: Index<uint, Output=Complex<f32>> + Clone,
              //IM: IndexMut<uint, Output=Complex<f32>> + Clone {
        //cooley_tukey(Stride::stride_trivial(signal, self.scratch.len()),
                     //StrideMut::stride_trivial(spectrum, self.scratch.len()),
                     //StrideMut::stride_trivial(self.scratch, self.scratch.len()),
                     //self.factors.as_slice());
    //}

}

fn cooley_tukey<I, IM1, IM2>(signal: Stride<I>, spectrum: StrideMut<IM1>,
                       scratch: StrideMut<IM2>, factors: &[uint])
where I: Index<uint, Output=Complex<f32>> + Clone,
      IM1: IndexMut<uint, Output=Complex<f32>> + Clone,
      IM2: IndexMut<uint, Output=Complex<f32>> + Clone {
    match factors {
        [1] | [] => dft(signal, spectrum),
        [n1, other_factors..] => {
            let n2 = signal.len() / n1;
            for i in range(0, n1)
            {
                cooley_tukey(signal.skip_some(i).stride(n1),
                             spectrum.skip_some(i).stride(n1),
                             scratch.skip_some(i).stride(n1),
                             other_factors);
            }

            multiply_by_twiddles(scratch.clone(), n1, n2);

            for (i, offset) in range_step(0u, scratch.len(), n1).enumerate() {
                let row = scratch.skip_some(offset).take_some(n1);
                dft_mut(row, spectrum.skip_some(i));
            }
        }
    }
}

fn multiply_by_twiddles<'a, I>(xs: I, n1: uint, n2: uint) where I: Iterator<Item=&'a mut Complex<f32>>
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
