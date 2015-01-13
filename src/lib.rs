extern crate num;

use num::{Complex, Zero};
use std::iter::{repeat, range_step_inclusive, range_step};
use std::f32;
use stride::{Stride, StrideMut};

mod stride;

pub struct FFT {
    scratch: Vec<Complex<f32>>,
    factors: Vec<usize>,
    twiddles: TwiddlePowerTable,
}

impl FFT {
    pub fn new(len: usize) -> Self {
        FFT {
            scratch: repeat(Zero::zero()).take(len).collect(),
            factors: factor(len),
            twiddles: TwiddlePowerTable::new(len),
        }
    }

    pub fn process(&mut self, signal: &[Complex<f32>], spectrum: &mut [Complex<f32>]) {
        cooley_tukey(Stride::from_slice(signal),
                     StrideMut::from_slice(spectrum),
                     StrideMut::from_slice(self.scratch.as_mut_slice()),
                     &self.twiddles,
                     self.factors.as_slice());
    }
}

fn cooley_tukey<'a>(signal: Stride<'a, Complex<f32>>,
                    spectrum: StrideMut<'a, Complex<f32>>,
                    scratch: StrideMut<'a, Complex<f32>>,
                    twiddles: &TwiddlePowerTable,
                    factors: &[usize]) {
    match factors {
        [_] | [] => dft_precalc_twiddles(signal, spectrum, twiddles),
        [n1, other_factors..] => {
            let n2 = signal.len() / n1;
            for i in range(0, n1)
            {
                cooley_tukey(signal.skip_some(i).stride(n1),
                             scratch.skip_some(i).stride(n1),
                             spectrum.skip_some(i).stride(n1),
                             twiddles,
                             other_factors);
            }

            butterfly(scratch, spectrum, twiddles, n1, n2);
        }
    }
}

fn butterfly<'a>(input: StrideMut<'a, Complex<f32>>,
                 output: StrideMut<'a, Complex<f32>>,
                 twiddles: &TwiddlePowerTable,
                 num_cols: usize,
                 num_rows: usize) {
    for (i, offset) in range_step(0us, input.len(), num_cols).enumerate() {
        let in_row = input.skip_some(offset).take_some(num_cols);
        let out_col = output.skip_some(i).stride(num_rows);
        for (j, spec_bin) in out_col.enumerate() {
            *spec_bin = Zero::zero();
            let in_row_copy = in_row.clone();
            for (k, sig_bin) in in_row_copy.enumerate() {
                let twiddle_power = i * k + j * k * num_rows;
                let twiddle = twiddles.get_twiddle(twiddle_power, num_rows * num_cols);
                *spec_bin = *spec_bin + twiddle * *sig_bin;
            }
        }
    }
}

fn dft_precalc_twiddles<'a, 'b, I, O>(signal: I, spectrum: O, twiddles: &TwiddlePowerTable)
where I: Iterator<Item=&'a Complex<f32>> + ExactSizeIterator + Clone,
      O: Iterator<Item=&'b mut Complex<f32>>
{
    let n = signal.len();
    for (k, spec_bin) in spectrum.enumerate()
    {
        let mut signal_iter = signal.clone();
        *spec_bin = Zero::zero();
        for (i, &x) in signal_iter.enumerate() {
            let twiddle = twiddles.get_twiddle(i * k, n);
            *spec_bin = *spec_bin + twiddle * x;
        }
    }
}

pub fn dft<'a, 'b, I, O>(signal: I, spectrum: O)
where I: Iterator<Item=&'a Complex<f32>> + ExactSizeIterator + Clone,
      O: Iterator<Item=&'b mut Complex<f32>>
{
    for (k, spec_bin) in spectrum.enumerate()
    {
        let mut signal_iter = signal.clone();
        let mut sum: Complex<f32> = Zero::zero();
        for (i, &x) in signal_iter.enumerate() {
            let angle = -1. * ((i * k) as f32) * f32::consts::PI_2 / (signal.len() as f32);
            let twiddle: Complex<f32> = Complex::from_polar(&1f32, &angle);
            sum = sum + twiddle * x;
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

/// Holds the powers of the `n`th root of unity
struct TwiddlePowerTable {
    n: usize,
    twiddles: Vec<Complex<f32>>,
}

impl TwiddlePowerTable {
    /// Calculates the powers of the `n`th root of unity
    fn new(n: usize) -> Self {
        TwiddlePowerTable {
            n: n,
            twiddles: range(0, n)
                      .map(|i| -1. * (i as f32) * f32::consts::PI_2 / (n as f32))
                      .map(|phase| Complex::from_polar(&1., &phase))
                      .collect(),
        }
    }

    /// Returns the `base`th root of unity raised to `power`.
    ///
    /// `base` must be a divisor of the value of `n` for which the table was calculated.
    #[inline]
    fn get_twiddle(&self, power: usize, base: usize) -> &Complex<f32> {
        unsafe {
            self.twiddles.get_unchecked((power * self.n / base) % self.n)
        }
    }
}
