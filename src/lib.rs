#![cfg_attr(all(feature = "bench", test), feature(test))]

extern crate num;

pub mod algorithm;
mod math_utils;
mod array_utils;
mod plan;
mod twiddles;
mod common;

use num::{Complex, Zero};
use std::rc::Rc;

use plan::Planner;

pub use common::FFTnum;

pub struct FFT<T> {
    len: usize,
    algorithm: Rc<algorithm::FFTAlgorithm<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: common::FFTnum> FFT<T> {
    /// Creates a new FFT context that will process signal of length
    /// `len`. If `inverse` is `true`, then this struct will run inverse
    /// FFTs. This implementation of the FFT doesn't do any scaling on both
    /// the forward and backward transforms, so doing a forward then backward
    /// FFT on a signal will scale the signal by its length.
    pub fn new(len: usize, inverse: bool) -> Self {

        let mut planner = Planner::new(inverse);

        FFT {
            len: len,
            algorithm: planner.plan_fft(len),
            scratch: vec![Zero::zero(); len],
        }
    }

    /// Runs the FFT on the input `signal` buffer, and places the output in the
    /// `spectrum` buffer.
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        common::verify_length(signal, spectrum, self.len);

        self.scratch.copy_from_slice(signal);

        self.algorithm.process(&mut self.scratch, spectrum);
    }
}

#[cfg(test)]
extern crate rand;
#[cfg(test)]
mod test_utils;