#![cfg_attr(all(feature = "bench", test), feature(test))]

pub extern crate num_complex;
pub extern crate num_traits;

/// Individual FFT algorithms
pub mod algorithm;
mod math_utils;
mod array_utils;
mod plan;
mod twiddles;
mod common;

use num_complex::Complex;

pub use plan::FFTplanner;
pub use common::FFTnum;

/// A trait that allows FFT algorithms to report their expected input/output size
pub trait Length {
    /// The FFT size that this algorithm can process
    fn len(&self) -> usize;
}

/// A trait that allows FFT algorithms to report whether they compute forward FFTs or inverse FFTs
pub trait IsInverse {
    /// Returns false if this instance computes forward FFTs, true for inverse FFTs
    fn is_inverse(&self) -> bool;
}

/// An umbrella trait for all available FFT algorithms
pub trait FFT<T: FFTnum>: Length + IsInverse {
    /// Performs an FFT on the `input` buffer, places the result in the `output` buffer.
    /// Uses the `input` buffer as scratch space
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);

    /// Divides the `input` and `output` buffers into self.len() chunks, then runs an FFT
    /// on each chunk. Uses the `input` buffer as scratch space
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);
}

#[cfg(test)]
extern crate rand;
#[cfg(test)]
mod test_utils;