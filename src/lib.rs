#![cfg_attr(all(feature = "bench", test), feature(test))]

//! RustFFT allows users to compute arbitrary-sized FFTs in O(nlogn) time.
//!
//! The recommended way to use RustFFT is to create a [`FFTplanner`](struct.FFTplanner.html) instance and then call its
//! `plan_fft` method. This method will automatically choose which FFT algorithms are best
//! for a given size and initialize the required buffers and precomputed data.
//!
//! ```
//! // Perform a forward FFT of size 1234
//! use rustfft::{FFTplanner, num_complex::Complex};
//!
//! let mut planner = FFTplanner::new(false);
//! let fft = planner.plan_fft(1234);
//!
//! let mut input:  Vec<Complex<f32>> = vec![Complex{ re: 0.0, im: 0.0 }; 4096];
//! let mut output: Vec<Complex<f32>> = vec![Complex{ re: 0.0, im: 0.0 }; 4096];
//!
//! fft.process(&mut input, &mut output);
//! ```
//! The planner returns trait objects of the [`FFT`](trait.FFT.html) trait, allowing for FFT sizes that aren't known
//! until runtime.
//!
//! RustFFT also exposes individual FFT algorithms. If you know beforehand that you need a power-of-two FFT, you can
//! avoid the overhead of the planner and trait object by directly creating instances of the Radix4 algorithm:
//!
//! ```
//! // Computes a forward FFT of size 4096
//! use rustfft::{FFT, algorithm::Radix4, num_complex::Complex};
//!
//! let fft = Radix4::new(4096, false);
//!
//! let mut input:  Vec<Complex<f32>> = vec![Complex{ re: 0.0, im: 0.0 }; 4096];
//! let mut output: Vec<Complex<f32>> = vec![Complex{ re: 0.0, im: 0.0 }; 4096];
//!
//! fft.process(&mut input, &mut output);
//! ```
//!
//! For the vast majority of situations, simply using the [`FFTplanner`](struct.FFTplanner.html) will be enough, but
//! advanced users may have better insight than the planner into which algorithms are best for a specific size. See the
//! [`algorithm`](algorithm/index.html) module for a complete list of algorithms implemented by RustFFT.
//!
//! ### Normalization
//!
//! RustFFT does not normalize outputs. Callers must manually normalize the results by scaling each element by
//! `1/len().sqrt()`. Multiple normalization steps can be merged into one via pairwise multiplication, so when
//! doing a forward FFT followed by an inverse FFT, callers can normalize once by scaling each element by `1/len()`
//!
//! ### Output Order
//!
//! Elements in the output are ordered by ascending frequency, with the first element corresponding to frequency 0.

#![allow(unknown_lints)] // The "bare trait objects" lint is unknown on rustc 1.26
#![allow(bare_trait_objects)]

pub use num_complex;
pub use num_traits;

/// Individual FFT algorithms
pub mod algorithm;
mod array_utils;
mod common;
mod math_utils;
mod plan;
mod twiddles;

use num_complex::Complex;

pub use crate::common::FFTnum;
pub use crate::plan::FFTplanner;

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
pub trait FFT<T: FFTnum>: Length + IsInverse + Sync + Send {
    /// Computes an FFT on the `input` buffer and places the result in the `output` buffer.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`. Multiple normalization steps can be merged into one via pairwise multiplication, so when
    /// doing a forward FFT followed by an inverse FFT, callers can normalize once by scaling each element by `1/len()`
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);

    /// Divides the `input` and `output` buffers into chunks of length self.len(), then computes an FFT on each chunk.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`. Multiple normalization steps can be merged into one via pairwise multiplication, so when
    /// doing a forward FFT followed by an inverse FFT, callers can normalize once by scaling each element by `1/len()`
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);
}

#[cfg(test)]
mod test_utils;
