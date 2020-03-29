#![cfg_attr(all(feature = "bench", test), feature(test))]

//! RustFFT allows users to compute arbitrary-sized FFTs in O(nlogn) time.
//!
//! The recommended way to use RustFFT is to create a [`FFTplanner`](struct.FFTplanner.html) instance and then call its
//! `plan_fft` method. This method will automatically choose which FFT algorithms are best
//! for a given size and initialize the required buffers and precomputed data.
//!
//! ```
//! // Perform a forward FFT of size 1234
//! use std::sync::Arc;
//! use rustfft::FFTplanner;
//! use rustfft::num_complex::Complex;
//! use rustfft::num_traits::Zero;
//!
//! let mut input:  Vec<Complex<f32>> = vec![Complex::zero(); 1234];
//! let mut output: Vec<Complex<f32>> = vec![Complex::zero(); 1234];
//!
//! let mut planner = FFTplanner::new(false);
//! let fft = planner.plan_fft(1234);
//! fft.process(&mut input, &mut output);
//! 
//! // The fft instance returned by the planner is stored behind an `Arc`, so it's cheap to clone
//! let fft_clone = Arc::clone(&fft);
//! ```
//! The planner returns trait objects of the [`Fft`](trait.Fft.html) trait, allowing for FFT sizes that aren't known
//! until runtime.
//! 
//! RustFFT also exposes individual FFT algorithms. If you know beforehand that you need a power-of-two you can
//! avoid the overhead of the planner and trait object by directly creating instances of the Radix4 algorithm:
//!
//! ```
//! // Computes a forward FFT of size 4096
//! use rustfft::algorithm::Radix4;
//! use rustfft::Fft;
//! use rustfft::num_complex::Complex;
//! use rustfft::num_traits::Zero;
//!
//! let mut input:  Vec<Complex<f32>> = vec![Complex::zero(); 4096];
//! let mut output: Vec<Complex<f32>> = vec![Complex::zero(); 4096];
//!
//! let fft = Radix4::new(4096, false);
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
//! doing a forward FFT followed by an inverse callers can normalize once by scaling each element by `1/len()`
//!
//! ### Output Order
//!
//! Elements in the output are ordered by ascending frequency, with the first element corresponding to frequency 0.

#![allow(unknown_lints)] // The "bare trait objects" lint is unknown on rustc 1.26
#![allow(bare_trait_objects)]
#![feature(specialization)]

pub extern crate num_complex;
pub extern crate num_traits;
extern crate num_integer;
extern crate strength_reduce;
extern crate transpose;


#[macro_use]
mod common;

/// Individual FFT algorithms
pub mod algorithm;
mod math_utils;
mod array_utils;
mod plan;
mod twiddles;

use num_complex::Complex;
use num_traits::Zero;

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
pub trait Fft<T: FFTnum>: Length + IsInverse + Sync + Send {
    /// Computes an FFT on the `input` buffer and places the result in the `output` buffer.
    ///
    /// The output is not normalized. Callers must manually normalize the results by scaling each element by
    /// `1/len().sqrt()`. Multiple normalization steps can be merged into one via pairwise multiplication, so when
    /// doing a forward FFT followed by an inverse callers can normalize once by scaling each element by `1/len()`
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_out_of_place_scratch_len()];
        self.process_with_scratch(input, output, &mut scratch);
    }
    fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_inplace_scratch_len()];
        self.process_inplace_with_scratch(buffer, &mut scratch);
    }

    fn process_with_scratch(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]);
    fn process_inplace_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]);
    fn process_inplace_multi(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);


    fn get_inplace_scratch_len(&self) -> usize;
    fn get_out_of_place_scratch_len(&self) -> usize;
}

#[cfg(test)]
extern crate rand;
#[cfg(test)]
mod test_utils;
