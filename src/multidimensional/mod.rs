//! Traits and utilities for computing multidimensional FFTs
//!
//! RustFFT can compute multidimensional FFTs of arbitrary dimensionality. The same API is used for 2d FFTs, 3D FFTs, and
//! for higher dimensionalities.
//!
//! To plan a multidimensional FFT, create a [`FftPlanner`](crate::FftPlanner) instance and then call its
//! [`plan_fft_multidimensional`](crate::FftPlanner::plan_fft_multidimensional) method. This method takes a shape array defining the multidimensional FFT's
//! exact dimensions and selects appropriate FFT algorithms for the given shape. The planner returns trait objects of the [`FftNd`]
//! trait, allowing for FFT sizes that aren't known until runtime.
//!
//! #### Example: 2D FFT
//!
//! For 2D FFTs, the data is interpreted as row-major order, where `shape[1]` specifies the row width, and `shape[0]` specifies the number of rows.
//!
//! ~~~
//! // Perform a 2d Fft of size 3x4
//! use std::sync::Arc;
//! use rustfft::{FftPlanner, FftDirection, num_complex::Complex};
//!
//! let mut planner = FftPlanner::<f32>::new();
//! let shape = [3, 4];
//! let fft = planner.plan_fft_multidimensional(shape, FftDirection::Forward);
//!
//! // Data layout:
//! // Since shape[1] is 4, each row will contain 4 elements.
//! // Since shape[0] is 3, there will be 3 rows.
//! let mut buffer = vec![
//!      Complex::from(11.0), Complex::from(12.0), Complex::from(13.0), Complex::from(14.0),
//!      Complex::from(21.0), Complex::from(22.0), Complex::from(23.0), Complex::from(24.0),
//!      Complex::from(31.0), Complex::from(32.0), Complex::from(33.0), Complex::from(34.0),
//! ];
//!
//! fft.process(&mut buffer);
//! ~~~
//!
//! #### Example: 3D FFT
//!
//! For 3D FFTs, the data layout is a logical extension of row-major order.
//! - `shape[2]` specifies the row width. In memory, rows are stored contiguously.
//! - `shape[1]` specifies the number of rows. In memory, each row starts where the previous ends, IE a stride of `shape[2]`.
//! - `shape[0]` specifies the number of "groups of rows". In memory, each group of rows starts where the previous group ends, IE a stride of `shape[2] * shape[1]`.
//!
//! ~~~
//! // Perform a 3d Fft of size 2x3x4
//! use std::sync::Arc;
//! use rustfft::{FftPlanner, FftDirection, num_complex::Complex};
//!
//! let mut planner = FftPlanner::<f32>::new();
//! let shape = [2, 3, 4];
//! let fft = planner.plan_fft_multidimensional(shape, FftDirection::Forward);
//!
//! // Data layout:
//! // Since shape[2] is 4, each row will contain 4 elements.
//! // Since shape[1] is 3, there will be 3 rows.
//! // Since shape[0] is 2, there will be 2 groups of rows.
//! let mut buffer = vec![
//!      Complex::from(111.0), Complex::from(112.0), Complex::from(113.0), Complex::from(114.0),
//!      Complex::from(121.0), Complex::from(122.0), Complex::from(123.0), Complex::from(124.0),
//!      Complex::from(131.0), Complex::from(132.0), Complex::from(133.0), Complex::from(134.0),
//!
//!      Complex::from(211.0), Complex::from(212.0), Complex::from(213.0), Complex::from(214.0),
//!      Complex::from(221.0), Complex::from(222.0), Complex::from(223.0), Complex::from(224.0),
//!      Complex::from(231.0), Complex::from(232.0), Complex::from(233.0), Complex::from(234.0),
//! ];
//!
//! fft.process(&mut buffer);
//! ~~~
//!
//! #### Higher Dimensions
//!
//! Higher dimensions follow a logical extension of the requirements for lower dimensions: In the shape array, the
//! last dimension in the shape array should be represented contiguously in memory, and each preceding dimension
//! should be represented in memory with a larger and larger stride.
//!
//! Generally speaking, dimension `i` should have a stride of `shape.iter().skip(i + 1).product()`.
//!
//! #### Degenerate Cases
//!
//! - A dimensionality of 1 is allowed and is exactly equivalent to computing a FFT with the non-multidimensional API.
//! - A dimensionality of 0 is allowed and is a no-op and will always report its len() to be 0.

use num_complex::Complex;
use num_traits::Zero;

use crate::{Direction, FftNum, Length};

pub(crate) mod fft_nd_transpose;

#[cfg(test)]
mod known_test_data;
#[cfg(test)]
pub(crate) mod multidimensional_test_utils;

/// Trait for algorithms that compute multidimensional FFTs.
///
/// This trait has a few methods for computing FFTs. Its most conveinent method is [`process(slice)`](crate::FftNd::process).
/// It takes in a slice of `Complex<T>` and computes a multidimensional FFT on that slice, in-place. It may copy the data over to internal scratch buffers
/// if that speeds up the computation, but the output will always end up in the same slice as the input.
pub trait FftNd<T: FftNum, const DIMENSIONS: usize>: Length + Direction + Sync + Send {
    /// Computes a multi-dimensional FFT in-place.
    ///
    /// Convenience method that allocates a `Vec` with the required scratch space and calls `self.process_with_scratch`.
    /// If you want to re-use that allocation across multiple FFT computations, consider calling `process_with_scratch` instead.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() % self.len_product() > 0`
    /// - `buffer.len() < self.len_product()`
    fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_inplace_scratch_len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    /// Divides `buffer` into chunks of size `self.len()`, and computes a multi-dimensional FFT on each chunk.
    ///
    /// Uses the `scratch` buffer as scratch space, so the contents of `scratch` should be considered garbage
    /// after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() % self.len() > 0`
    /// - `buffer.len() < self.len()`
    /// - `scratch.len() < self.get_inplace_scratch_len()`
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    /// Divides `input` and `output` into chunks of size `self.len()`, and computes a multi-dimensional FFT on each chunk.
    ///
    /// This method uses both the `input` buffer and `scratch` buffer as scratch space, so the contents of both should be
    /// considered garbage after calling.
    ///
    /// This is a more niche way of computing a FFT. It's useful to avoid a `copy_from_slice()` if you need the output
    /// in a different buffer than the input for some reason. This happens frequently in RustFFT internals, but is probably
    /// less common among RustFFT users.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `output.len() != input.len()`
    /// - `input.len() % self.len() > 0`
    /// - `input.len() < self.len()`
    /// - `scratch.len() < self.get_outofplace_scratch_len()`
    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );

    /// Divides `input` and `output` into chunks of `self.len()`, and computes a multi-dimensional FFT on each chunk while
    /// keeping `input` untouched.
    ///
    /// This method uses the `scratch` buffer as scratch space, so the contents should be considered garbage after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `output.len() ! input.len()`
    /// - `input.len() % self.len() > 0`
    /// - `input.len() < self.len()`
    /// - `scratch.len() < get_immutable_scratch_len()`
    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );

    /// Returns the size of the scratch buffer required by `process_with_scratch`
    ///
    /// The returned value may change from one version of RustFFT to the next.
    fn get_inplace_scratch_len(&self) -> usize;

    /// Returns the size of the scratch buffer required by `process_outofplace_with_scratch`
    ///
    /// The returned value may change from one version of RustFFT to the next.
    fn get_outofplace_scratch_len(&self) -> usize;

    /// Returns the size of the scratch buffer required by `process_immutable_with_scratch`
    ///
    /// The returned value may change from one version of RustFFT to the next.
    fn get_immutable_scratch_len(&self) -> usize;

    /// Returns FFT length of each dimension of this multidimensional FFT
    fn shape(&self) -> [usize; DIMENSIONS];
}
