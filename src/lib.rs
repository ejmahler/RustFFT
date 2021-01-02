#![cfg_attr(all(feature = "bench", test), feature(test))]

//! RustFFT is a high-performance FFT library written in pure Rust.
//!
//! RustFFT supports the AVX instruction set for increased performance. No special code is needed to activate AVX:
//! Simply plan a FFT using the FftPlanner on a machine that supports the `avx` and `fma` CPU features, and RustFFT
//! will automatically switch to faster AVX-accelerated algorithms.
//!
//! ### Usage
//!
//! The recommended way to use RustFFT is to create a [`FftPlanner`](crate::FftPlanner) instance and then call its
//! [`plan_fft`](crate::FftPlanner::plan_fft) method. This method will automatically choose which FFT algorithms are best
//! for a given size and initialize the required buffers and precomputed data.
//!
//! ```
//! // Perform a forward FFT of size 1234
//! use rustfft::{FftPlanner, num_complex::Complex};
//!
//! let mut planner = FftPlanner::new();
//! let fft = planner.plan_fft_forward(1234);
//!
//! let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
//! fft.process_inplace(&mut buffer);
//! ```
//! The planner returns trait objects of the [`Fft`](crate::Fft) trait, allowing for FFT sizes that aren't known
//! until runtime.
//!
//! RustFFT also exposes individual FFT algorithms. For example, if you know beforehand that you need a power-of-two FFT, you can
//! avoid the overhead of the planner and trait object by directly creating instances of the [`Radix4`](crate::algorithm::Radix4) algorithm:
//!
//! ```
//! // Computes a forward FFT of size 4096
//! use rustfft::{Fft, FftDirection, num_complex::Complex, algorithm::Radix4};
//!
//! let fft = Radix4::new(4096, FftDirection::Forward);
//!
//! let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 4096];
//! fft.process_inplace(&mut buffer);
//! ```
//!
//! For the vast majority of situations, simply using the [`FftPlanner`](crate::FftPlanner) will be enough, but
//! advanced users may have better insight than the planner into which algorithms are best for a specific size. See the
//! [`algorithm`](crate::algorithm) module for a complete list of scalar algorithms implemented by RustFFT.
//!
//! Users should beware, however, that bypassing the planner will disable all AVX optimizations.
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
//!
//! ### AVX Performance Tips
//!
//! In any FFT computation, the time required to compute a FFT of size N relies heavily on the [prime factorization](https://en.wikipedia.org/wiki/Integer_factorization) of N.
//! If N's prime factors are all very small, computing a FFT of size N will be fast, and it'll be slow if N has large prime
//! factors, or if N is a prime number.
//!
//! In most FFT libraries (Including RustFFT when using non-AVX code), power-of-two FFT sizes are the fastest, and users see a steep
//! falloff in performance when using non-power-of-two sizes. Thankfully, RustFFT using AVX acceleration is not quite as restrictive:
//!
//! - Any FFT whose size is of the form `2^n * 3^m` can be considered the "fastest" in RustFFT.
//! - Any FFT whose prime factors are all 11 or smaller will also be very fast, but the fewer the factors of 2 and 3 the slower it will be.
//!     For example, computing a FFT of size 13552 (2^4 * 7 * 11 * 11) is takes 12% longer to compute than 13824 (2^9 * 3^3),
//!     and computing a FFT of size 2541 (3*7*11*11) is takes 65% longer to compute than 2592 (2^5 * 3^4)
//! - Any other FFT size will be noticeably slower. A considerable amount of effort has been put into making these FFT sizes as fast as
//!     they can be, but some FFT sizes just take more work than others. For example, computing a FFT of size 5183 (71 * 73) takes about
//!     5x longer than computing a FFT of size 5184 (2^6 * 3^4).
//!
//! In most cases, even prime-sized FFTs will be fast enough for your application. In the example of 5183 above, even that "slow" FFT
//! only takes a few tens of microseconds to compute.
//!
//! Our advice is to start by trying the size that's most convenient to your application.
//! If that's too slow, see if you can find a nearby size whose prime factors are all 11 or smaller, and you can expect a 2x-5x speedup.
//! If that's still too slow, find a nearby size whose prime factors are all 2 or 3, and you can expect a 1.1x-1.5x speedup.

use std::fmt::Display;

pub use num_complex;
pub use num_traits;

#[macro_use]
mod common;

/// Individual FFT algorithms
pub mod algorithm;
mod array_utils;
mod fft_cache;
mod math_utils;
mod plan;
mod twiddles;

use num_complex::Complex;
use num_traits::Zero;

pub use crate::common::FftNum;
pub use crate::plan::{FftPlanner, FftPlannerScalar};

/// A trait that allows FFT algorithms to report their expected input/output size
pub trait Length {
    /// The FFT size that this algorithm can process
    fn len(&self) -> usize;
}

/// Represents a FFT direction, IE a forward FFT or an inverse FFT
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FftDirection {
    Forward,
    Inverse,
}
impl FftDirection {
    /// Returns the opposite direction of `self`.
    ///
    ///  - If `self` is `FftDirection::Forward`, returns `FftDirection::Inverse`
    ///  - If `self` is `FftDirection::Inverse`, returns `FftDirection::Forward`
    pub fn reverse(&self) -> FftDirection {
        match self {
            Self::Forward => Self::Inverse,
            Self::Inverse => Self::Forward,
        }
    }
}
impl Display for FftDirection {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        match self {
            Self::Forward => f.write_str("Forward"),
            Self::Inverse => f.write_str("Inverse"),
        }
    }
}

/// A trait that allows FFT algorithms to report whether they compute forward FFTs or inverse FFTs
pub trait Direction {
    /// Returns false if this instance computes forward FFTs, true for inverse FFTs
    fn fft_direction(&self) -> FftDirection;
}

/// Trait for algorithms that compute FFTs.
///
/// This trait has two main methods:
/// - [`process_inplace(buffer)`](crate::Fft::process_inplace) computes a FFT using `buffer` as input and store the result back into `buffer`.
/// - [`process(input, output)`](crate::Fft::process) computes a FFT using `input` as input and store the result into `output`.
///
/// Both methods may need to allocate additional scratch space. If you'd like re-use that allocation across multiple FFT computations, call
/// `process_inplace_with_scratch` or `process_with_scratch`, respectively.
pub trait Fft<T: FftNum>: Length + Direction + Sync + Send {
    /// Computes a FFT.
    ///
    /// Convenience method that allocates the required scratch space and and calls `self.process_with_scratch`.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `input.len() != self.len()`
    /// - `output.len() != self.len()`
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_out_of_place_scratch_len()];
        self.process_with_scratch(input, output, &mut scratch);
    }

    /// Computes a FFT.
    ///
    /// Convenience method that allocates the required scratch space and calls `self.process_inplace_with_scratch`.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() != self.len()`
    fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_inplace_scratch_len()];
        self.process_inplace_with_scratch(buffer, &mut scratch);
    }

    /// Computes a FFT.
    ///
    /// Uses both the `input` buffer and `scratch` buffer as scratch space, so the contents of both should be
    /// considered garbage after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `input.len() != self.len()`
    /// - `output.len() != self.len()`
    /// - `scratch.len() < self.get_out_of_place_scratch_len()`
    fn process_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );

    /// Computes a FFT, in-place.
    ///
    /// Uses the `scratch` buffer as scratch space, so the contents of `scratch` should be considered garbage
    /// after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() != self.len()`
    /// - `scratch.len() < self.get_inplace_scratch_len()`
    fn process_inplace_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    /// Computes multiple FFTs.
    ///
    /// Divides `input` and `output` into chunks of size `self.len()`, computes an FFT on each input chunk,
    /// and stores the result in the corresponding output chunk.
    ///
    /// This method uses both the `input` buffer and `scratch` buffer as scratch space, so the contents of both should
    /// be considered garbage after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `input.len() % self.len() != 0`
    /// - `output.len() != input.len()`
    /// - `scratch.len() < self.get_out_of_place_scratch_len()`
    fn process_multi(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );

    /// Computes multiple FFTs, in-place.
    ///
    /// Divides `buffer` into chunks of size `self.len()`, computes an FFT on each chunk, and stores the result back
    /// into `buffer`.
    ///
    /// This method uses the `scratch` buffer as scratch space, so its contents should be considered garbage after
    /// calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() % self.len() != 0`
    /// - `scratch.len() < self.get_inplace_scratch_len()`
    fn process_inplace_multi(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    /// Returns the size of the scratch buffer required by `process_inplace_with_scratch` and `process_inplace_multi`
    fn get_inplace_scratch_len(&self) -> usize;

    /// Returns the size of the scratch buffer required by `process_with_scratch` and `process_multi`
    fn get_out_of_place_scratch_len(&self) -> usize;
}

// Algorithms implemented to use AVX instructions. Only compiled on x86_64.
#[cfg(target_arch = "x86_64")]
mod avx;

// When we're not on avx, keep a stub implementation around that just does nothing
#[cfg(not(target_arch = "x86_64"))]
mod avx {
    pub mod avx_planner {
        use crate::{Fft, FftNum};
        use std::sync::Arc;
        pub struct FftPlannerAvx<T: FftNum> {
            _phantom: std::marker::PhantomData<T>,
        }
        impl<T: FftNum> FftPlannerAvx<T> {
            pub fn new(_direction: FftDirection) -> Result<Self, ()> {
                Err(())
            }
            pub fn plan_fft(&mut self, _len: usize) -> Arc<dyn Fft<T>> {
                unreachable!();
            }
        }
    }
}

pub use self::avx::avx_planner::FftPlannerAvx;

#[cfg(test)]
mod test_utils;
