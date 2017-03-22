use num::Complex;
use common::FFTnum;

mod good_thomas_algorithm;
mod mixed_radix;
mod raders_algorithm;
mod radix4;
mod dft;
pub mod butterflies;

pub trait Length {
	// The FFT size that this algorithm can process
    fn len(&self) -> usize;
}

pub trait FFTAlgorithm<T: FFTnum>: Length {
	/// Performs an FFT on the `input` buffer, places the result in the `output` bufer
	/// Uses the `input` buffer as scratch space
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);

    /// Divides the `input` and `output` buffers into self.len() chunks, then runs an FFT
    /// on each chunk. Uses the `input` buffer as scratch space
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);
}

pub use self::mixed_radix::{MixedRadix, MixedRadixDoubleButterfly};
pub use self::raders_algorithm::RadersAlgorithm;
pub use self::radix4::Radix4;
pub use self::good_thomas_algorithm::GoodThomasAlgorithm;
pub use self::dft::DFT;
