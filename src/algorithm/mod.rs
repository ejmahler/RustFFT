use num::Complex;
use common::FFTnum;

mod good_thomas_algorithm;
mod mixed_radix;
mod raders_algorithm;
mod radix4;
mod dft;
pub mod butterflies;


pub trait FFTAlgorithm<T: FFTnum> {
	/// Performs an FFT on the `input` buffer, places the result in the `output` bufer
	/// Uses the `input` buffer as scratch space
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);

    /// Divides the `input` and `output` buffers into self.len() chunks, then runs an FFT
    /// on each chunk. Uses the `input` buffer as scratch space
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);

    // The FFT size that this algorithm can process
    fn len(&self) -> usize;
}


pub trait FFTButterfly<T: FFTnum> {
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]);
}
pub enum ButterflyEnum<T> {
	Butterfly2(butterflies::Butterfly2),
	Butterfly3(butterflies::Butterfly3<T>),
	Butterfly4(butterflies::Butterfly4),
	Butterfly5(butterflies::Butterfly5<T>),
	Butterfly6(butterflies::Butterfly6<T>),
	Butterfly7(butterflies::Butterfly7<T>),
}

impl<T: FFTnum> ButterflyEnum<T> {
	#[inline(always)]
	pub unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
		use self::ButterflyEnum::*;
		match *self {
			Butterfly2(ref fft) => fft.process_multi_inplace(buffer),
			Butterfly3(ref fft) => fft.process_multi_inplace(buffer),
			Butterfly4(ref fft) => fft.process_multi_inplace(buffer),
			Butterfly5(ref fft) => fft.process_multi_inplace(buffer),
			Butterfly6(ref fft) => fft.process_multi_inplace(buffer),
			Butterfly7(ref fft) => fft.process_multi_inplace(buffer),
		}
	}

	#[inline(always)]
	pub fn len(&self) -> usize {
		use self::ButterflyEnum::*;
		match *self {
			Butterfly2(_) => 2,
			Butterfly3(_) => 3,
			Butterfly4(_) => 4,
			Butterfly5(_) => 5,
			Butterfly6(_) => 6,
			Butterfly7(_) => 7,
		}
	}
}


pub struct NoopAlgorithm {
	pub len: usize
}
impl<T: FFTnum> FFTAlgorithm<T> for NoopAlgorithm {
    fn process(&self, _: &mut [Complex<T>], _: &mut [Complex<T>]) { }
    fn process_multi(&self, _: &mut [Complex<T>], _: &mut [Complex<T>]) { }
    fn len(&self) -> usize {
        self.len
    }
}

pub use self::mixed_radix::{MixedRadix, MixedRadixDoubleButterfly};
pub use self::raders_algorithm::RadersAlgorithm;
pub use self::radix4::Radix4;
pub use self::good_thomas_algorithm::GoodThomasAlgorithm;
pub use self::dft::DFTAlgorithm;
