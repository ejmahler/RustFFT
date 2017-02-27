use num::{Complex, Signed, FromPrimitive};

mod good_thomas_algorithm;
mod mixed_radix_terminal;
mod mixed_radix_single;
mod raders_algorithm;
mod radix4;
mod dft;
pub mod butterflies;


pub trait FFTAlgorithm<T: Signed + FromPrimitive + Copy> {
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]);
}
pub trait FFTButterfly<T: Signed + FromPrimitive + Copy> {
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]);
}

pub struct NoopAlgorithm;
impl<T> FFTAlgorithm<T> for NoopAlgorithm
    where T: Signed + FromPrimitive + Copy
{
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        spectrum.copy_from_slice(signal);
    }
}

pub use self::mixed_radix_terminal::MixedRadixTerminal;
pub use self::mixed_radix_single::MixedRadixSingle;
pub use self::raders_algorithm::RadersAlgorithm;
pub use self::radix4::Radix4;
pub use self::good_thomas_algorithm::GoodThomasAlgorithm;
pub use self::dft::DFTAlgorithm;
