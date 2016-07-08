use num::Complex;

mod mixed_radix;
mod raders_algorithm;
mod radix4;

pub trait FFTAlgorithm<T> {
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]);
}

pub use self::mixed_radix::CooleyTukey;
pub use self::raders_algorithm::RadersAlgorithm;
pub use self::radix4::Radix4;
