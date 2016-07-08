use num::Complex;

pub trait FFTAlgorithm<T> {
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]);
}
