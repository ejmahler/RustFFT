use num::{Complex, Zero};
use common::FFTnum;

use algorithm::FFTAlgorithm;
use twiddles;

pub struct DFTAlgorithm<T> {
    twiddles: Vec<Complex<T>>,
}

impl<T: FFTnum> DFTAlgorithm<T> {
    pub fn new(len: usize, inverse: bool) -> Self {
        Self {
            twiddles: twiddles::generate_twiddle_factors(len, inverse),
        }
    }

    fn perform_fft(&self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        for (k, output_cell) in spectrum.iter_mut().enumerate() {
            let mut sum = Zero::zero();
            for (i, &input_cell) in signal.iter().enumerate() {
                let twiddle = unsafe { *self.twiddles.get_unchecked((k * i) % self.twiddles.len()) };
                sum = sum + twiddle * input_cell;
            }
            *output_cell = sum;
        }
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for DFTAlgorithm<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        self.perform_fft(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            self.perform_fft(in_chunk, out_chunk);
        }
    }
    fn len(&self) -> usize {
        self.twiddles.len()
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::{random_signal, compare_vectors};
    use dft;

    #[test]
    fn test_matches_dft() {
        for len in 1..50 {
            let mut input = random_signal(len);
            let mut expected = input.clone();
            dft(&input, &mut expected);

            let mut actual = input.clone();
            let wrapper = DFTAlgorithm::new(len, false);
            wrapper.process(&mut input, &mut actual);

            assert!(compare_vectors(&expected, &actual), "length = {}", len);
        }
    }
}