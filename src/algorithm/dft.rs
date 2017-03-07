use num::{Complex, Zero};
use common::{FFTnum, verify_length, verify_length_divisible};

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

    #[inline(always)]
    fn perform_fft(&self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        for k in 0..spectrum.len() {
            let output_cell = spectrum.get_mut(k).unwrap();

            *output_cell = Zero::zero();
            let mut twiddle_index = 0;

            for input_cell in signal {
                let twiddle = self.twiddles[twiddle_index];
                *output_cell = *output_cell + twiddle * input_cell;

                twiddle_index += k;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for DFTAlgorithm<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

        self.perform_fft(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

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