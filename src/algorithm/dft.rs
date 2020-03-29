use num_complex::Complex;
use num_traits::Zero;

use common::FFTnum;

use ::{Length, IsInverse, FFT, FftInline};
use twiddles;

/// Naive O(n^2 ) Discrete Fourier Transform implementation
///
/// This implementation is primarily used to test other FFT algorithms. In a few rare cases, such as small
/// [Cunningham Chain](https://en.wikipedia.org/wiki/Cunningham_chain) primes, this can be faster than the O(nlogn)
/// algorithms
///
/// ~~~
/// // Computes a naive DFT of size 1234
/// use rustfft::algorithm::DFT;
/// use rustfft::FFT;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1234];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1234];
///
/// let dft = DFT::new(1234, false);
/// dft.process(&mut input, &mut output);
/// ~~~
pub struct DFT<T> {
    twiddles: Vec<Complex<T>>,
    inverse: bool,
}

impl<T: FFTnum> DFT<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute DFT
    pub fn new(len: usize, inverse: bool) -> Self {
        DFT {
            twiddles: twiddles::generate_twiddle_factors(len, inverse),
            inverse: inverse
        }
    }

    fn perform_fft_out_of_place(&self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
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
boilerplate_fft_oop!(DFT, |this: &DFT<_>| this.twiddles.len());

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::f32;
    use test_utils::{random_signal, compare_vectors};
    use num_complex::Complex;
    use num_traits::Zero;

    fn dft(signal: &[Complex<f32>], spectrum: &mut [Complex<f32>]) {
        for (k, spec_bin) in spectrum.iter_mut().enumerate() {
            let mut sum = Zero::zero();
            for (i, &x) in signal.iter().enumerate() {
                let angle = -1f32 * (i * k) as f32 * 2f32 * f32::consts::PI / signal.len() as f32;
                let twiddle = Complex::from_polar(&1f32, &angle);

                sum = sum + twiddle * x;
            }
            *spec_bin = sum;
        }
    }

    #[test]
    fn test_matches_dft() {
        let n = 4;

        for len in 1..20 {
            let dft_instance = DFT::new(len, false);
            assert_eq!(dft_instance.len(), len, "DFT instance reported incorrect length");

            let input = random_signal(len * n);
            let mut expected_input = input.clone();
            let mut actual_input = input.clone();
            let mut multi_input = input.clone();
            let mut inline_buffer = input.clone();
            let mut inline_multi_buffer = input.clone();

            let mut expected_output = vec![Zero::zero(); len * n];
            let mut actual_output = expected_output.clone();
            let mut multi_output = expected_output.clone();
            let mut inline_scratch = vec![Zero::zero(); dft_instance.get_required_scratch_len()];
            let mut inline_multi_scratch = vec![Zero::zero(); dft_instance.get_required_scratch_len()];

            // perform the test
            dft_instance.process_multi(&mut multi_input, &mut multi_output);

            for (input_chunk, output_chunk) in actual_input.chunks_mut(len).zip(actual_output.chunks_mut(len)) {
                dft_instance.process(input_chunk, output_chunk);
            }

            for chunk in inline_buffer.chunks_mut(len) {
                dft_instance.process_inline(chunk, &mut inline_scratch);
            }
            dft_instance.process_inline_multi(&mut inline_multi_buffer, &mut inline_multi_scratch);

            for (input_chunk, output_chunk) in expected_input.chunks_mut(len).zip(expected_output.chunks_mut(len)) {
                dft(input_chunk, output_chunk);
            }

            assert!(compare_vectors(&expected_output, &actual_output), "process() failed, length = {}", len);
            assert!(compare_vectors(&expected_output, &multi_output), "process_multi() failed, length = {}", len);
            assert!(compare_vectors(&expected_output, &inline_buffer), "process_inline() failed, length = {}", len);
            assert!(compare_vectors(&expected_output, &inline_multi_buffer), "process_inline_multi() failed, length = {}", len);
            
            // one more thing: make sure that the DFT algorithm even works with dirty scratch space
            for item in inline_scratch.iter_mut() {
                *item = Complex::new(100.0,100.0);
            }
            let mut inline_buffer = input.clone();
            for chunk in inline_buffer.chunks_mut(len) {
                dft_instance.process_inline(chunk, &mut inline_scratch);
            }
            assert!(compare_vectors(&expected_output, &inline_buffer), "process_inline() failed the 'dirty scratch' test for len = {}", len);

            for item in inline_multi_scratch.iter_mut() {
                *item = Complex::new(100.0,100.0);
            }
            let mut inline_multi_buffer = input.clone();
            dft_instance.process_inline_multi(&mut inline_multi_buffer, &mut inline_multi_scratch);
            assert!(compare_vectors(&expected_output, &inline_multi_buffer), "process_inline_multi() failed the 'dirty scratch' test for len = {}", len);
        }

        //verify that it doesn't crash if we have a length of 0
        let zero_dft = DFT::new(0, false);
        let mut zero_input: Vec<Complex<f32>> = Vec::new();
        let mut zero_output: Vec<Complex<f32>> = Vec::new();
        let mut zero_scratch: Vec<Complex<f32>> = Vec::new();

        zero_dft.process(&mut zero_input, &mut zero_output);
        zero_dft.process_inline(&mut zero_input, &mut zero_scratch);
    }

    /// Returns true if our `dft` function calculates the given output from the
    /// given input, and if rustfft's DFT struct does the same
    fn test_dft_correct(input: &[Complex<f32>], output: &[Complex<f32>]) {
        assert_eq!(input.len(), output.len());
        let len = input.len();

        let mut reference_output = vec![Zero::zero(); len];
        dft(&input, &mut reference_output);
        assert!(compare_vectors(output, &reference_output), "Reference implementation failed for len={}", len);

        let dft_instance = DFT::new(len, false);
        let mut actual_input = input.to_vec();
        let mut actual_output = vec![Zero::zero(); len];
        let mut inline_buffer = input.to_vec();
        let mut inline_scratch = vec![Zero::zero(); dft_instance.get_required_scratch_len()];

        dft_instance.process(&mut actual_input, &mut actual_output);
        dft_instance.process_inline(&mut inline_buffer, &mut inline_scratch);
        assert!(compare_vectors(output, &actual_output), "process() failed for len = {}", len);
        assert!(compare_vectors(output, &inline_buffer), "process_inline() failed for len = {}", len);

        // one more thing: make sure that the DFT algorithm even works with dirty scratch space
        for item in inline_scratch.iter_mut() {
            *item = Complex::new(100.0,100.0);
        }
        let mut inline_buffer = input.to_vec();
        dft_instance.process_inline(&mut inline_buffer, &mut inline_scratch);
        assert!(compare_vectors(output, &inline_buffer), "process_inline() failed the 'dirty scratch' test for len = {}", len);
    }

    #[test]
    fn test_dft_known_len_2() {
        let signal = [Complex{re: 1f32, im: 0f32},
                      Complex{re:-1f32, im: 0f32}];
        let spectrum = [Complex{re: 0f32, im: 0f32},
                        Complex{re: 2f32, im: 0f32}];
        test_dft_correct(&signal[..], &spectrum[..]);
    }

    #[test]
    fn test_dft_known_len_3() {
        let signal = [Complex{re: 1f32, im: 1f32},
                      Complex{re: 2f32, im:-3f32},
                          Complex{re:-1f32, im: 4f32}];
        let spectrum = [Complex{re: 2f32, im: 2f32},
                        Complex{re: -5.562177f32, im: -2.098076f32},
                        Complex{re: 6.562178f32, im: 3.09807f32}];
        test_dft_correct(&signal[..], &spectrum[..]);
    }

    #[test]
    fn test_dft_known_len_4() {
        let signal = [Complex{re: 0f32, im: 1f32},
                      Complex{re: 2.5f32, im:-3f32},
                      Complex{re:-1f32, im: -1f32},
                      Complex{re: 4f32, im: 0f32}];
        let spectrum = [Complex{re: 5.5f32, im: -3f32},
                        Complex{re: -2f32, im: 3.5f32},
                        Complex{re: -7.5f32, im: 3f32},
                        Complex{re: 4f32, im: 0.5f32}];
        test_dft_correct(&signal[..], &spectrum[..]);
    }

    #[test]
    fn test_dft_known_len_6() {
        let signal = [Complex{re: 1f32, im: 1f32},
                      Complex{re: 2f32, im: 2f32},
                      Complex{re: 3f32, im: 3f32},
                      Complex{re: 4f32, im: 4f32},
                      Complex{re: 5f32, im: 5f32},
                      Complex{re: 6f32, im: 6f32}];
        let spectrum = [Complex{re: 21f32, im: 21f32},
                        Complex{re: -8.16f32, im: 2.16f32},
                        Complex{re: -4.76f32, im: -1.24f32},
                        Complex{re: -3f32, im: -3f32},
                        Complex{re: -1.24f32, im: -4.76f32},
                        Complex{re: 2.16f32, im: -8.16f32}];
        test_dft_correct(&signal[..], &spectrum[..]);
    }
}