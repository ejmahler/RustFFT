use std::rc::Rc;

use num::{Complex, Zero, FromPrimitive};
use common::{FFTnum, verify_length, verify_length_divisible};

use math_utils;
use twiddles;
use super::FFTAlgorithm;

pub struct RadersAlgorithm<T> {
    inner_fft: Rc<FFTAlgorithm<T>>,
    inner_fft_data: Box<[Complex<T>]>,

    input_map: Box<[usize]>,
    output_map: Box<[usize]>,
}

impl<T: FFTnum> RadersAlgorithm<T> {
    pub fn new(len: usize, inner_fft: Rc<FFTAlgorithm<T>>, inverse: bool) -> Self {
        assert_eq!(len - 1, inner_fft.len(), "For raders algorithm, inner_fft.len() must be self.len() - 1. Expected {}, got {}", len - 1, inner_fft.len());

        let inner_fft_len = len - 1;

        // compute the primitive root and its inverse for this size
        let primitive_root = math_utils::primitive_root(len as u64).unwrap();
        let root_inverse = math_utils::multiplicative_inverse(primitive_root, len as u64);

        // precompute the coefficients to use inside the process method
        let unity_scale: T = FromPrimitive::from_f64(1f64 / inner_fft_len as f64).unwrap();
        let mut inner_fft_input: Vec<Complex<T>> = (0..inner_fft_len)
            .map(|i| math_utils::modular_exponent(root_inverse, i as u64, len as u64) as usize)
            .map(|i| twiddles::single_twiddle(i, len, inverse))
            .map(|c| c * unity_scale)
            .collect();

        //precompute a FFT of our reordered twiddle factors
        let mut inner_fft_output = vec![Zero::zero(); inner_fft_len];
        inner_fft.process(&mut inner_fft_input, &mut inner_fft_output);

        //precompute the indexes we'll used to reorder the input and output
        let input_map: Vec<usize> = (0..len-1).map(|i| math_utils::modular_exponent(primitive_root, (i + 1) as u64, len as u64) as usize - 1).collect();
        let output_map: Vec<usize> = (0..len-1).map(|i| math_utils::modular_exponent(root_inverse, (i + 1) as u64, len as u64) as usize - 1).collect();

        RadersAlgorithm {
            inner_fft: inner_fft,
            inner_fft_data: inner_fft_output.into_boxed_slice(),

            input_map: input_map.into_boxed_slice(),
            output_map: output_map.into_boxed_slice(),
        }
    }

    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {

        //the input and output buffers are one element too large, split off the first and do the processing for the first now
        let (first_input, input) = input.split_first_mut().unwrap();
        let first_input_val = *first_input;

        let (first_output, output) = output.split_first_mut().unwrap();
        

        //the first element of the output will be the sum of all the inputs
        let input_sum: Complex<T> = input.iter().fold(Zero::zero(), |acc, &e| acc + e);
        *first_output = first_input_val + input_sum;


        // prepare the inner FFT by reordering the input buffer into the output buffer
        // we could compute the indexes here on the fly, but benchmarking shows it's faster to precompute and store them
        for (&input_index, output_element) in self.input_map.iter().zip(output.iter_mut()) {
            *output_element = input[input_index];
        }

        // perform the first of two inner FFTs
        self.inner_fft.process(output, input);

        // multiply the inner result with our cached setup data
        // also conjugate every entry. this sets us up to do an inverse FFT
        // (because an inverse FFT is equivalent to a normal FFT where you conjugate both the inputs and outputs)
        for ((&input_cell, output_cell), &multiple) in input.iter().zip(output.iter_mut()).zip(self.inner_fft_data.iter()) {
            *output_cell = (input_cell * multiple).conj();
        }

        // execute the second FFT
        self.inner_fft.process(output, input);

        // to finalize the inverse, compute the conjugate of every element
        for element in input.iter_mut() {
            *element = element.conj() + first_input_val;
        }

        // copy the input buffer to the output buffer, reordering the elements as we go
        // we could compute the indexes here on the fly, but benchmarking shows it's faster to precompute and store them
        for (&output_index, input_element) in self.output_map.iter().zip(input.iter()) {
            output[output_index] = *input_element;
        }
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for RadersAlgorithm<T> {
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
        self.inner_fft_data.len() + 1
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::rc::Rc;
    use num::Zero;
    use test_utils::{random_signal, compare_vectors};
    use algorithm::DFT;

    #[test]
    fn test_raders() {
        for &len in &[3,5,7,11,13] {
            let dft = DFT::new(len, false);
            let inner_fft = Rc::new(DFT::new(len - 1, false));

            let raders_fft = RadersAlgorithm::new(len, inner_fft, false);

            let mut dft_input = random_signal(len);
            let mut raders_input = dft_input.clone();

            let mut expected = vec![Zero::zero(); len];
            dft.process(&mut dft_input, &mut expected);

            let mut actual = vec![Zero::zero(); len];
            raders_fft.process(&mut raders_input, &mut actual);

            println!("Expected: {:?}", expected);
            println!("Actual:   {:?}", actual);

            assert!(compare_vectors(&actual, &expected), "len = {}", len);
        }
    }
}
