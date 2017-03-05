use std::rc::Rc;

use num::{Complex, Zero, FromPrimitive};
use common::FFTnum;

use math_utils;
use twiddles;
use super::FFTAlgorithm;

pub struct RadersAlgorithm<T> {
    len: usize,

    primitive_root: u64,
    root_inverse: u64,

    inner_fft_data: Vec<Complex<T>>,
    inner_fft: Rc<FFTAlgorithm<T>>,
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

        Self {
            len: len,
            primitive_root: primitive_root,
            root_inverse: root_inverse,

            inner_fft_data: inner_fft_output,
            inner_fft: inner_fft,
        }
    }

    unsafe fn copy_from_input(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        // it's not just a straight copy from the input to the scratch, we have
        // to compute the input index based on the scratch index and primitive root
        for (index, output_element) in output.iter_mut().enumerate() {
            let input_index = math_utils::modular_exponent(self.primitive_root, index as u64 + 1, self.len as u64) as usize - 1;
            
            *output_element = *input.get_unchecked(input_index);
        }
    }

    unsafe fn copy_to_output(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        // copy the data back into the input vector, but again it's not just a straight copy
        for (index, &input_element) in input.iter().enumerate() {
            let output_index = math_utils::modular_exponent(
                self.root_inverse,
                index as u64 + 1,
                self.len as u64) as usize - 1;

            *output.get_unchecked_mut(output_index) = input_element;
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



        // redorder the input as we copy it to the output buffer
        unsafe { self.copy_from_input(input, output) };

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

        unsafe { self.copy_to_output(input, output) };
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for RadersAlgorithm<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        self.perform_fft(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            self.perform_fft(in_chunk, out_chunk);
        }
    }
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::rc::Rc;
    use num::Zero;
    use test_utils::{random_signal, compare_vectors};
    use dft;
    use algorithm::{butterflies, DFTAlgorithm};

    #[test]
    fn test_raders() {
        for &len in &[3,5,7,11,13] {
            let dft = DFTAlgorithm::new(len, false);
            let inner_fft = Rc::new(DFTAlgorithm::new(len - 1, false));

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
