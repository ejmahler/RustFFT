use std::rc::Rc;

use num_complex::Complex;
use num_traits::{FromPrimitive, Zero};

use common::{FFTnum, verify_length, verify_length_divisible};

use math_utils;
use twiddles;
use ::{Length, IsInverse, FFT};

/// Implementation of Rader's Algorithm
///
/// This algorithm computes a prime-sized FFT in O(nlogn) time. It does this by converting this size n FFT into a
/// size (n - 1) FFT, which is guaranteed to be composite.
///
/// The worst case for this algorithm is when (n - 1) is 2 * prime, resulting in a
/// [Cunningham Chain](https://en.wikipedia.org/wiki/Cunningham_chain)
///
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Rader's Algorithm
/// use rustfft::algorithm::RadersAlgorithm;
/// use rustfft::{FFT, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1201];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1201];
///
/// // plan a FFT of size n - 1 = 1200
/// let mut planner = rustfft::FFTplanner::new(false);
/// let inner_fft = planner.plan_fft(1200);
///
/// let fft = RadersAlgorithm::new(1201, inner_fft);
/// fft.process(&mut input, &mut output);
/// ~~~
///
/// Rader's Algorithm is relatively expensive compared to other FFT algorithms. Benchmarking shows that it is up to
/// an order of magnitude slower than similar composite sizes. In the example size above of 1201, benchmarking shows
/// that it takes 2.5x more time to compute than a FFT of size 1200.

pub struct RadersAlgorithm<T> {
    inner_fft: Rc<FFT<T>>,
    inner_fft_data: Box<[Complex<T>]>,

    input_map: Box<[usize]>,
    output_map: Box<[usize]>,
}

impl<T: FFTnum> RadersAlgorithm<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `len`. `inner_fft.len()` must be `len - 1`
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must run a FFT of size n - 1 within the
    /// constructor. This further underlines the fact that Rader's Algorithm is more expensive to run than other
    /// FFT algorithms
    ///
    /// Note also that if `len` is not prime, this algorithm may silently produce garbage output
    pub fn new(len: usize, inner_fft: Rc<FFT<T>>) -> Self {
        assert_eq!(len - 1, inner_fft.len(), "For raders algorithm, inner_fft.len() must be self.len() - 1. Expected {}, got {}", len - 1, inner_fft.len());

        let inner_fft_len = len - 1;

        // compute the primitive root and its inverse for this size
        let primitive_root = math_utils::primitive_root(len as u64).unwrap();
        let root_inverse = math_utils::multiplicative_inverse(primitive_root, len as u64);

        // precompute the coefficients to use inside the process method
        let unity_scale: T = FromPrimitive::from_f64(1f64 / inner_fft_len as f64).unwrap();
        let mut inner_fft_input: Vec<Complex<T>> = (0..inner_fft_len)
            .map(|i| math_utils::modular_exponent(root_inverse, i as u64, len as u64) as usize)
            .map(|i| twiddles::single_twiddle(i, len, inner_fft.is_inverse()))
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

impl<T: FFTnum> FFT<T> for RadersAlgorithm<T> {
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
}
impl<T> Length for RadersAlgorithm<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.inner_fft_data.len() + 1
    }
}
impl<T> IsInverse for RadersAlgorithm<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inner_fft.is_inverse()
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::rc::Rc;
    use test_utils::check_fft_algorithm;
    use algorithm::DFT;

    #[test]
    fn test_raders() {
        for &len in &[3,5,7,11,13] {
            test_raders_with_length(len, false);
            test_raders_with_length(len, true);
        }
    }

    fn test_raders_with_length(len: usize, inverse: bool) {
        let inner_fft = Rc::new(DFT::new(len - 1, inverse));
        let fft = RadersAlgorithm::new(len, inner_fft);

        check_fft_algorithm(&fft, len, inverse);
    }
}
