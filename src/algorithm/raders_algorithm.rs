use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;
use strength_reduce::StrengthReducedUsize;

use common::FFTnum;

use math_utils;
use ::{Length, IsInverse, Fft};

/// Implementation of Rader's Algorithm
///
/// This algorithm computes a prime-sized FFT in O(nlogn) time. It does this by converting this size n FFT into a
/// size (n - 1) which is guaranteed to be composite.
///
/// The worst case for this algorithm is when (n - 1) is 2 * prime, resulting in a
/// [Cunningham Chain](https://en.wikipedia.org/wiki/Cunningham_chain)
///
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Rader's Algorithm
/// use rustfft::algorithm::RadersAlgorithm;
/// use rustfft::{Fft, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1201];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1201];
///
/// // plan a FFT of size n - 1 = 1200
/// let mut planner = FFTplanner::new(false);
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
    inner_fft: Arc<Fft<T>>,
    inner_fft_data: Box<[Complex<T>]>,

    primitive_root: usize,
    primitive_root_inverse: usize,

    len: StrengthReducedUsize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}

impl<T: FFTnum> RadersAlgorithm<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `len`. `inner_fft.len()` must be `len - 1`
    ///
    /// Note that this constructor is quite expensive to run; This algorithm must run a FFT of size n - 1 within the
    /// constructor. This further underlines the fact that Rader's Algorithm is more expensive to run than other
    /// FFT algorithms
    ///
    /// Note also that if `len` is not prime, this algorithm may silently produce garbage output
    pub fn new(len: usize, inner_fft: Arc<Fft<T>>) -> Self {
        assert_eq!(len - 1, inner_fft.len(), "For raders algorithm, inner_fft.len() must be self.len() - 1. Expected {}, got {}", len - 1, inner_fft.len());

        let inverse = inner_fft.is_inverse();
        let inner_fft_len = len - 1;
        let reduced_len = StrengthReducedUsize::new(len);

        // compute the primitive root and its inverse for this size
        let primitive_root = math_utils::primitive_root(len as u64).unwrap() as usize;
        let primitive_root_inverse = math_utils::multiplicative_inverse(primitive_root as usize, len);

        // precompute the coefficients to use inside the process method
        let unity_scale = T::from_f64(1f64 / inner_fft_len as f64).unwrap();
        let mut inner_fft_input = vec![Complex::zero(); inner_fft_len];
        let mut twiddle_input = 1;
        for input_cell in &mut inner_fft_input {
            let twiddle = T::generate_twiddle_factor(twiddle_input, len, inverse);
            *input_cell = twiddle * unity_scale;

            twiddle_input = (twiddle_input * primitive_root_inverse) % reduced_len;
        }

        let required_inner_scratch = inner_fft.get_inplace_scratch_len();
        let extra_inner_scratch = if required_inner_scratch <= inner_fft_len { 0 } else { required_inner_scratch };

        //precompute a FFT of our reordered twiddle factors
        let mut inner_fft_scratch = vec![Zero::zero(); required_inner_scratch];
        inner_fft.process_inplace_with_scratch(&mut inner_fft_input, &mut inner_fft_scratch);

        Self {
            inner_fft: inner_fft,
            inner_fft_data: inner_fft_input.into_boxed_slice(),

            primitive_root,
            primitive_root_inverse,

            len: reduced_len,
            inplace_scratch_len: inner_fft_len + extra_inner_scratch,
            outofplace_scratch_len: extra_inner_scratch,
            inverse,
        }
    }

    fn perform_fft_out_of_place(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {

        // The first output element is just the sum of all the input elements, and we need to store off the first input value
        let (output_first, output) = output.split_first_mut().unwrap();
        let (input_first, input) = input.split_first_mut().unwrap();
        let first_input_val = *input_first;
        *output_first = *input_first;

        // copy the inout into the output, reordering as we go. also compute a sum of all elements
        let mut input_index = 1;
        for output_element in output.iter_mut() {
            input_index = (input_index * self.primitive_root) % self.len;

            let input_element = input[input_index - 1];
            *output_first = *output_first + input_element;
            *output_element = input_element;
        }

        // perform the first of two inner FFTs
        let inner_scratch = if scratch.len() > 0 { &mut scratch[..] } else { &mut input[..] };
        self.inner_fft.process_inplace_with_scratch(output, inner_scratch);

        // multiply the inner result with our cached setup data
        // also conjugate every entry. this sets us up to do an inverse FFT
        // (because an inverse FFT is equivalent to a normal FFT where you conjugate both the inputs and outputs)
        for ((output_cell, input_cell), &multiple) in output.iter().zip(input.iter_mut()).zip(self.inner_fft_data.iter()) {
            *input_cell = (*output_cell * multiple).conj();
        }

        // execute the second FFT
        let inner_scratch = if scratch.len() > 0 { scratch } else { &mut output[..] };
        self.inner_fft.process_inplace_with_scratch(input, inner_scratch);

        // copy the final values into the output, reordering as we go
        let mut output_index = 1;
        for input_element in input {
            output_index = (output_index * self.primitive_root_inverse) % self.len;
            output[output_index - 1] = input_element.conj() + first_input_val;
        }
    }
    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {

        // The first output element is just the sum of all the input elements, and we need to store off the first input value
        let (buffer_first, buffer) = buffer.split_first_mut().unwrap();
        let buffer_first_val = *buffer_first;

        let (scratch, extra_scratch) = scratch.split_at_mut(self.len() - 1);

        // copy the buffer into the scratch, reordering as we go. also compute a sum of all elements
        let mut input_index = 1;
        for scratch_element in scratch.iter_mut() {
            input_index = (input_index * self.primitive_root) % self.len;

            let buffer_element = buffer[input_index - 1];
            *buffer_first = *buffer_first + buffer_element;
            *scratch_element = buffer_element;
        }

        // perform the first of two inner FFTs
        let inner_scratch = if extra_scratch.len() > 0 { extra_scratch } else { &mut buffer[..] };
        self.inner_fft.process_inplace_with_scratch(scratch, inner_scratch);

        // multiply the inner result with our cached setup data
        // also conjugate every entry. this sets us up to do an inverse FFT
        // (because an inverse FFT is equivalent to a normal FFT where you conjugate both the inputs and outputs)
        for (scratch_cell, &twiddle) in scratch.iter_mut().zip(self.inner_fft_data.iter()) {
            *scratch_cell = (*scratch_cell * twiddle).conj();
        }

        // execute the second FFT
        self.inner_fft.process_inplace_with_scratch(scratch, inner_scratch);

        // copy the final values into the output, reordering as we go
        let mut output_index = 1;
        for scratch_element in scratch {
            output_index = (output_index * self.primitive_root_inverse) % self.len;
            buffer[output_index - 1] = scratch_element.conj() + buffer_first_val;
        }
    }
}
boilerplate_fft!(RadersAlgorithm, 
    |this: &RadersAlgorithm<_>| this.len.get(),
    |this: &RadersAlgorithm<_>| this.inplace_scratch_len,
    |this: &RadersAlgorithm<_>| this.outofplace_scratch_len
);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
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
        let inner_fft = Arc::new(DFT::new(len - 1, inverse));
        let fft = RadersAlgorithm::new(len, inner_fft);

        check_fft_algorithm(&fft, len, inverse);
    }
}
