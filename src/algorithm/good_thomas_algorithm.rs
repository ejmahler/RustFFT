use std::rc::Rc;

use num::Complex;
use common::{FFTnum, verify_length, verify_length_divisible};

use math_utils;
use array_utils;

use algorithm::FFTAlgorithm;

pub struct GoodThomasAlgorithm<T> {
    width: usize,
    // width_inverse: usize,
    width_size_fft: Rc<FFTAlgorithm<T>>,

    height: usize,
    // height_inverse: usize,
    height_size_fft: Rc<FFTAlgorithm<T>>,

    input_map: Box<[usize]>,
    output_map: Box<[usize]>,
}

impl<T: FFTnum> GoodThomasAlgorithm<T> {
    #[allow(dead_code)]
    pub fn new(n1_fft: Rc<FFTAlgorithm<T>>, n2_fft: Rc<FFTAlgorithm<T>>) -> Self {

        let n1 = n1_fft.len();
        let n2 = n2_fft.len();

        // compute the nultiplicative inverse of n1 mod n2 and vice versa
        let (gcd, mut n1_inverse, mut n2_inverse) =
            math_utils::extended_euclidean_algorithm(n1 as i64, n2 as i64);
        assert!(gcd == 1,
                "Invalid input n1 and n2 to Good-Thomas Algorithm: ({},{}): Inputs must be \
                 coprime",
                n1,
                n2);

        // n1_inverse or n2_inverse might be negative, make it positive
        if n1_inverse < 0 {
            n1_inverse += n2 as i64;
        }
        if n2_inverse < 0 {
            n2_inverse += n1 as i64;
        }

        // NOTE: we are precomputing the input and output reordering indexes
        // benchmarking shows that it's 20-30% faster
        // If we wanted to optimize for memory use or setup time instead of multiple-FFT speed,
        // these can be computed at runtime
        let input_map: Vec<usize> = 
            (0..n1 * n2)
                .map(|i| (i % n1, i / n1))
                .map(|(x, y)| (x * n2 + y * n1) % (n1 * n2))
                .collect();
        let output_map: Vec<usize> = 
            (0..n1 * n2)
                .map(|i| (i % n2, i / n2))
                .map(|(y, x)| {
                    (x * n2 * n2_inverse as usize + y * n1 * n1_inverse as usize) % (n1 * n2)
                })
                .collect();

        GoodThomasAlgorithm {
            width: n1,
            width_size_fft: n1_fft,

            height: n2,
            height_size_fft: n2_fft,
            
            input_map: input_map.into_boxed_slice(),
            output_map: output_map.into_boxed_slice(),
        }
    }

    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {

        // copy the input using our reordering mapping
        for (output_element, &input_index) in output.iter_mut().zip(self.input_map.iter()) {
            *output_element = input[input_index];
        }

        // run FFTs of size `width`
        self.width_size_fft.process_multi(output, input);

        // transpose
        array_utils::transpose(self.width, self.height, input, output);

        // run FFTs of size 'height'
        self.height_size_fft.process_multi(output, input);

        // copy to the output, using our output redordeing mapping
        for (input_element, &output_index) in input.iter().zip(self.output_map.iter()) {
            output[output_index] = *input_element;
        }
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for GoodThomasAlgorithm<T> {
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
        self.input_map.len()
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::rc::Rc;
    use test_utils::{random_signal, compare_vectors};
    use algorithm::DFT;
    use num::Zero;

    #[test]
    fn test_good_thomas() {

        //gcd(n, n+1) is guaranteed to be 1, so we can generate some test sizes by just passing in n, n + 1
        for width in 2..20 {
            test_good_thomas_with_lengths(width, width - 1);
            test_good_thomas_with_lengths(width, width + 1);
        }

        //verify that it works correctly when width and/or height are 1
        test_good_thomas_with_lengths(1, 10);
        test_good_thomas_with_lengths(10, 1);
        test_good_thomas_with_lengths(1, 1);
    }

    fn test_good_thomas_with_lengths(width: usize, height: usize) {
        let n = 4;
        let len = width * height;

        // set up algorithms
        let dft = DFT::new(len, false);

        let width_fft = Rc::new(DFT::new(width, false)) as Rc<FFTAlgorithm<f32>>;
        let height_fft = Rc::new(DFT::new(height, false)) as Rc<FFTAlgorithm<f32>>;

        let good_thomas_fft = GoodThomasAlgorithm::new(width_fft, height_fft);

        assert_eq!(good_thomas_fft.len(), len, "Good thomas algorithm reported incorrect length");

        // set up buffers
        let mut expected_input = random_signal(len * n);
        let mut actual_input = expected_input.clone();
        let mut multi_input = expected_input.clone();

        let mut expected_output = vec![Zero::zero(); len * n];
        let mut actual_output = expected_output.clone();
        let mut multi_output = expected_output.clone();

        // perform the test
        dft.process_multi(&mut expected_input, &mut expected_output);
        good_thomas_fft.process_multi(&mut multi_input, &mut multi_output);

        for (input_chunk, output_chunk) in actual_input.chunks_mut(len).zip(actual_output.chunks_mut(len)) {
            good_thomas_fft.process(input_chunk, output_chunk);
        }

        assert!(compare_vectors(&expected_output, &actual_output), "process() failed, width = {}, height = {}", width, height);
        assert!(compare_vectors(&expected_output, &multi_output), "process_multi() failed, width = {}, height = {}", width, height);
    }
}
