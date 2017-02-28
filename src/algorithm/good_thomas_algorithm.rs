
use num::Complex;
use common::FFTnum;

use math_utils;
use array_utils;

use algorithm::FFTAlgorithm;

pub struct GoodThomasAlgorithm<T> {
    width: usize,
    // width_inverse: usize,
    width_size_fft: Box<FFTAlgorithm<T>>,

    height: usize,
    // height_inverse: usize,
    height_size_fft: Box<FFTAlgorithm<T>>,

    input_map: Vec<usize>,
    output_map: Vec<usize>,
}

impl<T: FFTnum> GoodThomasAlgorithm<T> {
    #[allow(dead_code)]
    pub fn new(n1_fft: Box<FFTAlgorithm<T>>, n2_fft: Box<FFTAlgorithm<T>>) -> Self {

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

        GoodThomasAlgorithm {
            width: n1,
            // width_inverse: n1_inverse as usize,
            width_size_fft: n1_fft,

            height: n2,
            // height_inverse: n2_inverse as usize,
            height_size_fft: n2_fft,

            // NOTE: we are precomputing the input and output mappings because
            // benchmarking shows that it's 20-30% faster
            // If we wanted to optimize for memory use or setup time instead of multiple-FFT speed,
            // these can be computed at runtime
            input_map: (0..n1 * n2)
                .map(|i| (i % n1, i / n1))
                .map(|(x, y)| (x * n2 + y * n1) % (n1 * n2))
                .collect(),
            output_map: (0..n1 * n2)
                .map(|i| (i % n2, i / n2))
                .map(|(y, x)| {
                    (x * n2 * n2_inverse as usize + y * n1 * n1_inverse as usize) % (n1 * n2)
                })
                .collect(),
        }
    }

    fn copy_from_input(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        for (output_element, input_index) in output.iter_mut().zip(self.input_map.iter()) {
            *output_element = unsafe { *input.get_unchecked(*input_index) };
        }
    }

    fn copy_to_output(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        for (input_element, output_index) in input.iter().zip(self.output_map.iter()) {
            unsafe { *output.get_unchecked_mut(*output_index) = *input_element };
        }
    }


    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {

        // copy the input using our reordering algorithm
        self.copy_from_input(input, output);

        // run FFTs of size `width`
        self.width_size_fft.process_multi(output, input);

        // transpose
        array_utils::transpose(self.width, self.height, input, output);

        // run 'width' FFTs of size 'height' from the spectrum back into scratch
        self.width_size_fft.process_multi(output, input);

        // we're done, copy to the output
        self.copy_to_output(input, output);
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for GoodThomasAlgorithm<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        self.perform_fft(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            self.perform_fft(in_chunk, out_chunk);
        }
    }
    fn len(&self) -> usize {
        self.input_map.len()
    }
}
