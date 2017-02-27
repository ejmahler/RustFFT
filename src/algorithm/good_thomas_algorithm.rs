
use num::{Complex, Zero};
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

    scratch: Vec<Complex<T>>,
}

impl<T: FFTnum> GoodThomasAlgorithm<T> {
    #[allow(dead_code)]
    pub fn new(n1: usize,
               n1_fft: Box<FFTAlgorithm<T>>,
               n2: usize,
               n2_fft: Box<FFTAlgorithm<T>>)
               -> Self {

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

            scratch: vec![Zero::zero(); n1 * n2],
        }
    }

    fn copy_from_input(&mut self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        for (output_element, input_index) in output.iter_mut().zip(self.input_map.iter()) {
            *output_element = unsafe { *input.get_unchecked(*input_index) };
        }
    }

    fn copy_transposed_scratch_to_output(&self, output: &mut [Complex<T>]) {
        for (scratch_element, output_index) in self.scratch.iter().zip(self.output_map.iter()) {
            unsafe { *output.get_unchecked_mut(*output_index) = *scratch_element };
        }
    }

    /// Runs the FFT on the input `signal` array, placing the output in the 'spectrum' array
    fn perform_fft(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        // copy the input into the spectrum
        self.copy_from_input(signal, spectrum);

        // run 'height' FFTs of size 'width' from the spectrum into scratch
        for (input, output) in spectrum.chunks(self.width)
            .zip(self.scratch.chunks_mut(self.width)) {
            self.width_size_fft.process(input, output);
        }

        // transpose the scratch back into the spectrum to prepare for the next round of FFT
        array_utils::transpose(self.width, self.height, self.scratch.as_slice(), spectrum);

        // run 'width' FFTs of size 'height' from the spectrum back into scratch
        for (input, output) in spectrum.chunks(self.height)
            .zip(self.scratch.chunks_mut(self.height)) {
            self.height_size_fft.process(input, output);
        }

        // we're done, copy to the output
        self.copy_transposed_scratch_to_output(spectrum);
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for GoodThomasAlgorithm<T> {
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        self.perform_fft(signal, spectrum);
    }
    fn process_multi(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        for (input, output) in signal.chunks(self.scratch.len()).zip(spectrum.chunks_mut(self.scratch.len())) {
            self.perform_fft(input, output);
        }
    }
}
