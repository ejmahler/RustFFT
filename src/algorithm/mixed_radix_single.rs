use num::{Complex, FromPrimitive, Signed, Zero};

use algorithm::FFTAlgorithm;
use array_utils;
use twiddles;

pub struct MixedRadixSingle<T> {
    width: usize,
    width_size_fft: Box<FFTAlgorithm<T>>,

    height: usize,
    height_size_fft: Box<FFTAlgorithm<T>>,

    twiddles: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T> MixedRadixSingle<T>
    where T: Signed + FromPrimitive + Copy
{
    pub fn new(width: usize,
               width_fft: Box<FFTAlgorithm<T>>,
               height: usize,
               height_fft: Box<FFTAlgorithm<T>>,
               inverse: bool) -> Self {

        let len = width * height;

        MixedRadixSingle {
            width: width,
            width_size_fft: width_fft,

            height: height,
            height_size_fft: height_fft,

            twiddles: twiddles::generate_twiddle_factors(len, inverse),
            scratch: vec![Zero::zero(); len],
        }
    }
}

impl<T> FFTAlgorithm<T> for MixedRadixSingle<T>
    where T: Signed + FromPrimitive + Copy
{
    /// Runs the FFT on the input `signal` array, placing the output in the 'spectrum' array
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        // SIX STEP FFT:

        // STEP 1: transpose
        array_utils::transpose(self.width, self.height, signal, spectrum);

        // STEP 2: perform FFTs of size 'height'
        for (row_index, (input, output)) in spectrum.chunks(self.height).zip(self.scratch.chunks_mut(self.height)).enumerate()
        {
            self.height_size_fft.process(input, output);

            // STEP 3: Apply twiddle factors
            for (column_index, output_cell) in output.iter_mut().enumerate()
            {

                let twiddle = unsafe { *self.twiddles.get_unchecked(row_index * column_index) };
                *output_cell = *output_cell * twiddle;
            }
        }

        // STEP 4: transpose again
        array_utils::transpose(self.height, self.width, self.scratch.as_slice(), spectrum);

        // STEP 5: perform FFTs of size 'width'
        for (input, output) in spectrum.chunks(self.width).zip(self.scratch.chunks_mut(self.width))
        {
            self.width_size_fft.process(input, output);
        }

        // STEP 6: transpose again
        array_utils::transpose(self.width, self.height, self.scratch.as_slice(), spectrum);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ::test::{MockFFTAlgorithm, random_signal, compare_vectors};
    use ::dft;

    #[test]
    fn test_mixed_radix_single() {
        for width in 2..11 {
            for height in 2..11 {
                let width_fft = Box::new(MockFFTAlgorithm {}) as Box<FFTAlgorithm<f32>>;
                let height_fft = Box::new(MockFFTAlgorithm {}) as Box<FFTAlgorithm<f32>>;

                let mut mixed_radix_fft = MixedRadixSingle::new(width, width_fft, height, height_fft, false);

                let input = random_signal(width * height);

                let mut expected = input.clone();
                dft(&input, &mut expected);

                let mut actual = input.clone();
                mixed_radix_fft.process(&input, &mut actual);

                println!("expected:");
                for expected_chunk in expected.chunks(width) {
                    println!("{:?}", expected_chunk);
                }
                println!("");
                println!("actual:");
                for actual_chunk in actual.chunks(width) {
                    println!("{:?}", actual_chunk);
                }

                assert!(compare_vectors(&actual, &expected), "width = {}, height = {}", width, height);
            }
        }
    }
}