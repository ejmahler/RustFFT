use num::{Complex, Zero};

use common::FFTnum;

use algorithm::{FFTAlgorithm, ButterflyEnum};
use array_utils;
use twiddles;

pub struct MixedRadix<T> {
    width: usize,
    width_size_fft: Box<FFTAlgorithm<T>>,

    height: usize,
    height_size_fft: Box<FFTAlgorithm<T>>,

    twiddles: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: FFTnum> MixedRadix<T> {
    pub fn new(width: usize,
               width_fft: Box<FFTAlgorithm<T>>,
               height: usize,
               height_fft: Box<FFTAlgorithm<T>>,
               inverse: bool) -> Self {

        let len = width * height;

        Self {
            width: width,
            width_size_fft: width_fft,

            height: height,
            height_size_fft: height_fft,

            twiddles: twiddles::generate_twiddle_factors(len, inverse),
            scratch: vec![Zero::zero(); len],
        }
    }

    /// Runs the FFT on the input `signal` array, placing the output in the 'spectrum' array
    fn perform_fft(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        // SIX STEP FFT:

        // STEP 1: transpose
        array_utils::transpose(self.width, self.height, signal, spectrum);

        // STEP 2: perform FFTs of size 'height'
        self.height_size_fft.process_multi(spectrum, &mut self.scratch);

        // STEP 3: Apply twiddle factors. we skip row 0 and column 0 because the
        // twiddle factor for row/column 0 is always the identity
        for (row, chunk) in self.scratch.chunks_mut(self.height).enumerate().skip(1)
        {
            for (column, cell) in chunk.iter_mut().enumerate().skip(1)
            {

                let twiddle = unsafe { *self.twiddles.get_unchecked(row * column) };
                *cell = *cell * twiddle;
            }
        }

        // STEP 4: transpose again
        array_utils::transpose(self.height, self.width, self.scratch.as_slice(), spectrum);

        // STEP 5: perform FFTs of size 'width'
        self.width_size_fft.process_multi(spectrum, &mut self.scratch);

        // STEP 6: transpose again
        array_utils::transpose(self.width, self.height, self.scratch.as_slice(), spectrum);
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for MixedRadix<T> {
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        self.perform_fft(signal, spectrum);
    }
    fn process_multi(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        for (input, output) in signal.chunks(self.twiddles.len()).zip(spectrum.chunks_mut(self.twiddles.len())) {
            self.perform_fft(input, output);
        }
    }
}







/// This struct is the same as MixedRadixSingle, except it's specialized for the case where one of the inner FFTs is a butterfly
pub struct MixedRadixSingleButterfly<T> {
    inner_fft_len: usize,
    inner_fft: Box<FFTAlgorithm<T>>,

    butterfly_len: usize,
    butterfly_fft: ButterflyEnum<T>,

    twiddles: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: FFTnum> MixedRadixSingleButterfly<T> {
    pub fn new(inner_fft_len: usize,
               inner_fft: Box<FFTAlgorithm<T>>,
               butterfly_len: usize,
               butterfly_fft: ButterflyEnum<T>,
               inverse: bool) -> Self {

        let len = inner_fft_len * butterfly_len;

        Self {
            inner_fft_len: inner_fft_len,
            inner_fft: inner_fft,

            butterfly_len: butterfly_len,
            butterfly_fft: butterfly_fft,

            twiddles: twiddles::generate_twiddle_factors(len, inverse),
            scratch: vec![Zero::zero(); len],
        }
    }

    /// Runs the FFT on the input `signal` array, placing the output in the 'spectrum' array
    fn perform_fft(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        // SIX STEP FFT:

        // STEP 1: transpose
        array_utils::transpose(self.inner_fft_len, self.butterfly_len, signal, &mut self.scratch);

        // STEP 2: perform the butterfly FFTs
        unsafe { self.butterfly_fft.process_multi_inplace(&mut self.scratch) };

        // STEP 3: Apply twiddle factors. we skip row 0 and column 0 because the
        // twiddle factor for row/column 0 is always the identity
        for (row, chunk) in self.scratch.chunks_mut(self.butterfly_len).enumerate().skip(1)
        {
            for (column, cell) in chunk.iter_mut().enumerate().skip(1)
            {

                let twiddle = unsafe { *self.twiddles.get_unchecked(row * column) };
                *cell = *cell * twiddle;
            }
        }

        // STEP 4: transpose again
        array_utils::transpose(self.butterfly_len, self.inner_fft_len, &self.scratch, spectrum);

        // STEP 5: perform the intter FFTs
        self.inner_fft.process_multi(spectrum, &mut self.scratch);

        // STEP 6: transpose again
        array_utils::transpose(self.inner_fft_len, self.butterfly_len, &self.scratch, spectrum);
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for MixedRadixSingleButterfly<T> {
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        self.perform_fft(signal, spectrum);
    }
    fn process_multi(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        for (input, output) in signal.chunks(self.twiddles.len()).zip(spectrum.chunks_mut(self.twiddles.len())) {
            self.perform_fft(input, output);
        }
    }
}




/// This struct is the same as MixedRadixSingle, except it's specialized for the case where both inner FFTs are butterflies
pub struct MixedRadixDoubleButterfly<T> {
    width: usize,
    width_size_fft: ButterflyEnum<T>,

    height: usize,
    height_size_fft: ButterflyEnum<T>,

    twiddles: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: FFTnum> MixedRadixDoubleButterfly<T> {
    pub fn new(width: usize,
               width_fft: ButterflyEnum<T>,
               height: usize,
               height_fft: ButterflyEnum<T>,
               inverse: bool) -> Self {

        let len = width * height;

        Self {
            width: width,
            width_size_fft: width_fft,

            height: height,
            height_size_fft: height_fft,

            twiddles: twiddles::generate_twiddle_factors(len, inverse),
            scratch: vec![Zero::zero(); len],
        }
    }

    /// Runs the FFT on the input `signal` array, placing the output in the 'spectrum' array
    fn perform_fft(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        // SIX STEP FFT:

        // STEP 1: transpose
        array_utils::transpose(self.width, self.height, signal, spectrum);

        // STEP 2: perform FFTs of size 'height'
        unsafe { self.height_size_fft.process_multi_inplace(spectrum) };

        // STEP 3: Apply twiddle factors. we skip row 0 and column 0 because the
        // twiddle factor for row/column 0 is always the identity
        for (row, chunk) in spectrum.chunks_mut(self.height).enumerate().skip(1)
        {
            for (column, cell) in chunk.iter_mut().enumerate().skip(1)
            {

                let twiddle = unsafe { *self.twiddles.get_unchecked(row * column) };
                *cell = *cell * twiddle;
            }
        }

        // STEP 4: transpose again
        array_utils::transpose(self.height, self.width, spectrum, &mut self.scratch);

        // STEP 5: perform FFTs of size 'width'
        unsafe { self.width_size_fft.process_multi_inplace(&mut self.scratch) };

        // STEP 6: transpose again
        array_utils::transpose(self.width, self.height, self.scratch.as_slice(), spectrum);
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for MixedRadixDoubleButterfly<T> {
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        self.perform_fft(signal, spectrum);
    }
    fn process_multi(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        for (input, output) in signal.chunks(self.twiddles.len()).zip(spectrum.chunks_mut(self.twiddles.len())) {
            self.perform_fft(input, output);
        }
    }
}



#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::{random_signal, compare_vectors};
    use dft;
    use algorithm::{butterflies, DFTAlgorithm};

    #[test]
    fn test_mixed_radix() {
        for width in 2..11 {
            for height in 2..11 {
                let width_fft = Box::new(DFTAlgorithm::new(width, false)) as Box<FFTAlgorithm<f32>>;
                let height_fft = Box::new(DFTAlgorithm::new(height, false)) as Box<FFTAlgorithm<f32>>;

                let mut mixed_radix_fft = MixedRadix::new(width, width_fft, height, height_fft, false);

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

    #[test]
    fn test_mixed_radix_single_butterfly() {
        for width in 2..11 {
            for &height in &[2,3,4,5,6] {
                let width_fft = Box::new(DFTAlgorithm::new(width, false)) as Box<FFTAlgorithm<f32>>;
                let height_fft = Box::new(DFTAlgorithm::new(height, false)) as Box<FFTAlgorithm<f32>>;
                let mut control_fft = MixedRadix::new(width, width_fft, height, height_fft, false);

                let inner_fft = Box::new(DFTAlgorithm::new(width, false)) as Box<FFTAlgorithm<f32>>;
                let butterfly_fft = make_butterfly(height, false);
                let mut test_fft = MixedRadixSingleButterfly::new(width, inner_fft, height, butterfly_fft, false);

                let input = random_signal(width * height);

                let mut expected = input.clone();
                control_fft.process(&input, &mut expected);

                let mut actual = input.clone();
                test_fft.process(&input, &mut actual);

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

    #[test]
    fn test_mixed_radix_double_butterfly() {
        for &width in &[2,3,4,5,6] {
            for &height in &[2,3,4,5,6] {
                let width_fft = Box::new(DFTAlgorithm::new(width, false)) as Box<FFTAlgorithm<f32>>;
                let height_fft = Box::new(DFTAlgorithm::new(height, false)) as Box<FFTAlgorithm<f32>>;
                let mut control_fft = MixedRadix::new(width, width_fft, height, height_fft, false);

                let width_butterfly = make_butterfly(width, false);
                let height_butterfly = make_butterfly(height, false);
                let mut test_fft = MixedRadixDoubleButterfly::new(width, width_butterfly, height, height_butterfly, false);

                let input = random_signal(width * height);

                let mut expected = input.clone();
                control_fft.process(&input, &mut expected);

                let mut actual = input.clone();
                test_fft.process(&input, &mut actual);

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
    fn make_butterfly<T: FFTnum>(len: usize, inverse: bool) -> ButterflyEnum<T> {
        match len {
            2 => ButterflyEnum::Butterfly2(butterflies::Butterfly2{}),
            3 => ButterflyEnum::Butterfly3(butterflies::Butterfly3::new(inverse)),
            4 => ButterflyEnum::Butterfly4(butterflies::Butterfly4::new(inverse)),
            5 => ButterflyEnum::Butterfly5(butterflies::Butterfly5::new(inverse)),
            6 => ButterflyEnum::Butterfly6(butterflies::Butterfly6::new(inverse)),
            _ => panic!("Invalid butterfly size: {}", len)
        }
    }
}