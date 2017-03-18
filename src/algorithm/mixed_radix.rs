use std::rc::Rc;

use num::Complex;

use common::{FFTnum, verify_length, verify_length_divisible};

use algorithm::{FFTAlgorithm, ButterflyEnum};
use array_utils;
use twiddles;

pub struct MixedRadix<T> {
    width: usize,
    width_size_fft: Rc<FFTAlgorithm<T>>,

    height: usize,
    height_size_fft: Rc<FFTAlgorithm<T>>,

    twiddles: Box<[Complex<T>]>,
}

impl<T: FFTnum> MixedRadix<T> {
    pub fn new(width_fft: Rc<FFTAlgorithm<T>>, height_fft: Rc<FFTAlgorithm<T>>, inverse: bool) -> Self {

        let width = width_fft.len();
        let height = height_fft.len();

        let len = width * height;

        let mut twiddles = Vec::with_capacity(len);
        for x in 0..width {
            for y in 0..height {
                twiddles.push(twiddles::single_twiddle(x * y, len, inverse));
            }
        }

        MixedRadix {
            width: width,
            width_size_fft: width_fft,

            height: height,
            height_size_fft: height_fft,

            twiddles: twiddles.into_boxed_slice(),
        }
    }


    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        // SIX STEP FFT:

        // STEP 1: transpose
        array_utils::transpose(self.width, self.height, input, output);

        // STEP 2: perform FFTs of size `height`
        self.height_size_fft.process_multi(output, input);

        // STEP 3: Apply twiddle factors
        for (element, &twiddle) in input.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        array_utils::transpose(self.height, self.width, input, output);

        // STEP 5: perform FFTs of size `width`
        self.width_size_fft.process_multi(output, input);

        // STEP 6: transpose again
        array_utils::transpose(self.width, self.height, input, output);
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for MixedRadix<T> {
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
        self.twiddles.len()
    }
}




/// This struct is the same as MixedRadixSingle, except it's specialized for the case where both inner FFTs are butterflies
pub struct MixedRadixDoubleButterfly<T> {
    width: usize,
    width_size_fft: ButterflyEnum<T>,

    height: usize,
    height_size_fft: ButterflyEnum<T>,

    twiddles: Box<[Complex<T>]>,
}

impl<T: FFTnum> MixedRadixDoubleButterfly<T> {
    pub fn new(width_fft: ButterflyEnum<T>, height_fft: ButterflyEnum<T>, inverse: bool) -> Self {
        let width = width_fft.len();
        let height = height_fft.len();

        let len = width * height;

        let mut twiddles = Vec::with_capacity(len);
        for x in 0..width {
            for y in 0..height {
                twiddles.push(twiddles::single_twiddle(x * y, len, inverse));
            }
        }

        MixedRadixDoubleButterfly {
            width: width,
            width_size_fft: width_fft,

            height: height,
            height_size_fft: height_fft,

            twiddles: twiddles.into_boxed_slice(),
        }
    }


    unsafe fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        // SIX STEP FFT:

        // STEP 1: transpose
        array_utils::transpose_small(self.width, self.height, input, output);

        // STEP 2: perform FFTs of size 'height'
        self.height_size_fft.process_multi_inplace(output);

        // STEP 3: Apply twiddle factors
        for (element, &twiddle) in output.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        array_utils::transpose_small(self.height, self.width, output, input);

        // STEP 5: perform FFTs of size 'width'
        self.width_size_fft.process_multi_inplace(input);

        // STEP 6: transpose again
        array_utils::transpose_small(self.width, self.height, input, output);
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for MixedRadixDoubleButterfly<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

        unsafe { self.perform_fft(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            unsafe { self.perform_fft(in_chunk, out_chunk) };
        }
    }
    fn len(&self) -> usize {
        self.twiddles.len()
    }
}



#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::rc::Rc;
    use test_utils::{random_signal, compare_vectors};
    use algorithm::{butterflies, DFT};
    use num::Zero;

    #[test]
    fn test_mixed_radix() {
        for width in 1..7 {
            for height in 1..7 {
                test_mixed_radix_with_lengths(width, height, false);
                test_mixed_radix_with_lengths(width, height, true);
            }
        }
    }

    #[test]
    fn test_mixed_radix_double_butterfly() {
        for width in 2..7 {
            for height in 2..7 {
                test_mixed_radix_butterfly_with_lengths(width, height, false);
                test_mixed_radix_butterfly_with_lengths(width, height, true);
            }
        }
    }




    fn test_mixed_radix_with_lengths(width: usize, height: usize, inverse: bool) {
        let n = 4;
        let len = width * height;

        // set up algorithms
        let dft = DFT::new(len, inverse);

        let width_fft = Rc::new(DFT::new(width, inverse)) as Rc<FFTAlgorithm<f32>>;
        let height_fft = Rc::new(DFT::new(height, inverse)) as Rc<FFTAlgorithm<f32>>;

        let mixed_fft = MixedRadix::new(width_fft, height_fft, inverse);

        assert_eq!(mixed_fft.len(), len, "Mixed radix algorithm reported incorrect length");

        // set up buffers
        let mut expected_input = random_signal(len * n);
        let mut actual_input = expected_input.clone();
        let mut multi_input = expected_input.clone();

        let mut expected_output = vec![Zero::zero(); len * n];
        let mut actual_output = expected_output.clone();
        let mut multi_output = expected_output.clone();

        // perform the test
        dft.process_multi(&mut expected_input, &mut expected_output);
        mixed_fft.process_multi(&mut multi_input, &mut multi_output);

        for (input_chunk, output_chunk) in actual_input.chunks_mut(len).zip(actual_output.chunks_mut(len)) {
            mixed_fft.process(input_chunk, output_chunk);
        }

        assert!(compare_vectors(&expected_output, &actual_output), "process() failed, width = {}, height = {}, inverse = {}", width, height, inverse);
        assert!(compare_vectors(&expected_output, &multi_output), "process_multi() failed, width = {}, height = {}, inverse = {}", width, height, inverse);
    }

    fn test_mixed_radix_butterfly_with_lengths(width: usize, height: usize, inverse: bool) {
        let n = 4;
        let len = width * height;

        // set up algorithms
        let dft = DFT::new(len, inverse);

        let width_fft = make_butterfly(width, inverse);
        let height_fft = make_butterfly(height, inverse);

        let mixed_fft = MixedRadixDoubleButterfly::new(width_fft, height_fft, inverse);

        assert_eq!(mixed_fft.len(), len, "Mixed radix butterfly algorithm reported incorrect length");

        // set up buffers
        let mut expected_input = random_signal(len * n);
        let mut actual_input = expected_input.clone();
        let mut multi_input = expected_input.clone();

        let mut expected_output = vec![Zero::zero(); len * n];
        let mut actual_output = expected_output.clone();
        let mut multi_output = expected_output.clone();

        // perform the test
        dft.process_multi(&mut expected_input, &mut expected_output);
        mixed_fft.process_multi(&mut multi_input, &mut multi_output);

        for (input_chunk, output_chunk) in actual_input.chunks_mut(len).zip(actual_output.chunks_mut(len)) {
            mixed_fft.process(input_chunk, output_chunk);
        }

        assert!(compare_vectors(&expected_output, &actual_output), "process() failed, width = {}, height = {}, inverse = {}", width, height, inverse);
        assert!(compare_vectors(&expected_output, &multi_output), "process_multi() failed, width = {}, height = {}, inverse = {}", width, height, inverse);
    }

    fn make_butterfly<T: FFTnum>(len: usize, inverse: bool) -> ButterflyEnum<T> {
        match len {
            2 => ButterflyEnum::Butterfly2(butterflies::Butterfly2{}),
            3 => ButterflyEnum::Butterfly3(butterflies::Butterfly3::new(inverse)),
            4 => ButterflyEnum::Butterfly4(butterflies::Butterfly4::new(inverse)),
            5 => ButterflyEnum::Butterfly5(Box::new(butterflies::Butterfly5::new(inverse))),
            6 => ButterflyEnum::Butterfly6(butterflies::Butterfly6::new(inverse)),
            _ => panic!("Invalid butterfly size: {}", len)
        }
    }
}