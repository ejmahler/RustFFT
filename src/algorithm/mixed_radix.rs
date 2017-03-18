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
        for width in 2..11 {
            for height in 2..11 {
                let width_fft = Rc::new(DFT::new(width, false)) as Rc<FFTAlgorithm<f32>>;
                let height_fft = Rc::new(DFT::new(height, false)) as Rc<FFTAlgorithm<f32>>;

                let mixed_radix_fft = MixedRadix::new(width_fft, height_fft, false);

                let mut expected_input = random_signal(width * height);
                let mut actual_input = expected_input.clone();

                let mut expected_output = vec![Zero::zero(); width * height];
                let mut actual_output = vec![Zero::zero(); width * height];

                let dft = DFT::new(width * height, false);
                dft.process(&mut expected_input, &mut expected_output);

                mixed_radix_fft.process(&mut actual_input, &mut actual_output);

                assert!(compare_vectors(&actual_output, &expected_output), "width = {}, height = {}", width, height);
            }
        }
    }

    #[test]
    fn test_mixed_radix_double_butterfly() {
        for &width in &[2,3,4,5,6] {
            for &height in &[2,3,4,5,6] {
                let width_fft = Rc::new(DFT::new(width, false)) as Rc<FFTAlgorithm<f32>>;
                let height_fft = Rc::new(DFT::new(height, false)) as Rc<FFTAlgorithm<f32>>;
                let control_fft = MixedRadix::new(width_fft, height_fft, false);

                let width_butterfly = make_butterfly(width, false);
                let height_butterfly = make_butterfly(height, false);
                let test_fft = MixedRadixDoubleButterfly::new(width_butterfly, height_butterfly, false);

                let mut control_input = random_signal(width * height);
                let mut test_input = control_input.clone();

                let mut expected = control_input.clone();
                control_fft.process(&mut control_input, &mut expected);

                let mut actual = test_input.clone();
                test_fft.process(&mut test_input, &mut actual);

                assert!(compare_vectors(&actual, &expected), "width = {}, height = {}", width, height);
            }
        }
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