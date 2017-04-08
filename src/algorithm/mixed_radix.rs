use std::rc::Rc;

use num_complex::Complex;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, FFT};
use algorithm::butterflies::FFTButterfly;
use array_utils;
use twiddles;

pub struct MixedRadix<T> {
    width: usize,
    width_size_fft: Rc<FFT<T>>,

    height: usize,
    height_size_fft: Rc<FFT<T>>,

    twiddles: Box<[Complex<T>]>,
}

impl<T: FFTnum> MixedRadix<T> {
    pub fn new(width_fft: Rc<FFT<T>>, height_fft: Rc<FFT<T>>, inverse: bool) -> Self {

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
impl<T: FFTnum> FFT<T> for MixedRadix<T> {
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
impl<T> Length for MixedRadix<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len()
    }
}




/// This struct is the same as MixedRadixSingle, except it's specialized for the case where both inner FFTs are butterflies
pub struct MixedRadixDoubleButterfly<T> {
    width: usize,
    width_size_fft: Rc<FFTButterfly<T>>,

    height: usize,
    height_size_fft: Rc<FFTButterfly<T>>,

    twiddles: Box<[Complex<T>]>,
}

impl<T: FFTnum> MixedRadixDoubleButterfly<T> {
    pub fn new(width_fft: Rc<FFTButterfly<T>>, height_fft: Rc<FFTButterfly<T>>, inverse: bool) -> Self {
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

impl<T: FFTnum> FFT<T> for MixedRadixDoubleButterfly<T> {
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
}
impl<T> Length for MixedRadixDoubleButterfly<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len()
    }
}



#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::rc::Rc;
    use test_utils::check_fft_algorithm;
    use algorithm::{butterflies, DFT};

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
        let width_fft = Rc::new(DFT::new(width, inverse)) as Rc<FFT<f32>>;
        let height_fft = Rc::new(DFT::new(height, inverse)) as Rc<FFT<f32>>;

        let fft = MixedRadix::new(width_fft, height_fft, inverse);

        check_fft_algorithm(&fft, width * height, inverse);
    }

    fn test_mixed_radix_butterfly_with_lengths(width: usize, height: usize, inverse: bool) {
        let width_fft = make_butterfly(width, inverse);
        let height_fft = make_butterfly(height, inverse);

        let fft = MixedRadixDoubleButterfly::new(width_fft, height_fft, inverse);

        check_fft_algorithm(&fft, width * height, inverse);
    }

    fn make_butterfly(len: usize, inverse: bool) -> Rc<FFTButterfly<f32>> {
        match len {
            2 => Rc::new(butterflies::Butterfly2 {}),
            3 => Rc::new(butterflies::Butterfly3::new(inverse)),
            4 => Rc::new(butterflies::Butterfly4::new(inverse)),
            5 => Rc::new(butterflies::Butterfly5::new(inverse)),
            6 => Rc::new(butterflies::Butterfly6::new(inverse)),
            7 => Rc::new(butterflies::Butterfly7::new(inverse)),
            8 => Rc::new(butterflies::Butterfly8::new(inverse)),
            16 => Rc::new(butterflies::Butterfly16::new(inverse)),
            _ => panic!("Invalid butterfly size: {}", len),
        }
    }
}