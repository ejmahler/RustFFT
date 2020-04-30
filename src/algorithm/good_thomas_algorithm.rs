use std::sync::Arc;
use std::cmp::max;

use num_complex::Complex;
use strength_reduce::StrengthReducedUsize;
use transpose;

use crate::common::FFTnum;

use crate::math_utils;
use crate::array_utils;

use crate::{Length, IsInverse, Fft};

/// Implementation of the [Good-Thomas Algorithm (AKA Prime Factor Algorithm)](https://en.wikipedia.org/wiki/Prime-factor_FFT_algorithm)
///
/// This algorithm factors a size n FFT into n1 * n2, where GCD(n1, n2) == 1
///
/// Conceptually, this algorithm is very similar to the Mixed-Radix except because GCD(n1, n2) == 1 we can do some
/// number theory trickery to reduce the number of floating-point multiplications and additions. Additionally, It can
/// be faster than Mixed-Radix at sizes below 10,000 or so.
///
/// ~~~
/// // Computes a forward FFT of size 1200, using the Good-Thomas Algorithm
/// use rustfft::algorithm::GoodThomasAlgorithm;
/// use rustfft::{Fft, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1200];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1200];
///
/// // we need to find an n1 and n2 such that n1 * n2 == 1200 and GCD(n1, n2) == 1
/// // n1 = 48 and n2 = 25 satisfies this
/// let mut planner = FFTplanner::new(false);
/// let inner_fft_n1 = planner.plan_fft(48);
/// let inner_fft_n2 = planner.plan_fft(25);
///
/// // the good-thomas FFT length will be inner_fft_n1.len() * inner_fft_n2.len() = 1200
/// let fft = GoodThomasAlgorithm::new(inner_fft_n1, inner_fft_n2);
/// fft.process(&mut input, &mut output);
/// ~~~
pub struct GoodThomasAlgorithm<T> {
    width: usize,
    width_size_fft: Arc<dyn Fft<T>>,

    height: usize,
    height_size_fft: Arc<dyn Fft<T>>,

    input_x_stride: usize,
    input_y_stride: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,

    len: StrengthReducedUsize,
    inverse: bool,
}

impl<T: FFTnum> GoodThomasAlgorithm<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `width_fft.len() * height_fft.len()`
    ///
    /// GCD(width_fft.len(), height_fft.len()) must be equal to 1
    pub fn new(width_fft: Arc<dyn Fft<T>>, height_fft: Arc<dyn Fft<T>>) -> Self {
        assert_eq!(
            width_fft.is_inverse(), height_fft.is_inverse(), 
            "width_fft and height_fft must both be inverse, or neither. got width inverse={}, height inverse={}",
            width_fft.is_inverse(), height_fft.is_inverse());

        let width = width_fft.len();
        let height = height_fft.len();
        let is_inverse = width_fft.is_inverse();

        // compute the nultiplicative inverse of width mod height and vice versa
        let (gcd, mut width_inverse, mut height_inverse) =
            math_utils::extended_euclidean_algorithm(width as i64, height as i64);
        assert!(gcd == 1,
                "Invalid input width and height to Good-Thomas Algorithm: ({},{}): Inputs must be coprime",
                width,
                height);

        // width_inverse or height_inverse might be negative, make it positive
        if width_inverse < 0 {
            width_inverse += height as i64;
        }
        if height_inverse < 0 {
            height_inverse += width as i64;
        }

        let len = width * height;
        let width_inplace_scratch = height_fft.get_inplace_scratch_len();
        let height_inplace_scratch = width_fft.get_inplace_scratch_len();
        let height_outofplace_scratch = width_fft.get_out_of_place_scratch_len();

        let outofplace_scratch = max(height_inplace_scratch, width_inplace_scratch);
        let inplace_extra = max(if width_inplace_scratch > len { width_inplace_scratch } else { 0 }, height_outofplace_scratch);

        Self {
            width: width,
            width_size_fft: width_fft,

            height: height,
            height_size_fft: height_fft,

            input_x_stride: height_inverse as usize * height,
            input_y_stride: width_inverse as usize * width,

            inplace_scratch_len: len + inplace_extra,
            outofplace_scratch_len: if outofplace_scratch > len { outofplace_scratch } else { 0 },

            len: StrengthReducedUsize::new(width * height),
            inverse: is_inverse,
        }
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.len());

        // copy the input into the output buffer
        for (y, row) in scratch.chunks_mut(self.width).enumerate() {
            let input_base = y * self.input_y_stride;
            for (x, output_cell) in row.iter_mut().enumerate() {
                let input_index = (input_base + x * self.input_x_stride) % self.len;
                *output_cell = buffer[input_index];
            }
        }

        // run FFTs of size `width`
        let width_scratch = if inner_scratch.len() > buffer.len() { &mut inner_scratch[..] } else { &mut buffer[..] };
        self.width_size_fft.process_inplace_multi(scratch, width_scratch);

        // transpose
        transpose::transpose(scratch, buffer, self.width, self.height);

        // run FFTs of size 'height'
        self.height_size_fft.process_multi(buffer, scratch, inner_scratch);

        // copy to the output, using our output redordering mapping
        for (x, row) in scratch.chunks(self.height).enumerate() {
            let output_base = x * self.height;
            for (y, input_cell) in row.iter().enumerate() {
                let output_index = (output_base + y * self.width) % self.len;
                buffer[output_index] = *input_cell;
            }
        }
    }

    fn perform_fft_out_of_place(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // copy the input into the output buffer
        for (y, row) in output.chunks_mut(self.width).enumerate() {
            let input_base = y * self.input_y_stride;
            for (x, output_cell) in row.iter_mut().enumerate() {
                let input_index = (input_base + x * self.input_x_stride) % self.len;
                *output_cell = input[input_index];
            }
        }

        // run FFTs of size `width`
        let width_scratch = if scratch.len() > input.len() { &mut scratch[..] } else { &mut input[..] };
        self.width_size_fft.process_inplace_multi(output, width_scratch);

        // transpose
        transpose::transpose(output, input, self.width, self.height);

        // run FFTs of size 'height'
        let height_scratch = if scratch.len() > output.len() { &mut scratch[..] } else { &mut output[..] };
        self.height_size_fft.process_inplace_multi(input, height_scratch);

        // copy to the output, using our output redordering mapping
        for (x, row) in input.chunks(self.height).enumerate() {
            let output_base = x * self.height;
            for (y, input_cell) in row.iter().enumerate() {
                let output_index = (output_base + y * self.width) % self.len;
                output[output_index] = *input_cell;
            }
        }
    }
}
boilerplate_fft!(GoodThomasAlgorithm,
    |this: &GoodThomasAlgorithm<_>| this.len.get(),
    |this: &GoodThomasAlgorithm<_>| this.inplace_scratch_len,
    |this: &GoodThomasAlgorithm<_>| this.outofplace_scratch_len
);

/// Implementation of the Good-Thomas Algorithm, specialized for smaller input sizes
///
/// This algorithm factors a size n FFT into n1 * n2, where GCD(n1, n2) == 1
///
/// Conceptually, this algorithm is very similar to MixedRadix except because GCD(n1, n2) == 1 we can do some
/// number theory trickery to reduce the number of floating point operations. It typically performs
/// better than MixedRadixSmall, especially at the smallest sizes.
///
/// ~~~
/// // Computes a forward FFT of size 56, using the Good-Thoma Butterfly Algorithm
/// use std::sync::Arc;
/// use rustfft::algorithm::GoodThomasAlgorithmSmall;
/// use rustfft::algorithm::butterflies::{Butterfly7, Butterfly8};
/// use rustfft::Fft;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 56];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 56];
///
/// // we need to find an n1 and n2 such that n1 * n2 == 56 and GCD(n1, n2) == 1
/// // n1 = 7 and n2 = 8 satisfies this
/// let inner_fft_n1 = Arc::new(Butterfly7::new(false));
/// let inner_fft_n2 = Arc::new(Butterfly8::new(false));
///
/// // the good-thomas FFT length will be inner_fft_n1.len() * inner_fft_n2.len() = 56
/// let fft = GoodThomasAlgorithmSmall::new(inner_fft_n1, inner_fft_n2);
/// fft.process(&mut input, &mut output);
/// ~~~
pub struct GoodThomasAlgorithmSmall<T> {
    width: usize,
    width_size_fft: Arc<dyn Fft<T>>,

    height: usize,
    height_size_fft: Arc<dyn Fft<T>>,

    input_output_map: Box<[usize]>,

    inverse: bool,
}

impl<T: FFTnum> GoodThomasAlgorithmSmall<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `width_fft.len() * height_fft.len()`
    ///
    /// GCD(n1.len(), n2.len()) must be equal to 1
    pub fn new(width_fft: Arc<dyn Fft<T>>, height_fft: Arc<dyn Fft<T>>) -> Self {
        assert_eq!(
            width_fft.is_inverse(), height_fft.is_inverse(), 
            "n1_fft and height_fft must both be inverse, or neither. got width inverse={}, height inverse={}",
            width_fft.is_inverse(), height_fft.is_inverse());

        let width = width_fft.len();
        let height = height_fft.len();
        let len = width * height;

        // compute the nultiplicative inverse of n1 mod height and vice versa
        let (gcd, mut width_inverse, mut height_inverse) =
            math_utils::extended_euclidean_algorithm(width as i64, height as i64);
        assert!(gcd == 1,
                "Invalid input n1 and height to Good-Thomas Algorithm: ({},{}): Inputs must be coprime",
                width,
                height);

        // width_inverse or height_inverse might be negative, make it positive
        if width_inverse < 0 {
            width_inverse += height as i64;
        }
        if height_inverse < 0 {
            height_inverse += width as i64;
        }

        // NOTE: we are precomputing the input and output reordering indexes, because benchmarking shows that it's 10-20% faster
        // If we wanted to optimize for memory use or setup time instead of multiple-FFT speed, we could compute these on the fly in the perform_fft() method
        let input_iter = (0..len)
                .map(|i| (i % width, i / width))
                .map(|(x, y)| (x * height + y * width) % len);
        let output_iter = (0..len)
                .map(|i| (i % height, i / height))
                .map(|(y, x)| (x * height * height_inverse as usize + y * width * width_inverse as usize) % len);

        let input_output_map: Vec<usize> = input_iter.chain(output_iter).collect();

        Self {
            inverse: width_fft.is_inverse(),

            width: width,
            width_size_fft: width_fft,

            height: height,
            height_size_fft: height_fft,
            
            input_output_map: input_output_map.into_boxed_slice(),
        }
    }

    fn perform_fft_out_of_place(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        // These asserts are for the unsafe blocks down below. we're relying on the optimizer to get rid of this assert
        assert_eq!(self.len(), input.len());
        assert_eq!(self.len(), output.len());

        let (input_map, output_map) = self.input_output_map.split_at(self.len());

        // copy the input using our reordering mapping
        for (output_element, &input_index) in output.iter_mut().zip(input_map.iter()) {
            *output_element = input[input_index];
        }

        // run FFTs of size `width`
        self.width_size_fft.process_inplace_multi(output, input);

        // transpose
        unsafe { array_utils::transpose_small(self.width, self.height, output, input) };

        // run FFTs of size 'height'
        self.height_size_fft.process_inplace_multi(input, output);

        // copy to the output, using our output redordeing mapping
        for (input_element, &output_index) in input.iter().zip(output_map.iter()) {
            output[output_index] = *input_element;
        }
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // These asserts are for the unsafe blocks down below. we're relying on the optimizer to get rid of this assert
        assert_eq!(self.len(), buffer.len());
        assert_eq!(self.len(), scratch.len());

        let (input_map, output_map) = self.input_output_map.split_at(self.len());

        // copy the input using our reordering mapping
        for (output_element, &input_index) in scratch.iter_mut().zip(input_map.iter()) {
            *output_element = buffer[input_index];
        }

        // run FFTs of size `width`
        self.width_size_fft.process_inplace_multi(scratch, buffer);

        // transpose
        unsafe { array_utils::transpose_small(self.width, self.height, scratch, buffer) };

        // run FFTs of size 'height'
        self.height_size_fft.process_multi(buffer, scratch, &mut []);

        // copy to the output, using our output redordeing mapping
        for (input_element, &output_index) in scratch.iter().zip(output_map.iter()) {
            buffer[output_index] = *input_element;
        }
    }
}
boilerplate_fft!(GoodThomasAlgorithmSmall, 
    |this: &GoodThomasAlgorithmSmall<_>| this.width * this.height,
    |this: &GoodThomasAlgorithmSmall<_>| this.len(),
    |_| 0
);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use crate::test_utils::check_fft_algorithm;
    use crate::algorithm::DFT;
    use num_integer::gcd;

    #[test]
    fn test_good_thomas() {
        for width in 1..12 {
            for height in 1..12 {
                if gcd(width, height) == 1 {
                    test_good_thomas_with_lengths(width, height, false);
                    test_good_thomas_with_lengths(width, height, true);
                }
            }
        }
    }

    #[test]
    fn test_good_thomas_small() {
        let butterfly_sizes = [2,3,4,5,6,7,8,16];
        for width in &butterfly_sizes {
            for height in &butterfly_sizes {
                if gcd(*width, *height) == 1 {
                    test_good_thomas_small_with_lengths(*width, *height, false);
                    test_good_thomas_small_with_lengths(*width, *height, true);
                }
            }
        }
    }

    fn test_good_thomas_with_lengths(width: usize, height: usize, inverse: bool) {
        let width_fft = Arc::new(DFT::new(width, inverse)) as Arc<dyn Fft<f32>>;
        let height_fft = Arc::new(DFT::new(height, inverse)) as Arc<dyn Fft<f32>>;

        let fft = GoodThomasAlgorithm::new(width_fft, height_fft);

        check_fft_algorithm(&fft, width * height, inverse);
    }

    fn test_good_thomas_small_with_lengths(width: usize, height: usize, inverse: bool) {
        let width_fft = Arc::new(DFT::new(width, inverse)) as Arc<dyn Fft<f32>>;
        let height_fft = Arc::new(DFT::new(height, inverse)) as Arc<dyn Fft<f32>>;

        let fft = GoodThomasAlgorithmSmall::new(width_fft, height_fft);

        check_fft_algorithm(&fft, width * height, inverse);
    }
}
