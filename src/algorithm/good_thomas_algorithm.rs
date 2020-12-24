use std::sync::Arc;

use num_complex::Complex;
use strength_reduce::StrengthReducedUsize;
use transpose;

use crate::common::{verify_length, verify_length_divisible, FFTnum};

use crate::array_utils;
use crate::math_utils;

use crate::algorithm::butterflies::FFTButterfly;
use crate::{IsInverse, Length, Fft};

/// Implementation of the [Good-Thomas Algorithm (AKA Prime Factor Algorithm)](https://en.wikipedia.org/wiki/Prime-factor_FFT_algorithm)
///
/// This algorithm factors a size n FFT into n1 * n2, where GCD(n1, n2) == 1
///
/// Conceptually, this algorithm is very similar to the Mixed-Radix FFT, except because GCD(n1, n2) == 1 we can do some
/// number theory trickery to reduce the number of floating-point multiplications and additions. Additionally, It can
/// be faster than Mixed-Radix at sizes below 10,000 or so.
///
/// ~~~
/// // Computes a forward FFT of size 1200, using the Good-Thomas Algorithm
/// use rustfft::algorithm::GoodThomasAlgorithm;
/// use rustfft::{FFT, FFTplanner};
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

    reduced_width: StrengthReducedUsize,
    reduced_width_plus_one: StrengthReducedUsize,

    inverse: bool,
}

impl<T: FFTnum> GoodThomasAlgorithm<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `width_fft.len() * height_fft.len()`
    ///
    /// GCD(width_fft.len(), height_fft.len()) must be equal to 1
    pub fn new(mut width_fft: Arc<dyn Fft<T>>, mut height_fft: Arc<dyn Fft<T>>) -> Self {
        assert_eq!(
            width_fft.is_inverse(), height_fft.is_inverse(),
            "width_fft and height_fft must both be inverse, or neither. got width inverse={}, height inverse={}",
            width_fft.is_inverse(), height_fft.is_inverse());

        let mut width = width_fft.len();
        let mut height = height_fft.len();
        let is_inverse = width_fft.is_inverse();

        // This algorithm doesn't work if width and height aren't coprime
        let gcd = num_integer::gcd(width as i64, height as i64);
        assert!(gcd == 1,
                "Invalid width and height for Good-Thomas Algorithm (width={}, height={}): Inputs must be coprime",
                width,
                height);

        // The trick we're using for our index remapping will only work if width < height, so just swap them if it isn't
        if width > height {
            std::mem::swap(&mut width, &mut height);
            std::mem::swap(&mut width_fft, &mut height_fft);
        }

        Self {
            width,
            width_size_fft: width_fft,

            height,
            height_size_fft: height_fft,

            reduced_width: StrengthReducedUsize::new(width),
            reduced_width_plus_one: StrengthReducedUsize::new(width + 1),

            inverse: is_inverse,
        }
    }

    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        // A critical part of the good-thomas algorithm is re-indexing the inputs and outputs.
        // To remap the inputs, we will use the CRT mapping, paired with the normal transpose we'd do for mixed radix.
        //
        // The algorithm for the CRT mapping will work like this:
        // 1: Keep an output index, initialized to 0
        // 2: The output index will be incremented by width + 1
        // 3: At the start of the row, compute if we will increment output_index past self.len()
        //      3a: If we will, then compute exactly how many increments it will take,
        //      3b: Increment however many times as we scan over the input row, copying each element to the output index
        //      3c: Subtract self.len() from output_index
        // 4: Scan over the rest of the row, incrementing output_index, and copying each element to output_index, thne incrementing output_index
        // 5: The first index of each row will be the final index of the previous row plus one, but because of our incrementing (width+1) inside the loop, we overshot, so at the end of the row, subtract width from output_index
        //
        // This ends up producing the same result as computing the multiplicative inverse of width mod height and etc by the CRT mapping, but with only one integer division per row, instead of one per element.
        let mut output_index = 0;
        for mut input_row in input.chunks_exact(self.width) {
            let increments_until_cycle =
                1 + (self.len() - output_index) / self.reduced_width_plus_one;

            // If we will have to rollover output_index on this row, do it in a separate loop
            if increments_until_cycle < self.width {
                let (pre_cycle_row, post_cycle_row) = input_row.split_at(increments_until_cycle);

                for input_element in pre_cycle_row {
                    output[output_index] = *input_element;
                    output_index += self.reduced_width_plus_one.get();
                }

                // Store the split slice back to input_row, os that outside the loop, we can finish the job of iterating the row
                input_row = post_cycle_row;
                output_index -= self.len();
            }

            // Loop over the entire row (if we did not roll over) or what's left of the row (if we did) and keep incrementing output_row
            for input_element in input_row {
                output[output_index] = *input_element;
                output_index += self.reduced_width_plus_one.get();
            }

            output_index -= self.width;
        }

        // run FFTs of size `width`
        self.width_size_fft.process_multi(output, input);

        // transpose
        transpose::transpose(input, output, self.width, self.height);

        // run FFTs of size 'height'
        self.height_size_fft.process_multi(output, input);

        // To remap the outputs, we will use the ruritanian mapping, paired with the normal transpose we'd do for mixed radix.
        //
        // The algorithm for the ruritanian mapping will work like this:
        // 1: At the start of every row, compute the output index = (y * self.height) % self.width
        // 2: We will increment this output index by self.width for every element
        // 3: Compute where in the row the output index will wrap around
        // 4: Instead of starting copying from the beginning of the row, start copying from after the rollover point
        // 5: When we hit the end of the row, continue from the beginning of the row, continuing to increment the output index by self.width
        //
        // This achieves the same result as the modular arithmetic ofthe ruritanian mapping, but with only one integer divison per row, instead of one per element
        for (y, input_chunk) in input.chunks_exact(self.height).enumerate() {
            let (quotient, remainder) =
                StrengthReducedUsize::div_rem(y * self.height, self.reduced_width);

            // Compute our base index and starting point in the row
            let mut output_index = remainder;
            let start_x = self.height - quotient;

            // Process the first part of the row
            for x in start_x..self.height {
                output[output_index] = input_chunk[x];
                output_index += self.width;
            }

            // Wrap back around to the beginning of the row and keep incrementing
            for x in 0..start_x {
                output[output_index] = input_chunk[x];
                output_index += self.width;
            }
        }
    }
}

impl<T: FFTnum> Fft<T> for GoodThomasAlgorithm<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

        self.perform_fft(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input
            .chunks_exact_mut(self.len())
            .zip(output.chunks_exact_mut(self.len()))
        {
            self.perform_fft(in_chunk, out_chunk);
        }
    }
}
impl<T> Length for GoodThomasAlgorithm<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.width * self.height
    }
}
impl<T> IsInverse for GoodThomasAlgorithm<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

/// Implementation of the Good-Thomas Algorithm, specialized for the case where both inner FFTs are butterflies
///
/// This algorithm factors a size n FFT into n1 * n2, where GCD(n1, n2) == 1
///
/// Conceptually, this algorithm is very similar to the Mixed-Radix FFT, except because GCD(n1, n2) == 1 we can do some
/// number theory trickery to reduce the number of floating-point multiplications and additions. It typically performs
/// better than Mixed-Radix Double Butterfly Algorithm, especially at small sizes.
///
/// ~~~
/// // Computes a forward FFT of size 56, using the Good-Thoma Butterfly Algorithm
/// use std::sync::Arc;
/// use rustfft::algorithm::GoodThomasAlgorithmDoubleButterfly;
/// use rustfft::algorithm::butterflies::{Butterfly7, Butterfly8};
/// use rustfft::FFT;
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
/// let fft = GoodThomasAlgorithmDoubleButterfly::new(inner_fft_n1, inner_fft_n2);
/// fft.process(&mut input, &mut output);
/// ~~~
pub struct GoodThomasAlgorithmDoubleButterfly<T> {
    width: usize,
    width_size_fft: Arc<dyn FFTButterfly<T>>,

    height: usize,
    height_size_fft: Arc<dyn FFTButterfly<T>>,

    input_output_map: Box<[usize]>,

    inverse: bool,
}

impl<T: FFTnum> GoodThomasAlgorithmDoubleButterfly<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `width_fft.len() * height_fft.len()`
    ///
    /// GCD(n1.len(), n2.len()) must be equal to 1
    pub fn new(width_fft: Arc<dyn FFTButterfly<T>>, height_fft: Arc<dyn FFTButterfly<T>>) -> Self {
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
        assert!(
            gcd == 1,
            "Invalid input n1 and height to Good-Thomas Algorithm: ({},{}): Inputs must be coprime",
            width,
            height
        );

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
        let output_iter = (0..len).map(|i| (i % height, i / height)).map(|(y, x)| {
            (x * height * height_inverse as usize + y * width * width_inverse as usize) % len
        });

        let input_output_map: Vec<usize> = input_iter.chain(output_iter).collect();

        Self {
            inverse: width_fft.is_inverse(),

            width,
            width_size_fft: width_fft,

            height,
            height_size_fft: height_fft,

            input_output_map: input_output_map.into_boxed_slice(),
        }
    }

    unsafe fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        let (input_map, output_map) = self.input_output_map.split_at(self.len());

        // copy the input using our reordering mapping
        for (output_element, &input_index) in output.iter_mut().zip(input_map.iter()) {
            *output_element = input[input_index];
        }

        // run FFTs of size `width`
        self.width_size_fft.process_multi_inplace(output);

        // transpose
        array_utils::transpose_small(self.width, self.height, output, input);

        // run FFTs of size 'height'
        self.height_size_fft.process_multi_inplace(input);

        // copy to the output, using our output redordeing mapping
        for (input_element, &output_index) in input.iter().zip(output_map.iter()) {
            output[output_index] = *input_element;
        }
    }
}

impl<T: FFTnum> Fft<T> for GoodThomasAlgorithmDoubleButterfly<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

        unsafe { self.perform_fft(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input
            .chunks_exact_mut(self.len())
            .zip(output.chunks_exact_mut(self.len()))
        {
            unsafe { self.perform_fft(in_chunk, out_chunk) };
        }
    }
}
impl<T> Length for GoodThomasAlgorithmDoubleButterfly<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.width * self.height
    }
}
impl<T> IsInverse for GoodThomasAlgorithmDoubleButterfly<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::algorithm::DFT;
    use crate::test_utils::{check_fft_algorithm, make_butterfly};
    use num_integer::gcd;
    use std::sync::Arc;

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
    fn test_good_thomas_double_butterfly() {
        let butterfly_sizes = [2, 3, 4, 5, 6, 7, 8, 16];
        for width in &butterfly_sizes {
            for height in &butterfly_sizes {
                if gcd(*width, *height) == 1 {
                    test_good_thomas_butterfly_with_lengths(*width, *height, false);
                    test_good_thomas_butterfly_with_lengths(*width, *height, true);
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

    fn test_good_thomas_butterfly_with_lengths(width: usize, height: usize, inverse: bool) {
        let width_fft = make_butterfly(width, inverse);
        let height_fft = make_butterfly(height, inverse);

        let fft = GoodThomasAlgorithmDoubleButterfly::new(width_fft, height_fft);

        check_fft_algorithm(&fft, width * height, inverse);
    }

    #[test]
    fn test_output_mapping() {
        let width = 15;
        for height in 3..width {
            if gcd(width, height) == 1 {
                let width_fft = Arc::new(DFT::new(width, false)) as Arc<dyn Fft<f32>>;
                let height_fft = Arc::new(DFT::new(height, false)) as Arc<dyn Fft<f32>>;

                let fft = GoodThomasAlgorithm::new(width_fft, height_fft);

                let mut input = vec![Complex { re: 0.0, im: 0.0 }; fft.len()];
                let mut output = vec![Complex { re: 0.0, im: 0.0 }; fft.len()];

                fft.process(&mut input, &mut output);
            }
        }
    }
}
