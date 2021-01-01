use std::cmp::max;
use std::sync::Arc;

use num_complex::Complex;
use transpose;

use crate::{FftDirection, common::FftNum, twiddles};

use crate::array_utils;
use crate::{Direction, Fft, Length};

/// Implementation of the Mixed-Radix FFT algorithm
///
/// This algorithm factors a size n FFT into n1 * n2, computes several inner FFTs of size n1 and n2, then combines the
/// results to get the final answer
///
/// ~~~
/// // Computes a forward FFT of size 1200, using the Mixed-Radix Algorithm
/// use rustfft::algorithm::MixedRadix;
/// use rustfft::{Fft, FftPlanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1200];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1200];
///
/// // we need to find an n1 and n2 such that n1 * n2 == 1200
/// // n1 = 30 and n2 = 40 satisfies this
/// let mut planner = FftPlanner::new();
/// let inner_fft_n1 = planner.plan_fft_forward(30);
/// let inner_fft_n2 = planner.plan_fft_forward(40);
///
/// // the mixed radix FFT length will be inner_fft_n1.len() * inner_fft_n2.len() = 1200
/// let fft = MixedRadix::new(inner_fft_n1, inner_fft_n2);
/// fft.process(&mut input, &mut output);
/// ~~~
pub struct MixedRadix<T> {
    twiddles: Box<[Complex<T>]>,

    width_size_fft: Arc<dyn Fft<T>>,
    width: usize,

    height_size_fft: Arc<dyn Fft<T>>,
    height: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,

    direction: FftDirection,
}

impl<T: FftNum> MixedRadix<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `width_fft.len() * height_fft.len()`
    pub fn new(width_fft: Arc<dyn Fft<T>>, height_fft: Arc<dyn Fft<T>>) -> Self {
        assert_eq!(
            width_fft.fft_direction(), height_fft.fft_direction(),
            "width_fft and height_fft must have the same direction. got width direction={}, height direction={}",
            width_fft.fft_direction(), height_fft.fft_direction());

        let direction = width_fft.fft_direction();

        let width = width_fft.len();
        let height = height_fft.len();

        let len = width * height;

        let mut twiddles = Vec::with_capacity(len);
        for x in 0..width {
            for y in 0..height {
                twiddles.push(twiddles::compute_twiddle(x * y, len, direction));
            }
        }

        let height_inplace_scratch = height_fft.get_inplace_scratch_len();
        let width_inplace_scratch = width_fft.get_inplace_scratch_len();
        let width_outofplace_scratch = width_fft.get_out_of_place_scratch_len();

        let outofplace_scratch = max(height_inplace_scratch, width_inplace_scratch);
        let inplace_extra = max(
            if height_inplace_scratch > len {
                height_inplace_scratch
            } else {
                0
            },
            width_outofplace_scratch,
        );

        Self {
            twiddles: twiddles.into_boxed_slice(),

            width_size_fft: width_fft,
            width: width,

            height_size_fft: height_fft,
            height: height,

            inplace_scratch_len: len + inplace_extra,
            outofplace_scratch_len: if outofplace_scratch > len {
                outofplace_scratch
            } else {
                0
            },

            direction,
        }
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // SIX STEP FFT:
        let (scratch, inner_scratch) = scratch.split_at_mut(self.len());

        // STEP 1: transpose
        transpose::transpose(buffer, scratch, self.width, self.height);

        // STEP 2: perform FFTs of size `height`
        let height_scratch = if inner_scratch.len() > buffer.len() {
            &mut inner_scratch[..]
        } else {
            &mut buffer[..]
        };
        self.height_size_fft
            .process_inplace_multi(scratch, height_scratch);

        // STEP 3: Apply twiddle factors
        for (element, twiddle) in scratch.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        transpose::transpose(scratch, buffer, self.height, self.width);

        // STEP 5: perform FFTs of size `width`
        self.width_size_fft
            .process_multi(buffer, scratch, inner_scratch);

        // STEP 6: transpose again
        transpose::transpose(scratch, buffer, self.width, self.height);
    }

    fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        // SIX STEP FFT:

        // STEP 1: transpose
        transpose::transpose(input, output, self.width, self.height);

        // STEP 2: perform FFTs of size `height`
        let height_scratch = if scratch.len() > input.len() {
            &mut scratch[..]
        } else {
            &mut input[..]
        };
        self.height_size_fft
            .process_inplace_multi(output, height_scratch);

        // STEP 3: Apply twiddle factors
        for (element, twiddle) in output.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        transpose::transpose(output, input, self.height, self.width);

        // STEP 5: perform FFTs of size `width`
        let width_scratch = if scratch.len() > output.len() {
            &mut scratch[..]
        } else {
            &mut output[..]
        };
        self.width_size_fft
            .process_inplace_multi(input, width_scratch);

        // STEP 6: transpose again
        transpose::transpose(input, output, self.width, self.height);
    }
}
boilerplate_fft!(
    MixedRadix,
    |this: &MixedRadix<_>| this.twiddles.len(),
    |this: &MixedRadix<_>| this.inplace_scratch_len,
    |this: &MixedRadix<_>| this.outofplace_scratch_len
);

/// Implementation of the Mixed-Radix FFT algorithm, specialized for smaller input sizes
///
/// This algorithm factors a size n FFT into n1 * n2, computes several inner FFTs of size n1 and n2, then combines the
/// results to get the final answer
///
/// ~~~
/// // Computes a forward FFT of size 40, using the Mixed-Radix Algorithm
/// use std::sync::Arc;
/// use rustfft::algorithm::MixedRadixSmall;
/// use rustfft::algorithm::butterflies::{Butterfly5, Butterfly8};
/// use rustfft::{Fft, FftDirection};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let len = 40;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); len];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); len];
///
/// // we need to find an n1 and n2 such that n1 * n2 == 40
/// // n1 = 5 and n2 = 8 satisfies this
/// let inner_fft_n1 = Arc::new(Butterfly5::new(FftDirection::Forward));
/// let inner_fft_n2 = Arc::new(Butterfly8::new(FftDirection::Forward));
///
/// // the mixed radix FFT length will be inner_fft_n1.len() * inner_fft_n2.len() = 40
/// let fft = MixedRadixSmall::new(inner_fft_n1, inner_fft_n2);
/// fft.process(&mut input, &mut output);
/// ~~~
pub struct MixedRadixSmall<T> {
    twiddles: Box<[Complex<T>]>,

    width_size_fft: Arc<dyn Fft<T>>,
    width: usize,

    height_size_fft: Arc<dyn Fft<T>>,
    height: usize,

    direction: FftDirection,
}

impl<T: FftNum> MixedRadixSmall<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `width_fft.len() * height_fft.len()`
    pub fn new(width_fft: Arc<dyn Fft<T>>, height_fft: Arc<dyn Fft<T>>) -> Self {
        assert_eq!(
            width_fft.fft_direction(), height_fft.fft_direction(),
            "width_fft and height_fft must have the same direction. got width direction={}, height direction={}",
            width_fft.fft_direction(), height_fft.fft_direction());

        let direction = width_fft.fft_direction();

        let width = width_fft.len();
        let height = height_fft.len();

        let len = width * height;

        let mut twiddles = Vec::with_capacity(len);
        for x in 0..width {
            for y in 0..height {
                twiddles.push(twiddles::compute_twiddle(x * y, len, direction));
            }
        }

        Self {
            twiddles: twiddles.into_boxed_slice(),

            width_size_fft: width_fft,
            width: width,

            height_size_fft: height_fft,
            height: height,

            direction,
        }
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // SIX STEP FFT:
        // STEP 1: transpose
        unsafe { array_utils::transpose_small(self.width, self.height, buffer, scratch) };

        // STEP 2: perform FFTs of size `height`
        self.height_size_fft.process_inplace_multi(scratch, buffer);

        // STEP 3: Apply twiddle factors
        for (element, twiddle) in scratch.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        unsafe { array_utils::transpose_small(self.height, self.width, scratch, buffer) };

        // STEP 5: perform FFTs of size `width`
        self.width_size_fft.process_multi(buffer, scratch, &mut []);

        // STEP 6: transpose again
        unsafe { array_utils::transpose_small(self.width, self.height, scratch, buffer) };
    }

    fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // SIX STEP FFT:
        // STEP 1: transpose
        unsafe { array_utils::transpose_small(self.width, self.height, input, output) };

        // STEP 2: perform FFTs of size `height`
        self.height_size_fft.process_inplace_multi(output, input);

        // STEP 3: Apply twiddle factors
        for (element, twiddle) in output.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        unsafe { array_utils::transpose_small(self.height, self.width, output, input) };

        // STEP 5: perform FFTs of size `width`
        self.width_size_fft.process_inplace_multi(input, output);

        // STEP 6: transpose again
        unsafe { array_utils::transpose_small(self.width, self.height, input, output) };
    }
}
boilerplate_fft!(
    MixedRadixSmall,
    |this: &MixedRadixSmall<_>| this.twiddles.len(),
    |this: &MixedRadixSmall<_>| this.len(),
    |_| 0
);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::algorithm::DFT;
    use crate::test_utils::check_fft_algorithm;
    use std::sync::Arc;

    #[test]
    fn test_mixed_radix() {
        for width in 1..7 {
            for height in 1..7 {
                test_mixed_radix_with_lengths(width, height, FftDirection::Forward);
                test_mixed_radix_with_lengths(width, height, FftDirection::Inverse);
            }
        }
    }

    #[test]
    fn test_mixed_radix_small() {
        for width in 2..7 {
            for height in 2..7 {
                test_mixed_radix_small_with_lengths(width, height, FftDirection::Forward);
                test_mixed_radix_small_with_lengths(width, height, FftDirection::Inverse);
            }
        }
    }

    fn test_mixed_radix_with_lengths(width: usize, height: usize, direction: FftDirection) {
        let width_fft = Arc::new(DFT::new(width, direction)) as Arc<dyn Fft<f32>>;
        let height_fft = Arc::new(DFT::new(height, direction)) as Arc<dyn Fft<f32>>;

        let fft = MixedRadix::new(width_fft, height_fft);

        check_fft_algorithm(&fft, width * height, direction);
    }

    fn test_mixed_radix_small_with_lengths(width: usize, height: usize, direction: FftDirection) {
        let width_fft = Arc::new(DFT::new(width, direction)) as Arc<dyn Fft<f32>>;
        let height_fft = Arc::new(DFT::new(height, direction)) as Arc<dyn Fft<f32>>;

        let fft = MixedRadixSmall::new(width_fft, height_fft);

        check_fft_algorithm(&fft, width * height, direction);
    }
}
