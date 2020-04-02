use std::sync::Arc;
use std::cmp::max;

use num_complex::Complex;
use transpose;

use common::FFTnum;

use ::{Length, IsInverse, Fft};
use algorithm::butterflies::FFTButterfly;
use array_utils;

/// Implementation of the Mixed-Radix FFT algorithm
///
/// This algorithm factors a size n FFT into n1 * n2, computes several inner FFTs of size n1 and n2, then combines the 
/// results to get the final answer
///
/// ~~~
/// // Computes a forward FFT of size 1200, using the Mixed-Radix Algorithm
/// use rustfft::algorithm::MixedRadix;
/// use rustfft::{Fft, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1200];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1200];
///
/// // we need to find an n1 and n2 such that n1 * n2 == 1200
/// // n1 = 30 and n2 = 40 satisfies this
/// let mut planner = FFTplanner::new(false);
/// let inner_fft_n1 = planner.plan_fft(30);
/// let inner_fft_n2 = planner.plan_fft(40);
///
/// // the mixed radix FFT length will be inner_fft_n1.len() * inner_fft_n2.len() = 1200
/// let fft = MixedRadix::new(inner_fft_n1, inner_fft_n2);
/// fft.process(&mut input, &mut output);
/// ~~~

pub struct MixedRadix<T> {
    twiddles: Box<[Complex<T>]>,

    width_size_fft: Arc<Fft<T>>,
    width: usize,

    height_size_fft: Arc<Fft<T>>,
    height: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,

    inverse: bool,
}

impl<T: FFTnum> MixedRadix<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `width_fft.len() * height_fft.len()`
    pub fn new(width_fft: Arc<Fft<T>>, height_fft: Arc<Fft<T>>) -> Self {
        assert_eq!(
            width_fft.is_inverse(), height_fft.is_inverse(), 
            "width_fft and height_fft must both be inverse, or neither. got width inverse={}, height inverse={}",
            width_fft.is_inverse(), height_fft.is_inverse());

        let inverse = width_fft.is_inverse();

        let width = width_fft.len();
        let height = height_fft.len();

        let len = width * height;

        let mut twiddles = Vec::with_capacity(len);
        for x in 0..width {
            for y in 0..height {
                twiddles.push(T::generate_twiddle_factor(x * y, len, inverse));
            }
        }

        let height_inplace_scratch = height_fft.get_inplace_scratch_len();
        let width_inplace_scratch = width_fft.get_inplace_scratch_len();
        let width_outofplace_scratch = width_fft.get_out_of_place_scratch_len();

        let outofplace_scratch = max(height_inplace_scratch, width_inplace_scratch);
        let inplace_extra = max(if height_inplace_scratch > len { height_inplace_scratch } else { 0 }, width_outofplace_scratch);

        Self {
            twiddles: twiddles.into_boxed_slice(),

            width_size_fft: width_fft,
            width: width,

            height_size_fft: height_fft,
            height: height,

            inplace_scratch_len: len + inplace_extra,
            outofplace_scratch_len: if outofplace_scratch > len { outofplace_scratch } else { 0 },

            inverse: inverse,
        }
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // SIX STEP FFT:
        let (scratch, inner_scratch) = scratch.split_at_mut(self.len());

        // STEP 1: transpose
        transpose::transpose(buffer, scratch, self.width, self.height);

        // STEP 2: perform FFTs of size `height`
        let height_scratch = if inner_scratch.len() > buffer.len() { &mut inner_scratch[..] } else { &mut buffer[..] };
        self.height_size_fft.process_inplace_multi(scratch, height_scratch);

        // STEP 3: Apply twiddle factors
        for (element, twiddle) in scratch.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        transpose::transpose(scratch, buffer, self.height, self.width);

        // STEP 5: perform FFTs of size `width`
        self.width_size_fft.process_multi(buffer, scratch, inner_scratch);

        // STEP 6: transpose again
        transpose::transpose(scratch, buffer, self.width, self.height);
    }
    
    fn perform_fft_out_of_place(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // SIX STEP FFT:

        // STEP 1: transpose
        transpose::transpose(input, output, self.width, self.height);

        // STEP 2: perform FFTs of size `height`
        let height_scratch = if scratch.len() > input.len() { &mut scratch[..] } else { &mut input[..] };
        self.height_size_fft.process_inplace_multi(output, height_scratch);

        // STEP 3: Apply twiddle factors
        for (element, twiddle) in output.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        transpose::transpose(output, input, self.height, self.width);

        // STEP 5: perform FFTs of size `width`
        let width_scratch = if scratch.len() > output.len() { &mut scratch[..] } else { &mut output[..] };
        self.width_size_fft.process_inplace_multi(input, width_scratch);

        // STEP 6: transpose again
        transpose::transpose(input, output, self.width, self.height);
    }
}
boilerplate_fft!(MixedRadix,
    |this: &MixedRadix<_>| this.twiddles.len(),
    |this: &MixedRadix<_>| this.inplace_scratch_len,
    |this: &MixedRadix<_>| this.outofplace_scratch_len
);

/// Implementation of the Mixed-Radix FFT algorithm, specialized for the case where both inner FFTs are butterflies
///
/// This algorithm factors a size n FFT into n1 * n2
///
/// ~~~
/// // Computes a forward FFT of size 56, using the Mixed-Radix Butterfly Algorithm
/// use std::sync::Arc;
/// use rustfft::algorithm::MixedRadixDoubleButterfly;
/// use rustfft::algorithm::butterflies::{Butterfly7, Butterfly8};
/// use rustfft::Fft;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 56];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 56];
///
/// // we need to find an n1 and n2 such that n1 * n2 == 56
/// // n1 = 7 and n2 = 8 satisfies this
/// let inner_fft_n1 = Arc::new(Butterfly7::new(false));
/// let inner_fft_n2 = Arc::new(Butterfly8::new(false));
///
/// // the mixed radix FFT length will be inner_fft_n1.len() * inner_fft_n2.len() = 56
/// let fft = MixedRadixDoubleButterfly::new(inner_fft_n1, inner_fft_n2);
/// fft.process(&mut input, &mut output);
/// ~~~
pub struct MixedRadixDoubleButterfly<T> {
    width: usize,
    width_size_fft: Arc<FFTButterfly<T>>,

    height: usize,
    height_size_fft: Arc<FFTButterfly<T>>,

    twiddles: Box<[Complex<T>]>,
    inverse: bool,
}

impl<T: FFTnum> MixedRadixDoubleButterfly<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `width_fft.len() * height_fft.len()`
    pub fn new(width_fft: Arc<FFTButterfly<T>>, height_fft: Arc<FFTButterfly<T>>) -> Self {
        assert_eq!(
            width_fft.is_inverse(), height_fft.is_inverse(), 
            "width_fft and height_fft must both be inverse, or neither. got width inverse={}, height inverse={}",
            width_fft.is_inverse(), height_fft.is_inverse());

        let inverse = width_fft.is_inverse();

        let width = width_fft.len();
        let height = height_fft.len();

        let len = width * height;

        let mut twiddles = Vec::with_capacity(len);
        for x in 0..width {
            for y in 0..height {
                twiddles.push(T::generate_twiddle_factor(x * y, len, inverse));
            }
        }

        Self {
            width: width,
            width_size_fft: width_fft,

            height: height,
            height_size_fft: height_fft,

            twiddles: twiddles.into_boxed_slice(),
            inverse: inverse
        }
    }


    fn perform_fft_out_of_place(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        // we're relying on the optimizer to get rid of this assert
        assert_eq!(self.len(), input.len());
        assert_eq!(self.len(), output.len());

        // SIX STEP FFT:

        // STEP 1: transpose
        unsafe { array_utils::transpose_small(self.width, self.height, input, output) };

        // STEP 2: perform FFTs of size 'height'
        unsafe { self.height_size_fft.process_butterfly_multi_inplace(output) };

        // STEP 3: Apply twiddle factors
        for (element, &twiddle) in output.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        unsafe { array_utils::transpose_small(self.height, self.width, output, input) };

        // STEP 5: perform FFTs of size 'width'
        unsafe { self.width_size_fft.process_butterfly_multi_inplace(input) };
        
        // STEP 6: transpose again
        unsafe { array_utils::transpose_small(self.width, self.height, input, output) };
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // we're relying on the optimizer to get rid of this assert
        assert_eq!(self.len(), buffer.len());
        assert_eq!(self.len(), scratch.len());

        // SIX STEP FFT:

        // STEP 1: transpose
        unsafe { array_utils::transpose_small(self.width, self.height, buffer, scratch) };

        // STEP 2: perform FFTs of size 'height'
        unsafe { self.height_size_fft.process_butterfly_multi_inplace(scratch) };

        // STEP 3: Apply twiddle factors
        for (element, &twiddle) in scratch.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * twiddle;
        }

        // STEP 4: transpose again
        unsafe { array_utils::transpose_small(self.height, self.width, scratch, buffer) };

        // STEP 5: perform FFTs of size 'width'
        unsafe { self.width_size_fft.process_butterfly_multi_inplace(buffer) };
        
        // STEP 6: transpose again
        unsafe { array_utils::transpose_small(self.width, self.height, buffer, scratch) };
        buffer.copy_from_slice(scratch);
    }
}
boilerplate_fft!(MixedRadixDoubleButterfly, 
    |this: &MixedRadixDoubleButterfly<_>| this.width * this.height,
    |this: &MixedRadixDoubleButterfly<_>| this.len(),
    |_| 0
);



#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use test_utils::{check_fft_algorithm, make_butterfly};
    use algorithm::DFT;

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
        let width_fft = Arc::new(DFT::new(width, inverse)) as Arc<Fft<f32>>;
        let height_fft = Arc::new(DFT::new(height, inverse)) as Arc<Fft<f32>>;

        let fft = MixedRadix::new(width_fft, height_fft);

        check_fft_algorithm(&fft, width * height, inverse);
    }

    fn test_mixed_radix_butterfly_with_lengths(width: usize, height: usize, inverse: bool) {
        let width_fft = make_butterfly(width, inverse);
        let height_fft = make_butterfly(height, inverse);

        let fft = MixedRadixDoubleButterfly::new(width_fft, height_fft);

        check_fft_algorithm(&fft, width * height, inverse);
    }
}
