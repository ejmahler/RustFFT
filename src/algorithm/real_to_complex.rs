use std::sync::Arc;

use num_complex::Complex;

use crate::array_utils::{into_complex_mut, zip3};
use crate::{twiddles, Fft, FftComplexToReal, FftDirection, FftNum, FftRealToComplex, Length};

/// Processes FFTs with real-only inputs. Restricted to even input sizes.
pub struct RealToComplexEven<T> {
    inner_fft: Arc<dyn Fft<T>>,
    twiddles: Box<[Complex<T>]>,

    len: usize,
    required_scratch: usize,
    direction: FftDirection,
}
impl<T: FftNum> RealToComplexEven<T> {
    /// Creates a FFT instance which will process forward FFTs with real-only inputs of size `inner_fft.len() * 2`
    #[allow(unused)]
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();
        let len = inner_fft_len * 2;
        let direction = inner_fft.fft_direction();

        // Compute our twiddle factors. We only need half as many twiddle factors as our FFT length,
        // and keep in mind that we're baking a multiply by half into the twiddles
        let twiddle_count = if inner_fft_len % 2 == 0 {
            inner_fft_len / 2
        } else {
            inner_fft_len / 2 + 1
        };
        let half = T::from_f32(0.5).unwrap();
        let twiddles: Box<[Complex<T>]> = (1..twiddle_count)
            .map(|i| twiddles::compute_twiddle(i, len, direction) * half)
            .collect();

        Self {
            required_scratch: 2 * inner_fft.get_outofplace_scratch_len(),

            inner_fft,
            twiddles: twiddles,

            len,
            direction,
        }
    }
}
impl<T: FftNum> FftRealToComplex<T> for RealToComplexEven<T> {
    #[inline(never)]
    fn process(&self, input: &mut [T], output: &mut [Complex<T>], scratch: &mut [T]) {
        if self.len() == 0 {
            return;
        }

        let half = T::from_f32(0.5).unwrap();

        // The simplest part of the process is computing the inner FFT. Just transmute the input and forward it to the FFT
        {
            let inner_fft_len = self.len() / 2;

            let input_complex = into_complex_mut(input);
            let chopped_output = &mut output[..inner_fft_len];
            let scratch_complex = into_complex_mut(scratch);

            self.inner_fft.process_outofplace_with_scratch(
                input_complex,
                chopped_output,
                scratch_complex,
            );
        }

        // Next step is to apply twiddle factors to our output array, in-place.
        // The process works by loading 2 elements from opposite ends of the array,
        // combining them, and writing them back where we found them.
        // To support reading/writing from opposite ends of the array simultaneously, split the output array in half
        let (mut output_left, mut output_right) = output.split_at_mut(output.len() / 2);

        // The first and last element don't require any twiddle factors, so skip that work
        match (output_left.first_mut(), output_right.last_mut()) {
            (Some(first_element), Some(last_element)) => {
                // The first and last elements are just a sum and difference of the first value's real and imaginary values
                let first_value = *first_element;
                *first_element = Complex {
                    re: first_value.re + first_value.im,
                    im: T::zero(),
                };
                *last_element = Complex {
                    re: first_value.re - first_value.im,
                    im: T::zero(),
                };

                // Chop the first and last element off of our slices so that the loop below doesn't have to deal with them
                output_left = &mut output_left[1..];
                let right_len = output_right.len();
                output_right = &mut output_right[..right_len - 1];
            }
            _ => {
                return;
            }
        }

        // Loop over the remaining elements and apply twiddle factors on them
        for (twiddle, out, out_rev) in zip3(
            self.twiddles.iter(),
            output_left.iter_mut(),
            output_right.iter_mut().rev(),
        ) {
            let sum = *out + *out_rev;
            let diff = *out - *out_rev;

            // let sumdiff_blended = Complex { re: sum.re, im: diff.im };
            // let diffsum_blended = Complex { re: diff.re, im: sum.im };
            // let diffsum_swapped = Complex { re: sum.im, im: diff.re };

            // let twiddled_diffsum_blended = diffsum_blended * twiddle.im;
            // let twiddled_diffsum_swapped = diffsum_swapped * twiddle.re;
            // let half_sumdiff = sumdiff_blended * half;

            // let twiddled_output = Complex {
            //     re: twiddled_diffsum_blended.re + twiddled_diffsum_swapped.re,
            //     im: twiddled_diffsum_blended.im - twiddled_diffsum_swapped.im,
            // };

            // #[allow(unused)]
            // let tmp_out = Complex {
            //     re: half_sumdiff.re + twiddled_output.re,
            //     im: half_sumdiff.im + twiddled_output.im,
            // };
            // #[allow(unused)]
            // let tmp_out_rev = Complex {
            //     re: half_sumdiff.re - twiddled_output.re,
            //     im: -half_sumdiff.im + twiddled_output.im,
            // };

            // Apply twiddle factors. Theoretically we'd have to load 2 separate twiddle factors here, one for the beginning
            // and one for the end. But the twiddle factor for the end is jsut the twiddle for the beginning, with the
            // real part negated. Since it's the same twiddle, we can factor out a ton of math ops and cut the number of
            // multiplications in half
            let twiddled_re_sum = sum * twiddle.re;
            let twiddled_im_sum = sum * twiddle.im;
            let twiddled_re_diff = diff * twiddle.re;
            let twiddled_im_diff = diff * twiddle.im;
            let half_sum_re = half * sum.re;
            let half_diff_im = half * diff.im;

            let output_twiddled_real = twiddled_re_sum.im + twiddled_im_diff.re;
            let output_twiddled_im = twiddled_im_sum.im - twiddled_re_diff.re;

            // We finally have all the data we need to write the transformed data back out where we found it
            *out = Complex {
                re: half_sum_re + output_twiddled_real,
                im: half_diff_im + output_twiddled_im,
            };

            *out_rev = Complex {
                re: half_sum_re - output_twiddled_real,
                im: output_twiddled_im - half_diff_im,
            };
        }

        // If the output len is odd, the loop above can't postprocess the centermost element, so handle that separately
        if self.direction == FftDirection::Forward && output.len() % 2 == 1 {
            if let Some(center_element) = output.get_mut(output.len() / 2) {
                center_element.im = -center_element.im;
            }
        }
    }

    fn get_scratch_len(&self) -> usize {
        self.required_scratch
    }
}
impl<T> Length for RealToComplexEven<T> {
    fn len(&self) -> usize {
        self.len
    }
}

/// Processes FFTs with real-only outputs. Restricted to even input sizes.
pub struct ComplexToRealEven<T> {
    inner_fft: Arc<dyn Fft<T>>,
    twiddles: Box<[Complex<T>]>,

    len: usize,
    required_scratch: usize,
    direction: FftDirection,
}
impl<T: FftNum> ComplexToRealEven<T> {
    /// Creates a FFT instance which will process FFTs of size `inner_fft.len() * 2`, with real-only outputs
    #[allow(unused)]
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();
        let len = inner_fft_len * 2;
        let direction = inner_fft.fft_direction();

        // Compute our twiddle factors. We only need half as many twiddle factors as our FFT length,
        // and keep in mind that we're baking a multiply by half into the twiddles
        let twiddle_count = inner_fft_len;
        let twiddles: Box<[Complex<T>]> = (0..twiddle_count)
            .map(|i| twiddles::compute_twiddle(i, len, direction))
            .collect();

        Self {
            required_scratch: 2 * inner_fft.get_outofplace_scratch_len(),

            inner_fft,
            twiddles: twiddles,

            len,
            direction,
        }
    }
}
impl<T: FftNum> FftComplexToReal<T> for ComplexToRealEven<T> {
    #[inline(never)]
    fn process(&self, input: &mut [Complex<T>], output: &mut [T], scratch: &mut [T]) {
        if self.len() == 0 {
            return;
        }

        let inner_fft_len = self.len() / 2;
        assert_eq!(input.len(), self.len() / 2 + 1);
        assert_eq!(output.len(), self.len());
        assert!(scratch.len() >= self.get_scratch_len());

        // We have to preprocess the input in-place before we send it to the FFT.
        // The first and centermost values have to be preprocessed separately from the rest, so do that now
        {
            let last_value = input[input.len() - 1];
            let first_value = input[0];
            let first_sum = first_value + last_value;
            let first_diff = first_value - last_value;

            input[0] = Complex {
                re: first_sum.re - first_sum.im,
                im: first_diff.re - first_diff.im,
            };
        }

        // now, in a loop, preprocess the rest of the elements 2 at a time
        let chopped_input = &mut input[1..inner_fft_len];
        let (input_left, input_right) = chopped_input.split_at_mut(chopped_input.len() / 2);

        for (twiddle, fft_input, fft_input_rev) in zip3(
            (&self.twiddles[1..]).iter(),
            input_left.iter_mut(),
            input_right.iter_mut().rev(),
        ) {
            let sum = *fft_input + *fft_input_rev;
            let diff = *fft_input - *fft_input_rev;

            // Apply twiddle factors. Theoretically we'd have to load 2 separate twiddle factors here, one for the beginning
            // and one for the end. But the twiddle factor for the end is jsut the twiddle for the beginning, with the
            // real part negated. Since it's the same twiddle, we can factor out a ton of math ops and cut the number of
            // multiplications in half
            let twiddled_re_sum = sum * twiddle.re;
            let twiddled_im_sum = sum * twiddle.im;
            let twiddled_re_diff = diff * twiddle.re;
            let twiddled_im_diff = diff * twiddle.im;

            let output_twiddled_real = twiddled_re_sum.im + twiddled_im_diff.re;
            let output_twiddled_im = twiddled_im_sum.im - twiddled_re_diff.re;

            // We finally have all the data we need to write our preprocessed data back where we got it from
            *fft_input = Complex {
                re: sum.re - output_twiddled_real,
                im: diff.im - output_twiddled_im,
            };

            *fft_input_rev = Complex {
                re: sum.re + output_twiddled_real,
                im: -output_twiddled_im - diff.im,
            }
        }

        // If the output len is odd, the loop above can't preprocess the centermost element, so handle that separately
        if input.len() % 2 == 1 {
            let center_element = input[input.len() / 2];
            let doubled = center_element + center_element;
            input[input.len() / 2] = match self.direction {
                FftDirection::Forward => doubled,
                FftDirection::Inverse => doubled.conj(),
            };
        }

        // The only theing left to do is compute the FFT.
        // The data is in `input`, and we want it in `output`, so do an out of place transform to get it there
        let complex_output = into_complex_mut(output);
        let complex_scratch = into_complex_mut(scratch);
        self.inner_fft.process_outofplace_with_scratch(
            &mut input[..inner_fft_len],
            complex_output,
            complex_scratch,
        );
    }

    fn get_scratch_len(&self) -> usize {
        self.required_scratch
    }
}
impl<T> Length for ComplexToRealEven<T> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::{
        algorithm::Dft,
        test_utils::{compare_vectors, random_real_signal, random_signal},
        FftDirection,
    };
    use num_traits::Zero;

    #[test]
    fn test_r2c_even_scalar() {
        for inner_len in 6..7 {
            test_r2c_even_with_inner(inner_len, FftDirection::Forward);

            // Note: Even though R2C is usually used with a forward FFT, it was pretty trivial to make it support inverse FFTs.
            // If there's a compelling performance reason to drop inverse support, go ahead.
            //test_r2c_even_with_inner(inner_len, FftDirection::Inverse);
        }
    }

    fn test_r2c_even_with_inner(inner_len: usize, direction: FftDirection) {
        let inner_fft: Arc<Dft<f64>> = Arc::new(Dft::new(inner_len, direction));
        let fft = RealToComplexEven::new(inner_fft);

        let control = Dft::new(inner_len * 2, direction);

        let mut real_input = random_real_signal(control.len());
        let mut complex_input: Vec<Complex<f64>> = real_input.iter().map(Complex::from).collect();

        control.process(&mut complex_input);

        let mut real_output = vec![Complex::zero(); inner_len + 1];
        let mut scratch = vec![0.0; fft.get_scratch_len()];
        fft.process(&mut real_input, &mut real_output, &mut scratch);

        if inner_len > 0 {
            assert!(
                compare_vectors(&complex_input[..inner_len + 1], &real_output),
                "process() failed, len = {}, direction = {}",
                inner_len * 2,
                direction,
            );
        }
    }

    #[test]
    fn test_c2r_even_scalar() {
        for inner_len in 6..7 {
            // Note: Even though C2R is usually used with an inverse FFT, it was pretty trivial to make it support forward FFTs.
            // If there's a compelling performance reason to drop forward support, go ahead.
            test_c2r_even_with_inner(inner_len, FftDirection::Forward);

            test_c2r_even_with_inner(inner_len, FftDirection::Inverse);
        }
    }

    fn test_c2r_even_with_inner(inner_len: usize, direction: FftDirection) {
        let inner_fft: Arc<Dft<f32>> = Arc::new(Dft::new(inner_len, direction));
        let fft = ComplexToRealEven::new(inner_fft);

        let control = Dft::new(inner_len * 2, direction);

        let mut real_input = random_signal(inner_len + 1);
        real_input[0].im = 0.0;
        real_input.last_mut().unwrap().im = 0.0;
        let mut complex_input = real_input.clone();

        for i in (1..complex_input.len() - 1).rev() {
            complex_input.push(complex_input[i].conj());
        }

        control.process(&mut complex_input);

        let mut real_output = vec![0.0; inner_len * 2];
        let mut scratch = vec![0.0; fft.get_scratch_len()];
        fft.process(&mut real_input, &mut real_output, &mut scratch);

        let real_output: Vec<_> = real_output.iter().map(Complex::from).collect();
        if inner_len > 0 {
            assert!(compare_vectors(&complex_input, &real_output));
        }
    }
}
