use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use crate::algorithm::butterflies::{Butterfly1, Butterfly27, Butterfly3, Butterfly9};
use crate::array_utils;
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::{
    array_utils::{RawSlice, RawSliceMut},
    common::FftNum,
    twiddles, FftDirection,
};
use crate::{Direction, Fft, Length};

/// FFT algorithm optimized for power-of-three sizes
///
/// ~~~
/// // Computes a forward FFT of size 2187
/// use rustfft::algorithm::Radix3;
/// use rustfft::{Fft, FftDirection};
/// use rustfft::num_complex::Complex;
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 2187];
///
/// let fft = Radix3::new(2187, FftDirection::Forward);
/// fft.process(&mut buffer);
/// ~~~

pub struct Radix3<T> {
    twiddles: Box<[Complex<T>]>,
    butterfly3: Butterfly3<T>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
}

impl<T: FftNum> Radix3<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-three FFT
    pub fn new(len: usize, direction: FftDirection) -> Self {
        // Compute the total power of 3 for this length. IE, len = 3^exponent
        let exponent = compute_logarithm(len, 3).unwrap_or_else(|| {
            panic!(
                "Radix3 algorithm requires a power-of-three input size. Got {}",
                len
            )
        });

        // figure out which base length we're going to use
        let (base_len, base_fft) = match exponent {
            0 => (len, Arc::new(Butterfly1::new(direction)) as Arc<dyn Fft<T>>),
            1 => (len, Arc::new(Butterfly3::new(direction)) as Arc<dyn Fft<T>>),
            2 => (len, Arc::new(Butterfly9::new(direction)) as Arc<dyn Fft<T>>),
            _ => (27, Arc::new(Butterfly27::new(direction)) as Arc<dyn Fft<T>>),
        };

        // precompute the twiddle factors this algorithm will use.
        // we're doing the same precomputation of twiddle factors as the mixed radix algorithm where width=3 and height=len/3
        // but mixed radix only does one step and then calls itself recusrively, and this algorithm does every layer all the way down
        // so we're going to pack all the "layers" of twiddle factors into a single array, starting with the bottom layer and going up
        let mut twiddle_stride = len / (base_len * 3);
        let mut twiddle_factors = Vec::with_capacity(len * 2);
        while twiddle_stride > 0 {
            let num_rows = len / (twiddle_stride * 3);
            for i in 0..num_rows {
                for k in 1..3 {
                    let twiddle = twiddles::compute_twiddle(i * k * twiddle_stride, len, direction);
                    twiddle_factors.push(twiddle);
                }
            }
            twiddle_stride /= 3;
        }

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),
            butterfly3: Butterfly3::new(direction),

            base_fft,
            base_len,

            len,
            direction,
        }
    }

    fn perform_fft_out_of_place(
        &self,
        signal: &[Complex<T>],
        spectrum: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // copy the data into the spectrum vector
        if self.len() == self.base_len {
            spectrum.copy_from_slice(signal);
        } else {
            bitreversed_transpose(self.base_len, signal, spectrum);
        }

        // Base-level FFTs
        self.base_fft.process_with_scratch(spectrum, &mut []);

        // cross-FFTs
        let mut current_size = self.base_len * 3;
        let mut layer_twiddles: &[Complex<T>] = &self.twiddles;

        while current_size <= signal.len() {
            let num_rows = signal.len() / current_size;

            for i in 0..num_rows {
                unsafe {
                    butterfly_3(
                        &mut spectrum[i * current_size..],
                        layer_twiddles,
                        current_size / 3,
                        &self.butterfly3,
                    )
                }
            }

            //skip past all the twiddle factors used in this layer
            let twiddle_offset = (current_size * 2) / 3;
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            current_size *= 3;
        }
    }
}
boilerplate_fft_oop!(Radix3, |this: &Radix3<_>| this.len);

// Preparing for radix 3 is similar to a transpose, where the column index is bit reversed.
// Use a lookup table to avoid repeating the slow bit reverse operations.
// Unrolling the outer loop by a factor 4 helps speed things up.
pub fn bitreversed_transpose<T: Copy>(height: usize, input: &[T], output: &mut [T]) {
    let width = input.len() / height;
    let third_width = width / 3;

    let rev_digits = compute_logarithm(width, 3).unwrap();

    // Let's make sure the arguments are ok
    assert!(input.len() == output.len());
    for x in 0..third_width {
        let x0 = 3 * x;
        let x1 = 3 * x + 1;
        let x2 = 3 * x + 2;

        let x_rev = [
            reverse_bits(x0, rev_digits),
            reverse_bits(x1, rev_digits),
            reverse_bits(x2, rev_digits),
        ];

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // The highest index the loop reaches is: (x_rev[n] + 1)*height - 1
        // The last element of the data is at index: width*height - 1
        // Thus it is sufficient to assert that x_rev[n]<width.
        assert!(x_rev[0] < width && x_rev[1] < width && x_rev[2] < width);

        for y in 0..height {
            let input_index0 = x0 + y * width;
            let input_index1 = x1 + y * width;
            let input_index2 = x2 + y * width;
            let output_index0 = y + x_rev[0] * height;
            let output_index1 = y + x_rev[1] * height;
            let output_index2 = y + x_rev[2] * height;

            unsafe {
                let temp0 = *input.get_unchecked(input_index0);
                let temp1 = *input.get_unchecked(input_index1);
                let temp2 = *input.get_unchecked(input_index2);

                *output.get_unchecked_mut(output_index0) = temp0;
                *output.get_unchecked_mut(output_index1) = temp1;
                *output.get_unchecked_mut(output_index2) = temp2;
            }
        }
    }
}

// computes `n` such that `base ^ n == value`. Returns `None` if `value` is not a perfect power of `base`, otherwise returns `Some(n)`
fn compute_logarithm(value: usize, base: usize) -> Option<usize> {
    if value == 0 || base == 0 {
        return None;
    }

    let mut current_exponent = 0;
    let mut current_value = value;

    while current_value % base == 0 {
        current_exponent += 1;
        current_value /= base;
    }

    if current_value == 1 {
        Some(current_exponent)
    } else {
        None
    }
}

// Sort of like reversing bits in radix4. We're not actually reversing bits, but the algorithm is exactly the same.
// Radix4's bit reversal does divisions by 4, multiplications by 4, and modulo 4 - all of which are easily represented by bit manipulation.
// As a result, it can be thought of as a bit reversal. But really, the "bit reversal"-ness of it is a special case of a more general "remainder reversal"
// IE, it's repeatedly taking the remainder of dividing by N, and building a new number where those remainders are reversed.
// So this algorithm does all the things that bit reversal does, but replaces the multiplications by 4 with multiplications by 3, etc, and ends up with the same conceptual result as a bit reversal.
pub fn reverse_bits(value: usize, reversal_iters: usize) -> usize {
    let mut result: usize = 0;
    let mut value = value;
    for _ in 0..reversal_iters {
        result = (result * 3) + (value % 3);
        value /= 3;
    }
    result
}

unsafe fn butterfly_3<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    num_ffts: usize,
    butterfly3: &Butterfly3<T>,
) {
    let mut idx = 0usize;
    let mut tw_idx = 0usize;
    let mut scratch = [Zero::zero(); 3];
    for _ in 0..num_ffts {
        scratch[0] = *data.get_unchecked(idx);
        scratch[1] = *data.get_unchecked(idx + 1 * num_ffts) * twiddles[tw_idx];
        scratch[2] = *data.get_unchecked(idx + 2 * num_ffts) * twiddles[tw_idx + 1];

        butterfly3.perform_fft_contiguous(RawSlice::new(&scratch), RawSliceMut::new(&mut scratch));

        *data.get_unchecked_mut(idx) = scratch[0];
        *data.get_unchecked_mut(idx + 1 * num_ffts) = scratch[1];
        *data.get_unchecked_mut(idx + 2 * num_ffts) = scratch[2];

        tw_idx += 2;
        idx += 1;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    #[test]
    fn test_radix3() {
        for pow in 0..8 {
            let len = 3usize.pow(pow);
            test_3adix3_with_length(len, FftDirection::Forward);
            test_3adix3_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_3adix3_with_length(len: usize, direction: FftDirection) {
        let fft = Radix3::new(len, direction);

        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
