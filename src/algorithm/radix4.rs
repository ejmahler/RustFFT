use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use crate::algorithm::butterflies::{Butterfly1, Butterfly16, Butterfly2, Butterfly4, Butterfly8};
use crate::array_utils;
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::{
    array_utils::{RawSlice, RawSliceMut},
    common::FftNum,
    twiddles, FftDirection,
};
use crate::{Direction, Fft, Length};

/// FFT algorithm optimized for power-of-two sizes
///
/// ~~~
/// // Computes a forward FFT of size 4096
/// use rustfft::algorithm::Radix4;
/// use rustfft::{Fft, FftDirection};
/// use rustfft::num_complex::Complex;
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 4096];
///
/// let fft = Radix4::new(4096, FftDirection::Forward);
/// fft.process(&mut buffer);
/// ~~~

pub struct Radix4<T> {
    twiddles: Box<[Complex<T>]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
}

impl<T: FftNum> Radix4<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(len: usize, direction: FftDirection) -> Self {
        assert!(
            len.is_power_of_two(),
            "Radix4 algorithm requires a power-of-two input size. Got {}",
            len
        );

        // figure out which base length we're going to use
        let num_bits = len.trailing_zeros();
        let (base_len, base_fft) = match num_bits {
            0 => (len, Arc::new(Butterfly1::new(direction)) as Arc<dyn Fft<T>>),
            1 => (len, Arc::new(Butterfly2::new(direction)) as Arc<dyn Fft<T>>),
            2 => (len, Arc::new(Butterfly4::new(direction)) as Arc<dyn Fft<T>>),
            _ => {
                if num_bits % 2 == 1 {
                    (8, Arc::new(Butterfly8::new(direction)) as Arc<dyn Fft<T>>)
                } else {
                    (16, Arc::new(Butterfly16::new(direction)) as Arc<dyn Fft<T>>)
                }
            }
        };

        // precompute the twiddle factors this algorithm will use.
        // we're doing the same precomputation of twiddle factors as the mixed radix algorithm where width=4 and height=len/4
        // but mixed radix only does one step and then calls itself recusrively, and this algorithm does every layer all the way down
        // so we're going to pack all the "layers" of twiddle factors into a single array, starting with the bottom layer and going up
        let mut twiddle_stride = len / (base_len * 4);
        let mut twiddle_factors = Vec::with_capacity(len * 2);
        while twiddle_stride > 0 {
            let num_rows = len / (twiddle_stride * 4);
            for i in 0..num_rows {
                for k in 1..4 {
                    let twiddle = twiddles::compute_twiddle(i * k * twiddle_stride, len, direction);
                    twiddle_factors.push(twiddle);
                }
            }
            twiddle_stride /= 4;
        }

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

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
        let mut current_size = self.base_len * 4;
        let mut layer_twiddles: &[Complex<T>] = &self.twiddles;

        while current_size <= signal.len() {
            let num_rows = signal.len() / current_size;

            for i in 0..num_rows {
                unsafe {
                    butterfly_4(
                        &mut spectrum[i * current_size..],
                        layer_twiddles,
                        current_size / 4,
                        self.direction,
                    )
                }
            }

            //skip past all the twiddle factors used in this layer
            let twiddle_offset = (current_size * 3) / 4;
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            current_size *= 4;
        }
    }
}
boilerplate_fft_oop!(Radix4, |this: &Radix4<_>| this.len);

// Preparing for radix 4 is similar to a transpose, where the column index is bit reversed.
// Use a lookup table to avoid repeating the slow bit reverse operations.
// Unrolling the outer loop by a factor 4 helps speed things up.
pub fn bitreversed_transpose<T: Copy>(height: usize, input: &[T], output: &mut [T]) {
    let width = input.len() / height;
    let quarter_width = width / 4;

    let rev_digits = (width.trailing_zeros() / 2) as usize;

    // Let's make sure the arguments are ok
    assert!(input.len() == output.len());
    for x in 0..quarter_width {
        let x0 = 4 * x;
        let x1 = 4 * x + 1;
        let x2 = 4 * x + 2;
        let x3 = 4 * x + 3;

        let x_rev = [
            reverse_bits(x0, rev_digits),
            reverse_bits(x1, rev_digits),
            reverse_bits(x2, rev_digits),
            reverse_bits(x3, rev_digits),
        ];

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // The highest index the loop reaches is: (x_rev[n] + 1)*height - 1
        // The last element of the data is at index: width*height - 1
        // Thus it is sufficient to assert that x_rev[n]<width.
        assert!(x_rev[0] < width && x_rev[1] < width && x_rev[2] < width && x_rev[3] < width);

        for y in 0..height {
            let input_index0 = x0 + y * width;
            let input_index1 = x1 + y * width;
            let input_index2 = x2 + y * width;
            let input_index3 = x3 + y * width;
            let output_index0 = y + x_rev[0] * height;
            let output_index1 = y + x_rev[1] * height;
            let output_index2 = y + x_rev[2] * height;
            let output_index3 = y + x_rev[3] * height;

            unsafe {
                let temp0 = *input.get_unchecked(input_index0);
                let temp1 = *input.get_unchecked(input_index1);
                let temp2 = *input.get_unchecked(input_index2);
                let temp3 = *input.get_unchecked(input_index3);

                *output.get_unchecked_mut(output_index0) = temp0;
                *output.get_unchecked_mut(output_index1) = temp1;
                *output.get_unchecked_mut(output_index2) = temp2;
                *output.get_unchecked_mut(output_index3) = temp3;
            }
        }
    }
}

// Reverse bits of value, in pairs.
// For 8 bits: abcdefgh -> ghefcdab
pub fn reverse_bits(value: usize, bitpairs: usize) -> usize {
    let mut result: usize = 0;
    let mut value = value;
    for _ in 0..bitpairs {
        result = (result << 2) + (value & 0x03);
        value = value >> 2;
    }
    result
}

unsafe fn butterfly_4<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    num_ffts: usize,
    direction: FftDirection,
) {
    let butterfly4 = Butterfly4::new(direction);

    let mut idx = 0usize;
    let mut tw_idx = 0usize;
    let mut scratch = [Zero::zero(); 4];
    for _ in 0..num_ffts {
        scratch[0] = *data.get_unchecked(idx);
        scratch[1] = *data.get_unchecked(idx + 1 * num_ffts) * twiddles[tw_idx];
        scratch[2] = *data.get_unchecked(idx + 2 * num_ffts) * twiddles[tw_idx + 1];
        scratch[3] = *data.get_unchecked(idx + 3 * num_ffts) * twiddles[tw_idx + 2];

        butterfly4.perform_fft_contiguous(RawSlice::new(&scratch), RawSliceMut::new(&mut scratch));

        *data.get_unchecked_mut(idx) = scratch[0];
        *data.get_unchecked_mut(idx + 1 * num_ffts) = scratch[1];
        *data.get_unchecked_mut(idx + 2 * num_ffts) = scratch[2];
        *data.get_unchecked_mut(idx + 3 * num_ffts) = scratch[3];

        tw_idx += 3;
        idx += 1;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    #[test]
    fn test_radix4() {
        for pow in 1..12 {
            let len = 1 << pow;
            test_radix4_with_length(len, FftDirection::Forward);
            //test_radix4_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_radix4_with_length(len: usize, direction: FftDirection) {
        let fft = Radix4::new(len, direction);

        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
