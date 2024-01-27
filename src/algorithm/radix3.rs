use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use crate::algorithm::butterflies::{Butterfly1, Butterfly27, Butterfly3, Butterfly9};
use crate::array_utils::{self, bitreversed_transpose, compute_logarithm};
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::{common::FftNum, twiddles, FftDirection};
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
        let exponent = compute_logarithm::<3>(len).unwrap_or_else(|| {
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
        const ROW_COUNT: usize = 3;
        let mut cross_fft_len = base_len * ROW_COUNT;
        let mut twiddle_factors = Vec::with_capacity(len * 2);
        while cross_fft_len <= len {
            let num_columns = cross_fft_len / ROW_COUNT;

            for i in 0..num_columns {
                for k in 1..ROW_COUNT {
                    let twiddle = twiddles::compute_twiddle(i * k, cross_fft_len, direction);
                    twiddle_factors.push(twiddle);
                }
            }
            cross_fft_len *= ROW_COUNT;
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
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // copy the data into the output vector
        if self.len() == self.base_len {
            output.copy_from_slice(input);
        } else {
            bitreversed_transpose::<Complex<T>, 3>(self.base_len, input, output);
        }

        // Base-level FFTs
        self.base_fft.process_with_scratch(output, &mut []);

        // cross-FFTs
        const ROW_COUNT: usize = 3;
        let mut cross_fft_len = self.base_len * ROW_COUNT;
        let mut layer_twiddles: &[Complex<T>] = &self.twiddles;

        while cross_fft_len <= input.len() {
            let num_rows = input.len() / cross_fft_len;
            let num_columns = cross_fft_len / ROW_COUNT;

            for i in 0..num_rows {
                unsafe {
                    butterfly_3(
                        &mut output[i * cross_fft_len..],
                        layer_twiddles,
                        num_columns,
                        &self.butterfly3,
                    )
                }
            }

            // skip past all the twiddle factors used in this layer
            let twiddle_offset = num_columns * (ROW_COUNT - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            cross_fft_len *= ROW_COUNT;
        }
    }
}
boilerplate_fft_oop!(Radix3, |this: &Radix3<_>| this.len);

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

        butterfly3.perform_fft_butterfly(&mut scratch);

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
