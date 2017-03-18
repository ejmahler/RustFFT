use num::{Complex, Zero};
use common::{FFTnum, verify_length, verify_length_divisible};

use algorithm::butterflies::{Butterfly2, Butterfly4, FFTButterfly};
use twiddles;
use algorithm::{FFTAlgorithm, Length};

pub struct Radix4<T> {
    twiddles: Box<[Complex<T>]>,
    inverse: bool,
}

impl<T: FFTnum> Radix4<T> {
    pub fn new(len: usize, inverse: bool) -> Self {
        assert!(len.is_power_of_two() && len > 1, "Radix4 algorithm requires a power-of-two input size greater than one. Got {}", len);
        Radix4 {
            twiddles: twiddles::generate_twiddle_factors(len, inverse).into_boxed_slice(),
            inverse: inverse,
        }
    }

    fn perform_fft(&self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        // copy the data into the spectrum vector
        prepare_radix4(signal.len(), signal, spectrum, 1);

        // perform the butterflies. the butterfly size depends on the input size
        let num_bits = signal.len().trailing_zeros();
        let mut current_size = if num_bits % 2 > 0 {
            let inner_fft = Butterfly2{};
            unsafe { inner_fft.process_multi_inplace(spectrum) };

            // for the cross-ffts we want to to start off with a size of 8 (2 * 4)
            8
        } else {
            let inner_fft = Butterfly4::new(self.inverse);
            unsafe { inner_fft.process_multi_inplace(spectrum) };

            // for the cross-ffts we want to to start off with a size of 16 (4 * 4)
            16
        };

        // now, perform all the cross-FFTs, one "layer" at a time
        while current_size <= signal.len() {
            let group_stride = signal.len() / current_size;

            for i in 0..group_stride {
                unsafe {
                    butterfly_4(&mut spectrum[i * current_size..],
                                group_stride,
                                &self.twiddles,
                                current_size / 4,
                                self.inverse)
                }
            }
            current_size *= 4;
        }
    }
}

impl<T: FFTnum> FFTAlgorithm<T> for Radix4<T> {
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
impl<T> Length for Radix4<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len()
    }
}
// after testing an iterative bit reversal algorithm, this recursive algorithm
// was almost an order of magnitude faster at setting up
fn prepare_radix4<T: FFTnum>(size: usize,
                           signal: &[Complex<T>],
                           spectrum: &mut [Complex<T>],
                           stride: usize) {
    match size {
        4 => unsafe {
            for i in 0..4 {
                *spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        },
        2 => unsafe {
            for i in 0..2 {
                *spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        },
        _ => {
            for i in 0..4 {
                prepare_radix4(size / 4,
                               &signal[i * stride..],
                               &mut spectrum[i * (size / 4)..],
                               stride * 4);
            }
        }
    }
}

unsafe fn butterfly_4<T: FFTnum>(data: &mut [Complex<T>],
                             stride: usize,
                             twiddles: &[Complex<T>],
                             num_ffts: usize,
                             inverse: bool)
{
    let mut idx = 0usize;
    let mut tw_idx_1 = 0usize;
    let mut tw_idx_2 = 0usize;
    let mut tw_idx_3 = 0usize;
    let mut scratch: [Complex<T>; 6] = [Zero::zero(); 6];
    for _ in 0..num_ffts {
        scratch[0] = data.get_unchecked(idx + 1 * num_ffts) * twiddles[tw_idx_1];
        scratch[1] = data.get_unchecked(idx + 2 * num_ffts) * twiddles[tw_idx_2];
        scratch[2] = data.get_unchecked(idx + 3 * num_ffts) * twiddles[tw_idx_3];
        scratch[5] = data.get_unchecked(idx) - scratch[1];
        *data.get_unchecked_mut(idx) = data.get_unchecked(idx) + scratch[1];
        scratch[3] = scratch[0] + scratch[2];
        scratch[4] = scratch[0] - scratch[2];
        *data.get_unchecked_mut(idx + 2 * num_ffts) = data.get_unchecked(idx) - scratch[3];
        *data.get_unchecked_mut(idx) = data.get_unchecked(idx) + scratch[3];
        if inverse {
            data.get_unchecked_mut(idx + num_ffts).re = scratch[5].re - scratch[4].im;
            data.get_unchecked_mut(idx + num_ffts).im = scratch[5].im + scratch[4].re;
            data.get_unchecked_mut(idx + 3 * num_ffts).re = scratch[5].re + scratch[4].im;
            data.get_unchecked_mut(idx + 3 * num_ffts).im = scratch[5].im - scratch[4].re;
        } else {
            data.get_unchecked_mut(idx + num_ffts).re = scratch[5].re + scratch[4].im;
            data.get_unchecked_mut(idx + num_ffts).im = scratch[5].im - scratch[4].re;
            data.get_unchecked_mut(idx + 3 * num_ffts).re = scratch[5].re - scratch[4].im;
            data.get_unchecked_mut(idx + 3 * num_ffts).im = scratch[5].im + scratch[4].re;
        }

        tw_idx_1 += 1 * stride;
        tw_idx_2 += 2 * stride;
        tw_idx_3 += 3 * stride;
        idx += 1;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_fft_algorithm;

    #[test]
    fn test_radix4() {
        for pow in 1..7 {
            let len = 1 << pow;
            test_radix4_with_length(len, false);
            test_radix4_with_length(len, true);
        }
    }

    fn test_radix4_with_length(len: usize, inverse: bool) {
        let fft = Radix4::new(len, inverse);

        check_fft_algorithm(&fft, len, inverse);
    }
}
