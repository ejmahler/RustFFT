use std::f32;

use num::{Complex, FromPrimitive, Signed};

use butterflies::{butterfly_2_single, butterfly_4_single, butterfly_4};
use super::FFTAlgorithm;

pub struct Radix4<T> {
    twiddles: Vec<Complex<T>>,
    inverse: bool,
}

impl<T> Radix4<T>
    where T: Signed + FromPrimitive + Copy
{
    pub fn new(len: usize, inverse: bool) -> Self {
        let dir = if inverse {
            1
        } else {
            -1
        };

        Radix4 {
            twiddles: (0..len)
                .map(|i| dir as f32 * i as f32 * 2.0 * f32::consts::PI / len as f32)
                .map(|phase| Complex::from_polar(&1.0, &phase))
                .map(|c| {
                    Complex {
                        re: FromPrimitive::from_f32(c.re).unwrap(),
                        im: FromPrimitive::from_f32(c.im).unwrap(),
                    }
                })
                .collect(),
            inverse: inverse,
        }
    }
}

impl<T> FFTAlgorithm<T> for Radix4<T>
    where T: Signed + FromPrimitive + Copy
{
    /// Runs the FFT on the input `signal` array, placing the output in the 'spectrum' array
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        // copy the data into the spectrum vector
        prepare_radix4(signal.len(), signal, spectrum, 1);

        // perform the butterflies. the butterfly size depends on the input size
        let num_bits = signal.len().trailing_zeros();
        let mut current_size = if num_bits % 2 > 0 {
            // the size is a power of 2, so we need to do size-2 butterflies,
            // with a stride of size / 2
            for chunk in spectrum.chunks_mut(2) {
                unsafe { butterfly_2_single(chunk, 1) }
            }

            // for the cross-ffts we want to to start off with a size of 8 (2 * 4)
            8
        } else {
            for chunk in spectrum.chunks_mut(4) {
                unsafe { butterfly_4_single(chunk, 1, self.inverse) }
            }

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
                                self.twiddles.as_slice(),
                                current_size / 4,
                                self.inverse)
                }
            }
            current_size *= 4;
        }
    }
}

// after testing an iterative bit reversal algorithm, this recursive algorithm
// was almost an order of magnitude faster at setting up
fn prepare_radix4<T: Copy>(size: usize,
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
