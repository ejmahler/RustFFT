use num::{Complex, FromPrimitive, Signed};

use butterflies::{butterfly_2, butterfly_4};

pub fn execute_radix4<T>(size: usize,
              spectrum: &mut [Complex<T>],
              stride: usize,
              twiddles: &[Complex<T>],
              inverse: bool) where T: Signed + FromPrimitive + Copy {
    match size {
        4 => unsafe { butterfly_4(spectrum, stride, twiddles, 1, inverse) },
        2 => unsafe { butterfly_2(spectrum, stride, twiddles, 1) },
        _ => {
            for i in 0..4 {
                execute_radix4(size / 4,
                        &mut spectrum[i * (size / 4)..],
                        stride * 4,
                        twiddles,
                        inverse);
            }
            unsafe { butterfly_4(spectrum, stride, twiddles, size / 4, inverse) };
        }
    }
}



// TODO: we should be able to do this in a simple loop, rather than recursively, using bit reversal
// the algorithm will be a little more complicated though due
// to us potentially using radix 2 for one of the teps
pub fn prepare_radix4<T: Copy>(size: usize,
                           signal: &[Complex<T>],
                           spectrum: &mut [Complex<T>],
                           stride: usize,
                           signal_offset: usize,
                           spectrum_offset: usize)
{
    match size {
        4 => unsafe {
            for i in 0..4 {
                *spectrum.get_unchecked_mut(spectrum_offset + i) =
                    *signal.get_unchecked(signal_offset + i * stride);
            }
        },
        2 => unsafe {
            for i in 0..2 {
                *spectrum.get_unchecked_mut(spectrum_offset + i) =
                    *signal.get_unchecked(signal_offset + i * stride);
            }
        },
        _ => {
            for i in 0..4 {
                prepare_radix4(size / 4,
                                     signal,
                                     spectrum,
                                     stride * 4,
                                     signal_offset + i * stride,
                                     spectrum_offset + i * (size / 4));
            }
        }
    }
}