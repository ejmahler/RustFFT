
use num::{Complex, FromPrimitive, Signed};

use butterflies::{butterfly_2, butterfly_3, butterfly_4, butterfly_5, butterfly};

pub fn cooley_tukey<T>(signal: &[Complex<T>],
                   spectrum: &mut [Complex<T>],
                   stride: usize,
                   twiddles: &[Complex<T>],
                   factors: &[(usize, usize)],
                   scratch: &mut [Complex<T>],
                   inverse: bool) where T: Signed + FromPrimitive + Copy {
    if let Some(&(n1, n2)) = factors.first() {
        if n2 == 1 {
            // we theoretically need to compute n1 ffts of size n2
            // but n2 is 1, and a fft of size 1 is just a copy
            copy_data(signal, spectrum, stride);
        } else {
            // Recursive call to perform n1 ffts of length n2
            for i in 0..n1 {
                cooley_tukey(&signal[i * stride..],
                             &mut spectrum[i * n2..],
                             stride * n1, twiddles, &factors[1..],
                             scratch, inverse);
            }
        }

        match n1 {
            5 => unsafe { butterfly_5(spectrum, stride, twiddles, n2) },
            4 => unsafe { butterfly_4(spectrum, stride, twiddles, n2, inverse) },
            3 => unsafe { butterfly_3(spectrum, stride, twiddles, n2) },
            2 => unsafe { butterfly_2(spectrum, stride, twiddles, n2) },
            _ => butterfly(spectrum, stride, twiddles, n2, n1, &mut scratch[..n1]),
        }
    }
}

fn copy_data<T: Copy>(signal: &[Complex<T>], spectrum: &mut [Complex<T>], stride: usize)
{
    let mut spectrum_idx = 0usize;
    let mut signal_idx = 0usize;
    while signal_idx < signal.len() {
        unsafe {
            *spectrum.get_unchecked_mut(spectrum_idx) = *signal.get_unchecked(signal_idx);
        }
        spectrum_idx += 1;
        signal_idx += stride;
    }
}