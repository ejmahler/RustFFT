use std::sync::Arc;

use num_complex::Complex;
//use num_traits::Zero;

use core::arch::x86_64::*;

use crate::sse::sse_butterflies::{Sse64Butterfly1, Sse64Butterfly16, Sse64Butterfly2, Sse64Butterfly4, Sse64Butterfly8};
use crate::array_utils;
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::{
    //array_utils::{RawSlice, RawSliceMut},
    common::FftNum,
    twiddles, FftDirection,
};
use crate::{Direction, Fft, Length};

use super::sse_utils::*;

use super::sse_common;


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

pub struct Sse64Radix4<T> {
    _phantom: std::marker::PhantomData<T>,
    twiddles: Box<[__m128d]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
    bf4: Sse64Butterfly4<T>,
}

impl<T: FftNum> Sse64Radix4<T> {
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
            0 => (len, Arc::new(Sse64Butterfly1::new(direction)) as Arc<dyn Fft<T>>),
            1 => (len, Arc::new(Sse64Butterfly2::new(direction)) as Arc<dyn Fft<T>>),
            2 => (len, Arc::new(Sse64Butterfly4::new(direction)) as Arc<dyn Fft<T>>),
            _ => {
                if num_bits % 2 == 1 {
                    (8, Arc::new(Sse64Butterfly8::new(direction)) as Arc<dyn Fft<T>>)
                } else {
                    (16, Arc::new(Sse64Butterfly16::new(direction)) as Arc<dyn Fft<T>>)
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
                    unsafe {
                        let twiddle = twiddles::compute_twiddle(i * k * twiddle_stride, len, direction);
                        let twiddle_packed = _mm_set_pd(twiddle.im, twiddle.re);
                        twiddle_factors.push(twiddle_packed);
                    }
                }
            }
            twiddle_stride >>= 2;
        }

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

            base_fft,
            base_len,

            len,
            direction,
            _phantom: std::marker::PhantomData,
            bf4: Sse64Butterfly4::<T>::new(direction),
        }
    }

    
    fn perform_fft_out_of_place(
        &self,
        signal: &[Complex<T>],
        spectrum: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // copy the data into the spectrum vector
        prepare_radix4(signal.len(), self.base_len, signal, spectrum, 1);

        // Base-level FFTs
        self.base_fft.process_with_scratch(spectrum, &mut []);

        // cross-FFTs
        let mut current_size = self.base_len * 4;
        let mut layer_twiddles: &[__m128d] = &self.twiddles;

        while current_size <= signal.len() {
            let num_rows = signal.len() / current_size;

            for i in 0..num_rows {
                unsafe {
                    butterfly_4(
                        &mut spectrum[i * current_size..],
                        layer_twiddles,
                        current_size / 4,
                        &self.bf4,
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
boilerplate_fft_sse_oop!(Sse64Radix4, |this: &Sse64Radix4<_>| this.len);

// after testing an iterative bit reversal algorithm, this recursive algorithm
// was almost an order of magnitude faster at setting up
fn prepare_radix4<T: FftNum>(
    size: usize,
    base_len: usize,
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    stride: usize,
) {
    if size == base_len {
        unsafe {
            for i in 0..size {
                *spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        }
    } else {
        for i in 0..4 {
            prepare_radix4(
                size / 4,
                base_len,
                &signal[i * stride..],
                &mut spectrum[i * (size / 4)..],
                stride * 4,
            );
        }
    }
}

unsafe fn butterfly_4<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[__m128d],
    num_ffts: usize,
    bf4: &Sse64Butterfly4<T>,
) {
    let mut idx = 0usize;
    let mut tw_idx = 0usize;
    for _ in 0..num_ffts {
        let scratch0 = _mm_loadu_pd(data.as_ptr().add(idx) as *const f64);
        let mut scratch1 = _mm_loadu_pd(data.as_ptr().add(idx + 1 * num_ffts) as *const f64);
        let mut scratch2 = _mm_loadu_pd(data.as_ptr().add(idx + 2 * num_ffts) as *const f64);
        let mut scratch3 = _mm_loadu_pd(data.as_ptr().add(idx + 3 * num_ffts) as *const f64);

        scratch1 = complex_mul_64(scratch1, twiddles[tw_idx]);
        scratch2 = complex_mul_64(scratch2, twiddles[tw_idx + 1]);
        scratch3 = complex_mul_64(scratch3, twiddles[tw_idx + 2]);

        let scratch = bf4.perform_fft_direct(scratch0, scratch1, scratch2, scratch3);

        let array = std::mem::transmute::<[__m128d; 4], [Complex<f64>; 4]>(scratch);

        let output_slice = data.as_mut_ptr() as *mut Complex<f64>;

        *output_slice.add(idx) = array[0];
        *output_slice.add(idx + 1 * num_ffts) = array[1];
        *output_slice.add(idx + 2 * num_ffts) = array[2];
        *output_slice.add(idx + 3 * num_ffts) = array[3];

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
        for pow in 0..8 {
            let len = 1 << pow;
            test_radix4_with_length(len, FftDirection::Forward);
            test_radix4_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_radix4_with_length(len: usize, direction: FftDirection) {
        let fft = Sse64Radix4::new(len, direction);

        check_fft_algorithm::<f64>(&fft, len, direction);
    }
}
