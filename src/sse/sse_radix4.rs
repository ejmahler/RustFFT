use num_complex::Complex;

use core::arch::x86_64::*;

use crate::array_utils;
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::sse::sse_butterflies::{
    SseF32Butterfly1, SseF32Butterfly16, SseF32Butterfly2, SseF32Butterfly32, SseF32Butterfly4,
    SseF32Butterfly8,
};
use crate::sse::sse_butterflies::{
    SseF64Butterfly1, SseF64Butterfly16, SseF64Butterfly2, SseF64Butterfly32, SseF64Butterfly4,
    SseF64Butterfly8,
};
use crate::{common::FftNum, twiddles, FftDirection};
use crate::{Direction, Fft, Length};

use super::sse_common::{assert_f32, assert_f64};
use super::sse_utils::*;

/// FFT algorithm optimized for power-of-two sizes, SSE accelerated version.
/// This is designed to be used via a Planner, and not created directly.

enum Sse32Butterfly<T> {
    Len1(SseF32Butterfly1<T>),
    Len2(SseF32Butterfly2<T>),
    Len4(SseF32Butterfly4<T>),
    Len8(SseF32Butterfly8<T>),
    Len16(SseF32Butterfly16<T>),
    Len32(SseF32Butterfly32<T>),
}

enum Sse64Butterfly<T> {
    Len1(SseF64Butterfly1<T>),
    Len2(SseF64Butterfly2<T>),
    Len4(SseF64Butterfly4<T>),
    Len8(SseF64Butterfly8<T>),
    Len16(SseF64Butterfly16<T>),
    Len32(SseF64Butterfly32<T>),
}

pub struct Sse32Radix4<T> {
    _phantom: std::marker::PhantomData<T>,
    twiddles: Box<[__m128]>,

    base_fft: Sse32Butterfly<T>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
    bf4: SseF32Butterfly4<T>,
}

impl<T: FftNum> Sse32Radix4<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(len: usize, direction: FftDirection) -> Self {
        assert!(
            len.is_power_of_two(),
            "Radix4 algorithm requires a power-of-two input size. Got {}",
            len
        );
        assert_f32::<T>();

        // figure out which base length we're going to use
        let num_bits = len.trailing_zeros();
        let (base_len, base_fft) = match num_bits {
            0 => (len, Sse32Butterfly::Len1(SseF32Butterfly1::new(direction))),
            1 => (len, Sse32Butterfly::Len2(SseF32Butterfly2::new(direction))),
            2 => (len, Sse32Butterfly::Len4(SseF32Butterfly4::new(direction))),
            3 => (len, Sse32Butterfly::Len8(SseF32Butterfly8::new(direction))),
            _ => {
                if num_bits % 2 == 1 {
                    (32, Sse32Butterfly::Len32(SseF32Butterfly32::new(direction)))
                } else {
                    (16, Sse32Butterfly::Len16(SseF32Butterfly16::new(direction)))
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
            for i in 0..num_rows / 2 {
                for k in 1..4 {
                    unsafe {
                        let twiddle_a =
                            twiddles::compute_twiddle(2 * i * k * twiddle_stride, len, direction);
                        let twiddle_b = twiddles::compute_twiddle(
                            (2 * i + 1) * k * twiddle_stride,
                            len,
                            direction,
                        );
                        let twiddles_packed =
                            _mm_set_ps(twiddle_b.im, twiddle_b.re, twiddle_a.im, twiddle_a.re);
                        twiddle_factors.push(twiddles_packed);
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
            bf4: SseF32Butterfly4::<T>::new(direction),
        }
    }

    #[target_feature(enable = "sse3")]
    unsafe fn perform_fft_out_of_place(
        &self,
        signal: &[Complex<T>],
        spectrum: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // copy the data into the spectrum vector, split the copying up into chunks to make it more cache friendly
        let mut num_chunks = signal.len() / 16384;
        if num_chunks == 0 {
            num_chunks = 1;
        } else if num_chunks > self.base_len {
            num_chunks = self.base_len;
        }
        for n in 0..num_chunks {
            prepare_radix4(
                signal.len(),
                self.base_len,
                signal,
                spectrum,
                1,
                n,
                num_chunks,
            );
        }

        // Base-level FFTs
        match &self.base_fft {
            Sse32Butterfly::Len1(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len2(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len4(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len8(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len16(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len32(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
        };

        // cross-FFTs
        let mut current_size = self.base_len * 4;
        let mut layer_twiddles: &[__m128] = &self.twiddles;

        while current_size <= signal.len() {
            let num_rows = signal.len() / current_size;

            for i in 0..num_rows {
                butterfly_4_32(
                    &mut spectrum[i * current_size..],
                    layer_twiddles,
                    current_size / 4,
                    &self.bf4,
                )
            }

            //skip past all the twiddle factors used in this layer
            let twiddle_offset = (current_size * 3) / 8;
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            current_size *= 4;
        }
    }
}
boilerplate_fft_sse_oop!(Sse32Radix4, |this: &Sse32Radix4<_>| this.len);

#[target_feature(enable = "sse3")]
unsafe fn butterfly_4_32<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[__m128],
    num_ffts: usize,
    bf4: &SseF32Butterfly4<T>,
) {
    let mut idx = 0usize;
    let mut tw_idx = 0usize;
    let output_slice = data.as_mut_ptr() as *mut Complex<f32>;
    for _ in 0..num_ffts / 4 {
        let scratch0 = _mm_loadu_ps(output_slice.add(idx) as *const f32);
        let scratch0b = _mm_loadu_ps(output_slice.add(idx + 2) as *const f32);

        let mut scratch1 = _mm_loadu_ps(output_slice.add(idx + 1 * num_ffts) as *const f32);
        let mut scratch1b = _mm_loadu_ps(output_slice.add(idx + 2 + 1 * num_ffts) as *const f32);

        let mut scratch2 = _mm_loadu_ps(output_slice.add(idx + 2 * num_ffts) as *const f32);
        let mut scratch2b = _mm_loadu_ps(output_slice.add(idx + 2 + 2 * num_ffts) as *const f32);

        let mut scratch3 = _mm_loadu_ps(output_slice.add(idx + 3 * num_ffts) as *const f32);
        let mut scratch3b = _mm_loadu_ps(output_slice.add(idx + 2 + 3 * num_ffts) as *const f32);

        scratch1 = complex_dual_mul_f32(scratch1, twiddles[tw_idx]);
        scratch2 = complex_dual_mul_f32(scratch2, twiddles[tw_idx + 1]);
        scratch3 = complex_dual_mul_f32(scratch3, twiddles[tw_idx + 2]);
        scratch1b = complex_dual_mul_f32(scratch1b, twiddles[tw_idx + 3]);
        scratch2b = complex_dual_mul_f32(scratch2b, twiddles[tw_idx + 4]);
        scratch3b = complex_dual_mul_f32(scratch3b, twiddles[tw_idx + 5]);

        let scratch = bf4.perform_dual_fft_direct(scratch0, scratch1, scratch2, scratch3);
        let scratchb = bf4.perform_dual_fft_direct(scratch0b, scratch1b, scratch2b, scratch3b);

        _mm_storeu_ps(output_slice.add(idx) as *mut f32, scratch[0]);
        _mm_storeu_ps(output_slice.add(idx + 2) as *mut f32, scratchb[0]);

        _mm_storeu_ps(output_slice.add(idx + 1 * num_ffts) as *mut f32, scratch[1]);
        _mm_storeu_ps(
            output_slice.add(idx + 2 + 1 * num_ffts) as *mut f32,
            scratchb[1],
        );

        _mm_storeu_ps(output_slice.add(idx + 2 * num_ffts) as *mut f32, scratch[2]);
        _mm_storeu_ps(
            output_slice.add(idx + 2 + 2 * num_ffts) as *mut f32,
            scratchb[2],
        );

        _mm_storeu_ps(output_slice.add(idx + 3 * num_ffts) as *mut f32, scratch[3]);
        _mm_storeu_ps(
            output_slice.add(idx + 2 + 3 * num_ffts) as *mut f32,
            scratchb[3],
        );
        tw_idx += 6;
        idx += 4;
    }
}

pub struct Sse64Radix4<T> {
    _phantom: std::marker::PhantomData<T>,
    twiddles: Box<[__m128d]>,

    base_fft: Sse64Butterfly<T>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
    bf4: SseF64Butterfly4<T>,
}

impl<T: FftNum> Sse64Radix4<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(len: usize, direction: FftDirection) -> Self {
        assert!(
            len.is_power_of_two(),
            "Radix4 algorithm requires a power-of-two input size. Got {}",
            len
        );

        assert_f64::<T>();

        // figure out which base length we're going to use
        let num_bits = len.trailing_zeros();
        let (base_len, base_fft) = match num_bits {
            0 => (len, Sse64Butterfly::Len1(SseF64Butterfly1::new(direction))),
            1 => (len, Sse64Butterfly::Len2(SseF64Butterfly2::new(direction))),
            2 => (len, Sse64Butterfly::Len4(SseF64Butterfly4::new(direction))),
            3 => (len, Sse64Butterfly::Len8(SseF64Butterfly8::new(direction))),
            _ => {
                if num_bits % 2 == 1 {
                    (32, Sse64Butterfly::Len32(SseF64Butterfly32::new(direction)))
                } else {
                    (16, Sse64Butterfly::Len16(SseF64Butterfly16::new(direction)))
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
                        let twiddle =
                            twiddles::compute_twiddle(i * k * twiddle_stride, len, direction);
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
            bf4: SseF64Butterfly4::<T>::new(direction),
        }
    }

    #[target_feature(enable = "sse3")]
    unsafe fn perform_fft_out_of_place(
        &self,
        signal: &[Complex<T>],
        spectrum: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // copy the data into the spectrum vector, split the copying up into chunks to make it more cache friendly
        let mut num_chunks = signal.len() / 8192;
        if num_chunks == 0 {
            num_chunks = 1;
        } else if num_chunks > self.base_len {
            num_chunks = self.base_len;
        }
        for n in 0..num_chunks {
            prepare_radix4(
                signal.len(),
                self.base_len,
                signal,
                spectrum,
                1,
                n,
                num_chunks,
            );
        }

        // Base-level FFTs
        match &self.base_fft {
            Sse64Butterfly::Len1(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len2(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len4(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len8(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len16(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len32(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
        }

        // cross-FFTs
        let mut current_size = self.base_len * 4;
        let mut layer_twiddles: &[__m128d] = &self.twiddles;

        while current_size <= signal.len() {
            let num_rows = signal.len() / current_size;

            for i in 0..num_rows {
                butterfly_4_64(
                    &mut spectrum[i * current_size..],
                    layer_twiddles,
                    current_size / 4,
                    &self.bf4,
                )
            }

            //skip past all the twiddle factors used in this layer
            let twiddle_offset = (current_size * 3) / 4;
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            current_size *= 4;
        }
    }
}
boilerplate_fft_sse_oop!(Sse64Radix4, |this: &Sse64Radix4<_>| this.len);

fn prepare_radix4<T: FftNum>(
    size: usize,
    base_len: usize,
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    stride: usize,
    chunk: usize,
    nbr_chunks: usize,
) {
    if size == (4 * base_len) {
        do_radix4_shuffle(size, signal, spectrum, stride, chunk, nbr_chunks);
    } else if size == base_len {
        unsafe {
            for i in (chunk * base_len / nbr_chunks)..((chunk + 1) * base_len / nbr_chunks) {
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
                chunk,
                nbr_chunks,
            );
        }
    }
}

fn do_radix4_shuffle<T: FftNum>(
    size: usize,
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    stride: usize,
    chunk: usize,
    nbr_chunks: usize,
) {
    let stepsize = size / 4;
    let stepstride = stride * 4;
    let signal_offset = stride;
    let spectrum_offset = size / 4;
    unsafe {
        for i in (chunk * stepsize / nbr_chunks)..((chunk + 1) * stepsize / nbr_chunks) {
            let val0 = *signal.get_unchecked(i * stepstride);
            let val1 = *signal.get_unchecked(i * stepstride + signal_offset);
            let val2 = *signal.get_unchecked(i * stepstride + 2 * signal_offset);
            let val3 = *signal.get_unchecked(i * stepstride + 3 * signal_offset);
            *spectrum.get_unchecked_mut(i) = val0;
            *spectrum.get_unchecked_mut(i + spectrum_offset) = val1;
            *spectrum.get_unchecked_mut(i + 2 * spectrum_offset) = val2;
            *spectrum.get_unchecked_mut(i + 3 * spectrum_offset) = val3;
        }
    }
}

#[target_feature(enable = "sse3")]
unsafe fn butterfly_4_64<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[__m128d],
    num_ffts: usize,
    bf4: &SseF64Butterfly4<T>,
) {
    let mut idx = 0usize;
    let mut tw_idx = 0usize;
    let output_slice = data.as_mut_ptr() as *mut Complex<f64>;
    for _ in 0..num_ffts / 2 {
        let scratch0 = _mm_loadu_pd(output_slice.add(idx) as *const f64);
        let scratch0b = _mm_loadu_pd(output_slice.add(idx + 1) as *const f64);
        let mut scratch1 = _mm_loadu_pd(output_slice.add(idx + 1 * num_ffts) as *const f64);
        let mut scratch1b = _mm_loadu_pd(output_slice.add(idx + 1 + 1 * num_ffts) as *const f64);
        let mut scratch2 = _mm_loadu_pd(output_slice.add(idx + 2 * num_ffts) as *const f64);
        let mut scratch2b = _mm_loadu_pd(output_slice.add(idx + 1 + 2 * num_ffts) as *const f64);
        let mut scratch3 = _mm_loadu_pd(output_slice.add(idx + 3 * num_ffts) as *const f64);
        let mut scratch3b = _mm_loadu_pd(output_slice.add(idx + 1 + 3 * num_ffts) as *const f64);

        scratch1 = complex_mul_f64(scratch1, twiddles[tw_idx]);
        scratch2 = complex_mul_f64(scratch2, twiddles[tw_idx + 1]);
        scratch3 = complex_mul_f64(scratch3, twiddles[tw_idx + 2]);
        scratch1b = complex_mul_f64(scratch1b, twiddles[tw_idx + 3]);
        scratch2b = complex_mul_f64(scratch2b, twiddles[tw_idx + 4]);
        scratch3b = complex_mul_f64(scratch3b, twiddles[tw_idx + 5]);

        let scratch = bf4.perform_fft_direct(scratch0, scratch1, scratch2, scratch3);
        let scratchb = bf4.perform_fft_direct(scratch0b, scratch1b, scratch2b, scratch3b);

        _mm_storeu_pd(output_slice.add(idx) as *mut f64, scratch[0]);
        _mm_storeu_pd(output_slice.add(idx + 1) as *mut f64, scratchb[0]);
        _mm_storeu_pd(output_slice.add(idx + 1 * num_ffts) as *mut f64, scratch[1]);
        _mm_storeu_pd(
            output_slice.add(idx + 1 + 1 * num_ffts) as *mut f64,
            scratchb[1],
        );
        _mm_storeu_pd(output_slice.add(idx + 2 * num_ffts) as *mut f64, scratch[2]);
        _mm_storeu_pd(
            output_slice.add(idx + 1 + 2 * num_ffts) as *mut f64,
            scratchb[2],
        );
        _mm_storeu_pd(output_slice.add(idx + 3 * num_ffts) as *mut f64, scratch[3]);
        _mm_storeu_pd(
            output_slice.add(idx + 1 + 3 * num_ffts) as *mut f64,
            scratchb[3],
        );

        tw_idx += 6;
        idx += 2;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    #[test]
    fn test_sse_radix4_64() {
        for pow in 4..12 {
            let len = 1 << pow;
            test_sse_radix4_64_with_length(len, FftDirection::Forward);
            test_sse_radix4_64_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_sse_radix4_64_with_length(len: usize, direction: FftDirection) {
        let fft = Sse64Radix4::new(len, direction);
        check_fft_algorithm::<f64>(&fft, len, direction);
    }

    #[test]
    fn test_sse_radix4_32() {
        for pow in 0..12 {
            let len = 1 << pow;
            test_sse_radix4_32_with_length(len, FftDirection::Forward);
            test_sse_radix4_32_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_sse_radix4_32_with_length(len: usize, direction: FftDirection) {
        let fft = Sse32Radix4::new(len, direction);
        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
