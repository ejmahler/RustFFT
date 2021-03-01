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
use crate::{
    //array_utils::{RawSlice, RawSliceMut},
    common::FftNum,
    twiddles,
    FftDirection,
};
use crate::{Direction, Fft, Length};

use super::sse_common::{assert_f32, assert_f64};
use super::sse_utils::*;

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
            for i in 0..num_rows/2 {
                // //for k in 1..4 {
                // unsafe {
                //     let twiddle1a = twiddles::compute_twiddle(2*i * twiddle_stride, len, direction);
                //     let twiddle2a = twiddles::compute_twiddle(2*i * 2 * twiddle_stride, len, direction);
                //     let twiddle3a = twiddles::compute_twiddle(2*i * 3 * twiddle_stride, len, direction);
                //     let twiddle1b = twiddles::compute_twiddle((2*i+1) * twiddle_stride, len, direction);
                //     let twiddle2b = twiddles::compute_twiddle((2*i+1) * 2 * twiddle_stride, len, direction);
                //     let twiddle3b = twiddles::compute_twiddle((2*i+1) * 3 * twiddle_stride, len, direction);
                //     let twiddle1_packed = _mm_set_ps(twiddle1b.im, twiddle1b.re, twiddle1a.im, twiddle1a.re);
                //     let twiddle2_packed = _mm_set_ps(twiddle2b.im, twiddle2b.re, twiddle2a.im, twiddle2a.re);
                //     let twiddle3_packed = _mm_set_ps(twiddle3b.im, twiddle3b.re, twiddle3a.im, twiddle3a.re);
                //     twiddle_factors.push(twiddle1_packed);
                //     twiddle_factors.push(twiddle2_packed);
                //     twiddle_factors.push(twiddle3_packed);
                // }
                // //}
                for k in 1..4 {
                    unsafe {
                        let twiddle_a =
                            twiddles::compute_twiddle(2*i * k * twiddle_stride, len, direction);
                        let twiddle_b =
                            twiddles::compute_twiddle((2*i + 1) * k * twiddle_stride, len, direction);
                        let twiddles_packed = _mm_set_ps(twiddle_b.im, twiddle_b.re, twiddle_a.im, twiddle_a.re);
                        twiddle_factors.push(twiddles_packed);
                    }
                }
            }
            twiddle_stride >>= 2;
        }

        // while twiddle_stride > 0 {
        //     let num_rows = len / (twiddle_stride * 4);
        //     for i in 0..num_rows {
        //         for k in 1..4 {
        //             unsafe {
        //                 let twiddle =
        //                     twiddles::compute_twiddle(i * k * twiddle_stride, len, direction);
        //                 let twiddle_packed = _mm_set_pd(twiddle.im, twiddle.re);
        //                 twiddle_factors.push(twiddle_packed);
        //             }
        //         }
        //     }
        //     twiddle_stride >>= 2;
        // }


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
        //let start = time::Instant::now();
        // copy the data into the spectrum vector

        let signal_tm: &[Complex<f32>] = array_utils::workaround_transmute(signal);
        let spectrum_tm: &mut [Complex<f32>] = array_utils::workaround_transmute_mut(spectrum);
        prepare_radix4_sse32(signal.len(), self.base_len, signal_tm, spectrum_tm, 1);

        //prepare_radix4_32(signal.len(), self.base_len, signal, spectrum, 1);
        //let end = time::Instant::now();
        //println!("prepare: {} ns", end.duration_since(start).as_nanos());
        //let start = time::Instant::now();
        // Base-level FFTs
        match &self.base_fft {
            Sse32Butterfly::Len1(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len2(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len4(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len8(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len16(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse32Butterfly::Len32(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
        }

        //let end = time::Instant::now();
        //println!("base fft: {} ns", end.duration_since(start).as_nanos());

        //let start = time::Instant::now();
        //self.base_fft.perform_fft_butterfly(spectrum); //, &mut []);

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
            //let twiddle_offset = current_size / 4;
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            current_size *= 4;
        }
        //let end = time::Instant::now();
        //println!("cross fft: {} ns", end.duration_since(start).as_nanos());
    }
}
boilerplate_fft_sse_oop!(Sse32Radix4, |this: &Sse32Radix4<_>| this.len);

// after testing an iterative bit reversal algorithm, this recursive algorithm
// was almost an order of magnitude faster at setting up
fn prepare_radix4_32<T: FftNum>(
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
            prepare_radix4_32(
                size / 4,
                base_len,
                &signal[i * stride..],
                &mut spectrum[i * (size / 4)..],
                stride * 4,
            );
        }
    }
}

// after testing an iterative bit reversal algorithm, this recursive algorithm
// was almost an order of magnitude faster at setting up
fn prepare_radix4_sse32(
    size: usize,
    base_len: usize,
    signal: &[Complex<f32>],
    spectrum: &mut [Complex<f32>],
    stride: usize,
) {
    if size == base_len {
        unsafe {
            for i in 0..size {
                _mm_storel_pd(spectrum.as_mut_ptr().add(i) as *mut f64, _mm_load1_pd(signal.as_ptr().add(i * stride) as *const f64));
                //*spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        }
    } else {
        for i in 0..4 {
            prepare_radix4_sse32(
                size / 4,
                base_len,
                &signal[i * stride..],
                &mut spectrum[i * (size / 4)..],
                stride * 4,
            );
        }
    }
}

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
    /*
    for _ in 0..num_ffts {
        // There is no intrinsic to load the lower or upper two singles. Instead we load them as a double and the cast the __m128d to a __m128.
        let scratch0 = _mm_load_sd(output_slice.add(idx) as *const f64);
        let mut scratch0 = _mm_castpd_ps(_mm_loadh_pd(
            scratch0,
            output_slice.add(idx + 1 * num_ffts) as *const f64,
        ));
        let scratch2 = _mm_load_sd(output_slice.add(idx + 2 * num_ffts) as *const f64);
        let mut scratch2 = _mm_castpd_ps(_mm_loadh_pd(
            scratch2,
            output_slice.add(idx + 3 * num_ffts) as *const f64,
        ));

        scratch0 = complex_dual_mul_f32(scratch0, twiddles[tw_idx]);
        scratch2 = complex_dual_mul_f32(scratch2, twiddles[tw_idx + 1]);

        let scratch = bf4.perform_fft_direct(scratch0, scratch2);

        // There is no intrinsic to store the lower or upper two elements of a __m128. Instead we cast the __m128d to a __m128 and store them as single f64.
        _mm_storel_pd(output_slice.add(idx) as *mut f64, _mm_castps_pd(scratch[0]));
        _mm_storeh_pd(
            output_slice.add(idx + 1 * num_ffts) as *mut f64,
            _mm_castps_pd(scratch[0]),
        );
        _mm_storel_pd(
            output_slice.add(idx + 2 * num_ffts) as *mut f64,
            _mm_castps_pd(scratch[1]),
        );
        _mm_storeh_pd(
            output_slice.add(idx + 3 * num_ffts) as *mut f64,
            _mm_castps_pd(scratch[1]),
        );

        tw_idx += 2;
        idx += 1;
    } */
    for _ in 0..num_ffts/4 {
        let scratch0 = _mm_loadu_ps(output_slice.add(idx) as *const f32);
        let scratch0b = _mm_loadu_ps(output_slice.add(idx+2) as *const f32);
        
        let mut scratch1 = _mm_loadu_ps(output_slice.add(idx + 1 * num_ffts) as *const f32);
        let mut scratch1b = _mm_loadu_ps(output_slice.add(idx+2 + 1 * num_ffts) as *const f32);

        let mut scratch2 = _mm_loadu_ps(output_slice.add(idx + 2 * num_ffts) as *const f32);
        let mut scratch2b = _mm_loadu_ps(output_slice.add(idx+2 + 2 * num_ffts) as *const f32);

        let mut scratch3 = _mm_loadu_ps(output_slice.add(idx + 3 * num_ffts) as *const f32);
        let mut scratch3b = _mm_loadu_ps(output_slice.add(idx+2 + 3 * num_ffts) as *const f32);

        scratch1 = complex_dual_mul_f32(scratch1, twiddles[tw_idx]);
        scratch2 = complex_dual_mul_f32(scratch2, twiddles[tw_idx + 1]);
        scratch3 = complex_dual_mul_f32(scratch3, twiddles[tw_idx + 2]);
        scratch1b = complex_dual_mul_f32(scratch1b, twiddles[tw_idx + 3]);
        scratch2b = complex_dual_mul_f32(scratch2b, twiddles[tw_idx + 4]);
        scratch3b = complex_dual_mul_f32(scratch3b, twiddles[tw_idx + 5]);

        let scratch = bf4.perform_dual_fft_direct(scratch0, scratch1, scratch2, scratch3);
        let scratchb = bf4.perform_dual_fft_direct(scratch0b, scratch1b, scratch2b, scratch3b);

        _mm_storeu_ps(output_slice.add(idx) as *mut f32, scratch[0]);
        _mm_storeu_ps(output_slice.add(idx+2) as *mut f32, scratchb[0]);

        _mm_storeu_ps(output_slice.add(idx + 1 * num_ffts) as *mut f32, scratch[1]);
        _mm_storeu_ps(output_slice.add(idx+2 + 1 * num_ffts) as *mut f32, scratchb[1]);

        _mm_storeu_ps(output_slice.add(idx + 2 * num_ffts) as *mut f32, scratch[2]);
        _mm_storeu_ps(output_slice.add(idx+2 + 2 * num_ffts) as *mut f32, scratchb[2]);

        _mm_storeu_ps(output_slice.add(idx + 3 * num_ffts) as *mut f32, scratch[3]);
        _mm_storeu_ps(output_slice.add(idx+2 + 3 * num_ffts) as *mut f32, scratchb[3]);
        tw_idx += 6;
        idx += 4;
        // tw_idx += 3;
        // idx += 2;
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
        //let start = time::Instant::now();
        // copy the data into the spectrum vector
        let signal_tm: &[Complex<f64>] = array_utils::workaround_transmute(signal);
        let spectrum_tm: &mut [Complex<f64>] = array_utils::workaround_transmute_mut(spectrum);
        prepare_radix4_sse64(signal.len(), self.base_len, signal_tm, spectrum_tm, 1);
        //prepare_radix4_64(signal.len(), self.base_len, signal, spectrum, 1);
        //let end = time::Instant::now();
        //println!("prepare: {} ns", end.duration_since(start).as_nanos());

        //let start = time::Instant::now();
        // Base-level FFTs
        //self.base_fft.process_with_scratch(spectrum, &mut []);
        match &self.base_fft {
            Sse64Butterfly::Len1(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len2(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len4(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len8(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len16(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Sse64Butterfly::Len32(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
        }

        //let end = time::Instant::now();
        //println!("base fft: {} ns", end.duration_since(start).as_nanos());
        //
        //let start = time::Instant::now();
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
        //let end = time::Instant::now();
        //println!("cross fft: {} ns", end.duration_since(start).as_nanos());
    }
}
boilerplate_fft_sse_oop!(Sse64Radix4, |this: &Sse64Radix4<_>| this.len);

// after testing an iterative bit reversal algorithm, this recursive algorithm
// was almost an order of magnitude faster at setting up
fn prepare_radix4_64<T: FftNum>(
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
            prepare_radix4_64(
                size / 4,
                base_len,
                &signal[i * stride..],
                &mut spectrum[i * (size / 4)..],
                stride * 4,
            );
        }
    }
}

// after testing an iterative bit reversal algorithm, this recursive algorithm
// was almost an order of magnitude faster at setting up
fn prepare_radix4_sse64(
    size: usize,
    base_len: usize,
    signal: &[Complex<f64>],
    spectrum: &mut [Complex<f64>],
    stride: usize,
) {
    if size == base_len {
        unsafe {
            for i in 0..size {
                _mm_storeu_pd(spectrum.as_mut_ptr().add(i) as *mut f64, _mm_loadu_pd(signal.as_ptr().add(i * stride) as *const f64));
                //*spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        }
    } else {
        for i in 0..4 {
            prepare_radix4_sse64(
                size / 4,
                base_len,
                &signal[i * stride..],
                &mut spectrum[i * (size / 4)..],
                stride * 4,
            );
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
    for _ in 0..num_ffts/2 {
        let scratch0 = _mm_loadu_pd(output_slice.add(idx) as *const f64);
        let scratch0b = _mm_loadu_pd(output_slice.add(idx+1) as *const f64);
        let mut scratch1 = _mm_loadu_pd(output_slice.add(idx + 1 * num_ffts) as *const f64);
        let mut scratch1b = _mm_loadu_pd(output_slice.add(idx+1 + 1 * num_ffts) as *const f64);
        let mut scratch2 = _mm_loadu_pd(output_slice.add(idx + 2 * num_ffts) as *const f64);
        let mut scratch2b = _mm_loadu_pd(output_slice.add(idx+1 + 2 * num_ffts) as *const f64);
        let mut scratch3 = _mm_loadu_pd(output_slice.add(idx + 3 * num_ffts) as *const f64);
        let mut scratch3b = _mm_loadu_pd(output_slice.add(idx+1 + 3 * num_ffts) as *const f64);

        scratch1 = complex_mul_f64(scratch1, twiddles[tw_idx]);
        scratch2 = complex_mul_f64(scratch2, twiddles[tw_idx + 1]);
        scratch3 = complex_mul_f64(scratch3, twiddles[tw_idx + 2]);
        scratch1b = complex_mul_f64(scratch1b, twiddles[tw_idx + 3]);
        scratch2b = complex_mul_f64(scratch2b, twiddles[tw_idx + 4]);
        scratch3b = complex_mul_f64(scratch3b, twiddles[tw_idx + 5]);

        let scratch = bf4.perform_fft_direct(scratch0, scratch1, scratch2, scratch3);
        let scratchb = bf4.perform_fft_direct(scratch0b, scratch1b, scratch2b, scratch3b);

        _mm_storeu_pd(output_slice.add(idx) as *mut f64, scratch[0]);
        _mm_storeu_pd(output_slice.add(idx+1) as *mut f64, scratchb[0]);
        _mm_storeu_pd(output_slice.add(idx + 1 * num_ffts) as *mut f64, scratch[1]);
        _mm_storeu_pd(output_slice.add(idx+1 + 1 * num_ffts) as *mut f64, scratchb[1]);
        _mm_storeu_pd(output_slice.add(idx + 2 * num_ffts) as *mut f64, scratch[2]);
        _mm_storeu_pd(output_slice.add(idx+1 + 2 * num_ffts) as *mut f64, scratchb[2]);
        _mm_storeu_pd(output_slice.add(idx + 3 * num_ffts) as *mut f64, scratch[3]);
        _mm_storeu_pd(output_slice.add(idx+1 + 3 * num_ffts) as *mut f64, scratchb[3]);

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
        for pow in 0..12 {
            let len = 1 << pow;
            test_sse_radix4_64_with_length(len, FftDirection::Forward);
            test_sse_radix4_64_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_sse_radix4_64_with_length(len: usize, direction: FftDirection) {
        let fft = Sse64Radix4::new(len, direction);
        check_fft_algorithm::<f64>(&fft, len, direction);
    }

    //#[test]
    //fn test_dummy_radix4_64() {
    //    let fft = Sse64Radix4::<f64>::new(65536, FftDirection::Forward);
    //    let mut data = vec![Complex::from(0.0); 65536];
    //    fft.process(&mut data);
    //    assert!(false);
    //}
    //
    //#[test]
    //fn test_dummy_radix4_32() {
    //    let fft = Sse32Radix4::<f32>::new(65536, FftDirection::Forward);
    //    let mut data = vec![Complex::<f32>::from(0.0); 65536];
    //    fft.process(&mut data);
    //    assert!(false);
    //}

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

    fn prepare_radix4(
        size: usize,
        base_len: usize,
        signal: &[f32],
        spectrum: &mut [f32],
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

    #[test]
    fn test_prepare() {
        let len = 128;
        let mut signal = vec![0.0; len];
        let mut spectrum = vec![0.0_f32; len];
        for n in 0..signal.len() {
            signal[n] = n as f32;
        }
        prepare_radix4(signal.len(), 32, &signal[..], &mut spectrum[..], 1);
        println!("{:?}", spectrum);
        assert!(false);
    }
}
