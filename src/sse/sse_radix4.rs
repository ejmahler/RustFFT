use num_complex::Complex;

use core::arch::x86_64::*;

use crate::algorithm::bitreversed_transpose;
use crate::array_utils::{self, workaround_transmute_mut};
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

use super::sse_vector::{SseArray, SseArrayMut};

/// FFT algorithm optimized for power-of-two sizes, SSE accelerated version.
/// This is designed to be used via a Planner, and not created directly.

const USE_BUTTERFLY32_FROM: usize = 262144; // Use length 32 butterfly starting from this length

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
                    if len < USE_BUTTERFLY32_FROM {
                        (8, Sse32Butterfly::Len8(SseF32Butterfly8::new(direction)))
                    } else {
                        (32, Sse32Butterfly::Len32(SseF32Butterfly32::new(direction)))
                    }
                } else {
                    (16, Sse32Butterfly::Len16(SseF32Butterfly16::new(direction)))
                }
            }
        };

        // precompute the twiddle factors this algorithm will use.
        // we're doing the same precomputation of twiddle factors as the mixed radix algorithm where width=4 and height=len/4
        // but mixed radix only does one step and then calls itself recusrively, and this algorithm does every layer all the way down
        // so we're going to pack all the "layers" of twiddle factors into a single array, starting with the bottom layer and going up
        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = base_len * ROW_COUNT;
        let mut twiddle_factors = Vec::with_capacity(len * 2);
        while cross_fft_len <= len {
            let num_scalar_columns = cross_fft_len / ROW_COUNT;
            let num_vector_columns = num_scalar_columns / 2; // 2 complex<T> per __m128

            for i in 0..num_vector_columns {
                for k in 1..ROW_COUNT {
                    unsafe {
                        let twiddle_a =
                            twiddles::compute_twiddle(2 * i * k, cross_fft_len, direction);
                        let twiddle_b =
                            twiddles::compute_twiddle((2 * i + 1) * k, cross_fft_len, direction);
                        let twiddles_packed =
                            _mm_set_ps(twiddle_b.im, twiddle_b.re, twiddle_a.im, twiddle_a.re);
                        twiddle_factors.push(twiddles_packed);
                    }
                }
            }
            cross_fft_len *= ROW_COUNT;
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

    #[target_feature(enable = "sse4.1")]
    unsafe fn perform_fft_out_of_place(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // copy the data into the output vector
        if self.len() == self.base_len {
            output.copy_from_slice(input);
        } else {
            bitreversed_transpose(self.base_len, input, output);
        }

        // Base-level FFTs
        match &self.base_fft {
            Sse32Butterfly::Len1(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse32Butterfly::Len2(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse32Butterfly::Len4(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse32Butterfly::Len8(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse32Butterfly::Len16(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse32Butterfly::Len32(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
        };

        // cross-FFTs
        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = self.base_len * ROW_COUNT;
        let mut layer_twiddles: &[__m128] = &self.twiddles;

        while cross_fft_len <= input.len() {
            let num_rows = input.len() / cross_fft_len;
            let num_columns = cross_fft_len / ROW_COUNT;

            for i in 0..num_rows {
                butterfly_4_32(
                    &mut output[i * cross_fft_len..],
                    layer_twiddles,
                    num_columns,
                    &self.bf4,
                )
            }

            // skip past all the twiddle factors used in this layer
            // half as many twiddles because sse vectors store 2 complex<f32> each
            let twiddle_offset = num_columns * (ROW_COUNT - 1) / 2;
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            cross_fft_len *= ROW_COUNT;
        }
    }
}
boilerplate_fft_sse_oop!(Sse32Radix4, |this: &Sse32Radix4<_>| this.len);

#[target_feature(enable = "sse4.1")]
unsafe fn butterfly_4_32<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[__m128],
    num_ffts: usize,
    bf4: &SseF32Butterfly4<T>,
) {
    let mut idx = 0usize;
    let mut buffer: &mut [Complex<f32>] = workaround_transmute_mut(data);
    for tw in twiddles.chunks_exact(6).take(num_ffts / 4) {
        let scratch0 = buffer.load_complex(idx);
        let scratch0b = buffer.load_complex(idx + 2);
        let mut scratch1 = buffer.load_complex(idx + 1 * num_ffts);
        let mut scratch1b = buffer.load_complex(idx + 2 + 1 * num_ffts);
        let mut scratch2 = buffer.load_complex(idx + 2 * num_ffts);
        let mut scratch2b = buffer.load_complex(idx + 2 + 2 * num_ffts);
        let mut scratch3 = buffer.load_complex(idx + 3 * num_ffts);
        let mut scratch3b = buffer.load_complex(idx + 2 + 3 * num_ffts);

        scratch1 = mul_complex_f32(scratch1, tw[0]);
        scratch2 = mul_complex_f32(scratch2, tw[1]);
        scratch3 = mul_complex_f32(scratch3, tw[2]);
        scratch1b = mul_complex_f32(scratch1b, tw[3]);
        scratch2b = mul_complex_f32(scratch2b, tw[4]);
        scratch3b = mul_complex_f32(scratch3b, tw[5]);

        let scratch = bf4.perform_parallel_fft_direct(scratch0, scratch1, scratch2, scratch3);
        let scratchb = bf4.perform_parallel_fft_direct(scratch0b, scratch1b, scratch2b, scratch3b);

        buffer.store_complex(scratch[0], idx);
        buffer.store_complex(scratchb[0], idx + 2);
        buffer.store_complex(scratch[1], idx + 1 * num_ffts);
        buffer.store_complex(scratchb[1], idx + 2 + 1 * num_ffts);
        buffer.store_complex(scratch[2], idx + 2 * num_ffts);
        buffer.store_complex(scratchb[2], idx + 2 + 2 * num_ffts);
        buffer.store_complex(scratch[3], idx + 3 * num_ffts);
        buffer.store_complex(scratchb[3], idx + 2 + 3 * num_ffts);

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
                    if len < USE_BUTTERFLY32_FROM {
                        (8, Sse64Butterfly::Len8(SseF64Butterfly8::new(direction)))
                    } else {
                        (32, Sse64Butterfly::Len32(SseF64Butterfly32::new(direction)))
                    }
                } else {
                    (16, Sse64Butterfly::Len16(SseF64Butterfly16::new(direction)))
                }
            }
        };

        // precompute the twiddle factors this algorithm will use.
        // we're doing the same precomputation of twiddle factors as the mixed radix algorithm where width=4 and height=len/4
        // but mixed radix only does one step and then calls itself recusrively, and this algorithm does every layer all the way down
        // so we're going to pack all the "layers" of twiddle factors into a single array, starting with the bottom layer and going up
        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = base_len * ROW_COUNT;
        let mut twiddle_factors = Vec::with_capacity(len * 2);
        while cross_fft_len <= len {
            let num_columns = cross_fft_len / ROW_COUNT;

            for i in 0..num_columns {
                for k in 1..ROW_COUNT {
                    unsafe {
                        let twiddle = twiddles::compute_twiddle(i * k, cross_fft_len, direction);
                        let twiddle_packed = _mm_set_pd(twiddle.im, twiddle.re);
                        twiddle_factors.push(twiddle_packed);
                    }
                }
            }
            cross_fft_len *= ROW_COUNT;
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

    #[target_feature(enable = "sse4.1")]
    unsafe fn perform_fft_out_of_place(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // copy the data into the output vector
        if self.len() == self.base_len {
            output.copy_from_slice(input);
        } else {
            bitreversed_transpose(self.base_len, input, output);
        }

        // Base-level FFTs
        match &self.base_fft {
            Sse64Butterfly::Len1(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse64Butterfly::Len2(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse64Butterfly::Len4(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse64Butterfly::Len8(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse64Butterfly::Len16(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
            Sse64Butterfly::Len32(bf) => bf.perform_fft_butterfly_multi(output).unwrap(),
        }

        // cross-FFTs
        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = self.base_len * ROW_COUNT;
        let mut layer_twiddles: &[__m128d] = &self.twiddles;

        while cross_fft_len <= input.len() {
            let num_rows = input.len() / cross_fft_len;
            let num_columns = cross_fft_len / ROW_COUNT;

            for i in 0..num_rows {
                butterfly_4_64(
                    &mut output[i * cross_fft_len..],
                    layer_twiddles,
                    num_columns,
                    &self.bf4,
                )
            }

            // skip past all the twiddle factors used in this layer
            let twiddle_offset = num_columns * (ROW_COUNT - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            cross_fft_len *= ROW_COUNT;
        }
    }
}
boilerplate_fft_sse_oop!(Sse64Radix4, |this: &Sse64Radix4<_>| this.len);

#[target_feature(enable = "sse4.1")]
unsafe fn butterfly_4_64<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[__m128d],
    num_ffts: usize,
    bf4: &SseF64Butterfly4<T>,
) {
    let mut idx = 0usize;
    let mut buffer: &mut [Complex<f64>] = workaround_transmute_mut(data);
    for tw in twiddles.chunks_exact(6).take(num_ffts / 2) {
        let scratch0 = buffer.load_complex(idx);
        let scratch0b = buffer.load_complex(idx + 1);
        let mut scratch1 = buffer.load_complex(idx + 1 * num_ffts);
        let mut scratch1b = buffer.load_complex(idx + 1 + 1 * num_ffts);
        let mut scratch2 = buffer.load_complex(idx + 2 * num_ffts);
        let mut scratch2b = buffer.load_complex(idx + 1 + 2 * num_ffts);
        let mut scratch3 = buffer.load_complex(idx + 3 * num_ffts);
        let mut scratch3b = buffer.load_complex(idx + 1 + 3 * num_ffts);

        scratch1 = mul_complex_f64(scratch1, tw[0]);
        scratch2 = mul_complex_f64(scratch2, tw[1]);
        scratch3 = mul_complex_f64(scratch3, tw[2]);
        scratch1b = mul_complex_f64(scratch1b, tw[3]);
        scratch2b = mul_complex_f64(scratch2b, tw[4]);
        scratch3b = mul_complex_f64(scratch3b, tw[5]);

        let scratch = bf4.perform_fft_direct(scratch0, scratch1, scratch2, scratch3);
        let scratchb = bf4.perform_fft_direct(scratch0b, scratch1b, scratch2b, scratch3b);

        buffer.store_complex(scratch[0], idx);
        buffer.store_complex(scratchb[0], idx + 1);
        buffer.store_complex(scratch[1], idx + 1 * num_ffts);
        buffer.store_complex(scratchb[1], idx + 1 + 1 * num_ffts);
        buffer.store_complex(scratch[2], idx + 2 * num_ffts);
        buffer.store_complex(scratchb[2], idx + 1 + 2 * num_ffts);
        buffer.store_complex(scratch[3], idx + 3 * num_ffts);
        buffer.store_complex(scratchb[3], idx + 1 + 3 * num_ffts);

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
