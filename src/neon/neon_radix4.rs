use num_complex::Complex;

use core::arch::aarch64::*;

use crate::algorithm::bitreversed_transpose;
use crate::array_utils;
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::neon::neon_butterflies::{
    NeonF32Butterfly1, NeonF32Butterfly16, NeonF32Butterfly2, NeonF32Butterfly32,
    NeonF32Butterfly4, NeonF32Butterfly8,
};
use crate::neon::neon_butterflies::{
    NeonF64Butterfly1, NeonF64Butterfly16, NeonF64Butterfly2, NeonF64Butterfly32,
    NeonF64Butterfly4, NeonF64Butterfly8,
};
use crate::{common::FftNum, twiddles, FftDirection};
use crate::{Direction, Fft, Length};

use super::neon_common::{assert_f32, assert_f64};
use super::neon_utils::*;

use super::neon_vector::{NeonArray, NeonArrayMut};
use crate::array_utils::{RawSlice, RawSliceMut};

/// FFT algorithm optimized for power-of-two sizes, Neon accelerated version.
/// This is designed to be used via a Planner, and not created directly.

const USE_BUTTERFLY32_FROM: usize = 262144; // Use length 32 butterfly starting from this length

enum Neon32Butterfly<T> {
    Len1(NeonF32Butterfly1<T>),
    Len2(NeonF32Butterfly2<T>),
    Len4(NeonF32Butterfly4<T>),
    Len8(NeonF32Butterfly8<T>),
    Len16(NeonF32Butterfly16<T>),
    Len32(NeonF32Butterfly32<T>),
}

enum Neon64Butterfly<T> {
    Len1(NeonF64Butterfly1<T>),
    Len2(NeonF64Butterfly2<T>),
    Len4(NeonF64Butterfly4<T>),
    Len8(NeonF64Butterfly8<T>),
    Len16(NeonF64Butterfly16<T>),
    Len32(NeonF64Butterfly32<T>),
}

pub struct Neon32Radix4<T> {
    _phantom: std::marker::PhantomData<T>,
    twiddles: Box<[float32x4_t]>,

    base_fft: Neon32Butterfly<T>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
    bf4: NeonF32Butterfly4<T>,
}

impl<T: FftNum> Neon32Radix4<T> {
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
            0 => (
                len,
                Neon32Butterfly::Len1(NeonF32Butterfly1::new(direction)),
            ),
            1 => (
                len,
                Neon32Butterfly::Len2(NeonF32Butterfly2::new(direction)),
            ),
            2 => (
                len,
                Neon32Butterfly::Len4(NeonF32Butterfly4::new(direction)),
            ),
            3 => (
                len,
                Neon32Butterfly::Len8(NeonF32Butterfly8::new(direction)),
            ),
            _ => {
                if num_bits % 2 == 1 {
                    if len < USE_BUTTERFLY32_FROM {
                        (8, Neon32Butterfly::Len8(NeonF32Butterfly8::new(direction)))
                    } else {
                        (
                            32,
                            Neon32Butterfly::Len32(NeonF32Butterfly32::new(direction)),
                        )
                    }
                } else {
                    (
                        16,
                        Neon32Butterfly::Len16(NeonF32Butterfly16::new(direction)),
                    )
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
                    let twiddle_a = twiddles::compute_twiddle::<f32>(
                        2 * i * k * twiddle_stride,
                        len,
                        direction,
                    );
                    let twiddle_b = twiddles::compute_twiddle::<f32>(
                        (2 * i + 1) * k * twiddle_stride,
                        len,
                        direction,
                    );
                    let twiddles_packed =
                        unsafe { RawSlice::new(&[twiddle_a, twiddle_b]).load_complex(0) };
                    twiddle_factors.push(twiddles_packed);
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
            bf4: NeonF32Butterfly4::<T>::new(direction),
        }
    }

    //#[target_feature(enable = "neon")]
    unsafe fn perform_fft_out_of_place(
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
        match &self.base_fft {
            Neon32Butterfly::Len1(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon32Butterfly::Len2(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon32Butterfly::Len4(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon32Butterfly::Len8(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon32Butterfly::Len16(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon32Butterfly::Len32(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
        };

        // cross-FFTs
        let mut current_size = self.base_len * 4;
        let mut layer_twiddles: &[float32x4_t] = &self.twiddles;

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
boilerplate_fft_neon_oop!(Neon32Radix4, |this: &Neon32Radix4<_>| this.len);

//#[target_feature(enable = "neon")]
unsafe fn butterfly_4_32<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[float32x4_t],
    num_ffts: usize,
    bf4: &NeonF32Butterfly4<T>,
) {
    let mut idx = 0usize;
    let input: RawSlice<Complex<f32>> = RawSlice::new_transmuted(data);
    let output: RawSliceMut<Complex<f32>> = RawSliceMut::new_transmuted(data);
    for tw in twiddles.chunks_exact(6).take(num_ffts / 4) {
        let scratch0 = input.load_complex(idx);
        let scratch0b = input.load_complex(idx + 2);
        let mut scratch1 = input.load_complex(idx + 1 * num_ffts);
        let mut scratch1b = input.load_complex(idx + 2 + 1 * num_ffts);
        let mut scratch2 = input.load_complex(idx + 2 * num_ffts);
        let mut scratch2b = input.load_complex(idx + 2 + 2 * num_ffts);
        let mut scratch3 = input.load_complex(idx + 3 * num_ffts);
        let mut scratch3b = input.load_complex(idx + 2 + 3 * num_ffts);

        scratch1 = mul_complex_f32(scratch1, tw[0]);
        scratch2 = mul_complex_f32(scratch2, tw[1]);
        scratch3 = mul_complex_f32(scratch3, tw[2]);
        scratch1b = mul_complex_f32(scratch1b, tw[3]);
        scratch2b = mul_complex_f32(scratch2b, tw[4]);
        scratch3b = mul_complex_f32(scratch3b, tw[5]);

        let scratch = bf4.perform_parallel_fft_direct(scratch0, scratch1, scratch2, scratch3);
        let scratchb = bf4.perform_parallel_fft_direct(scratch0b, scratch1b, scratch2b, scratch3b);

        output.store_complex(scratch[0], idx);
        output.store_complex(scratchb[0], idx + 2);
        output.store_complex(scratch[1], idx + 1 * num_ffts);
        output.store_complex(scratchb[1], idx + 2 + 1 * num_ffts);
        output.store_complex(scratch[2], idx + 2 * num_ffts);
        output.store_complex(scratchb[2], idx + 2 + 2 * num_ffts);
        output.store_complex(scratch[3], idx + 3 * num_ffts);
        output.store_complex(scratchb[3], idx + 2 + 3 * num_ffts);

        idx += 4;
    }
}

pub struct Neon64Radix4<T> {
    _phantom: std::marker::PhantomData<T>,
    twiddles: Box<[float64x2_t]>,

    base_fft: Neon64Butterfly<T>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
    bf4: NeonF64Butterfly4<T>,
}

impl<T: FftNum> Neon64Radix4<T> {
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
            0 => (
                len,
                Neon64Butterfly::Len1(NeonF64Butterfly1::new(direction)),
            ),
            1 => (
                len,
                Neon64Butterfly::Len2(NeonF64Butterfly2::new(direction)),
            ),
            2 => (
                len,
                Neon64Butterfly::Len4(NeonF64Butterfly4::new(direction)),
            ),
            3 => (
                len,
                Neon64Butterfly::Len8(NeonF64Butterfly8::new(direction)),
            ),
            _ => {
                if num_bits % 2 == 1 {
                    if len < USE_BUTTERFLY32_FROM {
                        (8, Neon64Butterfly::Len8(NeonF64Butterfly8::new(direction)))
                    } else {
                        (
                            32,
                            Neon64Butterfly::Len32(NeonF64Butterfly32::new(direction)),
                        )
                    }
                } else {
                    (
                        16,
                        Neon64Butterfly::Len16(NeonF64Butterfly16::new(direction)),
                    )
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
                    let twiddle =
                        twiddles::compute_twiddle::<f64>(i * k * twiddle_stride, len, direction);
                    let twiddle_packed = unsafe { RawSlice::new(&[twiddle]).load_complex(0) };
                    twiddle_factors.push(twiddle_packed);
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
            bf4: NeonF64Butterfly4::<T>::new(direction),
        }
    }

    //#[target_feature(enable = "neon")]
    unsafe fn perform_fft_out_of_place(
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
        match &self.base_fft {
            Neon64Butterfly::Len1(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon64Butterfly::Len2(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon64Butterfly::Len4(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon64Butterfly::Len8(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon64Butterfly::Len16(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
            Neon64Butterfly::Len32(bf) => bf.perform_fft_butterfly_multi(spectrum).unwrap(),
        }

        // cross-FFTs
        let mut current_size = self.base_len * 4;
        let mut layer_twiddles: &[float64x2_t] = &self.twiddles;

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
boilerplate_fft_neon_oop!(Neon64Radix4, |this: &Neon64Radix4<_>| this.len);

//#[target_feature(enable = "neon")]
unsafe fn butterfly_4_64<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[float64x2_t],
    num_ffts: usize,
    bf4: &NeonF64Butterfly4<T>,
) {
    let mut idx = 0usize;
    let input: RawSlice<Complex<f64>> = RawSlice::new_transmuted(data);
    let output: RawSliceMut<Complex<f64>> = RawSliceMut::new_transmuted(data);
    for tw in twiddles.chunks_exact(6).take(num_ffts / 2) {
        let scratch0 = input.load_complex(idx);
        let scratch0b = input.load_complex(idx + 1);
        let mut scratch1 = input.load_complex(idx + 1 * num_ffts);
        let mut scratch1b = input.load_complex(idx + 1 + 1 * num_ffts);
        let mut scratch2 = input.load_complex(idx + 2 * num_ffts);
        let mut scratch2b = input.load_complex(idx + 1 + 2 * num_ffts);
        let mut scratch3 = input.load_complex(idx + 3 * num_ffts);
        let mut scratch3b = input.load_complex(idx + 1 + 3 * num_ffts);

        scratch1 = mul_complex_f64(scratch1, tw[0]);
        scratch2 = mul_complex_f64(scratch2, tw[1]);
        scratch3 = mul_complex_f64(scratch3, tw[2]);
        scratch1b = mul_complex_f64(scratch1b, tw[3]);
        scratch2b = mul_complex_f64(scratch2b, tw[4]);
        scratch3b = mul_complex_f64(scratch3b, tw[5]);

        let scratch = bf4.perform_fft_direct(scratch0, scratch1, scratch2, scratch3);
        let scratchb = bf4.perform_fft_direct(scratch0b, scratch1b, scratch2b, scratch3b);

        output.store_complex(scratch[0], idx);
        output.store_complex(scratchb[0], idx + 1);
        output.store_complex(scratch[1], idx + 1 * num_ffts);
        output.store_complex(scratchb[1], idx + 1 + 1 * num_ffts);
        output.store_complex(scratch[2], idx + 2 * num_ffts);
        output.store_complex(scratchb[2], idx + 1 + 2 * num_ffts);
        output.store_complex(scratch[3], idx + 3 * num_ffts);
        output.store_complex(scratchb[3], idx + 1 + 3 * num_ffts);

        idx += 2;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    #[test]
    fn test_neon_radix4_64() {
        for pow in 4..12 {
            let len = 1 << pow;
            test_neon_radix4_64_with_length(len, FftDirection::Forward);
            test_neon_radix4_64_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_neon_radix4_64_with_length(len: usize, direction: FftDirection) {
        let fft = Neon64Radix4::new(len, direction);
        check_fft_algorithm::<f64>(&fft, len, direction);
    }

    #[test]
    fn test_neon_radix4_32() {
        for pow in 0..12 {
            let len = 1 << pow;
            test_neon_radix4_32_with_length(len, FftDirection::Forward);
            test_neon_radix4_32_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_neon_radix4_32_with_length(len: usize, direction: FftDirection) {
        let fft = Neon32Radix4::new(len, direction);
        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
