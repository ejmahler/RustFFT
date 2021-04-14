use core::arch::x86_64::*;
use num_complex::Complex;

use crate::{common::FftNum, FftDirection};

use crate::array_utils;
use crate::array_utils::{RawSlice, RawSliceMut};
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::twiddles;
use crate::{Direction, Fft, Length};

use super::sse_common::{assert_f32, assert_f64};
use super::sse_utils::*;
use super::sse_vector::{SseArray, SseArrayMut};

#[allow(unused)]
macro_rules! boilerplate_fft_sse_f32_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            #[target_feature(enable = "sse4.1")]
            //#[inline(always)]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(
                    RawSlice::new_transmuted(buffer),
                    RawSliceMut::new_transmuted(buffer),
                );
            }

            #[target_feature(enable = "sse4.1")]
            //#[inline(always)]
            pub(crate) unsafe fn perform_parallel_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_parallel_fft_contiguous(
                    RawSlice::new_transmuted(buffer),
                    RawSliceMut::new_transmuted(buffer),
                );
            }

            // Do multiple ffts over a longer vector inplace, called from "process_with_scratch" of Fft trait
            #[target_feature(enable = "sse4.1")]
            pub(crate) unsafe fn perform_fft_butterfly_multi(
                &self,
                buffer: &mut [Complex<T>],
            ) -> Result<(), ()> {
                let len = buffer.len();
                let alldone = array_utils::iter_chunks(buffer, 2 * self.len(), |chunk| {
                    self.perform_parallel_fft_butterfly(chunk)
                });
                if alldone.is_err() && buffer.len() >= self.len() {
                    self.perform_fft_butterfly(&mut buffer[len - self.len()..]);
                }
                Ok(())
            }

            // Do multiple ffts over a longer vector outofplace, called from "process_outofplace_with_scratch" of Fft trait
            #[target_feature(enable = "sse4.1")]
            pub(crate) unsafe fn perform_oop_fft_butterfly_multi(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
            ) -> Result<(), ()> {
                let len = input.len();
                let alldone = array_utils::iter_chunks_zipped(
                    input,
                    output,
                    2 * self.len(),
                    |in_chunk, out_chunk| {
                        self.perform_parallel_fft_contiguous(
                            RawSlice::new_transmuted(in_chunk),
                            RawSliceMut::new_transmuted(out_chunk),
                        )
                    },
                );
                if alldone.is_err() && input.len() >= self.len() {
                    self.perform_fft_contiguous(
                        RawSlice::new_transmuted(&input[len - self.len()..]),
                        RawSliceMut::new_transmuted(&mut output[len - self.len()..]),
                    );
                }
                Ok(())
            }
        }
    };
}

macro_rules! boilerplate_fft_sse_f64_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            // Do a single fft
            #[target_feature(enable = "sse4.1")]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(
                    RawSlice::new_transmuted(buffer),
                    RawSliceMut::new_transmuted(buffer),
                );
            }

            // Do multiple ffts over a longer vector inplace, called from "process_with_scratch" of Fft trait
            #[target_feature(enable = "sse4.1")]
            pub(crate) unsafe fn perform_fft_butterfly_multi(
                &self,
                buffer: &mut [Complex<T>],
            ) -> Result<(), ()> {
                array_utils::iter_chunks(buffer, self.len(), |chunk| {
                    self.perform_fft_butterfly(chunk)
                })
            }

            // Do multiple ffts over a longer vector outofplace, called from "process_outofplace_with_scratch" of Fft trait
            #[target_feature(enable = "sse4.1")]
            pub(crate) unsafe fn perform_oop_fft_butterfly_multi(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
            ) -> Result<(), ()> {
                array_utils::iter_chunks_zipped(input, output, self.len(), |in_chunk, out_chunk| {
                    self.perform_fft_contiguous(
                        RawSlice::new_transmuted(in_chunk),
                        RawSliceMut::new_transmuted(out_chunk),
                    )
                })
            }
        }
    };
}

#[allow(unused)]
macro_rules! boilerplate_fft_sse_common_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> Fft<T> for $struct_name<T> {
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                if input.len() < self.len() || output.len() != input.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                    return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
                }
                let result = unsafe { self.perform_oop_fft_butterfly_multi(input, output) };

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                }
            }
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                if buffer.len() < self.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_inplace(self.len(), buffer.len(), 0, 0);
                    return; // Unreachable, because fft_error_inplace asserts, but it helps codegen to put it here
                }

                let result = unsafe { self.perform_fft_butterfly_multi(buffer) };

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_inplace(self.len(), buffer.len(), 0, 0);
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                0
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len
            }
        }
        impl<T> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                $direction_fn(self)
            }
        }
    };
}

//   _            _________  _     _ _
//  / |          |___ /___ \| |__ (_) |_
//  | |   _____    |_ \ __) | '_ \| | __|
//  | |  |_____|  ___) / __/| |_) | | |_
//  |_|          |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly1<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly1, 1, |this: &SseF32Butterfly1<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly1, 1, |this: &SseF32Butterfly1<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly1<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        _input: RawSlice<Complex<f32>>,
        _output: RawSliceMut<Complex<f32>>,
    ) {
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        _input: RawSlice<Complex<f32>>,
        _output: RawSliceMut<Complex<f32>>,
    ) {
    }
}

//   _             __   _  _   _     _ _
//  / |           / /_ | || | | |__ (_) |_
//  | |   _____  | '_ \| || |_| '_ \| | __|
//  | |  |_____| | (_) |__   _| |_) | | |_
//  |_|           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly1<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly1, 1, |this: &SseF64Butterfly1<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly1, 1, |this: &SseF64Butterfly1<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly1<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        _input: RawSlice<Complex<f64>>,
        _output: RawSliceMut<Complex<f64>>,
    ) {
    }
}

//   ____            _________  _     _ _
//  |___ \          |___ /___ \| |__ (_) |_
//    __) |  _____    |_ \ __) | '_ \| | __|
//   / __/  |_____|  ___) / __/| |_) | | |_
//  |_____|         |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly2<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly2, 2, |this: &SseF32Butterfly2<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly2, 2, |this: &SseF32Butterfly2<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly2<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let values = input.load_complex(0);

        let temp = self.perform_fft_direct(values);

        output.store_complex(temp, 0);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let values_a = input.load_complex(0);
        let values_b = input.load_complex(2);

        let out = self.perform_parallel_fft_direct(values_a, values_b);

        let [out02, out13] = transpose_complex_2x2_f32(out[0], out[1]);

        output.store_complex(out02, 0);
        output.store_complex(out13, 2);
    }

    // length 2 fft of x, given as [x0, x1]
    // result is [X0, X1]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: __m128) -> __m128 {
        solo_fft2_f32(values)
    }

    // dual length 2 fft of x and y, given as [x0, x1], [y0, y1]
    // result is [X0, Y0], [X1, Y1]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        values_x: __m128,
        values_y: __m128,
    ) -> [__m128; 2] {
        parallel_fft2_contiguous_f32(values_x, values_y)
    }
}

// double lenth 2 fft of a and b, given as [x0, y0], [x1, y1]
// result is [X0, Y0], [X1, Y1]
#[inline(always)]
pub(crate) unsafe fn parallel_fft2_interleaved_f32(val02: __m128, val13: __m128) -> [__m128; 2] {
    let temp0 = _mm_add_ps(val02, val13);
    let temp1 = _mm_sub_ps(val02, val13);
    [temp0, temp1]
}

// double lenth 2 fft of a and b, given as [x0, x1], [y0, y1]
// result is [X0, Y0], [X1, Y1]
#[inline(always)]
unsafe fn parallel_fft2_contiguous_f32(left: __m128, right: __m128) -> [__m128; 2] {
    let [temp02, temp13] = transpose_complex_2x2_f32(left, right);
    parallel_fft2_interleaved_f32(temp02, temp13)
}

// length 2 fft of x, given as [x0, x1]
// result is [X0, X1]
#[inline(always)]
unsafe fn solo_fft2_f32(values: __m128) -> __m128 {
    let temp = reverse_complex_elements_f32(values);
    let temp2 = negate_hi_f32(values);
    _mm_add_ps(temp2, temp)
}

//   ____             __   _  _   _     _ _
//  |___ \           / /_ | || | | |__ (_) |_
//    __) |  _____  | '_ \| || |_| '_ \| | __|
//   / __/  |_____| | (_) |__   _| |_) | | |_
//  |_____|          \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly2<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly2, 2, |this: &SseF64Butterfly2<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly2, 2, |this: &SseF64Butterfly2<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly2<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let value0 = input.load_complex(0);
        let value1 = input.load_complex(1);

        let out = self.perform_fft_direct(value0, value1);

        output.store_complex(out[0], 0);
        output.store_complex(out[1], 1);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: __m128d,
        value1: __m128d,
    ) -> [__m128d; 2] {
        solo_fft2_f64(value0, value1)
    }
}

#[inline(always)]
pub(crate) unsafe fn solo_fft2_f64(left: __m128d, right: __m128d) -> [__m128d; 2] {
    let temp0 = _mm_add_pd(left, right);
    let temp1 = _mm_sub_pd(left, right);
    [temp0, temp1]
}

//   _____            _________  _     _ _
//  |___ /           |___ /___ \| |__ (_) |_
//    |_ \    _____    |_ \ __) | '_ \| | __|
//   ___) |  |_____|  ___) / __/| |_) | | |_
//  |____/           |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly3<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle: __m128,
    twiddle1re: __m128,
    twiddle1im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly3, 3, |this: &SseF32Butterfly3<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly3, 3, |this: &SseF32Butterfly3<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly3<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 3, direction);
        let twiddle = unsafe { _mm_set_ps(-tw1.im, -tw1.im, tw1.re, tw1.re) };
        let twiddle1re = unsafe { _mm_set_ps(tw1.re, tw1.re, tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_ps(tw1.im, tw1.im, tw1.im, tw1.im) };
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle,
            twiddle1re,
            twiddle1im,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let value0x = input.load_partial1_complex(0);
        let value12 = input.load_complex(1);

        let out = self.perform_fft_direct(value0x, value12);

        output.store_partial_lo_complex(out[0], 0);
        output.store_complex(out[1], 1);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let valuea0a1 = input.load_complex(0);
        let valuea2b0 = input.load_complex(2);
        let valueb1b2 = input.load_complex(4);

        let value0 = extract_lo_hi_f32(valuea0a1, valuea2b0);
        let value1 = extract_hi_lo_f32(valuea0a1, valueb1b2);
        let value2 = extract_lo_hi_f32(valuea2b0, valueb1b2);

        let out = self.perform_parallel_fft_direct(value0, value1, value2);

        let out0 = extract_lo_lo_f32(out[0], out[1]);
        let out1 = extract_lo_hi_f32(out[2], out[0]);
        let out2 = extract_hi_hi_f32(out[1], out[2]);

        output.store_complex(out0, 0);
        output.store_complex(out1, 2);
        output.store_complex(out2, 4);
    }

    // length 3 fft of a, given as [x0, 0.0], [x1, x2]
    // result is [X0, Z], [X1, X2]
    // The value Z should be discarded.
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0x: __m128,
        value12: __m128,
    ) -> [__m128; 2] {
        // This is a SSE translation of the scalar 3-point butterfly
        let rev12 = negate_hi_f32(reverse_complex_elements_f32(value12));
        let temp12pn = self.rotate.rotate_hi(_mm_add_ps(value12, rev12));
        let twiddled = _mm_mul_ps(temp12pn, self.twiddle);
        let temp = _mm_add_ps(value0x, twiddled);
        let out12 = solo_fft2_f32(temp);
        let out0x = _mm_add_ps(value0x, temp12pn);
        [out0x, out12]
    }

    // length 3 dual fft of a, given as (x0, y0), (x1, y1), (x2, y2).
    // result is [(X0, Y0), (X1, Y1), (X2, Y2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        value0: __m128,
        value1: __m128,
        value2: __m128,
    ) -> [__m128; 3] {
        // This is a SSE translation of the scalar 3-point butterfly
        let x12p = _mm_add_ps(value1, value2);
        let x12n = _mm_sub_ps(value1, value2);
        let sum = _mm_add_ps(value0, x12p);

        let temp_a = _mm_mul_ps(self.twiddle1re, x12p);
        let temp_a = _mm_add_ps(temp_a, value0);

        let n_rot = self.rotate.rotate_both(x12n);
        let temp_b = _mm_mul_ps(self.twiddle1im, n_rot);

        let x1 = _mm_add_ps(temp_a, temp_b);
        let x2 = _mm_sub_ps(temp_a, temp_b);
        [sum, x1, x2]
    }
}

//   _____             __   _  _   _     _ _
//  |___ /            / /_ | || | | |__ (_) |_
//    |_ \    _____  | '_ \| || |_| '_ \| | __|
//   ___) |  |_____| | (_) |__   _| |_) | | |_
//  |____/            \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly3<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly3, 3, |this: &SseF64Butterfly3<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly3, 3, |this: &SseF64Butterfly3<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly3<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 3, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let value0 = input.load_complex(0);
        let value1 = input.load_complex(1);
        let value2 = input.load_complex(2);

        let out = self.perform_fft_direct(value0, value1, value2);

        output.store_complex(out[0], 0);
        output.store_complex(out[1], 1);
        output.store_complex(out[2], 2);
    }

    // length 3 fft of x, given as x0, x1, x2.
    // result is [X0, X1, X2]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: __m128d,
        value1: __m128d,
        value2: __m128d,
    ) -> [__m128d; 3] {
        // This is a SSE translation of the scalar 3-point butterfly
        let x12p = _mm_add_pd(value1, value2);
        let x12n = _mm_sub_pd(value1, value2);
        let sum = _mm_add_pd(value0, x12p);

        let temp_a = _mm_mul_pd(self.twiddle1re, x12p);
        let temp_a = _mm_add_pd(temp_a, value0);

        let n_rot = self.rotate.rotate(x12n);
        let temp_b = _mm_mul_pd(self.twiddle1im, n_rot);

        let x1 = _mm_add_pd(temp_a, temp_b);
        let x2 = _mm_sub_pd(temp_a, temp_b);
        [sum, x1, x2]
    }
}

//   _  _             _________  _     _ _
//  | || |           |___ /___ \| |__ (_) |_
//  | || |_   _____    |_ \ __) | '_ \| | __|
//  |__   _| |_____|  ___) / __/| |_) | | |_
//     |_|           |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly4, 4, |this: &SseF32Butterfly4<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly4, 4, |this: &SseF32Butterfly4<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly4<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = if direction == FftDirection::Inverse {
            Rotate90F32::new(true)
        } else {
            Rotate90F32::new(false)
        };
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let value01 = input.load_complex(0);
        let value23 = input.load_complex(2);

        let out = self.perform_fft_direct(value01, value23);

        output.store_complex(out[0], 0);
        output.store_complex(out[1], 2);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let value01a = input.load_complex(0);
        let value23a = input.load_complex(2);
        let value01b = input.load_complex(4);
        let value23b = input.load_complex(6);

        let [value0ab, value1ab] = transpose_complex_2x2_f32(value01a, value01b);
        let [value2ab, value3ab] = transpose_complex_2x2_f32(value23a, value23b);

        let out = self.perform_parallel_fft_direct(value0ab, value1ab, value2ab, value3ab);

        let [out0, out1] = transpose_complex_2x2_f32(out[0], out[1]);
        let [out2, out3] = transpose_complex_2x2_f32(out[2], out[3]);

        output.store_complex(out0, 0);
        output.store_complex(out1, 4);
        output.store_complex(out2, 2);
        output.store_complex(out3, 6);
    }

    // length 4 fft of a, given as [x0, x1], [x2, x3]
    // result is [[X0, X1], [X2, X3]]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value01: __m128,
        value23: __m128,
    ) -> [__m128; 2] {
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose
        // and
        // step 2: column FFTs
        let mut temp = parallel_fft2_interleaved_f32(value01, value23);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        temp[1] = self.rotate.rotate_hi(temp[1]);

        // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

        // step 5: row FFTs
        // and
        // step 6: transpose by swapping index 1 and 2
        parallel_fft2_contiguous_f32(temp[0], temp[1])
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        values0: __m128,
        values1: __m128,
        values2: __m128,
        values3: __m128,
    ) -> [__m128; 4] {
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose
        // and
        // step 2: column FFTs
        let temp0 = parallel_fft2_interleaved_f32(values0, values2);
        let mut temp1 = parallel_fft2_interleaved_f32(values1, values3);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        temp1[1] = self.rotate.rotate_both(temp1[1]);

        // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

        // step 5: row FFTs
        let out0 = parallel_fft2_interleaved_f32(temp0[0], temp1[0]);
        let out2 = parallel_fft2_interleaved_f32(temp0[1], temp1[1]);

        // step 6: transpose by swapping index 1 and 2
        [out0[0], out2[0], out0[1], out2[1]]
    }
}

//   _  _              __   _  _   _     _ _
//  | || |            / /_ | || | | |__ (_) |_
//  | || |_   _____  | '_ \| || |_| '_ \| | __|
//  |__   _| |_____| | (_) |__   _| |_) | | |_
//     |_|            \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly4, 4, |this: &SseF64Butterfly4<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly4, 4, |this: &SseF64Butterfly4<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly4<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = if direction == FftDirection::Inverse {
            Rotate90F64::new(true)
        } else {
            Rotate90F64::new(false)
        };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let value0 = input.load_complex(0);
        let value1 = input.load_complex(1);
        let value2 = input.load_complex(2);
        let value3 = input.load_complex(3);

        let out = self.perform_fft_direct(value0, value1, value2, value3);

        output.store_complex(out[0], 0);
        output.store_complex(out[1], 1);
        output.store_complex(out[2], 2);
        output.store_complex(out[3], 3);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: __m128d,
        value1: __m128d,
        value2: __m128d,
        value3: __m128d,
    ) -> [__m128d; 4] {
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose
        // and
        // step 2: column FFTs
        let temp0 = solo_fft2_f64(value0, value2);
        let mut temp1 = solo_fft2_f64(value1, value3);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        temp1[1] = self.rotate.rotate(temp1[1]);

        // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

        // step 5: row FFTs
        let out0 = solo_fft2_f64(temp0[0], temp1[0]);
        let out2 = solo_fft2_f64(temp0[1], temp1[1]);

        // step 6: transpose by swapping index 1 and 2
        [out0[0], out2[0], out0[1], out2[1]]
    }
}

//   ____             _________  _     _ _
//  | ___|           |___ /___ \| |__ (_) |_
//  |___ \    _____    |_ \ __) | '_ \| | __|
//   ___) |  |_____|  ___) / __/| |_) | | |_
//  |____/           |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly5<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle12re: __m128,
    twiddle21re: __m128,
    twiddle12im: __m128,
    twiddle21im: __m128,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly5, 5, |this: &SseF32Butterfly5<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly5, 5, |this: &SseF32Butterfly5<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly5<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 5, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 5, direction);
        let twiddle12re = unsafe { _mm_set_ps(tw2.re, tw2.re, tw1.re, tw1.re) };
        let twiddle21re = unsafe { _mm_set_ps(tw1.re, tw1.re, tw2.re, tw2.re) };
        let twiddle12im = unsafe { _mm_set_ps(tw2.im, tw2.im, tw1.im, tw1.im) };
        let twiddle21im = unsafe { _mm_set_ps(-tw1.im, -tw1.im, tw2.im, tw2.im) };
        let twiddle1re = unsafe { _mm_set_ps(tw1.re, tw1.re, tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_ps(tw1.im, tw1.im, tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_ps(tw2.re, tw2.re, tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_ps(tw2.im, tw2.im, tw2.im, tw2.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle12re,
            twiddle21re,
            twiddle12im,
            twiddle21im,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let value00 = input.load1_complex(0);
        let value12 = input.load_complex(1);
        let value34 = input.load_complex(3);

        let out = self.perform_fft_direct(value00, value12, value34);

        output.store_partial_lo_complex(out[0], 0);
        output.store_complex(out[1], 1);
        output.store_complex(out[2], 3);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4 ,6, 8});

        let value0 = extract_lo_hi_f32(input_packed[0], input_packed[2]);
        let value1 = extract_hi_lo_f32(input_packed[0], input_packed[3]);
        let value2 = extract_lo_hi_f32(input_packed[1], input_packed[3]);
        let value3 = extract_hi_lo_f32(input_packed[1], input_packed[4]);
        let value4 = extract_lo_hi_f32(input_packed[2], input_packed[4]);

        let out = self.perform_parallel_fft_direct(value0, value1, value2, value3, value4);

        let out_packed = [
            extract_lo_lo_f32(out[0], out[1]),
            extract_lo_lo_f32(out[2], out[3]),
            extract_lo_hi_f32(out[4], out[0]),
            extract_hi_hi_f32(out[1], out[2]),
            extract_hi_hi_f32(out[3], out[4]),
        ];

        write_complex_to_array_strided!(out_packed, output, 2, {0, 1, 2, 3, 4});
    }

    // length 5 fft of a, given as [x0, x0], [x1, x2], [x3, x4].
    // result is [[X0, Z], [X1, X2], [X3, X4]]
    // Note that Z should not be used.
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value00: __m128,
        value12: __m128,
        value34: __m128,
    ) -> [__m128; 3] {
        // This is a SSE translation of the scalar 5-point butterfly
        let temp43 = reverse_complex_elements_f32(value34);
        let x1423p = _mm_add_ps(value12, temp43);
        let x1423n = _mm_sub_ps(value12, temp43);

        let x1414p = duplicate_lo_f32(x1423p);
        let x2323p = duplicate_hi_f32(x1423p);
        let x1414n = duplicate_lo_f32(x1423n);
        let x2323n = duplicate_hi_f32(x1423n);

        let temp_a1 = _mm_mul_ps(self.twiddle12re, x1414p);
        let temp_a2 = _mm_mul_ps(self.twiddle21re, x2323p);

        let temp_b1 = _mm_mul_ps(self.twiddle12im, x1414n);
        let temp_b2 = _mm_mul_ps(self.twiddle21im, x2323n);

        let temp_a = _mm_add_ps(temp_a1, temp_a2);
        let temp_a = _mm_add_ps(value00, temp_a);

        let temp_b = _mm_add_ps(temp_b1, temp_b2);

        let b_rot = self.rotate.rotate_both(temp_b);

        let x00 = _mm_add_ps(value00, _mm_add_ps(x1414p, x2323p));

        let x12 = _mm_add_ps(temp_a, b_rot);
        let x34 = reverse_complex_elements_f32(_mm_sub_ps(temp_a, b_rot));
        [x00, x12, x34]
    }

    // length 5 dual fft of x and y, given as (x0, y0), (x1, y1) ... (x4, y4).
    // result is [(X0, Y0), (X1, Y1) ... (X2, Y2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        value0: __m128,
        value1: __m128,
        value2: __m128,
        value3: __m128,
        value4: __m128,
    ) -> [__m128; 5] {
        // This is a SSE translation of the scalar 3-point butterfly
        let x14p = _mm_add_ps(value1, value4);
        let x14n = _mm_sub_ps(value1, value4);
        let x23p = _mm_add_ps(value2, value3);
        let x23n = _mm_sub_ps(value2, value3);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x14p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x23p);
        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x14n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x23n);
        let temp_a2_1 = _mm_mul_ps(self.twiddle1re, x23p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle2re, x14p);
        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x14n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle1im, x23n);

        let temp_a1 = _mm_add_ps(value0, _mm_add_ps(temp_a1_1, temp_a1_2));
        let temp_b1 = _mm_add_ps(temp_b1_1, temp_b1_2);
        let temp_a2 = _mm_add_ps(value0, _mm_add_ps(temp_a2_1, temp_a2_2));
        let temp_b2 = _mm_sub_ps(temp_b2_1, temp_b2_2);

        [
            _mm_add_ps(value0, _mm_add_ps(x14p, x23p)),
            _mm_add_ps(temp_a1, self.rotate.rotate_both(temp_b1)),
            _mm_add_ps(temp_a2, self.rotate.rotate_both(temp_b2)),
            _mm_sub_ps(temp_a2, self.rotate.rotate_both(temp_b2)),
            _mm_sub_ps(temp_a1, self.rotate.rotate_both(temp_b1)),
        ]
    }
}

//   ____              __   _  _   _     _ _
//  | ___|            / /_ | || | | |__ (_) |_
//  |___ \    _____  | '_ \| || |_| '_ \| | __|
//   ___) |  |_____| | (_) |__   _| |_) | | |_
//  |____/            \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly5<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly5, 5, |this: &SseF64Butterfly5<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly5, 5, |this: &SseF64Butterfly5<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly5<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 5, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 5, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let value0 = input.load_complex(0);
        let value1 = input.load_complex(1);
        let value2 = input.load_complex(2);
        let value3 = input.load_complex(3);
        let value4 = input.load_complex(4);

        let out = self.perform_fft_direct(value0, value1, value2, value3, value4);

        output.store_complex(out[0], 0);
        output.store_complex(out[1], 1);
        output.store_complex(out[2], 2);
        output.store_complex(out[3], 3);
        output.store_complex(out[4], 4);
    }

    // length 5 fft of x, given as x0, x1, x2, x3, x4.
    // result is [X0, X1, X2, X3, X4]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: __m128d,
        value1: __m128d,
        value2: __m128d,
        value3: __m128d,
        value4: __m128d,
    ) -> [__m128d; 5] {
        // This is a SSE translation of the scalar 5-point butterfly
        let x14p = _mm_add_pd(value1, value4);
        let x14n = _mm_sub_pd(value1, value4);
        let x23p = _mm_add_pd(value2, value3);
        let x23n = _mm_sub_pd(value2, value3);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x14p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x23p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x14p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle1re, x23p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x14n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x23n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x14n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle1im, x23n);

        let temp_a1 = _mm_add_pd(value0, _mm_add_pd(temp_a1_1, temp_a1_2));
        let temp_a2 = _mm_add_pd(value0, _mm_add_pd(temp_a2_1, temp_a2_2));

        let temp_b1 = _mm_add_pd(temp_b1_1, temp_b1_2);
        let temp_b2 = _mm_sub_pd(temp_b2_1, temp_b2_2);

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        [
            _mm_add_pd(value0, _mm_add_pd(x14p, x23p)),
            _mm_add_pd(temp_a1, temp_b1_rot),
            _mm_add_pd(temp_a2, temp_b2_rot),
            _mm_sub_pd(temp_a2, temp_b2_rot),
            _mm_sub_pd(temp_a1, temp_b1_rot),
        ]
    }
}

//    __             _________  _     _ _
//   / /_           |___ /___ \| |__ (_) |_
//  | '_ \   _____    |_ \ __) | '_ \| | __|
//  | (_) | |_____|  ___) / __/| |_) | | |_
//   \___/          |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly6<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: SseF32Butterfly3<T>,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly6, 6, |this: &SseF32Butterfly6<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly6, 6, |this: &SseF32Butterfly6<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = SseF32Butterfly3::new(direction);

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let value01 = input.load_complex(0);
        let value23 = input.load_complex(2);
        let value45 = input.load_complex(4);

        let out = self.perform_fft_direct(value01, value23, value45);

        output.store_complex(out[0], 0);
        output.store_complex(out[1], 2);
        output.store_complex(out[2], 4);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input,  {0, 2, 4, 6, 8, 10});

        let values = interleave_complex_f32!(input_packed, 3, {0, 1, 2});

        let out = self.perform_parallel_fft_direct(
            values[0], values[1], values[2], values[3], values[4], values[5],
        );

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4});
        write_complex_to_array_strided!(out_sorted, output, 2, {0, 1, 2, 3, 4, 5});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value01: __m128,
        value23: __m128,
        value45: __m128,
    ) -> [__m128; 3] {
        // Algorithm: 3x2 good-thomas

        // Size-3 FFTs down the columns of our reordered array
        let reord0 = extract_lo_hi_f32(value01, value23);
        let reord1 = extract_lo_hi_f32(value23, value45);
        let reord2 = extract_lo_hi_f32(value45, value01);

        let mid = self.bf3.perform_parallel_fft_direct(reord0, reord1, reord2);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = parallel_fft2_contiguous_f32(mid[0], mid[1]);
        let output2 = solo_fft2_f32(mid[2]);

        // Reorder into output
        [
            extract_lo_hi_f32(output0, output1),
            extract_lo_lo_f32(output2, output1),
            extract_hi_hi_f32(output0, output2),
        ]
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        value0: __m128,
        value1: __m128,
        value2: __m128,
        value3: __m128,
        value4: __m128,
        value5: __m128,
    ) -> [__m128; 6] {
        // Algorithm: 3x2 good-thomas

        // Size-3 FFTs down the columns of our reordered array
        let mid0 = self.bf3.perform_parallel_fft_direct(value0, value2, value4);
        let mid1 = self.bf3.perform_parallel_fft_direct(value3, value5, value1);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = parallel_fft2_interleaved_f32(mid0[0], mid1[0]);
        let [output2, output3] = parallel_fft2_interleaved_f32(mid0[1], mid1[1]);
        let [output4, output5] = parallel_fft2_interleaved_f32(mid0[2], mid1[2]);

        // Reorder into output
        [output0, output3, output4, output1, output2, output5]
    }
}

//    __              __   _  _   _     _ _
//   / /_            / /_ | || | | |__ (_) |_
//  | '_ \   _____  | '_ \| || |_| '_ \| | __|
//  | (_) | |_____| | (_) |__   _| |_) | | |_
//   \___/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly6<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: SseF64Butterfly3<T>,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly6, 6, |this: &SseF64Butterfly6<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly6, 6, |this: &SseF64Butterfly6<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = SseF64Butterfly3::new(direction);

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let value0 = input.load_complex(0);
        let value1 = input.load_complex(1);
        let value2 = input.load_complex(2);
        let value3 = input.load_complex(3);
        let value4 = input.load_complex(4);
        let value5 = input.load_complex(5);

        let out = self.perform_fft_direct(value0, value1, value2, value3, value4, value5);

        output.store_complex(out[0], 0);
        output.store_complex(out[1], 1);
        output.store_complex(out[2], 2);
        output.store_complex(out[3], 3);
        output.store_complex(out[4], 4);
        output.store_complex(out[5], 5);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: __m128d,
        value1: __m128d,
        value2: __m128d,
        value3: __m128d,
        value4: __m128d,
        value5: __m128d,
    ) -> [__m128d; 6] {
        // Algorithm: 3x2 good-thomas

        // Size-3 FFTs down the columns of our reordered array
        let mid0 = self.bf3.perform_fft_direct(value0, value2, value4);
        let mid1 = self.bf3.perform_fft_direct(value3, value5, value1);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = solo_fft2_f64(mid0[0], mid1[0]);
        let [output2, output3] = solo_fft2_f64(mid0[1], mid1[1]);
        let [output4, output5] = solo_fft2_f64(mid0[2], mid1[2]);

        // Reorder into output
        [output0, output3, output4, output1, output2, output5]
    }
}

//    ___            _________  _     _ _
//   ( _ )          |___ /___ \| |__ (_) |_
//   / _ \   _____    |_ \ __) | '_ \| | __|
//  | (_) | |_____|  ___) / __/| |_) | | |_
//   \___/          |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly8<T> {
    root2: __m128,
    root2_dual: __m128,
    direction: FftDirection,
    bf4: SseF32Butterfly4<T>,
    rotate90: Rotate90F32,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly8, 8, |this: &SseF32Butterfly8<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly8, 8, |this: &SseF32Butterfly8<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf4 = SseF32Butterfly4::new(direction);
        let root2 = unsafe { _mm_set_ps(0.5f32.sqrt(), 0.5f32.sqrt(), 1.0, 1.0) };
        let root2_dual = unsafe { _mm_load1_ps(&0.5f32.sqrt()) };
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F32::new(true)
        } else {
            Rotate90F32::new(false)
        };
        Self {
            root2,
            root2_dual,
            direction,
            bf4,
            rotate90,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6});

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, output, 2, {0,1,2,3});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14});

        let values = interleave_complex_f32!(input_packed, 4, {0, 1, 2, 3});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6});

        write_complex_to_array_strided!(out_sorted, output, 2, {0,1,2,3,4,5,6,7});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, values: [__m128; 4]) -> [__m128; 4] {
        // we're going to hardcode a step of mixed radix
        // step 1: copy and reorder the input into the scratch
        let [in02, in13] = transpose_complex_2x2_f32(values[0], values[1]);
        let [in46, in57] = transpose_complex_2x2_f32(values[2], values[3]);

        // step 2: column FFTs
        let val0 = self.bf4.perform_fft_direct(in02, in46);
        let mut val2 = self.bf4.perform_fft_direct(in13, in57);

        // step 3: apply twiddle factors
        let val2b = self.rotate90.rotate_hi(val2[0]);
        let val2c = _mm_add_ps(val2b, val2[0]);
        let val2d = _mm_mul_ps(val2c, self.root2);
        val2[0] = extract_lo_hi_f32(val2[0], val2d);

        let val3b = self.rotate90.rotate_both(val2[1]);
        let val3c = _mm_sub_ps(val3b, val2[1]);
        let val3d = _mm_mul_ps(val3c, self.root2);
        val2[1] = extract_lo_hi_f32(val3b, val3d);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        let out0 = parallel_fft2_interleaved_f32(val0[0], val2[0]);
        let out1 = parallel_fft2_interleaved_f32(val0[1], val2[1]);

        // step 6: rearrange and copy to buffer
        [out0[0], out1[0], out0[1], out1[1]]
    }

    #[inline(always)]
    unsafe fn perform_parallel_fft_direct(&self, values: [__m128; 8]) -> [__m128; 8] {
        // we're going to hardcode a step of mixed radix
        // step 1: copy and reorder the input into the scratch
        // and
        // step 2: column FFTs
        let val03 = self
            .bf4
            .perform_parallel_fft_direct(values[0], values[2], values[4], values[6]);
        let mut val47 = self
            .bf4
            .perform_parallel_fft_direct(values[1], values[3], values[5], values[7]);

        // step 3: apply twiddle factors
        let val5b = self.rotate90.rotate_both(val47[1]);
        let val5c = _mm_add_ps(val5b, val47[1]);
        val47[1] = _mm_mul_ps(val5c, self.root2_dual);
        val47[2] = self.rotate90.rotate_both(val47[2]);
        let val7b = self.rotate90.rotate_both(val47[3]);
        let val7c = _mm_sub_ps(val7b, val47[3]);
        val47[3] = _mm_mul_ps(val7c, self.root2_dual);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        let out0 = parallel_fft2_interleaved_f32(val03[0], val47[0]);
        let out1 = parallel_fft2_interleaved_f32(val03[1], val47[1]);
        let out2 = parallel_fft2_interleaved_f32(val03[2], val47[2]);
        let out3 = parallel_fft2_interleaved_f32(val03[3], val47[3]);

        // step 6: rearrange and copy to buffer
        [
            out0[0], out1[0], out2[0], out3[0], out0[1], out1[1], out2[1], out3[1],
        ]
    }
}

//    ___             __   _  _   _     _ _
//   ( _ )           / /_ | || | | |__ (_) |_
//   / _ \   _____  | '_ \| || |_| '_ \| | __|
//  | (_) | |_____| | (_) |__   _| |_) | | |_
//   \___/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly8<T> {
    root2: __m128d,
    direction: FftDirection,
    bf4: SseF64Butterfly4<T>,
    rotate90: Rotate90F64,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly8, 8, |this: &SseF64Butterfly8<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly8, 8, |this: &SseF64Butterfly8<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf4 = SseF64Butterfly4::new(direction);
        let root2 = unsafe { _mm_load1_pd(&0.5f64.sqrt()) };
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F64::new(true)
        } else {
            Rotate90F64::new(false)
        };
        Self {
            root2,
            direction,
            bf4,
            rotate90,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let values = read_complex_to_array!(input, {0, 1, 2, 3, 4, 5, 6, 7});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, output, {0, 1, 2, 3, 4, 5, 6, 7});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, values: [__m128d; 8]) -> [__m128d; 8] {
        // we're going to hardcode a step of mixed radix
        // step 1: copy and reorder the input into the scratch
        // and
        // step 2: column FFTs
        let val03 = self
            .bf4
            .perform_fft_direct(values[0], values[2], values[4], values[6]);
        let mut val47 = self
            .bf4
            .perform_fft_direct(values[1], values[3], values[5], values[7]);

        // step 3: apply twiddle factors
        let val5b = self.rotate90.rotate(val47[1]);
        let val5c = _mm_add_pd(val5b, val47[1]);
        val47[1] = _mm_mul_pd(val5c, self.root2);
        val47[2] = self.rotate90.rotate(val47[2]);
        let val7b = self.rotate90.rotate(val47[3]);
        let val7c = _mm_sub_pd(val7b, val47[3]);
        val47[3] = _mm_mul_pd(val7c, self.root2);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        let out0 = solo_fft2_f64(val03[0], val47[0]);
        let out1 = solo_fft2_f64(val03[1], val47[1]);
        let out2 = solo_fft2_f64(val03[2], val47[2]);
        let out3 = solo_fft2_f64(val03[3], val47[3]);

        // step 6: rearrange and copy to buffer
        [
            out0[0], out1[0], out2[0], out3[0], out0[1], out1[1], out2[1], out3[1],
        ]
    }
}

//    ___            _________  _     _ _
//   / _ \          |___ /___ \| |__ (_) |_
//  | (_) |  _____    |_ \ __) | '_ \| | __|
//   \__, | |_____|  ___) / __/| |_) | | |_
//     /_/          |____/_____|_.__/|_|\__|
//
pub struct SseF32Butterfly9<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: SseF32Butterfly3<T>,
    twiddle1: __m128,
    twiddle2: __m128,
    twiddle4: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly9, 9, |this: &SseF32Butterfly9<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly9, 9, |this: &SseF32Butterfly9<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly9<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = SseF32Butterfly3::new(direction);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 9, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 9, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 9, direction);
        let twiddle1 = unsafe { _mm_set_ps(tw1.im, tw1.re, tw1.im, tw1.re) };
        let twiddle2 = unsafe { _mm_set_ps(tw2.im, tw2.re, tw2.im, tw2.re) };
        let twiddle4 = unsafe { _mm_set_ps(tw4.im, tw4.re, tw4.im, tw4.re) };
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            twiddle1,
            twiddle2,
            twiddle4,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        // A single Sse 9-point will need a lot of shuffling, let's just reuse the dual one
        let values = read_partial1_complex_to_array!(input, {0,1,2,3,4,5,6,7,8});

        let out = self.perform_parallel_fft_direct(values);

        for n in 0..9 {
            output.store_partial_lo_complex(out[n], n);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14, 16});

        let values = [
            extract_lo_hi_f32(input_packed[0], input_packed[4]),
            extract_hi_lo_f32(input_packed[0], input_packed[5]),
            extract_lo_hi_f32(input_packed[1], input_packed[5]),
            extract_hi_lo_f32(input_packed[1], input_packed[6]),
            extract_lo_hi_f32(input_packed[2], input_packed[6]),
            extract_hi_lo_f32(input_packed[2], input_packed[7]),
            extract_lo_hi_f32(input_packed[3], input_packed[7]),
            extract_hi_lo_f32(input_packed[3], input_packed[8]),
            extract_lo_hi_f32(input_packed[4], input_packed[8]),
        ];

        let out = self.perform_parallel_fft_direct(values);

        let out_packed = [
            extract_lo_lo_f32(out[0], out[1]),
            extract_lo_lo_f32(out[2], out[3]),
            extract_lo_lo_f32(out[4], out[5]),
            extract_lo_lo_f32(out[6], out[7]),
            extract_lo_hi_f32(out[8], out[0]),
            extract_hi_hi_f32(out[1], out[2]),
            extract_hi_hi_f32(out[3], out[4]),
            extract_hi_hi_f32(out[5], out[6]),
            extract_hi_hi_f32(out[7], out[8]),
        ];

        write_complex_to_array_strided!(out_packed, output, 2, {0,1,2,3,4,5,6,7,8});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [__m128; 9]) -> [__m128; 9] {
        // Algorithm: 3x3 mixed radix

        // Size-3 FFTs down the columns
        let mid0 = self
            .bf3
            .perform_parallel_fft_direct(values[0], values[3], values[6]);
        let mut mid1 = self
            .bf3
            .perform_parallel_fft_direct(values[1], values[4], values[7]);
        let mut mid2 = self
            .bf3
            .perform_parallel_fft_direct(values[2], values[5], values[8]);

        // Apply twiddle factors. Note that we're re-using twiddle2
        mid1[1] = mul_complex_f32(self.twiddle1, mid1[1]);
        mid1[2] = mul_complex_f32(self.twiddle2, mid1[2]);
        mid2[1] = mul_complex_f32(self.twiddle2, mid2[1]);
        mid2[2] = mul_complex_f32(self.twiddle4, mid2[2]);

        let [output0, output1, output2] = self
            .bf3
            .perform_parallel_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] = self
            .bf3
            .perform_parallel_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] = self
            .bf3
            .perform_parallel_fft_direct(mid0[2], mid1[2], mid2[2]);

        [
            output0, output3, output6, output1, output4, output7, output2, output5, output8,
        ]
    }
}

//    ___             __   _  _   _     _ _
//   / _ \           / /_ | || | | |__ (_) |_
//  | (_) |  _____  | '_ \| || |_| '_ \| | __|
//   \__, | |_____| | (_) |__   _| |_) | | |_
//     /_/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly9<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: SseF64Butterfly3<T>,
    twiddle1: __m128d,
    twiddle2: __m128d,
    twiddle4: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly9, 9, |this: &SseF64Butterfly9<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly9, 9, |this: &SseF64Butterfly9<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly9<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = SseF64Butterfly3::new(direction);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 9, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 9, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 9, direction);
        let twiddle1 = unsafe { _mm_set_pd(tw1.im, tw1.re) };
        let twiddle2 = unsafe { _mm_set_pd(tw2.im, tw2.re) };
        let twiddle4 = unsafe { _mm_set_pd(tw4.im, tw4.re) };
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            twiddle1,
            twiddle2,
            twiddle4,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let values = read_complex_to_array!(input, {0, 1, 2, 3, 4, 5, 6, 7, 8});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, output, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 9]) -> [__m128d; 9] {
        // Algorithm: 3x3 mixed radix

        // Size-3 FFTs down the columns
        let mid0 = self.bf3.perform_fft_direct(values[0], values[3], values[6]);
        let mut mid1 = self.bf3.perform_fft_direct(values[1], values[4], values[7]);
        let mut mid2 = self.bf3.perform_fft_direct(values[2], values[5], values[8]);

        // Apply twiddle factors. Note that we're re-using twiddle2
        mid1[1] = mul_complex_f64(self.twiddle1, mid1[1]);
        mid1[2] = mul_complex_f64(self.twiddle2, mid1[2]);
        mid2[1] = mul_complex_f64(self.twiddle2, mid2[1]);
        mid2[2] = mul_complex_f64(self.twiddle4, mid2[2]);

        let [output0, output1, output2] = self.bf3.perform_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] = self.bf3.perform_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] = self.bf3.perform_fft_direct(mid0[2], mid1[2], mid2[2]);

        [
            output0, output3, output6, output1, output4, output7, output2, output5, output8,
        ]
    }
}

//   _  ___            _________  _     _ _
//  / |/ _ \          |___ /___ \| |__ (_) |_
//  | | | | |  _____    |_ \ __) | '_ \| | __|
//  | | |_| | |_____|  ___) / __/| |_) | | |_
//  |_|\___/          |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly10<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf5: SseF32Butterfly5<T>,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly10, 10, |this: &SseF32Butterfly10<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly10, 10, |this: &SseF32Butterfly10<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly10<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf5 = SseF32Butterfly5::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf5,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8});

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, output, 2, {0,1,2,3,4});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18});

        let values = interleave_complex_f32!(input_packed, 5, {0, 1, 2, 3, 4});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8});

        write_complex_to_array_strided!(out_sorted, output, 2, {0,1,2,3,4,5,6,7,8,9});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128; 5]) -> [__m128; 5] {
        // Algorithm: 5x2 good-thomas
        // Reorder and pack
        let reord0 = extract_lo_hi_f32(values[0], values[2]);
        let reord1 = extract_lo_hi_f32(values[1], values[3]);
        let reord2 = extract_lo_hi_f32(values[2], values[4]);
        let reord3 = extract_lo_hi_f32(values[3], values[0]);
        let reord4 = extract_lo_hi_f32(values[4], values[1]);

        // Size-5 FFTs down the columns of our reordered array
        let mids = self
            .bf5
            .perform_parallel_fft_direct(reord0, reord1, reord2, reord3, reord4);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [temp01, temp23] = parallel_fft2_contiguous_f32(mids[0], mids[1]);
        let [temp45, temp67] = parallel_fft2_contiguous_f32(mids[2], mids[3]);
        let temp89 = solo_fft2_f32(mids[4]);

        // Reorder
        let out01 = extract_lo_hi_f32(temp01, temp23);
        let out23 = extract_lo_hi_f32(temp45, temp67);
        let out45 = extract_lo_lo_f32(temp89, temp23);
        let out67 = extract_hi_lo_f32(temp01, temp67);
        let out89 = extract_hi_hi_f32(temp45, temp89);

        [out01, out23, out45, out67, out89]
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [__m128; 10]) -> [__m128; 10] {
        // Algorithm: 5x2 good-thomas

        // Size-5 FFTs down the columns of our reordered array
        let mid0 = self
            .bf5
            .perform_parallel_fft_direct(values[0], values[2], values[4], values[6], values[8]);
        let mid1 = self
            .bf5
            .perform_parallel_fft_direct(values[5], values[7], values[9], values[1], values[3]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = parallel_fft2_interleaved_f32(mid0[0], mid1[0]);
        let [output2, output3] = parallel_fft2_interleaved_f32(mid0[1], mid1[1]);
        let [output4, output5] = parallel_fft2_interleaved_f32(mid0[2], mid1[2]);
        let [output6, output7] = parallel_fft2_interleaved_f32(mid0[3], mid1[3]);
        let [output8, output9] = parallel_fft2_interleaved_f32(mid0[4], mid1[4]);

        // Reorder and return
        [
            output0, output3, output4, output7, output8, output1, output2, output5, output6,
            output9,
        ]
    }
}

//   _  ___             __   _  _   _     _ _
//  / |/ _ \           / /_ | || | | |__ (_) |_
//  | | | | |  _____  | '_ \| || |_| '_ \| | __|
//  | | |_| | |_____| | (_) |__   _| |_) | | |_
//  |_|\___/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly10<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf2: SseF64Butterfly2<T>,
    bf5: SseF64Butterfly5<T>,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly10, 10, |this: &SseF64Butterfly10<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly10, 10, |this: &SseF64Butterfly10<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly10<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf2 = SseF64Butterfly2::new(direction);
        let bf5 = SseF64Butterfly5::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf2,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let values = read_complex_to_array!(input, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, output, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 10]) -> [__m128d; 10] {
        // Algorithm: 5x2 good-thomas

        // Size-5 FFTs down the columns of our reordered array
        let mid0 = self
            .bf5
            .perform_fft_direct(values[0], values[2], values[4], values[6], values[8]);
        let mid1 = self
            .bf5
            .perform_fft_direct(values[5], values[7], values[9], values[1], values[3]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = self.bf2.perform_fft_direct(mid0[0], mid1[0]);
        let [output2, output3] = self.bf2.perform_fft_direct(mid0[1], mid1[1]);
        let [output4, output5] = self.bf2.perform_fft_direct(mid0[2], mid1[2]);
        let [output6, output7] = self.bf2.perform_fft_direct(mid0[3], mid1[3]);
        let [output8, output9] = self.bf2.perform_fft_direct(mid0[4], mid1[4]);

        // Reorder and return
        [
            output0, output3, output4, output7, output8, output1, output2, output5, output6,
            output9,
        ]
    }
}

//   _ ____            _________  _     _ _
//  / |___ \          |___ /___ \| |__ (_) |_
//  | | __) |  _____    |_ \ __) | '_ \| | __|
//  | |/ __/  |_____|  ___) / __/| |_) | | |_
//  |_|_____|         |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly12<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: SseF32Butterfly3<T>,
    bf4: SseF32Butterfly4<T>,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly12, 12, |this: &SseF32Butterfly12<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly12, 12, |this: &SseF32Butterfly12<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly12<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = SseF32Butterfly3::new(direction);
        let bf4 = SseF32Butterfly4::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            bf4,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8, 10 });

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, output, 2, {0,1,2,3,4,5});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed =
            read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});

        let values = interleave_complex_f32!(input_packed, 6, {0, 1, 2, 3, 4, 5});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8, 10});

        write_complex_to_array_strided!(out_sorted, output, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128; 6]) -> [__m128; 6] {
        // Algorithm: 4x3 good-thomas

        // Reorder and pack
        let packed03 = extract_lo_hi_f32(values[0], values[1]);
        let packed47 = extract_lo_hi_f32(values[2], values[3]);
        let packed69 = extract_lo_hi_f32(values[3], values[4]);
        let packed101 = extract_lo_hi_f32(values[5], values[0]);
        let packed811 = extract_lo_hi_f32(values[4], values[5]);
        let packed25 = extract_lo_hi_f32(values[1], values[2]);

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = self.bf4.perform_fft_direct(packed03, packed69);
        let mid1 = self.bf4.perform_fft_direct(packed47, packed101);
        let mid2 = self.bf4.perform_fft_direct(packed811, packed25);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [temp03, temp14, temp25] = self
            .bf3
            .perform_parallel_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [temp69, temp710, temp811] = self
            .bf3
            .perform_parallel_fft_direct(mid0[1], mid1[1], mid2[1]);

        // Reorder and return
        [
            extract_lo_hi_f32(temp03, temp14),
            extract_lo_hi_f32(temp811, temp69),
            extract_lo_hi_f32(temp14, temp25),
            extract_lo_hi_f32(temp69, temp710),
            extract_lo_hi_f32(temp25, temp03),
            extract_lo_hi_f32(temp710, temp811),
        ]
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [__m128; 12]) -> [__m128; 12] {
        // Algorithm: 4x3 good-thomas

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = self
            .bf4
            .perform_parallel_fft_direct(values[0], values[3], values[6], values[9]);
        let mid1 = self
            .bf4
            .perform_parallel_fft_direct(values[4], values[7], values[10], values[1]);
        let mid2 = self
            .bf4
            .perform_parallel_fft_direct(values[8], values[11], values[2], values[5]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [output0, output1, output2] = self
            .bf3
            .perform_parallel_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] = self
            .bf3
            .perform_parallel_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] = self
            .bf3
            .perform_parallel_fft_direct(mid0[2], mid1[2], mid2[2]);
        let [output9, output10, output11] = self
            .bf3
            .perform_parallel_fft_direct(mid0[3], mid1[3], mid2[3]);

        // Reorder and return
        [
            output0, output4, output8, output9, output1, output5, output6, output10, output2,
            output3, output7, output11,
        ]
    }
}

//   _ ____             __   _  _   _     _ _
//  / |___ \           / /_ | || | | |__ (_) |_
//  | | __) |  _____  | '_ \| || |_| '_ \| | __|
//  | |/ __/  |_____| | (_) |__   _| |_) | | |_
//  |_|_____|          \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly12<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: SseF64Butterfly3<T>,
    bf4: SseF64Butterfly4<T>,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly12, 12, |this: &SseF64Butterfly12<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly12, 12, |this: &SseF64Butterfly12<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly12<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = SseF64Butterfly3::new(direction);
        let bf4 = SseF64Butterfly4::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            bf4,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let values = read_complex_to_array!(input, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, output, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 12]) -> [__m128d; 12] {
        // Algorithm: 4x3 good-thomas

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = self
            .bf4
            .perform_fft_direct(values[0], values[3], values[6], values[9]);
        let mid1 = self
            .bf4
            .perform_fft_direct(values[4], values[7], values[10], values[1]);
        let mid2 = self
            .bf4
            .perform_fft_direct(values[8], values[11], values[2], values[5]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [output0, output1, output2] = self.bf3.perform_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] = self.bf3.perform_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] = self.bf3.perform_fft_direct(mid0[2], mid1[2], mid2[2]);
        let [output9, output10, output11] = self.bf3.perform_fft_direct(mid0[3], mid1[3], mid2[3]);

        [
            output0, output4, output8, output9, output1, output5, output6, output10, output2,
            output3, output7, output11,
        ]
    }
}

//   _ ____            _________  _     _ _
//  / | ___|          |___ /___ \| |__ (_) |_
//  | |___ \   _____    |_ \ __) | '_ \| | __|
//  | |___) | |_____|  ___) / __/| |_) | | |_
//  |_|____/          |____/_____|_.__/|_|\__|
//
pub struct SseF32Butterfly15<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: SseF32Butterfly3<T>,
    bf5: SseF32Butterfly5<T>,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly15, 15, |this: &SseF32Butterfly15<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly15, 15, |this: &SseF32Butterfly15<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly15<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = SseF32Butterfly3::new(direction);
        let bf5 = SseF32Butterfly5::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        // A single Sse 15-point will need a lot of shuffling, let's just reuse the dual one
        let values = read_partial1_complex_to_array!(input, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14});

        let out = self.perform_parallel_fft_direct(values);

        for n in 0..15 {
            output.store_partial_lo_complex(out[n], n);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed =
            read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28});

        let values = [
            extract_lo_hi_f32(input_packed[0], input_packed[7]),
            extract_hi_lo_f32(input_packed[0], input_packed[8]),
            extract_lo_hi_f32(input_packed[1], input_packed[8]),
            extract_hi_lo_f32(input_packed[1], input_packed[9]),
            extract_lo_hi_f32(input_packed[2], input_packed[9]),
            extract_hi_lo_f32(input_packed[2], input_packed[10]),
            extract_lo_hi_f32(input_packed[3], input_packed[10]),
            extract_hi_lo_f32(input_packed[3], input_packed[11]),
            extract_lo_hi_f32(input_packed[4], input_packed[11]),
            extract_hi_lo_f32(input_packed[4], input_packed[12]),
            extract_lo_hi_f32(input_packed[5], input_packed[12]),
            extract_hi_lo_f32(input_packed[5], input_packed[13]),
            extract_lo_hi_f32(input_packed[6], input_packed[13]),
            extract_hi_lo_f32(input_packed[6], input_packed[14]),
            extract_lo_hi_f32(input_packed[7], input_packed[14]),
        ];

        let out = self.perform_parallel_fft_direct(values);

        let out_packed = [
            extract_lo_lo_f32(out[0], out[1]),
            extract_lo_lo_f32(out[2], out[3]),
            extract_lo_lo_f32(out[4], out[5]),
            extract_lo_lo_f32(out[6], out[7]),
            extract_lo_lo_f32(out[8], out[9]),
            extract_lo_lo_f32(out[10], out[11]),
            extract_lo_lo_f32(out[12], out[13]),
            extract_lo_hi_f32(out[14], out[0]),
            extract_hi_hi_f32(out[1], out[2]),
            extract_hi_hi_f32(out[3], out[4]),
            extract_hi_hi_f32(out[5], out[6]),
            extract_hi_hi_f32(out[7], out[8]),
            extract_hi_hi_f32(out[9], out[10]),
            extract_hi_hi_f32(out[11], out[12]),
            extract_hi_hi_f32(out[13], out[14]),
        ];

        write_complex_to_array_strided!(out_packed, output, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11, 12, 13, 14});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [__m128; 15]) -> [__m128; 15] {
        // Algorithm: 5x3 good-thomas

        // Size-5 FFTs down the columns of our reordered array
        let mid0 = self
            .bf5
            .perform_parallel_fft_direct(values[0], values[3], values[6], values[9], values[12]);
        let mid1 = self
            .bf5
            .perform_parallel_fft_direct(values[5], values[8], values[11], values[14], values[2]);
        let mid2 = self
            .bf5
            .perform_parallel_fft_direct(values[10], values[13], values[1], values[4], values[7]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [output0, output1, output2] = self
            .bf3
            .perform_parallel_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] = self
            .bf3
            .perform_parallel_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] = self
            .bf3
            .perform_parallel_fft_direct(mid0[2], mid1[2], mid2[2]);
        let [output9, output10, output11] = self
            .bf3
            .perform_parallel_fft_direct(mid0[3], mid1[3], mid2[3]);
        let [output12, output13, output14] = self
            .bf3
            .perform_parallel_fft_direct(mid0[4], mid1[4], mid2[4]);

        [
            output0, output4, output8, output9, output13, output2, output3, output7, output11,
            output12, output1, output5, output6, output10, output14,
        ]
    }
}

//   _ ____             __   _  _   _     _ _
//  / | ___|           / /_ | || | | |__ (_) |_
//  | |___ \   _____  | '_ \| || |_| '_ \| | __|
//  | |___) | |_____| | (_) |__   _| |_) | | |_
//  |_|____/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly15<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: SseF64Butterfly3<T>,
    bf5: SseF64Butterfly5<T>,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly15, 15, |this: &SseF64Butterfly15<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly15, 15, |this: &SseF64Butterfly15<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly15<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = SseF64Butterfly3::new(direction);
        let bf5 = SseF64Butterfly5::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let values =
            read_complex_to_array!(input, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, output, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 15]) -> [__m128d; 15] {
        // Algorithm: 5x3 good-thomas

        // Size-5 FFTs down the columns of our reordered array
        let mid0 = self
            .bf5
            .perform_fft_direct(values[0], values[3], values[6], values[9], values[12]);
        let mid1 = self
            .bf5
            .perform_fft_direct(values[5], values[8], values[11], values[14], values[2]);
        let mid2 = self
            .bf5
            .perform_fft_direct(values[10], values[13], values[1], values[4], values[7]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [output0, output1, output2] = self.bf3.perform_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] = self.bf3.perform_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] = self.bf3.perform_fft_direct(mid0[2], mid1[2], mid2[2]);
        let [output9, output10, output11] = self.bf3.perform_fft_direct(mid0[3], mid1[3], mid2[3]);
        let [output12, output13, output14] = self.bf3.perform_fft_direct(mid0[4], mid1[4], mid2[4]);

        [
            output0, output4, output8, output9, output13, output2, output3, output7, output11,
            output12, output1, output5, output6, output10, output14,
        ]
    }
}

//   _  __             _________  _     _ _
//  / |/ /_           |___ /___ \| |__ (_) |_
//  | | '_ \   _____    |_ \ __) | '_ \| | __|
//  | | (_) | |_____|  ___) / __/| |_) | | |_
//  |_|\___/          |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly16<T> {
    direction: FftDirection,
    bf4: SseF32Butterfly4<T>,
    bf8: SseF32Butterfly8<T>,
    rotate90: Rotate90F32,
    twiddle01: __m128,
    twiddle23: __m128,
    twiddle01conj: __m128,
    twiddle23conj: __m128,
    twiddle1: __m128,
    twiddle2: __m128,
    twiddle3: __m128,
    twiddle1c: __m128,
    twiddle2c: __m128,
    twiddle3c: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly16, 16, |this: &SseF32Butterfly16<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly16, 16, |this: &SseF32Butterfly16<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf8 = SseF32Butterfly8::new(direction);
        let bf4 = SseF32Butterfly4::new(direction);
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F32::new(true)
        } else {
            Rotate90F32::new(false)
        };
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 16, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 16, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 16, direction);
        let twiddle01 = unsafe { _mm_set_ps(tw1.im, tw1.re, 0.0, 1.0) };
        let twiddle23 = unsafe { _mm_set_ps(tw3.im, tw3.re, tw2.im, tw2.re) };
        let twiddle01conj = unsafe { _mm_set_ps(-tw1.im, tw1.re, 0.0, 1.0) };
        let twiddle23conj = unsafe { _mm_set_ps(-tw3.im, tw3.re, -tw2.im, tw2.re) };
        let twiddle1 = unsafe { _mm_set_ps(tw1.im, tw1.re, tw1.im, tw1.re) };
        let twiddle2 = unsafe { _mm_set_ps(tw2.im, tw2.re, tw2.im, tw2.re) };
        let twiddle3 = unsafe { _mm_set_ps(tw3.im, tw3.re, tw3.im, tw3.re) };
        let twiddle1c = unsafe { _mm_set_ps(-tw1.im, tw1.re, -tw1.im, tw1.re) };
        let twiddle2c = unsafe { _mm_set_ps(-tw2.im, tw2.re, -tw2.im, tw2.re) };
        let twiddle3c = unsafe { _mm_set_ps(-tw3.im, tw3.re, -tw3.im, tw3.re) };
        Self {
            direction,
            bf4,
            bf8,
            rotate90,
            twiddle01,
            twiddle23,
            twiddle01conj,
            twiddle23conj,
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle1c,
            twiddle2c,
            twiddle3c,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14 });

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, output, 2, {0,1,2,3,4,5,6,7});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30});

        let values = interleave_complex_f32!(input_packed, 8, {0, 1, 2, 3 ,4 ,5 ,6 ,7});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8, 10, 12, 14});

        write_complex_to_array_strided!(out_sorted, output, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11,12,13,14, 15});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [__m128; 8]) -> [__m128; 8] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the input into the scratch
        let in0002 = extract_lo_lo_f32(input[0], input[1]);
        let in0406 = extract_lo_lo_f32(input[2], input[3]);
        let in0810 = extract_lo_lo_f32(input[4], input[5]);
        let in1214 = extract_lo_lo_f32(input[6], input[7]);

        let in0105 = extract_hi_hi_f32(input[0], input[2]);
        let in0913 = extract_hi_hi_f32(input[4], input[6]);
        let in1503 = extract_hi_hi_f32(input[7], input[1]);
        let in0711 = extract_hi_hi_f32(input[3], input[5]);

        let in_evens = [in0002, in0406, in0810, in1214];

        // step 2: column FFTs
        let evens = self.bf8.perform_fft_direct(in_evens);
        let mut odds1 = self.bf4.perform_fft_direct(in0105, in0913);
        let mut odds3 = self.bf4.perform_fft_direct(in1503, in0711);

        // step 3: apply twiddle factors
        odds1[0] = mul_complex_f32(odds1[0], self.twiddle01);
        odds3[0] = mul_complex_f32(odds3[0], self.twiddle01conj);

        odds1[1] = mul_complex_f32(odds1[1], self.twiddle23);
        odds3[1] = mul_complex_f32(odds3[1], self.twiddle23conj);

        // step 4: cross FFTs
        let mut temp0 = parallel_fft2_interleaved_f32(odds1[0], odds3[0]);
        let mut temp1 = parallel_fft2_interleaved_f32(odds1[1], odds3[1]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate_both(temp0[1]);
        temp1[1] = self.rotate90.rotate_both(temp1[1]);

        //step 5: copy/add/subtract data back to buffer
        [
            _mm_add_ps(evens[0], temp0[0]),
            _mm_add_ps(evens[1], temp1[0]),
            _mm_add_ps(evens[2], temp0[1]),
            _mm_add_ps(evens[3], temp1[1]),
            _mm_sub_ps(evens[0], temp0[0]),
            _mm_sub_ps(evens[1], temp1[0]),
            _mm_sub_ps(evens[2], temp0[1]),
            _mm_sub_ps(evens[3], temp1[1]),
        ]
    }

    #[inline(always)]
    unsafe fn perform_parallel_fft_direct(&self, input: [__m128; 16]) -> [__m128; 16] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        // and
        // step 2: column FFTs
        let evens = self.bf8.perform_parallel_fft_direct([
            input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14],
        ]);
        let mut odds1 = self
            .bf4
            .perform_parallel_fft_direct(input[1], input[5], input[9], input[13]);
        let mut odds3 = self
            .bf4
            .perform_parallel_fft_direct(input[15], input[3], input[7], input[11]);

        // step 3: apply twiddle factors
        odds1[1] = mul_complex_f32(odds1[1], self.twiddle1);
        odds3[1] = mul_complex_f32(odds3[1], self.twiddle1c);

        odds1[2] = mul_complex_f32(odds1[2], self.twiddle2);
        odds3[2] = mul_complex_f32(odds3[2], self.twiddle2c);

        odds1[3] = mul_complex_f32(odds1[3], self.twiddle3);
        odds3[3] = mul_complex_f32(odds3[3], self.twiddle3c);

        // step 4: cross FFTs
        let mut temp0 = parallel_fft2_interleaved_f32(odds1[0], odds3[0]);
        let mut temp1 = parallel_fft2_interleaved_f32(odds1[1], odds3[1]);
        let mut temp2 = parallel_fft2_interleaved_f32(odds1[2], odds3[2]);
        let mut temp3 = parallel_fft2_interleaved_f32(odds1[3], odds3[3]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate_both(temp0[1]);
        temp1[1] = self.rotate90.rotate_both(temp1[1]);
        temp2[1] = self.rotate90.rotate_both(temp2[1]);
        temp3[1] = self.rotate90.rotate_both(temp3[1]);

        //step 5: copy/add/subtract data back to buffer
        [
            _mm_add_ps(evens[0], temp0[0]),
            _mm_add_ps(evens[1], temp1[0]),
            _mm_add_ps(evens[2], temp2[0]),
            _mm_add_ps(evens[3], temp3[0]),
            _mm_add_ps(evens[4], temp0[1]),
            _mm_add_ps(evens[5], temp1[1]),
            _mm_add_ps(evens[6], temp2[1]),
            _mm_add_ps(evens[7], temp3[1]),
            _mm_sub_ps(evens[0], temp0[0]),
            _mm_sub_ps(evens[1], temp1[0]),
            _mm_sub_ps(evens[2], temp2[0]),
            _mm_sub_ps(evens[3], temp3[0]),
            _mm_sub_ps(evens[4], temp0[1]),
            _mm_sub_ps(evens[5], temp1[1]),
            _mm_sub_ps(evens[6], temp2[1]),
            _mm_sub_ps(evens[7], temp3[1]),
        ]
    }
}

//   _  __              __   _  _   _     _ _
//  / |/ /_            / /_ | || | | |__ (_) |_
//  | | '_ \   _____  | '_ \| || |_| '_ \| | __|
//  | | (_) | |_____| | (_) |__   _| |_) | | |_
//  |_|\___/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly16<T> {
    direction: FftDirection,
    bf4: SseF64Butterfly4<T>,
    bf8: SseF64Butterfly8<T>,
    rotate90: Rotate90F64,
    twiddle1: __m128d,
    twiddle2: __m128d,
    twiddle3: __m128d,
    twiddle1c: __m128d,
    twiddle2c: __m128d,
    twiddle3c: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly16, 16, |this: &SseF64Butterfly16<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly16, 16, |this: &SseF64Butterfly16<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf8 = SseF64Butterfly8::new(direction);
        let bf4 = SseF64Butterfly4::new(direction);
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F64::new(true)
        } else {
            Rotate90F64::new(false)
        };
        let twiddle1 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(1, 16, direction).re as *const f64) };
        let twiddle2 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(2, 16, direction).re as *const f64) };
        let twiddle3 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(3, 16, direction).re as *const f64) };
        let twiddle1c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(1, 16, direction).conj().re as *const f64)
        };
        let twiddle2c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(2, 16, direction).conj().re as *const f64)
        };
        let twiddle3c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(3, 16, direction).conj().re as *const f64)
        };

        Self {
            direction,
            bf4,
            bf8,
            rotate90,
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle1c,
            twiddle2c,
            twiddle3c,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let values =
            read_complex_to_array!(input, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, output, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [__m128d; 16]) -> [__m128d; 16] {
        // we're going to hardcode a step of split radix

        // step 1: copy and reorder the  input into the scratch
        // and
        // step 2: column FFTs
        let evens = self.bf8.perform_fft_direct([
            input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14],
        ]);
        let mut odds1 = self
            .bf4
            .perform_fft_direct(input[1], input[5], input[9], input[13]);
        let mut odds3 = self
            .bf4
            .perform_fft_direct(input[15], input[3], input[7], input[11]);

        // step 3: apply twiddle factors
        odds1[1] = mul_complex_f64(odds1[1], self.twiddle1);
        odds3[1] = mul_complex_f64(odds3[1], self.twiddle1c);

        odds1[2] = mul_complex_f64(odds1[2], self.twiddle2);
        odds3[2] = mul_complex_f64(odds3[2], self.twiddle2c);

        odds1[3] = mul_complex_f64(odds1[3], self.twiddle3);
        odds3[3] = mul_complex_f64(odds3[3], self.twiddle3c);

        // step 4: cross FFTs
        let mut temp0 = solo_fft2_f64(odds1[0], odds3[0]);
        let mut temp1 = solo_fft2_f64(odds1[1], odds3[1]);
        let mut temp2 = solo_fft2_f64(odds1[2], odds3[2]);
        let mut temp3 = solo_fft2_f64(odds1[3], odds3[3]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate(temp0[1]);
        temp1[1] = self.rotate90.rotate(temp1[1]);
        temp2[1] = self.rotate90.rotate(temp2[1]);
        temp3[1] = self.rotate90.rotate(temp3[1]);

        //step 5: copy/add/subtract data back to buffer
        [
            _mm_add_pd(evens[0], temp0[0]),
            _mm_add_pd(evens[1], temp1[0]),
            _mm_add_pd(evens[2], temp2[0]),
            _mm_add_pd(evens[3], temp3[0]),
            _mm_add_pd(evens[4], temp0[1]),
            _mm_add_pd(evens[5], temp1[1]),
            _mm_add_pd(evens[6], temp2[1]),
            _mm_add_pd(evens[7], temp3[1]),
            _mm_sub_pd(evens[0], temp0[0]),
            _mm_sub_pd(evens[1], temp1[0]),
            _mm_sub_pd(evens[2], temp2[0]),
            _mm_sub_pd(evens[3], temp3[0]),
            _mm_sub_pd(evens[4], temp0[1]),
            _mm_sub_pd(evens[5], temp1[1]),
            _mm_sub_pd(evens[6], temp2[1]),
            _mm_sub_pd(evens[7], temp3[1]),
        ]
    }
}

//   _________            _________  _     _ _
//  |___ /___ \          |___ /___ \| |__ (_) |_
//    |_ \ __) |  _____    |_ \ __) | '_ \| | __|
//   ___) / __/  |_____|  ___) / __/| |_) | | |_
//  |____/_____|         |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly32<T> {
    direction: FftDirection,
    bf8: SseF32Butterfly8<T>,
    bf16: SseF32Butterfly16<T>,
    rotate90: Rotate90F32,
    twiddle01: __m128,
    twiddle23: __m128,
    twiddle45: __m128,
    twiddle67: __m128,
    twiddle01conj: __m128,
    twiddle23conj: __m128,
    twiddle45conj: __m128,
    twiddle67conj: __m128,
    twiddle1: __m128,
    twiddle2: __m128,
    twiddle3: __m128,
    twiddle4: __m128,
    twiddle5: __m128,
    twiddle6: __m128,
    twiddle7: __m128,
    twiddle1c: __m128,
    twiddle2c: __m128,
    twiddle3c: __m128,
    twiddle4c: __m128,
    twiddle5c: __m128,
    twiddle6c: __m128,
    twiddle7c: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly32, 32, |this: &SseF32Butterfly32<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly32, 32, |this: &SseF32Butterfly32<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly32<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf8 = SseF32Butterfly8::new(direction);
        let bf16 = SseF32Butterfly16::new(direction);
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F32::new(true)
        } else {
            Rotate90F32::new(false)
        };
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 32, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 32, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 32, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 32, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 32, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 32, direction);
        let tw7: Complex<f32> = twiddles::compute_twiddle(7, 32, direction);
        let twiddle01 = unsafe { _mm_set_ps(tw1.im, tw1.re, 0.0, 1.0) };
        let twiddle23 = unsafe { _mm_set_ps(tw3.im, tw3.re, tw2.im, tw2.re) };
        let twiddle45 = unsafe { _mm_set_ps(tw5.im, tw5.re, tw4.im, tw4.re) };
        let twiddle67 = unsafe { _mm_set_ps(tw7.im, tw7.re, tw6.im, tw6.re) };
        let twiddle01conj = unsafe { _mm_set_ps(-tw1.im, tw1.re, 0.0, 1.0) };
        let twiddle23conj = unsafe { _mm_set_ps(-tw3.im, tw3.re, -tw2.im, tw2.re) };
        let twiddle45conj = unsafe { _mm_set_ps(-tw5.im, tw5.re, -tw4.im, tw4.re) };
        let twiddle67conj = unsafe { _mm_set_ps(-tw7.im, tw7.re, -tw6.im, tw6.re) };
        let twiddle1 = unsafe { _mm_set_ps(tw1.im, tw1.re, tw1.im, tw1.re) };
        let twiddle2 = unsafe { _mm_set_ps(tw2.im, tw2.re, tw2.im, tw2.re) };
        let twiddle3 = unsafe { _mm_set_ps(tw3.im, tw3.re, tw3.im, tw3.re) };
        let twiddle4 = unsafe { _mm_set_ps(tw4.im, tw4.re, tw4.im, tw4.re) };
        let twiddle5 = unsafe { _mm_set_ps(tw5.im, tw5.re, tw5.im, tw5.re) };
        let twiddle6 = unsafe { _mm_set_ps(tw6.im, tw6.re, tw6.im, tw6.re) };
        let twiddle7 = unsafe { _mm_set_ps(tw7.im, tw7.re, tw7.im, tw7.re) };
        let twiddle1c = unsafe { _mm_set_ps(-tw1.im, tw1.re, -tw1.im, tw1.re) };
        let twiddle2c = unsafe { _mm_set_ps(-tw2.im, tw2.re, -tw2.im, tw2.re) };
        let twiddle3c = unsafe { _mm_set_ps(-tw3.im, tw3.re, -tw3.im, tw3.re) };
        let twiddle4c = unsafe { _mm_set_ps(-tw4.im, tw4.re, -tw4.im, tw4.re) };
        let twiddle5c = unsafe { _mm_set_ps(-tw5.im, tw5.re, -tw5.im, tw5.re) };
        let twiddle6c = unsafe { _mm_set_ps(-tw6.im, tw6.re, -tw6.im, tw6.re) };
        let twiddle7c = unsafe { _mm_set_ps(-tw7.im, tw7.re, -tw7.im, tw7.re) };
        Self {
            direction,
            bf8,
            bf16,
            rotate90,
            twiddle01,
            twiddle23,
            twiddle45,
            twiddle67,
            twiddle01conj,
            twiddle23conj,
            twiddle45conj,
            twiddle67conj,
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            twiddle7,
            twiddle1c,
            twiddle2c,
            twiddle3c,
            twiddle4c,
            twiddle5c,
            twiddle6c,
            twiddle7c,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 });

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, output, 2, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        input: RawSlice<Complex<f32>>,
        output: RawSliceMut<Complex<f32>>,
    ) {
        let input_packed = read_complex_to_array!(input, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62});

        let values = interleave_complex_f32!(input_packed, 16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30});

        write_complex_to_array_strided!(out_sorted, output, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 });
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [__m128; 16]) -> [__m128; 16] {
        // we're going to hardcode a step of split radix

        // step 1: copy and reorder the input into the scratch
        let in0002 = extract_lo_lo_f32(input[0], input[1]);
        let in0406 = extract_lo_lo_f32(input[2], input[3]);
        let in0810 = extract_lo_lo_f32(input[4], input[5]);
        let in1214 = extract_lo_lo_f32(input[6], input[7]);
        let in1618 = extract_lo_lo_f32(input[8], input[9]);
        let in2022 = extract_lo_lo_f32(input[10], input[11]);
        let in2426 = extract_lo_lo_f32(input[12], input[13]);
        let in2830 = extract_lo_lo_f32(input[14], input[15]);

        let in0105 = extract_hi_hi_f32(input[0], input[2]);
        let in0913 = extract_hi_hi_f32(input[4], input[6]);
        let in1721 = extract_hi_hi_f32(input[8], input[10]);
        let in2529 = extract_hi_hi_f32(input[12], input[14]);

        let in3103 = extract_hi_hi_f32(input[15], input[1]);
        let in0711 = extract_hi_hi_f32(input[3], input[5]);
        let in1519 = extract_hi_hi_f32(input[7], input[9]);
        let in2327 = extract_hi_hi_f32(input[11], input[13]);

        let in_evens = [
            in0002, in0406, in0810, in1214, in1618, in2022, in2426, in2830,
        ];

        // step 2: column FFTs
        let evens = self.bf16.perform_fft_direct(in_evens);
        let mut odds1 = self
            .bf8
            .perform_fft_direct([in0105, in0913, in1721, in2529]);
        let mut odds3 = self
            .bf8
            .perform_fft_direct([in3103, in0711, in1519, in2327]);

        // step 3: apply twiddle factors
        odds1[0] = mul_complex_f32(odds1[0], self.twiddle01);
        odds3[0] = mul_complex_f32(odds3[0], self.twiddle01conj);

        odds1[1] = mul_complex_f32(odds1[1], self.twiddle23);
        odds3[1] = mul_complex_f32(odds3[1], self.twiddle23conj);

        odds1[2] = mul_complex_f32(odds1[2], self.twiddle45);
        odds3[2] = mul_complex_f32(odds3[2], self.twiddle45conj);

        odds1[3] = mul_complex_f32(odds1[3], self.twiddle67);
        odds3[3] = mul_complex_f32(odds3[3], self.twiddle67conj);

        // step 4: cross FFTs
        let mut temp0 = parallel_fft2_interleaved_f32(odds1[0], odds3[0]);
        let mut temp1 = parallel_fft2_interleaved_f32(odds1[1], odds3[1]);
        let mut temp2 = parallel_fft2_interleaved_f32(odds1[2], odds3[2]);
        let mut temp3 = parallel_fft2_interleaved_f32(odds1[3], odds3[3]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate_both(temp0[1]);
        temp1[1] = self.rotate90.rotate_both(temp1[1]);
        temp2[1] = self.rotate90.rotate_both(temp2[1]);
        temp3[1] = self.rotate90.rotate_both(temp3[1]);

        //step 5: copy/add/subtract data back to buffer
        [
            _mm_add_ps(evens[0], temp0[0]),
            _mm_add_ps(evens[1], temp1[0]),
            _mm_add_ps(evens[2], temp2[0]),
            _mm_add_ps(evens[3], temp3[0]),
            _mm_add_ps(evens[4], temp0[1]),
            _mm_add_ps(evens[5], temp1[1]),
            _mm_add_ps(evens[6], temp2[1]),
            _mm_add_ps(evens[7], temp3[1]),
            _mm_sub_ps(evens[0], temp0[0]),
            _mm_sub_ps(evens[1], temp1[0]),
            _mm_sub_ps(evens[2], temp2[0]),
            _mm_sub_ps(evens[3], temp3[0]),
            _mm_sub_ps(evens[4], temp0[1]),
            _mm_sub_ps(evens[5], temp1[1]),
            _mm_sub_ps(evens[6], temp2[1]),
            _mm_sub_ps(evens[7], temp3[1]),
        ]
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, input: [__m128; 32]) -> [__m128; 32] {
        // we're going to hardcode a step of split radix

        // step 1: copy and reorder the  input into the scratch
        // and
        // step 2: column FFTs
        let evens = self.bf16.perform_parallel_fft_direct([
            input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14],
            input[16], input[18], input[20], input[22], input[24], input[26], input[28], input[30],
        ]);
        let mut odds1 = self.bf8.perform_parallel_fft_direct([
            input[1], input[5], input[9], input[13], input[17], input[21], input[25], input[29],
        ]);
        let mut odds3 = self.bf8.perform_parallel_fft_direct([
            input[31], input[3], input[7], input[11], input[15], input[19], input[23], input[27],
        ]);

        // step 3: apply twiddle factors
        odds1[1] = mul_complex_f32(odds1[1], self.twiddle1);
        odds3[1] = mul_complex_f32(odds3[1], self.twiddle1c);

        odds1[2] = mul_complex_f32(odds1[2], self.twiddle2);
        odds3[2] = mul_complex_f32(odds3[2], self.twiddle2c);

        odds1[3] = mul_complex_f32(odds1[3], self.twiddle3);
        odds3[3] = mul_complex_f32(odds3[3], self.twiddle3c);

        odds1[4] = mul_complex_f32(odds1[4], self.twiddle4);
        odds3[4] = mul_complex_f32(odds3[4], self.twiddle4c);

        odds1[5] = mul_complex_f32(odds1[5], self.twiddle5);
        odds3[5] = mul_complex_f32(odds3[5], self.twiddle5c);

        odds1[6] = mul_complex_f32(odds1[6], self.twiddle6);
        odds3[6] = mul_complex_f32(odds3[6], self.twiddle6c);

        odds1[7] = mul_complex_f32(odds1[7], self.twiddle7);
        odds3[7] = mul_complex_f32(odds3[7], self.twiddle7c);

        // step 4: cross FFTs
        let mut temp0 = parallel_fft2_interleaved_f32(odds1[0], odds3[0]);
        let mut temp1 = parallel_fft2_interleaved_f32(odds1[1], odds3[1]);
        let mut temp2 = parallel_fft2_interleaved_f32(odds1[2], odds3[2]);
        let mut temp3 = parallel_fft2_interleaved_f32(odds1[3], odds3[3]);
        let mut temp4 = parallel_fft2_interleaved_f32(odds1[4], odds3[4]);
        let mut temp5 = parallel_fft2_interleaved_f32(odds1[5], odds3[5]);
        let mut temp6 = parallel_fft2_interleaved_f32(odds1[6], odds3[6]);
        let mut temp7 = parallel_fft2_interleaved_f32(odds1[7], odds3[7]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate_both(temp0[1]);
        temp1[1] = self.rotate90.rotate_both(temp1[1]);
        temp2[1] = self.rotate90.rotate_both(temp2[1]);
        temp3[1] = self.rotate90.rotate_both(temp3[1]);
        temp4[1] = self.rotate90.rotate_both(temp4[1]);
        temp5[1] = self.rotate90.rotate_both(temp5[1]);
        temp6[1] = self.rotate90.rotate_both(temp6[1]);
        temp7[1] = self.rotate90.rotate_both(temp7[1]);

        //step 5: copy/add/subtract data back to buffer
        [
            _mm_add_ps(evens[0], temp0[0]),
            _mm_add_ps(evens[1], temp1[0]),
            _mm_add_ps(evens[2], temp2[0]),
            _mm_add_ps(evens[3], temp3[0]),
            _mm_add_ps(evens[4], temp4[0]),
            _mm_add_ps(evens[5], temp5[0]),
            _mm_add_ps(evens[6], temp6[0]),
            _mm_add_ps(evens[7], temp7[0]),
            _mm_add_ps(evens[8], temp0[1]),
            _mm_add_ps(evens[9], temp1[1]),
            _mm_add_ps(evens[10], temp2[1]),
            _mm_add_ps(evens[11], temp3[1]),
            _mm_add_ps(evens[12], temp4[1]),
            _mm_add_ps(evens[13], temp5[1]),
            _mm_add_ps(evens[14], temp6[1]),
            _mm_add_ps(evens[15], temp7[1]),
            _mm_sub_ps(evens[0], temp0[0]),
            _mm_sub_ps(evens[1], temp1[0]),
            _mm_sub_ps(evens[2], temp2[0]),
            _mm_sub_ps(evens[3], temp3[0]),
            _mm_sub_ps(evens[4], temp4[0]),
            _mm_sub_ps(evens[5], temp5[0]),
            _mm_sub_ps(evens[6], temp6[0]),
            _mm_sub_ps(evens[7], temp7[0]),
            _mm_sub_ps(evens[8], temp0[1]),
            _mm_sub_ps(evens[9], temp1[1]),
            _mm_sub_ps(evens[10], temp2[1]),
            _mm_sub_ps(evens[11], temp3[1]),
            _mm_sub_ps(evens[12], temp4[1]),
            _mm_sub_ps(evens[13], temp5[1]),
            _mm_sub_ps(evens[14], temp6[1]),
            _mm_sub_ps(evens[15], temp7[1]),
        ]
    }
}

//   _________             __   _  _   _     _ _
//  |___ /___ \           / /_ | || | | |__ (_) |_
//    |_ \ __) |  _____  | '_ \| || |_| '_ \| | __|
//   ___) / __/  |_____| | (_) |__   _| |_) | | |_
//  |____/_____|          \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly32<T> {
    direction: FftDirection,
    bf8: SseF64Butterfly8<T>,
    bf16: SseF64Butterfly16<T>,
    rotate90: Rotate90F64,
    twiddle1: __m128d,
    twiddle2: __m128d,
    twiddle3: __m128d,
    twiddle4: __m128d,
    twiddle5: __m128d,
    twiddle6: __m128d,
    twiddle7: __m128d,
    twiddle1c: __m128d,
    twiddle2c: __m128d,
    twiddle3c: __m128d,
    twiddle4c: __m128d,
    twiddle5c: __m128d,
    twiddle6c: __m128d,
    twiddle7c: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly32, 32, |this: &SseF64Butterfly32<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly32, 32, |this: &SseF64Butterfly32<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly32<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf8 = SseF64Butterfly8::new(direction);
        let bf16 = SseF64Butterfly16::new(direction);
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F64::new(true)
        } else {
            Rotate90F64::new(false)
        };
        let twiddle1 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(1, 32, direction).re as *const f64) };
        let twiddle2 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(2, 32, direction).re as *const f64) };
        let twiddle3 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(3, 32, direction).re as *const f64) };
        let twiddle4 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(4, 32, direction).re as *const f64) };
        let twiddle5 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(5, 32, direction).re as *const f64) };
        let twiddle6 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(6, 32, direction).re as *const f64) };
        let twiddle7 =
            unsafe { _mm_loadu_pd(&twiddles::compute_twiddle(7, 32, direction).re as *const f64) };
        let twiddle1c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(1, 32, direction).conj().re as *const f64)
        };
        let twiddle2c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(2, 32, direction).conj().re as *const f64)
        };
        let twiddle3c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(3, 32, direction).conj().re as *const f64)
        };
        let twiddle4c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(4, 32, direction).conj().re as *const f64)
        };
        let twiddle5c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(5, 32, direction).conj().re as *const f64)
        };
        let twiddle6c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(6, 32, direction).conj().re as *const f64)
        };
        let twiddle7c = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(7, 32, direction).conj().re as *const f64)
        };

        Self {
            direction,
            bf8,
            bf16,
            rotate90,
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            twiddle7,
            twiddle1c,
            twiddle2c,
            twiddle3c,
            twiddle4c,
            twiddle5c,
            twiddle6c,
            twiddle7c,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<f64>>,
        output: RawSliceMut<Complex<f64>>,
    ) {
        let values = read_complex_to_array!(input, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, output, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [__m128d; 32]) -> [__m128d; 32] {
        // we're going to hardcode a step of split radix

        // step 1: copy and reorder the  input into the scratch
        // and
        // step 2: column FFTs
        let evens = self.bf16.perform_fft_direct([
            input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14],
            input[16], input[18], input[20], input[22], input[24], input[26], input[28], input[30],
        ]);
        let mut odds1 = self.bf8.perform_fft_direct([
            input[1], input[5], input[9], input[13], input[17], input[21], input[25], input[29],
        ]);
        let mut odds3 = self.bf8.perform_fft_direct([
            input[31], input[3], input[7], input[11], input[15], input[19], input[23], input[27],
        ]);

        // step 3: apply twiddle factors
        odds1[1] = mul_complex_f64(odds1[1], self.twiddle1);
        odds3[1] = mul_complex_f64(odds3[1], self.twiddle1c);

        odds1[2] = mul_complex_f64(odds1[2], self.twiddle2);
        odds3[2] = mul_complex_f64(odds3[2], self.twiddle2c);

        odds1[3] = mul_complex_f64(odds1[3], self.twiddle3);
        odds3[3] = mul_complex_f64(odds3[3], self.twiddle3c);

        odds1[4] = mul_complex_f64(odds1[4], self.twiddle4);
        odds3[4] = mul_complex_f64(odds3[4], self.twiddle4c);

        odds1[5] = mul_complex_f64(odds1[5], self.twiddle5);
        odds3[5] = mul_complex_f64(odds3[5], self.twiddle5c);

        odds1[6] = mul_complex_f64(odds1[6], self.twiddle6);
        odds3[6] = mul_complex_f64(odds3[6], self.twiddle6c);

        odds1[7] = mul_complex_f64(odds1[7], self.twiddle7);
        odds3[7] = mul_complex_f64(odds3[7], self.twiddle7c);

        // step 4: cross FFTs
        let mut temp0 = solo_fft2_f64(odds1[0], odds3[0]);
        let mut temp1 = solo_fft2_f64(odds1[1], odds3[1]);
        let mut temp2 = solo_fft2_f64(odds1[2], odds3[2]);
        let mut temp3 = solo_fft2_f64(odds1[3], odds3[3]);
        let mut temp4 = solo_fft2_f64(odds1[4], odds3[4]);
        let mut temp5 = solo_fft2_f64(odds1[5], odds3[5]);
        let mut temp6 = solo_fft2_f64(odds1[6], odds3[6]);
        let mut temp7 = solo_fft2_f64(odds1[7], odds3[7]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate(temp0[1]);
        temp1[1] = self.rotate90.rotate(temp1[1]);
        temp2[1] = self.rotate90.rotate(temp2[1]);
        temp3[1] = self.rotate90.rotate(temp3[1]);
        temp4[1] = self.rotate90.rotate(temp4[1]);
        temp5[1] = self.rotate90.rotate(temp5[1]);
        temp6[1] = self.rotate90.rotate(temp6[1]);
        temp7[1] = self.rotate90.rotate(temp7[1]);

        //step 5: copy/add/subtract data back to buffer
        [
            _mm_add_pd(evens[0], temp0[0]),
            _mm_add_pd(evens[1], temp1[0]),
            _mm_add_pd(evens[2], temp2[0]),
            _mm_add_pd(evens[3], temp3[0]),
            _mm_add_pd(evens[4], temp4[0]),
            _mm_add_pd(evens[5], temp5[0]),
            _mm_add_pd(evens[6], temp6[0]),
            _mm_add_pd(evens[7], temp7[0]),
            _mm_add_pd(evens[8], temp0[1]),
            _mm_add_pd(evens[9], temp1[1]),
            _mm_add_pd(evens[10], temp2[1]),
            _mm_add_pd(evens[11], temp3[1]),
            _mm_add_pd(evens[12], temp4[1]),
            _mm_add_pd(evens[13], temp5[1]),
            _mm_add_pd(evens[14], temp6[1]),
            _mm_add_pd(evens[15], temp7[1]),
            _mm_sub_pd(evens[0], temp0[0]),
            _mm_sub_pd(evens[1], temp1[0]),
            _mm_sub_pd(evens[2], temp2[0]),
            _mm_sub_pd(evens[3], temp3[0]),
            _mm_sub_pd(evens[4], temp4[0]),
            _mm_sub_pd(evens[5], temp5[0]),
            _mm_sub_pd(evens[6], temp6[0]),
            _mm_sub_pd(evens[7], temp7[0]),
            _mm_sub_pd(evens[8], temp0[1]),
            _mm_sub_pd(evens[9], temp1[1]),
            _mm_sub_pd(evens[10], temp2[1]),
            _mm_sub_pd(evens[11], temp3[1]),
            _mm_sub_pd(evens[12], temp4[1]),
            _mm_sub_pd(evens[13], temp5[1]),
            _mm_sub_pd(evens[14], temp6[1]),
            _mm_sub_pd(evens[15], temp7[1]),
        ]
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::algorithm::Dft;
    use crate::test_utils::{check_fft_algorithm, compare_vectors};

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_32_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(FftDirection::Forward);
                check_fft_algorithm::<f32>(&butterfly, $size, FftDirection::Forward);

                let butterfly_direction = $struct_name::new(FftDirection::Inverse);
                check_fft_algorithm::<f32>(&butterfly_direction, $size, FftDirection::Inverse);
            }
        };
    }
    test_butterfly_32_func!(test_ssef32_butterfly2, SseF32Butterfly2, 2);
    test_butterfly_32_func!(test_ssef32_butterfly3, SseF32Butterfly3, 3);
    test_butterfly_32_func!(test_ssef32_butterfly4, SseF32Butterfly4, 4);
    test_butterfly_32_func!(test_ssef32_butterfly5, SseF32Butterfly5, 5);
    test_butterfly_32_func!(test_ssef32_butterfly6, SseF32Butterfly6, 6);
    test_butterfly_32_func!(test_ssef32_butterfly8, SseF32Butterfly8, 8);
    test_butterfly_32_func!(test_ssef32_butterfly9, SseF32Butterfly9, 9);
    test_butterfly_32_func!(test_ssef32_butterfly10, SseF32Butterfly10, 10);
    test_butterfly_32_func!(test_ssef32_butterfly12, SseF32Butterfly12, 12);
    test_butterfly_32_func!(test_ssef32_butterfly15, SseF32Butterfly15, 15);
    test_butterfly_32_func!(test_ssef32_butterfly16, SseF32Butterfly16, 16);
    test_butterfly_32_func!(test_ssef32_butterfly32, SseF32Butterfly32, 32);

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_64_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(FftDirection::Forward);
                check_fft_algorithm::<f64>(&butterfly, $size, FftDirection::Forward);

                let butterfly_direction = $struct_name::new(FftDirection::Inverse);
                check_fft_algorithm::<f64>(&butterfly_direction, $size, FftDirection::Inverse);
            }
        };
    }
    test_butterfly_64_func!(test_ssef64_butterfly2, SseF64Butterfly2, 2);
    test_butterfly_64_func!(test_ssef64_butterfly3, SseF64Butterfly3, 3);
    test_butterfly_64_func!(test_ssef64_butterfly4, SseF64Butterfly4, 4);
    test_butterfly_64_func!(test_ssef64_butterfly5, SseF64Butterfly5, 5);
    test_butterfly_64_func!(test_ssef64_butterfly6, SseF64Butterfly6, 6);
    test_butterfly_64_func!(test_ssef64_butterfly8, SseF64Butterfly8, 8);
    test_butterfly_64_func!(test_ssef64_butterfly9, SseF64Butterfly9, 9);
    test_butterfly_64_func!(test_ssef64_butterfly10, SseF64Butterfly10, 10);
    test_butterfly_64_func!(test_ssef64_butterfly12, SseF64Butterfly12, 12);
    test_butterfly_64_func!(test_ssef64_butterfly15, SseF64Butterfly15, 15);
    test_butterfly_64_func!(test_ssef64_butterfly16, SseF64Butterfly16, 16);
    test_butterfly_64_func!(test_ssef64_butterfly32, SseF64Butterfly32, 32);

    #[test]
    fn test_mul_complex_f64() {
        unsafe {
            let right = _mm_set_pd(1.0, 2.0);
            let left = _mm_set_pd(5.0, 7.0);
            let res = mul_complex_f64(left, right);
            let expected = _mm_set_pd(2.0 * 5.0 + 1.0 * 7.0, 2.0 * 7.0 - 1.0 * 5.0);
            assert_eq!(
                std::mem::transmute::<__m128d, Complex<f64>>(res),
                std::mem::transmute::<__m128d, Complex<f64>>(expected)
            );
        }
    }

    #[test]
    fn test_mul_complex_f32() {
        unsafe {
            let val1 = Complex::<f32>::new(1.0, 2.5);
            let val2 = Complex::<f32>::new(3.2, 4.2);
            let val3 = Complex::<f32>::new(5.6, 6.2);
            let val4 = Complex::<f32>::new(7.4, 8.3);

            let nbr2 = _mm_set_ps(val4.im, val4.re, val3.im, val3.re);
            let nbr1 = _mm_set_ps(val2.im, val2.re, val1.im, val1.re);
            let res = mul_complex_f32(nbr1, nbr2);
            let res = std::mem::transmute::<__m128, [Complex<f32>; 2]>(res);
            let expected = [val1 * val3, val2 * val4];
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn test_parallel_fft4_32() {
        unsafe {
            let val_a1 = Complex::<f32>::new(1.0, 2.5);
            let val_a2 = Complex::<f32>::new(3.2, 4.2);
            let val_a3 = Complex::<f32>::new(5.6, 6.2);
            let val_a4 = Complex::<f32>::new(7.4, 8.3);

            let val_b1 = Complex::<f32>::new(6.0, 24.5);
            let val_b2 = Complex::<f32>::new(4.2, 34.2);
            let val_b3 = Complex::<f32>::new(9.6, 61.2);
            let val_b4 = Complex::<f32>::new(17.4, 81.3);

            let p1 = _mm_set_ps(val_b1.im, val_b1.re, val_a1.im, val_a1.re);
            let p2 = _mm_set_ps(val_b2.im, val_b2.re, val_a2.im, val_a2.re);
            let p3 = _mm_set_ps(val_b3.im, val_b3.re, val_a3.im, val_a3.re);
            let p4 = _mm_set_ps(val_b4.im, val_b4.re, val_a4.im, val_a4.re);

            let mut val_a = vec![val_a1, val_a2, val_a3, val_a4];
            let mut val_b = vec![val_b1, val_b2, val_b3, val_b4];

            let dft = Dft::new(4, FftDirection::Forward);

            let bf4 = SseF32Butterfly4::<f32>::new(FftDirection::Forward);

            dft.process(&mut val_a);
            dft.process(&mut val_b);
            let res_both = bf4.perform_parallel_fft_direct(p1, p2, p3, p4);

            let res = std::mem::transmute::<[__m128; 4], [Complex<f32>; 8]>(res_both);
            let sse_res_a = [res[0], res[2], res[4], res[6]];
            let sse_res_b = [res[1], res[3], res[5], res[7]];
            assert!(compare_vectors(&val_a, &sse_res_a));
            assert!(compare_vectors(&val_b, &sse_res_b));
        }
    }

    #[test]
    fn test_pack() {
        unsafe {
            let nbr2 = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
            let nbr1 = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
            let first = extract_lo_lo_f32(nbr1, nbr2);
            let second = extract_hi_hi_f32(nbr1, nbr2);
            let first = std::mem::transmute::<__m128, [Complex<f32>; 2]>(first);
            let second = std::mem::transmute::<__m128, [Complex<f32>; 2]>(second);
            let first_expected = [Complex::new(1.0, 2.0), Complex::new(5.0, 6.0)];
            let second_expected = [Complex::new(3.0, 4.0), Complex::new(7.0, 8.0)];
            assert_eq!(first, first_expected);
            assert_eq!(second, second_expected);
        }
    }
}
