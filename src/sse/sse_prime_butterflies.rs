use core::arch::x86_64::*;
use num_complex::Complex;
//use std::mem::transmute;
//use std::time::{Duration, Instant};

use crate::{common::FftNum, FftDirection};

use crate::array_utils;
use crate::array_utils::{RawSlice, RawSliceMut};
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::twiddles;
use crate::{Direction, Fft, Length};

use super::sse_common::{assert_f32, assert_f64};
use super::sse_utils::*;

//   _____           _________  _     _ _
//  |___  |         |___ /___ \| |__ (_) |_
//     / /   _____    |_ \ __) | '_ \| | __|
//    / /   |_____|  ___) / __/| |_) | | |_
//   /_/            |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly7<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
    twiddle3re: __m128,
    twiddle3im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly7, 7, |this: &SseF32Butterfly7<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly7, 7, |this: &SseF32Butterfly7<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly7<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 7, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 7, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 7, direction);
        let twiddle1re = unsafe { _mm_set_ps(tw1.re, tw1.re, tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_ps(tw1.im, tw1.im, tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_ps(tw2.re, tw2.re, tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_ps(tw2.im, tw2.im, tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_ps(tw3.re, tw3.re, tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_ps(tw3.im, tw3.im, tw3.im, tw3.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let v1 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(1) as *const f64));
        let v2 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(2) as *const f64));
        let v3 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(3) as *const f64));
        let v4 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(4) as *const f64));
        let v5 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(5) as *const f64));
        let v6 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(6) as *const f64));

        let temp = self.perform_dual_fft_direct([v0, v1, v2, v3, v4, v5, v6]);

        let array = std::mem::transmute::<[__m128; 7], [Complex<f32>; 14]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[2];
        *output_slice.add(2) = array[4];
        *output_slice.add(3) = array[6];
        *output_slice.add(4) = array[8];
        *output_slice.add(5) = array[10];
        *output_slice.add(6) = array[12];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4a5 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valuea6b0 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let valueb5b6 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);

        let v0 = pack_1and2_f32(valuea0a1, valuea6b0);
        let v1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let v2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let v3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let v4 = pack_1and2_f32(valuea4a5, valueb3b4);
        let v5 = pack_2and1_f32(valuea4a5, valueb5b6);
        let v6 = pack_1and2_f32(valuea6b0, valueb5b6);

        let out = self.perform_dual_fft_direct([v0, v1, v2, v3, v4, v5, v6]);

        let val = std::mem::transmute::<[__m128; 7], [Complex<f32>; 14]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[1];
        *output_slice.add(8) = val[3];
        *output_slice.add(9) = val[5];
        *output_slice.add(10) = val[7];
        *output_slice.add(11) = val[9];
        *output_slice.add(12) = val[11];
        *output_slice.add(13) = val[13];
    }

    // length 3 dual fft of a, given as (a0, b0), (a1, b1), (a2, b2).
    // result is [(A0, B0), (A1, B1), (A2, B2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values: [__m128; 7]) -> [__m128; 7] {
        // This is a SSE translation of the scalar 7-point butterfly
        let x16p = _mm_add_ps(values[1], values[6]);
        let x16n = _mm_sub_ps(values[1], values[6]);
        let x25p = _mm_add_ps(values[2], values[5]);
        let x25n = _mm_sub_ps(values[2], values[5]);
        let x34p = _mm_add_ps(values[3], values[4]);
        let x34n = _mm_sub_ps(values[3], values[4]);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x16p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x25p);
        let temp_a1_3 = _mm_mul_ps(self.twiddle3re, x34p);

        let temp_a2_1 = _mm_mul_ps(self.twiddle1re, x34p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle2re, x16p);
        let temp_a2_3 = _mm_mul_ps(self.twiddle3re, x25p);

        let temp_a3_1 = _mm_mul_ps(self.twiddle1re, x25p);
        let temp_a3_2 = _mm_mul_ps(self.twiddle2re, x34p);
        let temp_a3_3 = _mm_mul_ps(self.twiddle3re, x16p);

        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x16n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x25n);
        let temp_b1_3 = _mm_mul_ps(self.twiddle3im, x34n);

        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x16n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle3im, x25n);
        let temp_b2_3 = _mm_mul_ps(self.twiddle1im, x34n);

        let temp_b3_1 = _mm_mul_ps(self.twiddle3im, x16n);
        let temp_b3_2 = _mm_mul_ps(self.twiddle1im, x25n);
        let temp_b3_3 = _mm_mul_ps(self.twiddle2im, x34n);

        let temp_a1 = _mm_add_ps(
            _mm_add_ps(values[0], temp_a1_1),
            _mm_add_ps(temp_a1_2, temp_a1_3),
        );
        let temp_a2 = _mm_add_ps(
            _mm_add_ps(values[0], temp_a2_1),
            _mm_add_ps(temp_a2_2, temp_a2_3),
        );
        let temp_a3 = _mm_add_ps(
            _mm_add_ps(values[0], temp_a3_1),
            _mm_add_ps(temp_a3_2, temp_a3_3),
        );

        let temp_b1 = _mm_add_ps(temp_b1_1, _mm_add_ps(temp_b1_2, temp_b1_3));
        let temp_b2 = _mm_sub_ps(temp_b2_1, _mm_add_ps(temp_b2_2, temp_b2_3));
        let temp_b3 = _mm_sub_ps(temp_b3_1, _mm_sub_ps(temp_b3_2, temp_b3_3));

        let temp_b1_rot = self.rotate.rotate_both(temp_b1);
        let temp_b2_rot = self.rotate.rotate_both(temp_b2);
        let temp_b3_rot = self.rotate.rotate_both(temp_b3);
        let x0 = _mm_add_ps(_mm_add_ps(values[0], x16p), _mm_add_ps(x25p, x34p));
        let x1 = _mm_add_ps(temp_a1, temp_b1_rot);
        let x2 = _mm_add_ps(temp_a2, temp_b2_rot);
        let x3 = _mm_add_ps(temp_a3, temp_b3_rot);
        let x4 = _mm_sub_ps(temp_a3, temp_b3_rot);
        let x5 = _mm_sub_ps(temp_a2, temp_b2_rot);
        let x6 = _mm_sub_ps(temp_a1, temp_b1_rot);
        [x0, x1, x2, x3, x4, x5, x6]
    }
}

//   _____            __   _  _   _     _ _
//  |___  |          / /_ | || | | |__ (_) |_
//     / /   _____  | '_ \| || |_| '_ \| | __|
//    / /   |_____| | (_) |__   _| |_) | | |_
//   /_/             \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly7<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
    twiddle3re: __m128d,
    twiddle3im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly7, 7, |this: &SseF64Butterfly7<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly7, 7, |this: &SseF64Butterfly7<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly7<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 7, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 7, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 7, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_pd(tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_pd(tw3.im, tw3.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let v1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let v2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let v3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let v4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let v5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let v6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);

        let out = self.perform_fft_direct([v0, v1, v2, v3, v4, v5, v6]);

        let val = std::mem::transmute::<[__m128d; 7], [Complex<f64>; 7]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
        *output_slice.add(6) = val[6];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 7]) -> [__m128d; 7] {
        // This is a SSE translation of the scalar 7-point butterfly
        let x16p = _mm_add_pd(values[1], values[6]);
        let x16n = _mm_sub_pd(values[1], values[6]);
        let x25p = _mm_add_pd(values[2], values[5]);
        let x25n = _mm_sub_pd(values[2], values[5]);
        let x34p = _mm_add_pd(values[3], values[4]);
        let x34n = _mm_sub_pd(values[3], values[4]);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x16p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x25p);
        let temp_a1_3 = _mm_mul_pd(self.twiddle3re, x34p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x16p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle3re, x25p);
        let temp_a2_3 = _mm_mul_pd(self.twiddle1re, x34p);
        let temp_a3_1 = _mm_mul_pd(self.twiddle3re, x16p);
        let temp_a3_2 = _mm_mul_pd(self.twiddle1re, x25p);
        let temp_a3_3 = _mm_mul_pd(self.twiddle2re, x34p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x16n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x25n);
        let temp_b1_3 = _mm_mul_pd(self.twiddle3im, x34n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x16n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle3im, x25n);
        let temp_b2_3 = _mm_mul_pd(self.twiddle1im, x34n);
        let temp_b3_1 = _mm_mul_pd(self.twiddle3im, x16n);
        let temp_b3_2 = _mm_mul_pd(self.twiddle1im, x25n);
        let temp_b3_3 = _mm_mul_pd(self.twiddle2im, x34n);

        let temp_a1 = _mm_add_pd(
            values[0],
            _mm_add_pd(temp_a1_1, _mm_add_pd(temp_a1_2, temp_a1_3)),
        );
        let temp_a2 = _mm_add_pd(
            values[0],
            _mm_add_pd(temp_a2_1, _mm_add_pd(temp_a2_2, temp_a2_3)),
        );
        let temp_a3 = _mm_add_pd(
            values[0],
            _mm_add_pd(temp_a3_1, _mm_add_pd(temp_a3_2, temp_a3_3)),
        );

        let temp_b1 = _mm_add_pd(temp_b1_1, _mm_add_pd(temp_b1_2, temp_b1_3));
        let temp_b2 = _mm_sub_pd(temp_b2_1, _mm_add_pd(temp_b2_2, temp_b2_3));
        let temp_b3 = _mm_sub_pd(temp_b3_1, _mm_sub_pd(temp_b3_2, temp_b3_3));

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);

        let x0 = _mm_add_pd(values[0], _mm_add_pd(x16p, _mm_add_pd(x25p, x34p)));
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
        let x4 = _mm_sub_pd(temp_a3, temp_b3_rot);
        let x5 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x6 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [x0, x1, x2, x3, x4, x5, x6]
    }
}

//   _ _           _________  _     _ _
//  / / |         |___ /___ \| |__ (_) |_
//  | | |  _____    |_ \ __) | '_ \| | __|
//  | | | |_____|  ___) / __/| |_) | | |_
//  |_|_|         |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly11<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
    twiddle3re: __m128,
    twiddle3im: __m128,
    twiddle4re: __m128,
    twiddle4im: __m128,
    twiddle5re: __m128,
    twiddle5im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly11, 11, |this: &SseF32Butterfly11<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly11, 11, |this: &SseF32Butterfly11<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly11<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 11, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 11, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 11, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 11, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 11, direction);
        let twiddle1re = unsafe { _mm_load1_ps(&tw1.re) };
        let twiddle1im = unsafe { _mm_load1_ps(&tw1.im) };
        let twiddle2re = unsafe { _mm_load1_ps(&tw2.re) };
        let twiddle2im = unsafe { _mm_load1_ps(&tw2.im) };
        let twiddle3re = unsafe { _mm_load1_ps(&tw3.re) };
        let twiddle3im = unsafe { _mm_load1_ps(&tw3.im) };
        let twiddle4re = unsafe { _mm_load1_ps(&tw4.re) };
        let twiddle4im = unsafe { _mm_load1_ps(&tw4.im) };
        let twiddle5re = unsafe { _mm_load1_ps(&tw5.re) };
        let twiddle5im = unsafe { _mm_load1_ps(&tw5.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let v1 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(1) as *const f64));
        let v2 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(2) as *const f64));
        let v3 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(3) as *const f64));
        let v4 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(4) as *const f64));
        let v5 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(5) as *const f64));
        let v6 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(6) as *const f64));
        let v7 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(7) as *const f64));
        let v8 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(8) as *const f64));
        let v9 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(9) as *const f64));
        let v10 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(10) as *const f64));

        let out = self.perform_dual_fft_direct([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]);

        let val = std::mem::transmute::<[__m128; 11], [Complex<f32>; 22]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4a5 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valuea6a7 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valuea8a9 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let valuea10b0 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let valueb5b6 = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let valueb7b8 = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let valueb9b10 = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);

        let v0 = pack_1and2_f32(valuea0a1, valuea10b0);
        let v1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let v2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let v3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let v4 = pack_1and2_f32(valuea4a5, valueb3b4);
        let v5 = pack_2and1_f32(valuea4a5, valueb5b6);
        let v6 = pack_1and2_f32(valuea6a7, valueb5b6);
        let v7 = pack_2and1_f32(valuea6a7, valueb7b8);
        let v8 = pack_1and2_f32(valuea8a9, valueb7b8);
        let v9 = pack_2and1_f32(valuea8a9, valueb9b10);
        let v10 = pack_1and2_f32(valuea10b0, valueb9b10);

        let out = self.perform_dual_fft_direct([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]);

        let val = std::mem::transmute::<[__m128; 11], [Complex<f32>; 22]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[1];
        *output_slice.add(12) = val[3];
        *output_slice.add(13) = val[5];
        *output_slice.add(14) = val[7];
        *output_slice.add(15) = val[9];
        *output_slice.add(16) = val[11];
        *output_slice.add(17) = val[13];
        *output_slice.add(18) = val[15];
        *output_slice.add(19) = val[17];
        *output_slice.add(20) = val[19];
        *output_slice.add(21) = val[21];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values: [__m128; 11]) -> [__m128; 11] {
        // This is a SSE translation of the scalar 11-point butterfly
        let x110p = _mm_add_ps(values[1], values[10]);
        let x110n = _mm_sub_ps(values[1], values[10]);
        let x29p = _mm_add_ps(values[2], values[9]);
        let x29n = _mm_sub_ps(values[2], values[9]);
        let x38p = _mm_add_ps(values[3], values[8]);
        let x38n = _mm_sub_ps(values[3], values[8]);
        let x47p = _mm_add_ps(values[4], values[7]);
        let x47n = _mm_sub_ps(values[4], values[7]);
        let x56p = _mm_add_ps(values[5], values[6]);
        let x56n = _mm_sub_ps(values[5], values[6]);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x110p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x29p);
        let temp_a1_3 = _mm_mul_ps(self.twiddle3re, x38p);
        let temp_a1_4 = _mm_mul_ps(self.twiddle4re, x47p);
        let temp_a1_5 = _mm_mul_ps(self.twiddle5re, x56p);
        let temp_a2_1 = _mm_mul_ps(self.twiddle2re, x110p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle4re, x29p);
        let temp_a2_3 = _mm_mul_ps(self.twiddle5re, x38p);
        let temp_a2_4 = _mm_mul_ps(self.twiddle3re, x47p);
        let temp_a2_5 = _mm_mul_ps(self.twiddle1re, x56p);
        let temp_a3_1 = _mm_mul_ps(self.twiddle3re, x110p);
        let temp_a3_2 = _mm_mul_ps(self.twiddle5re, x29p);
        let temp_a3_3 = _mm_mul_ps(self.twiddle2re, x38p);
        let temp_a3_4 = _mm_mul_ps(self.twiddle1re, x47p);
        let temp_a3_5 = _mm_mul_ps(self.twiddle4re, x56p);
        let temp_a4_1 = _mm_mul_ps(self.twiddle4re, x110p);
        let temp_a4_2 = _mm_mul_ps(self.twiddle3re, x29p);
        let temp_a4_3 = _mm_mul_ps(self.twiddle1re, x38p);
        let temp_a4_4 = _mm_mul_ps(self.twiddle5re, x47p);
        let temp_a4_5 = _mm_mul_ps(self.twiddle2re, x56p);
        let temp_a5_1 = _mm_mul_ps(self.twiddle5re, x110p);
        let temp_a5_2 = _mm_mul_ps(self.twiddle1re, x29p);
        let temp_a5_3 = _mm_mul_ps(self.twiddle4re, x38p);
        let temp_a5_4 = _mm_mul_ps(self.twiddle2re, x47p);
        let temp_a5_5 = _mm_mul_ps(self.twiddle3re, x56p);

        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x110n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x29n);
        let temp_b1_3 = _mm_mul_ps(self.twiddle3im, x38n);
        let temp_b1_4 = _mm_mul_ps(self.twiddle4im, x47n);
        let temp_b1_5 = _mm_mul_ps(self.twiddle5im, x56n);
        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x110n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle4im, x29n);
        let temp_b2_3 = _mm_mul_ps(self.twiddle5im, x38n);
        let temp_b2_4 = _mm_mul_ps(self.twiddle3im, x47n);
        let temp_b2_5 = _mm_mul_ps(self.twiddle1im, x56n);
        let temp_b3_1 = _mm_mul_ps(self.twiddle3im, x110n);
        let temp_b3_2 = _mm_mul_ps(self.twiddle5im, x29n);
        let temp_b3_3 = _mm_mul_ps(self.twiddle2im, x38n);
        let temp_b3_4 = _mm_mul_ps(self.twiddle1im, x47n);
        let temp_b3_5 = _mm_mul_ps(self.twiddle4im, x56n);
        let temp_b4_1 = _mm_mul_ps(self.twiddle4im, x110n);
        let temp_b4_2 = _mm_mul_ps(self.twiddle3im, x29n);
        let temp_b4_3 = _mm_mul_ps(self.twiddle1im, x38n);
        let temp_b4_4 = _mm_mul_ps(self.twiddle5im, x47n);
        let temp_b4_5 = _mm_mul_ps(self.twiddle2im, x56n);
        let temp_b5_1 = _mm_mul_ps(self.twiddle5im, x110n);
        let temp_b5_2 = _mm_mul_ps(self.twiddle1im, x29n);
        let temp_b5_3 = _mm_mul_ps(self.twiddle4im, x38n);
        let temp_b5_4 = _mm_mul_ps(self.twiddle2im, x47n);
        let temp_b5_5 = _mm_mul_ps(self.twiddle3im, x56n);

        let temp_a1 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a1_1,
                _mm_add_ps(
                    temp_a1_2,
                    _mm_add_ps(temp_a1_3, _mm_add_ps(temp_a1_4, temp_a1_5)),
                ),
            ),
        );
        let temp_a2 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a2_1,
                _mm_add_ps(
                    temp_a2_2,
                    _mm_add_ps(temp_a2_3, _mm_add_ps(temp_a2_4, temp_a2_5)),
                ),
            ),
        );
        let temp_a3 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a3_1,
                _mm_add_ps(
                    temp_a3_2,
                    _mm_add_ps(temp_a3_3, _mm_add_ps(temp_a3_4, temp_a3_5)),
                ),
            ),
        );
        let temp_a4 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a4_1,
                _mm_add_ps(
                    temp_a4_2,
                    _mm_add_ps(temp_a4_3, _mm_add_ps(temp_a4_4, temp_a4_5)),
                ),
            ),
        );
        let temp_a5 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a5_1,
                _mm_add_ps(
                    temp_a5_2,
                    _mm_add_ps(temp_a5_3, _mm_add_ps(temp_a5_4, temp_a5_5)),
                ),
            ),
        );

        let temp_b1 = _mm_add_ps(
            temp_b1_1,
            _mm_add_ps(
                temp_b1_2,
                _mm_add_ps(temp_b1_3, _mm_add_ps(temp_b1_4, temp_b1_5)),
            ),
        );
        let temp_b2 = _mm_add_ps(
            temp_b2_1,
            _mm_sub_ps(
                temp_b2_2,
                _mm_add_ps(temp_b2_3, _mm_add_ps(temp_b2_4, temp_b2_5)),
            ),
        );
        let temp_b3 = _mm_sub_ps(
            temp_b3_1,
            _mm_add_ps(
                temp_b3_2,
                _mm_sub_ps(temp_b3_3, _mm_add_ps(temp_b3_4, temp_b3_5)),
            ),
        );
        let temp_b4 = _mm_sub_ps(
            temp_b4_1,
            _mm_sub_ps(
                temp_b4_2,
                _mm_add_ps(temp_b4_3, _mm_sub_ps(temp_b4_4, temp_b4_5)),
            ),
        );
        let temp_b5 = _mm_sub_ps(
            temp_b5_1,
            _mm_sub_ps(
                temp_b5_2,
                _mm_sub_ps(temp_b5_3, _mm_sub_ps(temp_b5_4, temp_b5_5)),
            ),
        );

        let temp_b1_rot = self.rotate.rotate_both(temp_b1);
        let temp_b2_rot = self.rotate.rotate_both(temp_b2);
        let temp_b3_rot = self.rotate.rotate_both(temp_b3);
        let temp_b4_rot = self.rotate.rotate_both(temp_b4);
        let temp_b5_rot = self.rotate.rotate_both(temp_b5);

        let x0 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                x110p,
                _mm_add_ps(x29p, _mm_add_ps(x38p, _mm_add_ps(x47p, x56p))),
            ),
        );
        let x1 = _mm_add_ps(temp_a1, temp_b1_rot);
        let x2 = _mm_add_ps(temp_a2, temp_b2_rot);
        let x3 = _mm_add_ps(temp_a3, temp_b3_rot);
        let x4 = _mm_add_ps(temp_a4, temp_b4_rot);
        let x5 = _mm_add_ps(temp_a5, temp_b5_rot);
        let x6 = _mm_sub_ps(temp_a5, temp_b5_rot);
        let x7 = _mm_sub_ps(temp_a4, temp_b4_rot);
        let x8 = _mm_sub_ps(temp_a3, temp_b3_rot);
        let x9 = _mm_sub_ps(temp_a2, temp_b2_rot);
        let x10 = _mm_sub_ps(temp_a1, temp_b1_rot);
        [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    }
}

//   _ _            __   _  _   _     _ _
//  / / |          / /_ | || | | |__ (_) |_
//  | | |  _____  | '_ \| || |_| '_ \| | __|
//  | | | |_____| | (_) |__   _| |_) | | |_
//  |_|_|          \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly11<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
    twiddle3re: __m128d,
    twiddle3im: __m128d,
    twiddle4re: __m128d,
    twiddle4im: __m128d,
    twiddle5re: __m128d,
    twiddle5im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly11, 11, |this: &SseF64Butterfly11<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly11, 11, |this: &SseF64Butterfly11<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly11<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 11, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 11, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 11, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 11, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 11, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_pd(tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_pd(tw3.im, tw3.im) };
        let twiddle4re = unsafe { _mm_set_pd(tw4.re, tw4.re) };
        let twiddle4im = unsafe { _mm_set_pd(tw4.im, tw4.im) };
        let twiddle5re = unsafe { _mm_set_pd(tw5.re, tw5.re) };
        let twiddle5im = unsafe { _mm_set_pd(tw5.im, tw5.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let v1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let v2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let v3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let v4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let v5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let v6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let v7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let v8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let v9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let v10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);

        let out = self.perform_fft_direct([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]);

        let val = std::mem::transmute::<[__m128d; 11], [Complex<f64>; 11]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
        *output_slice.add(6) = val[6];
        *output_slice.add(7) = val[7];
        *output_slice.add(8) = val[8];
        *output_slice.add(9) = val[9];
        *output_slice.add(10) = val[10];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 11]) -> [__m128d; 11] {
        // This is a SSE translation of the scalar 11-point butterfly
        let x110p = _mm_add_pd(values[1], values[10]);
        let x110n = _mm_sub_pd(values[1], values[10]);
        let x29p = _mm_add_pd(values[2], values[9]);
        let x29n = _mm_sub_pd(values[2], values[9]);
        let x38p = _mm_add_pd(values[3], values[8]);
        let x38n = _mm_sub_pd(values[3], values[8]);
        let x47p = _mm_add_pd(values[4], values[7]);
        let x47n = _mm_sub_pd(values[4], values[7]);
        let x56p = _mm_add_pd(values[5], values[6]);
        let x56n = _mm_sub_pd(values[5], values[6]);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x110p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x29p);
        let temp_a1_3 = _mm_mul_pd(self.twiddle3re, x38p);
        let temp_a1_4 = _mm_mul_pd(self.twiddle4re, x47p);
        let temp_a1_5 = _mm_mul_pd(self.twiddle5re, x56p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x110p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle4re, x29p);
        let temp_a2_3 = _mm_mul_pd(self.twiddle5re, x38p);
        let temp_a2_4 = _mm_mul_pd(self.twiddle3re, x47p);
        let temp_a2_5 = _mm_mul_pd(self.twiddle1re, x56p);
        let temp_a3_1 = _mm_mul_pd(self.twiddle3re, x110p);
        let temp_a3_2 = _mm_mul_pd(self.twiddle5re, x29p);
        let temp_a3_3 = _mm_mul_pd(self.twiddle2re, x38p);
        let temp_a3_4 = _mm_mul_pd(self.twiddle1re, x47p);
        let temp_a3_5 = _mm_mul_pd(self.twiddle4re, x56p);
        let temp_a4_1 = _mm_mul_pd(self.twiddle4re, x110p);
        let temp_a4_2 = _mm_mul_pd(self.twiddle3re, x29p);
        let temp_a4_3 = _mm_mul_pd(self.twiddle1re, x38p);
        let temp_a4_4 = _mm_mul_pd(self.twiddle5re, x47p);
        let temp_a4_5 = _mm_mul_pd(self.twiddle2re, x56p);
        let temp_a5_1 = _mm_mul_pd(self.twiddle5re, x110p);
        let temp_a5_2 = _mm_mul_pd(self.twiddle1re, x29p);
        let temp_a5_3 = _mm_mul_pd(self.twiddle4re, x38p);
        let temp_a5_4 = _mm_mul_pd(self.twiddle2re, x47p);
        let temp_a5_5 = _mm_mul_pd(self.twiddle3re, x56p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x110n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x29n);
        let temp_b1_3 = _mm_mul_pd(self.twiddle3im, x38n);
        let temp_b1_4 = _mm_mul_pd(self.twiddle4im, x47n);
        let temp_b1_5 = _mm_mul_pd(self.twiddle5im, x56n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x110n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle4im, x29n);
        let temp_b2_3 = _mm_mul_pd(self.twiddle5im, x38n);
        let temp_b2_4 = _mm_mul_pd(self.twiddle3im, x47n);
        let temp_b2_5 = _mm_mul_pd(self.twiddle1im, x56n);
        let temp_b3_1 = _mm_mul_pd(self.twiddle3im, x110n);
        let temp_b3_2 = _mm_mul_pd(self.twiddle5im, x29n);
        let temp_b3_3 = _mm_mul_pd(self.twiddle2im, x38n);
        let temp_b3_4 = _mm_mul_pd(self.twiddle1im, x47n);
        let temp_b3_5 = _mm_mul_pd(self.twiddle4im, x56n);
        let temp_b4_1 = _mm_mul_pd(self.twiddle4im, x110n);
        let temp_b4_2 = _mm_mul_pd(self.twiddle3im, x29n);
        let temp_b4_3 = _mm_mul_pd(self.twiddle1im, x38n);
        let temp_b4_4 = _mm_mul_pd(self.twiddle5im, x47n);
        let temp_b4_5 = _mm_mul_pd(self.twiddle2im, x56n);
        let temp_b5_1 = _mm_mul_pd(self.twiddle5im, x110n);
        let temp_b5_2 = _mm_mul_pd(self.twiddle1im, x29n);
        let temp_b5_3 = _mm_mul_pd(self.twiddle4im, x38n);
        let temp_b5_4 = _mm_mul_pd(self.twiddle2im, x47n);
        let temp_b5_5 = _mm_mul_pd(self.twiddle3im, x56n);

        let temp_a1 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a1_1,
                _mm_add_pd(
                    temp_a1_2,
                    _mm_add_pd(temp_a1_3, _mm_add_pd(temp_a1_4, temp_a1_5)),
                ),
            ),
        );
        let temp_a2 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a2_1,
                _mm_add_pd(
                    temp_a2_2,
                    _mm_add_pd(temp_a2_3, _mm_add_pd(temp_a2_4, temp_a2_5)),
                ),
            ),
        );
        let temp_a3 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a3_1,
                _mm_add_pd(
                    temp_a3_2,
                    _mm_add_pd(temp_a3_3, _mm_add_pd(temp_a3_4, temp_a3_5)),
                ),
            ),
        );
        let temp_a4 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a4_1,
                _mm_add_pd(
                    temp_a4_2,
                    _mm_add_pd(temp_a4_3, _mm_add_pd(temp_a4_4, temp_a4_5)),
                ),
            ),
        );
        let temp_a5 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a5_1,
                _mm_add_pd(
                    temp_a5_2,
                    _mm_add_pd(temp_a5_3, _mm_add_pd(temp_a5_4, temp_a5_5)),
                ),
            ),
        );

        let temp_b1 = _mm_add_pd(
            temp_b1_1,
            _mm_add_pd(
                temp_b1_2,
                _mm_add_pd(temp_b1_3, _mm_add_pd(temp_b1_4, temp_b1_5)),
            ),
        );
        let temp_b2 = _mm_add_pd(
            temp_b2_1,
            _mm_sub_pd(
                temp_b2_2,
                _mm_add_pd(temp_b2_3, _mm_add_pd(temp_b2_4, temp_b2_5)),
            ),
        );
        let temp_b3 = _mm_sub_pd(
            temp_b3_1,
            _mm_add_pd(
                temp_b3_2,
                _mm_sub_pd(temp_b3_3, _mm_add_pd(temp_b3_4, temp_b3_5)),
            ),
        );
        let temp_b4 = _mm_sub_pd(
            temp_b4_1,
            _mm_sub_pd(
                temp_b4_2,
                _mm_add_pd(temp_b4_3, _mm_sub_pd(temp_b4_4, temp_b4_5)),
            ),
        );
        let temp_b5 = _mm_sub_pd(
            temp_b5_1,
            _mm_sub_pd(
                temp_b5_2,
                _mm_sub_pd(temp_b5_3, _mm_sub_pd(temp_b5_4, temp_b5_5)),
            ),
        );

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);

        let x0 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                x110p,
                _mm_add_pd(x29p, _mm_add_pd(x38p, _mm_add_pd(x47p, x56p))),
            ),
        );
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
        let x4 = _mm_add_pd(temp_a4, temp_b4_rot);
        let x5 = _mm_add_pd(temp_a5, temp_b5_rot);
        let x6 = _mm_sub_pd(temp_a5, temp_b5_rot);
        let x7 = _mm_sub_pd(temp_a4, temp_b4_rot);
        let x8 = _mm_sub_pd(temp_a3, temp_b3_rot);
        let x9 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x10 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    }
}

//   _ _____           _________  _     _ _
//  / |___ /          |___ /___ \| |__ (_) |_
//  | | |_ \   _____    |_ \ __) | '_ \| | __|
//  | |___) | |_____|  ___) / __/| |_) | | |_
//  |_|____/          |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly13<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
    twiddle3re: __m128,
    twiddle3im: __m128,
    twiddle4re: __m128,
    twiddle4im: __m128,
    twiddle5re: __m128,
    twiddle5im: __m128,
    twiddle6re: __m128,
    twiddle6im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly13, 13, |this: &SseF32Butterfly13<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly13, 13, |this: &SseF32Butterfly13<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly13<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 13, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 13, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 13, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 13, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 13, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 13, direction);
        let twiddle1re = unsafe { _mm_load1_ps(&tw1.re) };
        let twiddle1im = unsafe { _mm_load1_ps(&tw1.im) };
        let twiddle2re = unsafe { _mm_load1_ps(&tw2.re) };
        let twiddle2im = unsafe { _mm_load1_ps(&tw2.im) };
        let twiddle3re = unsafe { _mm_load1_ps(&tw3.re) };
        let twiddle3im = unsafe { _mm_load1_ps(&tw3.im) };
        let twiddle4re = unsafe { _mm_load1_ps(&tw4.re) };
        let twiddle4im = unsafe { _mm_load1_ps(&tw4.im) };
        let twiddle5re = unsafe { _mm_load1_ps(&tw5.re) };
        let twiddle5im = unsafe { _mm_load1_ps(&tw5.im) };
        let twiddle6re = unsafe { _mm_load1_ps(&tw6.re) };
        let twiddle6im = unsafe { _mm_load1_ps(&tw6.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let v1 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(1) as *const f64));
        let v2 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(2) as *const f64));
        let v3 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(3) as *const f64));
        let v4 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(4) as *const f64));
        let v5 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(5) as *const f64));
        let v6 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(6) as *const f64));
        let v7 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(7) as *const f64));
        let v8 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(8) as *const f64));
        let v9 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(9) as *const f64));
        let v10 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(10) as *const f64));
        let v11 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(11) as *const f64));
        let v12 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(12) as *const f64));

        let out =
            self.perform_dual_fft_direct([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]);

        let val = std::mem::transmute::<[__m128; 13], [Complex<f32>; 26]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4a5 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valuea6a7 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valuea8a9 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let valuea10a11 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let valuea12b0 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let valueb5b6 = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let valueb7b8 = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let valueb9b10 = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let valueb11b12 = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);

        let v0 = pack_1and2_f32(valuea0a1, valuea12b0);
        let v1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let v2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let v3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let v4 = pack_1and2_f32(valuea4a5, valueb3b4);
        let v5 = pack_2and1_f32(valuea4a5, valueb5b6);
        let v6 = pack_1and2_f32(valuea6a7, valueb5b6);
        let v7 = pack_2and1_f32(valuea6a7, valueb7b8);
        let v8 = pack_1and2_f32(valuea8a9, valueb7b8);
        let v9 = pack_2and1_f32(valuea8a9, valueb9b10);
        let v10 = pack_1and2_f32(valuea10a11, valueb9b10);
        let v11 = pack_2and1_f32(valuea10a11, valueb11b12);
        let v12 = pack_1and2_f32(valuea12b0, valueb11b12);

        let out =
            self.perform_dual_fft_direct([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]);

        let val = std::mem::transmute::<[__m128; 13], [Complex<f32>; 26]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[1];
        *output_slice.add(14) = val[3];
        *output_slice.add(15) = val[5];
        *output_slice.add(16) = val[7];
        *output_slice.add(17) = val[9];
        *output_slice.add(18) = val[11];
        *output_slice.add(19) = val[13];
        *output_slice.add(20) = val[15];
        *output_slice.add(21) = val[17];
        *output_slice.add(22) = val[19];
        *output_slice.add(23) = val[21];
        *output_slice.add(24) = val[23];
        *output_slice.add(25) = val[25];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values: [__m128; 13]) -> [__m128; 13] {
        // This is a SSE translation of the scalar 13-point butterfly
        let x112p = _mm_add_ps(values[1], values[12]);
        let x112n = _mm_sub_ps(values[1], values[12]);
        let x211p = _mm_add_ps(values[2], values[11]);
        let x211n = _mm_sub_ps(values[2], values[11]);
        let x310p = _mm_add_ps(values[3], values[10]);
        let x310n = _mm_sub_ps(values[3], values[10]);
        let x49p = _mm_add_ps(values[4], values[9]);
        let x49n = _mm_sub_ps(values[4], values[9]);
        let x58p = _mm_add_ps(values[5], values[8]);
        let x58n = _mm_sub_ps(values[5], values[8]);
        let x67p = _mm_add_ps(values[6], values[7]);
        let x67n = _mm_sub_ps(values[6], values[7]);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x112p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x211p);
        let temp_a1_3 = _mm_mul_ps(self.twiddle3re, x310p);
        let temp_a1_4 = _mm_mul_ps(self.twiddle4re, x49p);
        let temp_a1_5 = _mm_mul_ps(self.twiddle5re, x58p);
        let temp_a1_6 = _mm_mul_ps(self.twiddle6re, x67p);
        let temp_a2_1 = _mm_mul_ps(self.twiddle2re, x112p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle4re, x211p);
        let temp_a2_3 = _mm_mul_ps(self.twiddle6re, x310p);
        let temp_a2_4 = _mm_mul_ps(self.twiddle5re, x49p);
        let temp_a2_5 = _mm_mul_ps(self.twiddle3re, x58p);
        let temp_a2_6 = _mm_mul_ps(self.twiddle1re, x67p);
        let temp_a3_1 = _mm_mul_ps(self.twiddle3re, x112p);
        let temp_a3_2 = _mm_mul_ps(self.twiddle6re, x211p);
        let temp_a3_3 = _mm_mul_ps(self.twiddle4re, x310p);
        let temp_a3_4 = _mm_mul_ps(self.twiddle1re, x49p);
        let temp_a3_5 = _mm_mul_ps(self.twiddle2re, x58p);
        let temp_a3_6 = _mm_mul_ps(self.twiddle5re, x67p);
        let temp_a4_1 = _mm_mul_ps(self.twiddle4re, x112p);
        let temp_a4_2 = _mm_mul_ps(self.twiddle5re, x211p);
        let temp_a4_3 = _mm_mul_ps(self.twiddle1re, x310p);
        let temp_a4_4 = _mm_mul_ps(self.twiddle3re, x49p);
        let temp_a4_5 = _mm_mul_ps(self.twiddle6re, x58p);
        let temp_a4_6 = _mm_mul_ps(self.twiddle2re, x67p);
        let temp_a5_1 = _mm_mul_ps(self.twiddle5re, x112p);
        let temp_a5_2 = _mm_mul_ps(self.twiddle3re, x211p);
        let temp_a5_3 = _mm_mul_ps(self.twiddle2re, x310p);
        let temp_a5_4 = _mm_mul_ps(self.twiddle6re, x49p);
        let temp_a5_5 = _mm_mul_ps(self.twiddle1re, x58p);
        let temp_a5_6 = _mm_mul_ps(self.twiddle4re, x67p);
        let temp_a6_1 = _mm_mul_ps(self.twiddle6re, x112p);
        let temp_a6_2 = _mm_mul_ps(self.twiddle1re, x211p);
        let temp_a6_3 = _mm_mul_ps(self.twiddle5re, x310p);
        let temp_a6_4 = _mm_mul_ps(self.twiddle2re, x49p);
        let temp_a6_5 = _mm_mul_ps(self.twiddle4re, x58p);
        let temp_a6_6 = _mm_mul_ps(self.twiddle3re, x67p);

        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x112n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x211n);
        let temp_b1_3 = _mm_mul_ps(self.twiddle3im, x310n);
        let temp_b1_4 = _mm_mul_ps(self.twiddle4im, x49n);
        let temp_b1_5 = _mm_mul_ps(self.twiddle5im, x58n);
        let temp_b1_6 = _mm_mul_ps(self.twiddle6im, x67n);
        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x112n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle4im, x211n);
        let temp_b2_3 = _mm_mul_ps(self.twiddle6im, x310n);
        let temp_b2_4 = _mm_mul_ps(self.twiddle5im, x49n);
        let temp_b2_5 = _mm_mul_ps(self.twiddle3im, x58n);
        let temp_b2_6 = _mm_mul_ps(self.twiddle1im, x67n);
        let temp_b3_1 = _mm_mul_ps(self.twiddle3im, x112n);
        let temp_b3_2 = _mm_mul_ps(self.twiddle6im, x211n);
        let temp_b3_3 = _mm_mul_ps(self.twiddle4im, x310n);
        let temp_b3_4 = _mm_mul_ps(self.twiddle1im, x49n);
        let temp_b3_5 = _mm_mul_ps(self.twiddle2im, x58n);
        let temp_b3_6 = _mm_mul_ps(self.twiddle5im, x67n);
        let temp_b4_1 = _mm_mul_ps(self.twiddle4im, x112n);
        let temp_b4_2 = _mm_mul_ps(self.twiddle5im, x211n);
        let temp_b4_3 = _mm_mul_ps(self.twiddle1im, x310n);
        let temp_b4_4 = _mm_mul_ps(self.twiddle3im, x49n);
        let temp_b4_5 = _mm_mul_ps(self.twiddle6im, x58n);
        let temp_b4_6 = _mm_mul_ps(self.twiddle2im, x67n);
        let temp_b5_1 = _mm_mul_ps(self.twiddle5im, x112n);
        let temp_b5_2 = _mm_mul_ps(self.twiddle3im, x211n);
        let temp_b5_3 = _mm_mul_ps(self.twiddle2im, x310n);
        let temp_b5_4 = _mm_mul_ps(self.twiddle6im, x49n);
        let temp_b5_5 = _mm_mul_ps(self.twiddle1im, x58n);
        let temp_b5_6 = _mm_mul_ps(self.twiddle4im, x67n);
        let temp_b6_1 = _mm_mul_ps(self.twiddle6im, x112n);
        let temp_b6_2 = _mm_mul_ps(self.twiddle1im, x211n);
        let temp_b6_3 = _mm_mul_ps(self.twiddle5im, x310n);
        let temp_b6_4 = _mm_mul_ps(self.twiddle2im, x49n);
        let temp_b6_5 = _mm_mul_ps(self.twiddle4im, x58n);
        let temp_b6_6 = _mm_mul_ps(self.twiddle3im, x67n);

        let temp_a1 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a1_1,
                _mm_add_ps(
                    temp_a1_2,
                    _mm_add_ps(
                        temp_a1_3,
                        _mm_add_ps(temp_a1_4, _mm_add_ps(temp_a1_5, temp_a1_6)),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a2_1,
                _mm_add_ps(
                    temp_a2_2,
                    _mm_add_ps(
                        temp_a2_3,
                        _mm_add_ps(temp_a2_4, _mm_add_ps(temp_a2_5, temp_a2_6)),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a3_1,
                _mm_add_ps(
                    temp_a3_2,
                    _mm_add_ps(
                        temp_a3_3,
                        _mm_add_ps(temp_a3_4, _mm_add_ps(temp_a3_5, temp_a3_6)),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a4_1,
                _mm_add_ps(
                    temp_a4_2,
                    _mm_add_ps(
                        temp_a4_3,
                        _mm_add_ps(temp_a4_4, _mm_add_ps(temp_a4_5, temp_a4_6)),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a5_1,
                _mm_add_ps(
                    temp_a5_2,
                    _mm_add_ps(
                        temp_a5_3,
                        _mm_add_ps(temp_a5_4, _mm_add_ps(temp_a5_5, temp_a5_6)),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a6_1,
                _mm_add_ps(
                    temp_a6_2,
                    _mm_add_ps(
                        temp_a6_3,
                        _mm_add_ps(temp_a6_4, _mm_add_ps(temp_a6_5, temp_a6_6)),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_ps(
            temp_b1_1,
            _mm_add_ps(
                temp_b1_2,
                _mm_add_ps(
                    temp_b1_3,
                    _mm_add_ps(temp_b1_4, _mm_add_ps(temp_b1_5, temp_b1_6)),
                ),
            ),
        );
        let temp_b2 = _mm_add_ps(
            temp_b2_1,
            _mm_add_ps(
                temp_b2_2,
                _mm_sub_ps(
                    temp_b2_3,
                    _mm_add_ps(temp_b2_4, _mm_add_ps(temp_b2_5, temp_b2_6)),
                ),
            ),
        );
        let temp_b3 = _mm_add_ps(
            temp_b3_1,
            _mm_sub_ps(
                temp_b3_2,
                _mm_add_ps(
                    temp_b3_3,
                    _mm_sub_ps(temp_b3_4, _mm_add_ps(temp_b3_5, temp_b3_6)),
                ),
            ),
        );
        let temp_b4 = _mm_sub_ps(
            temp_b4_1,
            _mm_add_ps(
                temp_b4_2,
                _mm_sub_ps(
                    temp_b4_3,
                    _mm_sub_ps(temp_b4_4, _mm_add_ps(temp_b4_5, temp_b4_6)),
                ),
            ),
        );
        let temp_b5 = _mm_sub_ps(
            temp_b5_1,
            _mm_sub_ps(
                temp_b5_2,
                _mm_sub_ps(
                    temp_b5_3,
                    _mm_add_ps(temp_b5_4, _mm_sub_ps(temp_b5_5, temp_b5_6)),
                ),
            ),
        );
        let temp_b6 = _mm_sub_ps(
            temp_b6_1,
            _mm_sub_ps(
                temp_b6_2,
                _mm_sub_ps(
                    temp_b6_3,
                    _mm_sub_ps(temp_b6_4, _mm_sub_ps(temp_b6_5, temp_b6_6)),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate_both(temp_b1);
        let temp_b2_rot = self.rotate.rotate_both(temp_b2);
        let temp_b3_rot = self.rotate.rotate_both(temp_b3);
        let temp_b4_rot = self.rotate.rotate_both(temp_b4);
        let temp_b5_rot = self.rotate.rotate_both(temp_b5);
        let temp_b6_rot = self.rotate.rotate_both(temp_b6);

        let x0 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                x112p,
                _mm_add_ps(
                    x211p,
                    _mm_add_ps(x310p, _mm_add_ps(x49p, _mm_add_ps(x58p, x67p))),
                ),
            ),
        );
        let x1 = _mm_add_ps(temp_a1, temp_b1_rot);
        let x2 = _mm_add_ps(temp_a2, temp_b2_rot);
        let x3 = _mm_add_ps(temp_a3, temp_b3_rot);
        let x4 = _mm_add_ps(temp_a4, temp_b4_rot);
        let x5 = _mm_add_ps(temp_a5, temp_b5_rot);
        let x6 = _mm_add_ps(temp_a6, temp_b6_rot);
        let x7 = _mm_sub_ps(temp_a6, temp_b6_rot);
        let x8 = _mm_sub_ps(temp_a5, temp_b5_rot);
        let x9 = _mm_sub_ps(temp_a4, temp_b4_rot);
        let x10 = _mm_sub_ps(temp_a3, temp_b3_rot);
        let x11 = _mm_sub_ps(temp_a2, temp_b2_rot);
        let x12 = _mm_sub_ps(temp_a1, temp_b1_rot);
        [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]
    }
}

//   _ _____            __   _  _   _     _ _
//  / |___ /           / /_ | || | | |__ (_) |_
//  | | |_ \   _____  | '_ \| || |_| '_ \| | __|
//  | |___) | |_____| | (_) |__   _| |_) | | |_
//  |_|____/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly13<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
    twiddle3re: __m128d,
    twiddle3im: __m128d,
    twiddle4re: __m128d,
    twiddle4im: __m128d,
    twiddle5re: __m128d,
    twiddle5im: __m128d,
    twiddle6re: __m128d,
    twiddle6im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly13, 13, |this: &SseF64Butterfly13<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly13, 13, |this: &SseF64Butterfly13<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly13<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 13, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 13, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 13, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 13, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 13, direction);
        let tw6: Complex<f64> = twiddles::compute_twiddle(6, 13, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_pd(tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_pd(tw3.im, tw3.im) };
        let twiddle4re = unsafe { _mm_set_pd(tw4.re, tw4.re) };
        let twiddle4im = unsafe { _mm_set_pd(tw4.im, tw4.im) };
        let twiddle5re = unsafe { _mm_set_pd(tw5.re, tw5.re) };
        let twiddle5im = unsafe { _mm_set_pd(tw5.im, tw5.im) };
        let twiddle6re = unsafe { _mm_set_pd(tw6.re, tw6.re) };
        let twiddle6im = unsafe { _mm_set_pd(tw6.im, tw6.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let v1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let v2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let v3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let v4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let v5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let v6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let v7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let v8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let v9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let v10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);
        let v11 = _mm_loadu_pd(input.as_ptr().add(11) as *const f64);
        let v12 = _mm_loadu_pd(input.as_ptr().add(12) as *const f64);

        let out = self.perform_fft_direct([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]);

        let val = std::mem::transmute::<[__m128d; 13], [Complex<f64>; 13]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
        *output_slice.add(6) = val[6];
        *output_slice.add(7) = val[7];
        *output_slice.add(8) = val[8];
        *output_slice.add(9) = val[9];
        *output_slice.add(10) = val[10];
        *output_slice.add(11) = val[11];
        *output_slice.add(12) = val[12];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 13]) -> [__m128d; 13] {
        // This is a SSE translation of the scalar 13-point butterfly
        let x112p = _mm_add_pd(values[1], values[12]);
        let x112n = _mm_sub_pd(values[1], values[12]);
        let x211p = _mm_add_pd(values[2], values[11]);
        let x211n = _mm_sub_pd(values[2], values[11]);
        let x310p = _mm_add_pd(values[3], values[10]);
        let x310n = _mm_sub_pd(values[3], values[10]);
        let x49p = _mm_add_pd(values[4], values[9]);
        let x49n = _mm_sub_pd(values[4], values[9]);
        let x58p = _mm_add_pd(values[5], values[8]);
        let x58n = _mm_sub_pd(values[5], values[8]);
        let x67p = _mm_add_pd(values[6], values[7]);
        let x67n = _mm_sub_pd(values[6], values[7]);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x112p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x211p);
        let temp_a1_3 = _mm_mul_pd(self.twiddle3re, x310p);
        let temp_a1_4 = _mm_mul_pd(self.twiddle4re, x49p);
        let temp_a1_5 = _mm_mul_pd(self.twiddle5re, x58p);
        let temp_a1_6 = _mm_mul_pd(self.twiddle6re, x67p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x112p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle4re, x211p);
        let temp_a2_3 = _mm_mul_pd(self.twiddle6re, x310p);
        let temp_a2_4 = _mm_mul_pd(self.twiddle5re, x49p);
        let temp_a2_5 = _mm_mul_pd(self.twiddle3re, x58p);
        let temp_a2_6 = _mm_mul_pd(self.twiddle1re, x67p);
        let temp_a3_1 = _mm_mul_pd(self.twiddle3re, x112p);
        let temp_a3_2 = _mm_mul_pd(self.twiddle6re, x211p);
        let temp_a3_3 = _mm_mul_pd(self.twiddle4re, x310p);
        let temp_a3_4 = _mm_mul_pd(self.twiddle1re, x49p);
        let temp_a3_5 = _mm_mul_pd(self.twiddle2re, x58p);
        let temp_a3_6 = _mm_mul_pd(self.twiddle5re, x67p);
        let temp_a4_1 = _mm_mul_pd(self.twiddle4re, x112p);
        let temp_a4_2 = _mm_mul_pd(self.twiddle5re, x211p);
        let temp_a4_3 = _mm_mul_pd(self.twiddle1re, x310p);
        let temp_a4_4 = _mm_mul_pd(self.twiddle3re, x49p);
        let temp_a4_5 = _mm_mul_pd(self.twiddle6re, x58p);
        let temp_a4_6 = _mm_mul_pd(self.twiddle2re, x67p);
        let temp_a5_1 = _mm_mul_pd(self.twiddle5re, x112p);
        let temp_a5_2 = _mm_mul_pd(self.twiddle3re, x211p);
        let temp_a5_3 = _mm_mul_pd(self.twiddle2re, x310p);
        let temp_a5_4 = _mm_mul_pd(self.twiddle6re, x49p);
        let temp_a5_5 = _mm_mul_pd(self.twiddle1re, x58p);
        let temp_a5_6 = _mm_mul_pd(self.twiddle4re, x67p);
        let temp_a6_1 = _mm_mul_pd(self.twiddle6re, x112p);
        let temp_a6_2 = _mm_mul_pd(self.twiddle1re, x211p);
        let temp_a6_3 = _mm_mul_pd(self.twiddle5re, x310p);
        let temp_a6_4 = _mm_mul_pd(self.twiddle2re, x49p);
        let temp_a6_5 = _mm_mul_pd(self.twiddle4re, x58p);
        let temp_a6_6 = _mm_mul_pd(self.twiddle3re, x67p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x112n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x211n);
        let temp_b1_3 = _mm_mul_pd(self.twiddle3im, x310n);
        let temp_b1_4 = _mm_mul_pd(self.twiddle4im, x49n);
        let temp_b1_5 = _mm_mul_pd(self.twiddle5im, x58n);
        let temp_b1_6 = _mm_mul_pd(self.twiddle6im, x67n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x112n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle4im, x211n);
        let temp_b2_3 = _mm_mul_pd(self.twiddle6im, x310n);
        let temp_b2_4 = _mm_mul_pd(self.twiddle5im, x49n);
        let temp_b2_5 = _mm_mul_pd(self.twiddle3im, x58n);
        let temp_b2_6 = _mm_mul_pd(self.twiddle1im, x67n);
        let temp_b3_1 = _mm_mul_pd(self.twiddle3im, x112n);
        let temp_b3_2 = _mm_mul_pd(self.twiddle6im, x211n);
        let temp_b3_3 = _mm_mul_pd(self.twiddle4im, x310n);
        let temp_b3_4 = _mm_mul_pd(self.twiddle1im, x49n);
        let temp_b3_5 = _mm_mul_pd(self.twiddle2im, x58n);
        let temp_b3_6 = _mm_mul_pd(self.twiddle5im, x67n);
        let temp_b4_1 = _mm_mul_pd(self.twiddle4im, x112n);
        let temp_b4_2 = _mm_mul_pd(self.twiddle5im, x211n);
        let temp_b4_3 = _mm_mul_pd(self.twiddle1im, x310n);
        let temp_b4_4 = _mm_mul_pd(self.twiddle3im, x49n);
        let temp_b4_5 = _mm_mul_pd(self.twiddle6im, x58n);
        let temp_b4_6 = _mm_mul_pd(self.twiddle2im, x67n);
        let temp_b5_1 = _mm_mul_pd(self.twiddle5im, x112n);
        let temp_b5_2 = _mm_mul_pd(self.twiddle3im, x211n);
        let temp_b5_3 = _mm_mul_pd(self.twiddle2im, x310n);
        let temp_b5_4 = _mm_mul_pd(self.twiddle6im, x49n);
        let temp_b5_5 = _mm_mul_pd(self.twiddle1im, x58n);
        let temp_b5_6 = _mm_mul_pd(self.twiddle4im, x67n);
        let temp_b6_1 = _mm_mul_pd(self.twiddle6im, x112n);
        let temp_b6_2 = _mm_mul_pd(self.twiddle1im, x211n);
        let temp_b6_3 = _mm_mul_pd(self.twiddle5im, x310n);
        let temp_b6_4 = _mm_mul_pd(self.twiddle2im, x49n);
        let temp_b6_5 = _mm_mul_pd(self.twiddle4im, x58n);
        let temp_b6_6 = _mm_mul_pd(self.twiddle3im, x67n);

        let temp_a1 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a1_1,
                _mm_add_pd(
                    temp_a1_2,
                    _mm_add_pd(
                        temp_a1_3,
                        _mm_add_pd(temp_a1_4, _mm_add_pd(temp_a1_5, temp_a1_6)),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a2_1,
                _mm_add_pd(
                    temp_a2_2,
                    _mm_add_pd(
                        temp_a2_3,
                        _mm_add_pd(temp_a2_4, _mm_add_pd(temp_a2_5, temp_a2_6)),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a3_1,
                _mm_add_pd(
                    temp_a3_2,
                    _mm_add_pd(
                        temp_a3_3,
                        _mm_add_pd(temp_a3_4, _mm_add_pd(temp_a3_5, temp_a3_6)),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a4_1,
                _mm_add_pd(
                    temp_a4_2,
                    _mm_add_pd(
                        temp_a4_3,
                        _mm_add_pd(temp_a4_4, _mm_add_pd(temp_a4_5, temp_a4_6)),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a5_1,
                _mm_add_pd(
                    temp_a5_2,
                    _mm_add_pd(
                        temp_a5_3,
                        _mm_add_pd(temp_a5_4, _mm_add_pd(temp_a5_5, temp_a5_6)),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a6_1,
                _mm_add_pd(
                    temp_a6_2,
                    _mm_add_pd(
                        temp_a6_3,
                        _mm_add_pd(temp_a6_4, _mm_add_pd(temp_a6_5, temp_a6_6)),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_pd(
            temp_b1_1,
            _mm_add_pd(
                temp_b1_2,
                _mm_add_pd(
                    temp_b1_3,
                    _mm_add_pd(temp_b1_4, _mm_add_pd(temp_b1_5, temp_b1_6)),
                ),
            ),
        );
        let temp_b2 = _mm_add_pd(
            temp_b2_1,
            _mm_add_pd(
                temp_b2_2,
                _mm_sub_pd(
                    temp_b2_3,
                    _mm_add_pd(temp_b2_4, _mm_add_pd(temp_b2_5, temp_b2_6)),
                ),
            ),
        );
        let temp_b3 = _mm_add_pd(
            temp_b3_1,
            _mm_sub_pd(
                temp_b3_2,
                _mm_add_pd(
                    temp_b3_3,
                    _mm_sub_pd(temp_b3_4, _mm_add_pd(temp_b3_5, temp_b3_6)),
                ),
            ),
        );
        let temp_b4 = _mm_sub_pd(
            temp_b4_1,
            _mm_add_pd(
                temp_b4_2,
                _mm_sub_pd(
                    temp_b4_3,
                    _mm_sub_pd(temp_b4_4, _mm_add_pd(temp_b4_5, temp_b4_6)),
                ),
            ),
        );
        let temp_b5 = _mm_sub_pd(
            temp_b5_1,
            _mm_sub_pd(
                temp_b5_2,
                _mm_sub_pd(
                    temp_b5_3,
                    _mm_add_pd(temp_b5_4, _mm_sub_pd(temp_b5_5, temp_b5_6)),
                ),
            ),
        );
        let temp_b6 = _mm_sub_pd(
            temp_b6_1,
            _mm_sub_pd(
                temp_b6_2,
                _mm_sub_pd(
                    temp_b6_3,
                    _mm_sub_pd(temp_b6_4, _mm_sub_pd(temp_b6_5, temp_b6_6)),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);
        let temp_b6_rot = self.rotate.rotate(temp_b6);

        let x0 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                x112p,
                _mm_add_pd(
                    x211p,
                    _mm_add_pd(x310p, _mm_add_pd(x49p, _mm_add_pd(x58p, x67p))),
                ),
            ),
        );
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
        let x4 = _mm_add_pd(temp_a4, temp_b4_rot);
        let x5 = _mm_add_pd(temp_a5, temp_b5_rot);
        let x6 = _mm_add_pd(temp_a6, temp_b6_rot);
        let x7 = _mm_sub_pd(temp_a6, temp_b6_rot);
        let x8 = _mm_sub_pd(temp_a5, temp_b5_rot);
        let x9 = _mm_sub_pd(temp_a4, temp_b4_rot);
        let x10 = _mm_sub_pd(temp_a3, temp_b3_rot);
        let x11 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x12 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]
    }
}

//   _ _____           _________  _     _ _
//  / |___  |         |___ /___ \| |__ (_) |_
//  | |  / /   _____    |_ \ __) | '_ \| | __|
//  | | / /   |_____|  ___) / __/| |_) | | |_
//  |_|/_/            |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly17<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
    twiddle3re: __m128,
    twiddle3im: __m128,
    twiddle4re: __m128,
    twiddle4im: __m128,
    twiddle5re: __m128,
    twiddle5im: __m128,
    twiddle6re: __m128,
    twiddle6im: __m128,
    twiddle7re: __m128,
    twiddle7im: __m128,
    twiddle8re: __m128,
    twiddle8im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly17, 17, |this: &SseF32Butterfly17<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly17, 17, |this: &SseF32Butterfly17<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly17<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 17, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 17, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 17, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 17, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 17, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 17, direction);
        let tw7: Complex<f32> = twiddles::compute_twiddle(7, 17, direction);
        let tw8: Complex<f32> = twiddles::compute_twiddle(8, 17, direction);
        let twiddle1re = unsafe { _mm_load1_ps(&tw1.re) };
        let twiddle1im = unsafe { _mm_load1_ps(&tw1.im) };
        let twiddle2re = unsafe { _mm_load1_ps(&tw2.re) };
        let twiddle2im = unsafe { _mm_load1_ps(&tw2.im) };
        let twiddle3re = unsafe { _mm_load1_ps(&tw3.re) };
        let twiddle3im = unsafe { _mm_load1_ps(&tw3.im) };
        let twiddle4re = unsafe { _mm_load1_ps(&tw4.re) };
        let twiddle4im = unsafe { _mm_load1_ps(&tw4.im) };
        let twiddle5re = unsafe { _mm_load1_ps(&tw5.re) };
        let twiddle5im = unsafe { _mm_load1_ps(&tw5.im) };
        let twiddle6re = unsafe { _mm_load1_ps(&tw6.re) };
        let twiddle6im = unsafe { _mm_load1_ps(&tw6.im) };
        let twiddle7re = unsafe { _mm_load1_ps(&tw7.re) };
        let twiddle7im = unsafe { _mm_load1_ps(&tw7.im) };
        let twiddle8re = unsafe { _mm_load1_ps(&tw8.re) };
        let twiddle8im = unsafe { _mm_load1_ps(&tw8.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let v1 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(1) as *const f64));
        let v2 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(2) as *const f64));
        let v3 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(3) as *const f64));
        let v4 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(4) as *const f64));
        let v5 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(5) as *const f64));
        let v6 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(6) as *const f64));
        let v7 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(7) as *const f64));
        let v8 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(8) as *const f64));
        let v9 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(9) as *const f64));
        let v10 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(10) as *const f64));
        let v11 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(11) as *const f64));
        let v12 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(12) as *const f64));
        let v13 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(13) as *const f64));
        let v14 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(14) as *const f64));
        let v15 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(15) as *const f64));
        let v16 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(16) as *const f64));

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16,
        ]);

        let val = std::mem::transmute::<[__m128; 17], [Complex<f32>; 34]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4a5 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valuea6a7 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valuea8a9 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let valuea10a11 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let valuea12a13 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let valuea14a15 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let valuea16b0 = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let valueb5b6 = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let valueb7b8 = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);
        let valueb9b10 = _mm_loadu_ps(input.as_ptr().add(26) as *const f32);
        let valueb11b12 = _mm_loadu_ps(input.as_ptr().add(28) as *const f32);
        let valueb13b14 = _mm_loadu_ps(input.as_ptr().add(30) as *const f32);
        let valueb15b16 = _mm_loadu_ps(input.as_ptr().add(32) as *const f32);

        let v0 = pack_1and2_f32(valuea0a1, valuea16b0);
        let v1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let v2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let v3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let v4 = pack_1and2_f32(valuea4a5, valueb3b4);
        let v5 = pack_2and1_f32(valuea4a5, valueb5b6);
        let v6 = pack_1and2_f32(valuea6a7, valueb5b6);
        let v7 = pack_2and1_f32(valuea6a7, valueb7b8);
        let v8 = pack_1and2_f32(valuea8a9, valueb7b8);
        let v9 = pack_2and1_f32(valuea8a9, valueb9b10);
        let v10 = pack_1and2_f32(valuea10a11, valueb9b10);
        let v11 = pack_2and1_f32(valuea10a11, valueb11b12);
        let v12 = pack_1and2_f32(valuea12a13, valueb11b12);
        let v13 = pack_2and1_f32(valuea12a13, valueb13b14);
        let v14 = pack_1and2_f32(valuea14a15, valueb13b14);
        let v15 = pack_2and1_f32(valuea14a15, valueb15b16);
        let v16 = pack_1and2_f32(valuea16b0, valueb15b16);

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16,
        ]);

        let val = std::mem::transmute::<[__m128; 17], [Complex<f32>; 34]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[1];
        *output_slice.add(18) = val[3];
        *output_slice.add(19) = val[5];
        *output_slice.add(20) = val[7];
        *output_slice.add(21) = val[9];
        *output_slice.add(22) = val[11];
        *output_slice.add(23) = val[13];
        *output_slice.add(24) = val[15];
        *output_slice.add(25) = val[17];
        *output_slice.add(26) = val[19];
        *output_slice.add(27) = val[21];
        *output_slice.add(28) = val[23];
        *output_slice.add(29) = val[25];
        *output_slice.add(30) = val[27];
        *output_slice.add(31) = val[29];
        *output_slice.add(32) = val[31];
        *output_slice.add(33) = val[33];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values: [__m128; 17]) -> [__m128; 17] {
        // This is a SSE translation of the scalar 17-point butterfly
        let x116p = _mm_add_ps(values[1], values[16]);
        let x116n = _mm_sub_ps(values[1], values[16]);
        let x215p = _mm_add_ps(values[2], values[15]);
        let x215n = _mm_sub_ps(values[2], values[15]);
        let x314p = _mm_add_ps(values[3], values[14]);
        let x314n = _mm_sub_ps(values[3], values[14]);
        let x413p = _mm_add_ps(values[4], values[13]);
        let x413n = _mm_sub_ps(values[4], values[13]);
        let x512p = _mm_add_ps(values[5], values[12]);
        let x512n = _mm_sub_ps(values[5], values[12]);
        let x611p = _mm_add_ps(values[6], values[11]);
        let x611n = _mm_sub_ps(values[6], values[11]);
        let x710p = _mm_add_ps(values[7], values[10]);
        let x710n = _mm_sub_ps(values[7], values[10]);
        let x89p = _mm_add_ps(values[8], values[9]);
        let x89n = _mm_sub_ps(values[8], values[9]);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x116p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x215p);
        let temp_a1_3 = _mm_mul_ps(self.twiddle3re, x314p);
        let temp_a1_4 = _mm_mul_ps(self.twiddle4re, x413p);
        let temp_a1_5 = _mm_mul_ps(self.twiddle5re, x512p);
        let temp_a1_6 = _mm_mul_ps(self.twiddle6re, x611p);
        let temp_a1_7 = _mm_mul_ps(self.twiddle7re, x710p);
        let temp_a1_8 = _mm_mul_ps(self.twiddle8re, x89p);
        let temp_a2_1 = _mm_mul_ps(self.twiddle2re, x116p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle4re, x215p);
        let temp_a2_3 = _mm_mul_ps(self.twiddle6re, x314p);
        let temp_a2_4 = _mm_mul_ps(self.twiddle8re, x413p);
        let temp_a2_5 = _mm_mul_ps(self.twiddle7re, x512p);
        let temp_a2_6 = _mm_mul_ps(self.twiddle5re, x611p);
        let temp_a2_7 = _mm_mul_ps(self.twiddle3re, x710p);
        let temp_a2_8 = _mm_mul_ps(self.twiddle1re, x89p);
        let temp_a3_1 = _mm_mul_ps(self.twiddle3re, x116p);
        let temp_a3_2 = _mm_mul_ps(self.twiddle6re, x215p);
        let temp_a3_3 = _mm_mul_ps(self.twiddle8re, x314p);
        let temp_a3_4 = _mm_mul_ps(self.twiddle5re, x413p);
        let temp_a3_5 = _mm_mul_ps(self.twiddle2re, x512p);
        let temp_a3_6 = _mm_mul_ps(self.twiddle1re, x611p);
        let temp_a3_7 = _mm_mul_ps(self.twiddle4re, x710p);
        let temp_a3_8 = _mm_mul_ps(self.twiddle7re, x89p);
        let temp_a4_1 = _mm_mul_ps(self.twiddle4re, x116p);
        let temp_a4_2 = _mm_mul_ps(self.twiddle8re, x215p);
        let temp_a4_3 = _mm_mul_ps(self.twiddle5re, x314p);
        let temp_a4_4 = _mm_mul_ps(self.twiddle1re, x413p);
        let temp_a4_5 = _mm_mul_ps(self.twiddle3re, x512p);
        let temp_a4_6 = _mm_mul_ps(self.twiddle7re, x611p);
        let temp_a4_7 = _mm_mul_ps(self.twiddle6re, x710p);
        let temp_a4_8 = _mm_mul_ps(self.twiddle2re, x89p);
        let temp_a5_1 = _mm_mul_ps(self.twiddle5re, x116p);
        let temp_a5_2 = _mm_mul_ps(self.twiddle7re, x215p);
        let temp_a5_3 = _mm_mul_ps(self.twiddle2re, x314p);
        let temp_a5_4 = _mm_mul_ps(self.twiddle3re, x413p);
        let temp_a5_5 = _mm_mul_ps(self.twiddle8re, x512p);
        let temp_a5_6 = _mm_mul_ps(self.twiddle4re, x611p);
        let temp_a5_7 = _mm_mul_ps(self.twiddle1re, x710p);
        let temp_a5_8 = _mm_mul_ps(self.twiddle6re, x89p);
        let temp_a6_1 = _mm_mul_ps(self.twiddle6re, x116p);
        let temp_a6_2 = _mm_mul_ps(self.twiddle5re, x215p);
        let temp_a6_3 = _mm_mul_ps(self.twiddle1re, x314p);
        let temp_a6_4 = _mm_mul_ps(self.twiddle7re, x413p);
        let temp_a6_5 = _mm_mul_ps(self.twiddle4re, x512p);
        let temp_a6_6 = _mm_mul_ps(self.twiddle2re, x611p);
        let temp_a6_7 = _mm_mul_ps(self.twiddle8re, x710p);
        let temp_a6_8 = _mm_mul_ps(self.twiddle3re, x89p);
        let temp_a7_1 = _mm_mul_ps(self.twiddle7re, x116p);
        let temp_a7_2 = _mm_mul_ps(self.twiddle3re, x215p);
        let temp_a7_3 = _mm_mul_ps(self.twiddle4re, x314p);
        let temp_a7_4 = _mm_mul_ps(self.twiddle6re, x413p);
        let temp_a7_5 = _mm_mul_ps(self.twiddle1re, x512p);
        let temp_a7_6 = _mm_mul_ps(self.twiddle8re, x611p);
        let temp_a7_7 = _mm_mul_ps(self.twiddle2re, x710p);
        let temp_a7_8 = _mm_mul_ps(self.twiddle5re, x89p);
        let temp_a8_1 = _mm_mul_ps(self.twiddle8re, x116p);
        let temp_a8_2 = _mm_mul_ps(self.twiddle1re, x215p);
        let temp_a8_3 = _mm_mul_ps(self.twiddle7re, x314p);
        let temp_a8_4 = _mm_mul_ps(self.twiddle2re, x413p);
        let temp_a8_5 = _mm_mul_ps(self.twiddle6re, x512p);
        let temp_a8_6 = _mm_mul_ps(self.twiddle3re, x611p);
        let temp_a8_7 = _mm_mul_ps(self.twiddle5re, x710p);
        let temp_a8_8 = _mm_mul_ps(self.twiddle4re, x89p);

        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x116n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x215n);
        let temp_b1_3 = _mm_mul_ps(self.twiddle3im, x314n);
        let temp_b1_4 = _mm_mul_ps(self.twiddle4im, x413n);
        let temp_b1_5 = _mm_mul_ps(self.twiddle5im, x512n);
        let temp_b1_6 = _mm_mul_ps(self.twiddle6im, x611n);
        let temp_b1_7 = _mm_mul_ps(self.twiddle7im, x710n);
        let temp_b1_8 = _mm_mul_ps(self.twiddle8im, x89n);
        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x116n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle4im, x215n);
        let temp_b2_3 = _mm_mul_ps(self.twiddle6im, x314n);
        let temp_b2_4 = _mm_mul_ps(self.twiddle8im, x413n);
        let temp_b2_5 = _mm_mul_ps(self.twiddle7im, x512n);
        let temp_b2_6 = _mm_mul_ps(self.twiddle5im, x611n);
        let temp_b2_7 = _mm_mul_ps(self.twiddle3im, x710n);
        let temp_b2_8 = _mm_mul_ps(self.twiddle1im, x89n);
        let temp_b3_1 = _mm_mul_ps(self.twiddle3im, x116n);
        let temp_b3_2 = _mm_mul_ps(self.twiddle6im, x215n);
        let temp_b3_3 = _mm_mul_ps(self.twiddle8im, x314n);
        let temp_b3_4 = _mm_mul_ps(self.twiddle5im, x413n);
        let temp_b3_5 = _mm_mul_ps(self.twiddle2im, x512n);
        let temp_b3_6 = _mm_mul_ps(self.twiddle1im, x611n);
        let temp_b3_7 = _mm_mul_ps(self.twiddle4im, x710n);
        let temp_b3_8 = _mm_mul_ps(self.twiddle7im, x89n);
        let temp_b4_1 = _mm_mul_ps(self.twiddle4im, x116n);
        let temp_b4_2 = _mm_mul_ps(self.twiddle8im, x215n);
        let temp_b4_3 = _mm_mul_ps(self.twiddle5im, x314n);
        let temp_b4_4 = _mm_mul_ps(self.twiddle1im, x413n);
        let temp_b4_5 = _mm_mul_ps(self.twiddle3im, x512n);
        let temp_b4_6 = _mm_mul_ps(self.twiddle7im, x611n);
        let temp_b4_7 = _mm_mul_ps(self.twiddle6im, x710n);
        let temp_b4_8 = _mm_mul_ps(self.twiddle2im, x89n);
        let temp_b5_1 = _mm_mul_ps(self.twiddle5im, x116n);
        let temp_b5_2 = _mm_mul_ps(self.twiddle7im, x215n);
        let temp_b5_3 = _mm_mul_ps(self.twiddle2im, x314n);
        let temp_b5_4 = _mm_mul_ps(self.twiddle3im, x413n);
        let temp_b5_5 = _mm_mul_ps(self.twiddle8im, x512n);
        let temp_b5_6 = _mm_mul_ps(self.twiddle4im, x611n);
        let temp_b5_7 = _mm_mul_ps(self.twiddle1im, x710n);
        let temp_b5_8 = _mm_mul_ps(self.twiddle6im, x89n);
        let temp_b6_1 = _mm_mul_ps(self.twiddle6im, x116n);
        let temp_b6_2 = _mm_mul_ps(self.twiddle5im, x215n);
        let temp_b6_3 = _mm_mul_ps(self.twiddle1im, x314n);
        let temp_b6_4 = _mm_mul_ps(self.twiddle7im, x413n);
        let temp_b6_5 = _mm_mul_ps(self.twiddle4im, x512n);
        let temp_b6_6 = _mm_mul_ps(self.twiddle2im, x611n);
        let temp_b6_7 = _mm_mul_ps(self.twiddle8im, x710n);
        let temp_b6_8 = _mm_mul_ps(self.twiddle3im, x89n);
        let temp_b7_1 = _mm_mul_ps(self.twiddle7im, x116n);
        let temp_b7_2 = _mm_mul_ps(self.twiddle3im, x215n);
        let temp_b7_3 = _mm_mul_ps(self.twiddle4im, x314n);
        let temp_b7_4 = _mm_mul_ps(self.twiddle6im, x413n);
        let temp_b7_5 = _mm_mul_ps(self.twiddle1im, x512n);
        let temp_b7_6 = _mm_mul_ps(self.twiddle8im, x611n);
        let temp_b7_7 = _mm_mul_ps(self.twiddle2im, x710n);
        let temp_b7_8 = _mm_mul_ps(self.twiddle5im, x89n);
        let temp_b8_1 = _mm_mul_ps(self.twiddle8im, x116n);
        let temp_b8_2 = _mm_mul_ps(self.twiddle1im, x215n);
        let temp_b8_3 = _mm_mul_ps(self.twiddle7im, x314n);
        let temp_b8_4 = _mm_mul_ps(self.twiddle2im, x413n);
        let temp_b8_5 = _mm_mul_ps(self.twiddle6im, x512n);
        let temp_b8_6 = _mm_mul_ps(self.twiddle3im, x611n);
        let temp_b8_7 = _mm_mul_ps(self.twiddle5im, x710n);
        let temp_b8_8 = _mm_mul_ps(self.twiddle4im, x89n);

        let temp_a1 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a1_1,
                _mm_add_ps(
                    temp_a1_2,
                    _mm_add_ps(
                        temp_a1_3,
                        _mm_add_ps(
                            temp_a1_4,
                            _mm_add_ps(
                                temp_a1_5,
                                _mm_add_ps(temp_a1_6, _mm_add_ps(temp_a1_7, temp_a1_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a2_1,
                _mm_add_ps(
                    temp_a2_2,
                    _mm_add_ps(
                        temp_a2_3,
                        _mm_add_ps(
                            temp_a2_4,
                            _mm_add_ps(
                                temp_a2_5,
                                _mm_add_ps(temp_a2_6, _mm_add_ps(temp_a2_7, temp_a2_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a3_1,
                _mm_add_ps(
                    temp_a3_2,
                    _mm_add_ps(
                        temp_a3_3,
                        _mm_add_ps(
                            temp_a3_4,
                            _mm_add_ps(
                                temp_a3_5,
                                _mm_add_ps(temp_a3_6, _mm_add_ps(temp_a3_7, temp_a3_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a4_1,
                _mm_add_ps(
                    temp_a4_2,
                    _mm_add_ps(
                        temp_a4_3,
                        _mm_add_ps(
                            temp_a4_4,
                            _mm_add_ps(
                                temp_a4_5,
                                _mm_add_ps(temp_a4_6, _mm_add_ps(temp_a4_7, temp_a4_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a5_1,
                _mm_add_ps(
                    temp_a5_2,
                    _mm_add_ps(
                        temp_a5_3,
                        _mm_add_ps(
                            temp_a5_4,
                            _mm_add_ps(
                                temp_a5_5,
                                _mm_add_ps(temp_a5_6, _mm_add_ps(temp_a5_7, temp_a5_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a6_1,
                _mm_add_ps(
                    temp_a6_2,
                    _mm_add_ps(
                        temp_a6_3,
                        _mm_add_ps(
                            temp_a6_4,
                            _mm_add_ps(
                                temp_a6_5,
                                _mm_add_ps(temp_a6_6, _mm_add_ps(temp_a6_7, temp_a6_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a7_1,
                _mm_add_ps(
                    temp_a7_2,
                    _mm_add_ps(
                        temp_a7_3,
                        _mm_add_ps(
                            temp_a7_4,
                            _mm_add_ps(
                                temp_a7_5,
                                _mm_add_ps(temp_a7_6, _mm_add_ps(temp_a7_7, temp_a7_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a8_1,
                _mm_add_ps(
                    temp_a8_2,
                    _mm_add_ps(
                        temp_a8_3,
                        _mm_add_ps(
                            temp_a8_4,
                            _mm_add_ps(
                                temp_a8_5,
                                _mm_add_ps(temp_a8_6, _mm_add_ps(temp_a8_7, temp_a8_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_ps(
            temp_b1_1,
            _mm_add_ps(
                temp_b1_2,
                _mm_add_ps(
                    temp_b1_3,
                    _mm_add_ps(
                        temp_b1_4,
                        _mm_add_ps(
                            temp_b1_5,
                            _mm_add_ps(temp_b1_6, _mm_add_ps(temp_b1_7, temp_b1_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_ps(
            temp_b2_1,
            _mm_add_ps(
                temp_b2_2,
                _mm_add_ps(
                    temp_b2_3,
                    _mm_sub_ps(
                        temp_b2_4,
                        _mm_add_ps(
                            temp_b2_5,
                            _mm_add_ps(temp_b2_6, _mm_add_ps(temp_b2_7, temp_b2_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_ps(
            temp_b3_1,
            _mm_sub_ps(
                temp_b3_2,
                _mm_add_ps(
                    temp_b3_3,
                    _mm_add_ps(
                        temp_b3_4,
                        _mm_sub_ps(
                            temp_b3_5,
                            _mm_add_ps(temp_b3_6, _mm_add_ps(temp_b3_7, temp_b3_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_ps(
            temp_b4_1,
            _mm_sub_ps(
                temp_b4_2,
                _mm_add_ps(
                    temp_b4_3,
                    _mm_sub_ps(
                        temp_b4_4,
                        _mm_add_ps(
                            temp_b4_5,
                            _mm_sub_ps(temp_b4_6, _mm_add_ps(temp_b4_7, temp_b4_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_sub_ps(
            temp_b5_1,
            _mm_add_ps(
                temp_b5_2,
                _mm_sub_ps(
                    temp_b5_3,
                    _mm_add_ps(
                        temp_b5_4,
                        _mm_sub_ps(
                            temp_b5_5,
                            _mm_sub_ps(temp_b5_6, _mm_add_ps(temp_b5_7, temp_b5_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_sub_ps(
            temp_b6_1,
            _mm_sub_ps(
                temp_b6_2,
                _mm_add_ps(
                    temp_b6_3,
                    _mm_sub_ps(
                        temp_b6_4,
                        _mm_sub_ps(
                            temp_b6_5,
                            _mm_add_ps(temp_b6_6, _mm_sub_ps(temp_b6_7, temp_b6_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_sub_ps(
            temp_b7_1,
            _mm_sub_ps(
                temp_b7_2,
                _mm_sub_ps(
                    temp_b7_3,
                    _mm_sub_ps(
                        temp_b7_4,
                        _mm_add_ps(
                            temp_b7_5,
                            _mm_sub_ps(temp_b7_6, _mm_sub_ps(temp_b7_7, temp_b7_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_ps(
            temp_b8_1,
            _mm_sub_ps(
                temp_b8_2,
                _mm_sub_ps(
                    temp_b8_3,
                    _mm_sub_ps(
                        temp_b8_4,
                        _mm_sub_ps(
                            temp_b8_5,
                            _mm_sub_ps(temp_b8_6, _mm_sub_ps(temp_b8_7, temp_b8_8)),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate_both(temp_b1);
        let temp_b2_rot = self.rotate.rotate_both(temp_b2);
        let temp_b3_rot = self.rotate.rotate_both(temp_b3);
        let temp_b4_rot = self.rotate.rotate_both(temp_b4);
        let temp_b5_rot = self.rotate.rotate_both(temp_b5);
        let temp_b6_rot = self.rotate.rotate_both(temp_b6);
        let temp_b7_rot = self.rotate.rotate_both(temp_b7);
        let temp_b8_rot = self.rotate.rotate_both(temp_b8);

        let x0 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                x116p,
                _mm_add_ps(
                    x215p,
                    _mm_add_ps(
                        x314p,
                        _mm_add_ps(
                            x413p,
                            _mm_add_ps(x512p, _mm_add_ps(x611p, _mm_add_ps(x710p, x89p))),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_ps(temp_a1, temp_b1_rot);
        let x2 = _mm_add_ps(temp_a2, temp_b2_rot);
        let x3 = _mm_add_ps(temp_a3, temp_b3_rot);
        let x4 = _mm_add_ps(temp_a4, temp_b4_rot);
        let x5 = _mm_add_ps(temp_a5, temp_b5_rot);
        let x6 = _mm_add_ps(temp_a6, temp_b6_rot);
        let x7 = _mm_add_ps(temp_a7, temp_b7_rot);
        let x8 = _mm_add_ps(temp_a8, temp_b8_rot);
        let x9 = _mm_sub_ps(temp_a8, temp_b8_rot);
        let x10 = _mm_sub_ps(temp_a7, temp_b7_rot);
        let x11 = _mm_sub_ps(temp_a6, temp_b6_rot);
        let x12 = _mm_sub_ps(temp_a5, temp_b5_rot);
        let x13 = _mm_sub_ps(temp_a4, temp_b4_rot);
        let x14 = _mm_sub_ps(temp_a3, temp_b3_rot);
        let x15 = _mm_sub_ps(temp_a2, temp_b2_rot);
        let x16 = _mm_sub_ps(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16,
        ]
    }
}

//   _ _____            __   _  _   _     _ _
//  / |___  |          / /_ | || | | |__ (_) |_
//  | |  / /   _____  | '_ \| || |_| '_ \| | __|
//  | | / /   |_____| | (_) |__   _| |_) | | |_
//  |_|/_/             \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly17<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
    twiddle3re: __m128d,
    twiddle3im: __m128d,
    twiddle4re: __m128d,
    twiddle4im: __m128d,
    twiddle5re: __m128d,
    twiddle5im: __m128d,
    twiddle6re: __m128d,
    twiddle6im: __m128d,
    twiddle7re: __m128d,
    twiddle7im: __m128d,
    twiddle8re: __m128d,
    twiddle8im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly17, 17, |this: &SseF64Butterfly17<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly17, 17, |this: &SseF64Butterfly17<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly17<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 17, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 17, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 17, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 17, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 17, direction);
        let tw6: Complex<f64> = twiddles::compute_twiddle(6, 17, direction);
        let tw7: Complex<f64> = twiddles::compute_twiddle(7, 17, direction);
        let tw8: Complex<f64> = twiddles::compute_twiddle(8, 17, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_pd(tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_pd(tw3.im, tw3.im) };
        let twiddle4re = unsafe { _mm_set_pd(tw4.re, tw4.re) };
        let twiddle4im = unsafe { _mm_set_pd(tw4.im, tw4.im) };
        let twiddle5re = unsafe { _mm_set_pd(tw5.re, tw5.re) };
        let twiddle5im = unsafe { _mm_set_pd(tw5.im, tw5.im) };
        let twiddle6re = unsafe { _mm_set_pd(tw6.re, tw6.re) };
        let twiddle6im = unsafe { _mm_set_pd(tw6.im, tw6.im) };
        let twiddle7re = unsafe { _mm_set_pd(tw7.re, tw7.re) };
        let twiddle7im = unsafe { _mm_set_pd(tw7.im, tw7.im) };
        let twiddle8re = unsafe { _mm_set_pd(tw8.re, tw8.re) };
        let twiddle8im = unsafe { _mm_set_pd(tw8.im, tw8.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let v1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let v2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let v3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let v4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let v5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let v6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let v7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let v8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let v9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let v10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);
        let v11 = _mm_loadu_pd(input.as_ptr().add(11) as *const f64);
        let v12 = _mm_loadu_pd(input.as_ptr().add(12) as *const f64);
        let v13 = _mm_loadu_pd(input.as_ptr().add(13) as *const f64);
        let v14 = _mm_loadu_pd(input.as_ptr().add(14) as *const f64);
        let v15 = _mm_loadu_pd(input.as_ptr().add(15) as *const f64);
        let v16 = _mm_loadu_pd(input.as_ptr().add(16) as *const f64);

        let out = self.perform_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16,
        ]);

        let val = std::mem::transmute::<[__m128d; 17], [Complex<f64>; 17]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
        *output_slice.add(6) = val[6];
        *output_slice.add(7) = val[7];
        *output_slice.add(8) = val[8];
        *output_slice.add(9) = val[9];
        *output_slice.add(10) = val[10];
        *output_slice.add(11) = val[11];
        *output_slice.add(12) = val[12];
        *output_slice.add(13) = val[13];
        *output_slice.add(14) = val[14];
        *output_slice.add(15) = val[15];
        *output_slice.add(16) = val[16];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 17]) -> [__m128d; 17] {
        // This is a SSE translation of the scalar 17-point butterfly
        let x116p = _mm_add_pd(values[1], values[16]);
        let x116n = _mm_sub_pd(values[1], values[16]);
        let x215p = _mm_add_pd(values[2], values[15]);
        let x215n = _mm_sub_pd(values[2], values[15]);
        let x314p = _mm_add_pd(values[3], values[14]);
        let x314n = _mm_sub_pd(values[3], values[14]);
        let x413p = _mm_add_pd(values[4], values[13]);
        let x413n = _mm_sub_pd(values[4], values[13]);
        let x512p = _mm_add_pd(values[5], values[12]);
        let x512n = _mm_sub_pd(values[5], values[12]);
        let x611p = _mm_add_pd(values[6], values[11]);
        let x611n = _mm_sub_pd(values[6], values[11]);
        let x710p = _mm_add_pd(values[7], values[10]);
        let x710n = _mm_sub_pd(values[7], values[10]);
        let x89p = _mm_add_pd(values[8], values[9]);
        let x89n = _mm_sub_pd(values[8], values[9]);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x116p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x215p);
        let temp_a1_3 = _mm_mul_pd(self.twiddle3re, x314p);
        let temp_a1_4 = _mm_mul_pd(self.twiddle4re, x413p);
        let temp_a1_5 = _mm_mul_pd(self.twiddle5re, x512p);
        let temp_a1_6 = _mm_mul_pd(self.twiddle6re, x611p);
        let temp_a1_7 = _mm_mul_pd(self.twiddle7re, x710p);
        let temp_a1_8 = _mm_mul_pd(self.twiddle8re, x89p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x116p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle4re, x215p);
        let temp_a2_3 = _mm_mul_pd(self.twiddle6re, x314p);
        let temp_a2_4 = _mm_mul_pd(self.twiddle8re, x413p);
        let temp_a2_5 = _mm_mul_pd(self.twiddle7re, x512p);
        let temp_a2_6 = _mm_mul_pd(self.twiddle5re, x611p);
        let temp_a2_7 = _mm_mul_pd(self.twiddle3re, x710p);
        let temp_a2_8 = _mm_mul_pd(self.twiddle1re, x89p);
        let temp_a3_1 = _mm_mul_pd(self.twiddle3re, x116p);
        let temp_a3_2 = _mm_mul_pd(self.twiddle6re, x215p);
        let temp_a3_3 = _mm_mul_pd(self.twiddle8re, x314p);
        let temp_a3_4 = _mm_mul_pd(self.twiddle5re, x413p);
        let temp_a3_5 = _mm_mul_pd(self.twiddle2re, x512p);
        let temp_a3_6 = _mm_mul_pd(self.twiddle1re, x611p);
        let temp_a3_7 = _mm_mul_pd(self.twiddle4re, x710p);
        let temp_a3_8 = _mm_mul_pd(self.twiddle7re, x89p);
        let temp_a4_1 = _mm_mul_pd(self.twiddle4re, x116p);
        let temp_a4_2 = _mm_mul_pd(self.twiddle8re, x215p);
        let temp_a4_3 = _mm_mul_pd(self.twiddle5re, x314p);
        let temp_a4_4 = _mm_mul_pd(self.twiddle1re, x413p);
        let temp_a4_5 = _mm_mul_pd(self.twiddle3re, x512p);
        let temp_a4_6 = _mm_mul_pd(self.twiddle7re, x611p);
        let temp_a4_7 = _mm_mul_pd(self.twiddle6re, x710p);
        let temp_a4_8 = _mm_mul_pd(self.twiddle2re, x89p);
        let temp_a5_1 = _mm_mul_pd(self.twiddle5re, x116p);
        let temp_a5_2 = _mm_mul_pd(self.twiddle7re, x215p);
        let temp_a5_3 = _mm_mul_pd(self.twiddle2re, x314p);
        let temp_a5_4 = _mm_mul_pd(self.twiddle3re, x413p);
        let temp_a5_5 = _mm_mul_pd(self.twiddle8re, x512p);
        let temp_a5_6 = _mm_mul_pd(self.twiddle4re, x611p);
        let temp_a5_7 = _mm_mul_pd(self.twiddle1re, x710p);
        let temp_a5_8 = _mm_mul_pd(self.twiddle6re, x89p);
        let temp_a6_1 = _mm_mul_pd(self.twiddle6re, x116p);
        let temp_a6_2 = _mm_mul_pd(self.twiddle5re, x215p);
        let temp_a6_3 = _mm_mul_pd(self.twiddle1re, x314p);
        let temp_a6_4 = _mm_mul_pd(self.twiddle7re, x413p);
        let temp_a6_5 = _mm_mul_pd(self.twiddle4re, x512p);
        let temp_a6_6 = _mm_mul_pd(self.twiddle2re, x611p);
        let temp_a6_7 = _mm_mul_pd(self.twiddle8re, x710p);
        let temp_a6_8 = _mm_mul_pd(self.twiddle3re, x89p);
        let temp_a7_1 = _mm_mul_pd(self.twiddle7re, x116p);
        let temp_a7_2 = _mm_mul_pd(self.twiddle3re, x215p);
        let temp_a7_3 = _mm_mul_pd(self.twiddle4re, x314p);
        let temp_a7_4 = _mm_mul_pd(self.twiddle6re, x413p);
        let temp_a7_5 = _mm_mul_pd(self.twiddle1re, x512p);
        let temp_a7_6 = _mm_mul_pd(self.twiddle8re, x611p);
        let temp_a7_7 = _mm_mul_pd(self.twiddle2re, x710p);
        let temp_a7_8 = _mm_mul_pd(self.twiddle5re, x89p);
        let temp_a8_1 = _mm_mul_pd(self.twiddle8re, x116p);
        let temp_a8_2 = _mm_mul_pd(self.twiddle1re, x215p);
        let temp_a8_3 = _mm_mul_pd(self.twiddle7re, x314p);
        let temp_a8_4 = _mm_mul_pd(self.twiddle2re, x413p);
        let temp_a8_5 = _mm_mul_pd(self.twiddle6re, x512p);
        let temp_a8_6 = _mm_mul_pd(self.twiddle3re, x611p);
        let temp_a8_7 = _mm_mul_pd(self.twiddle5re, x710p);
        let temp_a8_8 = _mm_mul_pd(self.twiddle4re, x89p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x116n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x215n);
        let temp_b1_3 = _mm_mul_pd(self.twiddle3im, x314n);
        let temp_b1_4 = _mm_mul_pd(self.twiddle4im, x413n);
        let temp_b1_5 = _mm_mul_pd(self.twiddle5im, x512n);
        let temp_b1_6 = _mm_mul_pd(self.twiddle6im, x611n);
        let temp_b1_7 = _mm_mul_pd(self.twiddle7im, x710n);
        let temp_b1_8 = _mm_mul_pd(self.twiddle8im, x89n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x116n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle4im, x215n);
        let temp_b2_3 = _mm_mul_pd(self.twiddle6im, x314n);
        let temp_b2_4 = _mm_mul_pd(self.twiddle8im, x413n);
        let temp_b2_5 = _mm_mul_pd(self.twiddle7im, x512n);
        let temp_b2_6 = _mm_mul_pd(self.twiddle5im, x611n);
        let temp_b2_7 = _mm_mul_pd(self.twiddle3im, x710n);
        let temp_b2_8 = _mm_mul_pd(self.twiddle1im, x89n);
        let temp_b3_1 = _mm_mul_pd(self.twiddle3im, x116n);
        let temp_b3_2 = _mm_mul_pd(self.twiddle6im, x215n);
        let temp_b3_3 = _mm_mul_pd(self.twiddle8im, x314n);
        let temp_b3_4 = _mm_mul_pd(self.twiddle5im, x413n);
        let temp_b3_5 = _mm_mul_pd(self.twiddle2im, x512n);
        let temp_b3_6 = _mm_mul_pd(self.twiddle1im, x611n);
        let temp_b3_7 = _mm_mul_pd(self.twiddle4im, x710n);
        let temp_b3_8 = _mm_mul_pd(self.twiddle7im, x89n);
        let temp_b4_1 = _mm_mul_pd(self.twiddle4im, x116n);
        let temp_b4_2 = _mm_mul_pd(self.twiddle8im, x215n);
        let temp_b4_3 = _mm_mul_pd(self.twiddle5im, x314n);
        let temp_b4_4 = _mm_mul_pd(self.twiddle1im, x413n);
        let temp_b4_5 = _mm_mul_pd(self.twiddle3im, x512n);
        let temp_b4_6 = _mm_mul_pd(self.twiddle7im, x611n);
        let temp_b4_7 = _mm_mul_pd(self.twiddle6im, x710n);
        let temp_b4_8 = _mm_mul_pd(self.twiddle2im, x89n);
        let temp_b5_1 = _mm_mul_pd(self.twiddle5im, x116n);
        let temp_b5_2 = _mm_mul_pd(self.twiddle7im, x215n);
        let temp_b5_3 = _mm_mul_pd(self.twiddle2im, x314n);
        let temp_b5_4 = _mm_mul_pd(self.twiddle3im, x413n);
        let temp_b5_5 = _mm_mul_pd(self.twiddle8im, x512n);
        let temp_b5_6 = _mm_mul_pd(self.twiddle4im, x611n);
        let temp_b5_7 = _mm_mul_pd(self.twiddle1im, x710n);
        let temp_b5_8 = _mm_mul_pd(self.twiddle6im, x89n);
        let temp_b6_1 = _mm_mul_pd(self.twiddle6im, x116n);
        let temp_b6_2 = _mm_mul_pd(self.twiddle5im, x215n);
        let temp_b6_3 = _mm_mul_pd(self.twiddle1im, x314n);
        let temp_b6_4 = _mm_mul_pd(self.twiddle7im, x413n);
        let temp_b6_5 = _mm_mul_pd(self.twiddle4im, x512n);
        let temp_b6_6 = _mm_mul_pd(self.twiddle2im, x611n);
        let temp_b6_7 = _mm_mul_pd(self.twiddle8im, x710n);
        let temp_b6_8 = _mm_mul_pd(self.twiddle3im, x89n);
        let temp_b7_1 = _mm_mul_pd(self.twiddle7im, x116n);
        let temp_b7_2 = _mm_mul_pd(self.twiddle3im, x215n);
        let temp_b7_3 = _mm_mul_pd(self.twiddle4im, x314n);
        let temp_b7_4 = _mm_mul_pd(self.twiddle6im, x413n);
        let temp_b7_5 = _mm_mul_pd(self.twiddle1im, x512n);
        let temp_b7_6 = _mm_mul_pd(self.twiddle8im, x611n);
        let temp_b7_7 = _mm_mul_pd(self.twiddle2im, x710n);
        let temp_b7_8 = _mm_mul_pd(self.twiddle5im, x89n);
        let temp_b8_1 = _mm_mul_pd(self.twiddle8im, x116n);
        let temp_b8_2 = _mm_mul_pd(self.twiddle1im, x215n);
        let temp_b8_3 = _mm_mul_pd(self.twiddle7im, x314n);
        let temp_b8_4 = _mm_mul_pd(self.twiddle2im, x413n);
        let temp_b8_5 = _mm_mul_pd(self.twiddle6im, x512n);
        let temp_b8_6 = _mm_mul_pd(self.twiddle3im, x611n);
        let temp_b8_7 = _mm_mul_pd(self.twiddle5im, x710n);
        let temp_b8_8 = _mm_mul_pd(self.twiddle4im, x89n);

        let temp_a1 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a1_1,
                _mm_add_pd(
                    temp_a1_2,
                    _mm_add_pd(
                        temp_a1_3,
                        _mm_add_pd(
                            temp_a1_4,
                            _mm_add_pd(
                                temp_a1_5,
                                _mm_add_pd(temp_a1_6, _mm_add_pd(temp_a1_7, temp_a1_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a2_1,
                _mm_add_pd(
                    temp_a2_2,
                    _mm_add_pd(
                        temp_a2_3,
                        _mm_add_pd(
                            temp_a2_4,
                            _mm_add_pd(
                                temp_a2_5,
                                _mm_add_pd(temp_a2_6, _mm_add_pd(temp_a2_7, temp_a2_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a3_1,
                _mm_add_pd(
                    temp_a3_2,
                    _mm_add_pd(
                        temp_a3_3,
                        _mm_add_pd(
                            temp_a3_4,
                            _mm_add_pd(
                                temp_a3_5,
                                _mm_add_pd(temp_a3_6, _mm_add_pd(temp_a3_7, temp_a3_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a4_1,
                _mm_add_pd(
                    temp_a4_2,
                    _mm_add_pd(
                        temp_a4_3,
                        _mm_add_pd(
                            temp_a4_4,
                            _mm_add_pd(
                                temp_a4_5,
                                _mm_add_pd(temp_a4_6, _mm_add_pd(temp_a4_7, temp_a4_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a5_1,
                _mm_add_pd(
                    temp_a5_2,
                    _mm_add_pd(
                        temp_a5_3,
                        _mm_add_pd(
                            temp_a5_4,
                            _mm_add_pd(
                                temp_a5_5,
                                _mm_add_pd(temp_a5_6, _mm_add_pd(temp_a5_7, temp_a5_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a6_1,
                _mm_add_pd(
                    temp_a6_2,
                    _mm_add_pd(
                        temp_a6_3,
                        _mm_add_pd(
                            temp_a6_4,
                            _mm_add_pd(
                                temp_a6_5,
                                _mm_add_pd(temp_a6_6, _mm_add_pd(temp_a6_7, temp_a6_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a7_1,
                _mm_add_pd(
                    temp_a7_2,
                    _mm_add_pd(
                        temp_a7_3,
                        _mm_add_pd(
                            temp_a7_4,
                            _mm_add_pd(
                                temp_a7_5,
                                _mm_add_pd(temp_a7_6, _mm_add_pd(temp_a7_7, temp_a7_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a8_1,
                _mm_add_pd(
                    temp_a8_2,
                    _mm_add_pd(
                        temp_a8_3,
                        _mm_add_pd(
                            temp_a8_4,
                            _mm_add_pd(
                                temp_a8_5,
                                _mm_add_pd(temp_a8_6, _mm_add_pd(temp_a8_7, temp_a8_8)),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_pd(
            temp_b1_1,
            _mm_add_pd(
                temp_b1_2,
                _mm_add_pd(
                    temp_b1_3,
                    _mm_add_pd(
                        temp_b1_4,
                        _mm_add_pd(
                            temp_b1_5,
                            _mm_add_pd(temp_b1_6, _mm_add_pd(temp_b1_7, temp_b1_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_pd(
            temp_b2_1,
            _mm_add_pd(
                temp_b2_2,
                _mm_add_pd(
                    temp_b2_3,
                    _mm_sub_pd(
                        temp_b2_4,
                        _mm_add_pd(
                            temp_b2_5,
                            _mm_add_pd(temp_b2_6, _mm_add_pd(temp_b2_7, temp_b2_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_pd(
            temp_b3_1,
            _mm_sub_pd(
                temp_b3_2,
                _mm_add_pd(
                    temp_b3_3,
                    _mm_add_pd(
                        temp_b3_4,
                        _mm_sub_pd(
                            temp_b3_5,
                            _mm_add_pd(temp_b3_6, _mm_add_pd(temp_b3_7, temp_b3_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_pd(
            temp_b4_1,
            _mm_sub_pd(
                temp_b4_2,
                _mm_add_pd(
                    temp_b4_3,
                    _mm_sub_pd(
                        temp_b4_4,
                        _mm_add_pd(
                            temp_b4_5,
                            _mm_sub_pd(temp_b4_6, _mm_add_pd(temp_b4_7, temp_b4_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_sub_pd(
            temp_b5_1,
            _mm_add_pd(
                temp_b5_2,
                _mm_sub_pd(
                    temp_b5_3,
                    _mm_add_pd(
                        temp_b5_4,
                        _mm_sub_pd(
                            temp_b5_5,
                            _mm_sub_pd(temp_b5_6, _mm_add_pd(temp_b5_7, temp_b5_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_sub_pd(
            temp_b6_1,
            _mm_sub_pd(
                temp_b6_2,
                _mm_add_pd(
                    temp_b6_3,
                    _mm_sub_pd(
                        temp_b6_4,
                        _mm_sub_pd(
                            temp_b6_5,
                            _mm_add_pd(temp_b6_6, _mm_sub_pd(temp_b6_7, temp_b6_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_sub_pd(
            temp_b7_1,
            _mm_sub_pd(
                temp_b7_2,
                _mm_sub_pd(
                    temp_b7_3,
                    _mm_sub_pd(
                        temp_b7_4,
                        _mm_add_pd(
                            temp_b7_5,
                            _mm_sub_pd(temp_b7_6, _mm_sub_pd(temp_b7_7, temp_b7_8)),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_pd(
            temp_b8_1,
            _mm_sub_pd(
                temp_b8_2,
                _mm_sub_pd(
                    temp_b8_3,
                    _mm_sub_pd(
                        temp_b8_4,
                        _mm_sub_pd(
                            temp_b8_5,
                            _mm_sub_pd(temp_b8_6, _mm_sub_pd(temp_b8_7, temp_b8_8)),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);
        let temp_b6_rot = self.rotate.rotate(temp_b6);
        let temp_b7_rot = self.rotate.rotate(temp_b7);
        let temp_b8_rot = self.rotate.rotate(temp_b8);

        let x0 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                x116p,
                _mm_add_pd(
                    x215p,
                    _mm_add_pd(
                        x314p,
                        _mm_add_pd(
                            x413p,
                            _mm_add_pd(x512p, _mm_add_pd(x611p, _mm_add_pd(x710p, x89p))),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
        let x4 = _mm_add_pd(temp_a4, temp_b4_rot);
        let x5 = _mm_add_pd(temp_a5, temp_b5_rot);
        let x6 = _mm_add_pd(temp_a6, temp_b6_rot);
        let x7 = _mm_add_pd(temp_a7, temp_b7_rot);
        let x8 = _mm_add_pd(temp_a8, temp_b8_rot);
        let x9 = _mm_sub_pd(temp_a8, temp_b8_rot);
        let x10 = _mm_sub_pd(temp_a7, temp_b7_rot);
        let x11 = _mm_sub_pd(temp_a6, temp_b6_rot);
        let x12 = _mm_sub_pd(temp_a5, temp_b5_rot);
        let x13 = _mm_sub_pd(temp_a4, temp_b4_rot);
        let x14 = _mm_sub_pd(temp_a3, temp_b3_rot);
        let x15 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x16 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16,
        ]
    }
}

//   _  ___            _________  _     _ _
//  / |/ _ \          |___ /___ \| |__ (_) |_
//  | | (_) |  _____    |_ \ __) | '_ \| | __|
//  | |\__, | |_____|  ___) / __/| |_) | | |_
//  |_|  /_/          |____/_____|_.__/|_|\__|
//
pub struct SseF32Butterfly19<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
    twiddle3re: __m128,
    twiddle3im: __m128,
    twiddle4re: __m128,
    twiddle4im: __m128,
    twiddle5re: __m128,
    twiddle5im: __m128,
    twiddle6re: __m128,
    twiddle6im: __m128,
    twiddle7re: __m128,
    twiddle7im: __m128,
    twiddle8re: __m128,
    twiddle8im: __m128,
    twiddle9re: __m128,
    twiddle9im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly19, 19, |this: &SseF32Butterfly19<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly19, 19, |this: &SseF32Butterfly19<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly19<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 19, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 19, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 19, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 19, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 19, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 19, direction);
        let tw7: Complex<f32> = twiddles::compute_twiddle(7, 19, direction);
        let tw8: Complex<f32> = twiddles::compute_twiddle(8, 19, direction);
        let tw9: Complex<f32> = twiddles::compute_twiddle(9, 19, direction);
        let twiddle1re = unsafe { _mm_load1_ps(&tw1.re) };
        let twiddle1im = unsafe { _mm_load1_ps(&tw1.im) };
        let twiddle2re = unsafe { _mm_load1_ps(&tw2.re) };
        let twiddle2im = unsafe { _mm_load1_ps(&tw2.im) };
        let twiddle3re = unsafe { _mm_load1_ps(&tw3.re) };
        let twiddle3im = unsafe { _mm_load1_ps(&tw3.im) };
        let twiddle4re = unsafe { _mm_load1_ps(&tw4.re) };
        let twiddle4im = unsafe { _mm_load1_ps(&tw4.im) };
        let twiddle5re = unsafe { _mm_load1_ps(&tw5.re) };
        let twiddle5im = unsafe { _mm_load1_ps(&tw5.im) };
        let twiddle6re = unsafe { _mm_load1_ps(&tw6.re) };
        let twiddle6im = unsafe { _mm_load1_ps(&tw6.im) };
        let twiddle7re = unsafe { _mm_load1_ps(&tw7.re) };
        let twiddle7im = unsafe { _mm_load1_ps(&tw7.im) };
        let twiddle8re = unsafe { _mm_load1_ps(&tw8.re) };
        let twiddle8im = unsafe { _mm_load1_ps(&tw8.im) };
        let twiddle9re = unsafe { _mm_load1_ps(&tw9.re) };
        let twiddle9im = unsafe { _mm_load1_ps(&tw9.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
            twiddle9re,
            twiddle9im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let v1 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(1) as *const f64));
        let v2 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(2) as *const f64));
        let v3 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(3) as *const f64));
        let v4 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(4) as *const f64));
        let v5 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(5) as *const f64));
        let v6 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(6) as *const f64));
        let v7 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(7) as *const f64));
        let v8 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(8) as *const f64));
        let v9 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(9) as *const f64));
        let v10 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(10) as *const f64));
        let v11 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(11) as *const f64));
        let v12 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(12) as *const f64));
        let v13 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(13) as *const f64));
        let v14 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(14) as *const f64));
        let v15 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(15) as *const f64));
        let v16 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(16) as *const f64));
        let v17 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(17) as *const f64));
        let v18 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(18) as *const f64));

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        ]);

        let val = std::mem::transmute::<[__m128; 19], [Complex<f32>; 38]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[34];
        *output_slice.add(18) = val[36];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4a5 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valuea6a7 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valuea8a9 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let valuea10a11 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let valuea12a13 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let valuea14a15 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let valuea16a17 = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let valuea18b0 = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let valueb5b6 = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);
        let valueb7b8 = _mm_loadu_ps(input.as_ptr().add(26) as *const f32);
        let valueb9b10 = _mm_loadu_ps(input.as_ptr().add(28) as *const f32);
        let valueb11b12 = _mm_loadu_ps(input.as_ptr().add(30) as *const f32);
        let valueb13b14 = _mm_loadu_ps(input.as_ptr().add(32) as *const f32);
        let valueb15b16 = _mm_loadu_ps(input.as_ptr().add(34) as *const f32);
        let valueb17b18 = _mm_loadu_ps(input.as_ptr().add(36) as *const f32);

        let v0 = pack_1and2_f32(valuea0a1, valuea18b0);
        let v1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let v2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let v3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let v4 = pack_1and2_f32(valuea4a5, valueb3b4);
        let v5 = pack_2and1_f32(valuea4a5, valueb5b6);
        let v6 = pack_1and2_f32(valuea6a7, valueb5b6);
        let v7 = pack_2and1_f32(valuea6a7, valueb7b8);
        let v8 = pack_1and2_f32(valuea8a9, valueb7b8);
        let v9 = pack_2and1_f32(valuea8a9, valueb9b10);
        let v10 = pack_1and2_f32(valuea10a11, valueb9b10);
        let v11 = pack_2and1_f32(valuea10a11, valueb11b12);
        let v12 = pack_1and2_f32(valuea12a13, valueb11b12);
        let v13 = pack_2and1_f32(valuea12a13, valueb13b14);
        let v14 = pack_1and2_f32(valuea14a15, valueb13b14);
        let v15 = pack_2and1_f32(valuea14a15, valueb15b16);
        let v16 = pack_1and2_f32(valuea16a17, valueb15b16);
        let v17 = pack_2and1_f32(valuea16a17, valueb17b18);
        let v18 = pack_1and2_f32(valuea18b0, valueb17b18);

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        ]);

        let val = std::mem::transmute::<[__m128; 19], [Complex<f32>; 38]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[34];
        *output_slice.add(18) = val[36];
        *output_slice.add(19) = val[1];
        *output_slice.add(20) = val[3];
        *output_slice.add(21) = val[5];
        *output_slice.add(22) = val[7];
        *output_slice.add(23) = val[9];
        *output_slice.add(24) = val[11];
        *output_slice.add(25) = val[13];
        *output_slice.add(26) = val[15];
        *output_slice.add(27) = val[17];
        *output_slice.add(28) = val[19];
        *output_slice.add(29) = val[21];
        *output_slice.add(30) = val[23];
        *output_slice.add(31) = val[25];
        *output_slice.add(32) = val[27];
        *output_slice.add(33) = val[29];
        *output_slice.add(34) = val[31];
        *output_slice.add(35) = val[33];
        *output_slice.add(36) = val[35];
        *output_slice.add(37) = val[37];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values: [__m128; 19]) -> [__m128; 19] {
        // This is a SSE translation of the scalar 19-point butterfly
        let x118p = _mm_add_ps(values[1], values[18]);
        let x118n = _mm_sub_ps(values[1], values[18]);
        let x217p = _mm_add_ps(values[2], values[17]);
        let x217n = _mm_sub_ps(values[2], values[17]);
        let x316p = _mm_add_ps(values[3], values[16]);
        let x316n = _mm_sub_ps(values[3], values[16]);
        let x415p = _mm_add_ps(values[4], values[15]);
        let x415n = _mm_sub_ps(values[4], values[15]);
        let x514p = _mm_add_ps(values[5], values[14]);
        let x514n = _mm_sub_ps(values[5], values[14]);
        let x613p = _mm_add_ps(values[6], values[13]);
        let x613n = _mm_sub_ps(values[6], values[13]);
        let x712p = _mm_add_ps(values[7], values[12]);
        let x712n = _mm_sub_ps(values[7], values[12]);
        let x811p = _mm_add_ps(values[8], values[11]);
        let x811n = _mm_sub_ps(values[8], values[11]);
        let x910p = _mm_add_ps(values[9], values[10]);
        let x910n = _mm_sub_ps(values[9], values[10]);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x118p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x217p);
        let temp_a1_3 = _mm_mul_ps(self.twiddle3re, x316p);
        let temp_a1_4 = _mm_mul_ps(self.twiddle4re, x415p);
        let temp_a1_5 = _mm_mul_ps(self.twiddle5re, x514p);
        let temp_a1_6 = _mm_mul_ps(self.twiddle6re, x613p);
        let temp_a1_7 = _mm_mul_ps(self.twiddle7re, x712p);
        let temp_a1_8 = _mm_mul_ps(self.twiddle8re, x811p);
        let temp_a1_9 = _mm_mul_ps(self.twiddle9re, x910p);
        let temp_a2_1 = _mm_mul_ps(self.twiddle2re, x118p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle4re, x217p);
        let temp_a2_3 = _mm_mul_ps(self.twiddle6re, x316p);
        let temp_a2_4 = _mm_mul_ps(self.twiddle8re, x415p);
        let temp_a2_5 = _mm_mul_ps(self.twiddle9re, x514p);
        let temp_a2_6 = _mm_mul_ps(self.twiddle7re, x613p);
        let temp_a2_7 = _mm_mul_ps(self.twiddle5re, x712p);
        let temp_a2_8 = _mm_mul_ps(self.twiddle3re, x811p);
        let temp_a2_9 = _mm_mul_ps(self.twiddle1re, x910p);
        let temp_a3_1 = _mm_mul_ps(self.twiddle3re, x118p);
        let temp_a3_2 = _mm_mul_ps(self.twiddle6re, x217p);
        let temp_a3_3 = _mm_mul_ps(self.twiddle9re, x316p);
        let temp_a3_4 = _mm_mul_ps(self.twiddle7re, x415p);
        let temp_a3_5 = _mm_mul_ps(self.twiddle4re, x514p);
        let temp_a3_6 = _mm_mul_ps(self.twiddle1re, x613p);
        let temp_a3_7 = _mm_mul_ps(self.twiddle2re, x712p);
        let temp_a3_8 = _mm_mul_ps(self.twiddle5re, x811p);
        let temp_a3_9 = _mm_mul_ps(self.twiddle8re, x910p);
        let temp_a4_1 = _mm_mul_ps(self.twiddle4re, x118p);
        let temp_a4_2 = _mm_mul_ps(self.twiddle8re, x217p);
        let temp_a4_3 = _mm_mul_ps(self.twiddle7re, x316p);
        let temp_a4_4 = _mm_mul_ps(self.twiddle3re, x415p);
        let temp_a4_5 = _mm_mul_ps(self.twiddle1re, x514p);
        let temp_a4_6 = _mm_mul_ps(self.twiddle5re, x613p);
        let temp_a4_7 = _mm_mul_ps(self.twiddle9re, x712p);
        let temp_a4_8 = _mm_mul_ps(self.twiddle6re, x811p);
        let temp_a4_9 = _mm_mul_ps(self.twiddle2re, x910p);
        let temp_a5_1 = _mm_mul_ps(self.twiddle5re, x118p);
        let temp_a5_2 = _mm_mul_ps(self.twiddle9re, x217p);
        let temp_a5_3 = _mm_mul_ps(self.twiddle4re, x316p);
        let temp_a5_4 = _mm_mul_ps(self.twiddle1re, x415p);
        let temp_a5_5 = _mm_mul_ps(self.twiddle6re, x514p);
        let temp_a5_6 = _mm_mul_ps(self.twiddle8re, x613p);
        let temp_a5_7 = _mm_mul_ps(self.twiddle3re, x712p);
        let temp_a5_8 = _mm_mul_ps(self.twiddle2re, x811p);
        let temp_a5_9 = _mm_mul_ps(self.twiddle7re, x910p);
        let temp_a6_1 = _mm_mul_ps(self.twiddle6re, x118p);
        let temp_a6_2 = _mm_mul_ps(self.twiddle7re, x217p);
        let temp_a6_3 = _mm_mul_ps(self.twiddle1re, x316p);
        let temp_a6_4 = _mm_mul_ps(self.twiddle5re, x415p);
        let temp_a6_5 = _mm_mul_ps(self.twiddle8re, x514p);
        let temp_a6_6 = _mm_mul_ps(self.twiddle2re, x613p);
        let temp_a6_7 = _mm_mul_ps(self.twiddle4re, x712p);
        let temp_a6_8 = _mm_mul_ps(self.twiddle9re, x811p);
        let temp_a6_9 = _mm_mul_ps(self.twiddle3re, x910p);
        let temp_a7_1 = _mm_mul_ps(self.twiddle7re, x118p);
        let temp_a7_2 = _mm_mul_ps(self.twiddle5re, x217p);
        let temp_a7_3 = _mm_mul_ps(self.twiddle2re, x316p);
        let temp_a7_4 = _mm_mul_ps(self.twiddle9re, x415p);
        let temp_a7_5 = _mm_mul_ps(self.twiddle3re, x514p);
        let temp_a7_6 = _mm_mul_ps(self.twiddle4re, x613p);
        let temp_a7_7 = _mm_mul_ps(self.twiddle8re, x712p);
        let temp_a7_8 = _mm_mul_ps(self.twiddle1re, x811p);
        let temp_a7_9 = _mm_mul_ps(self.twiddle6re, x910p);
        let temp_a8_1 = _mm_mul_ps(self.twiddle8re, x118p);
        let temp_a8_2 = _mm_mul_ps(self.twiddle3re, x217p);
        let temp_a8_3 = _mm_mul_ps(self.twiddle5re, x316p);
        let temp_a8_4 = _mm_mul_ps(self.twiddle6re, x415p);
        let temp_a8_5 = _mm_mul_ps(self.twiddle2re, x514p);
        let temp_a8_6 = _mm_mul_ps(self.twiddle9re, x613p);
        let temp_a8_7 = _mm_mul_ps(self.twiddle1re, x712p);
        let temp_a8_8 = _mm_mul_ps(self.twiddle7re, x811p);
        let temp_a8_9 = _mm_mul_ps(self.twiddle4re, x910p);
        let temp_a9_1 = _mm_mul_ps(self.twiddle9re, x118p);
        let temp_a9_2 = _mm_mul_ps(self.twiddle1re, x217p);
        let temp_a9_3 = _mm_mul_ps(self.twiddle8re, x316p);
        let temp_a9_4 = _mm_mul_ps(self.twiddle2re, x415p);
        let temp_a9_5 = _mm_mul_ps(self.twiddle7re, x514p);
        let temp_a9_6 = _mm_mul_ps(self.twiddle3re, x613p);
        let temp_a9_7 = _mm_mul_ps(self.twiddle6re, x712p);
        let temp_a9_8 = _mm_mul_ps(self.twiddle4re, x811p);
        let temp_a9_9 = _mm_mul_ps(self.twiddle5re, x910p);

        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x118n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x217n);
        let temp_b1_3 = _mm_mul_ps(self.twiddle3im, x316n);
        let temp_b1_4 = _mm_mul_ps(self.twiddle4im, x415n);
        let temp_b1_5 = _mm_mul_ps(self.twiddle5im, x514n);
        let temp_b1_6 = _mm_mul_ps(self.twiddle6im, x613n);
        let temp_b1_7 = _mm_mul_ps(self.twiddle7im, x712n);
        let temp_b1_8 = _mm_mul_ps(self.twiddle8im, x811n);
        let temp_b1_9 = _mm_mul_ps(self.twiddle9im, x910n);
        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x118n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle4im, x217n);
        let temp_b2_3 = _mm_mul_ps(self.twiddle6im, x316n);
        let temp_b2_4 = _mm_mul_ps(self.twiddle8im, x415n);
        let temp_b2_5 = _mm_mul_ps(self.twiddle9im, x514n);
        let temp_b2_6 = _mm_mul_ps(self.twiddle7im, x613n);
        let temp_b2_7 = _mm_mul_ps(self.twiddle5im, x712n);
        let temp_b2_8 = _mm_mul_ps(self.twiddle3im, x811n);
        let temp_b2_9 = _mm_mul_ps(self.twiddle1im, x910n);
        let temp_b3_1 = _mm_mul_ps(self.twiddle3im, x118n);
        let temp_b3_2 = _mm_mul_ps(self.twiddle6im, x217n);
        let temp_b3_3 = _mm_mul_ps(self.twiddle9im, x316n);
        let temp_b3_4 = _mm_mul_ps(self.twiddle7im, x415n);
        let temp_b3_5 = _mm_mul_ps(self.twiddle4im, x514n);
        let temp_b3_6 = _mm_mul_ps(self.twiddle1im, x613n);
        let temp_b3_7 = _mm_mul_ps(self.twiddle2im, x712n);
        let temp_b3_8 = _mm_mul_ps(self.twiddle5im, x811n);
        let temp_b3_9 = _mm_mul_ps(self.twiddle8im, x910n);
        let temp_b4_1 = _mm_mul_ps(self.twiddle4im, x118n);
        let temp_b4_2 = _mm_mul_ps(self.twiddle8im, x217n);
        let temp_b4_3 = _mm_mul_ps(self.twiddle7im, x316n);
        let temp_b4_4 = _mm_mul_ps(self.twiddle3im, x415n);
        let temp_b4_5 = _mm_mul_ps(self.twiddle1im, x514n);
        let temp_b4_6 = _mm_mul_ps(self.twiddle5im, x613n);
        let temp_b4_7 = _mm_mul_ps(self.twiddle9im, x712n);
        let temp_b4_8 = _mm_mul_ps(self.twiddle6im, x811n);
        let temp_b4_9 = _mm_mul_ps(self.twiddle2im, x910n);
        let temp_b5_1 = _mm_mul_ps(self.twiddle5im, x118n);
        let temp_b5_2 = _mm_mul_ps(self.twiddle9im, x217n);
        let temp_b5_3 = _mm_mul_ps(self.twiddle4im, x316n);
        let temp_b5_4 = _mm_mul_ps(self.twiddle1im, x415n);
        let temp_b5_5 = _mm_mul_ps(self.twiddle6im, x514n);
        let temp_b5_6 = _mm_mul_ps(self.twiddle8im, x613n);
        let temp_b5_7 = _mm_mul_ps(self.twiddle3im, x712n);
        let temp_b5_8 = _mm_mul_ps(self.twiddle2im, x811n);
        let temp_b5_9 = _mm_mul_ps(self.twiddle7im, x910n);
        let temp_b6_1 = _mm_mul_ps(self.twiddle6im, x118n);
        let temp_b6_2 = _mm_mul_ps(self.twiddle7im, x217n);
        let temp_b6_3 = _mm_mul_ps(self.twiddle1im, x316n);
        let temp_b6_4 = _mm_mul_ps(self.twiddle5im, x415n);
        let temp_b6_5 = _mm_mul_ps(self.twiddle8im, x514n);
        let temp_b6_6 = _mm_mul_ps(self.twiddle2im, x613n);
        let temp_b6_7 = _mm_mul_ps(self.twiddle4im, x712n);
        let temp_b6_8 = _mm_mul_ps(self.twiddle9im, x811n);
        let temp_b6_9 = _mm_mul_ps(self.twiddle3im, x910n);
        let temp_b7_1 = _mm_mul_ps(self.twiddle7im, x118n);
        let temp_b7_2 = _mm_mul_ps(self.twiddle5im, x217n);
        let temp_b7_3 = _mm_mul_ps(self.twiddle2im, x316n);
        let temp_b7_4 = _mm_mul_ps(self.twiddle9im, x415n);
        let temp_b7_5 = _mm_mul_ps(self.twiddle3im, x514n);
        let temp_b7_6 = _mm_mul_ps(self.twiddle4im, x613n);
        let temp_b7_7 = _mm_mul_ps(self.twiddle8im, x712n);
        let temp_b7_8 = _mm_mul_ps(self.twiddle1im, x811n);
        let temp_b7_9 = _mm_mul_ps(self.twiddle6im, x910n);
        let temp_b8_1 = _mm_mul_ps(self.twiddle8im, x118n);
        let temp_b8_2 = _mm_mul_ps(self.twiddle3im, x217n);
        let temp_b8_3 = _mm_mul_ps(self.twiddle5im, x316n);
        let temp_b8_4 = _mm_mul_ps(self.twiddle6im, x415n);
        let temp_b8_5 = _mm_mul_ps(self.twiddle2im, x514n);
        let temp_b8_6 = _mm_mul_ps(self.twiddle9im, x613n);
        let temp_b8_7 = _mm_mul_ps(self.twiddle1im, x712n);
        let temp_b8_8 = _mm_mul_ps(self.twiddle7im, x811n);
        let temp_b8_9 = _mm_mul_ps(self.twiddle4im, x910n);
        let temp_b9_1 = _mm_mul_ps(self.twiddle9im, x118n);
        let temp_b9_2 = _mm_mul_ps(self.twiddle1im, x217n);
        let temp_b9_3 = _mm_mul_ps(self.twiddle8im, x316n);
        let temp_b9_4 = _mm_mul_ps(self.twiddle2im, x415n);
        let temp_b9_5 = _mm_mul_ps(self.twiddle7im, x514n);
        let temp_b9_6 = _mm_mul_ps(self.twiddle3im, x613n);
        let temp_b9_7 = _mm_mul_ps(self.twiddle6im, x712n);
        let temp_b9_8 = _mm_mul_ps(self.twiddle4im, x811n);
        let temp_b9_9 = _mm_mul_ps(self.twiddle5im, x910n);

        let temp_a1 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a1_1,
                _mm_add_ps(
                    temp_a1_2,
                    _mm_add_ps(
                        temp_a1_3,
                        _mm_add_ps(
                            temp_a1_4,
                            _mm_add_ps(
                                temp_a1_5,
                                _mm_add_ps(
                                    temp_a1_6,
                                    _mm_add_ps(temp_a1_7, _mm_add_ps(temp_a1_8, temp_a1_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a2_1,
                _mm_add_ps(
                    temp_a2_2,
                    _mm_add_ps(
                        temp_a2_3,
                        _mm_add_ps(
                            temp_a2_4,
                            _mm_add_ps(
                                temp_a2_5,
                                _mm_add_ps(
                                    temp_a2_6,
                                    _mm_add_ps(temp_a2_7, _mm_add_ps(temp_a2_8, temp_a2_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a3_1,
                _mm_add_ps(
                    temp_a3_2,
                    _mm_add_ps(
                        temp_a3_3,
                        _mm_add_ps(
                            temp_a3_4,
                            _mm_add_ps(
                                temp_a3_5,
                                _mm_add_ps(
                                    temp_a3_6,
                                    _mm_add_ps(temp_a3_7, _mm_add_ps(temp_a3_8, temp_a3_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a4_1,
                _mm_add_ps(
                    temp_a4_2,
                    _mm_add_ps(
                        temp_a4_3,
                        _mm_add_ps(
                            temp_a4_4,
                            _mm_add_ps(
                                temp_a4_5,
                                _mm_add_ps(
                                    temp_a4_6,
                                    _mm_add_ps(temp_a4_7, _mm_add_ps(temp_a4_8, temp_a4_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a5_1,
                _mm_add_ps(
                    temp_a5_2,
                    _mm_add_ps(
                        temp_a5_3,
                        _mm_add_ps(
                            temp_a5_4,
                            _mm_add_ps(
                                temp_a5_5,
                                _mm_add_ps(
                                    temp_a5_6,
                                    _mm_add_ps(temp_a5_7, _mm_add_ps(temp_a5_8, temp_a5_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a6_1,
                _mm_add_ps(
                    temp_a6_2,
                    _mm_add_ps(
                        temp_a6_3,
                        _mm_add_ps(
                            temp_a6_4,
                            _mm_add_ps(
                                temp_a6_5,
                                _mm_add_ps(
                                    temp_a6_6,
                                    _mm_add_ps(temp_a6_7, _mm_add_ps(temp_a6_8, temp_a6_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a7_1,
                _mm_add_ps(
                    temp_a7_2,
                    _mm_add_ps(
                        temp_a7_3,
                        _mm_add_ps(
                            temp_a7_4,
                            _mm_add_ps(
                                temp_a7_5,
                                _mm_add_ps(
                                    temp_a7_6,
                                    _mm_add_ps(temp_a7_7, _mm_add_ps(temp_a7_8, temp_a7_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a8_1,
                _mm_add_ps(
                    temp_a8_2,
                    _mm_add_ps(
                        temp_a8_3,
                        _mm_add_ps(
                            temp_a8_4,
                            _mm_add_ps(
                                temp_a8_5,
                                _mm_add_ps(
                                    temp_a8_6,
                                    _mm_add_ps(temp_a8_7, _mm_add_ps(temp_a8_8, temp_a8_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a9 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a9_1,
                _mm_add_ps(
                    temp_a9_2,
                    _mm_add_ps(
                        temp_a9_3,
                        _mm_add_ps(
                            temp_a9_4,
                            _mm_add_ps(
                                temp_a9_5,
                                _mm_add_ps(
                                    temp_a9_6,
                                    _mm_add_ps(temp_a9_7, _mm_add_ps(temp_a9_8, temp_a9_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_ps(
            temp_b1_1,
            _mm_add_ps(
                temp_b1_2,
                _mm_add_ps(
                    temp_b1_3,
                    _mm_add_ps(
                        temp_b1_4,
                        _mm_add_ps(
                            temp_b1_5,
                            _mm_add_ps(
                                temp_b1_6,
                                _mm_add_ps(temp_b1_7, _mm_add_ps(temp_b1_8, temp_b1_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_ps(
            temp_b2_1,
            _mm_add_ps(
                temp_b2_2,
                _mm_add_ps(
                    temp_b2_3,
                    _mm_sub_ps(
                        temp_b2_4,
                        _mm_add_ps(
                            temp_b2_5,
                            _mm_add_ps(
                                temp_b2_6,
                                _mm_add_ps(temp_b2_7, _mm_add_ps(temp_b2_8, temp_b2_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_ps(
            temp_b3_1,
            _mm_add_ps(
                temp_b3_2,
                _mm_sub_ps(
                    temp_b3_3,
                    _mm_add_ps(
                        temp_b3_4,
                        _mm_add_ps(
                            temp_b3_5,
                            _mm_sub_ps(
                                temp_b3_6,
                                _mm_add_ps(temp_b3_7, _mm_add_ps(temp_b3_8, temp_b3_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_ps(
            temp_b4_1,
            _mm_sub_ps(
                temp_b4_2,
                _mm_add_ps(
                    temp_b4_3,
                    _mm_sub_ps(
                        temp_b4_4,
                        _mm_add_ps(
                            temp_b4_5,
                            _mm_add_ps(
                                temp_b4_6,
                                _mm_sub_ps(temp_b4_7, _mm_add_ps(temp_b4_8, temp_b4_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_sub_ps(
            temp_b5_1,
            _mm_add_ps(
                temp_b5_2,
                _mm_sub_ps(
                    temp_b5_3,
                    _mm_add_ps(
                        temp_b5_4,
                        _mm_sub_ps(
                            temp_b5_5,
                            _mm_add_ps(
                                temp_b5_6,
                                _mm_sub_ps(temp_b5_7, _mm_add_ps(temp_b5_8, temp_b5_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_sub_ps(
            temp_b6_1,
            _mm_add_ps(
                temp_b6_2,
                _mm_sub_ps(
                    temp_b6_3,
                    _mm_sub_ps(
                        temp_b6_4,
                        _mm_add_ps(
                            temp_b6_5,
                            _mm_sub_ps(
                                temp_b6_6,
                                _mm_sub_ps(temp_b6_7, _mm_add_ps(temp_b6_8, temp_b6_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_sub_ps(
            temp_b7_1,
            _mm_sub_ps(
                temp_b7_2,
                _mm_add_ps(
                    temp_b7_3,
                    _mm_sub_ps(
                        temp_b7_4,
                        _mm_sub_ps(
                            temp_b7_5,
                            _mm_sub_ps(
                                temp_b7_6,
                                _mm_add_ps(temp_b7_7, _mm_sub_ps(temp_b7_8, temp_b7_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_ps(
            temp_b8_1,
            _mm_sub_ps(
                temp_b8_2,
                _mm_sub_ps(
                    temp_b8_3,
                    _mm_sub_ps(
                        temp_b8_4,
                        _mm_sub_ps(
                            temp_b8_5,
                            _mm_add_ps(
                                temp_b8_6,
                                _mm_sub_ps(temp_b8_7, _mm_sub_ps(temp_b8_8, temp_b8_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b9 = _mm_sub_ps(
            temp_b9_1,
            _mm_sub_ps(
                temp_b9_2,
                _mm_sub_ps(
                    temp_b9_3,
                    _mm_sub_ps(
                        temp_b9_4,
                        _mm_sub_ps(
                            temp_b9_5,
                            _mm_sub_ps(
                                temp_b9_6,
                                _mm_sub_ps(temp_b9_7, _mm_sub_ps(temp_b9_8, temp_b9_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate_both(temp_b1);
        let temp_b2_rot = self.rotate.rotate_both(temp_b2);
        let temp_b3_rot = self.rotate.rotate_both(temp_b3);
        let temp_b4_rot = self.rotate.rotate_both(temp_b4);
        let temp_b5_rot = self.rotate.rotate_both(temp_b5);
        let temp_b6_rot = self.rotate.rotate_both(temp_b6);
        let temp_b7_rot = self.rotate.rotate_both(temp_b7);
        let temp_b8_rot = self.rotate.rotate_both(temp_b8);
        let temp_b9_rot = self.rotate.rotate_both(temp_b9);

        let x0 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                x118p,
                _mm_add_ps(
                    x217p,
                    _mm_add_ps(
                        x316p,
                        _mm_add_ps(
                            x415p,
                            _mm_add_ps(
                                x514p,
                                _mm_add_ps(x613p, _mm_add_ps(x712p, _mm_add_ps(x811p, x910p))),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_ps(temp_a1, temp_b1_rot);
        let x2 = _mm_add_ps(temp_a2, temp_b2_rot);
        let x3 = _mm_add_ps(temp_a3, temp_b3_rot);
        let x4 = _mm_add_ps(temp_a4, temp_b4_rot);
        let x5 = _mm_add_ps(temp_a5, temp_b5_rot);
        let x6 = _mm_add_ps(temp_a6, temp_b6_rot);
        let x7 = _mm_add_ps(temp_a7, temp_b7_rot);
        let x8 = _mm_add_ps(temp_a8, temp_b8_rot);
        let x9 = _mm_add_ps(temp_a9, temp_b9_rot);
        let x10 = _mm_sub_ps(temp_a9, temp_b9_rot);
        let x11 = _mm_sub_ps(temp_a8, temp_b8_rot);
        let x12 = _mm_sub_ps(temp_a7, temp_b7_rot);
        let x13 = _mm_sub_ps(temp_a6, temp_b6_rot);
        let x14 = _mm_sub_ps(temp_a5, temp_b5_rot);
        let x15 = _mm_sub_ps(temp_a4, temp_b4_rot);
        let x16 = _mm_sub_ps(temp_a3, temp_b3_rot);
        let x17 = _mm_sub_ps(temp_a2, temp_b2_rot);
        let x18 = _mm_sub_ps(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
        ]
    }
}

//   _  ___             __   _  _   _     _ _
//  / |/ _ \           / /_ | || | | |__ (_) |_
//  | | (_) |  _____  | '_ \| || |_| '_ \| | __|
//  | |\__, | |_____| | (_) |__   _| |_) | | |_
//  |_|  /_/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly19<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
    twiddle3re: __m128d,
    twiddle3im: __m128d,
    twiddle4re: __m128d,
    twiddle4im: __m128d,
    twiddle5re: __m128d,
    twiddle5im: __m128d,
    twiddle6re: __m128d,
    twiddle6im: __m128d,
    twiddle7re: __m128d,
    twiddle7im: __m128d,
    twiddle8re: __m128d,
    twiddle8im: __m128d,
    twiddle9re: __m128d,
    twiddle9im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly19, 19, |this: &SseF64Butterfly19<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly19, 19, |this: &SseF64Butterfly19<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly19<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 19, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 19, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 19, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 19, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 19, direction);
        let tw6: Complex<f64> = twiddles::compute_twiddle(6, 19, direction);
        let tw7: Complex<f64> = twiddles::compute_twiddle(7, 19, direction);
        let tw8: Complex<f64> = twiddles::compute_twiddle(8, 19, direction);
        let tw9: Complex<f64> = twiddles::compute_twiddle(9, 19, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_pd(tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_pd(tw3.im, tw3.im) };
        let twiddle4re = unsafe { _mm_set_pd(tw4.re, tw4.re) };
        let twiddle4im = unsafe { _mm_set_pd(tw4.im, tw4.im) };
        let twiddle5re = unsafe { _mm_set_pd(tw5.re, tw5.re) };
        let twiddle5im = unsafe { _mm_set_pd(tw5.im, tw5.im) };
        let twiddle6re = unsafe { _mm_set_pd(tw6.re, tw6.re) };
        let twiddle6im = unsafe { _mm_set_pd(tw6.im, tw6.im) };
        let twiddle7re = unsafe { _mm_set_pd(tw7.re, tw7.re) };
        let twiddle7im = unsafe { _mm_set_pd(tw7.im, tw7.im) };
        let twiddle8re = unsafe { _mm_set_pd(tw8.re, tw8.re) };
        let twiddle8im = unsafe { _mm_set_pd(tw8.im, tw8.im) };
        let twiddle9re = unsafe { _mm_set_pd(tw9.re, tw9.re) };
        let twiddle9im = unsafe { _mm_set_pd(tw9.im, tw9.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
            twiddle9re,
            twiddle9im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let v1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let v2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let v3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let v4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let v5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let v6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let v7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let v8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let v9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let v10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);
        let v11 = _mm_loadu_pd(input.as_ptr().add(11) as *const f64);
        let v12 = _mm_loadu_pd(input.as_ptr().add(12) as *const f64);
        let v13 = _mm_loadu_pd(input.as_ptr().add(13) as *const f64);
        let v14 = _mm_loadu_pd(input.as_ptr().add(14) as *const f64);
        let v15 = _mm_loadu_pd(input.as_ptr().add(15) as *const f64);
        let v16 = _mm_loadu_pd(input.as_ptr().add(16) as *const f64);
        let v17 = _mm_loadu_pd(input.as_ptr().add(17) as *const f64);
        let v18 = _mm_loadu_pd(input.as_ptr().add(18) as *const f64);

        let out = self.perform_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        ]);

        let val = std::mem::transmute::<[__m128d; 19], [Complex<f64>; 19]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
        *output_slice.add(6) = val[6];
        *output_slice.add(7) = val[7];
        *output_slice.add(8) = val[8];
        *output_slice.add(9) = val[9];
        *output_slice.add(10) = val[10];
        *output_slice.add(11) = val[11];
        *output_slice.add(12) = val[12];
        *output_slice.add(13) = val[13];
        *output_slice.add(14) = val[14];
        *output_slice.add(15) = val[15];
        *output_slice.add(16) = val[16];
        *output_slice.add(17) = val[17];
        *output_slice.add(18) = val[18];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 19]) -> [__m128d; 19] {
        // This is a SSE translation of the scalar 19-point butterfly
        let x118p = _mm_add_pd(values[1], values[18]);
        let x118n = _mm_sub_pd(values[1], values[18]);
        let x217p = _mm_add_pd(values[2], values[17]);
        let x217n = _mm_sub_pd(values[2], values[17]);
        let x316p = _mm_add_pd(values[3], values[16]);
        let x316n = _mm_sub_pd(values[3], values[16]);
        let x415p = _mm_add_pd(values[4], values[15]);
        let x415n = _mm_sub_pd(values[4], values[15]);
        let x514p = _mm_add_pd(values[5], values[14]);
        let x514n = _mm_sub_pd(values[5], values[14]);
        let x613p = _mm_add_pd(values[6], values[13]);
        let x613n = _mm_sub_pd(values[6], values[13]);
        let x712p = _mm_add_pd(values[7], values[12]);
        let x712n = _mm_sub_pd(values[7], values[12]);
        let x811p = _mm_add_pd(values[8], values[11]);
        let x811n = _mm_sub_pd(values[8], values[11]);
        let x910p = _mm_add_pd(values[9], values[10]);
        let x910n = _mm_sub_pd(values[9], values[10]);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x118p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x217p);
        let temp_a1_3 = _mm_mul_pd(self.twiddle3re, x316p);
        let temp_a1_4 = _mm_mul_pd(self.twiddle4re, x415p);
        let temp_a1_5 = _mm_mul_pd(self.twiddle5re, x514p);
        let temp_a1_6 = _mm_mul_pd(self.twiddle6re, x613p);
        let temp_a1_7 = _mm_mul_pd(self.twiddle7re, x712p);
        let temp_a1_8 = _mm_mul_pd(self.twiddle8re, x811p);
        let temp_a1_9 = _mm_mul_pd(self.twiddle9re, x910p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x118p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle4re, x217p);
        let temp_a2_3 = _mm_mul_pd(self.twiddle6re, x316p);
        let temp_a2_4 = _mm_mul_pd(self.twiddle8re, x415p);
        let temp_a2_5 = _mm_mul_pd(self.twiddle9re, x514p);
        let temp_a2_6 = _mm_mul_pd(self.twiddle7re, x613p);
        let temp_a2_7 = _mm_mul_pd(self.twiddle5re, x712p);
        let temp_a2_8 = _mm_mul_pd(self.twiddle3re, x811p);
        let temp_a2_9 = _mm_mul_pd(self.twiddle1re, x910p);
        let temp_a3_1 = _mm_mul_pd(self.twiddle3re, x118p);
        let temp_a3_2 = _mm_mul_pd(self.twiddle6re, x217p);
        let temp_a3_3 = _mm_mul_pd(self.twiddle9re, x316p);
        let temp_a3_4 = _mm_mul_pd(self.twiddle7re, x415p);
        let temp_a3_5 = _mm_mul_pd(self.twiddle4re, x514p);
        let temp_a3_6 = _mm_mul_pd(self.twiddle1re, x613p);
        let temp_a3_7 = _mm_mul_pd(self.twiddle2re, x712p);
        let temp_a3_8 = _mm_mul_pd(self.twiddle5re, x811p);
        let temp_a3_9 = _mm_mul_pd(self.twiddle8re, x910p);
        let temp_a4_1 = _mm_mul_pd(self.twiddle4re, x118p);
        let temp_a4_2 = _mm_mul_pd(self.twiddle8re, x217p);
        let temp_a4_3 = _mm_mul_pd(self.twiddle7re, x316p);
        let temp_a4_4 = _mm_mul_pd(self.twiddle3re, x415p);
        let temp_a4_5 = _mm_mul_pd(self.twiddle1re, x514p);
        let temp_a4_6 = _mm_mul_pd(self.twiddle5re, x613p);
        let temp_a4_7 = _mm_mul_pd(self.twiddle9re, x712p);
        let temp_a4_8 = _mm_mul_pd(self.twiddle6re, x811p);
        let temp_a4_9 = _mm_mul_pd(self.twiddle2re, x910p);
        let temp_a5_1 = _mm_mul_pd(self.twiddle5re, x118p);
        let temp_a5_2 = _mm_mul_pd(self.twiddle9re, x217p);
        let temp_a5_3 = _mm_mul_pd(self.twiddle4re, x316p);
        let temp_a5_4 = _mm_mul_pd(self.twiddle1re, x415p);
        let temp_a5_5 = _mm_mul_pd(self.twiddle6re, x514p);
        let temp_a5_6 = _mm_mul_pd(self.twiddle8re, x613p);
        let temp_a5_7 = _mm_mul_pd(self.twiddle3re, x712p);
        let temp_a5_8 = _mm_mul_pd(self.twiddle2re, x811p);
        let temp_a5_9 = _mm_mul_pd(self.twiddle7re, x910p);
        let temp_a6_1 = _mm_mul_pd(self.twiddle6re, x118p);
        let temp_a6_2 = _mm_mul_pd(self.twiddle7re, x217p);
        let temp_a6_3 = _mm_mul_pd(self.twiddle1re, x316p);
        let temp_a6_4 = _mm_mul_pd(self.twiddle5re, x415p);
        let temp_a6_5 = _mm_mul_pd(self.twiddle8re, x514p);
        let temp_a6_6 = _mm_mul_pd(self.twiddle2re, x613p);
        let temp_a6_7 = _mm_mul_pd(self.twiddle4re, x712p);
        let temp_a6_8 = _mm_mul_pd(self.twiddle9re, x811p);
        let temp_a6_9 = _mm_mul_pd(self.twiddle3re, x910p);
        let temp_a7_1 = _mm_mul_pd(self.twiddle7re, x118p);
        let temp_a7_2 = _mm_mul_pd(self.twiddle5re, x217p);
        let temp_a7_3 = _mm_mul_pd(self.twiddle2re, x316p);
        let temp_a7_4 = _mm_mul_pd(self.twiddle9re, x415p);
        let temp_a7_5 = _mm_mul_pd(self.twiddle3re, x514p);
        let temp_a7_6 = _mm_mul_pd(self.twiddle4re, x613p);
        let temp_a7_7 = _mm_mul_pd(self.twiddle8re, x712p);
        let temp_a7_8 = _mm_mul_pd(self.twiddle1re, x811p);
        let temp_a7_9 = _mm_mul_pd(self.twiddle6re, x910p);
        let temp_a8_1 = _mm_mul_pd(self.twiddle8re, x118p);
        let temp_a8_2 = _mm_mul_pd(self.twiddle3re, x217p);
        let temp_a8_3 = _mm_mul_pd(self.twiddle5re, x316p);
        let temp_a8_4 = _mm_mul_pd(self.twiddle6re, x415p);
        let temp_a8_5 = _mm_mul_pd(self.twiddle2re, x514p);
        let temp_a8_6 = _mm_mul_pd(self.twiddle9re, x613p);
        let temp_a8_7 = _mm_mul_pd(self.twiddle1re, x712p);
        let temp_a8_8 = _mm_mul_pd(self.twiddle7re, x811p);
        let temp_a8_9 = _mm_mul_pd(self.twiddle4re, x910p);
        let temp_a9_1 = _mm_mul_pd(self.twiddle9re, x118p);
        let temp_a9_2 = _mm_mul_pd(self.twiddle1re, x217p);
        let temp_a9_3 = _mm_mul_pd(self.twiddle8re, x316p);
        let temp_a9_4 = _mm_mul_pd(self.twiddle2re, x415p);
        let temp_a9_5 = _mm_mul_pd(self.twiddle7re, x514p);
        let temp_a9_6 = _mm_mul_pd(self.twiddle3re, x613p);
        let temp_a9_7 = _mm_mul_pd(self.twiddle6re, x712p);
        let temp_a9_8 = _mm_mul_pd(self.twiddle4re, x811p);
        let temp_a9_9 = _mm_mul_pd(self.twiddle5re, x910p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x118n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x217n);
        let temp_b1_3 = _mm_mul_pd(self.twiddle3im, x316n);
        let temp_b1_4 = _mm_mul_pd(self.twiddle4im, x415n);
        let temp_b1_5 = _mm_mul_pd(self.twiddle5im, x514n);
        let temp_b1_6 = _mm_mul_pd(self.twiddle6im, x613n);
        let temp_b1_7 = _mm_mul_pd(self.twiddle7im, x712n);
        let temp_b1_8 = _mm_mul_pd(self.twiddle8im, x811n);
        let temp_b1_9 = _mm_mul_pd(self.twiddle9im, x910n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x118n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle4im, x217n);
        let temp_b2_3 = _mm_mul_pd(self.twiddle6im, x316n);
        let temp_b2_4 = _mm_mul_pd(self.twiddle8im, x415n);
        let temp_b2_5 = _mm_mul_pd(self.twiddle9im, x514n);
        let temp_b2_6 = _mm_mul_pd(self.twiddle7im, x613n);
        let temp_b2_7 = _mm_mul_pd(self.twiddle5im, x712n);
        let temp_b2_8 = _mm_mul_pd(self.twiddle3im, x811n);
        let temp_b2_9 = _mm_mul_pd(self.twiddle1im, x910n);
        let temp_b3_1 = _mm_mul_pd(self.twiddle3im, x118n);
        let temp_b3_2 = _mm_mul_pd(self.twiddle6im, x217n);
        let temp_b3_3 = _mm_mul_pd(self.twiddle9im, x316n);
        let temp_b3_4 = _mm_mul_pd(self.twiddle7im, x415n);
        let temp_b3_5 = _mm_mul_pd(self.twiddle4im, x514n);
        let temp_b3_6 = _mm_mul_pd(self.twiddle1im, x613n);
        let temp_b3_7 = _mm_mul_pd(self.twiddle2im, x712n);
        let temp_b3_8 = _mm_mul_pd(self.twiddle5im, x811n);
        let temp_b3_9 = _mm_mul_pd(self.twiddle8im, x910n);
        let temp_b4_1 = _mm_mul_pd(self.twiddle4im, x118n);
        let temp_b4_2 = _mm_mul_pd(self.twiddle8im, x217n);
        let temp_b4_3 = _mm_mul_pd(self.twiddle7im, x316n);
        let temp_b4_4 = _mm_mul_pd(self.twiddle3im, x415n);
        let temp_b4_5 = _mm_mul_pd(self.twiddle1im, x514n);
        let temp_b4_6 = _mm_mul_pd(self.twiddle5im, x613n);
        let temp_b4_7 = _mm_mul_pd(self.twiddle9im, x712n);
        let temp_b4_8 = _mm_mul_pd(self.twiddle6im, x811n);
        let temp_b4_9 = _mm_mul_pd(self.twiddle2im, x910n);
        let temp_b5_1 = _mm_mul_pd(self.twiddle5im, x118n);
        let temp_b5_2 = _mm_mul_pd(self.twiddle9im, x217n);
        let temp_b5_3 = _mm_mul_pd(self.twiddle4im, x316n);
        let temp_b5_4 = _mm_mul_pd(self.twiddle1im, x415n);
        let temp_b5_5 = _mm_mul_pd(self.twiddle6im, x514n);
        let temp_b5_6 = _mm_mul_pd(self.twiddle8im, x613n);
        let temp_b5_7 = _mm_mul_pd(self.twiddle3im, x712n);
        let temp_b5_8 = _mm_mul_pd(self.twiddle2im, x811n);
        let temp_b5_9 = _mm_mul_pd(self.twiddle7im, x910n);
        let temp_b6_1 = _mm_mul_pd(self.twiddle6im, x118n);
        let temp_b6_2 = _mm_mul_pd(self.twiddle7im, x217n);
        let temp_b6_3 = _mm_mul_pd(self.twiddle1im, x316n);
        let temp_b6_4 = _mm_mul_pd(self.twiddle5im, x415n);
        let temp_b6_5 = _mm_mul_pd(self.twiddle8im, x514n);
        let temp_b6_6 = _mm_mul_pd(self.twiddle2im, x613n);
        let temp_b6_7 = _mm_mul_pd(self.twiddle4im, x712n);
        let temp_b6_8 = _mm_mul_pd(self.twiddle9im, x811n);
        let temp_b6_9 = _mm_mul_pd(self.twiddle3im, x910n);
        let temp_b7_1 = _mm_mul_pd(self.twiddle7im, x118n);
        let temp_b7_2 = _mm_mul_pd(self.twiddle5im, x217n);
        let temp_b7_3 = _mm_mul_pd(self.twiddle2im, x316n);
        let temp_b7_4 = _mm_mul_pd(self.twiddle9im, x415n);
        let temp_b7_5 = _mm_mul_pd(self.twiddle3im, x514n);
        let temp_b7_6 = _mm_mul_pd(self.twiddle4im, x613n);
        let temp_b7_7 = _mm_mul_pd(self.twiddle8im, x712n);
        let temp_b7_8 = _mm_mul_pd(self.twiddle1im, x811n);
        let temp_b7_9 = _mm_mul_pd(self.twiddle6im, x910n);
        let temp_b8_1 = _mm_mul_pd(self.twiddle8im, x118n);
        let temp_b8_2 = _mm_mul_pd(self.twiddle3im, x217n);
        let temp_b8_3 = _mm_mul_pd(self.twiddle5im, x316n);
        let temp_b8_4 = _mm_mul_pd(self.twiddle6im, x415n);
        let temp_b8_5 = _mm_mul_pd(self.twiddle2im, x514n);
        let temp_b8_6 = _mm_mul_pd(self.twiddle9im, x613n);
        let temp_b8_7 = _mm_mul_pd(self.twiddle1im, x712n);
        let temp_b8_8 = _mm_mul_pd(self.twiddle7im, x811n);
        let temp_b8_9 = _mm_mul_pd(self.twiddle4im, x910n);
        let temp_b9_1 = _mm_mul_pd(self.twiddle9im, x118n);
        let temp_b9_2 = _mm_mul_pd(self.twiddle1im, x217n);
        let temp_b9_3 = _mm_mul_pd(self.twiddle8im, x316n);
        let temp_b9_4 = _mm_mul_pd(self.twiddle2im, x415n);
        let temp_b9_5 = _mm_mul_pd(self.twiddle7im, x514n);
        let temp_b9_6 = _mm_mul_pd(self.twiddle3im, x613n);
        let temp_b9_7 = _mm_mul_pd(self.twiddle6im, x712n);
        let temp_b9_8 = _mm_mul_pd(self.twiddle4im, x811n);
        let temp_b9_9 = _mm_mul_pd(self.twiddle5im, x910n);

        let temp_a1 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a1_1,
                _mm_add_pd(
                    temp_a1_2,
                    _mm_add_pd(
                        temp_a1_3,
                        _mm_add_pd(
                            temp_a1_4,
                            _mm_add_pd(
                                temp_a1_5,
                                _mm_add_pd(
                                    temp_a1_6,
                                    _mm_add_pd(temp_a1_7, _mm_add_pd(temp_a1_8, temp_a1_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a2_1,
                _mm_add_pd(
                    temp_a2_2,
                    _mm_add_pd(
                        temp_a2_3,
                        _mm_add_pd(
                            temp_a2_4,
                            _mm_add_pd(
                                temp_a2_5,
                                _mm_add_pd(
                                    temp_a2_6,
                                    _mm_add_pd(temp_a2_7, _mm_add_pd(temp_a2_8, temp_a2_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a3_1,
                _mm_add_pd(
                    temp_a3_2,
                    _mm_add_pd(
                        temp_a3_3,
                        _mm_add_pd(
                            temp_a3_4,
                            _mm_add_pd(
                                temp_a3_5,
                                _mm_add_pd(
                                    temp_a3_6,
                                    _mm_add_pd(temp_a3_7, _mm_add_pd(temp_a3_8, temp_a3_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a4_1,
                _mm_add_pd(
                    temp_a4_2,
                    _mm_add_pd(
                        temp_a4_3,
                        _mm_add_pd(
                            temp_a4_4,
                            _mm_add_pd(
                                temp_a4_5,
                                _mm_add_pd(
                                    temp_a4_6,
                                    _mm_add_pd(temp_a4_7, _mm_add_pd(temp_a4_8, temp_a4_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a5_1,
                _mm_add_pd(
                    temp_a5_2,
                    _mm_add_pd(
                        temp_a5_3,
                        _mm_add_pd(
                            temp_a5_4,
                            _mm_add_pd(
                                temp_a5_5,
                                _mm_add_pd(
                                    temp_a5_6,
                                    _mm_add_pd(temp_a5_7, _mm_add_pd(temp_a5_8, temp_a5_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a6_1,
                _mm_add_pd(
                    temp_a6_2,
                    _mm_add_pd(
                        temp_a6_3,
                        _mm_add_pd(
                            temp_a6_4,
                            _mm_add_pd(
                                temp_a6_5,
                                _mm_add_pd(
                                    temp_a6_6,
                                    _mm_add_pd(temp_a6_7, _mm_add_pd(temp_a6_8, temp_a6_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a7_1,
                _mm_add_pd(
                    temp_a7_2,
                    _mm_add_pd(
                        temp_a7_3,
                        _mm_add_pd(
                            temp_a7_4,
                            _mm_add_pd(
                                temp_a7_5,
                                _mm_add_pd(
                                    temp_a7_6,
                                    _mm_add_pd(temp_a7_7, _mm_add_pd(temp_a7_8, temp_a7_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a8_1,
                _mm_add_pd(
                    temp_a8_2,
                    _mm_add_pd(
                        temp_a8_3,
                        _mm_add_pd(
                            temp_a8_4,
                            _mm_add_pd(
                                temp_a8_5,
                                _mm_add_pd(
                                    temp_a8_6,
                                    _mm_add_pd(temp_a8_7, _mm_add_pd(temp_a8_8, temp_a8_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a9 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a9_1,
                _mm_add_pd(
                    temp_a9_2,
                    _mm_add_pd(
                        temp_a9_3,
                        _mm_add_pd(
                            temp_a9_4,
                            _mm_add_pd(
                                temp_a9_5,
                                _mm_add_pd(
                                    temp_a9_6,
                                    _mm_add_pd(temp_a9_7, _mm_add_pd(temp_a9_8, temp_a9_9)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_pd(
            temp_b1_1,
            _mm_add_pd(
                temp_b1_2,
                _mm_add_pd(
                    temp_b1_3,
                    _mm_add_pd(
                        temp_b1_4,
                        _mm_add_pd(
                            temp_b1_5,
                            _mm_add_pd(
                                temp_b1_6,
                                _mm_add_pd(temp_b1_7, _mm_add_pd(temp_b1_8, temp_b1_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_pd(
            temp_b2_1,
            _mm_add_pd(
                temp_b2_2,
                _mm_add_pd(
                    temp_b2_3,
                    _mm_sub_pd(
                        temp_b2_4,
                        _mm_add_pd(
                            temp_b2_5,
                            _mm_add_pd(
                                temp_b2_6,
                                _mm_add_pd(temp_b2_7, _mm_add_pd(temp_b2_8, temp_b2_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_pd(
            temp_b3_1,
            _mm_add_pd(
                temp_b3_2,
                _mm_sub_pd(
                    temp_b3_3,
                    _mm_add_pd(
                        temp_b3_4,
                        _mm_add_pd(
                            temp_b3_5,
                            _mm_sub_pd(
                                temp_b3_6,
                                _mm_add_pd(temp_b3_7, _mm_add_pd(temp_b3_8, temp_b3_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_pd(
            temp_b4_1,
            _mm_sub_pd(
                temp_b4_2,
                _mm_add_pd(
                    temp_b4_3,
                    _mm_sub_pd(
                        temp_b4_4,
                        _mm_add_pd(
                            temp_b4_5,
                            _mm_add_pd(
                                temp_b4_6,
                                _mm_sub_pd(temp_b4_7, _mm_add_pd(temp_b4_8, temp_b4_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_sub_pd(
            temp_b5_1,
            _mm_add_pd(
                temp_b5_2,
                _mm_sub_pd(
                    temp_b5_3,
                    _mm_add_pd(
                        temp_b5_4,
                        _mm_sub_pd(
                            temp_b5_5,
                            _mm_add_pd(
                                temp_b5_6,
                                _mm_sub_pd(temp_b5_7, _mm_add_pd(temp_b5_8, temp_b5_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_sub_pd(
            temp_b6_1,
            _mm_add_pd(
                temp_b6_2,
                _mm_sub_pd(
                    temp_b6_3,
                    _mm_sub_pd(
                        temp_b6_4,
                        _mm_add_pd(
                            temp_b6_5,
                            _mm_sub_pd(
                                temp_b6_6,
                                _mm_sub_pd(temp_b6_7, _mm_add_pd(temp_b6_8, temp_b6_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_sub_pd(
            temp_b7_1,
            _mm_sub_pd(
                temp_b7_2,
                _mm_add_pd(
                    temp_b7_3,
                    _mm_sub_pd(
                        temp_b7_4,
                        _mm_sub_pd(
                            temp_b7_5,
                            _mm_sub_pd(
                                temp_b7_6,
                                _mm_add_pd(temp_b7_7, _mm_sub_pd(temp_b7_8, temp_b7_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_pd(
            temp_b8_1,
            _mm_sub_pd(
                temp_b8_2,
                _mm_sub_pd(
                    temp_b8_3,
                    _mm_sub_pd(
                        temp_b8_4,
                        _mm_sub_pd(
                            temp_b8_5,
                            _mm_add_pd(
                                temp_b8_6,
                                _mm_sub_pd(temp_b8_7, _mm_sub_pd(temp_b8_8, temp_b8_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b9 = _mm_sub_pd(
            temp_b9_1,
            _mm_sub_pd(
                temp_b9_2,
                _mm_sub_pd(
                    temp_b9_3,
                    _mm_sub_pd(
                        temp_b9_4,
                        _mm_sub_pd(
                            temp_b9_5,
                            _mm_sub_pd(
                                temp_b9_6,
                                _mm_sub_pd(temp_b9_7, _mm_sub_pd(temp_b9_8, temp_b9_9)),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);
        let temp_b6_rot = self.rotate.rotate(temp_b6);
        let temp_b7_rot = self.rotate.rotate(temp_b7);
        let temp_b8_rot = self.rotate.rotate(temp_b8);
        let temp_b9_rot = self.rotate.rotate(temp_b9);

        let x0 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                x118p,
                _mm_add_pd(
                    x217p,
                    _mm_add_pd(
                        x316p,
                        _mm_add_pd(
                            x415p,
                            _mm_add_pd(
                                x514p,
                                _mm_add_pd(x613p, _mm_add_pd(x712p, _mm_add_pd(x811p, x910p))),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
        let x4 = _mm_add_pd(temp_a4, temp_b4_rot);
        let x5 = _mm_add_pd(temp_a5, temp_b5_rot);
        let x6 = _mm_add_pd(temp_a6, temp_b6_rot);
        let x7 = _mm_add_pd(temp_a7, temp_b7_rot);
        let x8 = _mm_add_pd(temp_a8, temp_b8_rot);
        let x9 = _mm_add_pd(temp_a9, temp_b9_rot);
        let x10 = _mm_sub_pd(temp_a9, temp_b9_rot);
        let x11 = _mm_sub_pd(temp_a8, temp_b8_rot);
        let x12 = _mm_sub_pd(temp_a7, temp_b7_rot);
        let x13 = _mm_sub_pd(temp_a6, temp_b6_rot);
        let x14 = _mm_sub_pd(temp_a5, temp_b5_rot);
        let x15 = _mm_sub_pd(temp_a4, temp_b4_rot);
        let x16 = _mm_sub_pd(temp_a3, temp_b3_rot);
        let x17 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x18 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
        ]
    }
}

//   ____  _____           _________  _     _ _
//  |___ \|___ /          |___ /___ \| |__ (_) |_
//    __) | |_ \   _____    |_ \ __) | '_ \| | __|
//   / __/ ___) | |_____|  ___) / __/| |_) | | |_
//  |_____|____/          |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly23<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
    twiddle3re: __m128,
    twiddle3im: __m128,
    twiddle4re: __m128,
    twiddle4im: __m128,
    twiddle5re: __m128,
    twiddle5im: __m128,
    twiddle6re: __m128,
    twiddle6im: __m128,
    twiddle7re: __m128,
    twiddle7im: __m128,
    twiddle8re: __m128,
    twiddle8im: __m128,
    twiddle9re: __m128,
    twiddle9im: __m128,
    twiddle10re: __m128,
    twiddle10im: __m128,
    twiddle11re: __m128,
    twiddle11im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly23, 23, |this: &SseF32Butterfly23<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly23, 23, |this: &SseF32Butterfly23<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly23<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 23, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 23, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 23, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 23, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 23, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 23, direction);
        let tw7: Complex<f32> = twiddles::compute_twiddle(7, 23, direction);
        let tw8: Complex<f32> = twiddles::compute_twiddle(8, 23, direction);
        let tw9: Complex<f32> = twiddles::compute_twiddle(9, 23, direction);
        let tw10: Complex<f32> = twiddles::compute_twiddle(10, 23, direction);
        let tw11: Complex<f32> = twiddles::compute_twiddle(11, 23, direction);
        let twiddle1re = unsafe { _mm_load1_ps(&tw1.re) };
        let twiddle1im = unsafe { _mm_load1_ps(&tw1.im) };
        let twiddle2re = unsafe { _mm_load1_ps(&tw2.re) };
        let twiddle2im = unsafe { _mm_load1_ps(&tw2.im) };
        let twiddle3re = unsafe { _mm_load1_ps(&tw3.re) };
        let twiddle3im = unsafe { _mm_load1_ps(&tw3.im) };
        let twiddle4re = unsafe { _mm_load1_ps(&tw4.re) };
        let twiddle4im = unsafe { _mm_load1_ps(&tw4.im) };
        let twiddle5re = unsafe { _mm_load1_ps(&tw5.re) };
        let twiddle5im = unsafe { _mm_load1_ps(&tw5.im) };
        let twiddle6re = unsafe { _mm_load1_ps(&tw6.re) };
        let twiddle6im = unsafe { _mm_load1_ps(&tw6.im) };
        let twiddle7re = unsafe { _mm_load1_ps(&tw7.re) };
        let twiddle7im = unsafe { _mm_load1_ps(&tw7.im) };
        let twiddle8re = unsafe { _mm_load1_ps(&tw8.re) };
        let twiddle8im = unsafe { _mm_load1_ps(&tw8.im) };
        let twiddle9re = unsafe { _mm_load1_ps(&tw9.re) };
        let twiddle9im = unsafe { _mm_load1_ps(&tw9.im) };
        let twiddle10re = unsafe { _mm_load1_ps(&tw10.re) };
        let twiddle10im = unsafe { _mm_load1_ps(&tw10.im) };
        let twiddle11re = unsafe { _mm_load1_ps(&tw11.re) };
        let twiddle11im = unsafe { _mm_load1_ps(&tw11.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
            twiddle9re,
            twiddle9im,
            twiddle10re,
            twiddle10im,
            twiddle11re,
            twiddle11im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let v1 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(1) as *const f64));
        let v2 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(2) as *const f64));
        let v3 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(3) as *const f64));
        let v4 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(4) as *const f64));
        let v5 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(5) as *const f64));
        let v6 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(6) as *const f64));
        let v7 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(7) as *const f64));
        let v8 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(8) as *const f64));
        let v9 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(9) as *const f64));
        let v10 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(10) as *const f64));
        let v11 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(11) as *const f64));
        let v12 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(12) as *const f64));
        let v13 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(13) as *const f64));
        let v14 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(14) as *const f64));
        let v15 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(15) as *const f64));
        let v16 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(16) as *const f64));
        let v17 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(17) as *const f64));
        let v18 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(18) as *const f64));
        let v19 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(19) as *const f64));
        let v20 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(20) as *const f64));
        let v21 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(21) as *const f64));
        let v22 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(22) as *const f64));

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22,
        ]);

        let val = std::mem::transmute::<[__m128; 23], [Complex<f32>; 46]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[34];
        *output_slice.add(18) = val[36];
        *output_slice.add(19) = val[38];
        *output_slice.add(20) = val[40];
        *output_slice.add(21) = val[42];
        *output_slice.add(22) = val[44];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4a5 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valuea6a7 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valuea8a9 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let valuea10a11 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let valuea12a13 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let valuea14a15 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let valuea16a17 = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let valuea18a19 = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let valuea20a21 = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let valuea22b0 = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(26) as *const f32);
        let valueb5b6 = _mm_loadu_ps(input.as_ptr().add(28) as *const f32);
        let valueb7b8 = _mm_loadu_ps(input.as_ptr().add(30) as *const f32);
        let valueb9b10 = _mm_loadu_ps(input.as_ptr().add(32) as *const f32);
        let valueb11b12 = _mm_loadu_ps(input.as_ptr().add(34) as *const f32);
        let valueb13b14 = _mm_loadu_ps(input.as_ptr().add(36) as *const f32);
        let valueb15b16 = _mm_loadu_ps(input.as_ptr().add(38) as *const f32);
        let valueb17b18 = _mm_loadu_ps(input.as_ptr().add(40) as *const f32);
        let valueb19b20 = _mm_loadu_ps(input.as_ptr().add(42) as *const f32);
        let valueb21b22 = _mm_loadu_ps(input.as_ptr().add(44) as *const f32);

        let v0 = pack_1and2_f32(valuea0a1, valuea22b0);
        let v1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let v2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let v3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let v4 = pack_1and2_f32(valuea4a5, valueb3b4);
        let v5 = pack_2and1_f32(valuea4a5, valueb5b6);
        let v6 = pack_1and2_f32(valuea6a7, valueb5b6);
        let v7 = pack_2and1_f32(valuea6a7, valueb7b8);
        let v8 = pack_1and2_f32(valuea8a9, valueb7b8);
        let v9 = pack_2and1_f32(valuea8a9, valueb9b10);
        let v10 = pack_1and2_f32(valuea10a11, valueb9b10);
        let v11 = pack_2and1_f32(valuea10a11, valueb11b12);
        let v12 = pack_1and2_f32(valuea12a13, valueb11b12);
        let v13 = pack_2and1_f32(valuea12a13, valueb13b14);
        let v14 = pack_1and2_f32(valuea14a15, valueb13b14);
        let v15 = pack_2and1_f32(valuea14a15, valueb15b16);
        let v16 = pack_1and2_f32(valuea16a17, valueb15b16);
        let v17 = pack_2and1_f32(valuea16a17, valueb17b18);
        let v18 = pack_1and2_f32(valuea18a19, valueb17b18);
        let v19 = pack_2and1_f32(valuea18a19, valueb19b20);
        let v20 = pack_1and2_f32(valuea20a21, valueb19b20);
        let v21 = pack_2and1_f32(valuea20a21, valueb21b22);
        let v22 = pack_1and2_f32(valuea22b0, valueb21b22);

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22,
        ]);

        let val = std::mem::transmute::<[__m128; 23], [Complex<f32>; 46]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[34];
        *output_slice.add(18) = val[36];
        *output_slice.add(19) = val[38];
        *output_slice.add(20) = val[40];
        *output_slice.add(21) = val[42];
        *output_slice.add(22) = val[44];
        *output_slice.add(23) = val[1];
        *output_slice.add(24) = val[3];
        *output_slice.add(25) = val[5];
        *output_slice.add(26) = val[7];
        *output_slice.add(27) = val[9];
        *output_slice.add(28) = val[11];
        *output_slice.add(29) = val[13];
        *output_slice.add(30) = val[15];
        *output_slice.add(31) = val[17];
        *output_slice.add(32) = val[19];
        *output_slice.add(33) = val[21];
        *output_slice.add(34) = val[23];
        *output_slice.add(35) = val[25];
        *output_slice.add(36) = val[27];
        *output_slice.add(37) = val[29];
        *output_slice.add(38) = val[31];
        *output_slice.add(39) = val[33];
        *output_slice.add(40) = val[35];
        *output_slice.add(41) = val[37];
        *output_slice.add(42) = val[39];
        *output_slice.add(43) = val[41];
        *output_slice.add(44) = val[43];
        *output_slice.add(45) = val[45];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values: [__m128; 23]) -> [__m128; 23] {
        // This is a SSE translation of the scalar 17-point butterfly
        let x122p = _mm_add_ps(values[1], values[22]);
        let x122n = _mm_sub_ps(values[1], values[22]);
        let x221p = _mm_add_ps(values[2], values[21]);
        let x221n = _mm_sub_ps(values[2], values[21]);
        let x320p = _mm_add_ps(values[3], values[20]);
        let x320n = _mm_sub_ps(values[3], values[20]);
        let x419p = _mm_add_ps(values[4], values[19]);
        let x419n = _mm_sub_ps(values[4], values[19]);
        let x518p = _mm_add_ps(values[5], values[18]);
        let x518n = _mm_sub_ps(values[5], values[18]);
        let x617p = _mm_add_ps(values[6], values[17]);
        let x617n = _mm_sub_ps(values[6], values[17]);
        let x716p = _mm_add_ps(values[7], values[16]);
        let x716n = _mm_sub_ps(values[7], values[16]);
        let x815p = _mm_add_ps(values[8], values[15]);
        let x815n = _mm_sub_ps(values[8], values[15]);
        let x914p = _mm_add_ps(values[9], values[14]);
        let x914n = _mm_sub_ps(values[9], values[14]);
        let x1013p = _mm_add_ps(values[10], values[13]);
        let x1013n = _mm_sub_ps(values[10], values[13]);
        let x1112p = _mm_add_ps(values[11], values[12]);
        let x1112n = _mm_sub_ps(values[11], values[12]);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x122p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x221p);
        let temp_a1_3 = _mm_mul_ps(self.twiddle3re, x320p);
        let temp_a1_4 = _mm_mul_ps(self.twiddle4re, x419p);
        let temp_a1_5 = _mm_mul_ps(self.twiddle5re, x518p);
        let temp_a1_6 = _mm_mul_ps(self.twiddle6re, x617p);
        let temp_a1_7 = _mm_mul_ps(self.twiddle7re, x716p);
        let temp_a1_8 = _mm_mul_ps(self.twiddle8re, x815p);
        let temp_a1_9 = _mm_mul_ps(self.twiddle9re, x914p);
        let temp_a1_10 = _mm_mul_ps(self.twiddle10re, x1013p);
        let temp_a1_11 = _mm_mul_ps(self.twiddle11re, x1112p);
        let temp_a2_1 = _mm_mul_ps(self.twiddle2re, x122p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle4re, x221p);
        let temp_a2_3 = _mm_mul_ps(self.twiddle6re, x320p);
        let temp_a2_4 = _mm_mul_ps(self.twiddle8re, x419p);
        let temp_a2_5 = _mm_mul_ps(self.twiddle10re, x518p);
        let temp_a2_6 = _mm_mul_ps(self.twiddle11re, x617p);
        let temp_a2_7 = _mm_mul_ps(self.twiddle9re, x716p);
        let temp_a2_8 = _mm_mul_ps(self.twiddle7re, x815p);
        let temp_a2_9 = _mm_mul_ps(self.twiddle5re, x914p);
        let temp_a2_10 = _mm_mul_ps(self.twiddle3re, x1013p);
        let temp_a2_11 = _mm_mul_ps(self.twiddle1re, x1112p);
        let temp_a3_1 = _mm_mul_ps(self.twiddle3re, x122p);
        let temp_a3_2 = _mm_mul_ps(self.twiddle6re, x221p);
        let temp_a3_3 = _mm_mul_ps(self.twiddle9re, x320p);
        let temp_a3_4 = _mm_mul_ps(self.twiddle11re, x419p);
        let temp_a3_5 = _mm_mul_ps(self.twiddle8re, x518p);
        let temp_a3_6 = _mm_mul_ps(self.twiddle5re, x617p);
        let temp_a3_7 = _mm_mul_ps(self.twiddle2re, x716p);
        let temp_a3_8 = _mm_mul_ps(self.twiddle1re, x815p);
        let temp_a3_9 = _mm_mul_ps(self.twiddle4re, x914p);
        let temp_a3_10 = _mm_mul_ps(self.twiddle7re, x1013p);
        let temp_a3_11 = _mm_mul_ps(self.twiddle10re, x1112p);
        let temp_a4_1 = _mm_mul_ps(self.twiddle4re, x122p);
        let temp_a4_2 = _mm_mul_ps(self.twiddle8re, x221p);
        let temp_a4_3 = _mm_mul_ps(self.twiddle11re, x320p);
        let temp_a4_4 = _mm_mul_ps(self.twiddle7re, x419p);
        let temp_a4_5 = _mm_mul_ps(self.twiddle3re, x518p);
        let temp_a4_6 = _mm_mul_ps(self.twiddle1re, x617p);
        let temp_a4_7 = _mm_mul_ps(self.twiddle5re, x716p);
        let temp_a4_8 = _mm_mul_ps(self.twiddle9re, x815p);
        let temp_a4_9 = _mm_mul_ps(self.twiddle10re, x914p);
        let temp_a4_10 = _mm_mul_ps(self.twiddle6re, x1013p);
        let temp_a4_11 = _mm_mul_ps(self.twiddle2re, x1112p);
        let temp_a5_1 = _mm_mul_ps(self.twiddle5re, x122p);
        let temp_a5_2 = _mm_mul_ps(self.twiddle10re, x221p);
        let temp_a5_3 = _mm_mul_ps(self.twiddle8re, x320p);
        let temp_a5_4 = _mm_mul_ps(self.twiddle3re, x419p);
        let temp_a5_5 = _mm_mul_ps(self.twiddle2re, x518p);
        let temp_a5_6 = _mm_mul_ps(self.twiddle7re, x617p);
        let temp_a5_7 = _mm_mul_ps(self.twiddle11re, x716p);
        let temp_a5_8 = _mm_mul_ps(self.twiddle6re, x815p);
        let temp_a5_9 = _mm_mul_ps(self.twiddle1re, x914p);
        let temp_a5_10 = _mm_mul_ps(self.twiddle4re, x1013p);
        let temp_a5_11 = _mm_mul_ps(self.twiddle9re, x1112p);
        let temp_a6_1 = _mm_mul_ps(self.twiddle6re, x122p);
        let temp_a6_2 = _mm_mul_ps(self.twiddle11re, x221p);
        let temp_a6_3 = _mm_mul_ps(self.twiddle5re, x320p);
        let temp_a6_4 = _mm_mul_ps(self.twiddle1re, x419p);
        let temp_a6_5 = _mm_mul_ps(self.twiddle7re, x518p);
        let temp_a6_6 = _mm_mul_ps(self.twiddle10re, x617p);
        let temp_a6_7 = _mm_mul_ps(self.twiddle4re, x716p);
        let temp_a6_8 = _mm_mul_ps(self.twiddle2re, x815p);
        let temp_a6_9 = _mm_mul_ps(self.twiddle8re, x914p);
        let temp_a6_10 = _mm_mul_ps(self.twiddle9re, x1013p);
        let temp_a6_11 = _mm_mul_ps(self.twiddle3re, x1112p);
        let temp_a7_1 = _mm_mul_ps(self.twiddle7re, x122p);
        let temp_a7_2 = _mm_mul_ps(self.twiddle9re, x221p);
        let temp_a7_3 = _mm_mul_ps(self.twiddle2re, x320p);
        let temp_a7_4 = _mm_mul_ps(self.twiddle5re, x419p);
        let temp_a7_5 = _mm_mul_ps(self.twiddle11re, x518p);
        let temp_a7_6 = _mm_mul_ps(self.twiddle4re, x617p);
        let temp_a7_7 = _mm_mul_ps(self.twiddle3re, x716p);
        let temp_a7_8 = _mm_mul_ps(self.twiddle10re, x815p);
        let temp_a7_9 = _mm_mul_ps(self.twiddle6re, x914p);
        let temp_a7_10 = _mm_mul_ps(self.twiddle1re, x1013p);
        let temp_a7_11 = _mm_mul_ps(self.twiddle8re, x1112p);
        let temp_a8_1 = _mm_mul_ps(self.twiddle8re, x122p);
        let temp_a8_2 = _mm_mul_ps(self.twiddle7re, x221p);
        let temp_a8_3 = _mm_mul_ps(self.twiddle1re, x320p);
        let temp_a8_4 = _mm_mul_ps(self.twiddle9re, x419p);
        let temp_a8_5 = _mm_mul_ps(self.twiddle6re, x518p);
        let temp_a8_6 = _mm_mul_ps(self.twiddle2re, x617p);
        let temp_a8_7 = _mm_mul_ps(self.twiddle10re, x716p);
        let temp_a8_8 = _mm_mul_ps(self.twiddle5re, x815p);
        let temp_a8_9 = _mm_mul_ps(self.twiddle3re, x914p);
        let temp_a8_10 = _mm_mul_ps(self.twiddle11re, x1013p);
        let temp_a8_11 = _mm_mul_ps(self.twiddle4re, x1112p);
        let temp_a9_1 = _mm_mul_ps(self.twiddle9re, x122p);
        let temp_a9_2 = _mm_mul_ps(self.twiddle5re, x221p);
        let temp_a9_3 = _mm_mul_ps(self.twiddle4re, x320p);
        let temp_a9_4 = _mm_mul_ps(self.twiddle10re, x419p);
        let temp_a9_5 = _mm_mul_ps(self.twiddle1re, x518p);
        let temp_a9_6 = _mm_mul_ps(self.twiddle8re, x617p);
        let temp_a9_7 = _mm_mul_ps(self.twiddle6re, x716p);
        let temp_a9_8 = _mm_mul_ps(self.twiddle3re, x815p);
        let temp_a9_9 = _mm_mul_ps(self.twiddle11re, x914p);
        let temp_a9_10 = _mm_mul_ps(self.twiddle2re, x1013p);
        let temp_a9_11 = _mm_mul_ps(self.twiddle7re, x1112p);
        let temp_a10_1 = _mm_mul_ps(self.twiddle10re, x122p);
        let temp_a10_2 = _mm_mul_ps(self.twiddle3re, x221p);
        let temp_a10_3 = _mm_mul_ps(self.twiddle7re, x320p);
        let temp_a10_4 = _mm_mul_ps(self.twiddle6re, x419p);
        let temp_a10_5 = _mm_mul_ps(self.twiddle4re, x518p);
        let temp_a10_6 = _mm_mul_ps(self.twiddle9re, x617p);
        let temp_a10_7 = _mm_mul_ps(self.twiddle1re, x716p);
        let temp_a10_8 = _mm_mul_ps(self.twiddle11re, x815p);
        let temp_a10_9 = _mm_mul_ps(self.twiddle2re, x914p);
        let temp_a10_10 = _mm_mul_ps(self.twiddle8re, x1013p);
        let temp_a10_11 = _mm_mul_ps(self.twiddle5re, x1112p);
        let temp_a11_1 = _mm_mul_ps(self.twiddle11re, x122p);
        let temp_a11_2 = _mm_mul_ps(self.twiddle1re, x221p);
        let temp_a11_3 = _mm_mul_ps(self.twiddle10re, x320p);
        let temp_a11_4 = _mm_mul_ps(self.twiddle2re, x419p);
        let temp_a11_5 = _mm_mul_ps(self.twiddle9re, x518p);
        let temp_a11_6 = _mm_mul_ps(self.twiddle3re, x617p);
        let temp_a11_7 = _mm_mul_ps(self.twiddle8re, x716p);
        let temp_a11_8 = _mm_mul_ps(self.twiddle4re, x815p);
        let temp_a11_9 = _mm_mul_ps(self.twiddle7re, x914p);
        let temp_a11_10 = _mm_mul_ps(self.twiddle5re, x1013p);
        let temp_a11_11 = _mm_mul_ps(self.twiddle6re, x1112p);

        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x122n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x221n);
        let temp_b1_3 = _mm_mul_ps(self.twiddle3im, x320n);
        let temp_b1_4 = _mm_mul_ps(self.twiddle4im, x419n);
        let temp_b1_5 = _mm_mul_ps(self.twiddle5im, x518n);
        let temp_b1_6 = _mm_mul_ps(self.twiddle6im, x617n);
        let temp_b1_7 = _mm_mul_ps(self.twiddle7im, x716n);
        let temp_b1_8 = _mm_mul_ps(self.twiddle8im, x815n);
        let temp_b1_9 = _mm_mul_ps(self.twiddle9im, x914n);
        let temp_b1_10 = _mm_mul_ps(self.twiddle10im, x1013n);
        let temp_b1_11 = _mm_mul_ps(self.twiddle11im, x1112n);
        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x122n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle4im, x221n);
        let temp_b2_3 = _mm_mul_ps(self.twiddle6im, x320n);
        let temp_b2_4 = _mm_mul_ps(self.twiddle8im, x419n);
        let temp_b2_5 = _mm_mul_ps(self.twiddle10im, x518n);
        let temp_b2_6 = _mm_mul_ps(self.twiddle11im, x617n);
        let temp_b2_7 = _mm_mul_ps(self.twiddle9im, x716n);
        let temp_b2_8 = _mm_mul_ps(self.twiddle7im, x815n);
        let temp_b2_9 = _mm_mul_ps(self.twiddle5im, x914n);
        let temp_b2_10 = _mm_mul_ps(self.twiddle3im, x1013n);
        let temp_b2_11 = _mm_mul_ps(self.twiddle1im, x1112n);
        let temp_b3_1 = _mm_mul_ps(self.twiddle3im, x122n);
        let temp_b3_2 = _mm_mul_ps(self.twiddle6im, x221n);
        let temp_b3_3 = _mm_mul_ps(self.twiddle9im, x320n);
        let temp_b3_4 = _mm_mul_ps(self.twiddle11im, x419n);
        let temp_b3_5 = _mm_mul_ps(self.twiddle8im, x518n);
        let temp_b3_6 = _mm_mul_ps(self.twiddle5im, x617n);
        let temp_b3_7 = _mm_mul_ps(self.twiddle2im, x716n);
        let temp_b3_8 = _mm_mul_ps(self.twiddle1im, x815n);
        let temp_b3_9 = _mm_mul_ps(self.twiddle4im, x914n);
        let temp_b3_10 = _mm_mul_ps(self.twiddle7im, x1013n);
        let temp_b3_11 = _mm_mul_ps(self.twiddle10im, x1112n);
        let temp_b4_1 = _mm_mul_ps(self.twiddle4im, x122n);
        let temp_b4_2 = _mm_mul_ps(self.twiddle8im, x221n);
        let temp_b4_3 = _mm_mul_ps(self.twiddle11im, x320n);
        let temp_b4_4 = _mm_mul_ps(self.twiddle7im, x419n);
        let temp_b4_5 = _mm_mul_ps(self.twiddle3im, x518n);
        let temp_b4_6 = _mm_mul_ps(self.twiddle1im, x617n);
        let temp_b4_7 = _mm_mul_ps(self.twiddle5im, x716n);
        let temp_b4_8 = _mm_mul_ps(self.twiddle9im, x815n);
        let temp_b4_9 = _mm_mul_ps(self.twiddle10im, x914n);
        let temp_b4_10 = _mm_mul_ps(self.twiddle6im, x1013n);
        let temp_b4_11 = _mm_mul_ps(self.twiddle2im, x1112n);
        let temp_b5_1 = _mm_mul_ps(self.twiddle5im, x122n);
        let temp_b5_2 = _mm_mul_ps(self.twiddle10im, x221n);
        let temp_b5_3 = _mm_mul_ps(self.twiddle8im, x320n);
        let temp_b5_4 = _mm_mul_ps(self.twiddle3im, x419n);
        let temp_b5_5 = _mm_mul_ps(self.twiddle2im, x518n);
        let temp_b5_6 = _mm_mul_ps(self.twiddle7im, x617n);
        let temp_b5_7 = _mm_mul_ps(self.twiddle11im, x716n);
        let temp_b5_8 = _mm_mul_ps(self.twiddle6im, x815n);
        let temp_b5_9 = _mm_mul_ps(self.twiddle1im, x914n);
        let temp_b5_10 = _mm_mul_ps(self.twiddle4im, x1013n);
        let temp_b5_11 = _mm_mul_ps(self.twiddle9im, x1112n);
        let temp_b6_1 = _mm_mul_ps(self.twiddle6im, x122n);
        let temp_b6_2 = _mm_mul_ps(self.twiddle11im, x221n);
        let temp_b6_3 = _mm_mul_ps(self.twiddle5im, x320n);
        let temp_b6_4 = _mm_mul_ps(self.twiddle1im, x419n);
        let temp_b6_5 = _mm_mul_ps(self.twiddle7im, x518n);
        let temp_b6_6 = _mm_mul_ps(self.twiddle10im, x617n);
        let temp_b6_7 = _mm_mul_ps(self.twiddle4im, x716n);
        let temp_b6_8 = _mm_mul_ps(self.twiddle2im, x815n);
        let temp_b6_9 = _mm_mul_ps(self.twiddle8im, x914n);
        let temp_b6_10 = _mm_mul_ps(self.twiddle9im, x1013n);
        let temp_b6_11 = _mm_mul_ps(self.twiddle3im, x1112n);
        let temp_b7_1 = _mm_mul_ps(self.twiddle7im, x122n);
        let temp_b7_2 = _mm_mul_ps(self.twiddle9im, x221n);
        let temp_b7_3 = _mm_mul_ps(self.twiddle2im, x320n);
        let temp_b7_4 = _mm_mul_ps(self.twiddle5im, x419n);
        let temp_b7_5 = _mm_mul_ps(self.twiddle11im, x518n);
        let temp_b7_6 = _mm_mul_ps(self.twiddle4im, x617n);
        let temp_b7_7 = _mm_mul_ps(self.twiddle3im, x716n);
        let temp_b7_8 = _mm_mul_ps(self.twiddle10im, x815n);
        let temp_b7_9 = _mm_mul_ps(self.twiddle6im, x914n);
        let temp_b7_10 = _mm_mul_ps(self.twiddle1im, x1013n);
        let temp_b7_11 = _mm_mul_ps(self.twiddle8im, x1112n);
        let temp_b8_1 = _mm_mul_ps(self.twiddle8im, x122n);
        let temp_b8_2 = _mm_mul_ps(self.twiddle7im, x221n);
        let temp_b8_3 = _mm_mul_ps(self.twiddle1im, x320n);
        let temp_b8_4 = _mm_mul_ps(self.twiddle9im, x419n);
        let temp_b8_5 = _mm_mul_ps(self.twiddle6im, x518n);
        let temp_b8_6 = _mm_mul_ps(self.twiddle2im, x617n);
        let temp_b8_7 = _mm_mul_ps(self.twiddle10im, x716n);
        let temp_b8_8 = _mm_mul_ps(self.twiddle5im, x815n);
        let temp_b8_9 = _mm_mul_ps(self.twiddle3im, x914n);
        let temp_b8_10 = _mm_mul_ps(self.twiddle11im, x1013n);
        let temp_b8_11 = _mm_mul_ps(self.twiddle4im, x1112n);
        let temp_b9_1 = _mm_mul_ps(self.twiddle9im, x122n);
        let temp_b9_2 = _mm_mul_ps(self.twiddle5im, x221n);
        let temp_b9_3 = _mm_mul_ps(self.twiddle4im, x320n);
        let temp_b9_4 = _mm_mul_ps(self.twiddle10im, x419n);
        let temp_b9_5 = _mm_mul_ps(self.twiddle1im, x518n);
        let temp_b9_6 = _mm_mul_ps(self.twiddle8im, x617n);
        let temp_b9_7 = _mm_mul_ps(self.twiddle6im, x716n);
        let temp_b9_8 = _mm_mul_ps(self.twiddle3im, x815n);
        let temp_b9_9 = _mm_mul_ps(self.twiddle11im, x914n);
        let temp_b9_10 = _mm_mul_ps(self.twiddle2im, x1013n);
        let temp_b9_11 = _mm_mul_ps(self.twiddle7im, x1112n);
        let temp_b10_1 = _mm_mul_ps(self.twiddle10im, x122n);
        let temp_b10_2 = _mm_mul_ps(self.twiddle3im, x221n);
        let temp_b10_3 = _mm_mul_ps(self.twiddle7im, x320n);
        let temp_b10_4 = _mm_mul_ps(self.twiddle6im, x419n);
        let temp_b10_5 = _mm_mul_ps(self.twiddle4im, x518n);
        let temp_b10_6 = _mm_mul_ps(self.twiddle9im, x617n);
        let temp_b10_7 = _mm_mul_ps(self.twiddle1im, x716n);
        let temp_b10_8 = _mm_mul_ps(self.twiddle11im, x815n);
        let temp_b10_9 = _mm_mul_ps(self.twiddle2im, x914n);
        let temp_b10_10 = _mm_mul_ps(self.twiddle8im, x1013n);
        let temp_b10_11 = _mm_mul_ps(self.twiddle5im, x1112n);
        let temp_b11_1 = _mm_mul_ps(self.twiddle11im, x122n);
        let temp_b11_2 = _mm_mul_ps(self.twiddle1im, x221n);
        let temp_b11_3 = _mm_mul_ps(self.twiddle10im, x320n);
        let temp_b11_4 = _mm_mul_ps(self.twiddle2im, x419n);
        let temp_b11_5 = _mm_mul_ps(self.twiddle9im, x518n);
        let temp_b11_6 = _mm_mul_ps(self.twiddle3im, x617n);
        let temp_b11_7 = _mm_mul_ps(self.twiddle8im, x716n);
        let temp_b11_8 = _mm_mul_ps(self.twiddle4im, x815n);
        let temp_b11_9 = _mm_mul_ps(self.twiddle7im, x914n);
        let temp_b11_10 = _mm_mul_ps(self.twiddle5im, x1013n);
        let temp_b11_11 = _mm_mul_ps(self.twiddle6im, x1112n);

        let temp_a1 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a1_1,
                _mm_add_ps(
                    temp_a1_2,
                    _mm_add_ps(
                        temp_a1_3,
                        _mm_add_ps(
                            temp_a1_4,
                            _mm_add_ps(
                                temp_a1_5,
                                _mm_add_ps(
                                    temp_a1_6,
                                    _mm_add_ps(
                                        temp_a1_7,
                                        _mm_add_ps(
                                            temp_a1_8,
                                            _mm_add_ps(
                                                temp_a1_9,
                                                _mm_add_ps(temp_a1_10, temp_a1_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a2_1,
                _mm_add_ps(
                    temp_a2_2,
                    _mm_add_ps(
                        temp_a2_3,
                        _mm_add_ps(
                            temp_a2_4,
                            _mm_add_ps(
                                temp_a2_5,
                                _mm_add_ps(
                                    temp_a2_6,
                                    _mm_add_ps(
                                        temp_a2_7,
                                        _mm_add_ps(
                                            temp_a2_8,
                                            _mm_add_ps(
                                                temp_a2_9,
                                                _mm_add_ps(temp_a2_10, temp_a2_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a3_1,
                _mm_add_ps(
                    temp_a3_2,
                    _mm_add_ps(
                        temp_a3_3,
                        _mm_add_ps(
                            temp_a3_4,
                            _mm_add_ps(
                                temp_a3_5,
                                _mm_add_ps(
                                    temp_a3_6,
                                    _mm_add_ps(
                                        temp_a3_7,
                                        _mm_add_ps(
                                            temp_a3_8,
                                            _mm_add_ps(
                                                temp_a3_9,
                                                _mm_add_ps(temp_a3_10, temp_a3_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a4_1,
                _mm_add_ps(
                    temp_a4_2,
                    _mm_add_ps(
                        temp_a4_3,
                        _mm_add_ps(
                            temp_a4_4,
                            _mm_add_ps(
                                temp_a4_5,
                                _mm_add_ps(
                                    temp_a4_6,
                                    _mm_add_ps(
                                        temp_a4_7,
                                        _mm_add_ps(
                                            temp_a4_8,
                                            _mm_add_ps(
                                                temp_a4_9,
                                                _mm_add_ps(temp_a4_10, temp_a4_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a5_1,
                _mm_add_ps(
                    temp_a5_2,
                    _mm_add_ps(
                        temp_a5_3,
                        _mm_add_ps(
                            temp_a5_4,
                            _mm_add_ps(
                                temp_a5_5,
                                _mm_add_ps(
                                    temp_a5_6,
                                    _mm_add_ps(
                                        temp_a5_7,
                                        _mm_add_ps(
                                            temp_a5_8,
                                            _mm_add_ps(
                                                temp_a5_9,
                                                _mm_add_ps(temp_a5_10, temp_a5_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a6_1,
                _mm_add_ps(
                    temp_a6_2,
                    _mm_add_ps(
                        temp_a6_3,
                        _mm_add_ps(
                            temp_a6_4,
                            _mm_add_ps(
                                temp_a6_5,
                                _mm_add_ps(
                                    temp_a6_6,
                                    _mm_add_ps(
                                        temp_a6_7,
                                        _mm_add_ps(
                                            temp_a6_8,
                                            _mm_add_ps(
                                                temp_a6_9,
                                                _mm_add_ps(temp_a6_10, temp_a6_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a7_1,
                _mm_add_ps(
                    temp_a7_2,
                    _mm_add_ps(
                        temp_a7_3,
                        _mm_add_ps(
                            temp_a7_4,
                            _mm_add_ps(
                                temp_a7_5,
                                _mm_add_ps(
                                    temp_a7_6,
                                    _mm_add_ps(
                                        temp_a7_7,
                                        _mm_add_ps(
                                            temp_a7_8,
                                            _mm_add_ps(
                                                temp_a7_9,
                                                _mm_add_ps(temp_a7_10, temp_a7_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a8_1,
                _mm_add_ps(
                    temp_a8_2,
                    _mm_add_ps(
                        temp_a8_3,
                        _mm_add_ps(
                            temp_a8_4,
                            _mm_add_ps(
                                temp_a8_5,
                                _mm_add_ps(
                                    temp_a8_6,
                                    _mm_add_ps(
                                        temp_a8_7,
                                        _mm_add_ps(
                                            temp_a8_8,
                                            _mm_add_ps(
                                                temp_a8_9,
                                                _mm_add_ps(temp_a8_10, temp_a8_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a9 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a9_1,
                _mm_add_ps(
                    temp_a9_2,
                    _mm_add_ps(
                        temp_a9_3,
                        _mm_add_ps(
                            temp_a9_4,
                            _mm_add_ps(
                                temp_a9_5,
                                _mm_add_ps(
                                    temp_a9_6,
                                    _mm_add_ps(
                                        temp_a9_7,
                                        _mm_add_ps(
                                            temp_a9_8,
                                            _mm_add_ps(
                                                temp_a9_9,
                                                _mm_add_ps(temp_a9_10, temp_a9_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a10 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a10_1,
                _mm_add_ps(
                    temp_a10_2,
                    _mm_add_ps(
                        temp_a10_3,
                        _mm_add_ps(
                            temp_a10_4,
                            _mm_add_ps(
                                temp_a10_5,
                                _mm_add_ps(
                                    temp_a10_6,
                                    _mm_add_ps(
                                        temp_a10_7,
                                        _mm_add_ps(
                                            temp_a10_8,
                                            _mm_add_ps(
                                                temp_a10_9,
                                                _mm_add_ps(temp_a10_10, temp_a10_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a11 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a11_1,
                _mm_add_ps(
                    temp_a11_2,
                    _mm_add_ps(
                        temp_a11_3,
                        _mm_add_ps(
                            temp_a11_4,
                            _mm_add_ps(
                                temp_a11_5,
                                _mm_add_ps(
                                    temp_a11_6,
                                    _mm_add_ps(
                                        temp_a11_7,
                                        _mm_add_ps(
                                            temp_a11_8,
                                            _mm_add_ps(
                                                temp_a11_9,
                                                _mm_add_ps(temp_a11_10, temp_a11_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_ps(
            temp_b1_1,
            _mm_add_ps(
                temp_b1_2,
                _mm_add_ps(
                    temp_b1_3,
                    _mm_add_ps(
                        temp_b1_4,
                        _mm_add_ps(
                            temp_b1_5,
                            _mm_add_ps(
                                temp_b1_6,
                                _mm_add_ps(
                                    temp_b1_7,
                                    _mm_add_ps(
                                        temp_b1_8,
                                        _mm_add_ps(temp_b1_9, _mm_add_ps(temp_b1_10, temp_b1_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_ps(
            temp_b2_1,
            _mm_add_ps(
                temp_b2_2,
                _mm_add_ps(
                    temp_b2_3,
                    _mm_add_ps(
                        temp_b2_4,
                        _mm_sub_ps(
                            temp_b2_5,
                            _mm_add_ps(
                                temp_b2_6,
                                _mm_add_ps(
                                    temp_b2_7,
                                    _mm_add_ps(
                                        temp_b2_8,
                                        _mm_add_ps(temp_b2_9, _mm_add_ps(temp_b2_10, temp_b2_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_ps(
            temp_b3_1,
            _mm_add_ps(
                temp_b3_2,
                _mm_sub_ps(
                    temp_b3_3,
                    _mm_add_ps(
                        temp_b3_4,
                        _mm_add_ps(
                            temp_b3_5,
                            _mm_add_ps(
                                temp_b3_6,
                                _mm_sub_ps(
                                    temp_b3_7,
                                    _mm_add_ps(
                                        temp_b3_8,
                                        _mm_add_ps(temp_b3_9, _mm_add_ps(temp_b3_10, temp_b3_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_ps(
            temp_b4_1,
            _mm_sub_ps(
                temp_b4_2,
                _mm_add_ps(
                    temp_b4_3,
                    _mm_add_ps(
                        temp_b4_4,
                        _mm_sub_ps(
                            temp_b4_5,
                            _mm_add_ps(
                                temp_b4_6,
                                _mm_add_ps(
                                    temp_b4_7,
                                    _mm_sub_ps(
                                        temp_b4_8,
                                        _mm_add_ps(temp_b4_9, _mm_add_ps(temp_b4_10, temp_b4_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_add_ps(
            temp_b5_1,
            _mm_sub_ps(
                temp_b5_2,
                _mm_add_ps(
                    temp_b5_3,
                    _mm_sub_ps(
                        temp_b5_4,
                        _mm_add_ps(
                            temp_b5_5,
                            _mm_sub_ps(
                                temp_b5_6,
                                _mm_add_ps(
                                    temp_b5_7,
                                    _mm_add_ps(
                                        temp_b5_8,
                                        _mm_sub_ps(temp_b5_9, _mm_add_ps(temp_b5_10, temp_b5_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_sub_ps(
            temp_b6_1,
            _mm_add_ps(
                temp_b6_2,
                _mm_sub_ps(
                    temp_b6_3,
                    _mm_add_ps(
                        temp_b6_4,
                        _mm_sub_ps(
                            temp_b6_5,
                            _mm_add_ps(
                                temp_b6_6,
                                _mm_sub_ps(
                                    temp_b6_7,
                                    _mm_add_ps(
                                        temp_b6_8,
                                        _mm_sub_ps(temp_b6_9, _mm_add_ps(temp_b6_10, temp_b6_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_sub_ps(
            temp_b7_1,
            _mm_add_ps(
                temp_b7_2,
                _mm_sub_ps(
                    temp_b7_3,
                    _mm_sub_ps(
                        temp_b7_4,
                        _mm_add_ps(
                            temp_b7_5,
                            _mm_sub_ps(
                                temp_b7_6,
                                _mm_add_ps(
                                    temp_b7_7,
                                    _mm_sub_ps(
                                        temp_b7_8,
                                        _mm_sub_ps(temp_b7_9, _mm_add_ps(temp_b7_10, temp_b7_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_ps(
            temp_b8_1,
            _mm_sub_ps(
                temp_b8_2,
                _mm_add_ps(
                    temp_b8_3,
                    _mm_sub_ps(
                        temp_b8_4,
                        _mm_sub_ps(
                            temp_b8_5,
                            _mm_add_ps(
                                temp_b8_6,
                                _mm_sub_ps(
                                    temp_b8_7,
                                    _mm_sub_ps(
                                        temp_b8_8,
                                        _mm_add_ps(temp_b8_9, _mm_sub_ps(temp_b8_10, temp_b8_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b9 = _mm_sub_ps(
            temp_b9_1,
            _mm_sub_ps(
                temp_b9_2,
                _mm_sub_ps(
                    temp_b9_3,
                    _mm_add_ps(
                        temp_b9_4,
                        _mm_sub_ps(
                            temp_b9_5,
                            _mm_sub_ps(
                                temp_b9_6,
                                _mm_sub_ps(
                                    temp_b9_7,
                                    _mm_sub_ps(
                                        temp_b9_8,
                                        _mm_add_ps(temp_b9_9, _mm_sub_ps(temp_b9_10, temp_b9_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b10 = _mm_sub_ps(
            temp_b10_1,
            _mm_sub_ps(
                temp_b10_2,
                _mm_sub_ps(
                    temp_b10_3,
                    _mm_sub_ps(
                        temp_b10_4,
                        _mm_sub_ps(
                            temp_b10_5,
                            _mm_sub_ps(
                                temp_b10_6,
                                _mm_add_ps(
                                    temp_b10_7,
                                    _mm_sub_ps(
                                        temp_b10_8,
                                        _mm_sub_ps(
                                            temp_b10_9,
                                            _mm_sub_ps(temp_b10_10, temp_b10_11),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b11 = _mm_sub_ps(
            temp_b11_1,
            _mm_sub_ps(
                temp_b11_2,
                _mm_sub_ps(
                    temp_b11_3,
                    _mm_sub_ps(
                        temp_b11_4,
                        _mm_sub_ps(
                            temp_b11_5,
                            _mm_sub_ps(
                                temp_b11_6,
                                _mm_sub_ps(
                                    temp_b11_7,
                                    _mm_sub_ps(
                                        temp_b11_8,
                                        _mm_sub_ps(
                                            temp_b11_9,
                                            _mm_sub_ps(temp_b11_10, temp_b11_11),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate_both(temp_b1);
        let temp_b2_rot = self.rotate.rotate_both(temp_b2);
        let temp_b3_rot = self.rotate.rotate_both(temp_b3);
        let temp_b4_rot = self.rotate.rotate_both(temp_b4);
        let temp_b5_rot = self.rotate.rotate_both(temp_b5);
        let temp_b6_rot = self.rotate.rotate_both(temp_b6);
        let temp_b7_rot = self.rotate.rotate_both(temp_b7);
        let temp_b8_rot = self.rotate.rotate_both(temp_b8);
        let temp_b9_rot = self.rotate.rotate_both(temp_b9);
        let temp_b10_rot = self.rotate.rotate_both(temp_b10);
        let temp_b11_rot = self.rotate.rotate_both(temp_b11);

        let x0 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                x122p,
                _mm_add_ps(
                    x221p,
                    _mm_add_ps(
                        x320p,
                        _mm_add_ps(
                            x419p,
                            _mm_add_ps(
                                x518p,
                                _mm_add_ps(
                                    x617p,
                                    _mm_add_ps(
                                        x716p,
                                        _mm_add_ps(
                                            x815p,
                                            _mm_add_ps(x914p, _mm_add_ps(x1013p, x1112p)),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_ps(temp_a1, temp_b1_rot);
        let x2 = _mm_add_ps(temp_a2, temp_b2_rot);
        let x3 = _mm_add_ps(temp_a3, temp_b3_rot);
        let x4 = _mm_add_ps(temp_a4, temp_b4_rot);
        let x5 = _mm_add_ps(temp_a5, temp_b5_rot);
        let x6 = _mm_add_ps(temp_a6, temp_b6_rot);
        let x7 = _mm_add_ps(temp_a7, temp_b7_rot);
        let x8 = _mm_add_ps(temp_a8, temp_b8_rot);
        let x9 = _mm_add_ps(temp_a9, temp_b9_rot);
        let x10 = _mm_add_ps(temp_a10, temp_b10_rot);
        let x11 = _mm_add_ps(temp_a11, temp_b11_rot);
        let x12 = _mm_sub_ps(temp_a11, temp_b11_rot);
        let x13 = _mm_sub_ps(temp_a10, temp_b10_rot);
        let x14 = _mm_sub_ps(temp_a9, temp_b9_rot);
        let x15 = _mm_sub_ps(temp_a8, temp_b8_rot);
        let x16 = _mm_sub_ps(temp_a7, temp_b7_rot);
        let x17 = _mm_sub_ps(temp_a6, temp_b6_rot);
        let x18 = _mm_sub_ps(temp_a5, temp_b5_rot);
        let x19 = _mm_sub_ps(temp_a4, temp_b4_rot);
        let x20 = _mm_sub_ps(temp_a3, temp_b3_rot);
        let x21 = _mm_sub_ps(temp_a2, temp_b2_rot);
        let x22 = _mm_sub_ps(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
            x19, x20, x21, x22,
        ]
    }
}

//   ____  _____            __   _  _   _     _ _
//  |___ \|___ /           / /_ | || | | |__ (_) |_
//    __) | |_ \   _____  | '_ \| || |_| '_ \| | __|
//   / __/ ___) | |_____| | (_) |__   _| |_) | | |_
//  |_____|____/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly23<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
    twiddle3re: __m128d,
    twiddle3im: __m128d,
    twiddle4re: __m128d,
    twiddle4im: __m128d,
    twiddle5re: __m128d,
    twiddle5im: __m128d,
    twiddle6re: __m128d,
    twiddle6im: __m128d,
    twiddle7re: __m128d,
    twiddle7im: __m128d,
    twiddle8re: __m128d,
    twiddle8im: __m128d,
    twiddle9re: __m128d,
    twiddle9im: __m128d,
    twiddle10re: __m128d,
    twiddle10im: __m128d,
    twiddle11re: __m128d,
    twiddle11im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly23, 23, |this: &SseF64Butterfly23<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly23, 23, |this: &SseF64Butterfly23<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly23<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 23, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 23, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 23, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 23, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 23, direction);
        let tw6: Complex<f64> = twiddles::compute_twiddle(6, 23, direction);
        let tw7: Complex<f64> = twiddles::compute_twiddle(7, 23, direction);
        let tw8: Complex<f64> = twiddles::compute_twiddle(8, 23, direction);
        let tw9: Complex<f64> = twiddles::compute_twiddle(9, 23, direction);
        let tw10: Complex<f64> = twiddles::compute_twiddle(10, 23, direction);
        let tw11: Complex<f64> = twiddles::compute_twiddle(11, 23, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_pd(tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_pd(tw3.im, tw3.im) };
        let twiddle4re = unsafe { _mm_set_pd(tw4.re, tw4.re) };
        let twiddle4im = unsafe { _mm_set_pd(tw4.im, tw4.im) };
        let twiddle5re = unsafe { _mm_set_pd(tw5.re, tw5.re) };
        let twiddle5im = unsafe { _mm_set_pd(tw5.im, tw5.im) };
        let twiddle6re = unsafe { _mm_set_pd(tw6.re, tw6.re) };
        let twiddle6im = unsafe { _mm_set_pd(tw6.im, tw6.im) };
        let twiddle7re = unsafe { _mm_set_pd(tw7.re, tw7.re) };
        let twiddle7im = unsafe { _mm_set_pd(tw7.im, tw7.im) };
        let twiddle8re = unsafe { _mm_set_pd(tw8.re, tw8.re) };
        let twiddle8im = unsafe { _mm_set_pd(tw8.im, tw8.im) };
        let twiddle9re = unsafe { _mm_set_pd(tw9.re, tw9.re) };
        let twiddle9im = unsafe { _mm_set_pd(tw9.im, tw9.im) };
        let twiddle10re = unsafe { _mm_set_pd(tw10.re, tw10.re) };
        let twiddle10im = unsafe { _mm_set_pd(tw10.im, tw10.im) };
        let twiddle11re = unsafe { _mm_set_pd(tw11.re, tw11.re) };
        let twiddle11im = unsafe { _mm_set_pd(tw11.im, tw11.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
            twiddle9re,
            twiddle9im,
            twiddle10re,
            twiddle10im,
            twiddle11re,
            twiddle11im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let v1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let v2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let v3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let v4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let v5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let v6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let v7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let v8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let v9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let v10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);
        let v11 = _mm_loadu_pd(input.as_ptr().add(11) as *const f64);
        let v12 = _mm_loadu_pd(input.as_ptr().add(12) as *const f64);
        let v13 = _mm_loadu_pd(input.as_ptr().add(13) as *const f64);
        let v14 = _mm_loadu_pd(input.as_ptr().add(14) as *const f64);
        let v15 = _mm_loadu_pd(input.as_ptr().add(15) as *const f64);
        let v16 = _mm_loadu_pd(input.as_ptr().add(16) as *const f64);
        let v17 = _mm_loadu_pd(input.as_ptr().add(17) as *const f64);
        let v18 = _mm_loadu_pd(input.as_ptr().add(18) as *const f64);
        let v19 = _mm_loadu_pd(input.as_ptr().add(19) as *const f64);
        let v20 = _mm_loadu_pd(input.as_ptr().add(20) as *const f64);
        let v21 = _mm_loadu_pd(input.as_ptr().add(21) as *const f64);
        let v22 = _mm_loadu_pd(input.as_ptr().add(22) as *const f64);

        let out = self.perform_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22,
        ]);

        let val = std::mem::transmute::<[__m128d; 23], [Complex<f64>; 23]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
        *output_slice.add(6) = val[6];
        *output_slice.add(7) = val[7];
        *output_slice.add(8) = val[8];
        *output_slice.add(9) = val[9];
        *output_slice.add(10) = val[10];
        *output_slice.add(11) = val[11];
        *output_slice.add(12) = val[12];
        *output_slice.add(13) = val[13];
        *output_slice.add(14) = val[14];
        *output_slice.add(15) = val[15];
        *output_slice.add(16) = val[16];
        *output_slice.add(17) = val[17];
        *output_slice.add(18) = val[18];
        *output_slice.add(19) = val[19];
        *output_slice.add(20) = val[20];
        *output_slice.add(21) = val[21];
        *output_slice.add(22) = val[22];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 23]) -> [__m128d; 23] {
        // This is a SSE translation of the scalar 17-point butterfly
        let x122p = _mm_add_pd(values[1], values[22]);
        let x122n = _mm_sub_pd(values[1], values[22]);
        let x221p = _mm_add_pd(values[2], values[21]);
        let x221n = _mm_sub_pd(values[2], values[21]);
        let x320p = _mm_add_pd(values[3], values[20]);
        let x320n = _mm_sub_pd(values[3], values[20]);
        let x419p = _mm_add_pd(values[4], values[19]);
        let x419n = _mm_sub_pd(values[4], values[19]);
        let x518p = _mm_add_pd(values[5], values[18]);
        let x518n = _mm_sub_pd(values[5], values[18]);
        let x617p = _mm_add_pd(values[6], values[17]);
        let x617n = _mm_sub_pd(values[6], values[17]);
        let x716p = _mm_add_pd(values[7], values[16]);
        let x716n = _mm_sub_pd(values[7], values[16]);
        let x815p = _mm_add_pd(values[8], values[15]);
        let x815n = _mm_sub_pd(values[8], values[15]);
        let x914p = _mm_add_pd(values[9], values[14]);
        let x914n = _mm_sub_pd(values[9], values[14]);
        let x1013p = _mm_add_pd(values[10], values[13]);
        let x1013n = _mm_sub_pd(values[10], values[13]);
        let x1112p = _mm_add_pd(values[11], values[12]);
        let x1112n = _mm_sub_pd(values[11], values[12]);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x122p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x221p);
        let temp_a1_3 = _mm_mul_pd(self.twiddle3re, x320p);
        let temp_a1_4 = _mm_mul_pd(self.twiddle4re, x419p);
        let temp_a1_5 = _mm_mul_pd(self.twiddle5re, x518p);
        let temp_a1_6 = _mm_mul_pd(self.twiddle6re, x617p);
        let temp_a1_7 = _mm_mul_pd(self.twiddle7re, x716p);
        let temp_a1_8 = _mm_mul_pd(self.twiddle8re, x815p);
        let temp_a1_9 = _mm_mul_pd(self.twiddle9re, x914p);
        let temp_a1_10 = _mm_mul_pd(self.twiddle10re, x1013p);
        let temp_a1_11 = _mm_mul_pd(self.twiddle11re, x1112p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x122p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle4re, x221p);
        let temp_a2_3 = _mm_mul_pd(self.twiddle6re, x320p);
        let temp_a2_4 = _mm_mul_pd(self.twiddle8re, x419p);
        let temp_a2_5 = _mm_mul_pd(self.twiddle10re, x518p);
        let temp_a2_6 = _mm_mul_pd(self.twiddle11re, x617p);
        let temp_a2_7 = _mm_mul_pd(self.twiddle9re, x716p);
        let temp_a2_8 = _mm_mul_pd(self.twiddle7re, x815p);
        let temp_a2_9 = _mm_mul_pd(self.twiddle5re, x914p);
        let temp_a2_10 = _mm_mul_pd(self.twiddle3re, x1013p);
        let temp_a2_11 = _mm_mul_pd(self.twiddle1re, x1112p);
        let temp_a3_1 = _mm_mul_pd(self.twiddle3re, x122p);
        let temp_a3_2 = _mm_mul_pd(self.twiddle6re, x221p);
        let temp_a3_3 = _mm_mul_pd(self.twiddle9re, x320p);
        let temp_a3_4 = _mm_mul_pd(self.twiddle11re, x419p);
        let temp_a3_5 = _mm_mul_pd(self.twiddle8re, x518p);
        let temp_a3_6 = _mm_mul_pd(self.twiddle5re, x617p);
        let temp_a3_7 = _mm_mul_pd(self.twiddle2re, x716p);
        let temp_a3_8 = _mm_mul_pd(self.twiddle1re, x815p);
        let temp_a3_9 = _mm_mul_pd(self.twiddle4re, x914p);
        let temp_a3_10 = _mm_mul_pd(self.twiddle7re, x1013p);
        let temp_a3_11 = _mm_mul_pd(self.twiddle10re, x1112p);
        let temp_a4_1 = _mm_mul_pd(self.twiddle4re, x122p);
        let temp_a4_2 = _mm_mul_pd(self.twiddle8re, x221p);
        let temp_a4_3 = _mm_mul_pd(self.twiddle11re, x320p);
        let temp_a4_4 = _mm_mul_pd(self.twiddle7re, x419p);
        let temp_a4_5 = _mm_mul_pd(self.twiddle3re, x518p);
        let temp_a4_6 = _mm_mul_pd(self.twiddle1re, x617p);
        let temp_a4_7 = _mm_mul_pd(self.twiddle5re, x716p);
        let temp_a4_8 = _mm_mul_pd(self.twiddle9re, x815p);
        let temp_a4_9 = _mm_mul_pd(self.twiddle10re, x914p);
        let temp_a4_10 = _mm_mul_pd(self.twiddle6re, x1013p);
        let temp_a4_11 = _mm_mul_pd(self.twiddle2re, x1112p);
        let temp_a5_1 = _mm_mul_pd(self.twiddle5re, x122p);
        let temp_a5_2 = _mm_mul_pd(self.twiddle10re, x221p);
        let temp_a5_3 = _mm_mul_pd(self.twiddle8re, x320p);
        let temp_a5_4 = _mm_mul_pd(self.twiddle3re, x419p);
        let temp_a5_5 = _mm_mul_pd(self.twiddle2re, x518p);
        let temp_a5_6 = _mm_mul_pd(self.twiddle7re, x617p);
        let temp_a5_7 = _mm_mul_pd(self.twiddle11re, x716p);
        let temp_a5_8 = _mm_mul_pd(self.twiddle6re, x815p);
        let temp_a5_9 = _mm_mul_pd(self.twiddle1re, x914p);
        let temp_a5_10 = _mm_mul_pd(self.twiddle4re, x1013p);
        let temp_a5_11 = _mm_mul_pd(self.twiddle9re, x1112p);
        let temp_a6_1 = _mm_mul_pd(self.twiddle6re, x122p);
        let temp_a6_2 = _mm_mul_pd(self.twiddle11re, x221p);
        let temp_a6_3 = _mm_mul_pd(self.twiddle5re, x320p);
        let temp_a6_4 = _mm_mul_pd(self.twiddle1re, x419p);
        let temp_a6_5 = _mm_mul_pd(self.twiddle7re, x518p);
        let temp_a6_6 = _mm_mul_pd(self.twiddle10re, x617p);
        let temp_a6_7 = _mm_mul_pd(self.twiddle4re, x716p);
        let temp_a6_8 = _mm_mul_pd(self.twiddle2re, x815p);
        let temp_a6_9 = _mm_mul_pd(self.twiddle8re, x914p);
        let temp_a6_10 = _mm_mul_pd(self.twiddle9re, x1013p);
        let temp_a6_11 = _mm_mul_pd(self.twiddle3re, x1112p);
        let temp_a7_1 = _mm_mul_pd(self.twiddle7re, x122p);
        let temp_a7_2 = _mm_mul_pd(self.twiddle9re, x221p);
        let temp_a7_3 = _mm_mul_pd(self.twiddle2re, x320p);
        let temp_a7_4 = _mm_mul_pd(self.twiddle5re, x419p);
        let temp_a7_5 = _mm_mul_pd(self.twiddle11re, x518p);
        let temp_a7_6 = _mm_mul_pd(self.twiddle4re, x617p);
        let temp_a7_7 = _mm_mul_pd(self.twiddle3re, x716p);
        let temp_a7_8 = _mm_mul_pd(self.twiddle10re, x815p);
        let temp_a7_9 = _mm_mul_pd(self.twiddle6re, x914p);
        let temp_a7_10 = _mm_mul_pd(self.twiddle1re, x1013p);
        let temp_a7_11 = _mm_mul_pd(self.twiddle8re, x1112p);
        let temp_a8_1 = _mm_mul_pd(self.twiddle8re, x122p);
        let temp_a8_2 = _mm_mul_pd(self.twiddle7re, x221p);
        let temp_a8_3 = _mm_mul_pd(self.twiddle1re, x320p);
        let temp_a8_4 = _mm_mul_pd(self.twiddle9re, x419p);
        let temp_a8_5 = _mm_mul_pd(self.twiddle6re, x518p);
        let temp_a8_6 = _mm_mul_pd(self.twiddle2re, x617p);
        let temp_a8_7 = _mm_mul_pd(self.twiddle10re, x716p);
        let temp_a8_8 = _mm_mul_pd(self.twiddle5re, x815p);
        let temp_a8_9 = _mm_mul_pd(self.twiddle3re, x914p);
        let temp_a8_10 = _mm_mul_pd(self.twiddle11re, x1013p);
        let temp_a8_11 = _mm_mul_pd(self.twiddle4re, x1112p);
        let temp_a9_1 = _mm_mul_pd(self.twiddle9re, x122p);
        let temp_a9_2 = _mm_mul_pd(self.twiddle5re, x221p);
        let temp_a9_3 = _mm_mul_pd(self.twiddle4re, x320p);
        let temp_a9_4 = _mm_mul_pd(self.twiddle10re, x419p);
        let temp_a9_5 = _mm_mul_pd(self.twiddle1re, x518p);
        let temp_a9_6 = _mm_mul_pd(self.twiddle8re, x617p);
        let temp_a9_7 = _mm_mul_pd(self.twiddle6re, x716p);
        let temp_a9_8 = _mm_mul_pd(self.twiddle3re, x815p);
        let temp_a9_9 = _mm_mul_pd(self.twiddle11re, x914p);
        let temp_a9_10 = _mm_mul_pd(self.twiddle2re, x1013p);
        let temp_a9_11 = _mm_mul_pd(self.twiddle7re, x1112p);
        let temp_a10_1 = _mm_mul_pd(self.twiddle10re, x122p);
        let temp_a10_2 = _mm_mul_pd(self.twiddle3re, x221p);
        let temp_a10_3 = _mm_mul_pd(self.twiddle7re, x320p);
        let temp_a10_4 = _mm_mul_pd(self.twiddle6re, x419p);
        let temp_a10_5 = _mm_mul_pd(self.twiddle4re, x518p);
        let temp_a10_6 = _mm_mul_pd(self.twiddle9re, x617p);
        let temp_a10_7 = _mm_mul_pd(self.twiddle1re, x716p);
        let temp_a10_8 = _mm_mul_pd(self.twiddle11re, x815p);
        let temp_a10_9 = _mm_mul_pd(self.twiddle2re, x914p);
        let temp_a10_10 = _mm_mul_pd(self.twiddle8re, x1013p);
        let temp_a10_11 = _mm_mul_pd(self.twiddle5re, x1112p);
        let temp_a11_1 = _mm_mul_pd(self.twiddle11re, x122p);
        let temp_a11_2 = _mm_mul_pd(self.twiddle1re, x221p);
        let temp_a11_3 = _mm_mul_pd(self.twiddle10re, x320p);
        let temp_a11_4 = _mm_mul_pd(self.twiddle2re, x419p);
        let temp_a11_5 = _mm_mul_pd(self.twiddle9re, x518p);
        let temp_a11_6 = _mm_mul_pd(self.twiddle3re, x617p);
        let temp_a11_7 = _mm_mul_pd(self.twiddle8re, x716p);
        let temp_a11_8 = _mm_mul_pd(self.twiddle4re, x815p);
        let temp_a11_9 = _mm_mul_pd(self.twiddle7re, x914p);
        let temp_a11_10 = _mm_mul_pd(self.twiddle5re, x1013p);
        let temp_a11_11 = _mm_mul_pd(self.twiddle6re, x1112p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x122n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x221n);
        let temp_b1_3 = _mm_mul_pd(self.twiddle3im, x320n);
        let temp_b1_4 = _mm_mul_pd(self.twiddle4im, x419n);
        let temp_b1_5 = _mm_mul_pd(self.twiddle5im, x518n);
        let temp_b1_6 = _mm_mul_pd(self.twiddle6im, x617n);
        let temp_b1_7 = _mm_mul_pd(self.twiddle7im, x716n);
        let temp_b1_8 = _mm_mul_pd(self.twiddle8im, x815n);
        let temp_b1_9 = _mm_mul_pd(self.twiddle9im, x914n);
        let temp_b1_10 = _mm_mul_pd(self.twiddle10im, x1013n);
        let temp_b1_11 = _mm_mul_pd(self.twiddle11im, x1112n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x122n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle4im, x221n);
        let temp_b2_3 = _mm_mul_pd(self.twiddle6im, x320n);
        let temp_b2_4 = _mm_mul_pd(self.twiddle8im, x419n);
        let temp_b2_5 = _mm_mul_pd(self.twiddle10im, x518n);
        let temp_b2_6 = _mm_mul_pd(self.twiddle11im, x617n);
        let temp_b2_7 = _mm_mul_pd(self.twiddle9im, x716n);
        let temp_b2_8 = _mm_mul_pd(self.twiddle7im, x815n);
        let temp_b2_9 = _mm_mul_pd(self.twiddle5im, x914n);
        let temp_b2_10 = _mm_mul_pd(self.twiddle3im, x1013n);
        let temp_b2_11 = _mm_mul_pd(self.twiddle1im, x1112n);
        let temp_b3_1 = _mm_mul_pd(self.twiddle3im, x122n);
        let temp_b3_2 = _mm_mul_pd(self.twiddle6im, x221n);
        let temp_b3_3 = _mm_mul_pd(self.twiddle9im, x320n);
        let temp_b3_4 = _mm_mul_pd(self.twiddle11im, x419n);
        let temp_b3_5 = _mm_mul_pd(self.twiddle8im, x518n);
        let temp_b3_6 = _mm_mul_pd(self.twiddle5im, x617n);
        let temp_b3_7 = _mm_mul_pd(self.twiddle2im, x716n);
        let temp_b3_8 = _mm_mul_pd(self.twiddle1im, x815n);
        let temp_b3_9 = _mm_mul_pd(self.twiddle4im, x914n);
        let temp_b3_10 = _mm_mul_pd(self.twiddle7im, x1013n);
        let temp_b3_11 = _mm_mul_pd(self.twiddle10im, x1112n);
        let temp_b4_1 = _mm_mul_pd(self.twiddle4im, x122n);
        let temp_b4_2 = _mm_mul_pd(self.twiddle8im, x221n);
        let temp_b4_3 = _mm_mul_pd(self.twiddle11im, x320n);
        let temp_b4_4 = _mm_mul_pd(self.twiddle7im, x419n);
        let temp_b4_5 = _mm_mul_pd(self.twiddle3im, x518n);
        let temp_b4_6 = _mm_mul_pd(self.twiddle1im, x617n);
        let temp_b4_7 = _mm_mul_pd(self.twiddle5im, x716n);
        let temp_b4_8 = _mm_mul_pd(self.twiddle9im, x815n);
        let temp_b4_9 = _mm_mul_pd(self.twiddle10im, x914n);
        let temp_b4_10 = _mm_mul_pd(self.twiddle6im, x1013n);
        let temp_b4_11 = _mm_mul_pd(self.twiddle2im, x1112n);
        let temp_b5_1 = _mm_mul_pd(self.twiddle5im, x122n);
        let temp_b5_2 = _mm_mul_pd(self.twiddle10im, x221n);
        let temp_b5_3 = _mm_mul_pd(self.twiddle8im, x320n);
        let temp_b5_4 = _mm_mul_pd(self.twiddle3im, x419n);
        let temp_b5_5 = _mm_mul_pd(self.twiddle2im, x518n);
        let temp_b5_6 = _mm_mul_pd(self.twiddle7im, x617n);
        let temp_b5_7 = _mm_mul_pd(self.twiddle11im, x716n);
        let temp_b5_8 = _mm_mul_pd(self.twiddle6im, x815n);
        let temp_b5_9 = _mm_mul_pd(self.twiddle1im, x914n);
        let temp_b5_10 = _mm_mul_pd(self.twiddle4im, x1013n);
        let temp_b5_11 = _mm_mul_pd(self.twiddle9im, x1112n);
        let temp_b6_1 = _mm_mul_pd(self.twiddle6im, x122n);
        let temp_b6_2 = _mm_mul_pd(self.twiddle11im, x221n);
        let temp_b6_3 = _mm_mul_pd(self.twiddle5im, x320n);
        let temp_b6_4 = _mm_mul_pd(self.twiddle1im, x419n);
        let temp_b6_5 = _mm_mul_pd(self.twiddle7im, x518n);
        let temp_b6_6 = _mm_mul_pd(self.twiddle10im, x617n);
        let temp_b6_7 = _mm_mul_pd(self.twiddle4im, x716n);
        let temp_b6_8 = _mm_mul_pd(self.twiddle2im, x815n);
        let temp_b6_9 = _mm_mul_pd(self.twiddle8im, x914n);
        let temp_b6_10 = _mm_mul_pd(self.twiddle9im, x1013n);
        let temp_b6_11 = _mm_mul_pd(self.twiddle3im, x1112n);
        let temp_b7_1 = _mm_mul_pd(self.twiddle7im, x122n);
        let temp_b7_2 = _mm_mul_pd(self.twiddle9im, x221n);
        let temp_b7_3 = _mm_mul_pd(self.twiddle2im, x320n);
        let temp_b7_4 = _mm_mul_pd(self.twiddle5im, x419n);
        let temp_b7_5 = _mm_mul_pd(self.twiddle11im, x518n);
        let temp_b7_6 = _mm_mul_pd(self.twiddle4im, x617n);
        let temp_b7_7 = _mm_mul_pd(self.twiddle3im, x716n);
        let temp_b7_8 = _mm_mul_pd(self.twiddle10im, x815n);
        let temp_b7_9 = _mm_mul_pd(self.twiddle6im, x914n);
        let temp_b7_10 = _mm_mul_pd(self.twiddle1im, x1013n);
        let temp_b7_11 = _mm_mul_pd(self.twiddle8im, x1112n);
        let temp_b8_1 = _mm_mul_pd(self.twiddle8im, x122n);
        let temp_b8_2 = _mm_mul_pd(self.twiddle7im, x221n);
        let temp_b8_3 = _mm_mul_pd(self.twiddle1im, x320n);
        let temp_b8_4 = _mm_mul_pd(self.twiddle9im, x419n);
        let temp_b8_5 = _mm_mul_pd(self.twiddle6im, x518n);
        let temp_b8_6 = _mm_mul_pd(self.twiddle2im, x617n);
        let temp_b8_7 = _mm_mul_pd(self.twiddle10im, x716n);
        let temp_b8_8 = _mm_mul_pd(self.twiddle5im, x815n);
        let temp_b8_9 = _mm_mul_pd(self.twiddle3im, x914n);
        let temp_b8_10 = _mm_mul_pd(self.twiddle11im, x1013n);
        let temp_b8_11 = _mm_mul_pd(self.twiddle4im, x1112n);
        let temp_b9_1 = _mm_mul_pd(self.twiddle9im, x122n);
        let temp_b9_2 = _mm_mul_pd(self.twiddle5im, x221n);
        let temp_b9_3 = _mm_mul_pd(self.twiddle4im, x320n);
        let temp_b9_4 = _mm_mul_pd(self.twiddle10im, x419n);
        let temp_b9_5 = _mm_mul_pd(self.twiddle1im, x518n);
        let temp_b9_6 = _mm_mul_pd(self.twiddle8im, x617n);
        let temp_b9_7 = _mm_mul_pd(self.twiddle6im, x716n);
        let temp_b9_8 = _mm_mul_pd(self.twiddle3im, x815n);
        let temp_b9_9 = _mm_mul_pd(self.twiddle11im, x914n);
        let temp_b9_10 = _mm_mul_pd(self.twiddle2im, x1013n);
        let temp_b9_11 = _mm_mul_pd(self.twiddle7im, x1112n);
        let temp_b10_1 = _mm_mul_pd(self.twiddle10im, x122n);
        let temp_b10_2 = _mm_mul_pd(self.twiddle3im, x221n);
        let temp_b10_3 = _mm_mul_pd(self.twiddle7im, x320n);
        let temp_b10_4 = _mm_mul_pd(self.twiddle6im, x419n);
        let temp_b10_5 = _mm_mul_pd(self.twiddle4im, x518n);
        let temp_b10_6 = _mm_mul_pd(self.twiddle9im, x617n);
        let temp_b10_7 = _mm_mul_pd(self.twiddle1im, x716n);
        let temp_b10_8 = _mm_mul_pd(self.twiddle11im, x815n);
        let temp_b10_9 = _mm_mul_pd(self.twiddle2im, x914n);
        let temp_b10_10 = _mm_mul_pd(self.twiddle8im, x1013n);
        let temp_b10_11 = _mm_mul_pd(self.twiddle5im, x1112n);
        let temp_b11_1 = _mm_mul_pd(self.twiddle11im, x122n);
        let temp_b11_2 = _mm_mul_pd(self.twiddle1im, x221n);
        let temp_b11_3 = _mm_mul_pd(self.twiddle10im, x320n);
        let temp_b11_4 = _mm_mul_pd(self.twiddle2im, x419n);
        let temp_b11_5 = _mm_mul_pd(self.twiddle9im, x518n);
        let temp_b11_6 = _mm_mul_pd(self.twiddle3im, x617n);
        let temp_b11_7 = _mm_mul_pd(self.twiddle8im, x716n);
        let temp_b11_8 = _mm_mul_pd(self.twiddle4im, x815n);
        let temp_b11_9 = _mm_mul_pd(self.twiddle7im, x914n);
        let temp_b11_10 = _mm_mul_pd(self.twiddle5im, x1013n);
        let temp_b11_11 = _mm_mul_pd(self.twiddle6im, x1112n);

        let temp_a1 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a1_1,
                _mm_add_pd(
                    temp_a1_2,
                    _mm_add_pd(
                        temp_a1_3,
                        _mm_add_pd(
                            temp_a1_4,
                            _mm_add_pd(
                                temp_a1_5,
                                _mm_add_pd(
                                    temp_a1_6,
                                    _mm_add_pd(
                                        temp_a1_7,
                                        _mm_add_pd(
                                            temp_a1_8,
                                            _mm_add_pd(
                                                temp_a1_9,
                                                _mm_add_pd(temp_a1_10, temp_a1_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a2_1,
                _mm_add_pd(
                    temp_a2_2,
                    _mm_add_pd(
                        temp_a2_3,
                        _mm_add_pd(
                            temp_a2_4,
                            _mm_add_pd(
                                temp_a2_5,
                                _mm_add_pd(
                                    temp_a2_6,
                                    _mm_add_pd(
                                        temp_a2_7,
                                        _mm_add_pd(
                                            temp_a2_8,
                                            _mm_add_pd(
                                                temp_a2_9,
                                                _mm_add_pd(temp_a2_10, temp_a2_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a3_1,
                _mm_add_pd(
                    temp_a3_2,
                    _mm_add_pd(
                        temp_a3_3,
                        _mm_add_pd(
                            temp_a3_4,
                            _mm_add_pd(
                                temp_a3_5,
                                _mm_add_pd(
                                    temp_a3_6,
                                    _mm_add_pd(
                                        temp_a3_7,
                                        _mm_add_pd(
                                            temp_a3_8,
                                            _mm_add_pd(
                                                temp_a3_9,
                                                _mm_add_pd(temp_a3_10, temp_a3_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a4_1,
                _mm_add_pd(
                    temp_a4_2,
                    _mm_add_pd(
                        temp_a4_3,
                        _mm_add_pd(
                            temp_a4_4,
                            _mm_add_pd(
                                temp_a4_5,
                                _mm_add_pd(
                                    temp_a4_6,
                                    _mm_add_pd(
                                        temp_a4_7,
                                        _mm_add_pd(
                                            temp_a4_8,
                                            _mm_add_pd(
                                                temp_a4_9,
                                                _mm_add_pd(temp_a4_10, temp_a4_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a5_1,
                _mm_add_pd(
                    temp_a5_2,
                    _mm_add_pd(
                        temp_a5_3,
                        _mm_add_pd(
                            temp_a5_4,
                            _mm_add_pd(
                                temp_a5_5,
                                _mm_add_pd(
                                    temp_a5_6,
                                    _mm_add_pd(
                                        temp_a5_7,
                                        _mm_add_pd(
                                            temp_a5_8,
                                            _mm_add_pd(
                                                temp_a5_9,
                                                _mm_add_pd(temp_a5_10, temp_a5_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a6_1,
                _mm_add_pd(
                    temp_a6_2,
                    _mm_add_pd(
                        temp_a6_3,
                        _mm_add_pd(
                            temp_a6_4,
                            _mm_add_pd(
                                temp_a6_5,
                                _mm_add_pd(
                                    temp_a6_6,
                                    _mm_add_pd(
                                        temp_a6_7,
                                        _mm_add_pd(
                                            temp_a6_8,
                                            _mm_add_pd(
                                                temp_a6_9,
                                                _mm_add_pd(temp_a6_10, temp_a6_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a7_1,
                _mm_add_pd(
                    temp_a7_2,
                    _mm_add_pd(
                        temp_a7_3,
                        _mm_add_pd(
                            temp_a7_4,
                            _mm_add_pd(
                                temp_a7_5,
                                _mm_add_pd(
                                    temp_a7_6,
                                    _mm_add_pd(
                                        temp_a7_7,
                                        _mm_add_pd(
                                            temp_a7_8,
                                            _mm_add_pd(
                                                temp_a7_9,
                                                _mm_add_pd(temp_a7_10, temp_a7_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a8_1,
                _mm_add_pd(
                    temp_a8_2,
                    _mm_add_pd(
                        temp_a8_3,
                        _mm_add_pd(
                            temp_a8_4,
                            _mm_add_pd(
                                temp_a8_5,
                                _mm_add_pd(
                                    temp_a8_6,
                                    _mm_add_pd(
                                        temp_a8_7,
                                        _mm_add_pd(
                                            temp_a8_8,
                                            _mm_add_pd(
                                                temp_a8_9,
                                                _mm_add_pd(temp_a8_10, temp_a8_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a9 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a9_1,
                _mm_add_pd(
                    temp_a9_2,
                    _mm_add_pd(
                        temp_a9_3,
                        _mm_add_pd(
                            temp_a9_4,
                            _mm_add_pd(
                                temp_a9_5,
                                _mm_add_pd(
                                    temp_a9_6,
                                    _mm_add_pd(
                                        temp_a9_7,
                                        _mm_add_pd(
                                            temp_a9_8,
                                            _mm_add_pd(
                                                temp_a9_9,
                                                _mm_add_pd(temp_a9_10, temp_a9_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a10 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a10_1,
                _mm_add_pd(
                    temp_a10_2,
                    _mm_add_pd(
                        temp_a10_3,
                        _mm_add_pd(
                            temp_a10_4,
                            _mm_add_pd(
                                temp_a10_5,
                                _mm_add_pd(
                                    temp_a10_6,
                                    _mm_add_pd(
                                        temp_a10_7,
                                        _mm_add_pd(
                                            temp_a10_8,
                                            _mm_add_pd(
                                                temp_a10_9,
                                                _mm_add_pd(temp_a10_10, temp_a10_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a11 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a11_1,
                _mm_add_pd(
                    temp_a11_2,
                    _mm_add_pd(
                        temp_a11_3,
                        _mm_add_pd(
                            temp_a11_4,
                            _mm_add_pd(
                                temp_a11_5,
                                _mm_add_pd(
                                    temp_a11_6,
                                    _mm_add_pd(
                                        temp_a11_7,
                                        _mm_add_pd(
                                            temp_a11_8,
                                            _mm_add_pd(
                                                temp_a11_9,
                                                _mm_add_pd(temp_a11_10, temp_a11_11),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_pd(
            temp_b1_1,
            _mm_add_pd(
                temp_b1_2,
                _mm_add_pd(
                    temp_b1_3,
                    _mm_add_pd(
                        temp_b1_4,
                        _mm_add_pd(
                            temp_b1_5,
                            _mm_add_pd(
                                temp_b1_6,
                                _mm_add_pd(
                                    temp_b1_7,
                                    _mm_add_pd(
                                        temp_b1_8,
                                        _mm_add_pd(temp_b1_9, _mm_add_pd(temp_b1_10, temp_b1_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_pd(
            temp_b2_1,
            _mm_add_pd(
                temp_b2_2,
                _mm_add_pd(
                    temp_b2_3,
                    _mm_add_pd(
                        temp_b2_4,
                        _mm_sub_pd(
                            temp_b2_5,
                            _mm_add_pd(
                                temp_b2_6,
                                _mm_add_pd(
                                    temp_b2_7,
                                    _mm_add_pd(
                                        temp_b2_8,
                                        _mm_add_pd(temp_b2_9, _mm_add_pd(temp_b2_10, temp_b2_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_pd(
            temp_b3_1,
            _mm_add_pd(
                temp_b3_2,
                _mm_sub_pd(
                    temp_b3_3,
                    _mm_add_pd(
                        temp_b3_4,
                        _mm_add_pd(
                            temp_b3_5,
                            _mm_add_pd(
                                temp_b3_6,
                                _mm_sub_pd(
                                    temp_b3_7,
                                    _mm_add_pd(
                                        temp_b3_8,
                                        _mm_add_pd(temp_b3_9, _mm_add_pd(temp_b3_10, temp_b3_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_pd(
            temp_b4_1,
            _mm_sub_pd(
                temp_b4_2,
                _mm_add_pd(
                    temp_b4_3,
                    _mm_add_pd(
                        temp_b4_4,
                        _mm_sub_pd(
                            temp_b4_5,
                            _mm_add_pd(
                                temp_b4_6,
                                _mm_add_pd(
                                    temp_b4_7,
                                    _mm_sub_pd(
                                        temp_b4_8,
                                        _mm_add_pd(temp_b4_9, _mm_add_pd(temp_b4_10, temp_b4_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_add_pd(
            temp_b5_1,
            _mm_sub_pd(
                temp_b5_2,
                _mm_add_pd(
                    temp_b5_3,
                    _mm_sub_pd(
                        temp_b5_4,
                        _mm_add_pd(
                            temp_b5_5,
                            _mm_sub_pd(
                                temp_b5_6,
                                _mm_add_pd(
                                    temp_b5_7,
                                    _mm_add_pd(
                                        temp_b5_8,
                                        _mm_sub_pd(temp_b5_9, _mm_add_pd(temp_b5_10, temp_b5_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_sub_pd(
            temp_b6_1,
            _mm_add_pd(
                temp_b6_2,
                _mm_sub_pd(
                    temp_b6_3,
                    _mm_add_pd(
                        temp_b6_4,
                        _mm_sub_pd(
                            temp_b6_5,
                            _mm_add_pd(
                                temp_b6_6,
                                _mm_sub_pd(
                                    temp_b6_7,
                                    _mm_add_pd(
                                        temp_b6_8,
                                        _mm_sub_pd(temp_b6_9, _mm_add_pd(temp_b6_10, temp_b6_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_sub_pd(
            temp_b7_1,
            _mm_add_pd(
                temp_b7_2,
                _mm_sub_pd(
                    temp_b7_3,
                    _mm_sub_pd(
                        temp_b7_4,
                        _mm_add_pd(
                            temp_b7_5,
                            _mm_sub_pd(
                                temp_b7_6,
                                _mm_add_pd(
                                    temp_b7_7,
                                    _mm_sub_pd(
                                        temp_b7_8,
                                        _mm_sub_pd(temp_b7_9, _mm_add_pd(temp_b7_10, temp_b7_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_pd(
            temp_b8_1,
            _mm_sub_pd(
                temp_b8_2,
                _mm_add_pd(
                    temp_b8_3,
                    _mm_sub_pd(
                        temp_b8_4,
                        _mm_sub_pd(
                            temp_b8_5,
                            _mm_add_pd(
                                temp_b8_6,
                                _mm_sub_pd(
                                    temp_b8_7,
                                    _mm_sub_pd(
                                        temp_b8_8,
                                        _mm_add_pd(temp_b8_9, _mm_sub_pd(temp_b8_10, temp_b8_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b9 = _mm_sub_pd(
            temp_b9_1,
            _mm_sub_pd(
                temp_b9_2,
                _mm_sub_pd(
                    temp_b9_3,
                    _mm_add_pd(
                        temp_b9_4,
                        _mm_sub_pd(
                            temp_b9_5,
                            _mm_sub_pd(
                                temp_b9_6,
                                _mm_sub_pd(
                                    temp_b9_7,
                                    _mm_sub_pd(
                                        temp_b9_8,
                                        _mm_add_pd(temp_b9_9, _mm_sub_pd(temp_b9_10, temp_b9_11)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b10 = _mm_sub_pd(
            temp_b10_1,
            _mm_sub_pd(
                temp_b10_2,
                _mm_sub_pd(
                    temp_b10_3,
                    _mm_sub_pd(
                        temp_b10_4,
                        _mm_sub_pd(
                            temp_b10_5,
                            _mm_sub_pd(
                                temp_b10_6,
                                _mm_add_pd(
                                    temp_b10_7,
                                    _mm_sub_pd(
                                        temp_b10_8,
                                        _mm_sub_pd(
                                            temp_b10_9,
                                            _mm_sub_pd(temp_b10_10, temp_b10_11),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b11 = _mm_sub_pd(
            temp_b11_1,
            _mm_sub_pd(
                temp_b11_2,
                _mm_sub_pd(
                    temp_b11_3,
                    _mm_sub_pd(
                        temp_b11_4,
                        _mm_sub_pd(
                            temp_b11_5,
                            _mm_sub_pd(
                                temp_b11_6,
                                _mm_sub_pd(
                                    temp_b11_7,
                                    _mm_sub_pd(
                                        temp_b11_8,
                                        _mm_sub_pd(
                                            temp_b11_9,
                                            _mm_sub_pd(temp_b11_10, temp_b11_11),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);
        let temp_b6_rot = self.rotate.rotate(temp_b6);
        let temp_b7_rot = self.rotate.rotate(temp_b7);
        let temp_b8_rot = self.rotate.rotate(temp_b8);
        let temp_b9_rot = self.rotate.rotate(temp_b9);
        let temp_b10_rot = self.rotate.rotate(temp_b10);
        let temp_b11_rot = self.rotate.rotate(temp_b11);

        let x0 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                x122p,
                _mm_add_pd(
                    x221p,
                    _mm_add_pd(
                        x320p,
                        _mm_add_pd(
                            x419p,
                            _mm_add_pd(
                                x518p,
                                _mm_add_pd(
                                    x617p,
                                    _mm_add_pd(
                                        x716p,
                                        _mm_add_pd(
                                            x815p,
                                            _mm_add_pd(x914p, _mm_add_pd(x1013p, x1112p)),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
        let x4 = _mm_add_pd(temp_a4, temp_b4_rot);
        let x5 = _mm_add_pd(temp_a5, temp_b5_rot);
        let x6 = _mm_add_pd(temp_a6, temp_b6_rot);
        let x7 = _mm_add_pd(temp_a7, temp_b7_rot);
        let x8 = _mm_add_pd(temp_a8, temp_b8_rot);
        let x9 = _mm_add_pd(temp_a9, temp_b9_rot);
        let x10 = _mm_add_pd(temp_a10, temp_b10_rot);
        let x11 = _mm_add_pd(temp_a11, temp_b11_rot);
        let x12 = _mm_sub_pd(temp_a11, temp_b11_rot);
        let x13 = _mm_sub_pd(temp_a10, temp_b10_rot);
        let x14 = _mm_sub_pd(temp_a9, temp_b9_rot);
        let x15 = _mm_sub_pd(temp_a8, temp_b8_rot);
        let x16 = _mm_sub_pd(temp_a7, temp_b7_rot);
        let x17 = _mm_sub_pd(temp_a6, temp_b6_rot);
        let x18 = _mm_sub_pd(temp_a5, temp_b5_rot);
        let x19 = _mm_sub_pd(temp_a4, temp_b4_rot);
        let x20 = _mm_sub_pd(temp_a3, temp_b3_rot);
        let x21 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x22 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
            x19, x20, x21, x22,
        ]
    }
}

//   ____   ___            _________  _     _ _
//  |___ \ / _ \          |___ /___ \| |__ (_) |_
//    __) | (_) |  _____    |_ \ __) | '_ \| | __|
//   / __/ \__, | |_____|  ___) / __/| |_) | | |_
//  |_____|  /_/          |____/_____|_.__/|_|\__|
//

pub struct SseF32Butterfly29<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
    twiddle3re: __m128,
    twiddle3im: __m128,
    twiddle4re: __m128,
    twiddle4im: __m128,
    twiddle5re: __m128,
    twiddle5im: __m128,
    twiddle6re: __m128,
    twiddle6im: __m128,
    twiddle7re: __m128,
    twiddle7im: __m128,
    twiddle8re: __m128,
    twiddle8im: __m128,
    twiddle9re: __m128,
    twiddle9im: __m128,
    twiddle10re: __m128,
    twiddle10im: __m128,
    twiddle11re: __m128,
    twiddle11im: __m128,
    twiddle12re: __m128,
    twiddle12im: __m128,
    twiddle13re: __m128,
    twiddle13im: __m128,
    twiddle14re: __m128,
    twiddle14im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly29, 29, |this: &SseF32Butterfly29<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly29, 29, |this: &SseF32Butterfly29<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly29<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 29, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 29, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 29, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 29, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 29, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 29, direction);
        let tw7: Complex<f32> = twiddles::compute_twiddle(7, 29, direction);
        let tw8: Complex<f32> = twiddles::compute_twiddle(8, 29, direction);
        let tw9: Complex<f32> = twiddles::compute_twiddle(9, 29, direction);
        let tw10: Complex<f32> = twiddles::compute_twiddle(10, 29, direction);
        let tw11: Complex<f32> = twiddles::compute_twiddle(11, 29, direction);
        let tw12: Complex<f32> = twiddles::compute_twiddle(12, 29, direction);
        let tw13: Complex<f32> = twiddles::compute_twiddle(13, 29, direction);
        let tw14: Complex<f32> = twiddles::compute_twiddle(14, 29, direction);
        let twiddle1re = unsafe { _mm_load1_ps(&tw1.re) };
        let twiddle1im = unsafe { _mm_load1_ps(&tw1.im) };
        let twiddle2re = unsafe { _mm_load1_ps(&tw2.re) };
        let twiddle2im = unsafe { _mm_load1_ps(&tw2.im) };
        let twiddle3re = unsafe { _mm_load1_ps(&tw3.re) };
        let twiddle3im = unsafe { _mm_load1_ps(&tw3.im) };
        let twiddle4re = unsafe { _mm_load1_ps(&tw4.re) };
        let twiddle4im = unsafe { _mm_load1_ps(&tw4.im) };
        let twiddle5re = unsafe { _mm_load1_ps(&tw5.re) };
        let twiddle5im = unsafe { _mm_load1_ps(&tw5.im) };
        let twiddle6re = unsafe { _mm_load1_ps(&tw6.re) };
        let twiddle6im = unsafe { _mm_load1_ps(&tw6.im) };
        let twiddle7re = unsafe { _mm_load1_ps(&tw7.re) };
        let twiddle7im = unsafe { _mm_load1_ps(&tw7.im) };
        let twiddle8re = unsafe { _mm_load1_ps(&tw8.re) };
        let twiddle8im = unsafe { _mm_load1_ps(&tw8.im) };
        let twiddle9re = unsafe { _mm_load1_ps(&tw9.re) };
        let twiddle9im = unsafe { _mm_load1_ps(&tw9.im) };
        let twiddle10re = unsafe { _mm_load1_ps(&tw10.re) };
        let twiddle10im = unsafe { _mm_load1_ps(&tw10.im) };
        let twiddle11re = unsafe { _mm_load1_ps(&tw11.re) };
        let twiddle11im = unsafe { _mm_load1_ps(&tw11.im) };
        let twiddle12re = unsafe { _mm_load1_ps(&tw12.re) };
        let twiddle12im = unsafe { _mm_load1_ps(&tw12.im) };
        let twiddle13re = unsafe { _mm_load1_ps(&tw13.re) };
        let twiddle13im = unsafe { _mm_load1_ps(&tw13.im) };
        let twiddle14re = unsafe { _mm_load1_ps(&tw14.re) };
        let twiddle14im = unsafe { _mm_load1_ps(&tw14.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
            twiddle9re,
            twiddle9im,
            twiddle10re,
            twiddle10im,
            twiddle11re,
            twiddle11im,
            twiddle12re,
            twiddle12im,
            twiddle13re,
            twiddle13im,
            twiddle14re,
            twiddle14im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let v1 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(1) as *const f64));
        let v2 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(2) as *const f64));
        let v3 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(3) as *const f64));
        let v4 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(4) as *const f64));
        let v5 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(5) as *const f64));
        let v6 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(6) as *const f64));
        let v7 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(7) as *const f64));
        let v8 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(8) as *const f64));
        let v9 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(9) as *const f64));
        let v10 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(10) as *const f64));
        let v11 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(11) as *const f64));
        let v12 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(12) as *const f64));
        let v13 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(13) as *const f64));
        let v14 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(14) as *const f64));
        let v15 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(15) as *const f64));
        let v16 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(16) as *const f64));
        let v17 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(17) as *const f64));
        let v18 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(18) as *const f64));
        let v19 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(19) as *const f64));
        let v20 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(20) as *const f64));
        let v21 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(21) as *const f64));
        let v22 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(22) as *const f64));
        let v23 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(23) as *const f64));
        let v24 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(24) as *const f64));
        let v25 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(25) as *const f64));
        let v26 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(26) as *const f64));
        let v27 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(27) as *const f64));
        let v28 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(28) as *const f64));

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
        ]);

        let val = std::mem::transmute::<[__m128; 29], [Complex<f32>; 58]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[34];
        *output_slice.add(18) = val[36];
        *output_slice.add(19) = val[38];
        *output_slice.add(20) = val[40];
        *output_slice.add(21) = val[42];
        *output_slice.add(22) = val[44];
        *output_slice.add(23) = val[46];
        *output_slice.add(24) = val[48];
        *output_slice.add(25) = val[50];
        *output_slice.add(26) = val[52];
        *output_slice.add(27) = val[54];
        *output_slice.add(28) = val[56];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4a5 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valuea6a7 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valuea8a9 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let valuea10a11 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let valuea12a13 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let valuea14a15 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let valuea16a17 = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let valuea18a19 = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let valuea20a21 = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let valuea22a23 = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let valuea24a25 = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);
        let valuea26a27 = _mm_loadu_ps(input.as_ptr().add(26) as *const f32);
        let valuea28b0 = _mm_loadu_ps(input.as_ptr().add(28) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(30) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(32) as *const f32);
        let valueb5b6 = _mm_loadu_ps(input.as_ptr().add(34) as *const f32);
        let valueb7b8 = _mm_loadu_ps(input.as_ptr().add(36) as *const f32);
        let valueb9b10 = _mm_loadu_ps(input.as_ptr().add(38) as *const f32);
        let valueb11b12 = _mm_loadu_ps(input.as_ptr().add(40) as *const f32);
        let valueb13b14 = _mm_loadu_ps(input.as_ptr().add(42) as *const f32);
        let valueb15b16 = _mm_loadu_ps(input.as_ptr().add(44) as *const f32);
        let valueb17b18 = _mm_loadu_ps(input.as_ptr().add(46) as *const f32);
        let valueb19b20 = _mm_loadu_ps(input.as_ptr().add(48) as *const f32);
        let valueb21b22 = _mm_loadu_ps(input.as_ptr().add(50) as *const f32);
        let valueb23b24 = _mm_loadu_ps(input.as_ptr().add(52) as *const f32);
        let valueb25b26 = _mm_loadu_ps(input.as_ptr().add(54) as *const f32);
        let valueb27b28 = _mm_loadu_ps(input.as_ptr().add(56) as *const f32);

        let v0 = pack_1and2_f32(valuea0a1, valuea28b0);
        let v1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let v2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let v3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let v4 = pack_1and2_f32(valuea4a5, valueb3b4);
        let v5 = pack_2and1_f32(valuea4a5, valueb5b6);
        let v6 = pack_1and2_f32(valuea6a7, valueb5b6);
        let v7 = pack_2and1_f32(valuea6a7, valueb7b8);
        let v8 = pack_1and2_f32(valuea8a9, valueb7b8);
        let v9 = pack_2and1_f32(valuea8a9, valueb9b10);
        let v10 = pack_1and2_f32(valuea10a11, valueb9b10);
        let v11 = pack_2and1_f32(valuea10a11, valueb11b12);
        let v12 = pack_1and2_f32(valuea12a13, valueb11b12);
        let v13 = pack_2and1_f32(valuea12a13, valueb13b14);
        let v14 = pack_1and2_f32(valuea14a15, valueb13b14);
        let v15 = pack_2and1_f32(valuea14a15, valueb15b16);
        let v16 = pack_1and2_f32(valuea16a17, valueb15b16);
        let v17 = pack_2and1_f32(valuea16a17, valueb17b18);
        let v18 = pack_1and2_f32(valuea18a19, valueb17b18);
        let v19 = pack_2and1_f32(valuea18a19, valueb19b20);
        let v20 = pack_1and2_f32(valuea20a21, valueb19b20);
        let v21 = pack_2and1_f32(valuea20a21, valueb21b22);
        let v22 = pack_1and2_f32(valuea22a23, valueb21b22);
        let v23 = pack_2and1_f32(valuea22a23, valueb23b24);
        let v24 = pack_1and2_f32(valuea24a25, valueb23b24);
        let v25 = pack_2and1_f32(valuea24a25, valueb25b26);
        let v26 = pack_1and2_f32(valuea26a27, valueb25b26);
        let v27 = pack_2and1_f32(valuea26a27, valueb27b28);
        let v28 = pack_1and2_f32(valuea28b0, valueb27b28);

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
        ]);

        let val = std::mem::transmute::<[__m128; 29], [Complex<f32>; 58]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[34];
        *output_slice.add(18) = val[36];
        *output_slice.add(19) = val[38];
        *output_slice.add(20) = val[40];
        *output_slice.add(21) = val[42];
        *output_slice.add(22) = val[44];
        *output_slice.add(23) = val[46];
        *output_slice.add(24) = val[48];
        *output_slice.add(25) = val[50];
        *output_slice.add(26) = val[52];
        *output_slice.add(27) = val[54];
        *output_slice.add(28) = val[56];
        *output_slice.add(29) = val[1];
        *output_slice.add(30) = val[3];
        *output_slice.add(31) = val[5];
        *output_slice.add(32) = val[7];
        *output_slice.add(33) = val[9];
        *output_slice.add(34) = val[11];
        *output_slice.add(35) = val[13];
        *output_slice.add(36) = val[15];
        *output_slice.add(37) = val[17];
        *output_slice.add(38) = val[19];
        *output_slice.add(39) = val[21];
        *output_slice.add(40) = val[23];
        *output_slice.add(41) = val[25];
        *output_slice.add(42) = val[27];
        *output_slice.add(43) = val[29];
        *output_slice.add(44) = val[31];
        *output_slice.add(45) = val[33];
        *output_slice.add(46) = val[35];
        *output_slice.add(47) = val[37];
        *output_slice.add(48) = val[39];
        *output_slice.add(49) = val[41];
        *output_slice.add(50) = val[43];
        *output_slice.add(51) = val[45];
        *output_slice.add(52) = val[47];
        *output_slice.add(53) = val[49];
        *output_slice.add(54) = val[51];
        *output_slice.add(55) = val[53];
        *output_slice.add(56) = val[55];
        *output_slice.add(57) = val[57];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values: [__m128; 29]) -> [__m128; 29] {
        // This is a SSE translation of the scalar 17-point butterfly
        let x128p = _mm_add_ps(values[1], values[28]);
        let x128n = _mm_sub_ps(values[1], values[28]);
        let x227p = _mm_add_ps(values[2], values[27]);
        let x227n = _mm_sub_ps(values[2], values[27]);
        let x326p = _mm_add_ps(values[3], values[26]);
        let x326n = _mm_sub_ps(values[3], values[26]);
        let x425p = _mm_add_ps(values[4], values[25]);
        let x425n = _mm_sub_ps(values[4], values[25]);
        let x524p = _mm_add_ps(values[5], values[24]);
        let x524n = _mm_sub_ps(values[5], values[24]);
        let x623p = _mm_add_ps(values[6], values[23]);
        let x623n = _mm_sub_ps(values[6], values[23]);
        let x722p = _mm_add_ps(values[7], values[22]);
        let x722n = _mm_sub_ps(values[7], values[22]);
        let x821p = _mm_add_ps(values[8], values[21]);
        let x821n = _mm_sub_ps(values[8], values[21]);
        let x920p = _mm_add_ps(values[9], values[20]);
        let x920n = _mm_sub_ps(values[9], values[20]);
        let x1019p = _mm_add_ps(values[10], values[19]);
        let x1019n = _mm_sub_ps(values[10], values[19]);
        let x1118p = _mm_add_ps(values[11], values[18]);
        let x1118n = _mm_sub_ps(values[11], values[18]);
        let x1217p = _mm_add_ps(values[12], values[17]);
        let x1217n = _mm_sub_ps(values[12], values[17]);
        let x1316p = _mm_add_ps(values[13], values[16]);
        let x1316n = _mm_sub_ps(values[13], values[16]);
        let x1415p = _mm_add_ps(values[14], values[15]);
        let x1415n = _mm_sub_ps(values[14], values[15]);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x128p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x227p);
        let temp_a1_3 = _mm_mul_ps(self.twiddle3re, x326p);
        let temp_a1_4 = _mm_mul_ps(self.twiddle4re, x425p);
        let temp_a1_5 = _mm_mul_ps(self.twiddle5re, x524p);
        let temp_a1_6 = _mm_mul_ps(self.twiddle6re, x623p);
        let temp_a1_7 = _mm_mul_ps(self.twiddle7re, x722p);
        let temp_a1_8 = _mm_mul_ps(self.twiddle8re, x821p);
        let temp_a1_9 = _mm_mul_ps(self.twiddle9re, x920p);
        let temp_a1_10 = _mm_mul_ps(self.twiddle10re, x1019p);
        let temp_a1_11 = _mm_mul_ps(self.twiddle11re, x1118p);
        let temp_a1_12 = _mm_mul_ps(self.twiddle12re, x1217p);
        let temp_a1_13 = _mm_mul_ps(self.twiddle13re, x1316p);
        let temp_a1_14 = _mm_mul_ps(self.twiddle14re, x1415p);
        let temp_a2_1 = _mm_mul_ps(self.twiddle2re, x128p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle4re, x227p);
        let temp_a2_3 = _mm_mul_ps(self.twiddle6re, x326p);
        let temp_a2_4 = _mm_mul_ps(self.twiddle8re, x425p);
        let temp_a2_5 = _mm_mul_ps(self.twiddle10re, x524p);
        let temp_a2_6 = _mm_mul_ps(self.twiddle12re, x623p);
        let temp_a2_7 = _mm_mul_ps(self.twiddle14re, x722p);
        let temp_a2_8 = _mm_mul_ps(self.twiddle13re, x821p);
        let temp_a2_9 = _mm_mul_ps(self.twiddle11re, x920p);
        let temp_a2_10 = _mm_mul_ps(self.twiddle9re, x1019p);
        let temp_a2_11 = _mm_mul_ps(self.twiddle7re, x1118p);
        let temp_a2_12 = _mm_mul_ps(self.twiddle5re, x1217p);
        let temp_a2_13 = _mm_mul_ps(self.twiddle3re, x1316p);
        let temp_a2_14 = _mm_mul_ps(self.twiddle1re, x1415p);
        let temp_a3_1 = _mm_mul_ps(self.twiddle3re, x128p);
        let temp_a3_2 = _mm_mul_ps(self.twiddle6re, x227p);
        let temp_a3_3 = _mm_mul_ps(self.twiddle9re, x326p);
        let temp_a3_4 = _mm_mul_ps(self.twiddle12re, x425p);
        let temp_a3_5 = _mm_mul_ps(self.twiddle14re, x524p);
        let temp_a3_6 = _mm_mul_ps(self.twiddle11re, x623p);
        let temp_a3_7 = _mm_mul_ps(self.twiddle8re, x722p);
        let temp_a3_8 = _mm_mul_ps(self.twiddle5re, x821p);
        let temp_a3_9 = _mm_mul_ps(self.twiddle2re, x920p);
        let temp_a3_10 = _mm_mul_ps(self.twiddle1re, x1019p);
        let temp_a3_11 = _mm_mul_ps(self.twiddle4re, x1118p);
        let temp_a3_12 = _mm_mul_ps(self.twiddle7re, x1217p);
        let temp_a3_13 = _mm_mul_ps(self.twiddle10re, x1316p);
        let temp_a3_14 = _mm_mul_ps(self.twiddle13re, x1415p);
        let temp_a4_1 = _mm_mul_ps(self.twiddle4re, x128p);
        let temp_a4_2 = _mm_mul_ps(self.twiddle8re, x227p);
        let temp_a4_3 = _mm_mul_ps(self.twiddle12re, x326p);
        let temp_a4_4 = _mm_mul_ps(self.twiddle13re, x425p);
        let temp_a4_5 = _mm_mul_ps(self.twiddle9re, x524p);
        let temp_a4_6 = _mm_mul_ps(self.twiddle5re, x623p);
        let temp_a4_7 = _mm_mul_ps(self.twiddle1re, x722p);
        let temp_a4_8 = _mm_mul_ps(self.twiddle3re, x821p);
        let temp_a4_9 = _mm_mul_ps(self.twiddle7re, x920p);
        let temp_a4_10 = _mm_mul_ps(self.twiddle11re, x1019p);
        let temp_a4_11 = _mm_mul_ps(self.twiddle14re, x1118p);
        let temp_a4_12 = _mm_mul_ps(self.twiddle10re, x1217p);
        let temp_a4_13 = _mm_mul_ps(self.twiddle6re, x1316p);
        let temp_a4_14 = _mm_mul_ps(self.twiddle2re, x1415p);
        let temp_a5_1 = _mm_mul_ps(self.twiddle5re, x128p);
        let temp_a5_2 = _mm_mul_ps(self.twiddle10re, x227p);
        let temp_a5_3 = _mm_mul_ps(self.twiddle14re, x326p);
        let temp_a5_4 = _mm_mul_ps(self.twiddle9re, x425p);
        let temp_a5_5 = _mm_mul_ps(self.twiddle4re, x524p);
        let temp_a5_6 = _mm_mul_ps(self.twiddle1re, x623p);
        let temp_a5_7 = _mm_mul_ps(self.twiddle6re, x722p);
        let temp_a5_8 = _mm_mul_ps(self.twiddle11re, x821p);
        let temp_a5_9 = _mm_mul_ps(self.twiddle13re, x920p);
        let temp_a5_10 = _mm_mul_ps(self.twiddle8re, x1019p);
        let temp_a5_11 = _mm_mul_ps(self.twiddle3re, x1118p);
        let temp_a5_12 = _mm_mul_ps(self.twiddle2re, x1217p);
        let temp_a5_13 = _mm_mul_ps(self.twiddle7re, x1316p);
        let temp_a5_14 = _mm_mul_ps(self.twiddle12re, x1415p);
        let temp_a6_1 = _mm_mul_ps(self.twiddle6re, x128p);
        let temp_a6_2 = _mm_mul_ps(self.twiddle12re, x227p);
        let temp_a6_3 = _mm_mul_ps(self.twiddle11re, x326p);
        let temp_a6_4 = _mm_mul_ps(self.twiddle5re, x425p);
        let temp_a6_5 = _mm_mul_ps(self.twiddle1re, x524p);
        let temp_a6_6 = _mm_mul_ps(self.twiddle7re, x623p);
        let temp_a6_7 = _mm_mul_ps(self.twiddle13re, x722p);
        let temp_a6_8 = _mm_mul_ps(self.twiddle10re, x821p);
        let temp_a6_9 = _mm_mul_ps(self.twiddle4re, x920p);
        let temp_a6_10 = _mm_mul_ps(self.twiddle2re, x1019p);
        let temp_a6_11 = _mm_mul_ps(self.twiddle8re, x1118p);
        let temp_a6_12 = _mm_mul_ps(self.twiddle14re, x1217p);
        let temp_a6_13 = _mm_mul_ps(self.twiddle9re, x1316p);
        let temp_a6_14 = _mm_mul_ps(self.twiddle3re, x1415p);
        let temp_a7_1 = _mm_mul_ps(self.twiddle7re, x128p);
        let temp_a7_2 = _mm_mul_ps(self.twiddle14re, x227p);
        let temp_a7_3 = _mm_mul_ps(self.twiddle8re, x326p);
        let temp_a7_4 = _mm_mul_ps(self.twiddle1re, x425p);
        let temp_a7_5 = _mm_mul_ps(self.twiddle6re, x524p);
        let temp_a7_6 = _mm_mul_ps(self.twiddle13re, x623p);
        let temp_a7_7 = _mm_mul_ps(self.twiddle9re, x722p);
        let temp_a7_8 = _mm_mul_ps(self.twiddle2re, x821p);
        let temp_a7_9 = _mm_mul_ps(self.twiddle5re, x920p);
        let temp_a7_10 = _mm_mul_ps(self.twiddle12re, x1019p);
        let temp_a7_11 = _mm_mul_ps(self.twiddle10re, x1118p);
        let temp_a7_12 = _mm_mul_ps(self.twiddle3re, x1217p);
        let temp_a7_13 = _mm_mul_ps(self.twiddle4re, x1316p);
        let temp_a7_14 = _mm_mul_ps(self.twiddle11re, x1415p);
        let temp_a8_1 = _mm_mul_ps(self.twiddle8re, x128p);
        let temp_a8_2 = _mm_mul_ps(self.twiddle13re, x227p);
        let temp_a8_3 = _mm_mul_ps(self.twiddle5re, x326p);
        let temp_a8_4 = _mm_mul_ps(self.twiddle3re, x425p);
        let temp_a8_5 = _mm_mul_ps(self.twiddle11re, x524p);
        let temp_a8_6 = _mm_mul_ps(self.twiddle10re, x623p);
        let temp_a8_7 = _mm_mul_ps(self.twiddle2re, x722p);
        let temp_a8_8 = _mm_mul_ps(self.twiddle6re, x821p);
        let temp_a8_9 = _mm_mul_ps(self.twiddle14re, x920p);
        let temp_a8_10 = _mm_mul_ps(self.twiddle7re, x1019p);
        let temp_a8_11 = _mm_mul_ps(self.twiddle1re, x1118p);
        let temp_a8_12 = _mm_mul_ps(self.twiddle9re, x1217p);
        let temp_a8_13 = _mm_mul_ps(self.twiddle12re, x1316p);
        let temp_a8_14 = _mm_mul_ps(self.twiddle4re, x1415p);
        let temp_a9_1 = _mm_mul_ps(self.twiddle9re, x128p);
        let temp_a9_2 = _mm_mul_ps(self.twiddle11re, x227p);
        let temp_a9_3 = _mm_mul_ps(self.twiddle2re, x326p);
        let temp_a9_4 = _mm_mul_ps(self.twiddle7re, x425p);
        let temp_a9_5 = _mm_mul_ps(self.twiddle13re, x524p);
        let temp_a9_6 = _mm_mul_ps(self.twiddle4re, x623p);
        let temp_a9_7 = _mm_mul_ps(self.twiddle5re, x722p);
        let temp_a9_8 = _mm_mul_ps(self.twiddle14re, x821p);
        let temp_a9_9 = _mm_mul_ps(self.twiddle6re, x920p);
        let temp_a9_10 = _mm_mul_ps(self.twiddle3re, x1019p);
        let temp_a9_11 = _mm_mul_ps(self.twiddle12re, x1118p);
        let temp_a9_12 = _mm_mul_ps(self.twiddle8re, x1217p);
        let temp_a9_13 = _mm_mul_ps(self.twiddle1re, x1316p);
        let temp_a9_14 = _mm_mul_ps(self.twiddle10re, x1415p);
        let temp_a10_1 = _mm_mul_ps(self.twiddle10re, x128p);
        let temp_a10_2 = _mm_mul_ps(self.twiddle9re, x227p);
        let temp_a10_3 = _mm_mul_ps(self.twiddle1re, x326p);
        let temp_a10_4 = _mm_mul_ps(self.twiddle11re, x425p);
        let temp_a10_5 = _mm_mul_ps(self.twiddle8re, x524p);
        let temp_a10_6 = _mm_mul_ps(self.twiddle2re, x623p);
        let temp_a10_7 = _mm_mul_ps(self.twiddle12re, x722p);
        let temp_a10_8 = _mm_mul_ps(self.twiddle7re, x821p);
        let temp_a10_9 = _mm_mul_ps(self.twiddle3re, x920p);
        let temp_a10_10 = _mm_mul_ps(self.twiddle13re, x1019p);
        let temp_a10_11 = _mm_mul_ps(self.twiddle6re, x1118p);
        let temp_a10_12 = _mm_mul_ps(self.twiddle4re, x1217p);
        let temp_a10_13 = _mm_mul_ps(self.twiddle14re, x1316p);
        let temp_a10_14 = _mm_mul_ps(self.twiddle5re, x1415p);
        let temp_a11_1 = _mm_mul_ps(self.twiddle11re, x128p);
        let temp_a11_2 = _mm_mul_ps(self.twiddle7re, x227p);
        let temp_a11_3 = _mm_mul_ps(self.twiddle4re, x326p);
        let temp_a11_4 = _mm_mul_ps(self.twiddle14re, x425p);
        let temp_a11_5 = _mm_mul_ps(self.twiddle3re, x524p);
        let temp_a11_6 = _mm_mul_ps(self.twiddle8re, x623p);
        let temp_a11_7 = _mm_mul_ps(self.twiddle10re, x722p);
        let temp_a11_8 = _mm_mul_ps(self.twiddle1re, x821p);
        let temp_a11_9 = _mm_mul_ps(self.twiddle12re, x920p);
        let temp_a11_10 = _mm_mul_ps(self.twiddle6re, x1019p);
        let temp_a11_11 = _mm_mul_ps(self.twiddle5re, x1118p);
        let temp_a11_12 = _mm_mul_ps(self.twiddle13re, x1217p);
        let temp_a11_13 = _mm_mul_ps(self.twiddle2re, x1316p);
        let temp_a11_14 = _mm_mul_ps(self.twiddle9re, x1415p);
        let temp_a12_1 = _mm_mul_ps(self.twiddle12re, x128p);
        let temp_a12_2 = _mm_mul_ps(self.twiddle5re, x227p);
        let temp_a12_3 = _mm_mul_ps(self.twiddle7re, x326p);
        let temp_a12_4 = _mm_mul_ps(self.twiddle10re, x425p);
        let temp_a12_5 = _mm_mul_ps(self.twiddle2re, x524p);
        let temp_a12_6 = _mm_mul_ps(self.twiddle14re, x623p);
        let temp_a12_7 = _mm_mul_ps(self.twiddle3re, x722p);
        let temp_a12_8 = _mm_mul_ps(self.twiddle9re, x821p);
        let temp_a12_9 = _mm_mul_ps(self.twiddle8re, x920p);
        let temp_a12_10 = _mm_mul_ps(self.twiddle4re, x1019p);
        let temp_a12_11 = _mm_mul_ps(self.twiddle13re, x1118p);
        let temp_a12_12 = _mm_mul_ps(self.twiddle1re, x1217p);
        let temp_a12_13 = _mm_mul_ps(self.twiddle11re, x1316p);
        let temp_a12_14 = _mm_mul_ps(self.twiddle6re, x1415p);
        let temp_a13_1 = _mm_mul_ps(self.twiddle13re, x128p);
        let temp_a13_2 = _mm_mul_ps(self.twiddle3re, x227p);
        let temp_a13_3 = _mm_mul_ps(self.twiddle10re, x326p);
        let temp_a13_4 = _mm_mul_ps(self.twiddle6re, x425p);
        let temp_a13_5 = _mm_mul_ps(self.twiddle7re, x524p);
        let temp_a13_6 = _mm_mul_ps(self.twiddle9re, x623p);
        let temp_a13_7 = _mm_mul_ps(self.twiddle4re, x722p);
        let temp_a13_8 = _mm_mul_ps(self.twiddle12re, x821p);
        let temp_a13_9 = _mm_mul_ps(self.twiddle1re, x920p);
        let temp_a13_10 = _mm_mul_ps(self.twiddle14re, x1019p);
        let temp_a13_11 = _mm_mul_ps(self.twiddle2re, x1118p);
        let temp_a13_12 = _mm_mul_ps(self.twiddle11re, x1217p);
        let temp_a13_13 = _mm_mul_ps(self.twiddle5re, x1316p);
        let temp_a13_14 = _mm_mul_ps(self.twiddle8re, x1415p);
        let temp_a14_1 = _mm_mul_ps(self.twiddle14re, x128p);
        let temp_a14_2 = _mm_mul_ps(self.twiddle1re, x227p);
        let temp_a14_3 = _mm_mul_ps(self.twiddle13re, x326p);
        let temp_a14_4 = _mm_mul_ps(self.twiddle2re, x425p);
        let temp_a14_5 = _mm_mul_ps(self.twiddle12re, x524p);
        let temp_a14_6 = _mm_mul_ps(self.twiddle3re, x623p);
        let temp_a14_7 = _mm_mul_ps(self.twiddle11re, x722p);
        let temp_a14_8 = _mm_mul_ps(self.twiddle4re, x821p);
        let temp_a14_9 = _mm_mul_ps(self.twiddle10re, x920p);
        let temp_a14_10 = _mm_mul_ps(self.twiddle5re, x1019p);
        let temp_a14_11 = _mm_mul_ps(self.twiddle9re, x1118p);
        let temp_a14_12 = _mm_mul_ps(self.twiddle6re, x1217p);
        let temp_a14_13 = _mm_mul_ps(self.twiddle8re, x1316p);
        let temp_a14_14 = _mm_mul_ps(self.twiddle7re, x1415p);

        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x128n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x227n);
        let temp_b1_3 = _mm_mul_ps(self.twiddle3im, x326n);
        let temp_b1_4 = _mm_mul_ps(self.twiddle4im, x425n);
        let temp_b1_5 = _mm_mul_ps(self.twiddle5im, x524n);
        let temp_b1_6 = _mm_mul_ps(self.twiddle6im, x623n);
        let temp_b1_7 = _mm_mul_ps(self.twiddle7im, x722n);
        let temp_b1_8 = _mm_mul_ps(self.twiddle8im, x821n);
        let temp_b1_9 = _mm_mul_ps(self.twiddle9im, x920n);
        let temp_b1_10 = _mm_mul_ps(self.twiddle10im, x1019n);
        let temp_b1_11 = _mm_mul_ps(self.twiddle11im, x1118n);
        let temp_b1_12 = _mm_mul_ps(self.twiddle12im, x1217n);
        let temp_b1_13 = _mm_mul_ps(self.twiddle13im, x1316n);
        let temp_b1_14 = _mm_mul_ps(self.twiddle14im, x1415n);
        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x128n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle4im, x227n);
        let temp_b2_3 = _mm_mul_ps(self.twiddle6im, x326n);
        let temp_b2_4 = _mm_mul_ps(self.twiddle8im, x425n);
        let temp_b2_5 = _mm_mul_ps(self.twiddle10im, x524n);
        let temp_b2_6 = _mm_mul_ps(self.twiddle12im, x623n);
        let temp_b2_7 = _mm_mul_ps(self.twiddle14im, x722n);
        let temp_b2_8 = _mm_mul_ps(self.twiddle13im, x821n);
        let temp_b2_9 = _mm_mul_ps(self.twiddle11im, x920n);
        let temp_b2_10 = _mm_mul_ps(self.twiddle9im, x1019n);
        let temp_b2_11 = _mm_mul_ps(self.twiddle7im, x1118n);
        let temp_b2_12 = _mm_mul_ps(self.twiddle5im, x1217n);
        let temp_b2_13 = _mm_mul_ps(self.twiddle3im, x1316n);
        let temp_b2_14 = _mm_mul_ps(self.twiddle1im, x1415n);
        let temp_b3_1 = _mm_mul_ps(self.twiddle3im, x128n);
        let temp_b3_2 = _mm_mul_ps(self.twiddle6im, x227n);
        let temp_b3_3 = _mm_mul_ps(self.twiddle9im, x326n);
        let temp_b3_4 = _mm_mul_ps(self.twiddle12im, x425n);
        let temp_b3_5 = _mm_mul_ps(self.twiddle14im, x524n);
        let temp_b3_6 = _mm_mul_ps(self.twiddle11im, x623n);
        let temp_b3_7 = _mm_mul_ps(self.twiddle8im, x722n);
        let temp_b3_8 = _mm_mul_ps(self.twiddle5im, x821n);
        let temp_b3_9 = _mm_mul_ps(self.twiddle2im, x920n);
        let temp_b3_10 = _mm_mul_ps(self.twiddle1im, x1019n);
        let temp_b3_11 = _mm_mul_ps(self.twiddle4im, x1118n);
        let temp_b3_12 = _mm_mul_ps(self.twiddle7im, x1217n);
        let temp_b3_13 = _mm_mul_ps(self.twiddle10im, x1316n);
        let temp_b3_14 = _mm_mul_ps(self.twiddle13im, x1415n);
        let temp_b4_1 = _mm_mul_ps(self.twiddle4im, x128n);
        let temp_b4_2 = _mm_mul_ps(self.twiddle8im, x227n);
        let temp_b4_3 = _mm_mul_ps(self.twiddle12im, x326n);
        let temp_b4_4 = _mm_mul_ps(self.twiddle13im, x425n);
        let temp_b4_5 = _mm_mul_ps(self.twiddle9im, x524n);
        let temp_b4_6 = _mm_mul_ps(self.twiddle5im, x623n);
        let temp_b4_7 = _mm_mul_ps(self.twiddle1im, x722n);
        let temp_b4_8 = _mm_mul_ps(self.twiddle3im, x821n);
        let temp_b4_9 = _mm_mul_ps(self.twiddle7im, x920n);
        let temp_b4_10 = _mm_mul_ps(self.twiddle11im, x1019n);
        let temp_b4_11 = _mm_mul_ps(self.twiddle14im, x1118n);
        let temp_b4_12 = _mm_mul_ps(self.twiddle10im, x1217n);
        let temp_b4_13 = _mm_mul_ps(self.twiddle6im, x1316n);
        let temp_b4_14 = _mm_mul_ps(self.twiddle2im, x1415n);
        let temp_b5_1 = _mm_mul_ps(self.twiddle5im, x128n);
        let temp_b5_2 = _mm_mul_ps(self.twiddle10im, x227n);
        let temp_b5_3 = _mm_mul_ps(self.twiddle14im, x326n);
        let temp_b5_4 = _mm_mul_ps(self.twiddle9im, x425n);
        let temp_b5_5 = _mm_mul_ps(self.twiddle4im, x524n);
        let temp_b5_6 = _mm_mul_ps(self.twiddle1im, x623n);
        let temp_b5_7 = _mm_mul_ps(self.twiddle6im, x722n);
        let temp_b5_8 = _mm_mul_ps(self.twiddle11im, x821n);
        let temp_b5_9 = _mm_mul_ps(self.twiddle13im, x920n);
        let temp_b5_10 = _mm_mul_ps(self.twiddle8im, x1019n);
        let temp_b5_11 = _mm_mul_ps(self.twiddle3im, x1118n);
        let temp_b5_12 = _mm_mul_ps(self.twiddle2im, x1217n);
        let temp_b5_13 = _mm_mul_ps(self.twiddle7im, x1316n);
        let temp_b5_14 = _mm_mul_ps(self.twiddle12im, x1415n);
        let temp_b6_1 = _mm_mul_ps(self.twiddle6im, x128n);
        let temp_b6_2 = _mm_mul_ps(self.twiddle12im, x227n);
        let temp_b6_3 = _mm_mul_ps(self.twiddle11im, x326n);
        let temp_b6_4 = _mm_mul_ps(self.twiddle5im, x425n);
        let temp_b6_5 = _mm_mul_ps(self.twiddle1im, x524n);
        let temp_b6_6 = _mm_mul_ps(self.twiddle7im, x623n);
        let temp_b6_7 = _mm_mul_ps(self.twiddle13im, x722n);
        let temp_b6_8 = _mm_mul_ps(self.twiddle10im, x821n);
        let temp_b6_9 = _mm_mul_ps(self.twiddle4im, x920n);
        let temp_b6_10 = _mm_mul_ps(self.twiddle2im, x1019n);
        let temp_b6_11 = _mm_mul_ps(self.twiddle8im, x1118n);
        let temp_b6_12 = _mm_mul_ps(self.twiddle14im, x1217n);
        let temp_b6_13 = _mm_mul_ps(self.twiddle9im, x1316n);
        let temp_b6_14 = _mm_mul_ps(self.twiddle3im, x1415n);
        let temp_b7_1 = _mm_mul_ps(self.twiddle7im, x128n);
        let temp_b7_2 = _mm_mul_ps(self.twiddle14im, x227n);
        let temp_b7_3 = _mm_mul_ps(self.twiddle8im, x326n);
        let temp_b7_4 = _mm_mul_ps(self.twiddle1im, x425n);
        let temp_b7_5 = _mm_mul_ps(self.twiddle6im, x524n);
        let temp_b7_6 = _mm_mul_ps(self.twiddle13im, x623n);
        let temp_b7_7 = _mm_mul_ps(self.twiddle9im, x722n);
        let temp_b7_8 = _mm_mul_ps(self.twiddle2im, x821n);
        let temp_b7_9 = _mm_mul_ps(self.twiddle5im, x920n);
        let temp_b7_10 = _mm_mul_ps(self.twiddle12im, x1019n);
        let temp_b7_11 = _mm_mul_ps(self.twiddle10im, x1118n);
        let temp_b7_12 = _mm_mul_ps(self.twiddle3im, x1217n);
        let temp_b7_13 = _mm_mul_ps(self.twiddle4im, x1316n);
        let temp_b7_14 = _mm_mul_ps(self.twiddle11im, x1415n);
        let temp_b8_1 = _mm_mul_ps(self.twiddle8im, x128n);
        let temp_b8_2 = _mm_mul_ps(self.twiddle13im, x227n);
        let temp_b8_3 = _mm_mul_ps(self.twiddle5im, x326n);
        let temp_b8_4 = _mm_mul_ps(self.twiddle3im, x425n);
        let temp_b8_5 = _mm_mul_ps(self.twiddle11im, x524n);
        let temp_b8_6 = _mm_mul_ps(self.twiddle10im, x623n);
        let temp_b8_7 = _mm_mul_ps(self.twiddle2im, x722n);
        let temp_b8_8 = _mm_mul_ps(self.twiddle6im, x821n);
        let temp_b8_9 = _mm_mul_ps(self.twiddle14im, x920n);
        let temp_b8_10 = _mm_mul_ps(self.twiddle7im, x1019n);
        let temp_b8_11 = _mm_mul_ps(self.twiddle1im, x1118n);
        let temp_b8_12 = _mm_mul_ps(self.twiddle9im, x1217n);
        let temp_b8_13 = _mm_mul_ps(self.twiddle12im, x1316n);
        let temp_b8_14 = _mm_mul_ps(self.twiddle4im, x1415n);
        let temp_b9_1 = _mm_mul_ps(self.twiddle9im, x128n);
        let temp_b9_2 = _mm_mul_ps(self.twiddle11im, x227n);
        let temp_b9_3 = _mm_mul_ps(self.twiddle2im, x326n);
        let temp_b9_4 = _mm_mul_ps(self.twiddle7im, x425n);
        let temp_b9_5 = _mm_mul_ps(self.twiddle13im, x524n);
        let temp_b9_6 = _mm_mul_ps(self.twiddle4im, x623n);
        let temp_b9_7 = _mm_mul_ps(self.twiddle5im, x722n);
        let temp_b9_8 = _mm_mul_ps(self.twiddle14im, x821n);
        let temp_b9_9 = _mm_mul_ps(self.twiddle6im, x920n);
        let temp_b9_10 = _mm_mul_ps(self.twiddle3im, x1019n);
        let temp_b9_11 = _mm_mul_ps(self.twiddle12im, x1118n);
        let temp_b9_12 = _mm_mul_ps(self.twiddle8im, x1217n);
        let temp_b9_13 = _mm_mul_ps(self.twiddle1im, x1316n);
        let temp_b9_14 = _mm_mul_ps(self.twiddle10im, x1415n);
        let temp_b10_1 = _mm_mul_ps(self.twiddle10im, x128n);
        let temp_b10_2 = _mm_mul_ps(self.twiddle9im, x227n);
        let temp_b10_3 = _mm_mul_ps(self.twiddle1im, x326n);
        let temp_b10_4 = _mm_mul_ps(self.twiddle11im, x425n);
        let temp_b10_5 = _mm_mul_ps(self.twiddle8im, x524n);
        let temp_b10_6 = _mm_mul_ps(self.twiddle2im, x623n);
        let temp_b10_7 = _mm_mul_ps(self.twiddle12im, x722n);
        let temp_b10_8 = _mm_mul_ps(self.twiddle7im, x821n);
        let temp_b10_9 = _mm_mul_ps(self.twiddle3im, x920n);
        let temp_b10_10 = _mm_mul_ps(self.twiddle13im, x1019n);
        let temp_b10_11 = _mm_mul_ps(self.twiddle6im, x1118n);
        let temp_b10_12 = _mm_mul_ps(self.twiddle4im, x1217n);
        let temp_b10_13 = _mm_mul_ps(self.twiddle14im, x1316n);
        let temp_b10_14 = _mm_mul_ps(self.twiddle5im, x1415n);
        let temp_b11_1 = _mm_mul_ps(self.twiddle11im, x128n);
        let temp_b11_2 = _mm_mul_ps(self.twiddle7im, x227n);
        let temp_b11_3 = _mm_mul_ps(self.twiddle4im, x326n);
        let temp_b11_4 = _mm_mul_ps(self.twiddle14im, x425n);
        let temp_b11_5 = _mm_mul_ps(self.twiddle3im, x524n);
        let temp_b11_6 = _mm_mul_ps(self.twiddle8im, x623n);
        let temp_b11_7 = _mm_mul_ps(self.twiddle10im, x722n);
        let temp_b11_8 = _mm_mul_ps(self.twiddle1im, x821n);
        let temp_b11_9 = _mm_mul_ps(self.twiddle12im, x920n);
        let temp_b11_10 = _mm_mul_ps(self.twiddle6im, x1019n);
        let temp_b11_11 = _mm_mul_ps(self.twiddle5im, x1118n);
        let temp_b11_12 = _mm_mul_ps(self.twiddle13im, x1217n);
        let temp_b11_13 = _mm_mul_ps(self.twiddle2im, x1316n);
        let temp_b11_14 = _mm_mul_ps(self.twiddle9im, x1415n);
        let temp_b12_1 = _mm_mul_ps(self.twiddle12im, x128n);
        let temp_b12_2 = _mm_mul_ps(self.twiddle5im, x227n);
        let temp_b12_3 = _mm_mul_ps(self.twiddle7im, x326n);
        let temp_b12_4 = _mm_mul_ps(self.twiddle10im, x425n);
        let temp_b12_5 = _mm_mul_ps(self.twiddle2im, x524n);
        let temp_b12_6 = _mm_mul_ps(self.twiddle14im, x623n);
        let temp_b12_7 = _mm_mul_ps(self.twiddle3im, x722n);
        let temp_b12_8 = _mm_mul_ps(self.twiddle9im, x821n);
        let temp_b12_9 = _mm_mul_ps(self.twiddle8im, x920n);
        let temp_b12_10 = _mm_mul_ps(self.twiddle4im, x1019n);
        let temp_b12_11 = _mm_mul_ps(self.twiddle13im, x1118n);
        let temp_b12_12 = _mm_mul_ps(self.twiddle1im, x1217n);
        let temp_b12_13 = _mm_mul_ps(self.twiddle11im, x1316n);
        let temp_b12_14 = _mm_mul_ps(self.twiddle6im, x1415n);
        let temp_b13_1 = _mm_mul_ps(self.twiddle13im, x128n);
        let temp_b13_2 = _mm_mul_ps(self.twiddle3im, x227n);
        let temp_b13_3 = _mm_mul_ps(self.twiddle10im, x326n);
        let temp_b13_4 = _mm_mul_ps(self.twiddle6im, x425n);
        let temp_b13_5 = _mm_mul_ps(self.twiddle7im, x524n);
        let temp_b13_6 = _mm_mul_ps(self.twiddle9im, x623n);
        let temp_b13_7 = _mm_mul_ps(self.twiddle4im, x722n);
        let temp_b13_8 = _mm_mul_ps(self.twiddle12im, x821n);
        let temp_b13_9 = _mm_mul_ps(self.twiddle1im, x920n);
        let temp_b13_10 = _mm_mul_ps(self.twiddle14im, x1019n);
        let temp_b13_11 = _mm_mul_ps(self.twiddle2im, x1118n);
        let temp_b13_12 = _mm_mul_ps(self.twiddle11im, x1217n);
        let temp_b13_13 = _mm_mul_ps(self.twiddle5im, x1316n);
        let temp_b13_14 = _mm_mul_ps(self.twiddle8im, x1415n);
        let temp_b14_1 = _mm_mul_ps(self.twiddle14im, x128n);
        let temp_b14_2 = _mm_mul_ps(self.twiddle1im, x227n);
        let temp_b14_3 = _mm_mul_ps(self.twiddle13im, x326n);
        let temp_b14_4 = _mm_mul_ps(self.twiddle2im, x425n);
        let temp_b14_5 = _mm_mul_ps(self.twiddle12im, x524n);
        let temp_b14_6 = _mm_mul_ps(self.twiddle3im, x623n);
        let temp_b14_7 = _mm_mul_ps(self.twiddle11im, x722n);
        let temp_b14_8 = _mm_mul_ps(self.twiddle4im, x821n);
        let temp_b14_9 = _mm_mul_ps(self.twiddle10im, x920n);
        let temp_b14_10 = _mm_mul_ps(self.twiddle5im, x1019n);
        let temp_b14_11 = _mm_mul_ps(self.twiddle9im, x1118n);
        let temp_b14_12 = _mm_mul_ps(self.twiddle6im, x1217n);
        let temp_b14_13 = _mm_mul_ps(self.twiddle8im, x1316n);
        let temp_b14_14 = _mm_mul_ps(self.twiddle7im, x1415n);

        let temp_a1 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a1_1,
                _mm_add_ps(
                    temp_a1_2,
                    _mm_add_ps(
                        temp_a1_3,
                        _mm_add_ps(
                            temp_a1_4,
                            _mm_add_ps(
                                temp_a1_5,
                                _mm_add_ps(
                                    temp_a1_6,
                                    _mm_add_ps(
                                        temp_a1_7,
                                        _mm_add_ps(
                                            temp_a1_8,
                                            _mm_add_ps(
                                                temp_a1_9,
                                                _mm_add_ps(
                                                    temp_a1_10,
                                                    _mm_add_ps(
                                                        temp_a1_11,
                                                        _mm_add_ps(
                                                            temp_a1_12,
                                                            _mm_add_ps(temp_a1_13, temp_a1_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a2_1,
                _mm_add_ps(
                    temp_a2_2,
                    _mm_add_ps(
                        temp_a2_3,
                        _mm_add_ps(
                            temp_a2_4,
                            _mm_add_ps(
                                temp_a2_5,
                                _mm_add_ps(
                                    temp_a2_6,
                                    _mm_add_ps(
                                        temp_a2_7,
                                        _mm_add_ps(
                                            temp_a2_8,
                                            _mm_add_ps(
                                                temp_a2_9,
                                                _mm_add_ps(
                                                    temp_a2_10,
                                                    _mm_add_ps(
                                                        temp_a2_11,
                                                        _mm_add_ps(
                                                            temp_a2_12,
                                                            _mm_add_ps(temp_a2_13, temp_a2_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a3_1,
                _mm_add_ps(
                    temp_a3_2,
                    _mm_add_ps(
                        temp_a3_3,
                        _mm_add_ps(
                            temp_a3_4,
                            _mm_add_ps(
                                temp_a3_5,
                                _mm_add_ps(
                                    temp_a3_6,
                                    _mm_add_ps(
                                        temp_a3_7,
                                        _mm_add_ps(
                                            temp_a3_8,
                                            _mm_add_ps(
                                                temp_a3_9,
                                                _mm_add_ps(
                                                    temp_a3_10,
                                                    _mm_add_ps(
                                                        temp_a3_11,
                                                        _mm_add_ps(
                                                            temp_a3_12,
                                                            _mm_add_ps(temp_a3_13, temp_a3_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a4_1,
                _mm_add_ps(
                    temp_a4_2,
                    _mm_add_ps(
                        temp_a4_3,
                        _mm_add_ps(
                            temp_a4_4,
                            _mm_add_ps(
                                temp_a4_5,
                                _mm_add_ps(
                                    temp_a4_6,
                                    _mm_add_ps(
                                        temp_a4_7,
                                        _mm_add_ps(
                                            temp_a4_8,
                                            _mm_add_ps(
                                                temp_a4_9,
                                                _mm_add_ps(
                                                    temp_a4_10,
                                                    _mm_add_ps(
                                                        temp_a4_11,
                                                        _mm_add_ps(
                                                            temp_a4_12,
                                                            _mm_add_ps(temp_a4_13, temp_a4_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a5_1,
                _mm_add_ps(
                    temp_a5_2,
                    _mm_add_ps(
                        temp_a5_3,
                        _mm_add_ps(
                            temp_a5_4,
                            _mm_add_ps(
                                temp_a5_5,
                                _mm_add_ps(
                                    temp_a5_6,
                                    _mm_add_ps(
                                        temp_a5_7,
                                        _mm_add_ps(
                                            temp_a5_8,
                                            _mm_add_ps(
                                                temp_a5_9,
                                                _mm_add_ps(
                                                    temp_a5_10,
                                                    _mm_add_ps(
                                                        temp_a5_11,
                                                        _mm_add_ps(
                                                            temp_a5_12,
                                                            _mm_add_ps(temp_a5_13, temp_a5_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a6_1,
                _mm_add_ps(
                    temp_a6_2,
                    _mm_add_ps(
                        temp_a6_3,
                        _mm_add_ps(
                            temp_a6_4,
                            _mm_add_ps(
                                temp_a6_5,
                                _mm_add_ps(
                                    temp_a6_6,
                                    _mm_add_ps(
                                        temp_a6_7,
                                        _mm_add_ps(
                                            temp_a6_8,
                                            _mm_add_ps(
                                                temp_a6_9,
                                                _mm_add_ps(
                                                    temp_a6_10,
                                                    _mm_add_ps(
                                                        temp_a6_11,
                                                        _mm_add_ps(
                                                            temp_a6_12,
                                                            _mm_add_ps(temp_a6_13, temp_a6_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a7_1,
                _mm_add_ps(
                    temp_a7_2,
                    _mm_add_ps(
                        temp_a7_3,
                        _mm_add_ps(
                            temp_a7_4,
                            _mm_add_ps(
                                temp_a7_5,
                                _mm_add_ps(
                                    temp_a7_6,
                                    _mm_add_ps(
                                        temp_a7_7,
                                        _mm_add_ps(
                                            temp_a7_8,
                                            _mm_add_ps(
                                                temp_a7_9,
                                                _mm_add_ps(
                                                    temp_a7_10,
                                                    _mm_add_ps(
                                                        temp_a7_11,
                                                        _mm_add_ps(
                                                            temp_a7_12,
                                                            _mm_add_ps(temp_a7_13, temp_a7_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a8_1,
                _mm_add_ps(
                    temp_a8_2,
                    _mm_add_ps(
                        temp_a8_3,
                        _mm_add_ps(
                            temp_a8_4,
                            _mm_add_ps(
                                temp_a8_5,
                                _mm_add_ps(
                                    temp_a8_6,
                                    _mm_add_ps(
                                        temp_a8_7,
                                        _mm_add_ps(
                                            temp_a8_8,
                                            _mm_add_ps(
                                                temp_a8_9,
                                                _mm_add_ps(
                                                    temp_a8_10,
                                                    _mm_add_ps(
                                                        temp_a8_11,
                                                        _mm_add_ps(
                                                            temp_a8_12,
                                                            _mm_add_ps(temp_a8_13, temp_a8_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a9 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a9_1,
                _mm_add_ps(
                    temp_a9_2,
                    _mm_add_ps(
                        temp_a9_3,
                        _mm_add_ps(
                            temp_a9_4,
                            _mm_add_ps(
                                temp_a9_5,
                                _mm_add_ps(
                                    temp_a9_6,
                                    _mm_add_ps(
                                        temp_a9_7,
                                        _mm_add_ps(
                                            temp_a9_8,
                                            _mm_add_ps(
                                                temp_a9_9,
                                                _mm_add_ps(
                                                    temp_a9_10,
                                                    _mm_add_ps(
                                                        temp_a9_11,
                                                        _mm_add_ps(
                                                            temp_a9_12,
                                                            _mm_add_ps(temp_a9_13, temp_a9_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a10 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a10_1,
                _mm_add_ps(
                    temp_a10_2,
                    _mm_add_ps(
                        temp_a10_3,
                        _mm_add_ps(
                            temp_a10_4,
                            _mm_add_ps(
                                temp_a10_5,
                                _mm_add_ps(
                                    temp_a10_6,
                                    _mm_add_ps(
                                        temp_a10_7,
                                        _mm_add_ps(
                                            temp_a10_8,
                                            _mm_add_ps(
                                                temp_a10_9,
                                                _mm_add_ps(
                                                    temp_a10_10,
                                                    _mm_add_ps(
                                                        temp_a10_11,
                                                        _mm_add_ps(
                                                            temp_a10_12,
                                                            _mm_add_ps(temp_a10_13, temp_a10_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a11 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a11_1,
                _mm_add_ps(
                    temp_a11_2,
                    _mm_add_ps(
                        temp_a11_3,
                        _mm_add_ps(
                            temp_a11_4,
                            _mm_add_ps(
                                temp_a11_5,
                                _mm_add_ps(
                                    temp_a11_6,
                                    _mm_add_ps(
                                        temp_a11_7,
                                        _mm_add_ps(
                                            temp_a11_8,
                                            _mm_add_ps(
                                                temp_a11_9,
                                                _mm_add_ps(
                                                    temp_a11_10,
                                                    _mm_add_ps(
                                                        temp_a11_11,
                                                        _mm_add_ps(
                                                            temp_a11_12,
                                                            _mm_add_ps(temp_a11_13, temp_a11_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a12 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a12_1,
                _mm_add_ps(
                    temp_a12_2,
                    _mm_add_ps(
                        temp_a12_3,
                        _mm_add_ps(
                            temp_a12_4,
                            _mm_add_ps(
                                temp_a12_5,
                                _mm_add_ps(
                                    temp_a12_6,
                                    _mm_add_ps(
                                        temp_a12_7,
                                        _mm_add_ps(
                                            temp_a12_8,
                                            _mm_add_ps(
                                                temp_a12_9,
                                                _mm_add_ps(
                                                    temp_a12_10,
                                                    _mm_add_ps(
                                                        temp_a12_11,
                                                        _mm_add_ps(
                                                            temp_a12_12,
                                                            _mm_add_ps(temp_a12_13, temp_a12_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a13 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a13_1,
                _mm_add_ps(
                    temp_a13_2,
                    _mm_add_ps(
                        temp_a13_3,
                        _mm_add_ps(
                            temp_a13_4,
                            _mm_add_ps(
                                temp_a13_5,
                                _mm_add_ps(
                                    temp_a13_6,
                                    _mm_add_ps(
                                        temp_a13_7,
                                        _mm_add_ps(
                                            temp_a13_8,
                                            _mm_add_ps(
                                                temp_a13_9,
                                                _mm_add_ps(
                                                    temp_a13_10,
                                                    _mm_add_ps(
                                                        temp_a13_11,
                                                        _mm_add_ps(
                                                            temp_a13_12,
                                                            _mm_add_ps(temp_a13_13, temp_a13_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a14 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a14_1,
                _mm_add_ps(
                    temp_a14_2,
                    _mm_add_ps(
                        temp_a14_3,
                        _mm_add_ps(
                            temp_a14_4,
                            _mm_add_ps(
                                temp_a14_5,
                                _mm_add_ps(
                                    temp_a14_6,
                                    _mm_add_ps(
                                        temp_a14_7,
                                        _mm_add_ps(
                                            temp_a14_8,
                                            _mm_add_ps(
                                                temp_a14_9,
                                                _mm_add_ps(
                                                    temp_a14_10,
                                                    _mm_add_ps(
                                                        temp_a14_11,
                                                        _mm_add_ps(
                                                            temp_a14_12,
                                                            _mm_add_ps(temp_a14_13, temp_a14_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_ps(
            temp_b1_1,
            _mm_add_ps(
                temp_b1_2,
                _mm_add_ps(
                    temp_b1_3,
                    _mm_add_ps(
                        temp_b1_4,
                        _mm_add_ps(
                            temp_b1_5,
                            _mm_add_ps(
                                temp_b1_6,
                                _mm_add_ps(
                                    temp_b1_7,
                                    _mm_add_ps(
                                        temp_b1_8,
                                        _mm_add_ps(
                                            temp_b1_9,
                                            _mm_add_ps(
                                                temp_b1_10,
                                                _mm_add_ps(
                                                    temp_b1_11,
                                                    _mm_add_ps(
                                                        temp_b1_12,
                                                        _mm_add_ps(temp_b1_13, temp_b1_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_ps(
            temp_b2_1,
            _mm_add_ps(
                temp_b2_2,
                _mm_add_ps(
                    temp_b2_3,
                    _mm_add_ps(
                        temp_b2_4,
                        _mm_add_ps(
                            temp_b2_5,
                            _mm_add_ps(
                                temp_b2_6,
                                _mm_sub_ps(
                                    temp_b2_7,
                                    _mm_add_ps(
                                        temp_b2_8,
                                        _mm_add_ps(
                                            temp_b2_9,
                                            _mm_add_ps(
                                                temp_b2_10,
                                                _mm_add_ps(
                                                    temp_b2_11,
                                                    _mm_add_ps(
                                                        temp_b2_12,
                                                        _mm_add_ps(temp_b2_13, temp_b2_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_ps(
            temp_b3_1,
            _mm_add_ps(
                temp_b3_2,
                _mm_add_ps(
                    temp_b3_3,
                    _mm_sub_ps(
                        temp_b3_4,
                        _mm_add_ps(
                            temp_b3_5,
                            _mm_add_ps(
                                temp_b3_6,
                                _mm_add_ps(
                                    temp_b3_7,
                                    _mm_add_ps(
                                        temp_b3_8,
                                        _mm_sub_ps(
                                            temp_b3_9,
                                            _mm_add_ps(
                                                temp_b3_10,
                                                _mm_add_ps(
                                                    temp_b3_11,
                                                    _mm_add_ps(
                                                        temp_b3_12,
                                                        _mm_add_ps(temp_b3_13, temp_b3_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_ps(
            temp_b4_1,
            _mm_add_ps(
                temp_b4_2,
                _mm_sub_ps(
                    temp_b4_3,
                    _mm_add_ps(
                        temp_b4_4,
                        _mm_add_ps(
                            temp_b4_5,
                            _mm_add_ps(
                                temp_b4_6,
                                _mm_sub_ps(
                                    temp_b4_7,
                                    _mm_add_ps(
                                        temp_b4_8,
                                        _mm_add_ps(
                                            temp_b4_9,
                                            _mm_sub_ps(
                                                temp_b4_10,
                                                _mm_add_ps(
                                                    temp_b4_11,
                                                    _mm_add_ps(
                                                        temp_b4_12,
                                                        _mm_add_ps(temp_b4_13, temp_b4_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_add_ps(
            temp_b5_1,
            _mm_sub_ps(
                temp_b5_2,
                _mm_add_ps(
                    temp_b5_3,
                    _mm_add_ps(
                        temp_b5_4,
                        _mm_sub_ps(
                            temp_b5_5,
                            _mm_add_ps(
                                temp_b5_6,
                                _mm_add_ps(
                                    temp_b5_7,
                                    _mm_sub_ps(
                                        temp_b5_8,
                                        _mm_add_ps(
                                            temp_b5_9,
                                            _mm_add_ps(
                                                temp_b5_10,
                                                _mm_sub_ps(
                                                    temp_b5_11,
                                                    _mm_add_ps(
                                                        temp_b5_12,
                                                        _mm_add_ps(temp_b5_13, temp_b5_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_add_ps(
            temp_b6_1,
            _mm_sub_ps(
                temp_b6_2,
                _mm_add_ps(
                    temp_b6_3,
                    _mm_sub_ps(
                        temp_b6_4,
                        _mm_add_ps(
                            temp_b6_5,
                            _mm_add_ps(
                                temp_b6_6,
                                _mm_sub_ps(
                                    temp_b6_7,
                                    _mm_add_ps(
                                        temp_b6_8,
                                        _mm_sub_ps(
                                            temp_b6_9,
                                            _mm_add_ps(
                                                temp_b6_10,
                                                _mm_add_ps(
                                                    temp_b6_11,
                                                    _mm_sub_ps(
                                                        temp_b6_12,
                                                        _mm_add_ps(temp_b6_13, temp_b6_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_add_ps(
            temp_b7_1,
            _mm_sub_ps(
                temp_b7_2,
                _mm_add_ps(
                    temp_b7_3,
                    _mm_sub_ps(
                        temp_b7_4,
                        _mm_add_ps(
                            temp_b7_5,
                            _mm_sub_ps(
                                temp_b7_6,
                                _mm_add_ps(
                                    temp_b7_7,
                                    _mm_sub_ps(
                                        temp_b7_8,
                                        _mm_add_ps(
                                            temp_b7_9,
                                            _mm_sub_ps(
                                                temp_b7_10,
                                                _mm_add_ps(
                                                    temp_b7_11,
                                                    _mm_sub_ps(
                                                        temp_b7_12,
                                                        _mm_add_ps(temp_b7_13, temp_b7_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_ps(
            temp_b8_1,
            _mm_add_ps(
                temp_b8_2,
                _mm_sub_ps(
                    temp_b8_3,
                    _mm_add_ps(
                        temp_b8_4,
                        _mm_sub_ps(
                            temp_b8_5,
                            _mm_add_ps(
                                temp_b8_6,
                                _mm_sub_ps(
                                    temp_b8_7,
                                    _mm_add_ps(
                                        temp_b8_8,
                                        _mm_sub_ps(
                                            temp_b8_9,
                                            _mm_sub_ps(
                                                temp_b8_10,
                                                _mm_add_ps(
                                                    temp_b8_11,
                                                    _mm_sub_ps(
                                                        temp_b8_12,
                                                        _mm_add_ps(temp_b8_13, temp_b8_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b9 = _mm_sub_ps(
            temp_b9_1,
            _mm_add_ps(
                temp_b9_2,
                _mm_sub_ps(
                    temp_b9_3,
                    _mm_sub_ps(
                        temp_b9_4,
                        _mm_add_ps(
                            temp_b9_5,
                            _mm_sub_ps(
                                temp_b9_6,
                                _mm_add_ps(
                                    temp_b9_7,
                                    _mm_sub_ps(
                                        temp_b9_8,
                                        _mm_sub_ps(
                                            temp_b9_9,
                                            _mm_add_ps(
                                                temp_b9_10,
                                                _mm_sub_ps(
                                                    temp_b9_11,
                                                    _mm_sub_ps(
                                                        temp_b9_12,
                                                        _mm_add_ps(temp_b9_13, temp_b9_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b10 = _mm_sub_ps(
            temp_b10_1,
            _mm_sub_ps(
                temp_b10_2,
                _mm_add_ps(
                    temp_b10_3,
                    _mm_sub_ps(
                        temp_b10_4,
                        _mm_sub_ps(
                            temp_b10_5,
                            _mm_add_ps(
                                temp_b10_6,
                                _mm_sub_ps(
                                    temp_b10_7,
                                    _mm_sub_ps(
                                        temp_b10_8,
                                        _mm_add_ps(
                                            temp_b10_9,
                                            _mm_sub_ps(
                                                temp_b10_10,
                                                _mm_sub_ps(
                                                    temp_b10_11,
                                                    _mm_add_ps(
                                                        temp_b10_12,
                                                        _mm_sub_ps(temp_b10_13, temp_b10_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b11 = _mm_sub_ps(
            temp_b11_1,
            _mm_sub_ps(
                temp_b11_2,
                _mm_sub_ps(
                    temp_b11_3,
                    _mm_add_ps(
                        temp_b11_4,
                        _mm_sub_ps(
                            temp_b11_5,
                            _mm_sub_ps(
                                temp_b11_6,
                                _mm_sub_ps(
                                    temp_b11_7,
                                    _mm_add_ps(
                                        temp_b11_8,
                                        _mm_sub_ps(
                                            temp_b11_9,
                                            _mm_sub_ps(
                                                temp_b11_10,
                                                _mm_sub_ps(
                                                    temp_b11_11,
                                                    _mm_add_ps(
                                                        temp_b11_12,
                                                        _mm_sub_ps(temp_b11_13, temp_b11_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b12 = _mm_sub_ps(
            temp_b12_1,
            _mm_sub_ps(
                temp_b12_2,
                _mm_sub_ps(
                    temp_b12_3,
                    _mm_sub_ps(
                        temp_b12_4,
                        _mm_add_ps(
                            temp_b12_5,
                            _mm_sub_ps(
                                temp_b12_6,
                                _mm_sub_ps(
                                    temp_b12_7,
                                    _mm_sub_ps(
                                        temp_b12_8,
                                        _mm_sub_ps(
                                            temp_b12_9,
                                            _mm_sub_ps(
                                                temp_b12_10,
                                                _mm_add_ps(
                                                    temp_b12_11,
                                                    _mm_sub_ps(
                                                        temp_b12_12,
                                                        _mm_sub_ps(temp_b12_13, temp_b12_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b13 = _mm_sub_ps(
            temp_b13_1,
            _mm_sub_ps(
                temp_b13_2,
                _mm_sub_ps(
                    temp_b13_3,
                    _mm_sub_ps(
                        temp_b13_4,
                        _mm_sub_ps(
                            temp_b13_5,
                            _mm_sub_ps(
                                temp_b13_6,
                                _mm_sub_ps(
                                    temp_b13_7,
                                    _mm_sub_ps(
                                        temp_b13_8,
                                        _mm_add_ps(
                                            temp_b13_9,
                                            _mm_sub_ps(
                                                temp_b13_10,
                                                _mm_sub_ps(
                                                    temp_b13_11,
                                                    _mm_sub_ps(
                                                        temp_b13_12,
                                                        _mm_sub_ps(temp_b13_13, temp_b13_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b14 = _mm_sub_ps(
            temp_b14_1,
            _mm_sub_ps(
                temp_b14_2,
                _mm_sub_ps(
                    temp_b14_3,
                    _mm_sub_ps(
                        temp_b14_4,
                        _mm_sub_ps(
                            temp_b14_5,
                            _mm_sub_ps(
                                temp_b14_6,
                                _mm_sub_ps(
                                    temp_b14_7,
                                    _mm_sub_ps(
                                        temp_b14_8,
                                        _mm_sub_ps(
                                            temp_b14_9,
                                            _mm_sub_ps(
                                                temp_b14_10,
                                                _mm_sub_ps(
                                                    temp_b14_11,
                                                    _mm_sub_ps(
                                                        temp_b14_12,
                                                        _mm_sub_ps(temp_b14_13, temp_b14_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate_both(temp_b1);
        let temp_b2_rot = self.rotate.rotate_both(temp_b2);
        let temp_b3_rot = self.rotate.rotate_both(temp_b3);
        let temp_b4_rot = self.rotate.rotate_both(temp_b4);
        let temp_b5_rot = self.rotate.rotate_both(temp_b5);
        let temp_b6_rot = self.rotate.rotate_both(temp_b6);
        let temp_b7_rot = self.rotate.rotate_both(temp_b7);
        let temp_b8_rot = self.rotate.rotate_both(temp_b8);
        let temp_b9_rot = self.rotate.rotate_both(temp_b9);
        let temp_b10_rot = self.rotate.rotate_both(temp_b10);
        let temp_b11_rot = self.rotate.rotate_both(temp_b11);
        let temp_b12_rot = self.rotate.rotate_both(temp_b12);
        let temp_b13_rot = self.rotate.rotate_both(temp_b13);
        let temp_b14_rot = self.rotate.rotate_both(temp_b14);

        let x0 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                x128p,
                _mm_add_ps(
                    x227p,
                    _mm_add_ps(
                        x326p,
                        _mm_add_ps(
                            x425p,
                            _mm_add_ps(
                                x524p,
                                _mm_add_ps(
                                    x623p,
                                    _mm_add_ps(
                                        x722p,
                                        _mm_add_ps(
                                            x821p,
                                            _mm_add_ps(
                                                x920p,
                                                _mm_add_ps(
                                                    x1019p,
                                                    _mm_add_ps(
                                                        x1118p,
                                                        _mm_add_ps(
                                                            x1217p,
                                                            _mm_add_ps(x1316p, x1415p),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_ps(temp_a1, temp_b1_rot);
        let x2 = _mm_add_ps(temp_a2, temp_b2_rot);
        let x3 = _mm_add_ps(temp_a3, temp_b3_rot);
        let x4 = _mm_add_ps(temp_a4, temp_b4_rot);
        let x5 = _mm_add_ps(temp_a5, temp_b5_rot);
        let x6 = _mm_add_ps(temp_a6, temp_b6_rot);
        let x7 = _mm_add_ps(temp_a7, temp_b7_rot);
        let x8 = _mm_add_ps(temp_a8, temp_b8_rot);
        let x9 = _mm_add_ps(temp_a9, temp_b9_rot);
        let x10 = _mm_add_ps(temp_a10, temp_b10_rot);
        let x11 = _mm_add_ps(temp_a11, temp_b11_rot);
        let x12 = _mm_add_ps(temp_a12, temp_b12_rot);
        let x13 = _mm_add_ps(temp_a13, temp_b13_rot);
        let x14 = _mm_add_ps(temp_a14, temp_b14_rot);
        let x15 = _mm_sub_ps(temp_a14, temp_b14_rot);
        let x16 = _mm_sub_ps(temp_a13, temp_b13_rot);
        let x17 = _mm_sub_ps(temp_a12, temp_b12_rot);
        let x18 = _mm_sub_ps(temp_a11, temp_b11_rot);
        let x19 = _mm_sub_ps(temp_a10, temp_b10_rot);
        let x20 = _mm_sub_ps(temp_a9, temp_b9_rot);
        let x21 = _mm_sub_ps(temp_a8, temp_b8_rot);
        let x22 = _mm_sub_ps(temp_a7, temp_b7_rot);
        let x23 = _mm_sub_ps(temp_a6, temp_b6_rot);
        let x24 = _mm_sub_ps(temp_a5, temp_b5_rot);
        let x25 = _mm_sub_ps(temp_a4, temp_b4_rot);
        let x26 = _mm_sub_ps(temp_a3, temp_b3_rot);
        let x27 = _mm_sub_ps(temp_a2, temp_b2_rot);
        let x28 = _mm_sub_ps(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
            x19, x20, x21, x22, x23, x24, x25, x26, x27, x28,
        ]
    }
}

//   ____   ___             __   _  _   _     _ _
//  |___ \ / _ \           / /_ | || | | |__ (_) |_
//    __) | (_) |  _____  | '_ \| || |_| '_ \| | __|
//   / __/ \__, | |_____| | (_) |__   _| |_) | | |_
//  |_____|  /_/           \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly29<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
    twiddle3re: __m128d,
    twiddle3im: __m128d,
    twiddle4re: __m128d,
    twiddle4im: __m128d,
    twiddle5re: __m128d,
    twiddle5im: __m128d,
    twiddle6re: __m128d,
    twiddle6im: __m128d,
    twiddle7re: __m128d,
    twiddle7im: __m128d,
    twiddle8re: __m128d,
    twiddle8im: __m128d,
    twiddle9re: __m128d,
    twiddle9im: __m128d,
    twiddle10re: __m128d,
    twiddle10im: __m128d,
    twiddle11re: __m128d,
    twiddle11im: __m128d,
    twiddle12re: __m128d,
    twiddle12im: __m128d,
    twiddle13re: __m128d,
    twiddle13im: __m128d,
    twiddle14re: __m128d,
    twiddle14im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly29, 29, |this: &SseF64Butterfly29<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly29, 29, |this: &SseF64Butterfly29<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly29<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 29, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 29, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 29, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 29, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 29, direction);
        let tw6: Complex<f64> = twiddles::compute_twiddle(6, 29, direction);
        let tw7: Complex<f64> = twiddles::compute_twiddle(7, 29, direction);
        let tw8: Complex<f64> = twiddles::compute_twiddle(8, 29, direction);
        let tw9: Complex<f64> = twiddles::compute_twiddle(9, 29, direction);
        let tw10: Complex<f64> = twiddles::compute_twiddle(10, 29, direction);
        let tw11: Complex<f64> = twiddles::compute_twiddle(11, 29, direction);
        let tw12: Complex<f64> = twiddles::compute_twiddle(12, 29, direction);
        let tw13: Complex<f64> = twiddles::compute_twiddle(13, 29, direction);
        let tw14: Complex<f64> = twiddles::compute_twiddle(14, 29, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_pd(tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_pd(tw3.im, tw3.im) };
        let twiddle4re = unsafe { _mm_set_pd(tw4.re, tw4.re) };
        let twiddle4im = unsafe { _mm_set_pd(tw4.im, tw4.im) };
        let twiddle5re = unsafe { _mm_set_pd(tw5.re, tw5.re) };
        let twiddle5im = unsafe { _mm_set_pd(tw5.im, tw5.im) };
        let twiddle6re = unsafe { _mm_set_pd(tw6.re, tw6.re) };
        let twiddle6im = unsafe { _mm_set_pd(tw6.im, tw6.im) };
        let twiddle7re = unsafe { _mm_set_pd(tw7.re, tw7.re) };
        let twiddle7im = unsafe { _mm_set_pd(tw7.im, tw7.im) };
        let twiddle8re = unsafe { _mm_set_pd(tw8.re, tw8.re) };
        let twiddle8im = unsafe { _mm_set_pd(tw8.im, tw8.im) };
        let twiddle9re = unsafe { _mm_set_pd(tw9.re, tw9.re) };
        let twiddle9im = unsafe { _mm_set_pd(tw9.im, tw9.im) };
        let twiddle10re = unsafe { _mm_set_pd(tw10.re, tw10.re) };
        let twiddle10im = unsafe { _mm_set_pd(tw10.im, tw10.im) };
        let twiddle11re = unsafe { _mm_set_pd(tw11.re, tw11.re) };
        let twiddle11im = unsafe { _mm_set_pd(tw11.im, tw11.im) };
        let twiddle12re = unsafe { _mm_set_pd(tw12.re, tw12.re) };
        let twiddle12im = unsafe { _mm_set_pd(tw12.im, tw12.im) };
        let twiddle13re = unsafe { _mm_set_pd(tw13.re, tw13.re) };
        let twiddle13im = unsafe { _mm_set_pd(tw13.im, tw13.im) };
        let twiddle14re = unsafe { _mm_set_pd(tw14.re, tw14.re) };
        let twiddle14im = unsafe { _mm_set_pd(tw14.im, tw14.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
            twiddle9re,
            twiddle9im,
            twiddle10re,
            twiddle10im,
            twiddle11re,
            twiddle11im,
            twiddle12re,
            twiddle12im,
            twiddle13re,
            twiddle13im,
            twiddle14re,
            twiddle14im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let v1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let v2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let v3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let v4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let v5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let v6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let v7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let v8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let v9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let v10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);
        let v11 = _mm_loadu_pd(input.as_ptr().add(11) as *const f64);
        let v12 = _mm_loadu_pd(input.as_ptr().add(12) as *const f64);
        let v13 = _mm_loadu_pd(input.as_ptr().add(13) as *const f64);
        let v14 = _mm_loadu_pd(input.as_ptr().add(14) as *const f64);
        let v15 = _mm_loadu_pd(input.as_ptr().add(15) as *const f64);
        let v16 = _mm_loadu_pd(input.as_ptr().add(16) as *const f64);
        let v17 = _mm_loadu_pd(input.as_ptr().add(17) as *const f64);
        let v18 = _mm_loadu_pd(input.as_ptr().add(18) as *const f64);
        let v19 = _mm_loadu_pd(input.as_ptr().add(19) as *const f64);
        let v20 = _mm_loadu_pd(input.as_ptr().add(20) as *const f64);
        let v21 = _mm_loadu_pd(input.as_ptr().add(21) as *const f64);
        let v22 = _mm_loadu_pd(input.as_ptr().add(22) as *const f64);
        let v23 = _mm_loadu_pd(input.as_ptr().add(23) as *const f64);
        let v24 = _mm_loadu_pd(input.as_ptr().add(24) as *const f64);
        let v25 = _mm_loadu_pd(input.as_ptr().add(25) as *const f64);
        let v26 = _mm_loadu_pd(input.as_ptr().add(26) as *const f64);
        let v27 = _mm_loadu_pd(input.as_ptr().add(27) as *const f64);
        let v28 = _mm_loadu_pd(input.as_ptr().add(28) as *const f64);

        let out = self.perform_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
        ]);

        let val = std::mem::transmute::<[__m128d; 29], [Complex<f64>; 29]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
        *output_slice.add(6) = val[6];
        *output_slice.add(7) = val[7];
        *output_slice.add(8) = val[8];
        *output_slice.add(9) = val[9];
        *output_slice.add(10) = val[10];
        *output_slice.add(11) = val[11];
        *output_slice.add(12) = val[12];
        *output_slice.add(13) = val[13];
        *output_slice.add(14) = val[14];
        *output_slice.add(15) = val[15];
        *output_slice.add(16) = val[16];
        *output_slice.add(17) = val[17];
        *output_slice.add(18) = val[18];
        *output_slice.add(19) = val[19];
        *output_slice.add(20) = val[20];
        *output_slice.add(21) = val[21];
        *output_slice.add(22) = val[22];
        *output_slice.add(23) = val[23];
        *output_slice.add(24) = val[24];
        *output_slice.add(25) = val[25];
        *output_slice.add(26) = val[26];
        *output_slice.add(27) = val[27];
        *output_slice.add(28) = val[28];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 29]) -> [__m128d; 29] {
        // This is a SSE translation of the scalar 17-point butterfly
        let x128p = _mm_add_pd(values[1], values[28]);
        let x128n = _mm_sub_pd(values[1], values[28]);
        let x227p = _mm_add_pd(values[2], values[27]);
        let x227n = _mm_sub_pd(values[2], values[27]);
        let x326p = _mm_add_pd(values[3], values[26]);
        let x326n = _mm_sub_pd(values[3], values[26]);
        let x425p = _mm_add_pd(values[4], values[25]);
        let x425n = _mm_sub_pd(values[4], values[25]);
        let x524p = _mm_add_pd(values[5], values[24]);
        let x524n = _mm_sub_pd(values[5], values[24]);
        let x623p = _mm_add_pd(values[6], values[23]);
        let x623n = _mm_sub_pd(values[6], values[23]);
        let x722p = _mm_add_pd(values[7], values[22]);
        let x722n = _mm_sub_pd(values[7], values[22]);
        let x821p = _mm_add_pd(values[8], values[21]);
        let x821n = _mm_sub_pd(values[8], values[21]);
        let x920p = _mm_add_pd(values[9], values[20]);
        let x920n = _mm_sub_pd(values[9], values[20]);
        let x1019p = _mm_add_pd(values[10], values[19]);
        let x1019n = _mm_sub_pd(values[10], values[19]);
        let x1118p = _mm_add_pd(values[11], values[18]);
        let x1118n = _mm_sub_pd(values[11], values[18]);
        let x1217p = _mm_add_pd(values[12], values[17]);
        let x1217n = _mm_sub_pd(values[12], values[17]);
        let x1316p = _mm_add_pd(values[13], values[16]);
        let x1316n = _mm_sub_pd(values[13], values[16]);
        let x1415p = _mm_add_pd(values[14], values[15]);
        let x1415n = _mm_sub_pd(values[14], values[15]);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x128p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x227p);
        let temp_a1_3 = _mm_mul_pd(self.twiddle3re, x326p);
        let temp_a1_4 = _mm_mul_pd(self.twiddle4re, x425p);
        let temp_a1_5 = _mm_mul_pd(self.twiddle5re, x524p);
        let temp_a1_6 = _mm_mul_pd(self.twiddle6re, x623p);
        let temp_a1_7 = _mm_mul_pd(self.twiddle7re, x722p);
        let temp_a1_8 = _mm_mul_pd(self.twiddle8re, x821p);
        let temp_a1_9 = _mm_mul_pd(self.twiddle9re, x920p);
        let temp_a1_10 = _mm_mul_pd(self.twiddle10re, x1019p);
        let temp_a1_11 = _mm_mul_pd(self.twiddle11re, x1118p);
        let temp_a1_12 = _mm_mul_pd(self.twiddle12re, x1217p);
        let temp_a1_13 = _mm_mul_pd(self.twiddle13re, x1316p);
        let temp_a1_14 = _mm_mul_pd(self.twiddle14re, x1415p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x128p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle4re, x227p);
        let temp_a2_3 = _mm_mul_pd(self.twiddle6re, x326p);
        let temp_a2_4 = _mm_mul_pd(self.twiddle8re, x425p);
        let temp_a2_5 = _mm_mul_pd(self.twiddle10re, x524p);
        let temp_a2_6 = _mm_mul_pd(self.twiddle12re, x623p);
        let temp_a2_7 = _mm_mul_pd(self.twiddle14re, x722p);
        let temp_a2_8 = _mm_mul_pd(self.twiddle13re, x821p);
        let temp_a2_9 = _mm_mul_pd(self.twiddle11re, x920p);
        let temp_a2_10 = _mm_mul_pd(self.twiddle9re, x1019p);
        let temp_a2_11 = _mm_mul_pd(self.twiddle7re, x1118p);
        let temp_a2_12 = _mm_mul_pd(self.twiddle5re, x1217p);
        let temp_a2_13 = _mm_mul_pd(self.twiddle3re, x1316p);
        let temp_a2_14 = _mm_mul_pd(self.twiddle1re, x1415p);
        let temp_a3_1 = _mm_mul_pd(self.twiddle3re, x128p);
        let temp_a3_2 = _mm_mul_pd(self.twiddle6re, x227p);
        let temp_a3_3 = _mm_mul_pd(self.twiddle9re, x326p);
        let temp_a3_4 = _mm_mul_pd(self.twiddle12re, x425p);
        let temp_a3_5 = _mm_mul_pd(self.twiddle14re, x524p);
        let temp_a3_6 = _mm_mul_pd(self.twiddle11re, x623p);
        let temp_a3_7 = _mm_mul_pd(self.twiddle8re, x722p);
        let temp_a3_8 = _mm_mul_pd(self.twiddle5re, x821p);
        let temp_a3_9 = _mm_mul_pd(self.twiddle2re, x920p);
        let temp_a3_10 = _mm_mul_pd(self.twiddle1re, x1019p);
        let temp_a3_11 = _mm_mul_pd(self.twiddle4re, x1118p);
        let temp_a3_12 = _mm_mul_pd(self.twiddle7re, x1217p);
        let temp_a3_13 = _mm_mul_pd(self.twiddle10re, x1316p);
        let temp_a3_14 = _mm_mul_pd(self.twiddle13re, x1415p);
        let temp_a4_1 = _mm_mul_pd(self.twiddle4re, x128p);
        let temp_a4_2 = _mm_mul_pd(self.twiddle8re, x227p);
        let temp_a4_3 = _mm_mul_pd(self.twiddle12re, x326p);
        let temp_a4_4 = _mm_mul_pd(self.twiddle13re, x425p);
        let temp_a4_5 = _mm_mul_pd(self.twiddle9re, x524p);
        let temp_a4_6 = _mm_mul_pd(self.twiddle5re, x623p);
        let temp_a4_7 = _mm_mul_pd(self.twiddle1re, x722p);
        let temp_a4_8 = _mm_mul_pd(self.twiddle3re, x821p);
        let temp_a4_9 = _mm_mul_pd(self.twiddle7re, x920p);
        let temp_a4_10 = _mm_mul_pd(self.twiddle11re, x1019p);
        let temp_a4_11 = _mm_mul_pd(self.twiddle14re, x1118p);
        let temp_a4_12 = _mm_mul_pd(self.twiddle10re, x1217p);
        let temp_a4_13 = _mm_mul_pd(self.twiddle6re, x1316p);
        let temp_a4_14 = _mm_mul_pd(self.twiddle2re, x1415p);
        let temp_a5_1 = _mm_mul_pd(self.twiddle5re, x128p);
        let temp_a5_2 = _mm_mul_pd(self.twiddle10re, x227p);
        let temp_a5_3 = _mm_mul_pd(self.twiddle14re, x326p);
        let temp_a5_4 = _mm_mul_pd(self.twiddle9re, x425p);
        let temp_a5_5 = _mm_mul_pd(self.twiddle4re, x524p);
        let temp_a5_6 = _mm_mul_pd(self.twiddle1re, x623p);
        let temp_a5_7 = _mm_mul_pd(self.twiddle6re, x722p);
        let temp_a5_8 = _mm_mul_pd(self.twiddle11re, x821p);
        let temp_a5_9 = _mm_mul_pd(self.twiddle13re, x920p);
        let temp_a5_10 = _mm_mul_pd(self.twiddle8re, x1019p);
        let temp_a5_11 = _mm_mul_pd(self.twiddle3re, x1118p);
        let temp_a5_12 = _mm_mul_pd(self.twiddle2re, x1217p);
        let temp_a5_13 = _mm_mul_pd(self.twiddle7re, x1316p);
        let temp_a5_14 = _mm_mul_pd(self.twiddle12re, x1415p);
        let temp_a6_1 = _mm_mul_pd(self.twiddle6re, x128p);
        let temp_a6_2 = _mm_mul_pd(self.twiddle12re, x227p);
        let temp_a6_3 = _mm_mul_pd(self.twiddle11re, x326p);
        let temp_a6_4 = _mm_mul_pd(self.twiddle5re, x425p);
        let temp_a6_5 = _mm_mul_pd(self.twiddle1re, x524p);
        let temp_a6_6 = _mm_mul_pd(self.twiddle7re, x623p);
        let temp_a6_7 = _mm_mul_pd(self.twiddle13re, x722p);
        let temp_a6_8 = _mm_mul_pd(self.twiddle10re, x821p);
        let temp_a6_9 = _mm_mul_pd(self.twiddle4re, x920p);
        let temp_a6_10 = _mm_mul_pd(self.twiddle2re, x1019p);
        let temp_a6_11 = _mm_mul_pd(self.twiddle8re, x1118p);
        let temp_a6_12 = _mm_mul_pd(self.twiddle14re, x1217p);
        let temp_a6_13 = _mm_mul_pd(self.twiddle9re, x1316p);
        let temp_a6_14 = _mm_mul_pd(self.twiddle3re, x1415p);
        let temp_a7_1 = _mm_mul_pd(self.twiddle7re, x128p);
        let temp_a7_2 = _mm_mul_pd(self.twiddle14re, x227p);
        let temp_a7_3 = _mm_mul_pd(self.twiddle8re, x326p);
        let temp_a7_4 = _mm_mul_pd(self.twiddle1re, x425p);
        let temp_a7_5 = _mm_mul_pd(self.twiddle6re, x524p);
        let temp_a7_6 = _mm_mul_pd(self.twiddle13re, x623p);
        let temp_a7_7 = _mm_mul_pd(self.twiddle9re, x722p);
        let temp_a7_8 = _mm_mul_pd(self.twiddle2re, x821p);
        let temp_a7_9 = _mm_mul_pd(self.twiddle5re, x920p);
        let temp_a7_10 = _mm_mul_pd(self.twiddle12re, x1019p);
        let temp_a7_11 = _mm_mul_pd(self.twiddle10re, x1118p);
        let temp_a7_12 = _mm_mul_pd(self.twiddle3re, x1217p);
        let temp_a7_13 = _mm_mul_pd(self.twiddle4re, x1316p);
        let temp_a7_14 = _mm_mul_pd(self.twiddle11re, x1415p);
        let temp_a8_1 = _mm_mul_pd(self.twiddle8re, x128p);
        let temp_a8_2 = _mm_mul_pd(self.twiddle13re, x227p);
        let temp_a8_3 = _mm_mul_pd(self.twiddle5re, x326p);
        let temp_a8_4 = _mm_mul_pd(self.twiddle3re, x425p);
        let temp_a8_5 = _mm_mul_pd(self.twiddle11re, x524p);
        let temp_a8_6 = _mm_mul_pd(self.twiddle10re, x623p);
        let temp_a8_7 = _mm_mul_pd(self.twiddle2re, x722p);
        let temp_a8_8 = _mm_mul_pd(self.twiddle6re, x821p);
        let temp_a8_9 = _mm_mul_pd(self.twiddle14re, x920p);
        let temp_a8_10 = _mm_mul_pd(self.twiddle7re, x1019p);
        let temp_a8_11 = _mm_mul_pd(self.twiddle1re, x1118p);
        let temp_a8_12 = _mm_mul_pd(self.twiddle9re, x1217p);
        let temp_a8_13 = _mm_mul_pd(self.twiddle12re, x1316p);
        let temp_a8_14 = _mm_mul_pd(self.twiddle4re, x1415p);
        let temp_a9_1 = _mm_mul_pd(self.twiddle9re, x128p);
        let temp_a9_2 = _mm_mul_pd(self.twiddle11re, x227p);
        let temp_a9_3 = _mm_mul_pd(self.twiddle2re, x326p);
        let temp_a9_4 = _mm_mul_pd(self.twiddle7re, x425p);
        let temp_a9_5 = _mm_mul_pd(self.twiddle13re, x524p);
        let temp_a9_6 = _mm_mul_pd(self.twiddle4re, x623p);
        let temp_a9_7 = _mm_mul_pd(self.twiddle5re, x722p);
        let temp_a9_8 = _mm_mul_pd(self.twiddle14re, x821p);
        let temp_a9_9 = _mm_mul_pd(self.twiddle6re, x920p);
        let temp_a9_10 = _mm_mul_pd(self.twiddle3re, x1019p);
        let temp_a9_11 = _mm_mul_pd(self.twiddle12re, x1118p);
        let temp_a9_12 = _mm_mul_pd(self.twiddle8re, x1217p);
        let temp_a9_13 = _mm_mul_pd(self.twiddle1re, x1316p);
        let temp_a9_14 = _mm_mul_pd(self.twiddle10re, x1415p);
        let temp_a10_1 = _mm_mul_pd(self.twiddle10re, x128p);
        let temp_a10_2 = _mm_mul_pd(self.twiddle9re, x227p);
        let temp_a10_3 = _mm_mul_pd(self.twiddle1re, x326p);
        let temp_a10_4 = _mm_mul_pd(self.twiddle11re, x425p);
        let temp_a10_5 = _mm_mul_pd(self.twiddle8re, x524p);
        let temp_a10_6 = _mm_mul_pd(self.twiddle2re, x623p);
        let temp_a10_7 = _mm_mul_pd(self.twiddle12re, x722p);
        let temp_a10_8 = _mm_mul_pd(self.twiddle7re, x821p);
        let temp_a10_9 = _mm_mul_pd(self.twiddle3re, x920p);
        let temp_a10_10 = _mm_mul_pd(self.twiddle13re, x1019p);
        let temp_a10_11 = _mm_mul_pd(self.twiddle6re, x1118p);
        let temp_a10_12 = _mm_mul_pd(self.twiddle4re, x1217p);
        let temp_a10_13 = _mm_mul_pd(self.twiddle14re, x1316p);
        let temp_a10_14 = _mm_mul_pd(self.twiddle5re, x1415p);
        let temp_a11_1 = _mm_mul_pd(self.twiddle11re, x128p);
        let temp_a11_2 = _mm_mul_pd(self.twiddle7re, x227p);
        let temp_a11_3 = _mm_mul_pd(self.twiddle4re, x326p);
        let temp_a11_4 = _mm_mul_pd(self.twiddle14re, x425p);
        let temp_a11_5 = _mm_mul_pd(self.twiddle3re, x524p);
        let temp_a11_6 = _mm_mul_pd(self.twiddle8re, x623p);
        let temp_a11_7 = _mm_mul_pd(self.twiddle10re, x722p);
        let temp_a11_8 = _mm_mul_pd(self.twiddle1re, x821p);
        let temp_a11_9 = _mm_mul_pd(self.twiddle12re, x920p);
        let temp_a11_10 = _mm_mul_pd(self.twiddle6re, x1019p);
        let temp_a11_11 = _mm_mul_pd(self.twiddle5re, x1118p);
        let temp_a11_12 = _mm_mul_pd(self.twiddle13re, x1217p);
        let temp_a11_13 = _mm_mul_pd(self.twiddle2re, x1316p);
        let temp_a11_14 = _mm_mul_pd(self.twiddle9re, x1415p);
        let temp_a12_1 = _mm_mul_pd(self.twiddle12re, x128p);
        let temp_a12_2 = _mm_mul_pd(self.twiddle5re, x227p);
        let temp_a12_3 = _mm_mul_pd(self.twiddle7re, x326p);
        let temp_a12_4 = _mm_mul_pd(self.twiddle10re, x425p);
        let temp_a12_5 = _mm_mul_pd(self.twiddle2re, x524p);
        let temp_a12_6 = _mm_mul_pd(self.twiddle14re, x623p);
        let temp_a12_7 = _mm_mul_pd(self.twiddle3re, x722p);
        let temp_a12_8 = _mm_mul_pd(self.twiddle9re, x821p);
        let temp_a12_9 = _mm_mul_pd(self.twiddle8re, x920p);
        let temp_a12_10 = _mm_mul_pd(self.twiddle4re, x1019p);
        let temp_a12_11 = _mm_mul_pd(self.twiddle13re, x1118p);
        let temp_a12_12 = _mm_mul_pd(self.twiddle1re, x1217p);
        let temp_a12_13 = _mm_mul_pd(self.twiddle11re, x1316p);
        let temp_a12_14 = _mm_mul_pd(self.twiddle6re, x1415p);
        let temp_a13_1 = _mm_mul_pd(self.twiddle13re, x128p);
        let temp_a13_2 = _mm_mul_pd(self.twiddle3re, x227p);
        let temp_a13_3 = _mm_mul_pd(self.twiddle10re, x326p);
        let temp_a13_4 = _mm_mul_pd(self.twiddle6re, x425p);
        let temp_a13_5 = _mm_mul_pd(self.twiddle7re, x524p);
        let temp_a13_6 = _mm_mul_pd(self.twiddle9re, x623p);
        let temp_a13_7 = _mm_mul_pd(self.twiddle4re, x722p);
        let temp_a13_8 = _mm_mul_pd(self.twiddle12re, x821p);
        let temp_a13_9 = _mm_mul_pd(self.twiddle1re, x920p);
        let temp_a13_10 = _mm_mul_pd(self.twiddle14re, x1019p);
        let temp_a13_11 = _mm_mul_pd(self.twiddle2re, x1118p);
        let temp_a13_12 = _mm_mul_pd(self.twiddle11re, x1217p);
        let temp_a13_13 = _mm_mul_pd(self.twiddle5re, x1316p);
        let temp_a13_14 = _mm_mul_pd(self.twiddle8re, x1415p);
        let temp_a14_1 = _mm_mul_pd(self.twiddle14re, x128p);
        let temp_a14_2 = _mm_mul_pd(self.twiddle1re, x227p);
        let temp_a14_3 = _mm_mul_pd(self.twiddle13re, x326p);
        let temp_a14_4 = _mm_mul_pd(self.twiddle2re, x425p);
        let temp_a14_5 = _mm_mul_pd(self.twiddle12re, x524p);
        let temp_a14_6 = _mm_mul_pd(self.twiddle3re, x623p);
        let temp_a14_7 = _mm_mul_pd(self.twiddle11re, x722p);
        let temp_a14_8 = _mm_mul_pd(self.twiddle4re, x821p);
        let temp_a14_9 = _mm_mul_pd(self.twiddle10re, x920p);
        let temp_a14_10 = _mm_mul_pd(self.twiddle5re, x1019p);
        let temp_a14_11 = _mm_mul_pd(self.twiddle9re, x1118p);
        let temp_a14_12 = _mm_mul_pd(self.twiddle6re, x1217p);
        let temp_a14_13 = _mm_mul_pd(self.twiddle8re, x1316p);
        let temp_a14_14 = _mm_mul_pd(self.twiddle7re, x1415p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x128n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x227n);
        let temp_b1_3 = _mm_mul_pd(self.twiddle3im, x326n);
        let temp_b1_4 = _mm_mul_pd(self.twiddle4im, x425n);
        let temp_b1_5 = _mm_mul_pd(self.twiddle5im, x524n);
        let temp_b1_6 = _mm_mul_pd(self.twiddle6im, x623n);
        let temp_b1_7 = _mm_mul_pd(self.twiddle7im, x722n);
        let temp_b1_8 = _mm_mul_pd(self.twiddle8im, x821n);
        let temp_b1_9 = _mm_mul_pd(self.twiddle9im, x920n);
        let temp_b1_10 = _mm_mul_pd(self.twiddle10im, x1019n);
        let temp_b1_11 = _mm_mul_pd(self.twiddle11im, x1118n);
        let temp_b1_12 = _mm_mul_pd(self.twiddle12im, x1217n);
        let temp_b1_13 = _mm_mul_pd(self.twiddle13im, x1316n);
        let temp_b1_14 = _mm_mul_pd(self.twiddle14im, x1415n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x128n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle4im, x227n);
        let temp_b2_3 = _mm_mul_pd(self.twiddle6im, x326n);
        let temp_b2_4 = _mm_mul_pd(self.twiddle8im, x425n);
        let temp_b2_5 = _mm_mul_pd(self.twiddle10im, x524n);
        let temp_b2_6 = _mm_mul_pd(self.twiddle12im, x623n);
        let temp_b2_7 = _mm_mul_pd(self.twiddle14im, x722n);
        let temp_b2_8 = _mm_mul_pd(self.twiddle13im, x821n);
        let temp_b2_9 = _mm_mul_pd(self.twiddle11im, x920n);
        let temp_b2_10 = _mm_mul_pd(self.twiddle9im, x1019n);
        let temp_b2_11 = _mm_mul_pd(self.twiddle7im, x1118n);
        let temp_b2_12 = _mm_mul_pd(self.twiddle5im, x1217n);
        let temp_b2_13 = _mm_mul_pd(self.twiddle3im, x1316n);
        let temp_b2_14 = _mm_mul_pd(self.twiddle1im, x1415n);
        let temp_b3_1 = _mm_mul_pd(self.twiddle3im, x128n);
        let temp_b3_2 = _mm_mul_pd(self.twiddle6im, x227n);
        let temp_b3_3 = _mm_mul_pd(self.twiddle9im, x326n);
        let temp_b3_4 = _mm_mul_pd(self.twiddle12im, x425n);
        let temp_b3_5 = _mm_mul_pd(self.twiddle14im, x524n);
        let temp_b3_6 = _mm_mul_pd(self.twiddle11im, x623n);
        let temp_b3_7 = _mm_mul_pd(self.twiddle8im, x722n);
        let temp_b3_8 = _mm_mul_pd(self.twiddle5im, x821n);
        let temp_b3_9 = _mm_mul_pd(self.twiddle2im, x920n);
        let temp_b3_10 = _mm_mul_pd(self.twiddle1im, x1019n);
        let temp_b3_11 = _mm_mul_pd(self.twiddle4im, x1118n);
        let temp_b3_12 = _mm_mul_pd(self.twiddle7im, x1217n);
        let temp_b3_13 = _mm_mul_pd(self.twiddle10im, x1316n);
        let temp_b3_14 = _mm_mul_pd(self.twiddle13im, x1415n);
        let temp_b4_1 = _mm_mul_pd(self.twiddle4im, x128n);
        let temp_b4_2 = _mm_mul_pd(self.twiddle8im, x227n);
        let temp_b4_3 = _mm_mul_pd(self.twiddle12im, x326n);
        let temp_b4_4 = _mm_mul_pd(self.twiddle13im, x425n);
        let temp_b4_5 = _mm_mul_pd(self.twiddle9im, x524n);
        let temp_b4_6 = _mm_mul_pd(self.twiddle5im, x623n);
        let temp_b4_7 = _mm_mul_pd(self.twiddle1im, x722n);
        let temp_b4_8 = _mm_mul_pd(self.twiddle3im, x821n);
        let temp_b4_9 = _mm_mul_pd(self.twiddle7im, x920n);
        let temp_b4_10 = _mm_mul_pd(self.twiddle11im, x1019n);
        let temp_b4_11 = _mm_mul_pd(self.twiddle14im, x1118n);
        let temp_b4_12 = _mm_mul_pd(self.twiddle10im, x1217n);
        let temp_b4_13 = _mm_mul_pd(self.twiddle6im, x1316n);
        let temp_b4_14 = _mm_mul_pd(self.twiddle2im, x1415n);
        let temp_b5_1 = _mm_mul_pd(self.twiddle5im, x128n);
        let temp_b5_2 = _mm_mul_pd(self.twiddle10im, x227n);
        let temp_b5_3 = _mm_mul_pd(self.twiddle14im, x326n);
        let temp_b5_4 = _mm_mul_pd(self.twiddle9im, x425n);
        let temp_b5_5 = _mm_mul_pd(self.twiddle4im, x524n);
        let temp_b5_6 = _mm_mul_pd(self.twiddle1im, x623n);
        let temp_b5_7 = _mm_mul_pd(self.twiddle6im, x722n);
        let temp_b5_8 = _mm_mul_pd(self.twiddle11im, x821n);
        let temp_b5_9 = _mm_mul_pd(self.twiddle13im, x920n);
        let temp_b5_10 = _mm_mul_pd(self.twiddle8im, x1019n);
        let temp_b5_11 = _mm_mul_pd(self.twiddle3im, x1118n);
        let temp_b5_12 = _mm_mul_pd(self.twiddle2im, x1217n);
        let temp_b5_13 = _mm_mul_pd(self.twiddle7im, x1316n);
        let temp_b5_14 = _mm_mul_pd(self.twiddle12im, x1415n);
        let temp_b6_1 = _mm_mul_pd(self.twiddle6im, x128n);
        let temp_b6_2 = _mm_mul_pd(self.twiddle12im, x227n);
        let temp_b6_3 = _mm_mul_pd(self.twiddle11im, x326n);
        let temp_b6_4 = _mm_mul_pd(self.twiddle5im, x425n);
        let temp_b6_5 = _mm_mul_pd(self.twiddle1im, x524n);
        let temp_b6_6 = _mm_mul_pd(self.twiddle7im, x623n);
        let temp_b6_7 = _mm_mul_pd(self.twiddle13im, x722n);
        let temp_b6_8 = _mm_mul_pd(self.twiddle10im, x821n);
        let temp_b6_9 = _mm_mul_pd(self.twiddle4im, x920n);
        let temp_b6_10 = _mm_mul_pd(self.twiddle2im, x1019n);
        let temp_b6_11 = _mm_mul_pd(self.twiddle8im, x1118n);
        let temp_b6_12 = _mm_mul_pd(self.twiddle14im, x1217n);
        let temp_b6_13 = _mm_mul_pd(self.twiddle9im, x1316n);
        let temp_b6_14 = _mm_mul_pd(self.twiddle3im, x1415n);
        let temp_b7_1 = _mm_mul_pd(self.twiddle7im, x128n);
        let temp_b7_2 = _mm_mul_pd(self.twiddle14im, x227n);
        let temp_b7_3 = _mm_mul_pd(self.twiddle8im, x326n);
        let temp_b7_4 = _mm_mul_pd(self.twiddle1im, x425n);
        let temp_b7_5 = _mm_mul_pd(self.twiddle6im, x524n);
        let temp_b7_6 = _mm_mul_pd(self.twiddle13im, x623n);
        let temp_b7_7 = _mm_mul_pd(self.twiddle9im, x722n);
        let temp_b7_8 = _mm_mul_pd(self.twiddle2im, x821n);
        let temp_b7_9 = _mm_mul_pd(self.twiddle5im, x920n);
        let temp_b7_10 = _mm_mul_pd(self.twiddle12im, x1019n);
        let temp_b7_11 = _mm_mul_pd(self.twiddle10im, x1118n);
        let temp_b7_12 = _mm_mul_pd(self.twiddle3im, x1217n);
        let temp_b7_13 = _mm_mul_pd(self.twiddle4im, x1316n);
        let temp_b7_14 = _mm_mul_pd(self.twiddle11im, x1415n);
        let temp_b8_1 = _mm_mul_pd(self.twiddle8im, x128n);
        let temp_b8_2 = _mm_mul_pd(self.twiddle13im, x227n);
        let temp_b8_3 = _mm_mul_pd(self.twiddle5im, x326n);
        let temp_b8_4 = _mm_mul_pd(self.twiddle3im, x425n);
        let temp_b8_5 = _mm_mul_pd(self.twiddle11im, x524n);
        let temp_b8_6 = _mm_mul_pd(self.twiddle10im, x623n);
        let temp_b8_7 = _mm_mul_pd(self.twiddle2im, x722n);
        let temp_b8_8 = _mm_mul_pd(self.twiddle6im, x821n);
        let temp_b8_9 = _mm_mul_pd(self.twiddle14im, x920n);
        let temp_b8_10 = _mm_mul_pd(self.twiddle7im, x1019n);
        let temp_b8_11 = _mm_mul_pd(self.twiddle1im, x1118n);
        let temp_b8_12 = _mm_mul_pd(self.twiddle9im, x1217n);
        let temp_b8_13 = _mm_mul_pd(self.twiddle12im, x1316n);
        let temp_b8_14 = _mm_mul_pd(self.twiddle4im, x1415n);
        let temp_b9_1 = _mm_mul_pd(self.twiddle9im, x128n);
        let temp_b9_2 = _mm_mul_pd(self.twiddle11im, x227n);
        let temp_b9_3 = _mm_mul_pd(self.twiddle2im, x326n);
        let temp_b9_4 = _mm_mul_pd(self.twiddle7im, x425n);
        let temp_b9_5 = _mm_mul_pd(self.twiddle13im, x524n);
        let temp_b9_6 = _mm_mul_pd(self.twiddle4im, x623n);
        let temp_b9_7 = _mm_mul_pd(self.twiddle5im, x722n);
        let temp_b9_8 = _mm_mul_pd(self.twiddle14im, x821n);
        let temp_b9_9 = _mm_mul_pd(self.twiddle6im, x920n);
        let temp_b9_10 = _mm_mul_pd(self.twiddle3im, x1019n);
        let temp_b9_11 = _mm_mul_pd(self.twiddle12im, x1118n);
        let temp_b9_12 = _mm_mul_pd(self.twiddle8im, x1217n);
        let temp_b9_13 = _mm_mul_pd(self.twiddle1im, x1316n);
        let temp_b9_14 = _mm_mul_pd(self.twiddle10im, x1415n);
        let temp_b10_1 = _mm_mul_pd(self.twiddle10im, x128n);
        let temp_b10_2 = _mm_mul_pd(self.twiddle9im, x227n);
        let temp_b10_3 = _mm_mul_pd(self.twiddle1im, x326n);
        let temp_b10_4 = _mm_mul_pd(self.twiddle11im, x425n);
        let temp_b10_5 = _mm_mul_pd(self.twiddle8im, x524n);
        let temp_b10_6 = _mm_mul_pd(self.twiddle2im, x623n);
        let temp_b10_7 = _mm_mul_pd(self.twiddle12im, x722n);
        let temp_b10_8 = _mm_mul_pd(self.twiddle7im, x821n);
        let temp_b10_9 = _mm_mul_pd(self.twiddle3im, x920n);
        let temp_b10_10 = _mm_mul_pd(self.twiddle13im, x1019n);
        let temp_b10_11 = _mm_mul_pd(self.twiddle6im, x1118n);
        let temp_b10_12 = _mm_mul_pd(self.twiddle4im, x1217n);
        let temp_b10_13 = _mm_mul_pd(self.twiddle14im, x1316n);
        let temp_b10_14 = _mm_mul_pd(self.twiddle5im, x1415n);
        let temp_b11_1 = _mm_mul_pd(self.twiddle11im, x128n);
        let temp_b11_2 = _mm_mul_pd(self.twiddle7im, x227n);
        let temp_b11_3 = _mm_mul_pd(self.twiddle4im, x326n);
        let temp_b11_4 = _mm_mul_pd(self.twiddle14im, x425n);
        let temp_b11_5 = _mm_mul_pd(self.twiddle3im, x524n);
        let temp_b11_6 = _mm_mul_pd(self.twiddle8im, x623n);
        let temp_b11_7 = _mm_mul_pd(self.twiddle10im, x722n);
        let temp_b11_8 = _mm_mul_pd(self.twiddle1im, x821n);
        let temp_b11_9 = _mm_mul_pd(self.twiddle12im, x920n);
        let temp_b11_10 = _mm_mul_pd(self.twiddle6im, x1019n);
        let temp_b11_11 = _mm_mul_pd(self.twiddle5im, x1118n);
        let temp_b11_12 = _mm_mul_pd(self.twiddle13im, x1217n);
        let temp_b11_13 = _mm_mul_pd(self.twiddle2im, x1316n);
        let temp_b11_14 = _mm_mul_pd(self.twiddle9im, x1415n);
        let temp_b12_1 = _mm_mul_pd(self.twiddle12im, x128n);
        let temp_b12_2 = _mm_mul_pd(self.twiddle5im, x227n);
        let temp_b12_3 = _mm_mul_pd(self.twiddle7im, x326n);
        let temp_b12_4 = _mm_mul_pd(self.twiddle10im, x425n);
        let temp_b12_5 = _mm_mul_pd(self.twiddle2im, x524n);
        let temp_b12_6 = _mm_mul_pd(self.twiddle14im, x623n);
        let temp_b12_7 = _mm_mul_pd(self.twiddle3im, x722n);
        let temp_b12_8 = _mm_mul_pd(self.twiddle9im, x821n);
        let temp_b12_9 = _mm_mul_pd(self.twiddle8im, x920n);
        let temp_b12_10 = _mm_mul_pd(self.twiddle4im, x1019n);
        let temp_b12_11 = _mm_mul_pd(self.twiddle13im, x1118n);
        let temp_b12_12 = _mm_mul_pd(self.twiddle1im, x1217n);
        let temp_b12_13 = _mm_mul_pd(self.twiddle11im, x1316n);
        let temp_b12_14 = _mm_mul_pd(self.twiddle6im, x1415n);
        let temp_b13_1 = _mm_mul_pd(self.twiddle13im, x128n);
        let temp_b13_2 = _mm_mul_pd(self.twiddle3im, x227n);
        let temp_b13_3 = _mm_mul_pd(self.twiddle10im, x326n);
        let temp_b13_4 = _mm_mul_pd(self.twiddle6im, x425n);
        let temp_b13_5 = _mm_mul_pd(self.twiddle7im, x524n);
        let temp_b13_6 = _mm_mul_pd(self.twiddle9im, x623n);
        let temp_b13_7 = _mm_mul_pd(self.twiddle4im, x722n);
        let temp_b13_8 = _mm_mul_pd(self.twiddle12im, x821n);
        let temp_b13_9 = _mm_mul_pd(self.twiddle1im, x920n);
        let temp_b13_10 = _mm_mul_pd(self.twiddle14im, x1019n);
        let temp_b13_11 = _mm_mul_pd(self.twiddle2im, x1118n);
        let temp_b13_12 = _mm_mul_pd(self.twiddle11im, x1217n);
        let temp_b13_13 = _mm_mul_pd(self.twiddle5im, x1316n);
        let temp_b13_14 = _mm_mul_pd(self.twiddle8im, x1415n);
        let temp_b14_1 = _mm_mul_pd(self.twiddle14im, x128n);
        let temp_b14_2 = _mm_mul_pd(self.twiddle1im, x227n);
        let temp_b14_3 = _mm_mul_pd(self.twiddle13im, x326n);
        let temp_b14_4 = _mm_mul_pd(self.twiddle2im, x425n);
        let temp_b14_5 = _mm_mul_pd(self.twiddle12im, x524n);
        let temp_b14_6 = _mm_mul_pd(self.twiddle3im, x623n);
        let temp_b14_7 = _mm_mul_pd(self.twiddle11im, x722n);
        let temp_b14_8 = _mm_mul_pd(self.twiddle4im, x821n);
        let temp_b14_9 = _mm_mul_pd(self.twiddle10im, x920n);
        let temp_b14_10 = _mm_mul_pd(self.twiddle5im, x1019n);
        let temp_b14_11 = _mm_mul_pd(self.twiddle9im, x1118n);
        let temp_b14_12 = _mm_mul_pd(self.twiddle6im, x1217n);
        let temp_b14_13 = _mm_mul_pd(self.twiddle8im, x1316n);
        let temp_b14_14 = _mm_mul_pd(self.twiddle7im, x1415n);

        let temp_a1 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a1_1,
                _mm_add_pd(
                    temp_a1_2,
                    _mm_add_pd(
                        temp_a1_3,
                        _mm_add_pd(
                            temp_a1_4,
                            _mm_add_pd(
                                temp_a1_5,
                                _mm_add_pd(
                                    temp_a1_6,
                                    _mm_add_pd(
                                        temp_a1_7,
                                        _mm_add_pd(
                                            temp_a1_8,
                                            _mm_add_pd(
                                                temp_a1_9,
                                                _mm_add_pd(
                                                    temp_a1_10,
                                                    _mm_add_pd(
                                                        temp_a1_11,
                                                        _mm_add_pd(
                                                            temp_a1_12,
                                                            _mm_add_pd(temp_a1_13, temp_a1_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a2_1,
                _mm_add_pd(
                    temp_a2_2,
                    _mm_add_pd(
                        temp_a2_3,
                        _mm_add_pd(
                            temp_a2_4,
                            _mm_add_pd(
                                temp_a2_5,
                                _mm_add_pd(
                                    temp_a2_6,
                                    _mm_add_pd(
                                        temp_a2_7,
                                        _mm_add_pd(
                                            temp_a2_8,
                                            _mm_add_pd(
                                                temp_a2_9,
                                                _mm_add_pd(
                                                    temp_a2_10,
                                                    _mm_add_pd(
                                                        temp_a2_11,
                                                        _mm_add_pd(
                                                            temp_a2_12,
                                                            _mm_add_pd(temp_a2_13, temp_a2_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a3_1,
                _mm_add_pd(
                    temp_a3_2,
                    _mm_add_pd(
                        temp_a3_3,
                        _mm_add_pd(
                            temp_a3_4,
                            _mm_add_pd(
                                temp_a3_5,
                                _mm_add_pd(
                                    temp_a3_6,
                                    _mm_add_pd(
                                        temp_a3_7,
                                        _mm_add_pd(
                                            temp_a3_8,
                                            _mm_add_pd(
                                                temp_a3_9,
                                                _mm_add_pd(
                                                    temp_a3_10,
                                                    _mm_add_pd(
                                                        temp_a3_11,
                                                        _mm_add_pd(
                                                            temp_a3_12,
                                                            _mm_add_pd(temp_a3_13, temp_a3_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a4_1,
                _mm_add_pd(
                    temp_a4_2,
                    _mm_add_pd(
                        temp_a4_3,
                        _mm_add_pd(
                            temp_a4_4,
                            _mm_add_pd(
                                temp_a4_5,
                                _mm_add_pd(
                                    temp_a4_6,
                                    _mm_add_pd(
                                        temp_a4_7,
                                        _mm_add_pd(
                                            temp_a4_8,
                                            _mm_add_pd(
                                                temp_a4_9,
                                                _mm_add_pd(
                                                    temp_a4_10,
                                                    _mm_add_pd(
                                                        temp_a4_11,
                                                        _mm_add_pd(
                                                            temp_a4_12,
                                                            _mm_add_pd(temp_a4_13, temp_a4_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a5_1,
                _mm_add_pd(
                    temp_a5_2,
                    _mm_add_pd(
                        temp_a5_3,
                        _mm_add_pd(
                            temp_a5_4,
                            _mm_add_pd(
                                temp_a5_5,
                                _mm_add_pd(
                                    temp_a5_6,
                                    _mm_add_pd(
                                        temp_a5_7,
                                        _mm_add_pd(
                                            temp_a5_8,
                                            _mm_add_pd(
                                                temp_a5_9,
                                                _mm_add_pd(
                                                    temp_a5_10,
                                                    _mm_add_pd(
                                                        temp_a5_11,
                                                        _mm_add_pd(
                                                            temp_a5_12,
                                                            _mm_add_pd(temp_a5_13, temp_a5_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a6_1,
                _mm_add_pd(
                    temp_a6_2,
                    _mm_add_pd(
                        temp_a6_3,
                        _mm_add_pd(
                            temp_a6_4,
                            _mm_add_pd(
                                temp_a6_5,
                                _mm_add_pd(
                                    temp_a6_6,
                                    _mm_add_pd(
                                        temp_a6_7,
                                        _mm_add_pd(
                                            temp_a6_8,
                                            _mm_add_pd(
                                                temp_a6_9,
                                                _mm_add_pd(
                                                    temp_a6_10,
                                                    _mm_add_pd(
                                                        temp_a6_11,
                                                        _mm_add_pd(
                                                            temp_a6_12,
                                                            _mm_add_pd(temp_a6_13, temp_a6_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a7_1,
                _mm_add_pd(
                    temp_a7_2,
                    _mm_add_pd(
                        temp_a7_3,
                        _mm_add_pd(
                            temp_a7_4,
                            _mm_add_pd(
                                temp_a7_5,
                                _mm_add_pd(
                                    temp_a7_6,
                                    _mm_add_pd(
                                        temp_a7_7,
                                        _mm_add_pd(
                                            temp_a7_8,
                                            _mm_add_pd(
                                                temp_a7_9,
                                                _mm_add_pd(
                                                    temp_a7_10,
                                                    _mm_add_pd(
                                                        temp_a7_11,
                                                        _mm_add_pd(
                                                            temp_a7_12,
                                                            _mm_add_pd(temp_a7_13, temp_a7_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a8_1,
                _mm_add_pd(
                    temp_a8_2,
                    _mm_add_pd(
                        temp_a8_3,
                        _mm_add_pd(
                            temp_a8_4,
                            _mm_add_pd(
                                temp_a8_5,
                                _mm_add_pd(
                                    temp_a8_6,
                                    _mm_add_pd(
                                        temp_a8_7,
                                        _mm_add_pd(
                                            temp_a8_8,
                                            _mm_add_pd(
                                                temp_a8_9,
                                                _mm_add_pd(
                                                    temp_a8_10,
                                                    _mm_add_pd(
                                                        temp_a8_11,
                                                        _mm_add_pd(
                                                            temp_a8_12,
                                                            _mm_add_pd(temp_a8_13, temp_a8_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a9 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a9_1,
                _mm_add_pd(
                    temp_a9_2,
                    _mm_add_pd(
                        temp_a9_3,
                        _mm_add_pd(
                            temp_a9_4,
                            _mm_add_pd(
                                temp_a9_5,
                                _mm_add_pd(
                                    temp_a9_6,
                                    _mm_add_pd(
                                        temp_a9_7,
                                        _mm_add_pd(
                                            temp_a9_8,
                                            _mm_add_pd(
                                                temp_a9_9,
                                                _mm_add_pd(
                                                    temp_a9_10,
                                                    _mm_add_pd(
                                                        temp_a9_11,
                                                        _mm_add_pd(
                                                            temp_a9_12,
                                                            _mm_add_pd(temp_a9_13, temp_a9_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a10 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a10_1,
                _mm_add_pd(
                    temp_a10_2,
                    _mm_add_pd(
                        temp_a10_3,
                        _mm_add_pd(
                            temp_a10_4,
                            _mm_add_pd(
                                temp_a10_5,
                                _mm_add_pd(
                                    temp_a10_6,
                                    _mm_add_pd(
                                        temp_a10_7,
                                        _mm_add_pd(
                                            temp_a10_8,
                                            _mm_add_pd(
                                                temp_a10_9,
                                                _mm_add_pd(
                                                    temp_a10_10,
                                                    _mm_add_pd(
                                                        temp_a10_11,
                                                        _mm_add_pd(
                                                            temp_a10_12,
                                                            _mm_add_pd(temp_a10_13, temp_a10_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a11 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a11_1,
                _mm_add_pd(
                    temp_a11_2,
                    _mm_add_pd(
                        temp_a11_3,
                        _mm_add_pd(
                            temp_a11_4,
                            _mm_add_pd(
                                temp_a11_5,
                                _mm_add_pd(
                                    temp_a11_6,
                                    _mm_add_pd(
                                        temp_a11_7,
                                        _mm_add_pd(
                                            temp_a11_8,
                                            _mm_add_pd(
                                                temp_a11_9,
                                                _mm_add_pd(
                                                    temp_a11_10,
                                                    _mm_add_pd(
                                                        temp_a11_11,
                                                        _mm_add_pd(
                                                            temp_a11_12,
                                                            _mm_add_pd(temp_a11_13, temp_a11_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a12 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a12_1,
                _mm_add_pd(
                    temp_a12_2,
                    _mm_add_pd(
                        temp_a12_3,
                        _mm_add_pd(
                            temp_a12_4,
                            _mm_add_pd(
                                temp_a12_5,
                                _mm_add_pd(
                                    temp_a12_6,
                                    _mm_add_pd(
                                        temp_a12_7,
                                        _mm_add_pd(
                                            temp_a12_8,
                                            _mm_add_pd(
                                                temp_a12_9,
                                                _mm_add_pd(
                                                    temp_a12_10,
                                                    _mm_add_pd(
                                                        temp_a12_11,
                                                        _mm_add_pd(
                                                            temp_a12_12,
                                                            _mm_add_pd(temp_a12_13, temp_a12_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a13 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a13_1,
                _mm_add_pd(
                    temp_a13_2,
                    _mm_add_pd(
                        temp_a13_3,
                        _mm_add_pd(
                            temp_a13_4,
                            _mm_add_pd(
                                temp_a13_5,
                                _mm_add_pd(
                                    temp_a13_6,
                                    _mm_add_pd(
                                        temp_a13_7,
                                        _mm_add_pd(
                                            temp_a13_8,
                                            _mm_add_pd(
                                                temp_a13_9,
                                                _mm_add_pd(
                                                    temp_a13_10,
                                                    _mm_add_pd(
                                                        temp_a13_11,
                                                        _mm_add_pd(
                                                            temp_a13_12,
                                                            _mm_add_pd(temp_a13_13, temp_a13_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a14 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a14_1,
                _mm_add_pd(
                    temp_a14_2,
                    _mm_add_pd(
                        temp_a14_3,
                        _mm_add_pd(
                            temp_a14_4,
                            _mm_add_pd(
                                temp_a14_5,
                                _mm_add_pd(
                                    temp_a14_6,
                                    _mm_add_pd(
                                        temp_a14_7,
                                        _mm_add_pd(
                                            temp_a14_8,
                                            _mm_add_pd(
                                                temp_a14_9,
                                                _mm_add_pd(
                                                    temp_a14_10,
                                                    _mm_add_pd(
                                                        temp_a14_11,
                                                        _mm_add_pd(
                                                            temp_a14_12,
                                                            _mm_add_pd(temp_a14_13, temp_a14_14),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_pd(
            temp_b1_1,
            _mm_add_pd(
                temp_b1_2,
                _mm_add_pd(
                    temp_b1_3,
                    _mm_add_pd(
                        temp_b1_4,
                        _mm_add_pd(
                            temp_b1_5,
                            _mm_add_pd(
                                temp_b1_6,
                                _mm_add_pd(
                                    temp_b1_7,
                                    _mm_add_pd(
                                        temp_b1_8,
                                        _mm_add_pd(
                                            temp_b1_9,
                                            _mm_add_pd(
                                                temp_b1_10,
                                                _mm_add_pd(
                                                    temp_b1_11,
                                                    _mm_add_pd(
                                                        temp_b1_12,
                                                        _mm_add_pd(temp_b1_13, temp_b1_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_pd(
            temp_b2_1,
            _mm_add_pd(
                temp_b2_2,
                _mm_add_pd(
                    temp_b2_3,
                    _mm_add_pd(
                        temp_b2_4,
                        _mm_add_pd(
                            temp_b2_5,
                            _mm_add_pd(
                                temp_b2_6,
                                _mm_sub_pd(
                                    temp_b2_7,
                                    _mm_add_pd(
                                        temp_b2_8,
                                        _mm_add_pd(
                                            temp_b2_9,
                                            _mm_add_pd(
                                                temp_b2_10,
                                                _mm_add_pd(
                                                    temp_b2_11,
                                                    _mm_add_pd(
                                                        temp_b2_12,
                                                        _mm_add_pd(temp_b2_13, temp_b2_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_pd(
            temp_b3_1,
            _mm_add_pd(
                temp_b3_2,
                _mm_add_pd(
                    temp_b3_3,
                    _mm_sub_pd(
                        temp_b3_4,
                        _mm_add_pd(
                            temp_b3_5,
                            _mm_add_pd(
                                temp_b3_6,
                                _mm_add_pd(
                                    temp_b3_7,
                                    _mm_add_pd(
                                        temp_b3_8,
                                        _mm_sub_pd(
                                            temp_b3_9,
                                            _mm_add_pd(
                                                temp_b3_10,
                                                _mm_add_pd(
                                                    temp_b3_11,
                                                    _mm_add_pd(
                                                        temp_b3_12,
                                                        _mm_add_pd(temp_b3_13, temp_b3_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_pd(
            temp_b4_1,
            _mm_add_pd(
                temp_b4_2,
                _mm_sub_pd(
                    temp_b4_3,
                    _mm_add_pd(
                        temp_b4_4,
                        _mm_add_pd(
                            temp_b4_5,
                            _mm_add_pd(
                                temp_b4_6,
                                _mm_sub_pd(
                                    temp_b4_7,
                                    _mm_add_pd(
                                        temp_b4_8,
                                        _mm_add_pd(
                                            temp_b4_9,
                                            _mm_sub_pd(
                                                temp_b4_10,
                                                _mm_add_pd(
                                                    temp_b4_11,
                                                    _mm_add_pd(
                                                        temp_b4_12,
                                                        _mm_add_pd(temp_b4_13, temp_b4_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_add_pd(
            temp_b5_1,
            _mm_sub_pd(
                temp_b5_2,
                _mm_add_pd(
                    temp_b5_3,
                    _mm_add_pd(
                        temp_b5_4,
                        _mm_sub_pd(
                            temp_b5_5,
                            _mm_add_pd(
                                temp_b5_6,
                                _mm_add_pd(
                                    temp_b5_7,
                                    _mm_sub_pd(
                                        temp_b5_8,
                                        _mm_add_pd(
                                            temp_b5_9,
                                            _mm_add_pd(
                                                temp_b5_10,
                                                _mm_sub_pd(
                                                    temp_b5_11,
                                                    _mm_add_pd(
                                                        temp_b5_12,
                                                        _mm_add_pd(temp_b5_13, temp_b5_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_add_pd(
            temp_b6_1,
            _mm_sub_pd(
                temp_b6_2,
                _mm_add_pd(
                    temp_b6_3,
                    _mm_sub_pd(
                        temp_b6_4,
                        _mm_add_pd(
                            temp_b6_5,
                            _mm_add_pd(
                                temp_b6_6,
                                _mm_sub_pd(
                                    temp_b6_7,
                                    _mm_add_pd(
                                        temp_b6_8,
                                        _mm_sub_pd(
                                            temp_b6_9,
                                            _mm_add_pd(
                                                temp_b6_10,
                                                _mm_add_pd(
                                                    temp_b6_11,
                                                    _mm_sub_pd(
                                                        temp_b6_12,
                                                        _mm_add_pd(temp_b6_13, temp_b6_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_add_pd(
            temp_b7_1,
            _mm_sub_pd(
                temp_b7_2,
                _mm_add_pd(
                    temp_b7_3,
                    _mm_sub_pd(
                        temp_b7_4,
                        _mm_add_pd(
                            temp_b7_5,
                            _mm_sub_pd(
                                temp_b7_6,
                                _mm_add_pd(
                                    temp_b7_7,
                                    _mm_sub_pd(
                                        temp_b7_8,
                                        _mm_add_pd(
                                            temp_b7_9,
                                            _mm_sub_pd(
                                                temp_b7_10,
                                                _mm_add_pd(
                                                    temp_b7_11,
                                                    _mm_sub_pd(
                                                        temp_b7_12,
                                                        _mm_add_pd(temp_b7_13, temp_b7_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_pd(
            temp_b8_1,
            _mm_add_pd(
                temp_b8_2,
                _mm_sub_pd(
                    temp_b8_3,
                    _mm_add_pd(
                        temp_b8_4,
                        _mm_sub_pd(
                            temp_b8_5,
                            _mm_add_pd(
                                temp_b8_6,
                                _mm_sub_pd(
                                    temp_b8_7,
                                    _mm_add_pd(
                                        temp_b8_8,
                                        _mm_sub_pd(
                                            temp_b8_9,
                                            _mm_sub_pd(
                                                temp_b8_10,
                                                _mm_add_pd(
                                                    temp_b8_11,
                                                    _mm_sub_pd(
                                                        temp_b8_12,
                                                        _mm_add_pd(temp_b8_13, temp_b8_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b9 = _mm_sub_pd(
            temp_b9_1,
            _mm_add_pd(
                temp_b9_2,
                _mm_sub_pd(
                    temp_b9_3,
                    _mm_sub_pd(
                        temp_b9_4,
                        _mm_add_pd(
                            temp_b9_5,
                            _mm_sub_pd(
                                temp_b9_6,
                                _mm_add_pd(
                                    temp_b9_7,
                                    _mm_sub_pd(
                                        temp_b9_8,
                                        _mm_sub_pd(
                                            temp_b9_9,
                                            _mm_add_pd(
                                                temp_b9_10,
                                                _mm_sub_pd(
                                                    temp_b9_11,
                                                    _mm_sub_pd(
                                                        temp_b9_12,
                                                        _mm_add_pd(temp_b9_13, temp_b9_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b10 = _mm_sub_pd(
            temp_b10_1,
            _mm_sub_pd(
                temp_b10_2,
                _mm_add_pd(
                    temp_b10_3,
                    _mm_sub_pd(
                        temp_b10_4,
                        _mm_sub_pd(
                            temp_b10_5,
                            _mm_add_pd(
                                temp_b10_6,
                                _mm_sub_pd(
                                    temp_b10_7,
                                    _mm_sub_pd(
                                        temp_b10_8,
                                        _mm_add_pd(
                                            temp_b10_9,
                                            _mm_sub_pd(
                                                temp_b10_10,
                                                _mm_sub_pd(
                                                    temp_b10_11,
                                                    _mm_add_pd(
                                                        temp_b10_12,
                                                        _mm_sub_pd(temp_b10_13, temp_b10_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b11 = _mm_sub_pd(
            temp_b11_1,
            _mm_sub_pd(
                temp_b11_2,
                _mm_sub_pd(
                    temp_b11_3,
                    _mm_add_pd(
                        temp_b11_4,
                        _mm_sub_pd(
                            temp_b11_5,
                            _mm_sub_pd(
                                temp_b11_6,
                                _mm_sub_pd(
                                    temp_b11_7,
                                    _mm_add_pd(
                                        temp_b11_8,
                                        _mm_sub_pd(
                                            temp_b11_9,
                                            _mm_sub_pd(
                                                temp_b11_10,
                                                _mm_sub_pd(
                                                    temp_b11_11,
                                                    _mm_add_pd(
                                                        temp_b11_12,
                                                        _mm_sub_pd(temp_b11_13, temp_b11_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b12 = _mm_sub_pd(
            temp_b12_1,
            _mm_sub_pd(
                temp_b12_2,
                _mm_sub_pd(
                    temp_b12_3,
                    _mm_sub_pd(
                        temp_b12_4,
                        _mm_add_pd(
                            temp_b12_5,
                            _mm_sub_pd(
                                temp_b12_6,
                                _mm_sub_pd(
                                    temp_b12_7,
                                    _mm_sub_pd(
                                        temp_b12_8,
                                        _mm_sub_pd(
                                            temp_b12_9,
                                            _mm_sub_pd(
                                                temp_b12_10,
                                                _mm_add_pd(
                                                    temp_b12_11,
                                                    _mm_sub_pd(
                                                        temp_b12_12,
                                                        _mm_sub_pd(temp_b12_13, temp_b12_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b13 = _mm_sub_pd(
            temp_b13_1,
            _mm_sub_pd(
                temp_b13_2,
                _mm_sub_pd(
                    temp_b13_3,
                    _mm_sub_pd(
                        temp_b13_4,
                        _mm_sub_pd(
                            temp_b13_5,
                            _mm_sub_pd(
                                temp_b13_6,
                                _mm_sub_pd(
                                    temp_b13_7,
                                    _mm_sub_pd(
                                        temp_b13_8,
                                        _mm_add_pd(
                                            temp_b13_9,
                                            _mm_sub_pd(
                                                temp_b13_10,
                                                _mm_sub_pd(
                                                    temp_b13_11,
                                                    _mm_sub_pd(
                                                        temp_b13_12,
                                                        _mm_sub_pd(temp_b13_13, temp_b13_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b14 = _mm_sub_pd(
            temp_b14_1,
            _mm_sub_pd(
                temp_b14_2,
                _mm_sub_pd(
                    temp_b14_3,
                    _mm_sub_pd(
                        temp_b14_4,
                        _mm_sub_pd(
                            temp_b14_5,
                            _mm_sub_pd(
                                temp_b14_6,
                                _mm_sub_pd(
                                    temp_b14_7,
                                    _mm_sub_pd(
                                        temp_b14_8,
                                        _mm_sub_pd(
                                            temp_b14_9,
                                            _mm_sub_pd(
                                                temp_b14_10,
                                                _mm_sub_pd(
                                                    temp_b14_11,
                                                    _mm_sub_pd(
                                                        temp_b14_12,
                                                        _mm_sub_pd(temp_b14_13, temp_b14_14),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);
        let temp_b6_rot = self.rotate.rotate(temp_b6);
        let temp_b7_rot = self.rotate.rotate(temp_b7);
        let temp_b8_rot = self.rotate.rotate(temp_b8);
        let temp_b9_rot = self.rotate.rotate(temp_b9);
        let temp_b10_rot = self.rotate.rotate(temp_b10);
        let temp_b11_rot = self.rotate.rotate(temp_b11);
        let temp_b12_rot = self.rotate.rotate(temp_b12);
        let temp_b13_rot = self.rotate.rotate(temp_b13);
        let temp_b14_rot = self.rotate.rotate(temp_b14);

        let x0 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                x128p,
                _mm_add_pd(
                    x227p,
                    _mm_add_pd(
                        x326p,
                        _mm_add_pd(
                            x425p,
                            _mm_add_pd(
                                x524p,
                                _mm_add_pd(
                                    x623p,
                                    _mm_add_pd(
                                        x722p,
                                        _mm_add_pd(
                                            x821p,
                                            _mm_add_pd(
                                                x920p,
                                                _mm_add_pd(
                                                    x1019p,
                                                    _mm_add_pd(
                                                        x1118p,
                                                        _mm_add_pd(
                                                            x1217p,
                                                            _mm_add_pd(x1316p, x1415p),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
        let x4 = _mm_add_pd(temp_a4, temp_b4_rot);
        let x5 = _mm_add_pd(temp_a5, temp_b5_rot);
        let x6 = _mm_add_pd(temp_a6, temp_b6_rot);
        let x7 = _mm_add_pd(temp_a7, temp_b7_rot);
        let x8 = _mm_add_pd(temp_a8, temp_b8_rot);
        let x9 = _mm_add_pd(temp_a9, temp_b9_rot);
        let x10 = _mm_add_pd(temp_a10, temp_b10_rot);
        let x11 = _mm_add_pd(temp_a11, temp_b11_rot);
        let x12 = _mm_add_pd(temp_a12, temp_b12_rot);
        let x13 = _mm_add_pd(temp_a13, temp_b13_rot);
        let x14 = _mm_add_pd(temp_a14, temp_b14_rot);
        let x15 = _mm_sub_pd(temp_a14, temp_b14_rot);
        let x16 = _mm_sub_pd(temp_a13, temp_b13_rot);
        let x17 = _mm_sub_pd(temp_a12, temp_b12_rot);
        let x18 = _mm_sub_pd(temp_a11, temp_b11_rot);
        let x19 = _mm_sub_pd(temp_a10, temp_b10_rot);
        let x20 = _mm_sub_pd(temp_a9, temp_b9_rot);
        let x21 = _mm_sub_pd(temp_a8, temp_b8_rot);
        let x22 = _mm_sub_pd(temp_a7, temp_b7_rot);
        let x23 = _mm_sub_pd(temp_a6, temp_b6_rot);
        let x24 = _mm_sub_pd(temp_a5, temp_b5_rot);
        let x25 = _mm_sub_pd(temp_a4, temp_b4_rot);
        let x26 = _mm_sub_pd(temp_a3, temp_b3_rot);
        let x27 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x28 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
            x19, x20, x21, x22, x23, x24, x25, x26, x27, x28,
        ]
    }
}

//   _____ _           _________  _     _ _
//  |___ // |         |___ /___ \| |__ (_) |_
//    |_ \| |  _____    |_ \ __) | '_ \| | __|
//   ___) | | |_____|  ___) / __/| |_) | | |_
//  |____/|_|         |____/_____|_.__/|_|\__|
//
pub struct SseF32Butterfly31<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle1re: __m128,
    twiddle1im: __m128,
    twiddle2re: __m128,
    twiddle2im: __m128,
    twiddle3re: __m128,
    twiddle3im: __m128,
    twiddle4re: __m128,
    twiddle4im: __m128,
    twiddle5re: __m128,
    twiddle5im: __m128,
    twiddle6re: __m128,
    twiddle6im: __m128,
    twiddle7re: __m128,
    twiddle7im: __m128,
    twiddle8re: __m128,
    twiddle8im: __m128,
    twiddle9re: __m128,
    twiddle9im: __m128,
    twiddle10re: __m128,
    twiddle10im: __m128,
    twiddle11re: __m128,
    twiddle11im: __m128,
    twiddle12re: __m128,
    twiddle12im: __m128,
    twiddle13re: __m128,
    twiddle13im: __m128,
    twiddle14re: __m128,
    twiddle14im: __m128,
    twiddle15re: __m128,
    twiddle15im: __m128,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly31, 31, |this: &SseF32Butterfly31<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly31, 31, |this: &SseF32Butterfly31<_>| this
    .direction);
impl<T: FftNum> SseF32Butterfly31<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 31, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 31, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 31, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 31, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 31, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 31, direction);
        let tw7: Complex<f32> = twiddles::compute_twiddle(7, 31, direction);
        let tw8: Complex<f32> = twiddles::compute_twiddle(8, 31, direction);
        let tw9: Complex<f32> = twiddles::compute_twiddle(9, 31, direction);
        let tw10: Complex<f32> = twiddles::compute_twiddle(10, 31, direction);
        let tw11: Complex<f32> = twiddles::compute_twiddle(11, 31, direction);
        let tw12: Complex<f32> = twiddles::compute_twiddle(12, 31, direction);
        let tw13: Complex<f32> = twiddles::compute_twiddle(13, 31, direction);
        let tw14: Complex<f32> = twiddles::compute_twiddle(14, 31, direction);
        let tw15: Complex<f32> = twiddles::compute_twiddle(15, 31, direction);
        let twiddle1re = unsafe { _mm_load1_ps(&tw1.re) };
        let twiddle1im = unsafe { _mm_load1_ps(&tw1.im) };
        let twiddle2re = unsafe { _mm_load1_ps(&tw2.re) };
        let twiddle2im = unsafe { _mm_load1_ps(&tw2.im) };
        let twiddle3re = unsafe { _mm_load1_ps(&tw3.re) };
        let twiddle3im = unsafe { _mm_load1_ps(&tw3.im) };
        let twiddle4re = unsafe { _mm_load1_ps(&tw4.re) };
        let twiddle4im = unsafe { _mm_load1_ps(&tw4.im) };
        let twiddle5re = unsafe { _mm_load1_ps(&tw5.re) };
        let twiddle5im = unsafe { _mm_load1_ps(&tw5.im) };
        let twiddle6re = unsafe { _mm_load1_ps(&tw6.re) };
        let twiddle6im = unsafe { _mm_load1_ps(&tw6.im) };
        let twiddle7re = unsafe { _mm_load1_ps(&tw7.re) };
        let twiddle7im = unsafe { _mm_load1_ps(&tw7.im) };
        let twiddle8re = unsafe { _mm_load1_ps(&tw8.re) };
        let twiddle8im = unsafe { _mm_load1_ps(&tw8.im) };
        let twiddle9re = unsafe { _mm_load1_ps(&tw9.re) };
        let twiddle9im = unsafe { _mm_load1_ps(&tw9.im) };
        let twiddle10re = unsafe { _mm_load1_ps(&tw10.re) };
        let twiddle10im = unsafe { _mm_load1_ps(&tw10.im) };
        let twiddle11re = unsafe { _mm_load1_ps(&tw11.re) };
        let twiddle11im = unsafe { _mm_load1_ps(&tw11.im) };
        let twiddle12re = unsafe { _mm_load1_ps(&tw12.re) };
        let twiddle12im = unsafe { _mm_load1_ps(&tw12.im) };
        let twiddle13re = unsafe { _mm_load1_ps(&tw13.re) };
        let twiddle13im = unsafe { _mm_load1_ps(&tw13.im) };
        let twiddle14re = unsafe { _mm_load1_ps(&tw14.re) };
        let twiddle14im = unsafe { _mm_load1_ps(&tw14.im) };
        let twiddle15re = unsafe { _mm_load1_ps(&tw15.re) };
        let twiddle15im = unsafe { _mm_load1_ps(&tw15.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
            twiddle9re,
            twiddle9im,
            twiddle10re,
            twiddle10im,
            twiddle11re,
            twiddle11im,
            twiddle12re,
            twiddle12im,
            twiddle13re,
            twiddle13im,
            twiddle14re,
            twiddle14im,
            twiddle15re,
            twiddle15im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let v1 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(1) as *const f64));
        let v2 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(2) as *const f64));
        let v3 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(3) as *const f64));
        let v4 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(4) as *const f64));
        let v5 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(5) as *const f64));
        let v6 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(6) as *const f64));
        let v7 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(7) as *const f64));
        let v8 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(8) as *const f64));
        let v9 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(9) as *const f64));
        let v10 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(10) as *const f64));
        let v11 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(11) as *const f64));
        let v12 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(12) as *const f64));
        let v13 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(13) as *const f64));
        let v14 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(14) as *const f64));
        let v15 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(15) as *const f64));
        let v16 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(16) as *const f64));
        let v17 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(17) as *const f64));
        let v18 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(18) as *const f64));
        let v19 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(19) as *const f64));
        let v20 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(20) as *const f64));
        let v21 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(21) as *const f64));
        let v22 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(22) as *const f64));
        let v23 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(23) as *const f64));
        let v24 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(24) as *const f64));
        let v25 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(25) as *const f64));
        let v26 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(26) as *const f64));
        let v27 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(27) as *const f64));
        let v28 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(28) as *const f64));
        let v29 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(29) as *const f64));
        let v30 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr().add(30) as *const f64));

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
        ]);

        let val = std::mem::transmute::<[__m128; 31], [Complex<f32>; 62]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[34];
        *output_slice.add(18) = val[36];
        *output_slice.add(19) = val[38];
        *output_slice.add(20) = val[40];
        *output_slice.add(21) = val[42];
        *output_slice.add(22) = val[44];
        *output_slice.add(23) = val[46];
        *output_slice.add(24) = val[48];
        *output_slice.add(25) = val[50];
        *output_slice.add(26) = val[52];
        *output_slice.add(27) = val[54];
        *output_slice.add(28) = val[56];
        *output_slice.add(29) = val[58];
        *output_slice.add(30) = val[60];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4a5 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valuea6a7 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valuea8a9 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let valuea10a11 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let valuea12a13 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let valuea14a15 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let valuea16a17 = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let valuea18a19 = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let valuea20a21 = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let valuea22a23 = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let valuea24a25 = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);
        let valuea26a27 = _mm_loadu_ps(input.as_ptr().add(26) as *const f32);
        let valuea28a29 = _mm_loadu_ps(input.as_ptr().add(28) as *const f32);
        let valuea30b0 = _mm_loadu_ps(input.as_ptr().add(30) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(32) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(34) as *const f32);
        let valueb5b6 = _mm_loadu_ps(input.as_ptr().add(36) as *const f32);
        let valueb7b8 = _mm_loadu_ps(input.as_ptr().add(38) as *const f32);
        let valueb9b10 = _mm_loadu_ps(input.as_ptr().add(40) as *const f32);
        let valueb11b12 = _mm_loadu_ps(input.as_ptr().add(42) as *const f32);
        let valueb13b14 = _mm_loadu_ps(input.as_ptr().add(44) as *const f32);
        let valueb15b16 = _mm_loadu_ps(input.as_ptr().add(46) as *const f32);
        let valueb17b18 = _mm_loadu_ps(input.as_ptr().add(48) as *const f32);
        let valueb19b20 = _mm_loadu_ps(input.as_ptr().add(50) as *const f32);
        let valueb21b22 = _mm_loadu_ps(input.as_ptr().add(52) as *const f32);
        let valueb23b24 = _mm_loadu_ps(input.as_ptr().add(54) as *const f32);
        let valueb25b26 = _mm_loadu_ps(input.as_ptr().add(56) as *const f32);
        let valueb27b28 = _mm_loadu_ps(input.as_ptr().add(58) as *const f32);
        let valueb29b30 = _mm_loadu_ps(input.as_ptr().add(60) as *const f32);

        let v0 = pack_1and2_f32(valuea0a1, valuea30b0);
        let v1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let v2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let v3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let v4 = pack_1and2_f32(valuea4a5, valueb3b4);
        let v5 = pack_2and1_f32(valuea4a5, valueb5b6);
        let v6 = pack_1and2_f32(valuea6a7, valueb5b6);
        let v7 = pack_2and1_f32(valuea6a7, valueb7b8);
        let v8 = pack_1and2_f32(valuea8a9, valueb7b8);
        let v9 = pack_2and1_f32(valuea8a9, valueb9b10);
        let v10 = pack_1and2_f32(valuea10a11, valueb9b10);
        let v11 = pack_2and1_f32(valuea10a11, valueb11b12);
        let v12 = pack_1and2_f32(valuea12a13, valueb11b12);
        let v13 = pack_2and1_f32(valuea12a13, valueb13b14);
        let v14 = pack_1and2_f32(valuea14a15, valueb13b14);
        let v15 = pack_2and1_f32(valuea14a15, valueb15b16);
        let v16 = pack_1and2_f32(valuea16a17, valueb15b16);
        let v17 = pack_2and1_f32(valuea16a17, valueb17b18);
        let v18 = pack_1and2_f32(valuea18a19, valueb17b18);
        let v19 = pack_2and1_f32(valuea18a19, valueb19b20);
        let v20 = pack_1and2_f32(valuea20a21, valueb19b20);
        let v21 = pack_2and1_f32(valuea20a21, valueb21b22);
        let v22 = pack_1and2_f32(valuea22a23, valueb21b22);
        let v23 = pack_2and1_f32(valuea22a23, valueb23b24);
        let v24 = pack_1and2_f32(valuea24a25, valueb23b24);
        let v25 = pack_2and1_f32(valuea24a25, valueb25b26);
        let v26 = pack_1and2_f32(valuea26a27, valueb25b26);
        let v27 = pack_2and1_f32(valuea26a27, valueb27b28);
        let v28 = pack_1and2_f32(valuea28a29, valueb27b28);
        let v29 = pack_2and1_f32(valuea28a29, valueb29b30);
        let v30 = pack_1and2_f32(valuea30b0, valueb29b30);

        let out = self.perform_dual_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
        ]);

        let val = std::mem::transmute::<[__m128; 31], [Complex<f32>; 62]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[10];
        *output_slice.add(6) = val[12];
        *output_slice.add(7) = val[14];
        *output_slice.add(8) = val[16];
        *output_slice.add(9) = val[18];
        *output_slice.add(10) = val[20];
        *output_slice.add(11) = val[22];
        *output_slice.add(12) = val[24];
        *output_slice.add(13) = val[26];
        *output_slice.add(14) = val[28];
        *output_slice.add(15) = val[30];
        *output_slice.add(16) = val[32];
        *output_slice.add(17) = val[34];
        *output_slice.add(18) = val[36];
        *output_slice.add(19) = val[38];
        *output_slice.add(20) = val[40];
        *output_slice.add(21) = val[42];
        *output_slice.add(22) = val[44];
        *output_slice.add(23) = val[46];
        *output_slice.add(24) = val[48];
        *output_slice.add(25) = val[50];
        *output_slice.add(26) = val[52];
        *output_slice.add(27) = val[54];
        *output_slice.add(28) = val[56];
        *output_slice.add(29) = val[58];
        *output_slice.add(30) = val[60];
        *output_slice.add(31) = val[1];
        *output_slice.add(32) = val[3];
        *output_slice.add(33) = val[5];
        *output_slice.add(34) = val[7];
        *output_slice.add(35) = val[9];
        *output_slice.add(36) = val[11];
        *output_slice.add(37) = val[13];
        *output_slice.add(38) = val[15];
        *output_slice.add(39) = val[17];
        *output_slice.add(40) = val[19];
        *output_slice.add(41) = val[21];
        *output_slice.add(42) = val[23];
        *output_slice.add(43) = val[25];
        *output_slice.add(44) = val[27];
        *output_slice.add(45) = val[29];
        *output_slice.add(46) = val[31];
        *output_slice.add(47) = val[33];
        *output_slice.add(48) = val[35];
        *output_slice.add(49) = val[37];
        *output_slice.add(50) = val[39];
        *output_slice.add(51) = val[41];
        *output_slice.add(52) = val[43];
        *output_slice.add(53) = val[45];
        *output_slice.add(54) = val[47];
        *output_slice.add(55) = val[49];
        *output_slice.add(56) = val[51];
        *output_slice.add(57) = val[53];
        *output_slice.add(58) = val[55];
        *output_slice.add(59) = val[57];
        *output_slice.add(60) = val[59];
        *output_slice.add(61) = val[61];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values: [__m128; 31]) -> [__m128; 31] {
        // This is a SSE translation of the scalar 17-point butterfly
        let x130p = _mm_add_ps(values[1], values[30]);
        let x130n = _mm_sub_ps(values[1], values[30]);
        let x229p = _mm_add_ps(values[2], values[29]);
        let x229n = _mm_sub_ps(values[2], values[29]);
        let x328p = _mm_add_ps(values[3], values[28]);
        let x328n = _mm_sub_ps(values[3], values[28]);
        let x427p = _mm_add_ps(values[4], values[27]);
        let x427n = _mm_sub_ps(values[4], values[27]);
        let x526p = _mm_add_ps(values[5], values[26]);
        let x526n = _mm_sub_ps(values[5], values[26]);
        let x625p = _mm_add_ps(values[6], values[25]);
        let x625n = _mm_sub_ps(values[6], values[25]);
        let x724p = _mm_add_ps(values[7], values[24]);
        let x724n = _mm_sub_ps(values[7], values[24]);
        let x823p = _mm_add_ps(values[8], values[23]);
        let x823n = _mm_sub_ps(values[8], values[23]);
        let x922p = _mm_add_ps(values[9], values[22]);
        let x922n = _mm_sub_ps(values[9], values[22]);
        let x1021p = _mm_add_ps(values[10], values[21]);
        let x1021n = _mm_sub_ps(values[10], values[21]);
        let x1120p = _mm_add_ps(values[11], values[20]);
        let x1120n = _mm_sub_ps(values[11], values[20]);
        let x1219p = _mm_add_ps(values[12], values[19]);
        let x1219n = _mm_sub_ps(values[12], values[19]);
        let x1318p = _mm_add_ps(values[13], values[18]);
        let x1318n = _mm_sub_ps(values[13], values[18]);
        let x1417p = _mm_add_ps(values[14], values[17]);
        let x1417n = _mm_sub_ps(values[14], values[17]);
        let x1516p = _mm_add_ps(values[15], values[16]);
        let x1516n = _mm_sub_ps(values[15], values[16]);

        let temp_a1_1 = _mm_mul_ps(self.twiddle1re, x130p);
        let temp_a1_2 = _mm_mul_ps(self.twiddle2re, x229p);
        let temp_a1_3 = _mm_mul_ps(self.twiddle3re, x328p);
        let temp_a1_4 = _mm_mul_ps(self.twiddle4re, x427p);
        let temp_a1_5 = _mm_mul_ps(self.twiddle5re, x526p);
        let temp_a1_6 = _mm_mul_ps(self.twiddle6re, x625p);
        let temp_a1_7 = _mm_mul_ps(self.twiddle7re, x724p);
        let temp_a1_8 = _mm_mul_ps(self.twiddle8re, x823p);
        let temp_a1_9 = _mm_mul_ps(self.twiddle9re, x922p);
        let temp_a1_10 = _mm_mul_ps(self.twiddle10re, x1021p);
        let temp_a1_11 = _mm_mul_ps(self.twiddle11re, x1120p);
        let temp_a1_12 = _mm_mul_ps(self.twiddle12re, x1219p);
        let temp_a1_13 = _mm_mul_ps(self.twiddle13re, x1318p);
        let temp_a1_14 = _mm_mul_ps(self.twiddle14re, x1417p);
        let temp_a1_15 = _mm_mul_ps(self.twiddle15re, x1516p);
        let temp_a2_1 = _mm_mul_ps(self.twiddle2re, x130p);
        let temp_a2_2 = _mm_mul_ps(self.twiddle4re, x229p);
        let temp_a2_3 = _mm_mul_ps(self.twiddle6re, x328p);
        let temp_a2_4 = _mm_mul_ps(self.twiddle8re, x427p);
        let temp_a2_5 = _mm_mul_ps(self.twiddle10re, x526p);
        let temp_a2_6 = _mm_mul_ps(self.twiddle12re, x625p);
        let temp_a2_7 = _mm_mul_ps(self.twiddle14re, x724p);
        let temp_a2_8 = _mm_mul_ps(self.twiddle15re, x823p);
        let temp_a2_9 = _mm_mul_ps(self.twiddle13re, x922p);
        let temp_a2_10 = _mm_mul_ps(self.twiddle11re, x1021p);
        let temp_a2_11 = _mm_mul_ps(self.twiddle9re, x1120p);
        let temp_a2_12 = _mm_mul_ps(self.twiddle7re, x1219p);
        let temp_a2_13 = _mm_mul_ps(self.twiddle5re, x1318p);
        let temp_a2_14 = _mm_mul_ps(self.twiddle3re, x1417p);
        let temp_a2_15 = _mm_mul_ps(self.twiddle1re, x1516p);
        let temp_a3_1 = _mm_mul_ps(self.twiddle3re, x130p);
        let temp_a3_2 = _mm_mul_ps(self.twiddle6re, x229p);
        let temp_a3_3 = _mm_mul_ps(self.twiddle9re, x328p);
        let temp_a3_4 = _mm_mul_ps(self.twiddle12re, x427p);
        let temp_a3_5 = _mm_mul_ps(self.twiddle15re, x526p);
        let temp_a3_6 = _mm_mul_ps(self.twiddle13re, x625p);
        let temp_a3_7 = _mm_mul_ps(self.twiddle10re, x724p);
        let temp_a3_8 = _mm_mul_ps(self.twiddle7re, x823p);
        let temp_a3_9 = _mm_mul_ps(self.twiddle4re, x922p);
        let temp_a3_10 = _mm_mul_ps(self.twiddle1re, x1021p);
        let temp_a3_11 = _mm_mul_ps(self.twiddle2re, x1120p);
        let temp_a3_12 = _mm_mul_ps(self.twiddle5re, x1219p);
        let temp_a3_13 = _mm_mul_ps(self.twiddle8re, x1318p);
        let temp_a3_14 = _mm_mul_ps(self.twiddle11re, x1417p);
        let temp_a3_15 = _mm_mul_ps(self.twiddle14re, x1516p);
        let temp_a4_1 = _mm_mul_ps(self.twiddle4re, x130p);
        let temp_a4_2 = _mm_mul_ps(self.twiddle8re, x229p);
        let temp_a4_3 = _mm_mul_ps(self.twiddle12re, x328p);
        let temp_a4_4 = _mm_mul_ps(self.twiddle15re, x427p);
        let temp_a4_5 = _mm_mul_ps(self.twiddle11re, x526p);
        let temp_a4_6 = _mm_mul_ps(self.twiddle7re, x625p);
        let temp_a4_7 = _mm_mul_ps(self.twiddle3re, x724p);
        let temp_a4_8 = _mm_mul_ps(self.twiddle1re, x823p);
        let temp_a4_9 = _mm_mul_ps(self.twiddle5re, x922p);
        let temp_a4_10 = _mm_mul_ps(self.twiddle9re, x1021p);
        let temp_a4_11 = _mm_mul_ps(self.twiddle13re, x1120p);
        let temp_a4_12 = _mm_mul_ps(self.twiddle14re, x1219p);
        let temp_a4_13 = _mm_mul_ps(self.twiddle10re, x1318p);
        let temp_a4_14 = _mm_mul_ps(self.twiddle6re, x1417p);
        let temp_a4_15 = _mm_mul_ps(self.twiddle2re, x1516p);
        let temp_a5_1 = _mm_mul_ps(self.twiddle5re, x130p);
        let temp_a5_2 = _mm_mul_ps(self.twiddle10re, x229p);
        let temp_a5_3 = _mm_mul_ps(self.twiddle15re, x328p);
        let temp_a5_4 = _mm_mul_ps(self.twiddle11re, x427p);
        let temp_a5_5 = _mm_mul_ps(self.twiddle6re, x526p);
        let temp_a5_6 = _mm_mul_ps(self.twiddle1re, x625p);
        let temp_a5_7 = _mm_mul_ps(self.twiddle4re, x724p);
        let temp_a5_8 = _mm_mul_ps(self.twiddle9re, x823p);
        let temp_a5_9 = _mm_mul_ps(self.twiddle14re, x922p);
        let temp_a5_10 = _mm_mul_ps(self.twiddle12re, x1021p);
        let temp_a5_11 = _mm_mul_ps(self.twiddle7re, x1120p);
        let temp_a5_12 = _mm_mul_ps(self.twiddle2re, x1219p);
        let temp_a5_13 = _mm_mul_ps(self.twiddle3re, x1318p);
        let temp_a5_14 = _mm_mul_ps(self.twiddle8re, x1417p);
        let temp_a5_15 = _mm_mul_ps(self.twiddle13re, x1516p);
        let temp_a6_1 = _mm_mul_ps(self.twiddle6re, x130p);
        let temp_a6_2 = _mm_mul_ps(self.twiddle12re, x229p);
        let temp_a6_3 = _mm_mul_ps(self.twiddle13re, x328p);
        let temp_a6_4 = _mm_mul_ps(self.twiddle7re, x427p);
        let temp_a6_5 = _mm_mul_ps(self.twiddle1re, x526p);
        let temp_a6_6 = _mm_mul_ps(self.twiddle5re, x625p);
        let temp_a6_7 = _mm_mul_ps(self.twiddle11re, x724p);
        let temp_a6_8 = _mm_mul_ps(self.twiddle14re, x823p);
        let temp_a6_9 = _mm_mul_ps(self.twiddle8re, x922p);
        let temp_a6_10 = _mm_mul_ps(self.twiddle2re, x1021p);
        let temp_a6_11 = _mm_mul_ps(self.twiddle4re, x1120p);
        let temp_a6_12 = _mm_mul_ps(self.twiddle10re, x1219p);
        let temp_a6_13 = _mm_mul_ps(self.twiddle15re, x1318p);
        let temp_a6_14 = _mm_mul_ps(self.twiddle9re, x1417p);
        let temp_a6_15 = _mm_mul_ps(self.twiddle3re, x1516p);
        let temp_a7_1 = _mm_mul_ps(self.twiddle7re, x130p);
        let temp_a7_2 = _mm_mul_ps(self.twiddle14re, x229p);
        let temp_a7_3 = _mm_mul_ps(self.twiddle10re, x328p);
        let temp_a7_4 = _mm_mul_ps(self.twiddle3re, x427p);
        let temp_a7_5 = _mm_mul_ps(self.twiddle4re, x526p);
        let temp_a7_6 = _mm_mul_ps(self.twiddle11re, x625p);
        let temp_a7_7 = _mm_mul_ps(self.twiddle13re, x724p);
        let temp_a7_8 = _mm_mul_ps(self.twiddle6re, x823p);
        let temp_a7_9 = _mm_mul_ps(self.twiddle1re, x922p);
        let temp_a7_10 = _mm_mul_ps(self.twiddle8re, x1021p);
        let temp_a7_11 = _mm_mul_ps(self.twiddle15re, x1120p);
        let temp_a7_12 = _mm_mul_ps(self.twiddle9re, x1219p);
        let temp_a7_13 = _mm_mul_ps(self.twiddle2re, x1318p);
        let temp_a7_14 = _mm_mul_ps(self.twiddle5re, x1417p);
        let temp_a7_15 = _mm_mul_ps(self.twiddle12re, x1516p);
        let temp_a8_1 = _mm_mul_ps(self.twiddle8re, x130p);
        let temp_a8_2 = _mm_mul_ps(self.twiddle15re, x229p);
        let temp_a8_3 = _mm_mul_ps(self.twiddle7re, x328p);
        let temp_a8_4 = _mm_mul_ps(self.twiddle1re, x427p);
        let temp_a8_5 = _mm_mul_ps(self.twiddle9re, x526p);
        let temp_a8_6 = _mm_mul_ps(self.twiddle14re, x625p);
        let temp_a8_7 = _mm_mul_ps(self.twiddle6re, x724p);
        let temp_a8_8 = _mm_mul_ps(self.twiddle2re, x823p);
        let temp_a8_9 = _mm_mul_ps(self.twiddle10re, x922p);
        let temp_a8_10 = _mm_mul_ps(self.twiddle13re, x1021p);
        let temp_a8_11 = _mm_mul_ps(self.twiddle5re, x1120p);
        let temp_a8_12 = _mm_mul_ps(self.twiddle3re, x1219p);
        let temp_a8_13 = _mm_mul_ps(self.twiddle11re, x1318p);
        let temp_a8_14 = _mm_mul_ps(self.twiddle12re, x1417p);
        let temp_a8_15 = _mm_mul_ps(self.twiddle4re, x1516p);
        let temp_a9_1 = _mm_mul_ps(self.twiddle9re, x130p);
        let temp_a9_2 = _mm_mul_ps(self.twiddle13re, x229p);
        let temp_a9_3 = _mm_mul_ps(self.twiddle4re, x328p);
        let temp_a9_4 = _mm_mul_ps(self.twiddle5re, x427p);
        let temp_a9_5 = _mm_mul_ps(self.twiddle14re, x526p);
        let temp_a9_6 = _mm_mul_ps(self.twiddle8re, x625p);
        let temp_a9_7 = _mm_mul_ps(self.twiddle1re, x724p);
        let temp_a9_8 = _mm_mul_ps(self.twiddle10re, x823p);
        let temp_a9_9 = _mm_mul_ps(self.twiddle12re, x922p);
        let temp_a9_10 = _mm_mul_ps(self.twiddle3re, x1021p);
        let temp_a9_11 = _mm_mul_ps(self.twiddle6re, x1120p);
        let temp_a9_12 = _mm_mul_ps(self.twiddle15re, x1219p);
        let temp_a9_13 = _mm_mul_ps(self.twiddle7re, x1318p);
        let temp_a9_14 = _mm_mul_ps(self.twiddle2re, x1417p);
        let temp_a9_15 = _mm_mul_ps(self.twiddle11re, x1516p);
        let temp_a10_1 = _mm_mul_ps(self.twiddle10re, x130p);
        let temp_a10_2 = _mm_mul_ps(self.twiddle11re, x229p);
        let temp_a10_3 = _mm_mul_ps(self.twiddle1re, x328p);
        let temp_a10_4 = _mm_mul_ps(self.twiddle9re, x427p);
        let temp_a10_5 = _mm_mul_ps(self.twiddle12re, x526p);
        let temp_a10_6 = _mm_mul_ps(self.twiddle2re, x625p);
        let temp_a10_7 = _mm_mul_ps(self.twiddle8re, x724p);
        let temp_a10_8 = _mm_mul_ps(self.twiddle13re, x823p);
        let temp_a10_9 = _mm_mul_ps(self.twiddle3re, x922p);
        let temp_a10_10 = _mm_mul_ps(self.twiddle7re, x1021p);
        let temp_a10_11 = _mm_mul_ps(self.twiddle14re, x1120p);
        let temp_a10_12 = _mm_mul_ps(self.twiddle4re, x1219p);
        let temp_a10_13 = _mm_mul_ps(self.twiddle6re, x1318p);
        let temp_a10_14 = _mm_mul_ps(self.twiddle15re, x1417p);
        let temp_a10_15 = _mm_mul_ps(self.twiddle5re, x1516p);
        let temp_a11_1 = _mm_mul_ps(self.twiddle11re, x130p);
        let temp_a11_2 = _mm_mul_ps(self.twiddle9re, x229p);
        let temp_a11_3 = _mm_mul_ps(self.twiddle2re, x328p);
        let temp_a11_4 = _mm_mul_ps(self.twiddle13re, x427p);
        let temp_a11_5 = _mm_mul_ps(self.twiddle7re, x526p);
        let temp_a11_6 = _mm_mul_ps(self.twiddle4re, x625p);
        let temp_a11_7 = _mm_mul_ps(self.twiddle15re, x724p);
        let temp_a11_8 = _mm_mul_ps(self.twiddle5re, x823p);
        let temp_a11_9 = _mm_mul_ps(self.twiddle6re, x922p);
        let temp_a11_10 = _mm_mul_ps(self.twiddle14re, x1021p);
        let temp_a11_11 = _mm_mul_ps(self.twiddle3re, x1120p);
        let temp_a11_12 = _mm_mul_ps(self.twiddle8re, x1219p);
        let temp_a11_13 = _mm_mul_ps(self.twiddle12re, x1318p);
        let temp_a11_14 = _mm_mul_ps(self.twiddle1re, x1417p);
        let temp_a11_15 = _mm_mul_ps(self.twiddle10re, x1516p);
        let temp_a12_1 = _mm_mul_ps(self.twiddle12re, x130p);
        let temp_a12_2 = _mm_mul_ps(self.twiddle7re, x229p);
        let temp_a12_3 = _mm_mul_ps(self.twiddle5re, x328p);
        let temp_a12_4 = _mm_mul_ps(self.twiddle14re, x427p);
        let temp_a12_5 = _mm_mul_ps(self.twiddle2re, x526p);
        let temp_a12_6 = _mm_mul_ps(self.twiddle10re, x625p);
        let temp_a12_7 = _mm_mul_ps(self.twiddle9re, x724p);
        let temp_a12_8 = _mm_mul_ps(self.twiddle3re, x823p);
        let temp_a12_9 = _mm_mul_ps(self.twiddle15re, x922p);
        let temp_a12_10 = _mm_mul_ps(self.twiddle4re, x1021p);
        let temp_a12_11 = _mm_mul_ps(self.twiddle8re, x1120p);
        let temp_a12_12 = _mm_mul_ps(self.twiddle11re, x1219p);
        let temp_a12_13 = _mm_mul_ps(self.twiddle1re, x1318p);
        let temp_a12_14 = _mm_mul_ps(self.twiddle13re, x1417p);
        let temp_a12_15 = _mm_mul_ps(self.twiddle6re, x1516p);
        let temp_a13_1 = _mm_mul_ps(self.twiddle13re, x130p);
        let temp_a13_2 = _mm_mul_ps(self.twiddle5re, x229p);
        let temp_a13_3 = _mm_mul_ps(self.twiddle8re, x328p);
        let temp_a13_4 = _mm_mul_ps(self.twiddle10re, x427p);
        let temp_a13_5 = _mm_mul_ps(self.twiddle3re, x526p);
        let temp_a13_6 = _mm_mul_ps(self.twiddle15re, x625p);
        let temp_a13_7 = _mm_mul_ps(self.twiddle2re, x724p);
        let temp_a13_8 = _mm_mul_ps(self.twiddle11re, x823p);
        let temp_a13_9 = _mm_mul_ps(self.twiddle7re, x922p);
        let temp_a13_10 = _mm_mul_ps(self.twiddle6re, x1021p);
        let temp_a13_11 = _mm_mul_ps(self.twiddle12re, x1120p);
        let temp_a13_12 = _mm_mul_ps(self.twiddle1re, x1219p);
        let temp_a13_13 = _mm_mul_ps(self.twiddle14re, x1318p);
        let temp_a13_14 = _mm_mul_ps(self.twiddle4re, x1417p);
        let temp_a13_15 = _mm_mul_ps(self.twiddle9re, x1516p);
        let temp_a14_1 = _mm_mul_ps(self.twiddle14re, x130p);
        let temp_a14_2 = _mm_mul_ps(self.twiddle3re, x229p);
        let temp_a14_3 = _mm_mul_ps(self.twiddle11re, x328p);
        let temp_a14_4 = _mm_mul_ps(self.twiddle6re, x427p);
        let temp_a14_5 = _mm_mul_ps(self.twiddle8re, x526p);
        let temp_a14_6 = _mm_mul_ps(self.twiddle9re, x625p);
        let temp_a14_7 = _mm_mul_ps(self.twiddle5re, x724p);
        let temp_a14_8 = _mm_mul_ps(self.twiddle12re, x823p);
        let temp_a14_9 = _mm_mul_ps(self.twiddle2re, x922p);
        let temp_a14_10 = _mm_mul_ps(self.twiddle15re, x1021p);
        let temp_a14_11 = _mm_mul_ps(self.twiddle1re, x1120p);
        let temp_a14_12 = _mm_mul_ps(self.twiddle13re, x1219p);
        let temp_a14_13 = _mm_mul_ps(self.twiddle4re, x1318p);
        let temp_a14_14 = _mm_mul_ps(self.twiddle10re, x1417p);
        let temp_a14_15 = _mm_mul_ps(self.twiddle7re, x1516p);
        let temp_a15_1 = _mm_mul_ps(self.twiddle15re, x130p);
        let temp_a15_2 = _mm_mul_ps(self.twiddle1re, x229p);
        let temp_a15_3 = _mm_mul_ps(self.twiddle14re, x328p);
        let temp_a15_4 = _mm_mul_ps(self.twiddle2re, x427p);
        let temp_a15_5 = _mm_mul_ps(self.twiddle13re, x526p);
        let temp_a15_6 = _mm_mul_ps(self.twiddle3re, x625p);
        let temp_a15_7 = _mm_mul_ps(self.twiddle12re, x724p);
        let temp_a15_8 = _mm_mul_ps(self.twiddle4re, x823p);
        let temp_a15_9 = _mm_mul_ps(self.twiddle11re, x922p);
        let temp_a15_10 = _mm_mul_ps(self.twiddle5re, x1021p);
        let temp_a15_11 = _mm_mul_ps(self.twiddle10re, x1120p);
        let temp_a15_12 = _mm_mul_ps(self.twiddle6re, x1219p);
        let temp_a15_13 = _mm_mul_ps(self.twiddle9re, x1318p);
        let temp_a15_14 = _mm_mul_ps(self.twiddle7re, x1417p);
        let temp_a15_15 = _mm_mul_ps(self.twiddle8re, x1516p);

        let temp_b1_1 = _mm_mul_ps(self.twiddle1im, x130n);
        let temp_b1_2 = _mm_mul_ps(self.twiddle2im, x229n);
        let temp_b1_3 = _mm_mul_ps(self.twiddle3im, x328n);
        let temp_b1_4 = _mm_mul_ps(self.twiddle4im, x427n);
        let temp_b1_5 = _mm_mul_ps(self.twiddle5im, x526n);
        let temp_b1_6 = _mm_mul_ps(self.twiddle6im, x625n);
        let temp_b1_7 = _mm_mul_ps(self.twiddle7im, x724n);
        let temp_b1_8 = _mm_mul_ps(self.twiddle8im, x823n);
        let temp_b1_9 = _mm_mul_ps(self.twiddle9im, x922n);
        let temp_b1_10 = _mm_mul_ps(self.twiddle10im, x1021n);
        let temp_b1_11 = _mm_mul_ps(self.twiddle11im, x1120n);
        let temp_b1_12 = _mm_mul_ps(self.twiddle12im, x1219n);
        let temp_b1_13 = _mm_mul_ps(self.twiddle13im, x1318n);
        let temp_b1_14 = _mm_mul_ps(self.twiddle14im, x1417n);
        let temp_b1_15 = _mm_mul_ps(self.twiddle15im, x1516n);
        let temp_b2_1 = _mm_mul_ps(self.twiddle2im, x130n);
        let temp_b2_2 = _mm_mul_ps(self.twiddle4im, x229n);
        let temp_b2_3 = _mm_mul_ps(self.twiddle6im, x328n);
        let temp_b2_4 = _mm_mul_ps(self.twiddle8im, x427n);
        let temp_b2_5 = _mm_mul_ps(self.twiddle10im, x526n);
        let temp_b2_6 = _mm_mul_ps(self.twiddle12im, x625n);
        let temp_b2_7 = _mm_mul_ps(self.twiddle14im, x724n);
        let temp_b2_8 = _mm_mul_ps(self.twiddle15im, x823n);
        let temp_b2_9 = _mm_mul_ps(self.twiddle13im, x922n);
        let temp_b2_10 = _mm_mul_ps(self.twiddle11im, x1021n);
        let temp_b2_11 = _mm_mul_ps(self.twiddle9im, x1120n);
        let temp_b2_12 = _mm_mul_ps(self.twiddle7im, x1219n);
        let temp_b2_13 = _mm_mul_ps(self.twiddle5im, x1318n);
        let temp_b2_14 = _mm_mul_ps(self.twiddle3im, x1417n);
        let temp_b2_15 = _mm_mul_ps(self.twiddle1im, x1516n);
        let temp_b3_1 = _mm_mul_ps(self.twiddle3im, x130n);
        let temp_b3_2 = _mm_mul_ps(self.twiddle6im, x229n);
        let temp_b3_3 = _mm_mul_ps(self.twiddle9im, x328n);
        let temp_b3_4 = _mm_mul_ps(self.twiddle12im, x427n);
        let temp_b3_5 = _mm_mul_ps(self.twiddle15im, x526n);
        let temp_b3_6 = _mm_mul_ps(self.twiddle13im, x625n);
        let temp_b3_7 = _mm_mul_ps(self.twiddle10im, x724n);
        let temp_b3_8 = _mm_mul_ps(self.twiddle7im, x823n);
        let temp_b3_9 = _mm_mul_ps(self.twiddle4im, x922n);
        let temp_b3_10 = _mm_mul_ps(self.twiddle1im, x1021n);
        let temp_b3_11 = _mm_mul_ps(self.twiddle2im, x1120n);
        let temp_b3_12 = _mm_mul_ps(self.twiddle5im, x1219n);
        let temp_b3_13 = _mm_mul_ps(self.twiddle8im, x1318n);
        let temp_b3_14 = _mm_mul_ps(self.twiddle11im, x1417n);
        let temp_b3_15 = _mm_mul_ps(self.twiddle14im, x1516n);
        let temp_b4_1 = _mm_mul_ps(self.twiddle4im, x130n);
        let temp_b4_2 = _mm_mul_ps(self.twiddle8im, x229n);
        let temp_b4_3 = _mm_mul_ps(self.twiddle12im, x328n);
        let temp_b4_4 = _mm_mul_ps(self.twiddle15im, x427n);
        let temp_b4_5 = _mm_mul_ps(self.twiddle11im, x526n);
        let temp_b4_6 = _mm_mul_ps(self.twiddle7im, x625n);
        let temp_b4_7 = _mm_mul_ps(self.twiddle3im, x724n);
        let temp_b4_8 = _mm_mul_ps(self.twiddle1im, x823n);
        let temp_b4_9 = _mm_mul_ps(self.twiddle5im, x922n);
        let temp_b4_10 = _mm_mul_ps(self.twiddle9im, x1021n);
        let temp_b4_11 = _mm_mul_ps(self.twiddle13im, x1120n);
        let temp_b4_12 = _mm_mul_ps(self.twiddle14im, x1219n);
        let temp_b4_13 = _mm_mul_ps(self.twiddle10im, x1318n);
        let temp_b4_14 = _mm_mul_ps(self.twiddle6im, x1417n);
        let temp_b4_15 = _mm_mul_ps(self.twiddle2im, x1516n);
        let temp_b5_1 = _mm_mul_ps(self.twiddle5im, x130n);
        let temp_b5_2 = _mm_mul_ps(self.twiddle10im, x229n);
        let temp_b5_3 = _mm_mul_ps(self.twiddle15im, x328n);
        let temp_b5_4 = _mm_mul_ps(self.twiddle11im, x427n);
        let temp_b5_5 = _mm_mul_ps(self.twiddle6im, x526n);
        let temp_b5_6 = _mm_mul_ps(self.twiddle1im, x625n);
        let temp_b5_7 = _mm_mul_ps(self.twiddle4im, x724n);
        let temp_b5_8 = _mm_mul_ps(self.twiddle9im, x823n);
        let temp_b5_9 = _mm_mul_ps(self.twiddle14im, x922n);
        let temp_b5_10 = _mm_mul_ps(self.twiddle12im, x1021n);
        let temp_b5_11 = _mm_mul_ps(self.twiddle7im, x1120n);
        let temp_b5_12 = _mm_mul_ps(self.twiddle2im, x1219n);
        let temp_b5_13 = _mm_mul_ps(self.twiddle3im, x1318n);
        let temp_b5_14 = _mm_mul_ps(self.twiddle8im, x1417n);
        let temp_b5_15 = _mm_mul_ps(self.twiddle13im, x1516n);
        let temp_b6_1 = _mm_mul_ps(self.twiddle6im, x130n);
        let temp_b6_2 = _mm_mul_ps(self.twiddle12im, x229n);
        let temp_b6_3 = _mm_mul_ps(self.twiddle13im, x328n);
        let temp_b6_4 = _mm_mul_ps(self.twiddle7im, x427n);
        let temp_b6_5 = _mm_mul_ps(self.twiddle1im, x526n);
        let temp_b6_6 = _mm_mul_ps(self.twiddle5im, x625n);
        let temp_b6_7 = _mm_mul_ps(self.twiddle11im, x724n);
        let temp_b6_8 = _mm_mul_ps(self.twiddle14im, x823n);
        let temp_b6_9 = _mm_mul_ps(self.twiddle8im, x922n);
        let temp_b6_10 = _mm_mul_ps(self.twiddle2im, x1021n);
        let temp_b6_11 = _mm_mul_ps(self.twiddle4im, x1120n);
        let temp_b6_12 = _mm_mul_ps(self.twiddle10im, x1219n);
        let temp_b6_13 = _mm_mul_ps(self.twiddle15im, x1318n);
        let temp_b6_14 = _mm_mul_ps(self.twiddle9im, x1417n);
        let temp_b6_15 = _mm_mul_ps(self.twiddle3im, x1516n);
        let temp_b7_1 = _mm_mul_ps(self.twiddle7im, x130n);
        let temp_b7_2 = _mm_mul_ps(self.twiddle14im, x229n);
        let temp_b7_3 = _mm_mul_ps(self.twiddle10im, x328n);
        let temp_b7_4 = _mm_mul_ps(self.twiddle3im, x427n);
        let temp_b7_5 = _mm_mul_ps(self.twiddle4im, x526n);
        let temp_b7_6 = _mm_mul_ps(self.twiddle11im, x625n);
        let temp_b7_7 = _mm_mul_ps(self.twiddle13im, x724n);
        let temp_b7_8 = _mm_mul_ps(self.twiddle6im, x823n);
        let temp_b7_9 = _mm_mul_ps(self.twiddle1im, x922n);
        let temp_b7_10 = _mm_mul_ps(self.twiddle8im, x1021n);
        let temp_b7_11 = _mm_mul_ps(self.twiddle15im, x1120n);
        let temp_b7_12 = _mm_mul_ps(self.twiddle9im, x1219n);
        let temp_b7_13 = _mm_mul_ps(self.twiddle2im, x1318n);
        let temp_b7_14 = _mm_mul_ps(self.twiddle5im, x1417n);
        let temp_b7_15 = _mm_mul_ps(self.twiddle12im, x1516n);
        let temp_b8_1 = _mm_mul_ps(self.twiddle8im, x130n);
        let temp_b8_2 = _mm_mul_ps(self.twiddle15im, x229n);
        let temp_b8_3 = _mm_mul_ps(self.twiddle7im, x328n);
        let temp_b8_4 = _mm_mul_ps(self.twiddle1im, x427n);
        let temp_b8_5 = _mm_mul_ps(self.twiddle9im, x526n);
        let temp_b8_6 = _mm_mul_ps(self.twiddle14im, x625n);
        let temp_b8_7 = _mm_mul_ps(self.twiddle6im, x724n);
        let temp_b8_8 = _mm_mul_ps(self.twiddle2im, x823n);
        let temp_b8_9 = _mm_mul_ps(self.twiddle10im, x922n);
        let temp_b8_10 = _mm_mul_ps(self.twiddle13im, x1021n);
        let temp_b8_11 = _mm_mul_ps(self.twiddle5im, x1120n);
        let temp_b8_12 = _mm_mul_ps(self.twiddle3im, x1219n);
        let temp_b8_13 = _mm_mul_ps(self.twiddle11im, x1318n);
        let temp_b8_14 = _mm_mul_ps(self.twiddle12im, x1417n);
        let temp_b8_15 = _mm_mul_ps(self.twiddle4im, x1516n);
        let temp_b9_1 = _mm_mul_ps(self.twiddle9im, x130n);
        let temp_b9_2 = _mm_mul_ps(self.twiddle13im, x229n);
        let temp_b9_3 = _mm_mul_ps(self.twiddle4im, x328n);
        let temp_b9_4 = _mm_mul_ps(self.twiddle5im, x427n);
        let temp_b9_5 = _mm_mul_ps(self.twiddle14im, x526n);
        let temp_b9_6 = _mm_mul_ps(self.twiddle8im, x625n);
        let temp_b9_7 = _mm_mul_ps(self.twiddle1im, x724n);
        let temp_b9_8 = _mm_mul_ps(self.twiddle10im, x823n);
        let temp_b9_9 = _mm_mul_ps(self.twiddle12im, x922n);
        let temp_b9_10 = _mm_mul_ps(self.twiddle3im, x1021n);
        let temp_b9_11 = _mm_mul_ps(self.twiddle6im, x1120n);
        let temp_b9_12 = _mm_mul_ps(self.twiddle15im, x1219n);
        let temp_b9_13 = _mm_mul_ps(self.twiddle7im, x1318n);
        let temp_b9_14 = _mm_mul_ps(self.twiddle2im, x1417n);
        let temp_b9_15 = _mm_mul_ps(self.twiddle11im, x1516n);
        let temp_b10_1 = _mm_mul_ps(self.twiddle10im, x130n);
        let temp_b10_2 = _mm_mul_ps(self.twiddle11im, x229n);
        let temp_b10_3 = _mm_mul_ps(self.twiddle1im, x328n);
        let temp_b10_4 = _mm_mul_ps(self.twiddle9im, x427n);
        let temp_b10_5 = _mm_mul_ps(self.twiddle12im, x526n);
        let temp_b10_6 = _mm_mul_ps(self.twiddle2im, x625n);
        let temp_b10_7 = _mm_mul_ps(self.twiddle8im, x724n);
        let temp_b10_8 = _mm_mul_ps(self.twiddle13im, x823n);
        let temp_b10_9 = _mm_mul_ps(self.twiddle3im, x922n);
        let temp_b10_10 = _mm_mul_ps(self.twiddle7im, x1021n);
        let temp_b10_11 = _mm_mul_ps(self.twiddle14im, x1120n);
        let temp_b10_12 = _mm_mul_ps(self.twiddle4im, x1219n);
        let temp_b10_13 = _mm_mul_ps(self.twiddle6im, x1318n);
        let temp_b10_14 = _mm_mul_ps(self.twiddle15im, x1417n);
        let temp_b10_15 = _mm_mul_ps(self.twiddle5im, x1516n);
        let temp_b11_1 = _mm_mul_ps(self.twiddle11im, x130n);
        let temp_b11_2 = _mm_mul_ps(self.twiddle9im, x229n);
        let temp_b11_3 = _mm_mul_ps(self.twiddle2im, x328n);
        let temp_b11_4 = _mm_mul_ps(self.twiddle13im, x427n);
        let temp_b11_5 = _mm_mul_ps(self.twiddle7im, x526n);
        let temp_b11_6 = _mm_mul_ps(self.twiddle4im, x625n);
        let temp_b11_7 = _mm_mul_ps(self.twiddle15im, x724n);
        let temp_b11_8 = _mm_mul_ps(self.twiddle5im, x823n);
        let temp_b11_9 = _mm_mul_ps(self.twiddle6im, x922n);
        let temp_b11_10 = _mm_mul_ps(self.twiddle14im, x1021n);
        let temp_b11_11 = _mm_mul_ps(self.twiddle3im, x1120n);
        let temp_b11_12 = _mm_mul_ps(self.twiddle8im, x1219n);
        let temp_b11_13 = _mm_mul_ps(self.twiddle12im, x1318n);
        let temp_b11_14 = _mm_mul_ps(self.twiddle1im, x1417n);
        let temp_b11_15 = _mm_mul_ps(self.twiddle10im, x1516n);
        let temp_b12_1 = _mm_mul_ps(self.twiddle12im, x130n);
        let temp_b12_2 = _mm_mul_ps(self.twiddle7im, x229n);
        let temp_b12_3 = _mm_mul_ps(self.twiddle5im, x328n);
        let temp_b12_4 = _mm_mul_ps(self.twiddle14im, x427n);
        let temp_b12_5 = _mm_mul_ps(self.twiddle2im, x526n);
        let temp_b12_6 = _mm_mul_ps(self.twiddle10im, x625n);
        let temp_b12_7 = _mm_mul_ps(self.twiddle9im, x724n);
        let temp_b12_8 = _mm_mul_ps(self.twiddle3im, x823n);
        let temp_b12_9 = _mm_mul_ps(self.twiddle15im, x922n);
        let temp_b12_10 = _mm_mul_ps(self.twiddle4im, x1021n);
        let temp_b12_11 = _mm_mul_ps(self.twiddle8im, x1120n);
        let temp_b12_12 = _mm_mul_ps(self.twiddle11im, x1219n);
        let temp_b12_13 = _mm_mul_ps(self.twiddle1im, x1318n);
        let temp_b12_14 = _mm_mul_ps(self.twiddle13im, x1417n);
        let temp_b12_15 = _mm_mul_ps(self.twiddle6im, x1516n);
        let temp_b13_1 = _mm_mul_ps(self.twiddle13im, x130n);
        let temp_b13_2 = _mm_mul_ps(self.twiddle5im, x229n);
        let temp_b13_3 = _mm_mul_ps(self.twiddle8im, x328n);
        let temp_b13_4 = _mm_mul_ps(self.twiddle10im, x427n);
        let temp_b13_5 = _mm_mul_ps(self.twiddle3im, x526n);
        let temp_b13_6 = _mm_mul_ps(self.twiddle15im, x625n);
        let temp_b13_7 = _mm_mul_ps(self.twiddle2im, x724n);
        let temp_b13_8 = _mm_mul_ps(self.twiddle11im, x823n);
        let temp_b13_9 = _mm_mul_ps(self.twiddle7im, x922n);
        let temp_b13_10 = _mm_mul_ps(self.twiddle6im, x1021n);
        let temp_b13_11 = _mm_mul_ps(self.twiddle12im, x1120n);
        let temp_b13_12 = _mm_mul_ps(self.twiddle1im, x1219n);
        let temp_b13_13 = _mm_mul_ps(self.twiddle14im, x1318n);
        let temp_b13_14 = _mm_mul_ps(self.twiddle4im, x1417n);
        let temp_b13_15 = _mm_mul_ps(self.twiddle9im, x1516n);
        let temp_b14_1 = _mm_mul_ps(self.twiddle14im, x130n);
        let temp_b14_2 = _mm_mul_ps(self.twiddle3im, x229n);
        let temp_b14_3 = _mm_mul_ps(self.twiddle11im, x328n);
        let temp_b14_4 = _mm_mul_ps(self.twiddle6im, x427n);
        let temp_b14_5 = _mm_mul_ps(self.twiddle8im, x526n);
        let temp_b14_6 = _mm_mul_ps(self.twiddle9im, x625n);
        let temp_b14_7 = _mm_mul_ps(self.twiddle5im, x724n);
        let temp_b14_8 = _mm_mul_ps(self.twiddle12im, x823n);
        let temp_b14_9 = _mm_mul_ps(self.twiddle2im, x922n);
        let temp_b14_10 = _mm_mul_ps(self.twiddle15im, x1021n);
        let temp_b14_11 = _mm_mul_ps(self.twiddle1im, x1120n);
        let temp_b14_12 = _mm_mul_ps(self.twiddle13im, x1219n);
        let temp_b14_13 = _mm_mul_ps(self.twiddle4im, x1318n);
        let temp_b14_14 = _mm_mul_ps(self.twiddle10im, x1417n);
        let temp_b14_15 = _mm_mul_ps(self.twiddle7im, x1516n);
        let temp_b15_1 = _mm_mul_ps(self.twiddle15im, x130n);
        let temp_b15_2 = _mm_mul_ps(self.twiddle1im, x229n);
        let temp_b15_3 = _mm_mul_ps(self.twiddle14im, x328n);
        let temp_b15_4 = _mm_mul_ps(self.twiddle2im, x427n);
        let temp_b15_5 = _mm_mul_ps(self.twiddle13im, x526n);
        let temp_b15_6 = _mm_mul_ps(self.twiddle3im, x625n);
        let temp_b15_7 = _mm_mul_ps(self.twiddle12im, x724n);
        let temp_b15_8 = _mm_mul_ps(self.twiddle4im, x823n);
        let temp_b15_9 = _mm_mul_ps(self.twiddle11im, x922n);
        let temp_b15_10 = _mm_mul_ps(self.twiddle5im, x1021n);
        let temp_b15_11 = _mm_mul_ps(self.twiddle10im, x1120n);
        let temp_b15_12 = _mm_mul_ps(self.twiddle6im, x1219n);
        let temp_b15_13 = _mm_mul_ps(self.twiddle9im, x1318n);
        let temp_b15_14 = _mm_mul_ps(self.twiddle7im, x1417n);
        let temp_b15_15 = _mm_mul_ps(self.twiddle8im, x1516n);

        let temp_a1 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a1_1,
                _mm_add_ps(
                    temp_a1_2,
                    _mm_add_ps(
                        temp_a1_3,
                        _mm_add_ps(
                            temp_a1_4,
                            _mm_add_ps(
                                temp_a1_5,
                                _mm_add_ps(
                                    temp_a1_6,
                                    _mm_add_ps(
                                        temp_a1_7,
                                        _mm_add_ps(
                                            temp_a1_8,
                                            _mm_add_ps(
                                                temp_a1_9,
                                                _mm_add_ps(
                                                    temp_a1_10,
                                                    _mm_add_ps(
                                                        temp_a1_11,
                                                        _mm_add_ps(
                                                            temp_a1_12,
                                                            _mm_add_ps(
                                                                temp_a1_13,
                                                                _mm_add_ps(temp_a1_14, temp_a1_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a2_1,
                _mm_add_ps(
                    temp_a2_2,
                    _mm_add_ps(
                        temp_a2_3,
                        _mm_add_ps(
                            temp_a2_4,
                            _mm_add_ps(
                                temp_a2_5,
                                _mm_add_ps(
                                    temp_a2_6,
                                    _mm_add_ps(
                                        temp_a2_7,
                                        _mm_add_ps(
                                            temp_a2_8,
                                            _mm_add_ps(
                                                temp_a2_9,
                                                _mm_add_ps(
                                                    temp_a2_10,
                                                    _mm_add_ps(
                                                        temp_a2_11,
                                                        _mm_add_ps(
                                                            temp_a2_12,
                                                            _mm_add_ps(
                                                                temp_a2_13,
                                                                _mm_add_ps(temp_a2_14, temp_a2_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a3_1,
                _mm_add_ps(
                    temp_a3_2,
                    _mm_add_ps(
                        temp_a3_3,
                        _mm_add_ps(
                            temp_a3_4,
                            _mm_add_ps(
                                temp_a3_5,
                                _mm_add_ps(
                                    temp_a3_6,
                                    _mm_add_ps(
                                        temp_a3_7,
                                        _mm_add_ps(
                                            temp_a3_8,
                                            _mm_add_ps(
                                                temp_a3_9,
                                                _mm_add_ps(
                                                    temp_a3_10,
                                                    _mm_add_ps(
                                                        temp_a3_11,
                                                        _mm_add_ps(
                                                            temp_a3_12,
                                                            _mm_add_ps(
                                                                temp_a3_13,
                                                                _mm_add_ps(temp_a3_14, temp_a3_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a4_1,
                _mm_add_ps(
                    temp_a4_2,
                    _mm_add_ps(
                        temp_a4_3,
                        _mm_add_ps(
                            temp_a4_4,
                            _mm_add_ps(
                                temp_a4_5,
                                _mm_add_ps(
                                    temp_a4_6,
                                    _mm_add_ps(
                                        temp_a4_7,
                                        _mm_add_ps(
                                            temp_a4_8,
                                            _mm_add_ps(
                                                temp_a4_9,
                                                _mm_add_ps(
                                                    temp_a4_10,
                                                    _mm_add_ps(
                                                        temp_a4_11,
                                                        _mm_add_ps(
                                                            temp_a4_12,
                                                            _mm_add_ps(
                                                                temp_a4_13,
                                                                _mm_add_ps(temp_a4_14, temp_a4_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a5_1,
                _mm_add_ps(
                    temp_a5_2,
                    _mm_add_ps(
                        temp_a5_3,
                        _mm_add_ps(
                            temp_a5_4,
                            _mm_add_ps(
                                temp_a5_5,
                                _mm_add_ps(
                                    temp_a5_6,
                                    _mm_add_ps(
                                        temp_a5_7,
                                        _mm_add_ps(
                                            temp_a5_8,
                                            _mm_add_ps(
                                                temp_a5_9,
                                                _mm_add_ps(
                                                    temp_a5_10,
                                                    _mm_add_ps(
                                                        temp_a5_11,
                                                        _mm_add_ps(
                                                            temp_a5_12,
                                                            _mm_add_ps(
                                                                temp_a5_13,
                                                                _mm_add_ps(temp_a5_14, temp_a5_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a6_1,
                _mm_add_ps(
                    temp_a6_2,
                    _mm_add_ps(
                        temp_a6_3,
                        _mm_add_ps(
                            temp_a6_4,
                            _mm_add_ps(
                                temp_a6_5,
                                _mm_add_ps(
                                    temp_a6_6,
                                    _mm_add_ps(
                                        temp_a6_7,
                                        _mm_add_ps(
                                            temp_a6_8,
                                            _mm_add_ps(
                                                temp_a6_9,
                                                _mm_add_ps(
                                                    temp_a6_10,
                                                    _mm_add_ps(
                                                        temp_a6_11,
                                                        _mm_add_ps(
                                                            temp_a6_12,
                                                            _mm_add_ps(
                                                                temp_a6_13,
                                                                _mm_add_ps(temp_a6_14, temp_a6_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a7_1,
                _mm_add_ps(
                    temp_a7_2,
                    _mm_add_ps(
                        temp_a7_3,
                        _mm_add_ps(
                            temp_a7_4,
                            _mm_add_ps(
                                temp_a7_5,
                                _mm_add_ps(
                                    temp_a7_6,
                                    _mm_add_ps(
                                        temp_a7_7,
                                        _mm_add_ps(
                                            temp_a7_8,
                                            _mm_add_ps(
                                                temp_a7_9,
                                                _mm_add_ps(
                                                    temp_a7_10,
                                                    _mm_add_ps(
                                                        temp_a7_11,
                                                        _mm_add_ps(
                                                            temp_a7_12,
                                                            _mm_add_ps(
                                                                temp_a7_13,
                                                                _mm_add_ps(temp_a7_14, temp_a7_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a8_1,
                _mm_add_ps(
                    temp_a8_2,
                    _mm_add_ps(
                        temp_a8_3,
                        _mm_add_ps(
                            temp_a8_4,
                            _mm_add_ps(
                                temp_a8_5,
                                _mm_add_ps(
                                    temp_a8_6,
                                    _mm_add_ps(
                                        temp_a8_7,
                                        _mm_add_ps(
                                            temp_a8_8,
                                            _mm_add_ps(
                                                temp_a8_9,
                                                _mm_add_ps(
                                                    temp_a8_10,
                                                    _mm_add_ps(
                                                        temp_a8_11,
                                                        _mm_add_ps(
                                                            temp_a8_12,
                                                            _mm_add_ps(
                                                                temp_a8_13,
                                                                _mm_add_ps(temp_a8_14, temp_a8_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a9 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a9_1,
                _mm_add_ps(
                    temp_a9_2,
                    _mm_add_ps(
                        temp_a9_3,
                        _mm_add_ps(
                            temp_a9_4,
                            _mm_add_ps(
                                temp_a9_5,
                                _mm_add_ps(
                                    temp_a9_6,
                                    _mm_add_ps(
                                        temp_a9_7,
                                        _mm_add_ps(
                                            temp_a9_8,
                                            _mm_add_ps(
                                                temp_a9_9,
                                                _mm_add_ps(
                                                    temp_a9_10,
                                                    _mm_add_ps(
                                                        temp_a9_11,
                                                        _mm_add_ps(
                                                            temp_a9_12,
                                                            _mm_add_ps(
                                                                temp_a9_13,
                                                                _mm_add_ps(temp_a9_14, temp_a9_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a10 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a10_1,
                _mm_add_ps(
                    temp_a10_2,
                    _mm_add_ps(
                        temp_a10_3,
                        _mm_add_ps(
                            temp_a10_4,
                            _mm_add_ps(
                                temp_a10_5,
                                _mm_add_ps(
                                    temp_a10_6,
                                    _mm_add_ps(
                                        temp_a10_7,
                                        _mm_add_ps(
                                            temp_a10_8,
                                            _mm_add_ps(
                                                temp_a10_9,
                                                _mm_add_ps(
                                                    temp_a10_10,
                                                    _mm_add_ps(
                                                        temp_a10_11,
                                                        _mm_add_ps(
                                                            temp_a10_12,
                                                            _mm_add_ps(
                                                                temp_a10_13,
                                                                _mm_add_ps(
                                                                    temp_a10_14,
                                                                    temp_a10_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a11 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a11_1,
                _mm_add_ps(
                    temp_a11_2,
                    _mm_add_ps(
                        temp_a11_3,
                        _mm_add_ps(
                            temp_a11_4,
                            _mm_add_ps(
                                temp_a11_5,
                                _mm_add_ps(
                                    temp_a11_6,
                                    _mm_add_ps(
                                        temp_a11_7,
                                        _mm_add_ps(
                                            temp_a11_8,
                                            _mm_add_ps(
                                                temp_a11_9,
                                                _mm_add_ps(
                                                    temp_a11_10,
                                                    _mm_add_ps(
                                                        temp_a11_11,
                                                        _mm_add_ps(
                                                            temp_a11_12,
                                                            _mm_add_ps(
                                                                temp_a11_13,
                                                                _mm_add_ps(
                                                                    temp_a11_14,
                                                                    temp_a11_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a12 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a12_1,
                _mm_add_ps(
                    temp_a12_2,
                    _mm_add_ps(
                        temp_a12_3,
                        _mm_add_ps(
                            temp_a12_4,
                            _mm_add_ps(
                                temp_a12_5,
                                _mm_add_ps(
                                    temp_a12_6,
                                    _mm_add_ps(
                                        temp_a12_7,
                                        _mm_add_ps(
                                            temp_a12_8,
                                            _mm_add_ps(
                                                temp_a12_9,
                                                _mm_add_ps(
                                                    temp_a12_10,
                                                    _mm_add_ps(
                                                        temp_a12_11,
                                                        _mm_add_ps(
                                                            temp_a12_12,
                                                            _mm_add_ps(
                                                                temp_a12_13,
                                                                _mm_add_ps(
                                                                    temp_a12_14,
                                                                    temp_a12_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a13 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a13_1,
                _mm_add_ps(
                    temp_a13_2,
                    _mm_add_ps(
                        temp_a13_3,
                        _mm_add_ps(
                            temp_a13_4,
                            _mm_add_ps(
                                temp_a13_5,
                                _mm_add_ps(
                                    temp_a13_6,
                                    _mm_add_ps(
                                        temp_a13_7,
                                        _mm_add_ps(
                                            temp_a13_8,
                                            _mm_add_ps(
                                                temp_a13_9,
                                                _mm_add_ps(
                                                    temp_a13_10,
                                                    _mm_add_ps(
                                                        temp_a13_11,
                                                        _mm_add_ps(
                                                            temp_a13_12,
                                                            _mm_add_ps(
                                                                temp_a13_13,
                                                                _mm_add_ps(
                                                                    temp_a13_14,
                                                                    temp_a13_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a14 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a14_1,
                _mm_add_ps(
                    temp_a14_2,
                    _mm_add_ps(
                        temp_a14_3,
                        _mm_add_ps(
                            temp_a14_4,
                            _mm_add_ps(
                                temp_a14_5,
                                _mm_add_ps(
                                    temp_a14_6,
                                    _mm_add_ps(
                                        temp_a14_7,
                                        _mm_add_ps(
                                            temp_a14_8,
                                            _mm_add_ps(
                                                temp_a14_9,
                                                _mm_add_ps(
                                                    temp_a14_10,
                                                    _mm_add_ps(
                                                        temp_a14_11,
                                                        _mm_add_ps(
                                                            temp_a14_12,
                                                            _mm_add_ps(
                                                                temp_a14_13,
                                                                _mm_add_ps(
                                                                    temp_a14_14,
                                                                    temp_a14_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a15 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                temp_a15_1,
                _mm_add_ps(
                    temp_a15_2,
                    _mm_add_ps(
                        temp_a15_3,
                        _mm_add_ps(
                            temp_a15_4,
                            _mm_add_ps(
                                temp_a15_5,
                                _mm_add_ps(
                                    temp_a15_6,
                                    _mm_add_ps(
                                        temp_a15_7,
                                        _mm_add_ps(
                                            temp_a15_8,
                                            _mm_add_ps(
                                                temp_a15_9,
                                                _mm_add_ps(
                                                    temp_a15_10,
                                                    _mm_add_ps(
                                                        temp_a15_11,
                                                        _mm_add_ps(
                                                            temp_a15_12,
                                                            _mm_add_ps(
                                                                temp_a15_13,
                                                                _mm_add_ps(
                                                                    temp_a15_14,
                                                                    temp_a15_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_ps(
            temp_b1_1,
            _mm_add_ps(
                temp_b1_2,
                _mm_add_ps(
                    temp_b1_3,
                    _mm_add_ps(
                        temp_b1_4,
                        _mm_add_ps(
                            temp_b1_5,
                            _mm_add_ps(
                                temp_b1_6,
                                _mm_add_ps(
                                    temp_b1_7,
                                    _mm_add_ps(
                                        temp_b1_8,
                                        _mm_add_ps(
                                            temp_b1_9,
                                            _mm_add_ps(
                                                temp_b1_10,
                                                _mm_add_ps(
                                                    temp_b1_11,
                                                    _mm_add_ps(
                                                        temp_b1_12,
                                                        _mm_add_ps(
                                                            temp_b1_13,
                                                            _mm_add_ps(temp_b1_14, temp_b1_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_ps(
            temp_b2_1,
            _mm_add_ps(
                temp_b2_2,
                _mm_add_ps(
                    temp_b2_3,
                    _mm_add_ps(
                        temp_b2_4,
                        _mm_add_ps(
                            temp_b2_5,
                            _mm_add_ps(
                                temp_b2_6,
                                _mm_sub_ps(
                                    temp_b2_7,
                                    _mm_add_ps(
                                        temp_b2_8,
                                        _mm_add_ps(
                                            temp_b2_9,
                                            _mm_add_ps(
                                                temp_b2_10,
                                                _mm_add_ps(
                                                    temp_b2_11,
                                                    _mm_add_ps(
                                                        temp_b2_12,
                                                        _mm_add_ps(
                                                            temp_b2_13,
                                                            _mm_add_ps(temp_b2_14, temp_b2_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_ps(
            temp_b3_1,
            _mm_add_ps(
                temp_b3_2,
                _mm_add_ps(
                    temp_b3_3,
                    _mm_add_ps(
                        temp_b3_4,
                        _mm_sub_ps(
                            temp_b3_5,
                            _mm_add_ps(
                                temp_b3_6,
                                _mm_add_ps(
                                    temp_b3_7,
                                    _mm_add_ps(
                                        temp_b3_8,
                                        _mm_add_ps(
                                            temp_b3_9,
                                            _mm_sub_ps(
                                                temp_b3_10,
                                                _mm_add_ps(
                                                    temp_b3_11,
                                                    _mm_add_ps(
                                                        temp_b3_12,
                                                        _mm_add_ps(
                                                            temp_b3_13,
                                                            _mm_add_ps(temp_b3_14, temp_b3_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_ps(
            temp_b4_1,
            _mm_add_ps(
                temp_b4_2,
                _mm_sub_ps(
                    temp_b4_3,
                    _mm_add_ps(
                        temp_b4_4,
                        _mm_add_ps(
                            temp_b4_5,
                            _mm_add_ps(
                                temp_b4_6,
                                _mm_sub_ps(
                                    temp_b4_7,
                                    _mm_add_ps(
                                        temp_b4_8,
                                        _mm_add_ps(
                                            temp_b4_9,
                                            _mm_add_ps(
                                                temp_b4_10,
                                                _mm_sub_ps(
                                                    temp_b4_11,
                                                    _mm_add_ps(
                                                        temp_b4_12,
                                                        _mm_add_ps(
                                                            temp_b4_13,
                                                            _mm_add_ps(temp_b4_14, temp_b4_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_add_ps(
            temp_b5_1,
            _mm_add_ps(
                temp_b5_2,
                _mm_sub_ps(
                    temp_b5_3,
                    _mm_add_ps(
                        temp_b5_4,
                        _mm_add_ps(
                            temp_b5_5,
                            _mm_sub_ps(
                                temp_b5_6,
                                _mm_add_ps(
                                    temp_b5_7,
                                    _mm_add_ps(
                                        temp_b5_8,
                                        _mm_sub_ps(
                                            temp_b5_9,
                                            _mm_add_ps(
                                                temp_b5_10,
                                                _mm_add_ps(
                                                    temp_b5_11,
                                                    _mm_sub_ps(
                                                        temp_b5_12,
                                                        _mm_add_ps(
                                                            temp_b5_13,
                                                            _mm_add_ps(temp_b5_14, temp_b5_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_add_ps(
            temp_b6_1,
            _mm_sub_ps(
                temp_b6_2,
                _mm_add_ps(
                    temp_b6_3,
                    _mm_add_ps(
                        temp_b6_4,
                        _mm_sub_ps(
                            temp_b6_5,
                            _mm_add_ps(
                                temp_b6_6,
                                _mm_sub_ps(
                                    temp_b6_7,
                                    _mm_add_ps(
                                        temp_b6_8,
                                        _mm_add_ps(
                                            temp_b6_9,
                                            _mm_sub_ps(
                                                temp_b6_10,
                                                _mm_add_ps(
                                                    temp_b6_11,
                                                    _mm_sub_ps(
                                                        temp_b6_12,
                                                        _mm_add_ps(
                                                            temp_b6_13,
                                                            _mm_add_ps(temp_b6_14, temp_b6_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_add_ps(
            temp_b7_1,
            _mm_sub_ps(
                temp_b7_2,
                _mm_add_ps(
                    temp_b7_3,
                    _mm_sub_ps(
                        temp_b7_4,
                        _mm_add_ps(
                            temp_b7_5,
                            _mm_sub_ps(
                                temp_b7_6,
                                _mm_add_ps(
                                    temp_b7_7,
                                    _mm_sub_ps(
                                        temp_b7_8,
                                        _mm_add_ps(
                                            temp_b7_9,
                                            _mm_add_ps(
                                                temp_b7_10,
                                                _mm_sub_ps(
                                                    temp_b7_11,
                                                    _mm_add_ps(
                                                        temp_b7_12,
                                                        _mm_sub_ps(
                                                            temp_b7_13,
                                                            _mm_add_ps(temp_b7_14, temp_b7_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_ps(
            temp_b8_1,
            _mm_add_ps(
                temp_b8_2,
                _mm_sub_ps(
                    temp_b8_3,
                    _mm_add_ps(
                        temp_b8_4,
                        _mm_sub_ps(
                            temp_b8_5,
                            _mm_add_ps(
                                temp_b8_6,
                                _mm_sub_ps(
                                    temp_b8_7,
                                    _mm_add_ps(
                                        temp_b8_8,
                                        _mm_sub_ps(
                                            temp_b8_9,
                                            _mm_add_ps(
                                                temp_b8_10,
                                                _mm_sub_ps(
                                                    temp_b8_11,
                                                    _mm_add_ps(
                                                        temp_b8_12,
                                                        _mm_sub_ps(
                                                            temp_b8_13,
                                                            _mm_add_ps(temp_b8_14, temp_b8_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b9 = _mm_sub_ps(
            temp_b9_1,
            _mm_add_ps(
                temp_b9_2,
                _mm_sub_ps(
                    temp_b9_3,
                    _mm_add_ps(
                        temp_b9_4,
                        _mm_sub_ps(
                            temp_b9_5,
                            _mm_sub_ps(
                                temp_b9_6,
                                _mm_add_ps(
                                    temp_b9_7,
                                    _mm_sub_ps(
                                        temp_b9_8,
                                        _mm_add_ps(
                                            temp_b9_9,
                                            _mm_sub_ps(
                                                temp_b9_10,
                                                _mm_add_ps(
                                                    temp_b9_11,
                                                    _mm_sub_ps(
                                                        temp_b9_12,
                                                        _mm_sub_ps(
                                                            temp_b9_13,
                                                            _mm_add_ps(temp_b9_14, temp_b9_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b10 = _mm_sub_ps(
            temp_b10_1,
            _mm_add_ps(
                temp_b10_2,
                _mm_sub_ps(
                    temp_b10_3,
                    _mm_sub_ps(
                        temp_b10_4,
                        _mm_add_ps(
                            temp_b10_5,
                            _mm_sub_ps(
                                temp_b10_6,
                                _mm_sub_ps(
                                    temp_b10_7,
                                    _mm_add_ps(
                                        temp_b10_8,
                                        _mm_sub_ps(
                                            temp_b10_9,
                                            _mm_sub_ps(
                                                temp_b10_10,
                                                _mm_add_ps(
                                                    temp_b10_11,
                                                    _mm_sub_ps(
                                                        temp_b10_12,
                                                        _mm_sub_ps(
                                                            temp_b10_13,
                                                            _mm_add_ps(temp_b10_14, temp_b10_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b11 = _mm_sub_ps(
            temp_b11_1,
            _mm_sub_ps(
                temp_b11_2,
                _mm_add_ps(
                    temp_b11_3,
                    _mm_sub_ps(
                        temp_b11_4,
                        _mm_sub_ps(
                            temp_b11_5,
                            _mm_add_ps(
                                temp_b11_6,
                                _mm_sub_ps(
                                    temp_b11_7,
                                    _mm_sub_ps(
                                        temp_b11_8,
                                        _mm_sub_ps(
                                            temp_b11_9,
                                            _mm_add_ps(
                                                temp_b11_10,
                                                _mm_sub_ps(
                                                    temp_b11_11,
                                                    _mm_sub_ps(
                                                        temp_b11_12,
                                                        _mm_add_ps(
                                                            temp_b11_13,
                                                            _mm_sub_ps(temp_b11_14, temp_b11_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b12 = _mm_sub_ps(
            temp_b12_1,
            _mm_sub_ps(
                temp_b12_2,
                _mm_sub_ps(
                    temp_b12_3,
                    _mm_add_ps(
                        temp_b12_4,
                        _mm_sub_ps(
                            temp_b12_5,
                            _mm_sub_ps(
                                temp_b12_6,
                                _mm_sub_ps(
                                    temp_b12_7,
                                    _mm_add_ps(
                                        temp_b12_8,
                                        _mm_sub_ps(
                                            temp_b12_9,
                                            _mm_sub_ps(
                                                temp_b12_10,
                                                _mm_sub_ps(
                                                    temp_b12_11,
                                                    _mm_sub_ps(
                                                        temp_b12_12,
                                                        _mm_add_ps(
                                                            temp_b12_13,
                                                            _mm_sub_ps(temp_b12_14, temp_b12_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b13 = _mm_sub_ps(
            temp_b13_1,
            _mm_sub_ps(
                temp_b13_2,
                _mm_sub_ps(
                    temp_b13_3,
                    _mm_sub_ps(
                        temp_b13_4,
                        _mm_sub_ps(
                            temp_b13_5,
                            _mm_add_ps(
                                temp_b13_6,
                                _mm_sub_ps(
                                    temp_b13_7,
                                    _mm_sub_ps(
                                        temp_b13_8,
                                        _mm_sub_ps(
                                            temp_b13_9,
                                            _mm_sub_ps(
                                                temp_b13_10,
                                                _mm_sub_ps(
                                                    temp_b13_11,
                                                    _mm_add_ps(
                                                        temp_b13_12,
                                                        _mm_sub_ps(
                                                            temp_b13_13,
                                                            _mm_sub_ps(temp_b13_14, temp_b13_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b14 = _mm_sub_ps(
            temp_b14_1,
            _mm_sub_ps(
                temp_b14_2,
                _mm_sub_ps(
                    temp_b14_3,
                    _mm_sub_ps(
                        temp_b14_4,
                        _mm_sub_ps(
                            temp_b14_5,
                            _mm_sub_ps(
                                temp_b14_6,
                                _mm_sub_ps(
                                    temp_b14_7,
                                    _mm_sub_ps(
                                        temp_b14_8,
                                        _mm_sub_ps(
                                            temp_b14_9,
                                            _mm_add_ps(
                                                temp_b14_10,
                                                _mm_sub_ps(
                                                    temp_b14_11,
                                                    _mm_sub_ps(
                                                        temp_b14_12,
                                                        _mm_sub_ps(
                                                            temp_b14_13,
                                                            _mm_sub_ps(temp_b14_14, temp_b14_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b15 = _mm_sub_ps(
            temp_b15_1,
            _mm_sub_ps(
                temp_b15_2,
                _mm_sub_ps(
                    temp_b15_3,
                    _mm_sub_ps(
                        temp_b15_4,
                        _mm_sub_ps(
                            temp_b15_5,
                            _mm_sub_ps(
                                temp_b15_6,
                                _mm_sub_ps(
                                    temp_b15_7,
                                    _mm_sub_ps(
                                        temp_b15_8,
                                        _mm_sub_ps(
                                            temp_b15_9,
                                            _mm_sub_ps(
                                                temp_b15_10,
                                                _mm_sub_ps(
                                                    temp_b15_11,
                                                    _mm_sub_ps(
                                                        temp_b15_12,
                                                        _mm_sub_ps(
                                                            temp_b15_13,
                                                            _mm_sub_ps(temp_b15_14, temp_b15_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate_both(temp_b1);
        let temp_b2_rot = self.rotate.rotate_both(temp_b2);
        let temp_b3_rot = self.rotate.rotate_both(temp_b3);
        let temp_b4_rot = self.rotate.rotate_both(temp_b4);
        let temp_b5_rot = self.rotate.rotate_both(temp_b5);
        let temp_b6_rot = self.rotate.rotate_both(temp_b6);
        let temp_b7_rot = self.rotate.rotate_both(temp_b7);
        let temp_b8_rot = self.rotate.rotate_both(temp_b8);
        let temp_b9_rot = self.rotate.rotate_both(temp_b9);
        let temp_b10_rot = self.rotate.rotate_both(temp_b10);
        let temp_b11_rot = self.rotate.rotate_both(temp_b11);
        let temp_b12_rot = self.rotate.rotate_both(temp_b12);
        let temp_b13_rot = self.rotate.rotate_both(temp_b13);
        let temp_b14_rot = self.rotate.rotate_both(temp_b14);
        let temp_b15_rot = self.rotate.rotate_both(temp_b15);

        let x0 = _mm_add_ps(
            values[0],
            _mm_add_ps(
                x130p,
                _mm_add_ps(
                    x229p,
                    _mm_add_ps(
                        x328p,
                        _mm_add_ps(
                            x427p,
                            _mm_add_ps(
                                x526p,
                                _mm_add_ps(
                                    x625p,
                                    _mm_add_ps(
                                        x724p,
                                        _mm_add_ps(
                                            x823p,
                                            _mm_add_ps(
                                                x922p,
                                                _mm_add_ps(
                                                    x1021p,
                                                    _mm_add_ps(
                                                        x1120p,
                                                        _mm_add_ps(
                                                            x1219p,
                                                            _mm_add_ps(
                                                                x1318p,
                                                                _mm_add_ps(x1417p, x1516p),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_ps(temp_a1, temp_b1_rot);
        let x2 = _mm_add_ps(temp_a2, temp_b2_rot);
        let x3 = _mm_add_ps(temp_a3, temp_b3_rot);
        let x4 = _mm_add_ps(temp_a4, temp_b4_rot);
        let x5 = _mm_add_ps(temp_a5, temp_b5_rot);
        let x6 = _mm_add_ps(temp_a6, temp_b6_rot);
        let x7 = _mm_add_ps(temp_a7, temp_b7_rot);
        let x8 = _mm_add_ps(temp_a8, temp_b8_rot);
        let x9 = _mm_add_ps(temp_a9, temp_b9_rot);
        let x10 = _mm_add_ps(temp_a10, temp_b10_rot);
        let x11 = _mm_add_ps(temp_a11, temp_b11_rot);
        let x12 = _mm_add_ps(temp_a12, temp_b12_rot);
        let x13 = _mm_add_ps(temp_a13, temp_b13_rot);
        let x14 = _mm_add_ps(temp_a14, temp_b14_rot);
        let x15 = _mm_add_ps(temp_a15, temp_b15_rot);
        let x16 = _mm_sub_ps(temp_a15, temp_b15_rot);
        let x17 = _mm_sub_ps(temp_a14, temp_b14_rot);
        let x18 = _mm_sub_ps(temp_a13, temp_b13_rot);
        let x19 = _mm_sub_ps(temp_a12, temp_b12_rot);
        let x20 = _mm_sub_ps(temp_a11, temp_b11_rot);
        let x21 = _mm_sub_ps(temp_a10, temp_b10_rot);
        let x22 = _mm_sub_ps(temp_a9, temp_b9_rot);
        let x23 = _mm_sub_ps(temp_a8, temp_b8_rot);
        let x24 = _mm_sub_ps(temp_a7, temp_b7_rot);
        let x25 = _mm_sub_ps(temp_a6, temp_b6_rot);
        let x26 = _mm_sub_ps(temp_a5, temp_b5_rot);
        let x27 = _mm_sub_ps(temp_a4, temp_b4_rot);
        let x28 = _mm_sub_ps(temp_a3, temp_b3_rot);
        let x29 = _mm_sub_ps(temp_a2, temp_b2_rot);
        let x30 = _mm_sub_ps(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
            x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30,
        ]
    }
}

//   _____ _            __   _  _   _     _ _
//  |___ // |          / /_ | || | | |__ (_) |_
//    |_ \| |  _____  | '_ \| || |_| '_ \| | __|
//   ___) | | |_____| | (_) |__   _| |_) | | |_
//  |____/|_|          \___/   |_| |_.__/|_|\__|
//

pub struct SseF64Butterfly31<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: __m128d,
    twiddle1im: __m128d,
    twiddle2re: __m128d,
    twiddle2im: __m128d,
    twiddle3re: __m128d,
    twiddle3im: __m128d,
    twiddle4re: __m128d,
    twiddle4im: __m128d,
    twiddle5re: __m128d,
    twiddle5im: __m128d,
    twiddle6re: __m128d,
    twiddle6im: __m128d,
    twiddle7re: __m128d,
    twiddle7im: __m128d,
    twiddle8re: __m128d,
    twiddle8im: __m128d,
    twiddle9re: __m128d,
    twiddle9im: __m128d,
    twiddle10re: __m128d,
    twiddle10im: __m128d,
    twiddle11re: __m128d,
    twiddle11im: __m128d,
    twiddle12re: __m128d,
    twiddle12im: __m128d,
    twiddle13re: __m128d,
    twiddle13im: __m128d,
    twiddle14re: __m128d,
    twiddle14im: __m128d,
    twiddle15re: __m128d,
    twiddle15im: __m128d,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly31, 31, |this: &SseF64Butterfly31<_>| this
    .direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly31, 31, |this: &SseF64Butterfly31<_>| this
    .direction);
impl<T: FftNum> SseF64Butterfly31<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 31, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 31, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 31, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 31, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 31, direction);
        let tw6: Complex<f64> = twiddles::compute_twiddle(6, 31, direction);
        let tw7: Complex<f64> = twiddles::compute_twiddle(7, 31, direction);
        let tw8: Complex<f64> = twiddles::compute_twiddle(8, 31, direction);
        let tw9: Complex<f64> = twiddles::compute_twiddle(9, 31, direction);
        let tw10: Complex<f64> = twiddles::compute_twiddle(10, 31, direction);
        let tw11: Complex<f64> = twiddles::compute_twiddle(11, 31, direction);
        let tw12: Complex<f64> = twiddles::compute_twiddle(12, 31, direction);
        let tw13: Complex<f64> = twiddles::compute_twiddle(13, 31, direction);
        let tw14: Complex<f64> = twiddles::compute_twiddle(14, 31, direction);
        let tw15: Complex<f64> = twiddles::compute_twiddle(15, 31, direction);
        let twiddle1re = unsafe { _mm_set_pd(tw1.re, tw1.re) };
        let twiddle1im = unsafe { _mm_set_pd(tw1.im, tw1.im) };
        let twiddle2re = unsafe { _mm_set_pd(tw2.re, tw2.re) };
        let twiddle2im = unsafe { _mm_set_pd(tw2.im, tw2.im) };
        let twiddle3re = unsafe { _mm_set_pd(tw3.re, tw3.re) };
        let twiddle3im = unsafe { _mm_set_pd(tw3.im, tw3.im) };
        let twiddle4re = unsafe { _mm_set_pd(tw4.re, tw4.re) };
        let twiddle4im = unsafe { _mm_set_pd(tw4.im, tw4.im) };
        let twiddle5re = unsafe { _mm_set_pd(tw5.re, tw5.re) };
        let twiddle5im = unsafe { _mm_set_pd(tw5.im, tw5.im) };
        let twiddle6re = unsafe { _mm_set_pd(tw6.re, tw6.re) };
        let twiddle6im = unsafe { _mm_set_pd(tw6.im, tw6.im) };
        let twiddle7re = unsafe { _mm_set_pd(tw7.re, tw7.re) };
        let twiddle7im = unsafe { _mm_set_pd(tw7.im, tw7.im) };
        let twiddle8re = unsafe { _mm_set_pd(tw8.re, tw8.re) };
        let twiddle8im = unsafe { _mm_set_pd(tw8.im, tw8.im) };
        let twiddle9re = unsafe { _mm_set_pd(tw9.re, tw9.re) };
        let twiddle9im = unsafe { _mm_set_pd(tw9.im, tw9.im) };
        let twiddle10re = unsafe { _mm_set_pd(tw10.re, tw10.re) };
        let twiddle10im = unsafe { _mm_set_pd(tw10.im, tw10.im) };
        let twiddle11re = unsafe { _mm_set_pd(tw11.re, tw11.re) };
        let twiddle11im = unsafe { _mm_set_pd(tw11.im, tw11.im) };
        let twiddle12re = unsafe { _mm_set_pd(tw12.re, tw12.re) };
        let twiddle12im = unsafe { _mm_set_pd(tw12.im, tw12.im) };
        let twiddle13re = unsafe { _mm_set_pd(tw13.re, tw13.re) };
        let twiddle13im = unsafe { _mm_set_pd(tw13.im, tw13.im) };
        let twiddle14re = unsafe { _mm_set_pd(tw14.re, tw14.re) };
        let twiddle14im = unsafe { _mm_set_pd(tw14.im, tw14.im) };
        let twiddle15re = unsafe { _mm_set_pd(tw15.re, tw15.re) };
        let twiddle15im = unsafe { _mm_set_pd(tw15.im, tw15.im) };

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
            twiddle2re,
            twiddle2im,
            twiddle3re,
            twiddle3im,
            twiddle4re,
            twiddle4im,
            twiddle5re,
            twiddle5im,
            twiddle6re,
            twiddle6im,
            twiddle7re,
            twiddle7im,
            twiddle8re,
            twiddle8im,
            twiddle9re,
            twiddle9im,
            twiddle10re,
            twiddle10im,
            twiddle11re,
            twiddle11im,
            twiddle12re,
            twiddle12im,
            twiddle13re,
            twiddle13im,
            twiddle14re,
            twiddle14im,
            twiddle15re,
            twiddle15im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let v0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let v1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let v2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let v3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let v4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let v5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let v6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let v7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let v8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let v9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let v10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);
        let v11 = _mm_loadu_pd(input.as_ptr().add(11) as *const f64);
        let v12 = _mm_loadu_pd(input.as_ptr().add(12) as *const f64);
        let v13 = _mm_loadu_pd(input.as_ptr().add(13) as *const f64);
        let v14 = _mm_loadu_pd(input.as_ptr().add(14) as *const f64);
        let v15 = _mm_loadu_pd(input.as_ptr().add(15) as *const f64);
        let v16 = _mm_loadu_pd(input.as_ptr().add(16) as *const f64);
        let v17 = _mm_loadu_pd(input.as_ptr().add(17) as *const f64);
        let v18 = _mm_loadu_pd(input.as_ptr().add(18) as *const f64);
        let v19 = _mm_loadu_pd(input.as_ptr().add(19) as *const f64);
        let v20 = _mm_loadu_pd(input.as_ptr().add(20) as *const f64);
        let v21 = _mm_loadu_pd(input.as_ptr().add(21) as *const f64);
        let v22 = _mm_loadu_pd(input.as_ptr().add(22) as *const f64);
        let v23 = _mm_loadu_pd(input.as_ptr().add(23) as *const f64);
        let v24 = _mm_loadu_pd(input.as_ptr().add(24) as *const f64);
        let v25 = _mm_loadu_pd(input.as_ptr().add(25) as *const f64);
        let v26 = _mm_loadu_pd(input.as_ptr().add(26) as *const f64);
        let v27 = _mm_loadu_pd(input.as_ptr().add(27) as *const f64);
        let v28 = _mm_loadu_pd(input.as_ptr().add(28) as *const f64);
        let v29 = _mm_loadu_pd(input.as_ptr().add(29) as *const f64);
        let v30 = _mm_loadu_pd(input.as_ptr().add(30) as *const f64);

        let out = self.perform_fft_direct([
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
        ]);

        let val = std::mem::transmute::<[__m128d; 31], [Complex<f64>; 31]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
        *output_slice.add(6) = val[6];
        *output_slice.add(7) = val[7];
        *output_slice.add(8) = val[8];
        *output_slice.add(9) = val[9];
        *output_slice.add(10) = val[10];
        *output_slice.add(11) = val[11];
        *output_slice.add(12) = val[12];
        *output_slice.add(13) = val[13];
        *output_slice.add(14) = val[14];
        *output_slice.add(15) = val[15];
        *output_slice.add(16) = val[16];
        *output_slice.add(17) = val[17];
        *output_slice.add(18) = val[18];
        *output_slice.add(19) = val[19];
        *output_slice.add(20) = val[20];
        *output_slice.add(21) = val[21];
        *output_slice.add(22) = val[22];
        *output_slice.add(23) = val[23];
        *output_slice.add(24) = val[24];
        *output_slice.add(25) = val[25];
        *output_slice.add(26) = val[26];
        *output_slice.add(27) = val[27];
        *output_slice.add(28) = val[28];
        *output_slice.add(29) = val[29];
        *output_slice.add(30) = val[30];
    }

    // length 7 fft of a, given as a0, a1, a2, a3, a4, a5, a6.
    // result is [A0, A1, A2, A3, A4, A5, A6]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; 31]) -> [__m128d; 31] {
        // This is a SSE translation of the scalar 17-point butterfly
        let x130p = _mm_add_pd(values[1], values[30]);
        let x130n = _mm_sub_pd(values[1], values[30]);
        let x229p = _mm_add_pd(values[2], values[29]);
        let x229n = _mm_sub_pd(values[2], values[29]);
        let x328p = _mm_add_pd(values[3], values[28]);
        let x328n = _mm_sub_pd(values[3], values[28]);
        let x427p = _mm_add_pd(values[4], values[27]);
        let x427n = _mm_sub_pd(values[4], values[27]);
        let x526p = _mm_add_pd(values[5], values[26]);
        let x526n = _mm_sub_pd(values[5], values[26]);
        let x625p = _mm_add_pd(values[6], values[25]);
        let x625n = _mm_sub_pd(values[6], values[25]);
        let x724p = _mm_add_pd(values[7], values[24]);
        let x724n = _mm_sub_pd(values[7], values[24]);
        let x823p = _mm_add_pd(values[8], values[23]);
        let x823n = _mm_sub_pd(values[8], values[23]);
        let x922p = _mm_add_pd(values[9], values[22]);
        let x922n = _mm_sub_pd(values[9], values[22]);
        let x1021p = _mm_add_pd(values[10], values[21]);
        let x1021n = _mm_sub_pd(values[10], values[21]);
        let x1120p = _mm_add_pd(values[11], values[20]);
        let x1120n = _mm_sub_pd(values[11], values[20]);
        let x1219p = _mm_add_pd(values[12], values[19]);
        let x1219n = _mm_sub_pd(values[12], values[19]);
        let x1318p = _mm_add_pd(values[13], values[18]);
        let x1318n = _mm_sub_pd(values[13], values[18]);
        let x1417p = _mm_add_pd(values[14], values[17]);
        let x1417n = _mm_sub_pd(values[14], values[17]);
        let x1516p = _mm_add_pd(values[15], values[16]);
        let x1516n = _mm_sub_pd(values[15], values[16]);

        let temp_a1_1 = _mm_mul_pd(self.twiddle1re, x130p);
        let temp_a1_2 = _mm_mul_pd(self.twiddle2re, x229p);
        let temp_a1_3 = _mm_mul_pd(self.twiddle3re, x328p);
        let temp_a1_4 = _mm_mul_pd(self.twiddle4re, x427p);
        let temp_a1_5 = _mm_mul_pd(self.twiddle5re, x526p);
        let temp_a1_6 = _mm_mul_pd(self.twiddle6re, x625p);
        let temp_a1_7 = _mm_mul_pd(self.twiddle7re, x724p);
        let temp_a1_8 = _mm_mul_pd(self.twiddle8re, x823p);
        let temp_a1_9 = _mm_mul_pd(self.twiddle9re, x922p);
        let temp_a1_10 = _mm_mul_pd(self.twiddle10re, x1021p);
        let temp_a1_11 = _mm_mul_pd(self.twiddle11re, x1120p);
        let temp_a1_12 = _mm_mul_pd(self.twiddle12re, x1219p);
        let temp_a1_13 = _mm_mul_pd(self.twiddle13re, x1318p);
        let temp_a1_14 = _mm_mul_pd(self.twiddle14re, x1417p);
        let temp_a1_15 = _mm_mul_pd(self.twiddle15re, x1516p);
        let temp_a2_1 = _mm_mul_pd(self.twiddle2re, x130p);
        let temp_a2_2 = _mm_mul_pd(self.twiddle4re, x229p);
        let temp_a2_3 = _mm_mul_pd(self.twiddle6re, x328p);
        let temp_a2_4 = _mm_mul_pd(self.twiddle8re, x427p);
        let temp_a2_5 = _mm_mul_pd(self.twiddle10re, x526p);
        let temp_a2_6 = _mm_mul_pd(self.twiddle12re, x625p);
        let temp_a2_7 = _mm_mul_pd(self.twiddle14re, x724p);
        let temp_a2_8 = _mm_mul_pd(self.twiddle15re, x823p);
        let temp_a2_9 = _mm_mul_pd(self.twiddle13re, x922p);
        let temp_a2_10 = _mm_mul_pd(self.twiddle11re, x1021p);
        let temp_a2_11 = _mm_mul_pd(self.twiddle9re, x1120p);
        let temp_a2_12 = _mm_mul_pd(self.twiddle7re, x1219p);
        let temp_a2_13 = _mm_mul_pd(self.twiddle5re, x1318p);
        let temp_a2_14 = _mm_mul_pd(self.twiddle3re, x1417p);
        let temp_a2_15 = _mm_mul_pd(self.twiddle1re, x1516p);
        let temp_a3_1 = _mm_mul_pd(self.twiddle3re, x130p);
        let temp_a3_2 = _mm_mul_pd(self.twiddle6re, x229p);
        let temp_a3_3 = _mm_mul_pd(self.twiddle9re, x328p);
        let temp_a3_4 = _mm_mul_pd(self.twiddle12re, x427p);
        let temp_a3_5 = _mm_mul_pd(self.twiddle15re, x526p);
        let temp_a3_6 = _mm_mul_pd(self.twiddle13re, x625p);
        let temp_a3_7 = _mm_mul_pd(self.twiddle10re, x724p);
        let temp_a3_8 = _mm_mul_pd(self.twiddle7re, x823p);
        let temp_a3_9 = _mm_mul_pd(self.twiddle4re, x922p);
        let temp_a3_10 = _mm_mul_pd(self.twiddle1re, x1021p);
        let temp_a3_11 = _mm_mul_pd(self.twiddle2re, x1120p);
        let temp_a3_12 = _mm_mul_pd(self.twiddle5re, x1219p);
        let temp_a3_13 = _mm_mul_pd(self.twiddle8re, x1318p);
        let temp_a3_14 = _mm_mul_pd(self.twiddle11re, x1417p);
        let temp_a3_15 = _mm_mul_pd(self.twiddle14re, x1516p);
        let temp_a4_1 = _mm_mul_pd(self.twiddle4re, x130p);
        let temp_a4_2 = _mm_mul_pd(self.twiddle8re, x229p);
        let temp_a4_3 = _mm_mul_pd(self.twiddle12re, x328p);
        let temp_a4_4 = _mm_mul_pd(self.twiddle15re, x427p);
        let temp_a4_5 = _mm_mul_pd(self.twiddle11re, x526p);
        let temp_a4_6 = _mm_mul_pd(self.twiddle7re, x625p);
        let temp_a4_7 = _mm_mul_pd(self.twiddle3re, x724p);
        let temp_a4_8 = _mm_mul_pd(self.twiddle1re, x823p);
        let temp_a4_9 = _mm_mul_pd(self.twiddle5re, x922p);
        let temp_a4_10 = _mm_mul_pd(self.twiddle9re, x1021p);
        let temp_a4_11 = _mm_mul_pd(self.twiddle13re, x1120p);
        let temp_a4_12 = _mm_mul_pd(self.twiddle14re, x1219p);
        let temp_a4_13 = _mm_mul_pd(self.twiddle10re, x1318p);
        let temp_a4_14 = _mm_mul_pd(self.twiddle6re, x1417p);
        let temp_a4_15 = _mm_mul_pd(self.twiddle2re, x1516p);
        let temp_a5_1 = _mm_mul_pd(self.twiddle5re, x130p);
        let temp_a5_2 = _mm_mul_pd(self.twiddle10re, x229p);
        let temp_a5_3 = _mm_mul_pd(self.twiddle15re, x328p);
        let temp_a5_4 = _mm_mul_pd(self.twiddle11re, x427p);
        let temp_a5_5 = _mm_mul_pd(self.twiddle6re, x526p);
        let temp_a5_6 = _mm_mul_pd(self.twiddle1re, x625p);
        let temp_a5_7 = _mm_mul_pd(self.twiddle4re, x724p);
        let temp_a5_8 = _mm_mul_pd(self.twiddle9re, x823p);
        let temp_a5_9 = _mm_mul_pd(self.twiddle14re, x922p);
        let temp_a5_10 = _mm_mul_pd(self.twiddle12re, x1021p);
        let temp_a5_11 = _mm_mul_pd(self.twiddle7re, x1120p);
        let temp_a5_12 = _mm_mul_pd(self.twiddle2re, x1219p);
        let temp_a5_13 = _mm_mul_pd(self.twiddle3re, x1318p);
        let temp_a5_14 = _mm_mul_pd(self.twiddle8re, x1417p);
        let temp_a5_15 = _mm_mul_pd(self.twiddle13re, x1516p);
        let temp_a6_1 = _mm_mul_pd(self.twiddle6re, x130p);
        let temp_a6_2 = _mm_mul_pd(self.twiddle12re, x229p);
        let temp_a6_3 = _mm_mul_pd(self.twiddle13re, x328p);
        let temp_a6_4 = _mm_mul_pd(self.twiddle7re, x427p);
        let temp_a6_5 = _mm_mul_pd(self.twiddle1re, x526p);
        let temp_a6_6 = _mm_mul_pd(self.twiddle5re, x625p);
        let temp_a6_7 = _mm_mul_pd(self.twiddle11re, x724p);
        let temp_a6_8 = _mm_mul_pd(self.twiddle14re, x823p);
        let temp_a6_9 = _mm_mul_pd(self.twiddle8re, x922p);
        let temp_a6_10 = _mm_mul_pd(self.twiddle2re, x1021p);
        let temp_a6_11 = _mm_mul_pd(self.twiddle4re, x1120p);
        let temp_a6_12 = _mm_mul_pd(self.twiddle10re, x1219p);
        let temp_a6_13 = _mm_mul_pd(self.twiddle15re, x1318p);
        let temp_a6_14 = _mm_mul_pd(self.twiddle9re, x1417p);
        let temp_a6_15 = _mm_mul_pd(self.twiddle3re, x1516p);
        let temp_a7_1 = _mm_mul_pd(self.twiddle7re, x130p);
        let temp_a7_2 = _mm_mul_pd(self.twiddle14re, x229p);
        let temp_a7_3 = _mm_mul_pd(self.twiddle10re, x328p);
        let temp_a7_4 = _mm_mul_pd(self.twiddle3re, x427p);
        let temp_a7_5 = _mm_mul_pd(self.twiddle4re, x526p);
        let temp_a7_6 = _mm_mul_pd(self.twiddle11re, x625p);
        let temp_a7_7 = _mm_mul_pd(self.twiddle13re, x724p);
        let temp_a7_8 = _mm_mul_pd(self.twiddle6re, x823p);
        let temp_a7_9 = _mm_mul_pd(self.twiddle1re, x922p);
        let temp_a7_10 = _mm_mul_pd(self.twiddle8re, x1021p);
        let temp_a7_11 = _mm_mul_pd(self.twiddle15re, x1120p);
        let temp_a7_12 = _mm_mul_pd(self.twiddle9re, x1219p);
        let temp_a7_13 = _mm_mul_pd(self.twiddle2re, x1318p);
        let temp_a7_14 = _mm_mul_pd(self.twiddle5re, x1417p);
        let temp_a7_15 = _mm_mul_pd(self.twiddle12re, x1516p);
        let temp_a8_1 = _mm_mul_pd(self.twiddle8re, x130p);
        let temp_a8_2 = _mm_mul_pd(self.twiddle15re, x229p);
        let temp_a8_3 = _mm_mul_pd(self.twiddle7re, x328p);
        let temp_a8_4 = _mm_mul_pd(self.twiddle1re, x427p);
        let temp_a8_5 = _mm_mul_pd(self.twiddle9re, x526p);
        let temp_a8_6 = _mm_mul_pd(self.twiddle14re, x625p);
        let temp_a8_7 = _mm_mul_pd(self.twiddle6re, x724p);
        let temp_a8_8 = _mm_mul_pd(self.twiddle2re, x823p);
        let temp_a8_9 = _mm_mul_pd(self.twiddle10re, x922p);
        let temp_a8_10 = _mm_mul_pd(self.twiddle13re, x1021p);
        let temp_a8_11 = _mm_mul_pd(self.twiddle5re, x1120p);
        let temp_a8_12 = _mm_mul_pd(self.twiddle3re, x1219p);
        let temp_a8_13 = _mm_mul_pd(self.twiddle11re, x1318p);
        let temp_a8_14 = _mm_mul_pd(self.twiddle12re, x1417p);
        let temp_a8_15 = _mm_mul_pd(self.twiddle4re, x1516p);
        let temp_a9_1 = _mm_mul_pd(self.twiddle9re, x130p);
        let temp_a9_2 = _mm_mul_pd(self.twiddle13re, x229p);
        let temp_a9_3 = _mm_mul_pd(self.twiddle4re, x328p);
        let temp_a9_4 = _mm_mul_pd(self.twiddle5re, x427p);
        let temp_a9_5 = _mm_mul_pd(self.twiddle14re, x526p);
        let temp_a9_6 = _mm_mul_pd(self.twiddle8re, x625p);
        let temp_a9_7 = _mm_mul_pd(self.twiddle1re, x724p);
        let temp_a9_8 = _mm_mul_pd(self.twiddle10re, x823p);
        let temp_a9_9 = _mm_mul_pd(self.twiddle12re, x922p);
        let temp_a9_10 = _mm_mul_pd(self.twiddle3re, x1021p);
        let temp_a9_11 = _mm_mul_pd(self.twiddle6re, x1120p);
        let temp_a9_12 = _mm_mul_pd(self.twiddle15re, x1219p);
        let temp_a9_13 = _mm_mul_pd(self.twiddle7re, x1318p);
        let temp_a9_14 = _mm_mul_pd(self.twiddle2re, x1417p);
        let temp_a9_15 = _mm_mul_pd(self.twiddle11re, x1516p);
        let temp_a10_1 = _mm_mul_pd(self.twiddle10re, x130p);
        let temp_a10_2 = _mm_mul_pd(self.twiddle11re, x229p);
        let temp_a10_3 = _mm_mul_pd(self.twiddle1re, x328p);
        let temp_a10_4 = _mm_mul_pd(self.twiddle9re, x427p);
        let temp_a10_5 = _mm_mul_pd(self.twiddle12re, x526p);
        let temp_a10_6 = _mm_mul_pd(self.twiddle2re, x625p);
        let temp_a10_7 = _mm_mul_pd(self.twiddle8re, x724p);
        let temp_a10_8 = _mm_mul_pd(self.twiddle13re, x823p);
        let temp_a10_9 = _mm_mul_pd(self.twiddle3re, x922p);
        let temp_a10_10 = _mm_mul_pd(self.twiddle7re, x1021p);
        let temp_a10_11 = _mm_mul_pd(self.twiddle14re, x1120p);
        let temp_a10_12 = _mm_mul_pd(self.twiddle4re, x1219p);
        let temp_a10_13 = _mm_mul_pd(self.twiddle6re, x1318p);
        let temp_a10_14 = _mm_mul_pd(self.twiddle15re, x1417p);
        let temp_a10_15 = _mm_mul_pd(self.twiddle5re, x1516p);
        let temp_a11_1 = _mm_mul_pd(self.twiddle11re, x130p);
        let temp_a11_2 = _mm_mul_pd(self.twiddle9re, x229p);
        let temp_a11_3 = _mm_mul_pd(self.twiddle2re, x328p);
        let temp_a11_4 = _mm_mul_pd(self.twiddle13re, x427p);
        let temp_a11_5 = _mm_mul_pd(self.twiddle7re, x526p);
        let temp_a11_6 = _mm_mul_pd(self.twiddle4re, x625p);
        let temp_a11_7 = _mm_mul_pd(self.twiddle15re, x724p);
        let temp_a11_8 = _mm_mul_pd(self.twiddle5re, x823p);
        let temp_a11_9 = _mm_mul_pd(self.twiddle6re, x922p);
        let temp_a11_10 = _mm_mul_pd(self.twiddle14re, x1021p);
        let temp_a11_11 = _mm_mul_pd(self.twiddle3re, x1120p);
        let temp_a11_12 = _mm_mul_pd(self.twiddle8re, x1219p);
        let temp_a11_13 = _mm_mul_pd(self.twiddle12re, x1318p);
        let temp_a11_14 = _mm_mul_pd(self.twiddle1re, x1417p);
        let temp_a11_15 = _mm_mul_pd(self.twiddle10re, x1516p);
        let temp_a12_1 = _mm_mul_pd(self.twiddle12re, x130p);
        let temp_a12_2 = _mm_mul_pd(self.twiddle7re, x229p);
        let temp_a12_3 = _mm_mul_pd(self.twiddle5re, x328p);
        let temp_a12_4 = _mm_mul_pd(self.twiddle14re, x427p);
        let temp_a12_5 = _mm_mul_pd(self.twiddle2re, x526p);
        let temp_a12_6 = _mm_mul_pd(self.twiddle10re, x625p);
        let temp_a12_7 = _mm_mul_pd(self.twiddle9re, x724p);
        let temp_a12_8 = _mm_mul_pd(self.twiddle3re, x823p);
        let temp_a12_9 = _mm_mul_pd(self.twiddle15re, x922p);
        let temp_a12_10 = _mm_mul_pd(self.twiddle4re, x1021p);
        let temp_a12_11 = _mm_mul_pd(self.twiddle8re, x1120p);
        let temp_a12_12 = _mm_mul_pd(self.twiddle11re, x1219p);
        let temp_a12_13 = _mm_mul_pd(self.twiddle1re, x1318p);
        let temp_a12_14 = _mm_mul_pd(self.twiddle13re, x1417p);
        let temp_a12_15 = _mm_mul_pd(self.twiddle6re, x1516p);
        let temp_a13_1 = _mm_mul_pd(self.twiddle13re, x130p);
        let temp_a13_2 = _mm_mul_pd(self.twiddle5re, x229p);
        let temp_a13_3 = _mm_mul_pd(self.twiddle8re, x328p);
        let temp_a13_4 = _mm_mul_pd(self.twiddle10re, x427p);
        let temp_a13_5 = _mm_mul_pd(self.twiddle3re, x526p);
        let temp_a13_6 = _mm_mul_pd(self.twiddle15re, x625p);
        let temp_a13_7 = _mm_mul_pd(self.twiddle2re, x724p);
        let temp_a13_8 = _mm_mul_pd(self.twiddle11re, x823p);
        let temp_a13_9 = _mm_mul_pd(self.twiddle7re, x922p);
        let temp_a13_10 = _mm_mul_pd(self.twiddle6re, x1021p);
        let temp_a13_11 = _mm_mul_pd(self.twiddle12re, x1120p);
        let temp_a13_12 = _mm_mul_pd(self.twiddle1re, x1219p);
        let temp_a13_13 = _mm_mul_pd(self.twiddle14re, x1318p);
        let temp_a13_14 = _mm_mul_pd(self.twiddle4re, x1417p);
        let temp_a13_15 = _mm_mul_pd(self.twiddle9re, x1516p);
        let temp_a14_1 = _mm_mul_pd(self.twiddle14re, x130p);
        let temp_a14_2 = _mm_mul_pd(self.twiddle3re, x229p);
        let temp_a14_3 = _mm_mul_pd(self.twiddle11re, x328p);
        let temp_a14_4 = _mm_mul_pd(self.twiddle6re, x427p);
        let temp_a14_5 = _mm_mul_pd(self.twiddle8re, x526p);
        let temp_a14_6 = _mm_mul_pd(self.twiddle9re, x625p);
        let temp_a14_7 = _mm_mul_pd(self.twiddle5re, x724p);
        let temp_a14_8 = _mm_mul_pd(self.twiddle12re, x823p);
        let temp_a14_9 = _mm_mul_pd(self.twiddle2re, x922p);
        let temp_a14_10 = _mm_mul_pd(self.twiddle15re, x1021p);
        let temp_a14_11 = _mm_mul_pd(self.twiddle1re, x1120p);
        let temp_a14_12 = _mm_mul_pd(self.twiddle13re, x1219p);
        let temp_a14_13 = _mm_mul_pd(self.twiddle4re, x1318p);
        let temp_a14_14 = _mm_mul_pd(self.twiddle10re, x1417p);
        let temp_a14_15 = _mm_mul_pd(self.twiddle7re, x1516p);
        let temp_a15_1 = _mm_mul_pd(self.twiddle15re, x130p);
        let temp_a15_2 = _mm_mul_pd(self.twiddle1re, x229p);
        let temp_a15_3 = _mm_mul_pd(self.twiddle14re, x328p);
        let temp_a15_4 = _mm_mul_pd(self.twiddle2re, x427p);
        let temp_a15_5 = _mm_mul_pd(self.twiddle13re, x526p);
        let temp_a15_6 = _mm_mul_pd(self.twiddle3re, x625p);
        let temp_a15_7 = _mm_mul_pd(self.twiddle12re, x724p);
        let temp_a15_8 = _mm_mul_pd(self.twiddle4re, x823p);
        let temp_a15_9 = _mm_mul_pd(self.twiddle11re, x922p);
        let temp_a15_10 = _mm_mul_pd(self.twiddle5re, x1021p);
        let temp_a15_11 = _mm_mul_pd(self.twiddle10re, x1120p);
        let temp_a15_12 = _mm_mul_pd(self.twiddle6re, x1219p);
        let temp_a15_13 = _mm_mul_pd(self.twiddle9re, x1318p);
        let temp_a15_14 = _mm_mul_pd(self.twiddle7re, x1417p);
        let temp_a15_15 = _mm_mul_pd(self.twiddle8re, x1516p);

        let temp_b1_1 = _mm_mul_pd(self.twiddle1im, x130n);
        let temp_b1_2 = _mm_mul_pd(self.twiddle2im, x229n);
        let temp_b1_3 = _mm_mul_pd(self.twiddle3im, x328n);
        let temp_b1_4 = _mm_mul_pd(self.twiddle4im, x427n);
        let temp_b1_5 = _mm_mul_pd(self.twiddle5im, x526n);
        let temp_b1_6 = _mm_mul_pd(self.twiddle6im, x625n);
        let temp_b1_7 = _mm_mul_pd(self.twiddle7im, x724n);
        let temp_b1_8 = _mm_mul_pd(self.twiddle8im, x823n);
        let temp_b1_9 = _mm_mul_pd(self.twiddle9im, x922n);
        let temp_b1_10 = _mm_mul_pd(self.twiddle10im, x1021n);
        let temp_b1_11 = _mm_mul_pd(self.twiddle11im, x1120n);
        let temp_b1_12 = _mm_mul_pd(self.twiddle12im, x1219n);
        let temp_b1_13 = _mm_mul_pd(self.twiddle13im, x1318n);
        let temp_b1_14 = _mm_mul_pd(self.twiddle14im, x1417n);
        let temp_b1_15 = _mm_mul_pd(self.twiddle15im, x1516n);
        let temp_b2_1 = _mm_mul_pd(self.twiddle2im, x130n);
        let temp_b2_2 = _mm_mul_pd(self.twiddle4im, x229n);
        let temp_b2_3 = _mm_mul_pd(self.twiddle6im, x328n);
        let temp_b2_4 = _mm_mul_pd(self.twiddle8im, x427n);
        let temp_b2_5 = _mm_mul_pd(self.twiddle10im, x526n);
        let temp_b2_6 = _mm_mul_pd(self.twiddle12im, x625n);
        let temp_b2_7 = _mm_mul_pd(self.twiddle14im, x724n);
        let temp_b2_8 = _mm_mul_pd(self.twiddle15im, x823n);
        let temp_b2_9 = _mm_mul_pd(self.twiddle13im, x922n);
        let temp_b2_10 = _mm_mul_pd(self.twiddle11im, x1021n);
        let temp_b2_11 = _mm_mul_pd(self.twiddle9im, x1120n);
        let temp_b2_12 = _mm_mul_pd(self.twiddle7im, x1219n);
        let temp_b2_13 = _mm_mul_pd(self.twiddle5im, x1318n);
        let temp_b2_14 = _mm_mul_pd(self.twiddle3im, x1417n);
        let temp_b2_15 = _mm_mul_pd(self.twiddle1im, x1516n);
        let temp_b3_1 = _mm_mul_pd(self.twiddle3im, x130n);
        let temp_b3_2 = _mm_mul_pd(self.twiddle6im, x229n);
        let temp_b3_3 = _mm_mul_pd(self.twiddle9im, x328n);
        let temp_b3_4 = _mm_mul_pd(self.twiddle12im, x427n);
        let temp_b3_5 = _mm_mul_pd(self.twiddle15im, x526n);
        let temp_b3_6 = _mm_mul_pd(self.twiddle13im, x625n);
        let temp_b3_7 = _mm_mul_pd(self.twiddle10im, x724n);
        let temp_b3_8 = _mm_mul_pd(self.twiddle7im, x823n);
        let temp_b3_9 = _mm_mul_pd(self.twiddle4im, x922n);
        let temp_b3_10 = _mm_mul_pd(self.twiddle1im, x1021n);
        let temp_b3_11 = _mm_mul_pd(self.twiddle2im, x1120n);
        let temp_b3_12 = _mm_mul_pd(self.twiddle5im, x1219n);
        let temp_b3_13 = _mm_mul_pd(self.twiddle8im, x1318n);
        let temp_b3_14 = _mm_mul_pd(self.twiddle11im, x1417n);
        let temp_b3_15 = _mm_mul_pd(self.twiddle14im, x1516n);
        let temp_b4_1 = _mm_mul_pd(self.twiddle4im, x130n);
        let temp_b4_2 = _mm_mul_pd(self.twiddle8im, x229n);
        let temp_b4_3 = _mm_mul_pd(self.twiddle12im, x328n);
        let temp_b4_4 = _mm_mul_pd(self.twiddle15im, x427n);
        let temp_b4_5 = _mm_mul_pd(self.twiddle11im, x526n);
        let temp_b4_6 = _mm_mul_pd(self.twiddle7im, x625n);
        let temp_b4_7 = _mm_mul_pd(self.twiddle3im, x724n);
        let temp_b4_8 = _mm_mul_pd(self.twiddle1im, x823n);
        let temp_b4_9 = _mm_mul_pd(self.twiddle5im, x922n);
        let temp_b4_10 = _mm_mul_pd(self.twiddle9im, x1021n);
        let temp_b4_11 = _mm_mul_pd(self.twiddle13im, x1120n);
        let temp_b4_12 = _mm_mul_pd(self.twiddle14im, x1219n);
        let temp_b4_13 = _mm_mul_pd(self.twiddle10im, x1318n);
        let temp_b4_14 = _mm_mul_pd(self.twiddle6im, x1417n);
        let temp_b4_15 = _mm_mul_pd(self.twiddle2im, x1516n);
        let temp_b5_1 = _mm_mul_pd(self.twiddle5im, x130n);
        let temp_b5_2 = _mm_mul_pd(self.twiddle10im, x229n);
        let temp_b5_3 = _mm_mul_pd(self.twiddle15im, x328n);
        let temp_b5_4 = _mm_mul_pd(self.twiddle11im, x427n);
        let temp_b5_5 = _mm_mul_pd(self.twiddle6im, x526n);
        let temp_b5_6 = _mm_mul_pd(self.twiddle1im, x625n);
        let temp_b5_7 = _mm_mul_pd(self.twiddle4im, x724n);
        let temp_b5_8 = _mm_mul_pd(self.twiddle9im, x823n);
        let temp_b5_9 = _mm_mul_pd(self.twiddle14im, x922n);
        let temp_b5_10 = _mm_mul_pd(self.twiddle12im, x1021n);
        let temp_b5_11 = _mm_mul_pd(self.twiddle7im, x1120n);
        let temp_b5_12 = _mm_mul_pd(self.twiddle2im, x1219n);
        let temp_b5_13 = _mm_mul_pd(self.twiddle3im, x1318n);
        let temp_b5_14 = _mm_mul_pd(self.twiddle8im, x1417n);
        let temp_b5_15 = _mm_mul_pd(self.twiddle13im, x1516n);
        let temp_b6_1 = _mm_mul_pd(self.twiddle6im, x130n);
        let temp_b6_2 = _mm_mul_pd(self.twiddle12im, x229n);
        let temp_b6_3 = _mm_mul_pd(self.twiddle13im, x328n);
        let temp_b6_4 = _mm_mul_pd(self.twiddle7im, x427n);
        let temp_b6_5 = _mm_mul_pd(self.twiddle1im, x526n);
        let temp_b6_6 = _mm_mul_pd(self.twiddle5im, x625n);
        let temp_b6_7 = _mm_mul_pd(self.twiddle11im, x724n);
        let temp_b6_8 = _mm_mul_pd(self.twiddle14im, x823n);
        let temp_b6_9 = _mm_mul_pd(self.twiddle8im, x922n);
        let temp_b6_10 = _mm_mul_pd(self.twiddle2im, x1021n);
        let temp_b6_11 = _mm_mul_pd(self.twiddle4im, x1120n);
        let temp_b6_12 = _mm_mul_pd(self.twiddle10im, x1219n);
        let temp_b6_13 = _mm_mul_pd(self.twiddle15im, x1318n);
        let temp_b6_14 = _mm_mul_pd(self.twiddle9im, x1417n);
        let temp_b6_15 = _mm_mul_pd(self.twiddle3im, x1516n);
        let temp_b7_1 = _mm_mul_pd(self.twiddle7im, x130n);
        let temp_b7_2 = _mm_mul_pd(self.twiddle14im, x229n);
        let temp_b7_3 = _mm_mul_pd(self.twiddle10im, x328n);
        let temp_b7_4 = _mm_mul_pd(self.twiddle3im, x427n);
        let temp_b7_5 = _mm_mul_pd(self.twiddle4im, x526n);
        let temp_b7_6 = _mm_mul_pd(self.twiddle11im, x625n);
        let temp_b7_7 = _mm_mul_pd(self.twiddle13im, x724n);
        let temp_b7_8 = _mm_mul_pd(self.twiddle6im, x823n);
        let temp_b7_9 = _mm_mul_pd(self.twiddle1im, x922n);
        let temp_b7_10 = _mm_mul_pd(self.twiddle8im, x1021n);
        let temp_b7_11 = _mm_mul_pd(self.twiddle15im, x1120n);
        let temp_b7_12 = _mm_mul_pd(self.twiddle9im, x1219n);
        let temp_b7_13 = _mm_mul_pd(self.twiddle2im, x1318n);
        let temp_b7_14 = _mm_mul_pd(self.twiddle5im, x1417n);
        let temp_b7_15 = _mm_mul_pd(self.twiddle12im, x1516n);
        let temp_b8_1 = _mm_mul_pd(self.twiddle8im, x130n);
        let temp_b8_2 = _mm_mul_pd(self.twiddle15im, x229n);
        let temp_b8_3 = _mm_mul_pd(self.twiddle7im, x328n);
        let temp_b8_4 = _mm_mul_pd(self.twiddle1im, x427n);
        let temp_b8_5 = _mm_mul_pd(self.twiddle9im, x526n);
        let temp_b8_6 = _mm_mul_pd(self.twiddle14im, x625n);
        let temp_b8_7 = _mm_mul_pd(self.twiddle6im, x724n);
        let temp_b8_8 = _mm_mul_pd(self.twiddle2im, x823n);
        let temp_b8_9 = _mm_mul_pd(self.twiddle10im, x922n);
        let temp_b8_10 = _mm_mul_pd(self.twiddle13im, x1021n);
        let temp_b8_11 = _mm_mul_pd(self.twiddle5im, x1120n);
        let temp_b8_12 = _mm_mul_pd(self.twiddle3im, x1219n);
        let temp_b8_13 = _mm_mul_pd(self.twiddle11im, x1318n);
        let temp_b8_14 = _mm_mul_pd(self.twiddle12im, x1417n);
        let temp_b8_15 = _mm_mul_pd(self.twiddle4im, x1516n);
        let temp_b9_1 = _mm_mul_pd(self.twiddle9im, x130n);
        let temp_b9_2 = _mm_mul_pd(self.twiddle13im, x229n);
        let temp_b9_3 = _mm_mul_pd(self.twiddle4im, x328n);
        let temp_b9_4 = _mm_mul_pd(self.twiddle5im, x427n);
        let temp_b9_5 = _mm_mul_pd(self.twiddle14im, x526n);
        let temp_b9_6 = _mm_mul_pd(self.twiddle8im, x625n);
        let temp_b9_7 = _mm_mul_pd(self.twiddle1im, x724n);
        let temp_b9_8 = _mm_mul_pd(self.twiddle10im, x823n);
        let temp_b9_9 = _mm_mul_pd(self.twiddle12im, x922n);
        let temp_b9_10 = _mm_mul_pd(self.twiddle3im, x1021n);
        let temp_b9_11 = _mm_mul_pd(self.twiddle6im, x1120n);
        let temp_b9_12 = _mm_mul_pd(self.twiddle15im, x1219n);
        let temp_b9_13 = _mm_mul_pd(self.twiddle7im, x1318n);
        let temp_b9_14 = _mm_mul_pd(self.twiddle2im, x1417n);
        let temp_b9_15 = _mm_mul_pd(self.twiddle11im, x1516n);
        let temp_b10_1 = _mm_mul_pd(self.twiddle10im, x130n);
        let temp_b10_2 = _mm_mul_pd(self.twiddle11im, x229n);
        let temp_b10_3 = _mm_mul_pd(self.twiddle1im, x328n);
        let temp_b10_4 = _mm_mul_pd(self.twiddle9im, x427n);
        let temp_b10_5 = _mm_mul_pd(self.twiddle12im, x526n);
        let temp_b10_6 = _mm_mul_pd(self.twiddle2im, x625n);
        let temp_b10_7 = _mm_mul_pd(self.twiddle8im, x724n);
        let temp_b10_8 = _mm_mul_pd(self.twiddle13im, x823n);
        let temp_b10_9 = _mm_mul_pd(self.twiddle3im, x922n);
        let temp_b10_10 = _mm_mul_pd(self.twiddle7im, x1021n);
        let temp_b10_11 = _mm_mul_pd(self.twiddle14im, x1120n);
        let temp_b10_12 = _mm_mul_pd(self.twiddle4im, x1219n);
        let temp_b10_13 = _mm_mul_pd(self.twiddle6im, x1318n);
        let temp_b10_14 = _mm_mul_pd(self.twiddle15im, x1417n);
        let temp_b10_15 = _mm_mul_pd(self.twiddle5im, x1516n);
        let temp_b11_1 = _mm_mul_pd(self.twiddle11im, x130n);
        let temp_b11_2 = _mm_mul_pd(self.twiddle9im, x229n);
        let temp_b11_3 = _mm_mul_pd(self.twiddle2im, x328n);
        let temp_b11_4 = _mm_mul_pd(self.twiddle13im, x427n);
        let temp_b11_5 = _mm_mul_pd(self.twiddle7im, x526n);
        let temp_b11_6 = _mm_mul_pd(self.twiddle4im, x625n);
        let temp_b11_7 = _mm_mul_pd(self.twiddle15im, x724n);
        let temp_b11_8 = _mm_mul_pd(self.twiddle5im, x823n);
        let temp_b11_9 = _mm_mul_pd(self.twiddle6im, x922n);
        let temp_b11_10 = _mm_mul_pd(self.twiddle14im, x1021n);
        let temp_b11_11 = _mm_mul_pd(self.twiddle3im, x1120n);
        let temp_b11_12 = _mm_mul_pd(self.twiddle8im, x1219n);
        let temp_b11_13 = _mm_mul_pd(self.twiddle12im, x1318n);
        let temp_b11_14 = _mm_mul_pd(self.twiddle1im, x1417n);
        let temp_b11_15 = _mm_mul_pd(self.twiddle10im, x1516n);
        let temp_b12_1 = _mm_mul_pd(self.twiddle12im, x130n);
        let temp_b12_2 = _mm_mul_pd(self.twiddle7im, x229n);
        let temp_b12_3 = _mm_mul_pd(self.twiddle5im, x328n);
        let temp_b12_4 = _mm_mul_pd(self.twiddle14im, x427n);
        let temp_b12_5 = _mm_mul_pd(self.twiddle2im, x526n);
        let temp_b12_6 = _mm_mul_pd(self.twiddle10im, x625n);
        let temp_b12_7 = _mm_mul_pd(self.twiddle9im, x724n);
        let temp_b12_8 = _mm_mul_pd(self.twiddle3im, x823n);
        let temp_b12_9 = _mm_mul_pd(self.twiddle15im, x922n);
        let temp_b12_10 = _mm_mul_pd(self.twiddle4im, x1021n);
        let temp_b12_11 = _mm_mul_pd(self.twiddle8im, x1120n);
        let temp_b12_12 = _mm_mul_pd(self.twiddle11im, x1219n);
        let temp_b12_13 = _mm_mul_pd(self.twiddle1im, x1318n);
        let temp_b12_14 = _mm_mul_pd(self.twiddle13im, x1417n);
        let temp_b12_15 = _mm_mul_pd(self.twiddle6im, x1516n);
        let temp_b13_1 = _mm_mul_pd(self.twiddle13im, x130n);
        let temp_b13_2 = _mm_mul_pd(self.twiddle5im, x229n);
        let temp_b13_3 = _mm_mul_pd(self.twiddle8im, x328n);
        let temp_b13_4 = _mm_mul_pd(self.twiddle10im, x427n);
        let temp_b13_5 = _mm_mul_pd(self.twiddle3im, x526n);
        let temp_b13_6 = _mm_mul_pd(self.twiddle15im, x625n);
        let temp_b13_7 = _mm_mul_pd(self.twiddle2im, x724n);
        let temp_b13_8 = _mm_mul_pd(self.twiddle11im, x823n);
        let temp_b13_9 = _mm_mul_pd(self.twiddle7im, x922n);
        let temp_b13_10 = _mm_mul_pd(self.twiddle6im, x1021n);
        let temp_b13_11 = _mm_mul_pd(self.twiddle12im, x1120n);
        let temp_b13_12 = _mm_mul_pd(self.twiddle1im, x1219n);
        let temp_b13_13 = _mm_mul_pd(self.twiddle14im, x1318n);
        let temp_b13_14 = _mm_mul_pd(self.twiddle4im, x1417n);
        let temp_b13_15 = _mm_mul_pd(self.twiddle9im, x1516n);
        let temp_b14_1 = _mm_mul_pd(self.twiddle14im, x130n);
        let temp_b14_2 = _mm_mul_pd(self.twiddle3im, x229n);
        let temp_b14_3 = _mm_mul_pd(self.twiddle11im, x328n);
        let temp_b14_4 = _mm_mul_pd(self.twiddle6im, x427n);
        let temp_b14_5 = _mm_mul_pd(self.twiddle8im, x526n);
        let temp_b14_6 = _mm_mul_pd(self.twiddle9im, x625n);
        let temp_b14_7 = _mm_mul_pd(self.twiddle5im, x724n);
        let temp_b14_8 = _mm_mul_pd(self.twiddle12im, x823n);
        let temp_b14_9 = _mm_mul_pd(self.twiddle2im, x922n);
        let temp_b14_10 = _mm_mul_pd(self.twiddle15im, x1021n);
        let temp_b14_11 = _mm_mul_pd(self.twiddle1im, x1120n);
        let temp_b14_12 = _mm_mul_pd(self.twiddle13im, x1219n);
        let temp_b14_13 = _mm_mul_pd(self.twiddle4im, x1318n);
        let temp_b14_14 = _mm_mul_pd(self.twiddle10im, x1417n);
        let temp_b14_15 = _mm_mul_pd(self.twiddle7im, x1516n);
        let temp_b15_1 = _mm_mul_pd(self.twiddle15im, x130n);
        let temp_b15_2 = _mm_mul_pd(self.twiddle1im, x229n);
        let temp_b15_3 = _mm_mul_pd(self.twiddle14im, x328n);
        let temp_b15_4 = _mm_mul_pd(self.twiddle2im, x427n);
        let temp_b15_5 = _mm_mul_pd(self.twiddle13im, x526n);
        let temp_b15_6 = _mm_mul_pd(self.twiddle3im, x625n);
        let temp_b15_7 = _mm_mul_pd(self.twiddle12im, x724n);
        let temp_b15_8 = _mm_mul_pd(self.twiddle4im, x823n);
        let temp_b15_9 = _mm_mul_pd(self.twiddle11im, x922n);
        let temp_b15_10 = _mm_mul_pd(self.twiddle5im, x1021n);
        let temp_b15_11 = _mm_mul_pd(self.twiddle10im, x1120n);
        let temp_b15_12 = _mm_mul_pd(self.twiddle6im, x1219n);
        let temp_b15_13 = _mm_mul_pd(self.twiddle9im, x1318n);
        let temp_b15_14 = _mm_mul_pd(self.twiddle7im, x1417n);
        let temp_b15_15 = _mm_mul_pd(self.twiddle8im, x1516n);

        let temp_a1 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a1_1,
                _mm_add_pd(
                    temp_a1_2,
                    _mm_add_pd(
                        temp_a1_3,
                        _mm_add_pd(
                            temp_a1_4,
                            _mm_add_pd(
                                temp_a1_5,
                                _mm_add_pd(
                                    temp_a1_6,
                                    _mm_add_pd(
                                        temp_a1_7,
                                        _mm_add_pd(
                                            temp_a1_8,
                                            _mm_add_pd(
                                                temp_a1_9,
                                                _mm_add_pd(
                                                    temp_a1_10,
                                                    _mm_add_pd(
                                                        temp_a1_11,
                                                        _mm_add_pd(
                                                            temp_a1_12,
                                                            _mm_add_pd(
                                                                temp_a1_13,
                                                                _mm_add_pd(temp_a1_14, temp_a1_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a2 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a2_1,
                _mm_add_pd(
                    temp_a2_2,
                    _mm_add_pd(
                        temp_a2_3,
                        _mm_add_pd(
                            temp_a2_4,
                            _mm_add_pd(
                                temp_a2_5,
                                _mm_add_pd(
                                    temp_a2_6,
                                    _mm_add_pd(
                                        temp_a2_7,
                                        _mm_add_pd(
                                            temp_a2_8,
                                            _mm_add_pd(
                                                temp_a2_9,
                                                _mm_add_pd(
                                                    temp_a2_10,
                                                    _mm_add_pd(
                                                        temp_a2_11,
                                                        _mm_add_pd(
                                                            temp_a2_12,
                                                            _mm_add_pd(
                                                                temp_a2_13,
                                                                _mm_add_pd(temp_a2_14, temp_a2_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a3 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a3_1,
                _mm_add_pd(
                    temp_a3_2,
                    _mm_add_pd(
                        temp_a3_3,
                        _mm_add_pd(
                            temp_a3_4,
                            _mm_add_pd(
                                temp_a3_5,
                                _mm_add_pd(
                                    temp_a3_6,
                                    _mm_add_pd(
                                        temp_a3_7,
                                        _mm_add_pd(
                                            temp_a3_8,
                                            _mm_add_pd(
                                                temp_a3_9,
                                                _mm_add_pd(
                                                    temp_a3_10,
                                                    _mm_add_pd(
                                                        temp_a3_11,
                                                        _mm_add_pd(
                                                            temp_a3_12,
                                                            _mm_add_pd(
                                                                temp_a3_13,
                                                                _mm_add_pd(temp_a3_14, temp_a3_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a4 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a4_1,
                _mm_add_pd(
                    temp_a4_2,
                    _mm_add_pd(
                        temp_a4_3,
                        _mm_add_pd(
                            temp_a4_4,
                            _mm_add_pd(
                                temp_a4_5,
                                _mm_add_pd(
                                    temp_a4_6,
                                    _mm_add_pd(
                                        temp_a4_7,
                                        _mm_add_pd(
                                            temp_a4_8,
                                            _mm_add_pd(
                                                temp_a4_9,
                                                _mm_add_pd(
                                                    temp_a4_10,
                                                    _mm_add_pd(
                                                        temp_a4_11,
                                                        _mm_add_pd(
                                                            temp_a4_12,
                                                            _mm_add_pd(
                                                                temp_a4_13,
                                                                _mm_add_pd(temp_a4_14, temp_a4_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a5 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a5_1,
                _mm_add_pd(
                    temp_a5_2,
                    _mm_add_pd(
                        temp_a5_3,
                        _mm_add_pd(
                            temp_a5_4,
                            _mm_add_pd(
                                temp_a5_5,
                                _mm_add_pd(
                                    temp_a5_6,
                                    _mm_add_pd(
                                        temp_a5_7,
                                        _mm_add_pd(
                                            temp_a5_8,
                                            _mm_add_pd(
                                                temp_a5_9,
                                                _mm_add_pd(
                                                    temp_a5_10,
                                                    _mm_add_pd(
                                                        temp_a5_11,
                                                        _mm_add_pd(
                                                            temp_a5_12,
                                                            _mm_add_pd(
                                                                temp_a5_13,
                                                                _mm_add_pd(temp_a5_14, temp_a5_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a6 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a6_1,
                _mm_add_pd(
                    temp_a6_2,
                    _mm_add_pd(
                        temp_a6_3,
                        _mm_add_pd(
                            temp_a6_4,
                            _mm_add_pd(
                                temp_a6_5,
                                _mm_add_pd(
                                    temp_a6_6,
                                    _mm_add_pd(
                                        temp_a6_7,
                                        _mm_add_pd(
                                            temp_a6_8,
                                            _mm_add_pd(
                                                temp_a6_9,
                                                _mm_add_pd(
                                                    temp_a6_10,
                                                    _mm_add_pd(
                                                        temp_a6_11,
                                                        _mm_add_pd(
                                                            temp_a6_12,
                                                            _mm_add_pd(
                                                                temp_a6_13,
                                                                _mm_add_pd(temp_a6_14, temp_a6_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a7 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a7_1,
                _mm_add_pd(
                    temp_a7_2,
                    _mm_add_pd(
                        temp_a7_3,
                        _mm_add_pd(
                            temp_a7_4,
                            _mm_add_pd(
                                temp_a7_5,
                                _mm_add_pd(
                                    temp_a7_6,
                                    _mm_add_pd(
                                        temp_a7_7,
                                        _mm_add_pd(
                                            temp_a7_8,
                                            _mm_add_pd(
                                                temp_a7_9,
                                                _mm_add_pd(
                                                    temp_a7_10,
                                                    _mm_add_pd(
                                                        temp_a7_11,
                                                        _mm_add_pd(
                                                            temp_a7_12,
                                                            _mm_add_pd(
                                                                temp_a7_13,
                                                                _mm_add_pd(temp_a7_14, temp_a7_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a8 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a8_1,
                _mm_add_pd(
                    temp_a8_2,
                    _mm_add_pd(
                        temp_a8_3,
                        _mm_add_pd(
                            temp_a8_4,
                            _mm_add_pd(
                                temp_a8_5,
                                _mm_add_pd(
                                    temp_a8_6,
                                    _mm_add_pd(
                                        temp_a8_7,
                                        _mm_add_pd(
                                            temp_a8_8,
                                            _mm_add_pd(
                                                temp_a8_9,
                                                _mm_add_pd(
                                                    temp_a8_10,
                                                    _mm_add_pd(
                                                        temp_a8_11,
                                                        _mm_add_pd(
                                                            temp_a8_12,
                                                            _mm_add_pd(
                                                                temp_a8_13,
                                                                _mm_add_pd(temp_a8_14, temp_a8_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a9 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a9_1,
                _mm_add_pd(
                    temp_a9_2,
                    _mm_add_pd(
                        temp_a9_3,
                        _mm_add_pd(
                            temp_a9_4,
                            _mm_add_pd(
                                temp_a9_5,
                                _mm_add_pd(
                                    temp_a9_6,
                                    _mm_add_pd(
                                        temp_a9_7,
                                        _mm_add_pd(
                                            temp_a9_8,
                                            _mm_add_pd(
                                                temp_a9_9,
                                                _mm_add_pd(
                                                    temp_a9_10,
                                                    _mm_add_pd(
                                                        temp_a9_11,
                                                        _mm_add_pd(
                                                            temp_a9_12,
                                                            _mm_add_pd(
                                                                temp_a9_13,
                                                                _mm_add_pd(temp_a9_14, temp_a9_15),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a10 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a10_1,
                _mm_add_pd(
                    temp_a10_2,
                    _mm_add_pd(
                        temp_a10_3,
                        _mm_add_pd(
                            temp_a10_4,
                            _mm_add_pd(
                                temp_a10_5,
                                _mm_add_pd(
                                    temp_a10_6,
                                    _mm_add_pd(
                                        temp_a10_7,
                                        _mm_add_pd(
                                            temp_a10_8,
                                            _mm_add_pd(
                                                temp_a10_9,
                                                _mm_add_pd(
                                                    temp_a10_10,
                                                    _mm_add_pd(
                                                        temp_a10_11,
                                                        _mm_add_pd(
                                                            temp_a10_12,
                                                            _mm_add_pd(
                                                                temp_a10_13,
                                                                _mm_add_pd(
                                                                    temp_a10_14,
                                                                    temp_a10_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a11 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a11_1,
                _mm_add_pd(
                    temp_a11_2,
                    _mm_add_pd(
                        temp_a11_3,
                        _mm_add_pd(
                            temp_a11_4,
                            _mm_add_pd(
                                temp_a11_5,
                                _mm_add_pd(
                                    temp_a11_6,
                                    _mm_add_pd(
                                        temp_a11_7,
                                        _mm_add_pd(
                                            temp_a11_8,
                                            _mm_add_pd(
                                                temp_a11_9,
                                                _mm_add_pd(
                                                    temp_a11_10,
                                                    _mm_add_pd(
                                                        temp_a11_11,
                                                        _mm_add_pd(
                                                            temp_a11_12,
                                                            _mm_add_pd(
                                                                temp_a11_13,
                                                                _mm_add_pd(
                                                                    temp_a11_14,
                                                                    temp_a11_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a12 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a12_1,
                _mm_add_pd(
                    temp_a12_2,
                    _mm_add_pd(
                        temp_a12_3,
                        _mm_add_pd(
                            temp_a12_4,
                            _mm_add_pd(
                                temp_a12_5,
                                _mm_add_pd(
                                    temp_a12_6,
                                    _mm_add_pd(
                                        temp_a12_7,
                                        _mm_add_pd(
                                            temp_a12_8,
                                            _mm_add_pd(
                                                temp_a12_9,
                                                _mm_add_pd(
                                                    temp_a12_10,
                                                    _mm_add_pd(
                                                        temp_a12_11,
                                                        _mm_add_pd(
                                                            temp_a12_12,
                                                            _mm_add_pd(
                                                                temp_a12_13,
                                                                _mm_add_pd(
                                                                    temp_a12_14,
                                                                    temp_a12_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a13 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a13_1,
                _mm_add_pd(
                    temp_a13_2,
                    _mm_add_pd(
                        temp_a13_3,
                        _mm_add_pd(
                            temp_a13_4,
                            _mm_add_pd(
                                temp_a13_5,
                                _mm_add_pd(
                                    temp_a13_6,
                                    _mm_add_pd(
                                        temp_a13_7,
                                        _mm_add_pd(
                                            temp_a13_8,
                                            _mm_add_pd(
                                                temp_a13_9,
                                                _mm_add_pd(
                                                    temp_a13_10,
                                                    _mm_add_pd(
                                                        temp_a13_11,
                                                        _mm_add_pd(
                                                            temp_a13_12,
                                                            _mm_add_pd(
                                                                temp_a13_13,
                                                                _mm_add_pd(
                                                                    temp_a13_14,
                                                                    temp_a13_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a14 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a14_1,
                _mm_add_pd(
                    temp_a14_2,
                    _mm_add_pd(
                        temp_a14_3,
                        _mm_add_pd(
                            temp_a14_4,
                            _mm_add_pd(
                                temp_a14_5,
                                _mm_add_pd(
                                    temp_a14_6,
                                    _mm_add_pd(
                                        temp_a14_7,
                                        _mm_add_pd(
                                            temp_a14_8,
                                            _mm_add_pd(
                                                temp_a14_9,
                                                _mm_add_pd(
                                                    temp_a14_10,
                                                    _mm_add_pd(
                                                        temp_a14_11,
                                                        _mm_add_pd(
                                                            temp_a14_12,
                                                            _mm_add_pd(
                                                                temp_a14_13,
                                                                _mm_add_pd(
                                                                    temp_a14_14,
                                                                    temp_a14_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_a15 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                temp_a15_1,
                _mm_add_pd(
                    temp_a15_2,
                    _mm_add_pd(
                        temp_a15_3,
                        _mm_add_pd(
                            temp_a15_4,
                            _mm_add_pd(
                                temp_a15_5,
                                _mm_add_pd(
                                    temp_a15_6,
                                    _mm_add_pd(
                                        temp_a15_7,
                                        _mm_add_pd(
                                            temp_a15_8,
                                            _mm_add_pd(
                                                temp_a15_9,
                                                _mm_add_pd(
                                                    temp_a15_10,
                                                    _mm_add_pd(
                                                        temp_a15_11,
                                                        _mm_add_pd(
                                                            temp_a15_12,
                                                            _mm_add_pd(
                                                                temp_a15_13,
                                                                _mm_add_pd(
                                                                    temp_a15_14,
                                                                    temp_a15_15,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1 = _mm_add_pd(
            temp_b1_1,
            _mm_add_pd(
                temp_b1_2,
                _mm_add_pd(
                    temp_b1_3,
                    _mm_add_pd(
                        temp_b1_4,
                        _mm_add_pd(
                            temp_b1_5,
                            _mm_add_pd(
                                temp_b1_6,
                                _mm_add_pd(
                                    temp_b1_7,
                                    _mm_add_pd(
                                        temp_b1_8,
                                        _mm_add_pd(
                                            temp_b1_9,
                                            _mm_add_pd(
                                                temp_b1_10,
                                                _mm_add_pd(
                                                    temp_b1_11,
                                                    _mm_add_pd(
                                                        temp_b1_12,
                                                        _mm_add_pd(
                                                            temp_b1_13,
                                                            _mm_add_pd(temp_b1_14, temp_b1_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b2 = _mm_add_pd(
            temp_b2_1,
            _mm_add_pd(
                temp_b2_2,
                _mm_add_pd(
                    temp_b2_3,
                    _mm_add_pd(
                        temp_b2_4,
                        _mm_add_pd(
                            temp_b2_5,
                            _mm_add_pd(
                                temp_b2_6,
                                _mm_sub_pd(
                                    temp_b2_7,
                                    _mm_add_pd(
                                        temp_b2_8,
                                        _mm_add_pd(
                                            temp_b2_9,
                                            _mm_add_pd(
                                                temp_b2_10,
                                                _mm_add_pd(
                                                    temp_b2_11,
                                                    _mm_add_pd(
                                                        temp_b2_12,
                                                        _mm_add_pd(
                                                            temp_b2_13,
                                                            _mm_add_pd(temp_b2_14, temp_b2_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b3 = _mm_add_pd(
            temp_b3_1,
            _mm_add_pd(
                temp_b3_2,
                _mm_add_pd(
                    temp_b3_3,
                    _mm_add_pd(
                        temp_b3_4,
                        _mm_sub_pd(
                            temp_b3_5,
                            _mm_add_pd(
                                temp_b3_6,
                                _mm_add_pd(
                                    temp_b3_7,
                                    _mm_add_pd(
                                        temp_b3_8,
                                        _mm_add_pd(
                                            temp_b3_9,
                                            _mm_sub_pd(
                                                temp_b3_10,
                                                _mm_add_pd(
                                                    temp_b3_11,
                                                    _mm_add_pd(
                                                        temp_b3_12,
                                                        _mm_add_pd(
                                                            temp_b3_13,
                                                            _mm_add_pd(temp_b3_14, temp_b3_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b4 = _mm_add_pd(
            temp_b4_1,
            _mm_add_pd(
                temp_b4_2,
                _mm_sub_pd(
                    temp_b4_3,
                    _mm_add_pd(
                        temp_b4_4,
                        _mm_add_pd(
                            temp_b4_5,
                            _mm_add_pd(
                                temp_b4_6,
                                _mm_sub_pd(
                                    temp_b4_7,
                                    _mm_add_pd(
                                        temp_b4_8,
                                        _mm_add_pd(
                                            temp_b4_9,
                                            _mm_add_pd(
                                                temp_b4_10,
                                                _mm_sub_pd(
                                                    temp_b4_11,
                                                    _mm_add_pd(
                                                        temp_b4_12,
                                                        _mm_add_pd(
                                                            temp_b4_13,
                                                            _mm_add_pd(temp_b4_14, temp_b4_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b5 = _mm_add_pd(
            temp_b5_1,
            _mm_add_pd(
                temp_b5_2,
                _mm_sub_pd(
                    temp_b5_3,
                    _mm_add_pd(
                        temp_b5_4,
                        _mm_add_pd(
                            temp_b5_5,
                            _mm_sub_pd(
                                temp_b5_6,
                                _mm_add_pd(
                                    temp_b5_7,
                                    _mm_add_pd(
                                        temp_b5_8,
                                        _mm_sub_pd(
                                            temp_b5_9,
                                            _mm_add_pd(
                                                temp_b5_10,
                                                _mm_add_pd(
                                                    temp_b5_11,
                                                    _mm_sub_pd(
                                                        temp_b5_12,
                                                        _mm_add_pd(
                                                            temp_b5_13,
                                                            _mm_add_pd(temp_b5_14, temp_b5_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b6 = _mm_add_pd(
            temp_b6_1,
            _mm_sub_pd(
                temp_b6_2,
                _mm_add_pd(
                    temp_b6_3,
                    _mm_add_pd(
                        temp_b6_4,
                        _mm_sub_pd(
                            temp_b6_5,
                            _mm_add_pd(
                                temp_b6_6,
                                _mm_sub_pd(
                                    temp_b6_7,
                                    _mm_add_pd(
                                        temp_b6_8,
                                        _mm_add_pd(
                                            temp_b6_9,
                                            _mm_sub_pd(
                                                temp_b6_10,
                                                _mm_add_pd(
                                                    temp_b6_11,
                                                    _mm_sub_pd(
                                                        temp_b6_12,
                                                        _mm_add_pd(
                                                            temp_b6_13,
                                                            _mm_add_pd(temp_b6_14, temp_b6_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b7 = _mm_add_pd(
            temp_b7_1,
            _mm_sub_pd(
                temp_b7_2,
                _mm_add_pd(
                    temp_b7_3,
                    _mm_sub_pd(
                        temp_b7_4,
                        _mm_add_pd(
                            temp_b7_5,
                            _mm_sub_pd(
                                temp_b7_6,
                                _mm_add_pd(
                                    temp_b7_7,
                                    _mm_sub_pd(
                                        temp_b7_8,
                                        _mm_add_pd(
                                            temp_b7_9,
                                            _mm_add_pd(
                                                temp_b7_10,
                                                _mm_sub_pd(
                                                    temp_b7_11,
                                                    _mm_add_pd(
                                                        temp_b7_12,
                                                        _mm_sub_pd(
                                                            temp_b7_13,
                                                            _mm_add_pd(temp_b7_14, temp_b7_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b8 = _mm_sub_pd(
            temp_b8_1,
            _mm_add_pd(
                temp_b8_2,
                _mm_sub_pd(
                    temp_b8_3,
                    _mm_add_pd(
                        temp_b8_4,
                        _mm_sub_pd(
                            temp_b8_5,
                            _mm_add_pd(
                                temp_b8_6,
                                _mm_sub_pd(
                                    temp_b8_7,
                                    _mm_add_pd(
                                        temp_b8_8,
                                        _mm_sub_pd(
                                            temp_b8_9,
                                            _mm_add_pd(
                                                temp_b8_10,
                                                _mm_sub_pd(
                                                    temp_b8_11,
                                                    _mm_add_pd(
                                                        temp_b8_12,
                                                        _mm_sub_pd(
                                                            temp_b8_13,
                                                            _mm_add_pd(temp_b8_14, temp_b8_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b9 = _mm_sub_pd(
            temp_b9_1,
            _mm_add_pd(
                temp_b9_2,
                _mm_sub_pd(
                    temp_b9_3,
                    _mm_add_pd(
                        temp_b9_4,
                        _mm_sub_pd(
                            temp_b9_5,
                            _mm_sub_pd(
                                temp_b9_6,
                                _mm_add_pd(
                                    temp_b9_7,
                                    _mm_sub_pd(
                                        temp_b9_8,
                                        _mm_add_pd(
                                            temp_b9_9,
                                            _mm_sub_pd(
                                                temp_b9_10,
                                                _mm_add_pd(
                                                    temp_b9_11,
                                                    _mm_sub_pd(
                                                        temp_b9_12,
                                                        _mm_sub_pd(
                                                            temp_b9_13,
                                                            _mm_add_pd(temp_b9_14, temp_b9_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b10 = _mm_sub_pd(
            temp_b10_1,
            _mm_add_pd(
                temp_b10_2,
                _mm_sub_pd(
                    temp_b10_3,
                    _mm_sub_pd(
                        temp_b10_4,
                        _mm_add_pd(
                            temp_b10_5,
                            _mm_sub_pd(
                                temp_b10_6,
                                _mm_sub_pd(
                                    temp_b10_7,
                                    _mm_add_pd(
                                        temp_b10_8,
                                        _mm_sub_pd(
                                            temp_b10_9,
                                            _mm_sub_pd(
                                                temp_b10_10,
                                                _mm_add_pd(
                                                    temp_b10_11,
                                                    _mm_sub_pd(
                                                        temp_b10_12,
                                                        _mm_sub_pd(
                                                            temp_b10_13,
                                                            _mm_add_pd(temp_b10_14, temp_b10_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b11 = _mm_sub_pd(
            temp_b11_1,
            _mm_sub_pd(
                temp_b11_2,
                _mm_add_pd(
                    temp_b11_3,
                    _mm_sub_pd(
                        temp_b11_4,
                        _mm_sub_pd(
                            temp_b11_5,
                            _mm_add_pd(
                                temp_b11_6,
                                _mm_sub_pd(
                                    temp_b11_7,
                                    _mm_sub_pd(
                                        temp_b11_8,
                                        _mm_sub_pd(
                                            temp_b11_9,
                                            _mm_add_pd(
                                                temp_b11_10,
                                                _mm_sub_pd(
                                                    temp_b11_11,
                                                    _mm_sub_pd(
                                                        temp_b11_12,
                                                        _mm_add_pd(
                                                            temp_b11_13,
                                                            _mm_sub_pd(temp_b11_14, temp_b11_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b12 = _mm_sub_pd(
            temp_b12_1,
            _mm_sub_pd(
                temp_b12_2,
                _mm_sub_pd(
                    temp_b12_3,
                    _mm_add_pd(
                        temp_b12_4,
                        _mm_sub_pd(
                            temp_b12_5,
                            _mm_sub_pd(
                                temp_b12_6,
                                _mm_sub_pd(
                                    temp_b12_7,
                                    _mm_add_pd(
                                        temp_b12_8,
                                        _mm_sub_pd(
                                            temp_b12_9,
                                            _mm_sub_pd(
                                                temp_b12_10,
                                                _mm_sub_pd(
                                                    temp_b12_11,
                                                    _mm_sub_pd(
                                                        temp_b12_12,
                                                        _mm_add_pd(
                                                            temp_b12_13,
                                                            _mm_sub_pd(temp_b12_14, temp_b12_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b13 = _mm_sub_pd(
            temp_b13_1,
            _mm_sub_pd(
                temp_b13_2,
                _mm_sub_pd(
                    temp_b13_3,
                    _mm_sub_pd(
                        temp_b13_4,
                        _mm_sub_pd(
                            temp_b13_5,
                            _mm_add_pd(
                                temp_b13_6,
                                _mm_sub_pd(
                                    temp_b13_7,
                                    _mm_sub_pd(
                                        temp_b13_8,
                                        _mm_sub_pd(
                                            temp_b13_9,
                                            _mm_sub_pd(
                                                temp_b13_10,
                                                _mm_sub_pd(
                                                    temp_b13_11,
                                                    _mm_add_pd(
                                                        temp_b13_12,
                                                        _mm_sub_pd(
                                                            temp_b13_13,
                                                            _mm_sub_pd(temp_b13_14, temp_b13_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b14 = _mm_sub_pd(
            temp_b14_1,
            _mm_sub_pd(
                temp_b14_2,
                _mm_sub_pd(
                    temp_b14_3,
                    _mm_sub_pd(
                        temp_b14_4,
                        _mm_sub_pd(
                            temp_b14_5,
                            _mm_sub_pd(
                                temp_b14_6,
                                _mm_sub_pd(
                                    temp_b14_7,
                                    _mm_sub_pd(
                                        temp_b14_8,
                                        _mm_sub_pd(
                                            temp_b14_9,
                                            _mm_add_pd(
                                                temp_b14_10,
                                                _mm_sub_pd(
                                                    temp_b14_11,
                                                    _mm_sub_pd(
                                                        temp_b14_12,
                                                        _mm_sub_pd(
                                                            temp_b14_13,
                                                            _mm_sub_pd(temp_b14_14, temp_b14_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let temp_b15 = _mm_sub_pd(
            temp_b15_1,
            _mm_sub_pd(
                temp_b15_2,
                _mm_sub_pd(
                    temp_b15_3,
                    _mm_sub_pd(
                        temp_b15_4,
                        _mm_sub_pd(
                            temp_b15_5,
                            _mm_sub_pd(
                                temp_b15_6,
                                _mm_sub_pd(
                                    temp_b15_7,
                                    _mm_sub_pd(
                                        temp_b15_8,
                                        _mm_sub_pd(
                                            temp_b15_9,
                                            _mm_sub_pd(
                                                temp_b15_10,
                                                _mm_sub_pd(
                                                    temp_b15_11,
                                                    _mm_sub_pd(
                                                        temp_b15_12,
                                                        _mm_sub_pd(
                                                            temp_b15_13,
                                                            _mm_sub_pd(temp_b15_14, temp_b15_15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);
        let temp_b6_rot = self.rotate.rotate(temp_b6);
        let temp_b7_rot = self.rotate.rotate(temp_b7);
        let temp_b8_rot = self.rotate.rotate(temp_b8);
        let temp_b9_rot = self.rotate.rotate(temp_b9);
        let temp_b10_rot = self.rotate.rotate(temp_b10);
        let temp_b11_rot = self.rotate.rotate(temp_b11);
        let temp_b12_rot = self.rotate.rotate(temp_b12);
        let temp_b13_rot = self.rotate.rotate(temp_b13);
        let temp_b14_rot = self.rotate.rotate(temp_b14);
        let temp_b15_rot = self.rotate.rotate(temp_b15);

        let x0 = _mm_add_pd(
            values[0],
            _mm_add_pd(
                x130p,
                _mm_add_pd(
                    x229p,
                    _mm_add_pd(
                        x328p,
                        _mm_add_pd(
                            x427p,
                            _mm_add_pd(
                                x526p,
                                _mm_add_pd(
                                    x625p,
                                    _mm_add_pd(
                                        x724p,
                                        _mm_add_pd(
                                            x823p,
                                            _mm_add_pd(
                                                x922p,
                                                _mm_add_pd(
                                                    x1021p,
                                                    _mm_add_pd(
                                                        x1120p,
                                                        _mm_add_pd(
                                                            x1219p,
                                                            _mm_add_pd(
                                                                x1318p,
                                                                _mm_add_pd(x1417p, x1516p),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
        let x4 = _mm_add_pd(temp_a4, temp_b4_rot);
        let x5 = _mm_add_pd(temp_a5, temp_b5_rot);
        let x6 = _mm_add_pd(temp_a6, temp_b6_rot);
        let x7 = _mm_add_pd(temp_a7, temp_b7_rot);
        let x8 = _mm_add_pd(temp_a8, temp_b8_rot);
        let x9 = _mm_add_pd(temp_a9, temp_b9_rot);
        let x10 = _mm_add_pd(temp_a10, temp_b10_rot);
        let x11 = _mm_add_pd(temp_a11, temp_b11_rot);
        let x12 = _mm_add_pd(temp_a12, temp_b12_rot);
        let x13 = _mm_add_pd(temp_a13, temp_b13_rot);
        let x14 = _mm_add_pd(temp_a14, temp_b14_rot);
        let x15 = _mm_add_pd(temp_a15, temp_b15_rot);
        let x16 = _mm_sub_pd(temp_a15, temp_b15_rot);
        let x17 = _mm_sub_pd(temp_a14, temp_b14_rot);
        let x18 = _mm_sub_pd(temp_a13, temp_b13_rot);
        let x19 = _mm_sub_pd(temp_a12, temp_b12_rot);
        let x20 = _mm_sub_pd(temp_a11, temp_b11_rot);
        let x21 = _mm_sub_pd(temp_a10, temp_b10_rot);
        let x22 = _mm_sub_pd(temp_a9, temp_b9_rot);
        let x23 = _mm_sub_pd(temp_a8, temp_b8_rot);
        let x24 = _mm_sub_pd(temp_a7, temp_b7_rot);
        let x25 = _mm_sub_pd(temp_a6, temp_b6_rot);
        let x26 = _mm_sub_pd(temp_a5, temp_b5_rot);
        let x27 = _mm_sub_pd(temp_a4, temp_b4_rot);
        let x28 = _mm_sub_pd(temp_a3, temp_b3_rot);
        let x29 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x30 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
            x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30,
        ]
    }
}

//   _____ _____ ____ _____ ____
//  |_   _| ____/ ___|_   _/ ___|
//    | | |  _| \___ \ | | \___ \
//    | | | |___ ___) || |  ___) |
//    |_| |_____|____/ |_| |____/
//

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

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
    test_butterfly_32_func!(test_ssef32_butterfly7, SseF32Butterfly7, 7);
    test_butterfly_32_func!(test_ssef32_butterfly11, SseF32Butterfly11, 11);
    test_butterfly_32_func!(test_ssef32_butterfly13, SseF32Butterfly13, 13);
    test_butterfly_32_func!(test_ssef32_butterfly17, SseF32Butterfly17, 17);
    test_butterfly_32_func!(test_ssef32_butterfly19, SseF32Butterfly19, 19);
    test_butterfly_32_func!(test_ssef32_butterfly23, SseF32Butterfly23, 23);
    test_butterfly_32_func!(test_ssef32_butterfly29, SseF32Butterfly29, 29);
    test_butterfly_32_func!(test_ssef32_butterfly31, SseF32Butterfly31, 31);

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
    test_butterfly_64_func!(test_ssef64_butterfly7, SseF64Butterfly7, 7);
    test_butterfly_64_func!(test_ssef64_butterfly11, SseF64Butterfly11, 11);
    test_butterfly_64_func!(test_ssef64_butterfly13, SseF64Butterfly13, 13);
    test_butterfly_64_func!(test_ssef64_butterfly17, SseF64Butterfly17, 17);
    test_butterfly_64_func!(test_ssef64_butterfly19, SseF64Butterfly19, 19);
    test_butterfly_64_func!(test_ssef64_butterfly23, SseF64Butterfly23, 23);
    test_butterfly_64_func!(test_ssef64_butterfly29, SseF64Butterfly29, 29);
    test_butterfly_64_func!(test_ssef64_butterfly31, SseF64Butterfly31, 31);
}
