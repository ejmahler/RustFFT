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
    pub(crate) unsafe fn perform_dual_fft_direct(
        &self,
        values: [__m128; 7],
    ) -> [__m128; 7] {
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

        let temp_a1 = _mm_add_ps(_mm_add_ps(values[0], temp_a1_1), _mm_add_ps(temp_a1_2, temp_a1_3));
        let temp_a2 = _mm_add_ps(_mm_add_ps(values[0], temp_a2_1), _mm_add_ps(temp_a2_2, temp_a2_3));
        let temp_a3 = _mm_add_ps(_mm_add_ps(values[0], temp_a3_1), _mm_add_ps(temp_a3_2, temp_a3_3));

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
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [__m128d; 7]
    ) -> [__m128d; 7] {
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

        let temp_a1 = _mm_add_pd(values[0], _mm_add_pd(temp_a1_1, _mm_add_pd(temp_a1_2, temp_a1_3)));
        let temp_a2 = _mm_add_pd(values[0], _mm_add_pd(temp_a2_1, _mm_add_pd(temp_a2_2, temp_a2_3)));
        let temp_a3 = _mm_add_pd(values[0], _mm_add_pd(temp_a3_1, _mm_add_pd(temp_a3_2, temp_a3_3)));

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
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [__m128d; 11],
    ) -> [__m128d; 11] {
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

        let temp_a1 = _mm_add_pd(values[0], _mm_add_pd(temp_a1_1, _mm_add_pd(temp_a1_2, _mm_add_pd(temp_a1_3, _mm_add_pd(temp_a1_4, temp_a1_5)))));
        let temp_a2 = _mm_add_pd(values[0], _mm_add_pd(temp_a2_1, _mm_add_pd(temp_a2_2, _mm_add_pd(temp_a2_3, _mm_add_pd(temp_a2_4, temp_a2_5)))));
        let temp_a3 = _mm_add_pd(values[0], _mm_add_pd(temp_a3_1, _mm_add_pd(temp_a3_2, _mm_add_pd(temp_a3_3, _mm_add_pd(temp_a3_4, temp_a3_5)))));
        let temp_a4 = _mm_add_pd(values[0], _mm_add_pd(temp_a4_1, _mm_add_pd(temp_a4_2, _mm_add_pd(temp_a4_3, _mm_add_pd(temp_a4_4, temp_a4_5)))));
        let temp_a5 = _mm_add_pd(values[0], _mm_add_pd(temp_a5_1, _mm_add_pd(temp_a5_2, _mm_add_pd(temp_a5_3, _mm_add_pd(temp_a5_4, temp_a5_5)))));

        let temp_b1 = _mm_add_pd(temp_b1_1, _mm_add_pd(temp_b1_2, _mm_add_pd(temp_b1_3, _mm_add_pd(temp_b1_4, temp_b1_5))));
        let temp_b2 = _mm_add_pd(temp_b2_1, _mm_sub_pd(temp_b2_2, _mm_add_pd(temp_b2_3, _mm_add_pd(temp_b2_4, temp_b2_5))));
        let temp_b3 = _mm_sub_pd(temp_b3_1, _mm_add_pd(temp_b3_2, _mm_sub_pd(temp_b3_3, _mm_add_pd(temp_b3_4, temp_b3_5))));
        let temp_b4 = _mm_sub_pd(temp_b4_1, _mm_sub_pd(temp_b4_2, _mm_add_pd(temp_b4_3, _mm_sub_pd(temp_b4_4, temp_b4_5))));
        let temp_b5 = _mm_sub_pd(temp_b5_1, _mm_sub_pd(temp_b5_2, _mm_sub_pd(temp_b5_3, _mm_sub_pd(temp_b5_4, temp_b5_5))));

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);

        let x0 = _mm_add_pd(values[0], _mm_add_pd(x110p, _mm_add_pd(x29p, _mm_add_pd(x38p, _mm_add_pd(x47p, x56p)))));
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
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [__m128d; 13],
    ) -> [__m128d; 13] {
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
        
        let temp_a1 = _mm_add_pd(values[0], _mm_add_pd(temp_a1_1, _mm_add_pd(temp_a1_2, _mm_add_pd(temp_a1_3, _mm_add_pd(temp_a1_4, _mm_add_pd(temp_a1_5, temp_a1_6))))));
        let temp_a2 = _mm_add_pd(values[0], _mm_add_pd(temp_a2_1, _mm_add_pd(temp_a2_2, _mm_add_pd(temp_a2_3, _mm_add_pd(temp_a2_4, _mm_add_pd(temp_a2_5, temp_a2_6))))));
        let temp_a3 = _mm_add_pd(values[0], _mm_add_pd(temp_a3_1, _mm_add_pd(temp_a3_2, _mm_add_pd(temp_a3_3, _mm_add_pd(temp_a3_4, _mm_add_pd(temp_a3_5, temp_a3_6))))));
        let temp_a4 = _mm_add_pd(values[0], _mm_add_pd(temp_a4_1, _mm_add_pd(temp_a4_2, _mm_add_pd(temp_a4_3, _mm_add_pd(temp_a4_4, _mm_add_pd(temp_a4_5, temp_a4_6))))));
        let temp_a5 = _mm_add_pd(values[0], _mm_add_pd(temp_a5_1, _mm_add_pd(temp_a5_2, _mm_add_pd(temp_a5_3, _mm_add_pd(temp_a5_4, _mm_add_pd(temp_a5_5, temp_a5_6))))));
        let temp_a6 = _mm_add_pd(values[0], _mm_add_pd(temp_a6_1, _mm_add_pd(temp_a6_2, _mm_add_pd(temp_a6_3, _mm_add_pd(temp_a6_4, _mm_add_pd(temp_a6_5, temp_a6_6))))));
        
        let temp_b1 = _mm_add_pd(temp_b1_1, _mm_add_pd(temp_b1_2, _mm_add_pd(temp_b1_3, _mm_add_pd(temp_b1_4, _mm_add_pd(temp_b1_5, temp_b1_6)))));
        let temp_b2 = _mm_add_pd(temp_b2_1, _mm_add_pd(temp_b2_2, _mm_sub_pd(temp_b2_3, _mm_add_pd(temp_b2_4, _mm_add_pd(temp_b2_5, temp_b2_6)))));
        let temp_b3 = _mm_add_pd(temp_b3_1, _mm_sub_pd(temp_b3_2, _mm_add_pd(temp_b3_3, _mm_sub_pd(temp_b3_4, _mm_add_pd(temp_b3_5, temp_b3_6)))));
        let temp_b4 = _mm_sub_pd(temp_b4_1, _mm_add_pd(temp_b4_2, _mm_sub_pd(temp_b4_3, _mm_sub_pd(temp_b4_4, _mm_add_pd(temp_b4_5, temp_b4_6)))));
        let temp_b5 = _mm_sub_pd(temp_b5_1, _mm_sub_pd(temp_b5_2, _mm_sub_pd(temp_b5_3, _mm_add_pd(temp_b5_4, _mm_sub_pd(temp_b5_5, temp_b5_6)))));
        let temp_b6 = _mm_sub_pd(temp_b6_1, _mm_sub_pd(temp_b6_2, _mm_sub_pd(temp_b6_3, _mm_sub_pd(temp_b6_4, _mm_sub_pd(temp_b6_5, temp_b6_6)))));
        
        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        let temp_b3_rot = self.rotate.rotate(temp_b3);
        let temp_b4_rot = self.rotate.rotate(temp_b4);
        let temp_b5_rot = self.rotate.rotate(temp_b5);
        let temp_b6_rot = self.rotate.rotate(temp_b6);
        
        let x0 = _mm_add_pd(values[0], _mm_add_pd(x112p, _mm_add_pd(x211p, _mm_add_pd(x310p, _mm_add_pd(x49p, _mm_add_pd(x58p, x67p))))));
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


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::{check_fft_algorithm, compare_vectors};
    use crate::{algorithm::Dft, Direction, FftNum, Length};

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

}
