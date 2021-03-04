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

#[allow(unused)]
macro_rules! boilerplate_fft_sse_f32_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            #[target_feature(enable = "sse3")]
            //#[inline(always)]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(RawSlice::new(buffer), RawSliceMut::new(buffer));
            }

            #[target_feature(enable = "sse3")]
            //#[inline(always)]
            pub(crate) unsafe fn perform_dual_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_dual_fft_contiguous(RawSlice::new(buffer), RawSliceMut::new(buffer));
            }

            // Do multiple ffts over a longer vector inplace, called from "process_with_scratch" of Fft trait 
            #[target_feature(enable = "sse3")]
            pub(crate) unsafe fn perform_fft_butterfly_multi(
                &self,
                buffer: &mut [Complex<T>],
            ) -> Result<(), ()> {
                let len = buffer.len();
                let alldone = array_utils::iter_chunks(buffer, 2*self.len(), |chunk| {
                    self.perform_dual_fft_butterfly(chunk)
                });
                if alldone.is_err() && buffer.len()>=self.len()  {
                    self.perform_fft_butterfly(&mut buffer[len-self.len() ..]);
                }
                Ok(())
            }

            // Do multiple ffts over a longer vector outofplace, called from "process_outofplace_with_scratch" of Fft trait 
            #[target_feature(enable = "sse3")]
            pub(crate) unsafe fn perform_oop_fft_butterfly_multi(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
            ) -> Result<(), ()> {
                let len = input.len();
                let alldone = array_utils::iter_chunks_zipped(input, output, 2*self.len(), |in_chunk, out_chunk| {
                    self.perform_dual_fft_contiguous(RawSlice::new(in_chunk), RawSliceMut::new(out_chunk))
                });
                if alldone.is_err() && input.len()>=self.len()  {
                    self.perform_fft_contiguous(RawSlice::new(&input[len-self.len() ..]), RawSliceMut::new(&mut output[len-self.len() ..]), );
                }
                Ok(())
            }
        }
    }
}

macro_rules! boilerplate_fft_sse_f64_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            // Do a single fft
            #[target_feature(enable = "sse3")]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(RawSlice::new(buffer), RawSliceMut::new(buffer));
            }

            // Do multiple ffts over a longer vector inplace, called from "process_with_scratch" of Fft trait 
            #[target_feature(enable = "sse3")]
            pub(crate) unsafe fn perform_fft_butterfly_multi(
                &self,
                buffer: &mut [Complex<T>],
            ) -> Result<(), ()> {
                array_utils::iter_chunks(buffer, self.len(), |chunk| {
                    self.perform_fft_butterfly(chunk)
                })
            }

            // Do multiple ffts over a longer vector outofplace, called from "process_outofplace_with_scratch" of Fft trait 
            #[target_feature(enable = "sse3")]
            pub(crate) unsafe fn perform_oop_fft_butterfly_multi(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
            ) -> Result<(), ()> {
                array_utils::iter_chunks_zipped(input, output, self.len(), |in_chunk, out_chunk| {
                    self.perform_fft_contiguous(
                        RawSlice::new(in_chunk),
                        RawSliceMut::new(out_chunk),
                    )
                })
            }
        }
    }
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

                //let result = array_utils::iter_chunks(buffer, self.len(), |chunk| unsafe {
                //    self.perform_fft_butterfly(chunk)
                //});
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
        _input: RawSlice<Complex<T>>,
        _output: RawSliceMut<Complex<T>>,
    ) {
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        _input: RawSlice<Complex<T>>,
        _output: RawSliceMut<Complex<T>>,
    ) {
    }

    //#[inline(always)]
    //pub(crate) unsafe fn perform_fft_direct(
    //    &self,
    //    values: __m128,
    //) -> __m128 {
    //    values
    //}
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
        _input: RawSlice<Complex<T>>,
        _output: RawSliceMut<Complex<T>>,
    ) {
    }

    //#[inline(always)]
    //pub(crate) unsafe fn perform_fft_direct(
    //    &self,
    //    values: __m128d,
    //) -> __m128d {
    //    values
    //}
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let values = _mm_loadu_ps(input.as_ptr() as *const f32);

        let temp = self.perform_fft_direct(values);

        let array = std::mem::transmute::<__m128, [Complex<f32>; 2]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[1];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let values_a = _mm_loadu_ps(input.as_ptr() as *const f32);
        let values_b = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);

        let temp = self.perform_dual_fft_direct(values_a, values_b);

        let array = std::mem::transmute::<[__m128; 2], [Complex<f32>; 4]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[2];
        *output_slice.add(2) = array[1];
        *output_slice.add(3) = array[3];
    }

    // length 2 fft of a, given as [a0, a1]
    // result is [A0, A1]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: __m128) -> __m128 {
        solo_fft2_f32(values)
    }

    // dual length 2 fft of a, given as [a0, a1]
    // result is [A0, A1]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(&self, values_a: __m128, values_b: __m128) -> [__m128; 2] {
        dual_fft2_contiguous_f32(values_a, values_b)
    }
}

// double lenth 2 fft of a and b, given as [a0, b0], [a1, b1]
// result is [A0, B0], [A1, B1]
#[inline(always)]
unsafe fn dual_fft2_interleaved_f32(val02: __m128, val13: __m128) -> [__m128; 2] {
    let temp0 = _mm_add_ps(val02, val13);
    let temp1 = _mm_sub_ps(val02, val13);
    [temp0, temp1]
}

// double lenth 2 fft of a and b, given as [a0, a1], [b0, b1]
// result is [A0, B0], [A1, B1]
#[inline(always)]
unsafe fn dual_fft2_contiguous_f32(left: __m128, right: __m128) -> [__m128; 2] {
    let temp02 = _mm_shuffle_ps(left, right, 0x44);
    let temp13 = _mm_shuffle_ps(left, right, 0xEE);
    let temp0 = _mm_add_ps(temp02, temp13);
    let temp1 = _mm_sub_ps(temp02, temp13);
    [temp0, temp1]
}

// length 2 fft of a, given as [a0, a1]
// result is [A0, A1]
#[inline(always)]
unsafe fn solo_fft2_f32(values: __m128) -> __m128 {
    let temp = _mm_shuffle_ps(values, values, 0x4E);
    let sign = _mm_set_ps(-0.0, -0.0, 0.0, 0.0);
    let temp2 = _mm_xor_ps(values, sign);
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let value1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);

        let out = self.perform_fft_direct(value0, value1);

        let val = std::mem::transmute::<[__m128d; 2], [Complex<f64>; 2]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
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
unsafe fn solo_fft2_f64(left: __m128d, right: __m128d) -> [__m128d; 2] {
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value0x = _mm_castpd_ps(_mm_load_sd(input.as_ptr() as *const f64));
        let value12 = _mm_loadu_ps(input.as_ptr().add(1) as *const f32);

        let temp = self.perform_fft_direct(value0x, value12);

        let array = std::mem::transmute::<[__m128; 2], [Complex<f32>; 4]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[2];
        *output_slice.add(2) = array[3];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2b0 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);

        let value0 = pack_1and2_f32(valuea0a1, valuea2b0);
        let value1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let value2 = pack_1and2_f32(valuea2b0, valueb1b2);

        let out = self.perform_dual_fft_direct(value0, value1, value2);

        let val = std::mem::transmute::<[__m128; 3], [Complex<f32>; 6]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[1];
        *output_slice.add(4) = val[3];
        *output_slice.add(5) = val[5];
    }

    // length 3 fft of a, given as [a0, 0.0], [a1, a2]
    // result is [A0, X], [A1, A2]
    // The value X should be discarded.
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0x: __m128,
        value12: __m128,
    ) -> [__m128; 2] {
        // This is a SSE translation of the scalar 5-point butterfly 
        let rev12 = invert_2nd_f32(reverse_f32(value12));
        let temp12pn = self.rotate.rotate_2nd(_mm_add_ps(value12, rev12));
        let twiddled = _mm_mul_ps(temp12pn, self.twiddle);
        let temp = _mm_add_ps(value0x, twiddled);
        let out12 = solo_fft2_f32(temp);
        let out0x = _mm_add_ps(value0x, temp12pn);
        [out0x, out12]
    }

    // length 3 dual fft of a, given as (a0, b0), (a1, b1), (a2, b2).
    // result is [(A0, B0), (A1, B1), (A2, B2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(
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
        let temp_b =  _mm_mul_ps(self.twiddle1im, n_rot);

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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let value1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let value2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);

        let out = self.perform_fft_direct(value0, value1, value2);

        let val = std::mem::transmute::<[__m128d; 3], [Complex<f64>; 3]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
    }

    // length 3 fft of a, given as a0, a1, a2.
    // result is [A0, A1, A2]
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
        let temp_b =  _mm_mul_pd(self.twiddle1im, n_rot);

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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value01 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let value23 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);

        let temp = self.perform_fft_direct(value01, value23);

        let array = std::mem::transmute::<[__m128; 2], [Complex<f32>; 4]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[1];
        *output_slice.add(2) = array[2];
        *output_slice.add(3) = array[3];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value01a = _mm_loadu_ps(input.as_ptr() as *const f32);
        let value23a = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let value01b = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let value23b = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);

        let value0ab = pack_1st_f32(value01a, value01b);
        let value1ab = pack_2nd_f32(value01a, value01b);
        let value2ab = pack_1st_f32(value23a, value23b);
        let value3ab = pack_2nd_f32(value23a, value23b);

        let temp = self.perform_dual_fft_direct(value0ab, value1ab, value2ab, value3ab);

        let array = std::mem::transmute::<[__m128; 4], [Complex<f32>; 8]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[2];
        *output_slice.add(2) = array[4];
        *output_slice.add(3) = array[6];
        *output_slice.add(4) = array[1];
        *output_slice.add(5) = array[3];
        *output_slice.add(6) = array[5];
        *output_slice.add(7) = array[7];
    }

    // length 4 fft of a, given as [a0, a1], [a2, a3]
    // result is [[A0, A1], [A2, A3]]
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
        let mut temp = dual_fft2_interleaved_f32(value01, value23);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        temp[1] = self.rotate.rotate_2nd(temp[1]);

        // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

        // step 5: row FFTs
        // and
        // step 6: transpose by swapping index 1 and 2
        dual_fft2_contiguous_f32(temp[0], temp[1])
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(
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
        let temp0 = dual_fft2_interleaved_f32(values0, values2);
        let mut temp1 = dual_fft2_interleaved_f32(values1, values3);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        temp1[1] = self.rotate.rotate_both(temp1[1]);

        // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

        // step 5: row FFTs
        let out0 = dual_fft2_interleaved_f32(temp0[0], temp1[0]);
        let out2 = dual_fft2_interleaved_f32(temp0[1], temp1[1]);

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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let value1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let value2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let value3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);

        let out = self.perform_fft_direct(value0, value1, value2, value3);

        let val = std::mem::transmute::<[__m128d; 4], [Complex<f64>; 4]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value00 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let value12 = _mm_loadu_ps(input.as_ptr().add(1) as *const f32);
        let value34 = _mm_loadu_ps(input.as_ptr().add(3) as *const f32);

        let temp = self.perform_fft_direct(value00, value12, value34);

        let array = std::mem::transmute::<[__m128; 3], [Complex<f32>; 6]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[2];
        *output_slice.add(2) = array[3];
        *output_slice.add(3) = array[5];
        *output_slice.add(4) = array[4];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let valuea0a1 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let valuea2a3 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let valuea4b0 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let valueb1b2 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let valueb3b4 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);

        let value0 = pack_1and2_f32(valuea0a1, valuea4b0);
        let value1 = pack_2and1_f32(valuea0a1, valueb1b2);
        let value2 = pack_1and2_f32(valuea2a3, valueb1b2);
        let value3 = pack_2and1_f32(valuea2a3, valueb3b4);
        let value4 = pack_1and2_f32(valuea4b0, valueb3b4);

        let out = self.perform_dual_fft_direct(value0, value1, value2, value3, value4);

        let val = std::mem::transmute::<[__m128; 5], [Complex<f32>; 10]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[2];
        *output_slice.add(2) = val[4];
        *output_slice.add(3) = val[6];
        *output_slice.add(4) = val[8];
        *output_slice.add(5) = val[1];
        *output_slice.add(6) = val[3];
        *output_slice.add(7) = val[5];
        *output_slice.add(8) = val[7];
        *output_slice.add(9) = val[9];
    }

    // length 5 fft of a, given as [a0, a0], [a1, a2], [a3, a4].
    // result is [[A0, X], [A1, A2], [A4, A3]]
    // Note that X should not be used, and A4 and A3 are returned in reversed order.
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value00: __m128,
        value12: __m128,
        value34: __m128,
    ) -> [__m128; 3] {
        // This is a SSE translation of the scalar 5-point butterfly 
        let temp43 = reverse_f32(value34);
        let x1423p = _mm_add_ps(value12, temp43);
        let x1423n = _mm_sub_ps(value12, temp43);

        let x1414p = duplicate_1st_f32(x1423p); 
        let x2323p = duplicate_2nd_f32(x1423p); 
        let x1414n = duplicate_1st_f32(x1423n); 
        let x2323n = duplicate_2nd_f32(x1423n); 

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
        let x43 = _mm_sub_ps(temp_a, b_rot);
        [x00, x12, x43]
    }

        // length 3 dual fft of a, given as (a0, b0), (a1, b1), (a2, b2).
    // result is [(A0, B0), (A1, B1), (A2, B2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(
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

        let x0 = _mm_add_ps(value0, _mm_add_ps(x14p, x23p));
        let x1 = _mm_add_ps(temp_a1, self.rotate.rotate_both(temp_b1));
        let x2 = _mm_add_ps(temp_a2, self.rotate.rotate_both(temp_b2));
        let x3 = _mm_sub_ps(temp_a2, self.rotate.rotate_both(temp_b2));
        let x4 = _mm_sub_ps(temp_a1, self.rotate.rotate_both(temp_b1));
        [x0, x1, x2, x3, x4]
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let value1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let value2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let value3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let value4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);

        let out = self.perform_fft_direct(value0, value1, value2, value3, value4);

        let val = std::mem::transmute::<[__m128d; 5], [Complex<f64>; 5]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
    }

    // length 5 fft of a, given as a0, a1, a2, a3, a4.
    // result is [A0, A1, A2, A3, A4]
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
        
        let x0 = _mm_add_pd(value0, _mm_add_pd(x14p, x23p));
        let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
        let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
        let x3 = _mm_sub_pd(temp_a2, temp_b2_rot);
        let x4 = _mm_sub_pd(temp_a1, temp_b1_rot);
        [x0, x1, x2, x3, x4]
    }
}



//    __             _________  _     _ _   
//   / /_           |___ /___ \| |__ (_) |_ 
//  | '_ \   _____    |_ \ __) | '_ \| | __|
//  | (_) | |_____|  ___) / __/| |_) | | |_ 
//   \___/          |____/_____|_.__/|_|\__|
//                                         




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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let value1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let value2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let value3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let value4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let value5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);

        let out = self.perform_fft_direct(value0, value1, value2, value3, value4, value5);

        let val = std::mem::transmute::<[__m128d; 6], [Complex<f64>; 6]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val[0];
        *output_slice.add(1) = val[1];
        *output_slice.add(2) = val[2];
        *output_slice.add(3) = val[3];
        *output_slice.add(4) = val[4];
        *output_slice.add(5) = val[5];
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let in01 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let in23 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let in45 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let in67 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);

        let out = self.perform_fft_direct([in01, in23, in45, in67]);

        let outvals = std::mem::transmute::<[__m128; 4], [Complex<f32>; 8]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = outvals[0];
        *output_slice.add(1) = outvals[1];
        *output_slice.add(2) = outvals[2];
        *output_slice.add(3) = outvals[3];
        *output_slice.add(4) = outvals[4];
        *output_slice.add(5) = outvals[5];
        *output_slice.add(6) = outvals[6];
        *output_slice.add(7) = outvals[7];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value01a = _mm_loadu_ps(input.as_ptr() as *const f32);
        let value23a = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let value45a = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let value67a = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let value01b = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let value23b = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let value45b = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let value67b = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        

        let values0 = pack_1st_f32(value01a, value01b);
        let values1 = pack_2nd_f32(value01a, value01b);
        let values2 = pack_1st_f32(value23a, value23b);
        let values3 = pack_2nd_f32(value23a, value23b);
        let values4 = pack_1st_f32(value45a, value45b);
        let values5 = pack_2nd_f32(value45a, value45b);
        let values6 = pack_1st_f32(value67a, value67b);
        let values7 = pack_2nd_f32(value67a, value67b);

        let temp = self.perform_dual_fft_direct([values0, values1, values2, values3, values4, values5, values6, values7]);

        let array = std::mem::transmute::<[__m128; 8], [Complex<f32>; 16]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[2];
        *output_slice.add(2) = array[4];
        *output_slice.add(3) = array[6];
        *output_slice.add(4) = array[8];
        *output_slice.add(5) = array[10];
        *output_slice.add(6) = array[12];
        *output_slice.add(7) = array[14];
        *output_slice.add(8) = array[1];
        *output_slice.add(9) = array[3];
        *output_slice.add(10) = array[5];
        *output_slice.add(11) = array[7];
        *output_slice.add(12) = array[9];
        *output_slice.add(13) = array[11];
        *output_slice.add(14) = array[13];
        *output_slice.add(15) = array[15];
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, values: [__m128; 4]) -> [__m128; 4] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the input into the scratch
        let in02 = _mm_shuffle_ps(values[0], values[1], 0x44);
        let in13 = _mm_shuffle_ps(values[0], values[1], 0xEE);
        let in46 = _mm_shuffle_ps(values[2], values[3], 0x44);
        let in57 = _mm_shuffle_ps(values[2], values[3], 0xEE);

        // step 2: column FFTs
        let val0 = self.bf4.perform_fft_direct(in02, in46);
        let mut val2 = self.bf4.perform_fft_direct(in13, in57);

        // step 3: apply twiddle factors
        let val2b = self.rotate90.rotate_2nd(val2[0]);
        let val2c = _mm_add_ps(val2b, val2[0]);
        let val2d = _mm_mul_ps(val2c, self.root2);
        val2[0] = _mm_shuffle_ps(val2[0], val2d, 0xE4);

        let val3b = self.rotate90.rotate_both(val2[1]);
        let val3c = _mm_sub_ps(val3b, val2[1]);
        let val3d = _mm_mul_ps(val3c, self.root2);
        val2[1] = _mm_shuffle_ps(val3b, val3d, 0xE4);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        let out0 = dual_fft2_interleaved_f32(val0[0], val2[0]);
        let out1 = dual_fft2_interleaved_f32(val0[1], val2[1]);

        // step 6: rearrange and copy to buffer
        [out0[0], out1[0], out0[1], out1[1]]
    }

    #[inline(always)]
    unsafe fn perform_dual_fft_direct(&self, values: [__m128; 8]) -> [__m128; 8] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the input into the scratch
        // and
        // step 2: column FFTs
        let val03 = self
            .bf4
            .perform_dual_fft_direct(values[0], values[2], values[4], values[6]);
        let mut val47 = self
            .bf4
            .perform_dual_fft_direct(values[1], values[3], values[5], values[7]);

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
        let out0 = dual_fft2_interleaved_f32(val03[0], val47[0]);
        let out1 = dual_fft2_interleaved_f32(val03[1], val47[1]);
        let out2 = dual_fft2_interleaved_f32(val03[2], val47[2]);
        let out3 = dual_fft2_interleaved_f32(val03[3], val47[3]);

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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let in0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let in1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let in2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let in3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let in4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let in5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let in6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let in7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);

        let out = self.perform_fft_direct([in0, in1, in2, in3, in4, in5, in6, in7]);

        let outvals = std::mem::transmute::<[__m128d; 8], [Complex<f64>; 8]>(out);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;

        *output_slice.add(0) = outvals[0];
        *output_slice.add(1) = outvals[1];
        *output_slice.add(2) = outvals[2];
        *output_slice.add(3) = outvals[3];
        *output_slice.add(4) = outvals[4];
        *output_slice.add(5) = outvals[5];
        *output_slice.add(6) = outvals[6];
        *output_slice.add(7) = outvals[7];
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, values: [__m128d; 8]) -> [__m128d; 8] {
        // we're going to hardcode a step of split radix
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

        let out = self.perform_fft_direct([v0, v1, v2, v3, v4, v5,v6, v7, v8]);

        let val = std::mem::transmute::<[__m128d; 9], [Complex<f64>; 9]>(out);

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
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [__m128d; 9],
    ) -> [__m128d; 9] {
        // Algorithm: 3x3 mixed radix

        // Size-3 FFTs down the columns
        let mid0 = self.bf3.perform_fft_direct(values[0], values[3], values[6]);
        let mut mid1 = self.bf3.perform_fft_direct(values[1], values[4], values[7]);
        let mut mid2 = self.bf3.perform_fft_direct(values[2], values[5], values[8]);
 
        // Apply twiddle factors. Note that we're re-using twiddle2
        mid1[1] = complex_mul_f64(self.twiddle1, mid1[1]);
        mid1[2] = complex_mul_f64(self.twiddle2, mid1[2]);
        mid2[1] = complex_mul_f64(self.twiddle2, mid2[1]);
        mid2[2] = complex_mul_f64(self.twiddle4, mid2[2]);

        let [output0, output1, output2] =
            self.bf3.perform_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] =
            self.bf3.perform_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] =
            self.bf3.perform_fft_direct(mid0[2], mid1[2], mid2[2]);

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

        let out = self.perform_fft_direct([v0, v1, v2, v3, v4, v5,v6, v7, v8, v9]);

        let val = std::mem::transmute::<[__m128d; 10], [Complex<f64>; 10]>(out);

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
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [__m128d; 10],
    ) -> [__m128d; 10] {
        // Algorithm: 5x2 good-thomas

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = self.bf5.perform_fft_direct(values[0], values[2], values[4], values[6], values[8]);
        let mid1 = self.bf5.perform_fft_direct(values[5], values[7], values[9], values[1], values[3]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [output0, output1] =
            self.bf2.perform_fft_direct(mid0[0], mid1[0]);
        let [output2, output3] =
            self.bf2.perform_fft_direct(mid0[1], mid1[1]);
        let [output4, output5] =
            self.bf2.perform_fft_direct(mid0[2], mid1[2]);
        let [output6, output7] =
            self.bf2.perform_fft_direct(mid0[3], mid1[3]);
        let [output8, output9] =
            self.bf2.perform_fft_direct(mid0[4], mid1[4]);
        [
            output0, output3, output4, output7, output8, output1, output2, output5, output6, output9
        ]
    }
}


//   _ ____            _________  _     _ _   
//  / |___ \          |___ /___ \| |__ (_) |_ 
//  | | __) |  _____    |_ \ __) | '_ \| | __|
//  | |/ __/  |_____|  ___) / __/| |_) | | |_ 
//  |_|_____|         |____/_____|_.__/|_|\__|
//      

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

        let out = self.perform_fft_direct([v0, v1, v2, v3, v4, v5,v6, v7, v8, v9, v10, v11]);

        let val = std::mem::transmute::<[__m128d; 12], [Complex<f64>; 12]>(out);

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
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [__m128d; 12],
    ) -> [__m128d; 12] {
        // Algorithm: 4x3 good-thomas

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = self.bf4.perform_fft_direct(values[0], values[3], values[6], values[9]);
        let mid1 = self.bf4.perform_fft_direct(values[4], values[7], values[10], values[1]);
        let mid2 = self.bf4.perform_fft_direct(values[8], values[11], values[2], values[5]);


        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [output0, output1, output2] =
            self.bf3.perform_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] =
            self.bf3.perform_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] =
            self.bf3.perform_fft_direct(mid0[2], mid1[2], mid2[2]);
        let [output9, output10, output11] =
            self.bf3.perform_fft_direct(mid0[3], mid1[3], mid2[3]);

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

        let out = self.perform_fft_direct([v0, v1, v2, v3, v4, v5,v6, v7, v8, v9, v10, v11, v12, v13, v14]);

        let val = std::mem::transmute::<[__m128d; 15], [Complex<f64>; 15]>(out);

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
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [__m128d; 15],
    ) -> [__m128d; 15] {
        // Algorithm: 5x3 good-thomas

        // Size-5 FFTs down the columns of our reordered array
        let mid0 = self.bf5.perform_fft_direct(values[0], values[3], values[6], values[9], values[12]);
        let mid1 = self.bf5.perform_fft_direct(values[5], values[8], values[11], values[14], values[2]);
        let mid2 = self.bf5.perform_fft_direct(values[10], values[13], values[1], values[4], values[7]);


        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [output0, output1, output2] =
            self.bf3.perform_fft_direct(mid0[0], mid1[0], mid2[0]);
        let [output3, output4, output5] =
            self.bf3.perform_fft_direct(mid0[1], mid1[1], mid2[1]);
        let [output6, output7, output8] =
            self.bf3.perform_fft_direct(mid0[2], mid1[2], mid2[2]);
        let [output9, output10, output11] =
            self.bf3.perform_fft_direct(mid0[3], mid1[3], mid2[3]);
        let [output12, output13, output14] =
            self.bf3.perform_fft_direct(mid0[4], mid1[4], mid2[4]);

        [
            output0, output4, output8, output9, output13, output2, output3, output7, output11,
            output12, output1, output5, output6, output10, output14
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let in0 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let in2 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let in4 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let in6 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let in8 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let in10 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let in12 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let in14 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);

        let result = self.perform_fft_direct([in0, in2, in4, in6, in8, in10, in12, in14]);

        let values = std::mem::transmute::<[__m128; 8], [Complex<f32>; 16]>(result);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = values[0];
        *output_slice.add(1) = values[1];
        *output_slice.add(2) = values[2];
        *output_slice.add(3) = values[3];
        *output_slice.add(4) = values[4];
        *output_slice.add(5) = values[5];
        *output_slice.add(6) = values[6];
        *output_slice.add(7) = values[7];
        *output_slice.add(8) = values[8];
        *output_slice.add(9) = values[9];
        *output_slice.add(10) = values[10];
        *output_slice.add(11) = values[11];
        *output_slice.add(12) = values[12];
        *output_slice.add(13) = values[13];
        *output_slice.add(14) = values[14];
        *output_slice.add(15) = values[15];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value01a = _mm_loadu_ps(input.as_ptr() as *const f32);
        let value23a = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let value45a = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let value67a = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let value89a = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let value1011a = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let value1213a = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let value1415a = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let value01b = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let value23b = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let value45b = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let value67b = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let value89b = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);
        let value1011b = _mm_loadu_ps(input.as_ptr().add(26) as *const f32);
        let value1213b = _mm_loadu_ps(input.as_ptr().add(28) as *const f32);
        let value1415b = _mm_loadu_ps(input.as_ptr().add(30) as *const f32);
        
        

        let values0 = pack_1st_f32(value01a, value01b);
        let values1 = pack_2nd_f32(value01a, value01b);
        let values2 = pack_1st_f32(value23a, value23b);
        let values3 = pack_2nd_f32(value23a, value23b);
        let values4 = pack_1st_f32(value45a, value45b);
        let values5 = pack_2nd_f32(value45a, value45b);
        let values6 = pack_1st_f32(value67a, value67b);
        let values7 = pack_2nd_f32(value67a, value67b);
        let values8 = pack_1st_f32(value89a, value89b);
        let values9 = pack_2nd_f32(value89a, value89b);
        let values10 = pack_1st_f32(value1011a, value1011b);
        let values11 = pack_2nd_f32(value1011a, value1011b);
        let values12 = pack_1st_f32(value1213a, value1213b);
        let values13 = pack_2nd_f32(value1213a, value1213b);
        let values14 = pack_1st_f32(value1415a, value1415b);
        let values15 = pack_2nd_f32(value1415a, value1415b);

        let temp = self.perform_dual_fft_direct([values0, values1, values2, values3, values4, values5, values6, values7, values8, values9, values10, values11, values12, values13, values14, values15]);

        let array = std::mem::transmute::<[__m128; 16], [Complex<f32>; 32]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[2];
        *output_slice.add(2) = array[4];
        *output_slice.add(3) = array[6];
        *output_slice.add(4) = array[8];
        *output_slice.add(5) = array[10];
        *output_slice.add(6) = array[12];
        *output_slice.add(7) = array[14];
        *output_slice.add(8) = array[16];
        *output_slice.add(9) = array[18];
        *output_slice.add(10) = array[20];
        *output_slice.add(11) = array[22];
        *output_slice.add(12) = array[24];
        *output_slice.add(13) = array[26];
        *output_slice.add(14) = array[28];
        *output_slice.add(15) = array[30];
        *output_slice.add(16) = array[1];
        *output_slice.add(17) = array[3];
        *output_slice.add(18) = array[5];
        *output_slice.add(19) = array[7];
        *output_slice.add(20) = array[9];
        *output_slice.add(21) = array[11];
        *output_slice.add(22) = array[13];
        *output_slice.add(23) = array[15];
        *output_slice.add(24) = array[17];
        *output_slice.add(25) = array[19];
        *output_slice.add(26) = array[21];
        *output_slice.add(27) = array[23];
        *output_slice.add(28) = array[25];
        *output_slice.add(29) = array[27];
        *output_slice.add(30) = array[29];
        *output_slice.add(31) = array[31];
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [__m128; 8]) -> [__m128; 8] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the input into the scratch
        let in0002 = pack_1st_f32(input[0], input[1]);
        let in0406 = pack_1st_f32(input[2], input[3]);
        let in0810 = pack_1st_f32(input[4], input[5]);
        let in1214 = pack_1st_f32(input[6], input[7]);

        let in0105 = pack_2nd_f32(input[0], input[2]);
        let in0913 = pack_2nd_f32(input[4], input[6]);
        let in1503 = pack_2nd_f32(input[7], input[1]);
        let in0711 = pack_2nd_f32(input[3], input[5]);

        let in_evens = [in0002, in0406, in0810, in1214];

        // step 2: column FFTs
        let evens = self.bf8.perform_fft_direct(in_evens);
        let mut odds1 = self.bf4.perform_fft_direct(in0105, in0913);
        let mut odds3 = self.bf4.perform_fft_direct(in1503, in0711);

        // step 3: apply twiddle factors
        odds1[0] = complex_dual_mul_f32(odds1[0], self.twiddle01);
        odds3[0] = complex_dual_mul_f32(odds3[0], self.twiddle01conj);

        odds1[1] = complex_dual_mul_f32(odds1[1], self.twiddle23);
        odds3[1] = complex_dual_mul_f32(odds3[1], self.twiddle23conj);

        // step 4: cross FFTs
        let mut temp0 = dual_fft2_interleaved_f32(odds1[0], odds3[0]);
        let mut temp1 = dual_fft2_interleaved_f32(odds1[1], odds3[1]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate_both(temp0[1]);
        temp1[1] = self.rotate90.rotate_both(temp1[1]);

        //step 5: copy/add/subtract data back to buffer
        let out0 = _mm_add_ps(evens[0], temp0[0]);
        let out1 = _mm_add_ps(evens[1], temp1[0]);
        let out2 = _mm_add_ps(evens[2], temp0[1]);
        let out3 = _mm_add_ps(evens[3], temp1[1]);
        let out4 = _mm_sub_ps(evens[0], temp0[0]);
        let out5 = _mm_sub_ps(evens[1], temp1[0]);
        let out6 = _mm_sub_ps(evens[2], temp0[1]);
        let out7 = _mm_sub_ps(evens[3], temp1[1]);

        [out0, out1, out2, out3, out4, out5, out6, out7]
    }

    #[inline(always)]
    unsafe fn perform_dual_fft_direct(&self, input: [__m128; 16]) -> [__m128; 16] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        // and
        // step 2: column FFTs
        let evens = self.bf8.perform_dual_fft_direct([
            input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14],
        ]);
        let mut odds1 = self
            .bf4
            .perform_dual_fft_direct(input[1], input[5], input[9], input[13]);
        let mut odds3 = self
            .bf4
            .perform_dual_fft_direct(input[15], input[3], input[7], input[11]);

        // step 3: apply twiddle factors
        odds1[1] = complex_dual_mul_f32(odds1[1], self.twiddle1);
        odds3[1] = complex_dual_mul_f32(odds3[1], self.twiddle1c);

        odds1[2] = complex_dual_mul_f32(odds1[2], self.twiddle2);
        odds3[2] = complex_dual_mul_f32(odds3[2], self.twiddle2c);

        odds1[3] = complex_dual_mul_f32(odds1[3], self.twiddle3);
        odds3[3] = complex_dual_mul_f32(odds3[3], self.twiddle3c);

        // step 4: cross FFTs
        let mut temp0 = dual_fft2_interleaved_f32(odds1[0], odds3[0]);
        let mut temp1 = dual_fft2_interleaved_f32(odds1[1], odds3[1]);
        let mut temp2 = dual_fft2_interleaved_f32(odds1[2], odds3[2]);
        let mut temp3 = dual_fft2_interleaved_f32(odds1[3], odds3[3]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate_both(temp0[1]);
        temp1[1] = self.rotate90.rotate_both(temp1[1]);
        temp2[1] = self.rotate90.rotate_both(temp2[1]);
        temp3[1] = self.rotate90.rotate_both(temp3[1]);

        //step 5: copy/add/subtract data back to buffer
        let out0 = _mm_add_ps(evens[0], temp0[0]);
        let out1 = _mm_add_ps(evens[1], temp1[0]);
        let out2 = _mm_add_ps(evens[2], temp2[0]);
        let out3 = _mm_add_ps(evens[3], temp3[0]);
        let out4 = _mm_add_ps(evens[4], temp0[1]);
        let out5 = _mm_add_ps(evens[5], temp1[1]);
        let out6 = _mm_add_ps(evens[6], temp2[1]);
        let out7 = _mm_add_ps(evens[7], temp3[1]);
        let out8 = _mm_sub_ps(evens[0], temp0[0]);
        let out9 = _mm_sub_ps(evens[1], temp1[0]);
        let out10 = _mm_sub_ps(evens[2], temp2[0]);
        let out11 = _mm_sub_ps(evens[3], temp3[0]);
        let out12 = _mm_sub_ps(evens[4], temp0[1]);
        let out13 = _mm_sub_ps(evens[5], temp1[1]);
        let out14 = _mm_sub_ps(evens[6], temp2[1]);
        let out15 = _mm_sub_ps(evens[7], temp3[1]);

        [
            out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13,
            out14, out15,
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let in0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let in1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let in2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let in3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let in4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let in5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let in6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let in7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let in8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let in9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let in10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);
        let in11 = _mm_loadu_pd(input.as_ptr().add(11) as *const f64);
        let in12 = _mm_loadu_pd(input.as_ptr().add(12) as *const f64);
        let in13 = _mm_loadu_pd(input.as_ptr().add(13) as *const f64);
        let in14 = _mm_loadu_pd(input.as_ptr().add(14) as *const f64);
        let in15 = _mm_loadu_pd(input.as_ptr().add(15) as *const f64);

        let result = self.perform_fft_direct([
            in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15,
        ]);

        let values = std::mem::transmute::<[__m128d; 16], [Complex<f64>; 16]>(result);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = values[0];
        *output_slice.add(1) = values[1];
        *output_slice.add(2) = values[2];
        *output_slice.add(3) = values[3];
        *output_slice.add(4) = values[4];
        *output_slice.add(5) = values[5];
        *output_slice.add(6) = values[6];
        *output_slice.add(7) = values[7];
        *output_slice.add(8) = values[8];
        *output_slice.add(9) = values[9];
        *output_slice.add(10) = values[10];
        *output_slice.add(11) = values[11];
        *output_slice.add(12) = values[12];
        *output_slice.add(13) = values[13];
        *output_slice.add(14) = values[14];
        *output_slice.add(15) = values[15];
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
        odds1[1] = complex_mul_f64(odds1[1], self.twiddle1);
        odds3[1] = complex_mul_f64(odds3[1], self.twiddle1c);

        odds1[2] = complex_mul_f64(odds1[2], self.twiddle2);
        odds3[2] = complex_mul_f64(odds3[2], self.twiddle2c);

        odds1[3] = complex_mul_f64(odds1[3], self.twiddle3);
        odds3[3] = complex_mul_f64(odds3[3], self.twiddle3c);

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
        let out0 = _mm_add_pd(evens[0], temp0[0]);
        let out1 = _mm_add_pd(evens[1], temp1[0]);
        let out2 = _mm_add_pd(evens[2], temp2[0]);
        let out3 = _mm_add_pd(evens[3], temp3[0]);
        let out4 = _mm_add_pd(evens[4], temp0[1]);
        let out5 = _mm_add_pd(evens[5], temp1[1]);
        let out6 = _mm_add_pd(evens[6], temp2[1]);
        let out7 = _mm_add_pd(evens[7], temp3[1]);
        let out8 = _mm_sub_pd(evens[0], temp0[0]);
        let out9 = _mm_sub_pd(evens[1], temp1[0]);
        let out10 = _mm_sub_pd(evens[2], temp2[0]);
        let out11 = _mm_sub_pd(evens[3], temp3[0]);
        let out12 = _mm_sub_pd(evens[4], temp0[1]);
        let out13 = _mm_sub_pd(evens[5], temp1[1]);
        let out14 = _mm_sub_pd(evens[6], temp2[1]);
        let out15 = _mm_sub_pd(evens[7], temp3[1]);

        [
            out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13,
            out14, out15,
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let in0 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let in2 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let in4 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let in6 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let in8 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let in10 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let in12 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let in14 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let in16 = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let in18 = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let in20 = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let in22 = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let in24 = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);
        let in26 = _mm_loadu_ps(input.as_ptr().add(26) as *const f32);
        let in28 = _mm_loadu_ps(input.as_ptr().add(28) as *const f32);
        let in30 = _mm_loadu_ps(input.as_ptr().add(30) as *const f32);

        let result = self.perform_fft_direct([in0, in2, in4, in6, in8, in10, in12, in14, in16, in18, in20, in22, in24, in26, in28, in30]);

        let values = std::mem::transmute::<[__m128; 16], [Complex<f32>; 32]>(result);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = values[0];
        *output_slice.add(1) = values[1];
        *output_slice.add(2) = values[2];
        *output_slice.add(3) = values[3];
        *output_slice.add(4) = values[4];
        *output_slice.add(5) = values[5];
        *output_slice.add(6) = values[6];
        *output_slice.add(7) = values[7];
        *output_slice.add(8) = values[8];
        *output_slice.add(9) = values[9];
        *output_slice.add(10) = values[10];
        *output_slice.add(11) = values[11];
        *output_slice.add(12) = values[12];
        *output_slice.add(13) = values[13];
        *output_slice.add(14) = values[14];
        *output_slice.add(15) = values[15];
        *output_slice.add(16) = values[16];
        *output_slice.add(17) = values[17];
        *output_slice.add(18) = values[18];
        *output_slice.add(19) = values[19];
        *output_slice.add(20) = values[20];
        *output_slice.add(21) = values[21];
        *output_slice.add(22) = values[22];
        *output_slice.add(23) = values[23];
        *output_slice.add(24) = values[24];
        *output_slice.add(25) = values[25];
        *output_slice.add(26) = values[26];
        *output_slice.add(27) = values[27];
        *output_slice.add(28) = values[28];
        *output_slice.add(29) = values[29];
        *output_slice.add(30) = values[30];
        *output_slice.add(31) = values[31];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value01a = _mm_loadu_ps(input.as_ptr() as *const f32);
        let value23a = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let value45a = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let value67a = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let value89a = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let value1011a = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let value1213a = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let value1415a = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);
        let value1617a = _mm_loadu_ps(input.as_ptr().add(16) as *const f32);
        let value1819a = _mm_loadu_ps(input.as_ptr().add(18) as *const f32);
        let value2021a = _mm_loadu_ps(input.as_ptr().add(20) as *const f32);
        let value2223a = _mm_loadu_ps(input.as_ptr().add(22) as *const f32);
        let value2425a = _mm_loadu_ps(input.as_ptr().add(24) as *const f32);
        let value2627a = _mm_loadu_ps(input.as_ptr().add(26) as *const f32);
        let value2829a = _mm_loadu_ps(input.as_ptr().add(28) as *const f32);
        let value3031a = _mm_loadu_ps(input.as_ptr().add(30) as *const f32);
        let value01b = _mm_loadu_ps(input.as_ptr().add(32) as *const f32);
        let value23b = _mm_loadu_ps(input.as_ptr().add(34) as *const f32);
        let value45b = _mm_loadu_ps(input.as_ptr().add(36) as *const f32);
        let value67b = _mm_loadu_ps(input.as_ptr().add(38) as *const f32);
        let value89b = _mm_loadu_ps(input.as_ptr().add(40) as *const f32);
        let value1011b = _mm_loadu_ps(input.as_ptr().add(42) as *const f32);
        let value1213b = _mm_loadu_ps(input.as_ptr().add(44) as *const f32);
        let value1415b = _mm_loadu_ps(input.as_ptr().add(46) as *const f32);
        let value1617b = _mm_loadu_ps(input.as_ptr().add(48) as *const f32);
        let value1819b = _mm_loadu_ps(input.as_ptr().add(50) as *const f32);
        let value2021b = _mm_loadu_ps(input.as_ptr().add(52) as *const f32);
        let value2223b = _mm_loadu_ps(input.as_ptr().add(54) as *const f32);
        let value2425b = _mm_loadu_ps(input.as_ptr().add(56) as *const f32);
        let value2627b = _mm_loadu_ps(input.as_ptr().add(58) as *const f32);
        let value2829b = _mm_loadu_ps(input.as_ptr().add(60) as *const f32);
        let value3031b = _mm_loadu_ps(input.as_ptr().add(62) as *const f32);

        let values0 = pack_1st_f32(value01a, value01b);
        let values1 = pack_2nd_f32(value01a, value01b);
        let values2 = pack_1st_f32(value23a, value23b);
        let values3 = pack_2nd_f32(value23a, value23b);
        let values4 = pack_1st_f32(value45a, value45b);
        let values5 = pack_2nd_f32(value45a, value45b);
        let values6 = pack_1st_f32(value67a, value67b);
        let values7 = pack_2nd_f32(value67a, value67b);
        let values8 = pack_1st_f32(value89a, value89b);
        let values9 = pack_2nd_f32(value89a, value89b);
        let values10 = pack_1st_f32(value1011a, value1011b);
        let values11 = pack_2nd_f32(value1011a, value1011b);
        let values12 = pack_1st_f32(value1213a, value1213b);
        let values13 = pack_2nd_f32(value1213a, value1213b);
        let values14 = pack_1st_f32(value1415a, value1415b);
        let values15 = pack_2nd_f32(value1415a, value1415b);
        let values16 = pack_1st_f32(value1617a, value1617b);
        let values17 = pack_2nd_f32(value1617a, value1617b);
        let values18 = pack_1st_f32(value1819a, value1819b);
        let values19 = pack_2nd_f32(value1819a, value1819b);
        let values20 = pack_1st_f32(value2021a, value2021b);
        let values21 = pack_2nd_f32(value2021a, value2021b);
        let values22 = pack_1st_f32(value2223a, value2223b);
        let values23 = pack_2nd_f32(value2223a, value2223b);
        let values24 = pack_1st_f32(value2425a, value2425b);
        let values25 = pack_2nd_f32(value2425a, value2425b);
        let values26 = pack_1st_f32(value2627a, value2627b);
        let values27 = pack_2nd_f32(value2627a, value2627b);
        let values28 = pack_1st_f32(value2829a, value2829b);
        let values29 = pack_2nd_f32(value2829a, value2829b);
        let values30 = pack_1st_f32(value3031a, value3031b);
        let values31 = pack_2nd_f32(value3031a, value3031b);

        let temp = self.perform_dual_fft_direct([values0, values1, values2, values3, values4, values5, values6, values7, values8, values9,
                                                 values10, values11, values12, values13, values14, values15, values16, values17, values18, values19,
                                                 values20, values21, values22, values23, values24, values25, values26, values27, values28, values29,
                                                 values30, values31]);

        let array = std::mem::transmute::<[__m128; 32], [Complex<f32>; 64]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[2];
        *output_slice.add(2) = array[4];
        *output_slice.add(3) = array[6];
        *output_slice.add(4) = array[8];
        *output_slice.add(5) = array[10];
        *output_slice.add(6) = array[12];
        *output_slice.add(7) = array[14];
        *output_slice.add(8) = array[16];
        *output_slice.add(9) = array[18];
        *output_slice.add(10) = array[20];
        *output_slice.add(11) = array[22];
        *output_slice.add(12) = array[24];
        *output_slice.add(13) = array[26];
        *output_slice.add(14) = array[28];
        *output_slice.add(15) = array[30];
        *output_slice.add(16) = array[32];
        *output_slice.add(17) = array[34];
        *output_slice.add(18) = array[36];
        *output_slice.add(19) = array[38];
        *output_slice.add(20) = array[40];
        *output_slice.add(21) = array[42];
        *output_slice.add(22) = array[44];
        *output_slice.add(23) = array[46];
        *output_slice.add(24) = array[48];
        *output_slice.add(25) = array[50];
        *output_slice.add(26) = array[52];
        *output_slice.add(27) = array[54];
        *output_slice.add(28) = array[56];
        *output_slice.add(29) = array[58];
        *output_slice.add(30) = array[60];
        *output_slice.add(31) = array[62];

        *output_slice.add(32) = array[1];
        *output_slice.add(33) = array[3];
        *output_slice.add(34) = array[5];
        *output_slice.add(35) = array[7];
        *output_slice.add(36) = array[9];
        *output_slice.add(37) = array[11];
        *output_slice.add(38) = array[13];
        *output_slice.add(39) = array[15];
        *output_slice.add(40) = array[17];
        *output_slice.add(41) = array[19];
        *output_slice.add(42) = array[21];
        *output_slice.add(43) = array[23];
        *output_slice.add(44) = array[25];
        *output_slice.add(45) = array[27];
        *output_slice.add(46) = array[29];
        *output_slice.add(47) = array[31];
        *output_slice.add(48) = array[33];
        *output_slice.add(49) = array[35];
        *output_slice.add(50) = array[37];
        *output_slice.add(51) = array[39];
        *output_slice.add(52) = array[41];
        *output_slice.add(53) = array[43];
        *output_slice.add(54) = array[45];
        *output_slice.add(55) = array[47];
        *output_slice.add(56) = array[49];
        *output_slice.add(57) = array[51];
        *output_slice.add(58) = array[53];
        *output_slice.add(59) = array[55];
        *output_slice.add(60) = array[57];
        *output_slice.add(61) = array[59];
        *output_slice.add(62) = array[61];
        *output_slice.add(63) = array[63];
    }
    
    #[inline(always)]
    unsafe fn perform_fft_direct(
        &self,
        input: [__m128; 16]
    ) -> [__m128; 16] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the input into the scratch
        let in0002 = pack_1st_f32(input[0], input[1]);
        let in0406 = pack_1st_f32(input[2], input[3]);
        let in0810 = pack_1st_f32(input[4], input[5]);
        let in1214 = pack_1st_f32(input[6], input[7]);
        let in1618 = pack_1st_f32(input[8], input[9]);
        let in2022 = pack_1st_f32(input[10], input[11]);
        let in2426 = pack_1st_f32(input[12], input[13]);
        let in2830 = pack_1st_f32(input[14], input[15]);

        let in0105 = pack_2nd_f32(input[0], input[2]);
        let in0913 = pack_2nd_f32(input[4], input[6]);
        let in1721 = pack_2nd_f32(input[8], input[10]);
        let in2529 = pack_2nd_f32(input[12], input[14]);

        let in3103 = pack_2nd_f32(input[15], input[1]);
        let in0711 = pack_2nd_f32(input[3], input[5]);
        let in1519 = pack_2nd_f32(input[7], input[9]);
        let in2327 = pack_2nd_f32(input[11], input[13]);

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
        odds1[0] = complex_dual_mul_f32(odds1[0], self.twiddle01);
        odds3[0] = complex_dual_mul_f32(odds3[0], self.twiddle01conj);

        odds1[1] = complex_dual_mul_f32(odds1[1], self.twiddle23);
        odds3[1] = complex_dual_mul_f32(odds3[1], self.twiddle23conj);

        odds1[2] = complex_dual_mul_f32(odds1[2], self.twiddle45);
        odds3[2] = complex_dual_mul_f32(odds3[2], self.twiddle45conj);

        odds1[3] = complex_dual_mul_f32(odds1[3], self.twiddle67);
        odds3[3] = complex_dual_mul_f32(odds3[3], self.twiddle67conj);

        // step 4: cross FFTs
        let mut temp0 = dual_fft2_interleaved_f32(odds1[0], odds3[0]);
        let mut temp1 = dual_fft2_interleaved_f32(odds1[1], odds3[1]);
        let mut temp2 = dual_fft2_interleaved_f32(odds1[2], odds3[2]);
        let mut temp3 = dual_fft2_interleaved_f32(odds1[3], odds3[3]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        temp0[1] = self.rotate90.rotate_both(temp0[1]);
        temp1[1] = self.rotate90.rotate_both(temp1[1]);
        temp2[1] = self.rotate90.rotate_both(temp2[1]);
        temp3[1] = self.rotate90.rotate_both(temp3[1]);

        //step 5: copy/add/subtract data back to buffer
        let out0 = _mm_add_ps(evens[0], temp0[0]);
        let out1 = _mm_add_ps(evens[1], temp1[0]);
        let out2 = _mm_add_ps(evens[2], temp2[0]);
        let out3 = _mm_add_ps(evens[3], temp3[0]);
        let out4 = _mm_add_ps(evens[4], temp0[1]);
        let out5 = _mm_add_ps(evens[5], temp1[1]);
        let out6 = _mm_add_ps(evens[6], temp2[1]);
        let out7 = _mm_add_ps(evens[7], temp3[1]);
        let out8 = _mm_sub_ps(evens[0], temp0[0]);
        let out9 = _mm_sub_ps(evens[1], temp1[0]);
        let out10 = _mm_sub_ps(evens[2], temp2[0]);
        let out11 = _mm_sub_ps(evens[3], temp3[0]);
        let out12 = _mm_sub_ps(evens[4], temp0[1]);
        let out13 = _mm_sub_ps(evens[5], temp1[1]);
        let out14 = _mm_sub_ps(evens[6], temp2[1]);
        let out15 = _mm_sub_ps(evens[7], temp3[1]);

        [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15]
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_dual_fft_direct(
        &self,
        input: [__m128; 32]
    ) -> [__m128; 32] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        // step 2: column FFTs
        let evens = self.bf16.perform_dual_fft_direct([
            input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14], input[16], input[18], input[20], input[22], input[24], input[26], input[28],
            input[30],
        ]);
        let mut odds1 = self
            .bf8
            .perform_dual_fft_direct([input[1], input[5], input[9], input[13], input[17], input[21], input[25], input[29]]);
        let mut odds3 = self
            .bf8
            .perform_dual_fft_direct([input[31], input[3], input[7], input[11], input[15], input[19], input[23], input[27]]);

        // step 3: apply twiddle factors
        odds1[1] = complex_dual_mul_f32(odds1[1], self.twiddle1);
        odds3[1] = complex_dual_mul_f32(odds3[1], self.twiddle1c);

        odds1[2] = complex_dual_mul_f32(odds1[2], self.twiddle2);
        odds3[2] = complex_dual_mul_f32(odds3[2], self.twiddle2c);

        odds1[3] = complex_dual_mul_f32(odds1[3], self.twiddle3);
        odds3[3] = complex_dual_mul_f32(odds3[3], self.twiddle3c);

        odds1[4] = complex_dual_mul_f32(odds1[4], self.twiddle4);
        odds3[4] = complex_dual_mul_f32(odds3[4], self.twiddle4c);

        odds1[5] = complex_dual_mul_f32(odds1[5], self.twiddle5);
        odds3[5] = complex_dual_mul_f32(odds3[5], self.twiddle5c);

        odds1[6] = complex_dual_mul_f32(odds1[6], self.twiddle6);
        odds3[6] = complex_dual_mul_f32(odds3[6], self.twiddle6c);

        odds1[7] = complex_dual_mul_f32(odds1[7], self.twiddle7);
        odds3[7] = complex_dual_mul_f32(odds3[7], self.twiddle7c);

        // step 4: cross FFTs
        let mut temp0 = dual_fft2_interleaved_f32(odds1[0], odds3[0]);
        let mut temp1 = dual_fft2_interleaved_f32(odds1[1], odds3[1]);
        let mut temp2 = dual_fft2_interleaved_f32(odds1[2], odds3[2]);
        let mut temp3 = dual_fft2_interleaved_f32(odds1[3], odds3[3]);
        let mut temp4 = dual_fft2_interleaved_f32(odds1[4], odds3[4]);
        let mut temp5 = dual_fft2_interleaved_f32(odds1[5], odds3[5]);
        let mut temp6 = dual_fft2_interleaved_f32(odds1[6], odds3[6]);
        let mut temp7 = dual_fft2_interleaved_f32(odds1[7], odds3[7]);

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
        let out0 = _mm_add_ps(evens[0], temp0[0]);
        let out1 = _mm_add_ps(evens[1], temp1[0]);
        let out2 = _mm_add_ps(evens[2], temp2[0]);
        let out3 = _mm_add_ps(evens[3], temp3[0]);
        let out4 = _mm_add_ps(evens[4], temp4[0]);
        let out5 = _mm_add_ps(evens[5], temp5[0]);
        let out6 = _mm_add_ps(evens[6], temp6[0]);
        let out7 = _mm_add_ps(evens[7], temp7[0]);
        let out8 = _mm_add_ps(evens[8], temp0[1]);
        let out9 = _mm_add_ps(evens[9], temp1[1]);
        let out10 = _mm_add_ps(evens[10], temp2[1]);
        let out11 = _mm_add_ps(evens[11], temp3[1]);
        let out12 = _mm_add_ps(evens[12], temp4[1]);
        let out13 = _mm_add_ps(evens[13], temp5[1]);
        let out14 = _mm_add_ps(evens[14], temp6[1]);
        let out15 = _mm_add_ps(evens[15], temp7[1]);
        let out16 = _mm_sub_ps(evens[0], temp0[0]);
        let out17 = _mm_sub_ps(evens[1], temp1[0]);
        let out18 = _mm_sub_ps(evens[2], temp2[0]);
        let out19 = _mm_sub_ps(evens[3], temp3[0]);
        let out20 = _mm_sub_ps(evens[4], temp4[0]);
        let out21 = _mm_sub_ps(evens[5], temp5[0]);
        let out22 = _mm_sub_ps(evens[6], temp6[0]);
        let out23 = _mm_sub_ps(evens[7], temp7[0]);
        let out24 = _mm_sub_ps(evens[8], temp0[1]);
        let out25 = _mm_sub_ps(evens[9], temp1[1]);
        let out26 = _mm_sub_ps(evens[10], temp2[1]);
        let out27 = _mm_sub_ps(evens[11], temp3[1]);
        let out28 = _mm_sub_ps(evens[12], temp4[1]);
        let out29 = _mm_sub_ps(evens[13], temp5[1]);
        let out30 = _mm_sub_ps(evens[14], temp6[1]);
        let out31 = _mm_sub_ps(evens[15], temp7[1]);

        [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, 
        out10, out11, out12, out13, out14, out15, out16, out17, out18, out19,
        out20, out21, out22, out23, out24, out25, out26, out27, out28, out29,
        out30, out31]
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
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let in0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let in1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let in2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let in3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let in4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let in5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let in6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let in7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);
        let in8 = _mm_loadu_pd(input.as_ptr().add(8) as *const f64);
        let in9 = _mm_loadu_pd(input.as_ptr().add(9) as *const f64);
        let in10 = _mm_loadu_pd(input.as_ptr().add(10) as *const f64);
        let in11 = _mm_loadu_pd(input.as_ptr().add(11) as *const f64);
        let in12 = _mm_loadu_pd(input.as_ptr().add(12) as *const f64);
        let in13 = _mm_loadu_pd(input.as_ptr().add(13) as *const f64);
        let in14 = _mm_loadu_pd(input.as_ptr().add(14) as *const f64);
        let in15 = _mm_loadu_pd(input.as_ptr().add(15) as *const f64);
        let in16 = _mm_loadu_pd(input.as_ptr().add(16) as *const f64);
        let in17 = _mm_loadu_pd(input.as_ptr().add(17) as *const f64);
        let in18 = _mm_loadu_pd(input.as_ptr().add(18) as *const f64);
        let in19 = _mm_loadu_pd(input.as_ptr().add(19) as *const f64);
        let in20 = _mm_loadu_pd(input.as_ptr().add(20) as *const f64);
        let in21 = _mm_loadu_pd(input.as_ptr().add(21) as *const f64);
        let in22 = _mm_loadu_pd(input.as_ptr().add(22) as *const f64);
        let in23 = _mm_loadu_pd(input.as_ptr().add(23) as *const f64);
        let in24 = _mm_loadu_pd(input.as_ptr().add(24) as *const f64);
        let in25 = _mm_loadu_pd(input.as_ptr().add(25) as *const f64);
        let in26 = _mm_loadu_pd(input.as_ptr().add(26) as *const f64);
        let in27 = _mm_loadu_pd(input.as_ptr().add(27) as *const f64);
        let in28 = _mm_loadu_pd(input.as_ptr().add(28) as *const f64);
        let in29 = _mm_loadu_pd(input.as_ptr().add(29) as *const f64);
        let in30 = _mm_loadu_pd(input.as_ptr().add(30) as *const f64);
        let in31 = _mm_loadu_pd(input.as_ptr().add(31) as *const f64);

        let result = self.perform_fft_direct([
            in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, 
            in10, in11, in12, in13, in14, in15, in16, in17, in18, in19,
            in20, in21, in22, in23, in24, in25, in26, in27, in28, in29,
            in30, in31
        ]);

        let values = std::mem::transmute::<[__m128d; 32], [Complex<f64>; 32]>(result);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = values[0];
        *output_slice.add(1) = values[1];
        *output_slice.add(2) = values[2];
        *output_slice.add(3) = values[3];
        *output_slice.add(4) = values[4];
        *output_slice.add(5) = values[5];
        *output_slice.add(6) = values[6];
        *output_slice.add(7) = values[7];
        *output_slice.add(8) = values[8];
        *output_slice.add(9) = values[9];
        *output_slice.add(10) = values[10];
        *output_slice.add(11) = values[11];
        *output_slice.add(12) = values[12];
        *output_slice.add(13) = values[13];
        *output_slice.add(14) = values[14];
        *output_slice.add(15) = values[15];
        *output_slice.add(16) = values[16];
        *output_slice.add(17) = values[17];
        *output_slice.add(18) = values[18];
        *output_slice.add(19) = values[19];
        *output_slice.add(20) = values[20];
        *output_slice.add(21) = values[21];
        *output_slice.add(22) = values[22];
        *output_slice.add(23) = values[23];
        *output_slice.add(24) = values[24];
        *output_slice.add(25) = values[25];
        *output_slice.add(26) = values[26];
        *output_slice.add(27) = values[27];
        *output_slice.add(28) = values[28];
        *output_slice.add(29) = values[29];
        *output_slice.add(30) = values[30];
        *output_slice.add(31) = values[31];
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(
        &self,
        input: [__m128d; 32]
    ) -> [__m128d; 32] {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        // step 2: column FFTs
        let evens = self.bf16.perform_fft_direct([
            input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14], input[16], input[18], input[20], input[22], input[24], input[26], input[28],
            input[30],
        ]);
        let mut odds1 = self
            .bf8
            .perform_fft_direct([input[1], input[5], input[9], input[13], input[17], input[21], input[25], input[29]]);
        let mut odds3 = self
            .bf8
            .perform_fft_direct([input[31], input[3], input[7], input[11], input[15], input[19], input[23], input[27]]);

        // step 3: apply twiddle factors
        odds1[1] = complex_mul_f64(odds1[1], self.twiddle1);
        odds3[1] = complex_mul_f64(odds3[1], self.twiddle1c);

        odds1[2] = complex_mul_f64(odds1[2], self.twiddle2);
        odds3[2] = complex_mul_f64(odds3[2], self.twiddle2c);

        odds1[3] = complex_mul_f64(odds1[3], self.twiddle3);
        odds3[3] = complex_mul_f64(odds3[3], self.twiddle3c);

        odds1[4] = complex_mul_f64(odds1[4], self.twiddle4);
        odds3[4] = complex_mul_f64(odds3[4], self.twiddle4c);

        odds1[5] = complex_mul_f64(odds1[5], self.twiddle5);
        odds3[5] = complex_mul_f64(odds3[5], self.twiddle5c);

        odds1[6] = complex_mul_f64(odds1[6], self.twiddle6);
        odds3[6] = complex_mul_f64(odds3[6], self.twiddle6c);

        odds1[7] = complex_mul_f64(odds1[7], self.twiddle7);
        odds3[7] = complex_mul_f64(odds3[7], self.twiddle7c);

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
        let out0 = _mm_add_pd(evens[0], temp0[0]);
        let out1 = _mm_add_pd(evens[1], temp1[0]);
        let out2 = _mm_add_pd(evens[2], temp2[0]);
        let out3 = _mm_add_pd(evens[3], temp3[0]);
        let out4 = _mm_add_pd(evens[4], temp4[0]);
        let out5 = _mm_add_pd(evens[5], temp5[0]);
        let out6 = _mm_add_pd(evens[6], temp6[0]);
        let out7 = _mm_add_pd(evens[7], temp7[0]);
        let out8 = _mm_add_pd(evens[8], temp0[1]);
        let out9 = _mm_add_pd(evens[9], temp1[1]);
        let out10 = _mm_add_pd(evens[10], temp2[1]);
        let out11 = _mm_add_pd(evens[11], temp3[1]);
        let out12 = _mm_add_pd(evens[12], temp4[1]);
        let out13 = _mm_add_pd(evens[13], temp5[1]);
        let out14 = _mm_add_pd(evens[14], temp6[1]);
        let out15 = _mm_add_pd(evens[15], temp7[1]);
        let out16 = _mm_sub_pd(evens[0], temp0[0]);
        let out17 = _mm_sub_pd(evens[1], temp1[0]);
        let out18 = _mm_sub_pd(evens[2], temp2[0]);
        let out19 = _mm_sub_pd(evens[3], temp3[0]);
        let out20 = _mm_sub_pd(evens[4], temp4[0]);
        let out21 = _mm_sub_pd(evens[5], temp5[0]);
        let out22 = _mm_sub_pd(evens[6], temp6[0]);
        let out23 = _mm_sub_pd(evens[7], temp7[0]);
        let out24 = _mm_sub_pd(evens[8], temp0[1]);
        let out25 = _mm_sub_pd(evens[9], temp1[1]);
        let out26 = _mm_sub_pd(evens[10], temp2[1]);
        let out27 = _mm_sub_pd(evens[11], temp3[1]);
        let out28 = _mm_sub_pd(evens[12], temp4[1]);
        let out29 = _mm_sub_pd(evens[13], temp5[1]);
        let out30 = _mm_sub_pd(evens[14], temp6[1]);
        let out31 = _mm_sub_pd(evens[15], temp7[1]);

        [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, 
        out10, out11, out12, out13, out14, out15, out16, out17, out18, out19,
        out20, out21, out22, out23, out24, out25, out26, out27, out28, out29,
        out30, out31]
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
    test_butterfly_32_func!(test_ssef32_butterfly2, SseF32Butterfly2, 2);
    test_butterfly_32_func!(test_ssef32_butterfly3, SseF32Butterfly3, 3);
    test_butterfly_32_func!(test_ssef32_butterfly4, SseF32Butterfly4, 4);
    test_butterfly_32_func!(test_ssef32_butterfly5, SseF32Butterfly5, 5);
    test_butterfly_32_func!(test_ssef32_butterfly8, SseF32Butterfly8, 8);
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

    //#[test]
    //fn check_type() {
    //    let butterfly = Butterfly4::new(FftDirection::Forward);
    //    let mut input = vec![Complex::<f64>::new(1.0, 1.5),Complex::<f64>::new(2.0, 2.4),Complex::<f64>::new(7.0, 9.5),Complex::<f64>::new(-4.0, -4.5)];
    //    let mut scratch = vec![Complex::<f64>::from(0.0); 0];
    //    butterfly.process_with_scratch(&mut input, &mut scratch);
    //    assert!(false);
    //}
    //
    //#[test]
    //fn check_type_32() {
    //    let butterfly = Butterfly4::new(FftDirection::Forward);
    //    let mut input = vec![Complex::<f32>::new(1.0, 1.5),Complex::<f32>::new(2.0, 2.4),Complex::<f32>::new(7.0, 9.5),Complex::<f32>::new(-4.0, -4.5)];
    //    let mut scratch = vec![Complex::<f32>::from(0.0); 0];
    //    butterfly.process_with_scratch(&mut input, &mut scratch);
    //    assert!(false);
    //}

    //#[test]
    //fn check_scalar_dummy() {
    //    let butterfly = SseF64Butterfly16::new(FftDirection::Forward);
    //    let mut input = vec![Complex::new(1.0, 1.5), Complex::new(2.0, 2.4),Complex::new(7.0, 9.5),Complex::new(-4.0, -4.5),
    //                        Complex::new(-1.0, 5.5), Complex::new(3.3, 2.8),Complex::new(7.5, 3.5),Complex::new(-14.0, -6.5),
    //                        Complex::new(-7.6, 53.5), Complex::new(-4.3, 2.2),Complex::new(8.1, 1.123),Complex::new(-24.0, -16.5),
    //                        Complex::new(-11.0, 55.0), Complex::new(33.3, 62.8),Complex::new(17.2, 23.5),Complex::new(-54.0, -3.8)];
    //    let mut scratch = vec![Complex::<f64>::from(0.0); 0];
    //    butterfly.process_with_scratch(&mut input, &mut scratch);
    //    assert!(false);
    //}
    //
    //#[test]
    //fn check_scalar_dummy32() {
    //    let butterfly = SseF32Butterfly16::new(FftDirection::Forward);
    //    let mut input = vec![Complex::new(1.0, 1.5), Complex::new(2.0, 2.4),Complex::new(7.0, 9.5),Complex::new(-4.0, -4.5),
    //                    Complex::new(-1.0, 5.5), Complex::new(3.3, 2.8),Complex::new(7.5, 3.5),Complex::new(-14.0, -6.5),
    //                    Complex::new(-7.6, 53.5), Complex::new(-4.3, 2.2),Complex::new(8.1, 1.123),Complex::new(-24.0, -16.5),
    //                    Complex::new(-11.0, 55.0), Complex::new(33.3, 62.8),Complex::new(17.2, 23.5),Complex::new(-54.0, -3.8)];
    //    let mut scratch = vec![Complex::<f32>::from(0.0); 0];
    //    butterfly.process_with_scratch(&mut input, &mut scratch);
    //    assert!(false);
    //}

    //#[test]
    //fn check_3_dummy() {
    //    let butterfly = SseF64Butterfly3::new(FftDirection::Forward);
    //    let mut input = vec![Complex::new(1.0, 1.5), Complex::new(2.0, 2.4),Complex::new(7.0, 9.5)];
    //    let mut scratch = vec![Complex::<f64>::from(0.0); 0];
    //    butterfly.process_with_scratch(&mut input, &mut scratch);
    //    assert!(false);
    //}
    //
    //#[test]
    //fn check_3_dummy32() {
    //    let butterfly = SseF32Butterfly3::new(FftDirection::Forward);
    //    let mut input = vec![Complex::new(1.0, 1.5), Complex::new(2.0, 2.4),Complex::new(7.0, 9.5)];
    //    let mut scratch = vec![Complex::<f32>::from(0.0); 0];
    //    butterfly.process_with_scratch(&mut input, &mut scratch);
    //    assert!(false);
    //}

    #[test]
    fn test_complex_mul_f64() {
        unsafe {
            let right = _mm_set_pd(1.0, 2.0);
            let left = _mm_set_pd(5.0, 7.0);
            println!("left: {:?}", left);
            println!("right: {:?}", right);
            let res = complex_mul_f64(left, right);
            println!("res: {:?}", res);
            let expected = _mm_set_pd(2.0 * 5.0 + 1.0 * 7.0, 2.0 * 7.0 - 1.0 * 5.0);
            assert_eq!(
                std::mem::transmute::<__m128d, Complex<f64>>(res),
                std::mem::transmute::<__m128d, Complex<f64>>(expected)
            );
        }
    }

    #[test]
    fn test_complex_dual_mul_f32() {
        unsafe {
            let val1 = Complex::<f32>::new(1.0, 2.5);
            let val2 = Complex::<f32>::new(3.2, 4.2);
            let val3 = Complex::<f32>::new(5.6, 6.2);
            let val4 = Complex::<f32>::new(7.4, 8.3);

            let nbr2 = _mm_set_ps(val4.im, val4.re, val3.im, val3.re);
            let nbr1 = _mm_set_ps(val2.im, val2.re, val1.im, val1.re);
            println!("left: {:?}", nbr1);
            println!("right: {:?}", nbr2);
            let res = complex_dual_mul_f32(nbr1, nbr2);
            let res = std::mem::transmute::<__m128, [Complex<f32>; 2]>(res);
            println!("res: {:?}", res);
            let expected = [val1 * val3, val2 * val4];
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn test_dual_fft4_32() {
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

            //println!("left: {:?}", nbr1);
            //println!("right: {:?}", nbr2);
            dft.process(&mut val_a);
            dft.process(&mut val_b);
            let res_both = bf4.perform_dual_fft_direct(p1, p2, p3, p4);

            let res = std::mem::transmute::<[__m128; 4], [Complex<f32>; 8]>(res_both);
            println!("sse: {:?}", res);
            println!("dft a: {:?}", val_a);
            println!("dft b: {:?}", val_b);
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
            println!("nbr1: {:?}", nbr1);
            println!("nbr2: {:?}", nbr2);
            let first = pack_1st_f32(nbr1, nbr2);
            let second = pack_2nd_f32(nbr1, nbr2);
            println!("first: {:?}", first);
            println!("second: {:?}", first);
            let first = std::mem::transmute::<__m128, [Complex<f32>; 2]>(first);
            let second = std::mem::transmute::<__m128, [Complex<f32>; 2]>(second);
            let first_expected = [Complex::new(1.0, 2.0), Complex::new(5.0, 6.0)];
            let second_expected = [Complex::new(3.0, 4.0), Complex::new(7.0, 8.0)];
            assert_eq!(first, first_expected);
            assert_eq!(second, second_expected);
        }
    }
}
