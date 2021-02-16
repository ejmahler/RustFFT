use num_complex::Complex;
use core::arch::x86_64::*;
//use std::mem::transmute;
//use std::time::{Duration, Instant};

use crate::{common::FftNum, FftDirection};

use crate::array_utils;
use crate::array_utils::{RawSlice, RawSliceMut};
use crate::common::{fft_error_inplace, fft_error_outofplace};
//use crate::twiddles;
use crate::{Direction, Fft, Length};


#[allow(unused)]
macro_rules! boilerplate_fft_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            #[inline(always)]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(RawSlice::new(buffer), RawSliceMut::new(buffer));
            }
        }
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

                let result = array_utils::iter_chunks_zipped(
                    input,
                    output,
                    self.len(),
                    |in_chunk, out_chunk| {
                        unsafe {
                            self.perform_fft_contiguous(
                                RawSlice::new(in_chunk),
                                RawSliceMut::new(out_chunk),
                            )
                        };
                    },
                );

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

                let result = array_utils::iter_chunks(buffer, self.len(), |chunk| unsafe {
                    self.perform_fft_butterfly(chunk)
                });

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

// double lenth 2 fft of a and b, given as [a0, b0], [a1, b1]
#[inline(always)]
unsafe fn double_fft2_interleaved_32(val02: __m128, val13: __m128) -> (__m128, __m128) {
    let temp0 = _mm_add_ps(val02, val13);
    let temp1 = _mm_sub_ps(val02, val13);
    (temp0, temp1)
}

// double lenth 2 fft of a and b, given as [a0, a1], [b0, b1]
#[inline(always)]
unsafe fn double_fft2_contiguous_32(left: __m128, right: __m128) -> (__m128, __m128) {
    let temp02 = _mm_shuffle_ps(left, right, 0x44);
    let temp13 = _mm_shuffle_ps(left, right, 0xEE);
    let temp0 = _mm_add_ps(temp02, temp13);
    let temp1 = _mm_sub_ps(temp02, temp13);
    (temp0, temp1)
}

struct Rotate90_32 {
    sign_1st: __m128,
    sign_2nd: __m128,
    sign_both: __m128,
}

impl Rotate90_32 {
    fn new(positive: bool) -> Self {
        let sign_1st = unsafe {
            if positive {
                _mm_set_ps(0.0, 0.0, 0.0, -0.0)
            }
            else {
                _mm_set_ps(0.0, 0.0, -0.0, 0.0)
            }
        };
        let sign_2nd = unsafe {
            if positive {
                _mm_set_ps(0.0, -0.0, 0.0, 0.0)
            }
            else {
                _mm_set_ps(-0.0, 0.0, 0.0, 0.0)
            }
        };
        let sign_both = unsafe {
            if positive {
                _mm_set_ps(0.0, -0.0, 0.0, -0.0)
            }
            else {
                _mm_set_ps(-0.0, 0.0, -0.0, 0.0)
            }
        };
        Self {
            sign_1st,
            sign_2nd,
            sign_both,
        }
    }

    #[inline(always)]
    unsafe fn rotate_2nd(&self, values: __m128) -> __m128 {
        let temp = _mm_shuffle_ps(values, values, 0xB4);
        _mm_xor_ps(temp, self.sign_2nd)
    }

    #[inline(always)]
    unsafe fn rotate_1st(&self, values: __m128) -> __m128 {
        let temp = _mm_shuffle_ps(values, values, 0xE1);
        _mm_xor_ps(temp, self.sign_1st)
    }

    #[inline(always)]
    unsafe fn rotate_both(&self, values: __m128) -> __m128 {
        let temp = _mm_shuffle_ps(values, values, 0xB1);
        _mm_xor_ps(temp, self.sign_1st)
    }
}

pub(crate) struct Rotate90_64 {
    sign: __m128d,
}

impl Rotate90_64 {
    fn new(positive: bool) -> Self {
        let sign = unsafe {
            if positive {
                _mm_set_pd(0.0, -0.0)
            }
            else {
                _mm_set_pd(-0.0, 0.0)
            }
        };
        Self {
            sign,
        }
    }

    #[inline(always)]
    unsafe fn rotate(&self, values: __m128d) -> __m128d {
        let temp = _mm_shuffle_pd(values, values, 0x01);
        _mm_xor_pd(temp, self.sign)
    }
}



#[inline(always)]
unsafe fn single_fft2_64(left: __m128d, right: __m128d) -> (__m128d, __m128d) {
    let temp0 = _mm_add_pd(left, right);
    let temp1 = _mm_sub_pd(left, right);
    (temp0, temp1)
}


pub struct Sse32Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90_32,
    //rot: __m128,
}

pub struct Sse64Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90_64,
}



boilerplate_fft_butterfly!(Sse32Butterfly4, 4, |this: &Sse32Butterfly4<_>| this.direction);
impl<T: FftNum> Sse32Butterfly4<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        let rotate = if direction == FftDirection::Inverse {
            Rotate90_32::new(true)
        }
        else {
            Rotate90_32::new(false)
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

        let (temp0, temp1) = self.perform_fft_direct(value01, value23);

        let array0 = std::mem::transmute::<__m128, [Complex<f32>; 2]>(temp0);
        let array1 = std::mem::transmute::<__m128, [Complex<f32>; 2]>(temp1);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array0[0];
        *output_slice.add(1) = array0[1];
        *output_slice.add(2) = array1[0];
        *output_slice.add(3) = array1[1];
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value01: __m128,
        value23: __m128,
    ) -> (__m128, __m128) {
        let (temp0, mut temp1) = double_fft2_interleaved_32(value01, value23);
        temp1 = self.rotate.rotate_2nd(temp1);
        double_fft2_contiguous_32(temp0, temp1)
    }
}


boilerplate_fft_butterfly!(Sse64Butterfly4, 4, |this: &Sse64Butterfly4<_>| this.direction);
impl<T: FftNum> Sse64Butterfly4<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        let rotate = if direction == FftDirection::Inverse {
            Rotate90_64::new(true)
        }
        else {
            Rotate90_64::new(false)
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
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm
        let value0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let value1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let value2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let value3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);

        let (out0, out1, out2, out3) = self.perform_fft_direct(value0, value1, value2, value3);

        let val0 = std::mem::transmute::<__m128d, Complex<f64>>(out0);
        let val1 = std::mem::transmute::<__m128d, Complex<f64>>(out1);
        let val2 = std::mem::transmute::<__m128d, Complex<f64>>(out2);
        let val3 = std::mem::transmute::<__m128d, Complex<f64>>(out3);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val0;
        *output_slice.add(1) = val1;
        *output_slice.add(2) = val2;
        *output_slice.add(3) = val3;
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: __m128d,
        value1: __m128d,
        value2: __m128d,
        value3: __m128d,
    ) -> (__m128d, __m128d, __m128d, __m128d) {
        let (temp0, temp2) = single_fft2_64(value0, value2);
        let (temp1, temp3) = single_fft2_64(value1, value3);

        let temp3 = self.rotate.rotate(temp3);

        let (out0, out1) = single_fft2_64(temp0, temp1);
        let (out2, out3) = single_fft2_64(temp2, temp3);
        (out0, out2, out1, out3)
    }
    
}


pub struct Sse32Butterfly8<T> {
    root2: __m128,
    direction: FftDirection,
    bf4: Sse32Butterfly4<T>,
    rotate90: Rotate90_32,
}
boilerplate_fft_butterfly!(Sse32Butterfly8, 8, |this: &Sse32Butterfly8<_>| this.direction);
impl<T: FftNum> Sse32Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        let bf4 = Sse32Butterfly4::new(direction);
        let root2 = unsafe {
            _mm_load1_ps(&0.5f32.sqrt())
        };
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90_32::new(true)
        }
        else {
            Rotate90_32::new(false)
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
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm
        let in01 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let in23 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let in45 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let in67 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);

        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose the input into the scratch
        //let mut scratch0 = [input.load(0), input.load(2), input.load(4), input.load(6)];
        //let mut scratch1 = [input.load(1), input.load(3), input.load(5), input.load(7)];

        // step 2: column FFTs
        //butterfly4.perform_fft_butterfly(&mut scratch0);
        //butterfly4.perform_fft_butterfly(&mut scratch1);
        println!("scr0 {:?} {:?}", in01, in23);
        println!("scr1 {:?} {:?}", in45, in67);

        // transpose
        let in02 = _mm_shuffle_ps(in01, in23, 0x44);
        let in13 = _mm_shuffle_ps(in01, in23, 0xEE);
        let in46 = _mm_shuffle_ps(in45, in67, 0x44);
        let in57 = _mm_shuffle_ps(in45, in67, 0xEE);

        println!("scr0 sh {:?} {:?}", in02, in13);
        println!("scr1 sh {:?} {:?}", in46, in57);

        let (val0, val1) = self.bf4.perform_fft_direct(in02, in46);
        let (val2, val3) = self.bf4.perform_fft_direct(in13, in57);

        println!("scr0 fft {:?} {:?} ", val0, val1);
        println!("scr1 fft {:?} {:?} ", val2, val3);

        // step 3: apply twiddle factors
        //scratch1[1] = (twiddles::rotate_90(scratch1[1], self.direction) + scratch1[1]) * self.root2;
        //scratch1[2] = twiddles::rotate_90(scratch1[2], self.direction);
        //scratch1[3] = (twiddles::rotate_90(scratch1[3], self.direction) - scratch1[3]) * self.root2;

        //let val5b = self.rotate90.rotate(val5);
        //let val5c = _mm_add_pd(val5b, val5);
        //let val5 = _mm_mul_pd(val5c, self.root2);
//
        //let val6 = self.rotate90.rotate(val6);
//
        //let val7b = self.rotate90.rotate(val7);
        //let val7c = _mm_sub_pd(val7b, val7);
        //let val7 = _mm_mul_pd(val7c, self.root2);

        //println!("scr1 rot {:?} {:?} {:?} {:?}", val4, val5, val6, val7);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        //for i in 0..4 {
        //    Butterfly2::perform_fft_strided(&mut scratch0[i], &mut scratch1[i]);
        //}
        //let (out0, out4) = single_fft2_64(val0, val4);
        //let (out1, out5) = single_fft2_64(val1, val5);
        //let (out2, out6) = single_fft2_64(val2, val6);
        //let (out3, out7) = single_fft2_64(val3, val7);
//
        //println!("scr0 fft2 {:?} {:?} {:?} {:?}", out0, out1, out2, out3);
        //println!("scr1 fft2 {:?} {:?} {:?} {:?}", out4, out5, out6, out7);

        // step 6: copy data to the output. we don't need to transpose, because we skipped the step 4 transpose
        //for i in 0..4 {
        //    output.store(scratch0[i], i);
        //}
        //for i in 0..4 {
        //    output.store(scratch1[i], i + 4);
        //}

        let array0 = std::mem::transmute::<__m128, [Complex<f32>; 2]>(val0);
        let array1 = std::mem::transmute::<__m128, [Complex<f32>; 2]>(val1);
        let array2 = std::mem::transmute::<__m128, [Complex<f32>; 2]>(val2);
        let array3 = std::mem::transmute::<__m128, [Complex<f32>; 2]>(val3);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array0[0];
        *output_slice.add(1) = array0[1];
        *output_slice.add(2) = array1[0];
        *output_slice.add(3) = array1[1];
        *output_slice.add(4) = array2[0];
        *output_slice.add(5) = array2[1];
        *output_slice.add(6) = array3[0];
        *output_slice.add(7) = array3[1];
    }
}

pub struct Sse64Butterfly8<T> {
    root2: __m128d,
    direction: FftDirection,
    bf4: Sse64Butterfly4<T>,
    rotate90: Rotate90_64,
}
boilerplate_fft_butterfly!(Sse64Butterfly8, 8, |this: &Sse64Butterfly8<_>| this.direction);
impl<T: FftNum> Sse64Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        let bf4 = Sse64Butterfly4::new(direction);
        let root2 = unsafe {
            _mm_load1_pd(&0.5f64.sqrt())
        };
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90_64::new(true)
        }
        else {
            Rotate90_64::new(false)
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
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm
        let in0 = _mm_loadu_pd(input.as_ptr() as *const f64);
        let in1 = _mm_loadu_pd(input.as_ptr().add(1) as *const f64);
        let in2 = _mm_loadu_pd(input.as_ptr().add(2) as *const f64);
        let in3 = _mm_loadu_pd(input.as_ptr().add(3) as *const f64);
        let in4 = _mm_loadu_pd(input.as_ptr().add(4) as *const f64);
        let in5 = _mm_loadu_pd(input.as_ptr().add(5) as *const f64);
        let in6 = _mm_loadu_pd(input.as_ptr().add(6) as *const f64);
        let in7 = _mm_loadu_pd(input.as_ptr().add(7) as *const f64);

        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose the input into the scratch
        //let mut scratch0 = [input.load(0), input.load(2), input.load(4), input.load(6)];
        //let mut scratch1 = [input.load(1), input.load(3), input.load(5), input.load(7)];

        // step 2: column FFTs
        //butterfly4.perform_fft_butterfly(&mut scratch0);
        //butterfly4.perform_fft_butterfly(&mut scratch1);
        println!("scr0 {:?} {:?} {:?} {:?}", in0, in2, in4, in6);
        println!("scr1 {:?} {:?} {:?} {:?}", in1, in3, in5, in7);

        let (val0, val1, val2, val3) = self.bf4.perform_fft_direct(in0, in2, in4, in6);
        let (val4, val5, val6, val7) = self.bf4.perform_fft_direct(in1, in3, in5, in7);

        println!("scr0 fft {:?} {:?} {:?} {:?}", val0, val1, val2, val3);
        println!("scr1 fft {:?} {:?} {:?} {:?}", val4, val5, val6, val7);

        // step 3: apply twiddle factors
        //scratch1[1] = (twiddles::rotate_90(scratch1[1], self.direction) + scratch1[1]) * self.root2;
        //scratch1[2] = twiddles::rotate_90(scratch1[2], self.direction);
        //scratch1[3] = (twiddles::rotate_90(scratch1[3], self.direction) - scratch1[3]) * self.root2;

        let val5b = self.rotate90.rotate(val5);
        let val5c = _mm_add_pd(val5b, val5);
        let val5 = _mm_mul_pd(val5c, self.root2);

        let val6 = self.rotate90.rotate(val6);

        let val7b = self.rotate90.rotate(val7);
        let val7c = _mm_sub_pd(val7b, val7);
        let val7 = _mm_mul_pd(val7c, self.root2);

        println!("scr1 rot {:?} {:?} {:?} {:?}", val4, val5, val6, val7);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        //for i in 0..4 {
        //    Butterfly2::perform_fft_strided(&mut scratch0[i], &mut scratch1[i]);
        //}
        let (out0, out4) = single_fft2_64(val0, val4);
        let (out1, out5) = single_fft2_64(val1, val5);
        let (out2, out6) = single_fft2_64(val2, val6);
        let (out3, out7) = single_fft2_64(val3, val7);

        println!("scr0 fft2 {:?} {:?} {:?} {:?}", out0, out1, out2, out3);
        println!("scr1 fft2 {:?} {:?} {:?} {:?}", out4, out5, out6, out7);

        // step 6: copy data to the output. we don't need to transpose, because we skipped the step 4 transpose
        //for i in 0..4 {
        //    output.store(scratch0[i], i);
        //}
        //for i in 0..4 {
        //    output.store(scratch1[i], i + 4);
        //}

        let val0 = std::mem::transmute::<__m128d, Complex<f64>>(out0);
        let val1 = std::mem::transmute::<__m128d, Complex<f64>>(out1);
        let val2 = std::mem::transmute::<__m128d, Complex<f64>>(out2);
        let val3 = std::mem::transmute::<__m128d, Complex<f64>>(out3);
        let val4 = std::mem::transmute::<__m128d, Complex<f64>>(out4);
        let val5 = std::mem::transmute::<__m128d, Complex<f64>>(out5);
        let val6 = std::mem::transmute::<__m128d, Complex<f64>>(out6);
        let val7 = std::mem::transmute::<__m128d, Complex<f64>>(out7);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val0;
        *output_slice.add(1) = val1;
        *output_slice.add(2) = val2;
        *output_slice.add(3) = val3;
        *output_slice.add(4) = val4;
        *output_slice.add(5) = val5;
        *output_slice.add(6) = val6;
        *output_slice.add(7) = val7;
    }
}

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
    test_butterfly_32_func!(test_ssef32_butterfly4, Sse32Butterfly4, 4);

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
    test_butterfly_64_func!(test_ssef64_butterfly4, Sse64Butterfly4, 4);
    test_butterfly_64_func!(test_ssef64_butterfly8, Sse64Butterfly8, 8);

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
    //    let butterfly = Sse64Butterfly8::new(FftDirection::Forward);
    //    let mut input = vec![Complex::<f64>::new(1.0, 1.5), Complex::<f64>::new(2.0, 2.4),Complex::<f64>::new(7.0, 9.5),Complex::<f64>::new(-4.0, -4.5),
    //                    Complex::<f64>::new(-1.0, 5.5), Complex::<f64>::new(3.3, 2.8),Complex::<f64>::new(7.5, 3.5),Complex::<f64>::new(-14.0, -6.5)];
    //    let mut scratch = vec![Complex::<f64>::from(0.0); 0];
    //    butterfly.process_with_scratch(&mut input, &mut scratch);
    //    assert!(false);
    //}

    #[test]
    fn check_scalar_dummy32() {
        let butterfly = Sse32Butterfly8::new(FftDirection::Forward);
        let mut input = vec![Complex::<f32>::new(1.0, 1.5), Complex::<f32>::new(2.0, 2.4),Complex::<f32>::new(7.0, 9.5),Complex::<f32>::new(-4.0, -4.5),
                        Complex::<f32>::new(-1.0, 5.5), Complex::<f32>::new(3.3, 2.8),Complex::<f32>::new(7.5, 3.5),Complex::<f32>::new(-14.0, -6.5)];
        let mut scratch = vec![Complex::<f32>::from(0.0); 0];
        butterfly.process_with_scratch(&mut input, &mut scratch);
        assert!(false);
    }
}
