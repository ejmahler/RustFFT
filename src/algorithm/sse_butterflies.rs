use num_complex::Complex;
use core::arch::x86_64::*;
//use std::mem::transmute;
//use std::time::{Duration, Instant};

use crate::{common::FftNum, FftDirection};

use crate::array_utils;
use crate::array_utils::{RawSlice, RawSliceMut};
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::twiddles;
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




//  __  __       _   _               _________  _     _ _   
// |  \/  | __ _| |_| |__           |___ /___ \| |__ (_) |_ 
// | |\/| |/ _` | __| '_ \   _____    |_ \ __) | '_ \| | __|
// | |  | | (_| | |_| | | | |_____|  ___) / __/| |_) | | |_ 
// |_|  |_|\__,_|\__|_| |_|         |____/_____|_.__/|_|\__|
//                                                          



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
        _mm_xor_ps(temp, self.sign_both)
    }
}

#[inline(always)]
unsafe fn pack_1st_32(left: __m128, right: __m128) -> __m128 {
    _mm_shuffle_ps(left, right, 0x44)
}

#[inline(always)]
unsafe fn pack_2nd_32(left: __m128, right: __m128) -> __m128 {
    _mm_shuffle_ps(left, right, 0xEE)
}

// http://microperf.blogspot.com/2016/12/multiplying-two-complex-numbers-by-two.html
#[inline(always)]
unsafe fn complex_double_mul_32(ar_ai_cr_ci: __m128, br_bi_dr_di: __m128) -> __m128 {
    let sign = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
    let ar_ar_cr_cr = _mm_shuffle_ps(ar_ai_cr_ci, ar_ai_cr_ci, 0xA0);
    let ai_ai_ci_ci = _mm_shuffle_ps(ar_ai_cr_ci, ar_ai_cr_ci, 0xF5);
    let bi_br_di_dr = _mm_shuffle_ps(br_bi_dr_di, br_bi_dr_di, 0xB1);
    let arbr_arbi_crdr_crdi = _mm_mul_ps(ar_ar_cr_cr, br_bi_dr_di);
    let aibi_aibr_cidi_cidr = _mm_mul_ps(ai_ai_ci_ci, bi_br_di_dr);
    let naibi_aibr_ncidi_cidr = _mm_xor_ps(aibi_aibr_cidi_cidr, sign);
    _mm_add_ps(arbr_arbi_crdr_crdi, naibi_aibr_ncidi_cidr)
}


//  __  __       _   _                __   _  _   _     _ _   
// |  \/  | __ _| |_| |__            / /_ | || | | |__ (_) |_ 
// | |\/| |/ _` | __| '_ \   _____  | '_ \| || |_| '_ \| | __|
// | |  | | (_| | |_| | | | |_____| | (_) |__   _| |_) | | |_ 
// |_|  |_|\__,_|\__|_| |_|          \___/   |_| |_.__/|_|\__|
//                                                            



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
unsafe fn complex_mul_64(left: __m128d, right: __m128d) -> __m128d {
    let mul1 = _mm_mul_pd(left, right);
    let right_flipped = _mm_shuffle_pd(right, right, 0x01);
    let mul2 = _mm_mul_pd(left, right_flipped);
    let sign = _mm_set_pd(-0.0, 0.0);
    let mul1 = _mm_xor_pd(mul1, sign);
    let temp1 = _mm_shuffle_pd(mul1, mul2, 0x00);
    let temp2 = _mm_shuffle_pd(mul1, mul2, 0x03);
    _mm_add_pd(temp1, temp2)
}


pub struct Sse32Butterfly1<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_butterfly!(Sse32Butterfly1, 1, |this: &Sse32Butterfly1<_>| this.direction);
impl<T: FftNum> Sse32Butterfly1<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
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
    ) {}

    // length 2 fft of a, given as [a0, a1]
    // result is [A0, A1]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: __m128,
    ) -> __m128 {
        values
    }
}

pub struct Sse64Butterfly1<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_butterfly!(Sse64Butterfly1, 1, |this: &Sse64Butterfly1<_>| this.direction);
impl<T: FftNum> Sse64Butterfly1<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
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
    ) {}

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: __m128d,
    ) -> __m128d {
        values
    }
}


//  ____            _________  _     _ _   
// |___ \          |___ /___ \| |__ (_) |_ 
//   __) |  _____    |_ \ __) | '_ \| | __|
//  / __/  |_____|  ___) / __/| |_) | | |_ 
// |_____|         |____/_____|_.__/|_|\__|
//    

pub struct Sse32Butterfly2<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_butterfly!(Sse32Butterfly2, 2, |this: &Sse32Butterfly2<_>| this.direction);
impl<T: FftNum> Sse32Butterfly2<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
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

    // length 2 fft of a, given as [a0, a1]
    // result is [A0, A1]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: __m128,
    ) -> __m128 {
        let mut temp = _mm_shuffle_ps(values, values, 0x4E);
        let sign = _mm_set_ps(-0.0, -0.0, 0.0, 0.0);
        let temp2 = _mm_xor_ps(values, sign);
        _mm_add_ps(temp2, temp)
    }
}

// double lenth 2 fft of a and b, given as [a0, b0], [a1, b1]
// result is [A0, B0], [A1, B1]
#[inline(always)]
unsafe fn double_fft2_interleaved_32(val02: __m128, val13: __m128) -> [__m128; 2] {
    let temp0 = _mm_add_ps(val02, val13);
    let temp1 = _mm_sub_ps(val02, val13);
    [temp0, temp1]
}

// double lenth 2 fft of a and b, given as [a0, a1], [b0, b1]
// result is [A0, B0], [A1, B1]
#[inline(always)]
unsafe fn double_fft2_contiguous_32(left: __m128, right: __m128) -> [__m128; 2] {
    let temp02 = _mm_shuffle_ps(left, right, 0x44);
    let temp13 = _mm_shuffle_ps(left, right, 0xEE);
    let temp0 = _mm_add_ps(temp02, temp13);
    let temp1 = _mm_sub_ps(temp02, temp13);
    [temp0, temp1]
}




//  ____             __   _  _   _     _ _   
// |___ \           / /_ | || | | |__ (_) |_ 
//   __) |  _____  | '_ \| || |_| '_ \| | __|
//  / __/  |_____| | (_) |__   _| |_) | | |_ 
// |_____|          \___/   |_| |_.__/|_|\__|
//                                           


pub struct Sse64Butterfly2<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_butterfly!(Sse64Butterfly2, 2, |this: &Sse64Butterfly2<_>| this.direction);
impl<T: FftNum> Sse64Butterfly2<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
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
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm
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
        single_fft2_64(value0, value1)
    }
}

#[inline(always)]
unsafe fn single_fft2_64(left: __m128d, right: __m128d) -> [__m128d; 2] {
    let temp0 = _mm_add_pd(left, right);
    let temp1 = _mm_sub_pd(left, right);
    [temp0, temp1]
}





//  _  _             _________  _     _ _   
// | || |           |___ /___ \| |__ (_) |_ 
// | || |_   _____    |_ \ __) | '_ \| | __|
// |__   _| |_____|  ___) / __/| |_) | | |_ 
//    |_|           |____/_____|_.__/|_|\__|
//                                          



pub struct Sse32Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90_32,
    //rot: __m128,
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

        let temp = self.perform_fft_direct(value01, value23);

        let array = std::mem::transmute::<[__m128;2], [Complex<f32>; 4]>(temp);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;

        *output_slice.add(0) = array[0];
        *output_slice.add(1) = array[1];
        *output_slice.add(2) = array[2];
        *output_slice.add(3) = array[3];
    }

    // length 4 fft of a, given as [a0, a1], [a2, a3]
    // result is [A0, A1], [A2, A3]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value01: __m128,
        value23: __m128,
    ) -> [__m128; 2] {
        let mut temp = double_fft2_interleaved_32(value01, value23);
        temp[1] = self.rotate.rotate_2nd(temp[1]);
        double_fft2_contiguous_32(temp[0], temp[1])
    }
}


//  _  _              __   _  _   _     _ _   
// | || |            / /_ | || | | |__ (_) |_ 
// | || |_   _____  | '_ \| || |_| '_ \| | __|
// |__   _| |_____| | (_) |__   _| |_) | | |_ 
//    |_|            \___/   |_| |_.__/|_|\__|
//

pub struct Sse64Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90_64,
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
        let temp0 = single_fft2_64(value0, value2);
        let mut temp1 = single_fft2_64(value1, value3);

        temp1[1] = self.rotate.rotate(temp1[1]);

        let out0 = single_fft2_64(temp0[0], temp1[0]);
        let out2 = single_fft2_64(temp0[1], temp1[1]);
        [out0[0], out2[0], out0[1], out2[1]]
    }
    
}



//   ___            _________  _     _ _   
//  ( _ )          |___ /___ \| |__ (_) |_ 
//  / _ \   _____    |_ \ __) | '_ \| | __|
// | (_) | |_____|  ___) / __/| |_) | | |_ 
//  \___/          |____/_____|_.__/|_|\__|
//                                        


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
            _mm_set_ps(0.5f32.sqrt(), 0.5f32.sqrt(), 1.0, 1.0)
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
        let in01 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let in23 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let in45 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let in67 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);

        let out = self.perform_fft_direct([in01, in23, in45, in67]);

        let outvals = std::mem::transmute::<[__m128;4], [Complex<f32>; 8]>(out);

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
    unsafe fn perform_fft_direct(&self, values: [__m128; 4]) -> [__m128; 4] {
        // transpose
        let in02 = _mm_shuffle_ps(values[0], values[1], 0x44);
        let in13 = _mm_shuffle_ps(values[0], values[1], 0xEE);
        let in46 = _mm_shuffle_ps(values[2], values[3], 0x44);
        let in57 = _mm_shuffle_ps(values[2], values[3], 0xEE);
        
        let mut val0 = self.bf4.perform_fft_direct(in02, in46);
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
        let out0 = double_fft2_interleaved_32(val0[0], val2[0]);
        let out1 = double_fft2_interleaved_32(val0[1], val2[1]);

        [out0[0], out1[0], out0[1], out1[1]]
    }
}



//   ___             __   _  _   _     _ _   
//  ( _ )           / /_ | || | | |__ (_) |_ 
//  / _ \   _____  | '_ \| || |_| '_ \| | __|
// | (_) | |_____| | (_) |__   _| |_) | | |_ 
//  \___/           \___/   |_| |_.__/|_|\__|
//                                         



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

        let out = self.perform_fft_direct([in0, in1, in2, in3, in4, in5, in6, in7]);

        let outvals = std::mem::transmute::<[__m128d; 8], [Complex<f64>;8]>(out);

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
        let val03 = self.bf4.perform_fft_direct(values[0], values[2], values[4], values[6]);
        let mut val47 = self.bf4.perform_fft_direct(values[1], values[3], values[5], values[7]);

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
        let out0 = single_fft2_64(val03[0], val47[0]);
        let out1 = single_fft2_64(val03[1], val47[1]);
        let out2 = single_fft2_64(val03[2], val47[2]);
        let out3 = single_fft2_64(val03[3], val47[3]);
        [out0[0], out1[0], out2[0], out3[0], out0[1], out1[1], out2[1], out3[1]]
    }
}



//  _  __             _________  _     _ _   
// / |/ /_           |___ /___ \| |__ (_) |_ 
// | | '_ \   _____    |_ \ __) | '_ \| | __|
// | | (_) | |_____|  ___) / __/| |_) | | |_ 
// |_|\___/          |____/_____|_.__/|_|\__|
//                                           

pub struct Sse32Butterfly16<T> {
    root2: __m128,
    direction: FftDirection,
    bf4: Sse32Butterfly4<T>,
    bf8: Sse32Butterfly8<T>,
    rotate90: Rotate90_32,
    twiddle01: __m128,
    twiddle23: __m128,
    twiddle01conj: __m128,
    twiddle23conj: __m128,
}

boilerplate_fft_butterfly!(Sse32Butterfly16, 16, |this: &Sse32Butterfly16<_>| this.direction);
impl<T: FftNum> Sse32Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        let bf8 = Sse32Butterfly8::new(direction);
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
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 16, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 16, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 16, direction);
        let twiddle01 = unsafe {
            _mm_set_ps(tw1.im, tw1.re, 0.0, 1.0)
        };
        let twiddle23 = unsafe {
            _mm_set_ps(tw3.im, tw3.re, tw2.im, tw2.re)
        };
        let twiddle01conj = unsafe {
            _mm_set_ps(-tw1.im, tw1.re, 0.0, 1.0)
        };
        let twiddle23conj = unsafe {
            _mm_set_ps(-tw3.im, tw3.re, -tw2.im, tw2.re)
        };
        Self {
            root2,
            direction,
            bf4,
            bf8,
            rotate90,
            twiddle01,
            twiddle23,
            twiddle01conj,
            twiddle23conj,
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

        //let mut scratch_evens = [
        //    input.load(0),
        //    input.load(2),
        //    input.load(4),
        //    input.load(6),
        //    input.load(8),
        //    input.load(10),
        //    input.load(12),
        //    input.load(14),
        //];

        //let mut scratch_odds_n1 = [input.load(1), input.load(5), input.load(9), input.load(13)];
        //let mut scratch_odds_n3 = [input.load(15), input.load(3), input.load(7), input.load(11)];

        let in0 = _mm_loadu_ps(input.as_ptr() as *const f32);
        let in2 = _mm_loadu_ps(input.as_ptr().add(2) as *const f32);
        let in4 = _mm_loadu_ps(input.as_ptr().add(4) as *const f32);
        let in6 = _mm_loadu_ps(input.as_ptr().add(6) as *const f32);
        let in8 = _mm_loadu_ps(input.as_ptr().add(8) as *const f32);
        let in10 = _mm_loadu_ps(input.as_ptr().add(10) as *const f32);
        let in12 = _mm_loadu_ps(input.as_ptr().add(12) as *const f32);
        let in14 = _mm_loadu_ps(input.as_ptr().add(14) as *const f32);

        let in0002 = pack_1st_32(in0, in2);
        let in0406 = pack_1st_32(in4, in6);
        let in0810 = pack_1st_32(in8, in10);
        let in1214 = pack_1st_32(in12, in14);

        let in0105 = pack_2nd_32(in0, in4);
        let in0913 = pack_2nd_32(in8, in12);
        let in1503 = pack_2nd_32(in14, in2);
        let in0711 = pack_2nd_32(in6, in10);



        let in_evens = [in0002, in0406, in0810, in1214];

        //println!("evens {:?}", in_evens);
        //println!("odds1 {:?}", [in0105, in0913]);
        //println!("odds1 {:?}", [in1503, in0711]);

        // step 2: column FFTs
        let evens = self.bf8.perform_fft_direct(in_evens);
        let mut odds1 = self.bf4.perform_fft_direct(in0105, in0913);
        let mut odds3 = self.bf4.perform_fft_direct(in1503, in0711);

        //println!("evens fft {:?}", evens);
        //println!("odds fft {:?} {:?} ", odds1, odds3);

        //println!("scr0 fft {:?} {:?} {:?} {:?}", val0, val1, val2, val3);
        //println!("scr1 fft {:?} {:?} {:?} {:?}", val4, val5, val6, val7);

        // step 3: apply twiddle factors
        //odds1[1] = complex_mul_64(odds1[1], self.twiddle1);
        //odds3[1] = complex_mul_64(odds3[1], self.twiddle1c);

        //odds1[2] = complex_mul_64(odds1[2], self.twiddle2);
        //odds3[2] = complex_mul_64(odds3[2], self.twiddle2c);

        //odds1[3] = complex_mul_64(odds1[3], self.twiddle3);
        //odds3[3] = complex_mul_64(odds3[3], self.twiddle3c);

        odds1[0] = complex_double_mul_32(odds1[0], self.twiddle01);
        odds3[0] = complex_double_mul_32(odds3[0], self.twiddle01conj);

        odds1[1] = complex_double_mul_32(odds1[1], self.twiddle23);
        odds3[1] = complex_double_mul_32(odds3[1], self.twiddle23conj);

        //println!("odds fft tw {:?} {:?} ", odds1, odds3);

        //println!("scr1 rot {:?} {:?} {:?} {:?}", val4, val5, val6, val7);

        // step 4: cross FFTs
        //Butterfly2::perform_fft_strided(&mut scratch_odds_n1[0], &mut scratch_odds_n3[0]);
        //Butterfly2::perform_fft_strided(&mut scratch_odds_n1[1], &mut scratch_odds_n3[1]);
        //Butterfly2::perform_fft_strided(&mut scratch_odds_n1[2], &mut scratch_odds_n3[2]);
        //Butterfly2::perform_fft_strided(&mut scratch_odds_n1[3], &mut scratch_odds_n3[3]);

        //let mut temp0 = single_fft2_64(odds1[0], odds3[0]);
        //let mut temp1 = single_fft2_64(odds1[1], odds3[1]);
        //let mut temp2 = single_fft2_64(odds1[2], odds3[2]);
        //let mut temp3 = single_fft2_64(odds1[3], odds3[3]);

        let mut temp0 = double_fft2_interleaved_32(odds1[0], odds3[0]);
        let mut temp1 = double_fft2_interleaved_32(odds1[1], odds3[1]);

        //println!("odds fft2 {:?} {:?} ",  temp0, temp1);
        //temp0[1] = self.rotate90.rotate(temp0[1]);
        //temp1[1] = self.rotate90.rotate(temp1[1]);
        //temp2[1] = self.rotate90.rotate(temp2[1]);
        //temp3[1] = self.rotate90.rotate(temp3[1]);

        temp0[1] = self.rotate90.rotate_both(temp0[1]);
        temp1[1] = self.rotate90.rotate_both(temp1[1]);

        //println!("odds fft2 rot {:?} {:?} ", temp0, temp1);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        //scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.fft_direction());
        //scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.fft_direction());
        //scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.fft_direction());
        //scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.fft_direction());

        //println!("scr0 fft2 {:?} {:?} {:?} {:?}", out0, out1, out2, out3);
        //println!("scr1 fft2 {:?} {:?} {:?} {:?}", out4, out5, out6, out7);

        //step 5: copy/add/subtract data back to buffer
        //output.store(scratch_evens[0] + scratch_odds_n1[0], 0);
        //output.store(scratch_evens[1] + scratch_odds_n1[1], 1);
        //output.store(scratch_evens[2] + scratch_odds_n1[2], 2);
        //output.store(scratch_evens[3] + scratch_odds_n1[3], 3);
        //output.store(scratch_evens[4] + scratch_odds_n3[0], 4);
        //output.store(scratch_evens[5] + scratch_odds_n3[1], 5);
        //output.store(scratch_evens[6] + scratch_odds_n3[2], 6);
        //output.store(scratch_evens[7] + scratch_odds_n3[3], 7);
        //output.store(scratch_evens[0] - scratch_odds_n1[0], 8);
        //output.store(scratch_evens[1] - scratch_odds_n1[1], 9);
        //output.store(scratch_evens[2] - scratch_odds_n1[2], 10);
        //output.store(scratch_evens[3] - scratch_odds_n1[3], 11);
        //output.store(scratch_evens[4] - scratch_odds_n3[0], 12);
        //output.store(scratch_evens[5] - scratch_odds_n3[1], 13);
        //output.store(scratch_evens[6] - scratch_odds_n3[2], 14);
        //output.store(scratch_evens[7] - scratch_odds_n3[3], 15);

        //let out0  = _mm_add_pd(evens[0], temp0[0]);
        //let out1  = _mm_add_pd(evens[1], temp1[0]);
        //let out2  = _mm_add_pd(evens[2], temp2[0]);
        //let out3  = _mm_add_pd(evens[3], temp3[0]);
        //let out4  = _mm_add_pd(evens[4], temp0[1]);
        //let out5  = _mm_add_pd(evens[5], temp1[1]);
        //let out6  = _mm_add_pd(evens[6], temp2[1]);
        //let out7  = _mm_add_pd(evens[7], temp3[1]);
        //let out8  = _mm_sub_pd(evens[0], temp0[0]);
        //let out9  = _mm_sub_pd(evens[1], temp1[0]);
        //let out10 = _mm_sub_pd(evens[2], temp2[0]);
        //let out11 = _mm_sub_pd(evens[3], temp3[0]);
        //let out12 = _mm_sub_pd(evens[4], temp0[1]);
        //let out13 = _mm_sub_pd(evens[5], temp1[1]);
        //let out14 = _mm_sub_pd(evens[6], temp2[1]);
        //let out15 = _mm_sub_pd(evens[7], temp3[1]);

        //println!("adds {:?}, {:?}, {:?}, {:?}", temp0[0],temp0[1],temp1[0],temp1[1]);

        let out0 = _mm_add_ps(evens[0], temp0[0]);
        let out1 = _mm_add_ps(evens[1], temp1[0]);
        let out2 = _mm_add_ps(evens[2], temp0[1]);
        let out3 = _mm_add_ps(evens[3], temp1[1]);
        let out4 = _mm_sub_ps(evens[0], temp0[0]);
        let out5 = _mm_sub_ps(evens[1], temp1[0]);
        let out6 = _mm_sub_ps(evens[2], temp0[1]);
        let out7 = _mm_sub_ps(evens[3], temp1[1]);



        let val0  = std::mem::transmute::<__m128, [Complex<f32>;2]>(out0);
        let val1  = std::mem::transmute::<__m128, [Complex<f32>;2]>(out1);
        let val2  = std::mem::transmute::<__m128, [Complex<f32>;2]>(out2);
        let val3  = std::mem::transmute::<__m128, [Complex<f32>;2]>(out3);
        let val4  = std::mem::transmute::<__m128, [Complex<f32>;2]>(out4);
        let val5  = std::mem::transmute::<__m128, [Complex<f32>;2]>(out5);
        let val6  = std::mem::transmute::<__m128, [Complex<f32>;2]>(out6);
        let val7  = std::mem::transmute::<__m128, [Complex<f32>;2]>(out7);

        let output_slice = output.as_mut_ptr() as *mut Complex<f32>;
        *output_slice.add(0) = val0[0];
        *output_slice.add(1) = val0[1];
        *output_slice.add(2) = val1[0];
        *output_slice.add(3) = val1[1];
        *output_slice.add(4) = val2[0];
        *output_slice.add(5) = val2[1];
        *output_slice.add(6) = val3[0];
        *output_slice.add(7) = val3[1];
        *output_slice.add(8) = val4[0];
        *output_slice.add(9) = val4[1];
        *output_slice.add(10) = val5[0];
        *output_slice.add(11) = val5[1];
        *output_slice.add(12) = val6[0];
        *output_slice.add(13) = val6[1];
        *output_slice.add(14) = val7[0];
        *output_slice.add(15) = val7[1];
    }
}



//  _  __              __   _  _   _     _ _   
// / |/ /_            / /_ | || | | |__ (_) |_ 
// | | '_ \   _____  | '_ \| || |_| '_ \| | __|
// | | (_) | |_____| | (_) |__   _| |_) | | |_ 
// |_|\___/           \___/   |_| |_.__/|_|\__|
//                                             


pub struct Sse64Butterfly16<T> {
    root2: __m128d,
    direction: FftDirection,
    bf4: Sse64Butterfly4<T>,
    bf8: Sse64Butterfly8<T>,
    rotate90: Rotate90_64,
    twiddle1: __m128d,
    twiddle2: __m128d,
    twiddle3: __m128d,
    twiddle1c: __m128d,
    twiddle2c: __m128d,
    twiddle3c: __m128d,
}

boilerplate_fft_butterfly!(Sse64Butterfly16, 16, |this: &Sse64Butterfly16<_>| this.direction);
impl<T: FftNum> Sse64Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        let bf8 = Sse64Butterfly8::new(direction);
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
        let twiddle1 = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(1, 16, direction).re as *const f64)
        };
        let twiddle2 = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(2, 16, direction).re as *const f64)
        };
        let twiddle3 = unsafe {
            _mm_loadu_pd(&twiddles::compute_twiddle(3, 16, direction).re as *const f64)
        };
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
            root2,
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
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        //let mut scratch_evens = [
        //    input.load(0),
        //    input.load(2),
        //    input.load(4),
        //    input.load(6),
        //    input.load(8),
        //    input.load(10),
        //    input.load(12),
        //    input.load(14),
        //];

        //let mut scratch_odds_n1 = [input.load(1), input.load(5), input.load(9), input.load(13)];
        //let mut scratch_odds_n3 = [input.load(15), input.load(3), input.load(7), input.load(11)];

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

        //println!("evens: {:?}",[in0, in2, in4, in6, in8, in10, in12, in14]);
        //println!("odds1: {:?}",[in1, in5, in9, in13]);
        //println!("odds3: {:?}",[in15, in3, in7, in11]);

        // step 2: column FFTs
        let evens = self.bf8.perform_fft_direct([in0, in2, in4, in6, in8, in10, in12, in14]);
        let mut odds1 = self.bf4.perform_fft_direct(in1, in5, in9, in13);
        let mut odds3 = self.bf4.perform_fft_direct(in15, in3, in7, in11);

        //println!("evens fft {:?}", evens);
        //println!("odds fft {:?} {:?} ", odds1, odds3);

        // step 3: apply twiddle factors
        odds1[1] = complex_mul_64(odds1[1], self.twiddle1);
        odds3[1] = complex_mul_64(odds3[1], self.twiddle1c);

        odds1[2] = complex_mul_64(odds1[2], self.twiddle2);
        odds3[2] = complex_mul_64(odds3[2], self.twiddle2c);

        odds1[3] = complex_mul_64(odds1[3], self.twiddle3);
        odds3[3] = complex_mul_64(odds3[3], self.twiddle3c);

        //println!("odds fft tw {:?} {:?} ", odds1, odds3);

        //println!("scr1 rot {:?} {:?} {:?} {:?}", val4, val5, val6, val7);

        // step 4: cross FFTs
        //Butterfly2::perform_fft_strided(&mut scratch_odds_n1[0], &mut scratch_odds_n3[0]);
        //Butterfly2::perform_fft_strided(&mut scratch_odds_n1[1], &mut scratch_odds_n3[1]);
        //Butterfly2::perform_fft_strided(&mut scratch_odds_n1[2], &mut scratch_odds_n3[2]);
        //Butterfly2::perform_fft_strided(&mut scratch_odds_n1[3], &mut scratch_odds_n3[3]);

        let mut temp0 = single_fft2_64(odds1[0], odds3[0]);
        let mut temp1 = single_fft2_64(odds1[1], odds3[1]);
        let mut temp2 = single_fft2_64(odds1[2], odds3[2]);
        let mut temp3 = single_fft2_64(odds1[3], odds3[3]);

        //println!("odds fft2 {:?} {:?} {:?} {:?} ", temp0, temp1, temp2, temp3);

        temp0[1] = self.rotate90.rotate(temp0[1]);
        temp1[1] = self.rotate90.rotate(temp1[1]);
        temp2[1] = self.rotate90.rotate(temp2[1]);
        temp3[1] = self.rotate90.rotate(temp3[1]);

        //println!("odds fft2 rot {:?} {:?} {:?} {:?} ", temp0, temp1, temp2, temp3);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        //scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.fft_direction());
        //scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.fft_direction());
        //scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.fft_direction());
        //scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.fft_direction());

        //println!("scr0 fft2 {:?} {:?} {:?} {:?}", out0, out1, out2, out3);
        //println!("scr1 fft2 {:?} {:?} {:?} {:?}", out4, out5, out6, out7);

        //step 5: copy/add/subtract data back to buffer
        //output.store(scratch_evens[0] + scratch_odds_n1[0], 0);
        //output.store(scratch_evens[1] + scratch_odds_n1[1], 1);
        //output.store(scratch_evens[2] + scratch_odds_n1[2], 2);
        //output.store(scratch_evens[3] + scratch_odds_n1[3], 3);
        //output.store(scratch_evens[4] + scratch_odds_n3[0], 4);
        //output.store(scratch_evens[5] + scratch_odds_n3[1], 5);
        //output.store(scratch_evens[6] + scratch_odds_n3[2], 6);
        //output.store(scratch_evens[7] + scratch_odds_n3[3], 7);
        //output.store(scratch_evens[0] - scratch_odds_n1[0], 8);
        //output.store(scratch_evens[1] - scratch_odds_n1[1], 9);
        //output.store(scratch_evens[2] - scratch_odds_n1[2], 10);
        //output.store(scratch_evens[3] - scratch_odds_n1[3], 11);
        //output.store(scratch_evens[4] - scratch_odds_n3[0], 12);
        //output.store(scratch_evens[5] - scratch_odds_n3[1], 13);
        //output.store(scratch_evens[6] - scratch_odds_n3[2], 14);
        //output.store(scratch_evens[7] - scratch_odds_n3[3], 15);

        //println!("adds {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", temp0[0],temp1[0],temp2[0],temp3[0],temp0[1],temp1[1],temp2[1],temp3[1]);
        let out0  = _mm_add_pd(evens[0], temp0[0]);
        let out1  = _mm_add_pd(evens[1], temp1[0]);
        let out2  = _mm_add_pd(evens[2], temp2[0]);
        let out3  = _mm_add_pd(evens[3], temp3[0]);
        let out4  = _mm_add_pd(evens[4], temp0[1]);
        let out5  = _mm_add_pd(evens[5], temp1[1]);
        let out6  = _mm_add_pd(evens[6], temp2[1]);
        let out7  = _mm_add_pd(evens[7], temp3[1]);
        let out8  = _mm_sub_pd(evens[0], temp0[0]);
        let out9  = _mm_sub_pd(evens[1], temp1[0]);
        let out10 = _mm_sub_pd(evens[2], temp2[0]);
        let out11 = _mm_sub_pd(evens[3], temp3[0]);
        let out12 = _mm_sub_pd(evens[4], temp0[1]);
        let out13 = _mm_sub_pd(evens[5], temp1[1]);
        let out14 = _mm_sub_pd(evens[6], temp2[1]);
        let out15 = _mm_sub_pd(evens[7], temp3[1]);



        let val0  = std::mem::transmute::<__m128d, Complex<f64>>(out0);
        let val1  = std::mem::transmute::<__m128d, Complex<f64>>(out1);
        let val2  = std::mem::transmute::<__m128d, Complex<f64>>(out2);
        let val3  = std::mem::transmute::<__m128d, Complex<f64>>(out3);
        let val4  = std::mem::transmute::<__m128d, Complex<f64>>(out4);
        let val5  = std::mem::transmute::<__m128d, Complex<f64>>(out5);
        let val6  = std::mem::transmute::<__m128d, Complex<f64>>(out6);
        let val7  = std::mem::transmute::<__m128d, Complex<f64>>(out7);
        let val8  = std::mem::transmute::<__m128d, Complex<f64>>(out8);
        let val9  = std::mem::transmute::<__m128d, Complex<f64>>(out9);
        let val10 = std::mem::transmute::<__m128d, Complex<f64>>(out10);
        let val11 = std::mem::transmute::<__m128d, Complex<f64>>(out11);
        let val12 = std::mem::transmute::<__m128d, Complex<f64>>(out12);
        let val13 = std::mem::transmute::<__m128d, Complex<f64>>(out13);
        let val14 = std::mem::transmute::<__m128d, Complex<f64>>(out14);
        let val15 = std::mem::transmute::<__m128d, Complex<f64>>(out15);

        let output_slice = output.as_mut_ptr() as *mut Complex<f64>;
        *output_slice.add(0) = val0;
        *output_slice.add(1) = val1;
        *output_slice.add(2) = val2;
        *output_slice.add(3) = val3;
        *output_slice.add(4) = val4;
        *output_slice.add(5) = val5;
        *output_slice.add(6) = val6;
        *output_slice.add(7) = val7;
        *output_slice.add(8) = val8;
        *output_slice.add(9) = val9;
        *output_slice.add(10) = val10;
        *output_slice.add(11) = val11;
        *output_slice.add(12) = val12;
        *output_slice.add(13) = val13;
        *output_slice.add(14) = val14;
        *output_slice.add(15) = val15;
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
    test_butterfly_32_func!(test_ssef32_butterfly2, Sse32Butterfly2, 2);
    test_butterfly_32_func!(test_ssef32_butterfly4, Sse32Butterfly4, 4);
    test_butterfly_32_func!(test_ssef32_butterfly8, Sse32Butterfly8, 8);
    test_butterfly_32_func!(test_ssef32_butterfly16, Sse32Butterfly16, 16);

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
    test_butterfly_64_func!(test_ssef64_butterfly2, Sse64Butterfly2, 2);
    test_butterfly_64_func!(test_ssef64_butterfly4, Sse64Butterfly4, 4);
    test_butterfly_64_func!(test_ssef64_butterfly8, Sse64Butterfly8, 8);
    test_butterfly_64_func!(test_ssef64_butterfly16, Sse64Butterfly16, 16);

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

    #[test]
    fn check_scalar_dummy() {
        let butterfly = Sse64Butterfly16::new(FftDirection::Forward);
        let mut input = vec![Complex::new(1.0, 1.5), Complex::new(2.0, 2.4),Complex::new(7.0, 9.5),Complex::new(-4.0, -4.5),
                            Complex::new(-1.0, 5.5), Complex::new(3.3, 2.8),Complex::new(7.5, 3.5),Complex::new(-14.0, -6.5),
                            Complex::new(-7.6, 53.5), Complex::new(-4.3, 2.2),Complex::new(8.1, 1.123),Complex::new(-24.0, -16.5),
                            Complex::new(-11.0, 55.0), Complex::new(33.3, 62.8),Complex::new(17.2, 23.5),Complex::new(-54.0, -3.8)];
        let mut scratch = vec![Complex::<f64>::from(0.0); 0];
        butterfly.process_with_scratch(&mut input, &mut scratch);
        assert!(false);
    }

    #[test]
    fn check_scalar_dummy32() {
        let butterfly = Sse32Butterfly16::new(FftDirection::Forward);
        let mut input = vec![Complex::new(1.0, 1.5), Complex::new(2.0, 2.4),Complex::new(7.0, 9.5),Complex::new(-4.0, -4.5),
                        Complex::new(-1.0, 5.5), Complex::new(3.3, 2.8),Complex::new(7.5, 3.5),Complex::new(-14.0, -6.5),
                        Complex::new(-7.6, 53.5), Complex::new(-4.3, 2.2),Complex::new(8.1, 1.123),Complex::new(-24.0, -16.5),
                        Complex::new(-11.0, 55.0), Complex::new(33.3, 62.8),Complex::new(17.2, 23.5),Complex::new(-54.0, -3.8)];
        let mut scratch = vec![Complex::<f32>::from(0.0); 0];
        butterfly.process_with_scratch(&mut input, &mut scratch);
        assert!(false);
    }

    #[test]
    fn test_complex_mul_64() {
        unsafe {
            let right = _mm_set_pd(1.0, 2.0);
            let left = _mm_set_pd(5.0, 7.0);
            println!("left: {:?}", left);
            println!("right: {:?}", right);
            let res = complex_mul_64(left, right);
            println!("res: {:?}", res);
            let expected = _mm_set_pd(2.0*5.0 + 1.0*7.0, 2.0*7.0 - 1.0*5.0);
            assert_eq!(std::mem::transmute::<__m128d, Complex<f64>>(res), std::mem::transmute::<__m128d, Complex<f64>>(expected));
        }
    }

    #[test]
    fn test_complex_double_mul_32() {
        unsafe {
            let val1 = Complex::<f32>::new(1.0, 2.5);
            let val2 = Complex::<f32>::new(3.2, 4.2);
            let val3 = Complex::<f32>::new(5.6, 6.2);
            let val4 = Complex::<f32>::new(7.4, 8.3);

            let nbr2 = _mm_set_ps(val4.im, val4.re, val3.im, val3.re);
            let nbr1 = _mm_set_ps(val2.im, val2.re, val1.im, val1.re);
            println!("left: {:?}", nbr1);
            println!("right: {:?}", nbr2);
            let res = complex_double_mul_32(nbr1, nbr2);
            let res = std::mem::transmute::<__m128, [Complex<f32>;2]>(res);
            println!("res: {:?}", res);
            let expected = [val1*val3, val2*val4];
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn test_pack() {
        unsafe {
            let nbr2 = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
            let nbr1 = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
            println!("nbr1: {:?}", nbr1);
            println!("nbr2: {:?}", nbr2);
            let first = pack_1st_32(nbr1, nbr2);
            let second = pack_2nd_32(nbr1, nbr2);
            println!("first: {:?}", first);
            println!("second: {:?}", first);
            let first = std::mem::transmute::<__m128, [Complex<f32>;2]>(first);
            let second = std::mem::transmute::<__m128, [Complex<f32>;2]>(second);
            let first_expected = [Complex::new(1.0, 2.0), Complex::new(5.0, 6.0)];
            let second_expected = [Complex::new(3.0, 4.0), Complex::new(7.0, 8.0)];
            assert_eq!(first, first_expected);
            assert_eq!(second, second_expected);
        }
    }
}
