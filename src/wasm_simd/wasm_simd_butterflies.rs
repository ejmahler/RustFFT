use core::arch::wasm32::*;
use num_complex::Complex;

use crate::{common::FftNum, FftDirection};

use crate::array_utils;
use crate::array_utils::workaround_transmute_mut;
use crate::array_utils::DoubleBuf;
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::twiddles;
use crate::{Direction, Fft, Length};

use super::wasm_simd_common::{assert_f32, assert_f64};
use super::wasm_simd_utils::*;
use super::wasm_simd_vector::WasmSimdArrayMut;

#[allow(unused)]
macro_rules! boilerplate_fft_wasm_simd_f32_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            #[target_feature(enable = "simd128")]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(workaround_transmute_mut::<_, Complex<f32>>(buffer));
            }

            #[target_feature(enable = "simd128")]
            pub(crate) unsafe fn perform_parallel_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_parallel_fft_contiguous(workaround_transmute_mut::<_, Complex<f32>>(
                    buffer,
                ));
            }

            // Do multiple ffts over a longer vector inplace, called from "process_with_scratch" of Fft trait
            #[target_feature(enable = "simd128")]
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
            #[target_feature(enable = "simd128")]
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
                        let input_slice = workaround_transmute_mut(in_chunk);
                        let output_slice = workaround_transmute_mut(out_chunk);
                        self.perform_parallel_fft_contiguous(DoubleBuf {
                            input: input_slice,
                            output: output_slice,
                        })
                    },
                );
                if alldone.is_err() && input.len() >= self.len() {
                    let input_slice = workaround_transmute_mut(input);
                    let output_slice = workaround_transmute_mut(output);
                    self.perform_fft_contiguous(DoubleBuf {
                        input: &mut input_slice[len - self.len()..],
                        output: &mut output_slice[len - self.len()..],
                    })
                }
                Ok(())
            }
        }
    };
}

macro_rules! boilerplate_fft_wasm_simd_f64_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            // Do a single fft
            #[target_feature(enable = "simd128")]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(workaround_transmute_mut::<_, Complex<f64>>(buffer));
            }

            // Do multiple ffts over a longer vector inplace, called from "process_with_scratch" of Fft trait
            #[target_feature(enable = "simd128")]
            pub(crate) unsafe fn perform_fft_butterfly_multi(
                &self,
                buffer: &mut [Complex<T>],
            ) -> Result<(), ()> {
                array_utils::iter_chunks(buffer, self.len(), |chunk| {
                    self.perform_fft_butterfly(chunk)
                })
            }

            // Do multiple ffts over a longer vector outofplace, called from "process_outofplace_with_scratch" of Fft trait
            #[target_feature(enable = "simd128")]
            pub(crate) unsafe fn perform_oop_fft_butterfly_multi(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
            ) -> Result<(), ()> {
                array_utils::iter_chunks_zipped(input, output, self.len(), |in_chunk, out_chunk| {
                    let input_slice = workaround_transmute_mut(in_chunk);
                    let output_slice = workaround_transmute_mut(out_chunk);
                    self.perform_fft_contiguous(DoubleBuf {
                        input: input_slice,
                        output: output_slice,
                    })
                })
            }
        }
    };
}

#[allow(unused)]
macro_rules! boilerplate_fft_wasm_simd_common_butterfly {
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

pub struct WasmSimdF32Butterfly1<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly1,
    1,
    |this: &WasmSimdF32Butterfly1<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly1,
    1,
    |this: &WasmSimdF32Butterfly1<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly1<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let value = buffer.load_partial_lo_complex_v128(0);
        buffer.store_partial_lo_complex_v128(value, 0);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let value = buffer.load_complex_v128(0);
        buffer.store_complex_v128(value, 0);
    }
}

//   _             __   _  _   _     _ _
//  / |           / /_ | || | | |__ (_) |_
//  | |   _____  | '_ \| || |_| '_ \| | __|
//  | |  |_____| | (_) |__   _| |_) | | |_
//  |_|           \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly1<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly1,
    1,
    |this: &WasmSimdF64Butterfly1<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly1,
    1,
    |this: &WasmSimdF64Butterfly1<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly1<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let value = buffer.load_complex_v128(0);
        buffer.store_complex_v128(value, 0);
    }
}

//   ____            _________  _     _ _
//  |___ \          |___ /___ \| |__ (_) |_
//    __) |  _____    |_ \ __) | '_ \| | __|
//   / __/  |_____|  ___) / __/| |_) | | |_
//  |_____|         |____/_____|_.__/|_|\__|
//

pub struct WasmSimdF32Butterfly2<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly2,
    2,
    |this: &WasmSimdF32Butterfly2<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly2,
    2,
    |this: &WasmSimdF32Butterfly2<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly2<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let values = buffer.load_complex_v128(0);

        let temp = self.perform_fft_direct(values);

        buffer.store_complex_v128(temp, 0);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let values_a = buffer.load_complex_v128(0);
        let values_b = buffer.load_complex_v128(2);

        let out = self.perform_parallel_fft_direct(values_a, values_b);

        let [out02, out13] = transpose_complex_2x2_f32(out[0], out[1]);

        buffer.store_complex_v128(out02, 0);
        buffer.store_complex_v128(out13, 2);
    }

    // length 2 fft of x, given as [x0, x1]
    // result is [X0, X1]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: v128) -> v128 {
        solo_fft2_f32(values)
    }

    // dual length 2 fft of x and y, given as [x0, x1], [y0, y1]
    // result is [X0, Y0], [X1, Y1]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        values_x: v128,
        values_y: v128,
    ) -> [v128; 2] {
        parallel_fft2_contiguous_f32(values_x, values_y)
    }
}

// double lenth 2 fft of a and b, given as [x0, y0], [x1, y1]
// result is [X0, Y0], [X1, Y1]
#[inline(always)]
pub(crate) unsafe fn parallel_fft2_interleaved_f32(val02: v128, val13: v128) -> [v128; 2] {
    let temp0 = f32x4_add(val02, val13);
    let temp1 = f32x4_sub(val02, val13);
    [temp0, temp1]
}

// double lenth 2 fft of a and b, given as [x0, x1], [y0, y1]
// result is [X0, Y0], [X1, Y1]
#[inline(always)]
unsafe fn parallel_fft2_contiguous_f32(left: v128, right: v128) -> [v128; 2] {
    let [temp02, temp13] = transpose_complex_2x2_f32(left, right);
    parallel_fft2_interleaved_f32(temp02, temp13)
}

// length 2 fft of x, given as [x0, x1]
// result is [X0, X1]
#[inline(always)]
unsafe fn solo_fft2_f32(values: v128) -> v128 {
    let high = u64x2_shuffle::<0, 0>(values, values);
    let low = u64x2_shuffle::<1, 1>(values, values);
    let low = f32x4_mul(low, f32x4(1.0, 1.0, -1.0, -1.0));
    f32x4_add(high, low)
}

//   ____             __   _  _   _     _ _
//  |___ \           / /_ | || | | |__ (_) |_
//    __) |  _____  | '_ \| || |_| '_ \| | __|
//   / __/  |_____| | (_) |__   _| |_) | | |_
//  |_____|          \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly2<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly2,
    2,
    |this: &WasmSimdF64Butterfly2<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly2,
    2,
    |this: &WasmSimdF64Butterfly2<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly2<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let value0 = buffer.load_complex_v128(0);
        let value1 = buffer.load_complex_v128(1);

        let out = self.perform_fft_direct(value0, value1);

        buffer.store_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 1);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, value0: v128, value1: v128) -> [v128; 2] {
        solo_fft2_f64(value0, value1)
    }
}

#[inline(always)]
pub(crate) unsafe fn solo_fft2_f64(left: v128, right: v128) -> [v128; 2] {
    let temp0 = f64x2_add(left, right);
    let temp1 = f64x2_sub(left, right);
    [temp0, temp1]
}

//   _____            _________  _     _ _
//  |___ /           |___ /___ \| |__ (_) |_
//    |_ \    _____    |_ \ __) | '_ \| | __|
//   ___) |  |_____|  ___) / __/| |_) | | |_
//  |____/           |____/_____|_.__/|_|\__|
//

pub struct WasmSimdF32Butterfly3<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle: v128,
    twiddle1re: v128,
    twiddle1im: v128,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly3,
    3,
    |this: &WasmSimdF32Butterfly3<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly3,
    3,
    |this: &WasmSimdF32Butterfly3<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly3<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 3, direction);
        let twiddle = f32x4(tw1.re, tw1.re, -tw1.im, -tw1.im);
        let twiddle1re = f32x4_splat(tw1.re);
        let twiddle1im = f32x4_splat(tw1.im);
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
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let value0x = buffer.load_partial_lo_complex_v128(0);
        let value12 = buffer.load_complex_v128(1);

        let out = self.perform_fft_direct(value0x, value12);

        buffer.store_partial_lo_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 1);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let valuea0a1 = buffer.load_complex_v128(0);
        let valuea2b0 = buffer.load_complex_v128(2);
        let valueb1b2 = buffer.load_complex_v128(4);

        let value0 = extract_lo_hi_f32(valuea0a1, valuea2b0);
        let value1 = extract_hi_lo_f32(valuea0a1, valueb1b2);
        let value2 = extract_lo_hi_f32(valuea2b0, valueb1b2);

        let out = self.perform_parallel_fft_direct(value0, value1, value2);

        let out0 = extract_lo_lo_f32(out[0], out[1]);
        let out1 = extract_lo_hi_f32(out[2], out[0]);
        let out2 = extract_hi_hi_f32(out[1], out[2]);

        buffer.store_complex_v128(out0, 0);
        buffer.store_complex_v128(out1, 2);
        buffer.store_complex_v128(out2, 4);
    }

    // length 3 fft of a, given as [x0, 0.0], [x1, x2]
    // result is [X0, Z], [X1, X2]
    // The value Z should be discarded.
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, value0x: v128, value12: v128) -> [v128; 2] {
        // This is a WasmSimd translation of the scalar 3-point butterfly
        let rev12 = reverse_complex_and_negate_hi_f32(value12);
        let temp12pn = self.rotate.rotate_hi(f32x4_add(value12, rev12));
        let twiddled = f32x4_mul(temp12pn, self.twiddle);
        let temp = f32x4_add(value0x, twiddled);

        let out12 = solo_fft2_f32(temp);
        let out0x = f32x4_add(value0x, temp12pn);
        [out0x, out12]
    }

    // length 3 dual fft of a, given as (x0, y0), (x1, y1), (x2, y2).
    // result is [(X0, Y0), (X1, Y1), (X2, Y2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        value0: v128,
        value1: v128,
        value2: v128,
    ) -> [v128; 3] {
        // This is a WasmSimd translation of the scalar 3-point butterfly
        let x12p = f32x4_add(value1, value2);
        let x12n = f32x4_sub(value1, value2);
        let sum = f32x4_add(value0, x12p);

        let temp_a = f32x4_mul(self.twiddle1re, x12p);
        let temp_a = f32x4_add(temp_a, value0);

        let n_rot = self.rotate.rotate_both(x12n);
        let temp_b = f32x4_mul(self.twiddle1im, n_rot);

        let x1 = f32x4_add(temp_a, temp_b);
        let x2 = f32x4_sub(temp_a, temp_b);
        [sum, x1, x2]
    }
}

//   _____             __   _  _   _     _ _
//  |___ /            / /_ | || | | |__ (_) |_
//    |_ \    _____  | '_ \| || |_| '_ \| | __|
//   ___) |  |_____| | (_) |__   _| |_) | | |_
//  |____/            \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly3<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: v128,
    twiddle1im: v128,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly3,
    3,
    |this: &WasmSimdF64Butterfly3<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly3,
    3,
    |this: &WasmSimdF64Butterfly3<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly3<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 3, direction);
        let twiddle1re = f64x2_splat(tw1.re);
        let twiddle1im = f64x2_splat(tw1.im);

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            rotate,
            twiddle1re,
            twiddle1im,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let value0 = buffer.load_complex_v128(0);
        let value1 = buffer.load_complex_v128(1);
        let value2 = buffer.load_complex_v128(2);

        let out = self.perform_fft_direct(value0, value1, value2);

        buffer.store_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 1);
        buffer.store_complex_v128(out[2], 2);
    }

    // length 3 fft of x, given as x0, x1, x2.
    // result is [X0, X1, X2]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: v128,
        value1: v128,
        value2: v128,
    ) -> [v128; 3] {
        // This is a WasmSimd translation of the scalar 3-point butterfly
        let x12p = f64x2_add(value1, value2);
        let x12n = f64x2_sub(value1, value2);
        let sum = f64x2_add(value0, x12p);

        // let temp_a = vfmaq_f64(value0, self.twiddle1re, x12p);
        let temp_a = f64x2_add(value0, f64x2_mul(self.twiddle1re, x12p));

        let n_rot = self.rotate.rotate(x12n);
        let temp_b = f64x2_mul(self.twiddle1im, n_rot);

        let x1 = f64x2_add(temp_a, temp_b);
        let x2 = f64x2_sub(temp_a, temp_b);
        [sum, x1, x2]
    }
}

//   _  _             _________  _     _ _
//  | || |           |___ /___ \| |__ (_) |_
//  | || |_   _____    |_ \ __) | '_ \| | __|
//  |__   _| |_____|  ___) / __/| |_) | | |_
//     |_|           |____/_____|_.__/|_|\__|
//

pub struct WasmSimdF32Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly4,
    4,
    |this: &WasmSimdF32Butterfly4<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly4,
    4,
    |this: &WasmSimdF32Butterfly4<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly4<T> {
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
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let value01 = buffer.load_complex_v128(0);
        let value23 = buffer.load_complex_v128(2);

        let out = self.perform_fft_direct(value01, value23);

        buffer.store_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 2);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let value01a = buffer.load_complex_v128(0);
        let value23a = buffer.load_complex_v128(2);
        let value01b = buffer.load_complex_v128(4);
        let value23b = buffer.load_complex_v128(6);

        let [value0ab, value1ab] = transpose_complex_2x2_f32(value01a, value01b);
        let [value2ab, value3ab] = transpose_complex_2x2_f32(value23a, value23b);

        let out = self.perform_parallel_fft_direct(value0ab, value1ab, value2ab, value3ab);

        let [out0, out1] = transpose_complex_2x2_f32(out[0], out[1]);
        let [out2, out3] = transpose_complex_2x2_f32(out[2], out[3]);

        buffer.store_complex_v128(out0, 0);
        buffer.store_complex_v128(out1, 4);
        buffer.store_complex_v128(out2, 2);
        buffer.store_complex_v128(out3, 6);
    }

    // length 4 fft of a, given as [x0, x1], [x2, x3]
    // result is [[X0, X1], [X2, X3]]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, value01: v128, value23: v128) -> [v128; 2] {
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
        values0: v128,
        values1: v128,
        values2: v128,
        values3: v128,
    ) -> [v128; 4] {
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

pub struct WasmSimdF64Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly4,
    4,
    |this: &WasmSimdF64Butterfly4<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly4,
    4,
    |this: &WasmSimdF64Butterfly4<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly4<T> {
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
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let value0 = buffer.load_complex_v128(0);
        let value1 = buffer.load_complex_v128(1);
        let value2 = buffer.load_complex_v128(2);
        let value3 = buffer.load_complex_v128(3);

        let out = self.perform_fft_direct(value0, value1, value2, value3);

        buffer.store_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 1);
        buffer.store_complex_v128(out[2], 2);
        buffer.store_complex_v128(out[3], 3);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: v128,
        value1: v128,
        value2: v128,
        value3: v128,
    ) -> [v128; 4] {
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

pub struct WasmSimdF32Butterfly5<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F32,
    twiddle12re: v128,
    twiddle21re: v128,
    twiddle12im: v128,
    twiddle21im: v128,
    twiddle1re: v128,
    twiddle1im: v128,
    twiddle2re: v128,
    twiddle2im: v128,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly5,
    5,
    |this: &WasmSimdF32Butterfly5<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly5,
    5,
    |this: &WasmSimdF32Butterfly5<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly5<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let rotate = Rotate90F32::new(true);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 5, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 5, direction);
        let twiddle12re = f32x4(tw1.re, tw1.re, tw2.re, tw2.re);
        let twiddle21re = f32x4(tw2.re, tw2.re, tw1.re, tw1.re);
        let twiddle12im = f32x4(tw1.im, tw1.im, tw2.im, tw2.im);
        let twiddle21im = f32x4(tw2.im, tw2.im, -tw1.im, -tw1.im);
        let twiddle1re = f32x4_splat(tw1.re);
        let twiddle1im = f32x4_splat(tw1.im);
        let twiddle2re = f32x4_splat(tw2.re);
        let twiddle2im = f32x4_splat(tw2.im);

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
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let value00 = buffer.load1_complex_v128(0);
        let value12 = buffer.load_complex_v128(1);
        let value34 = buffer.load_complex_v128(3);

        let out = self.perform_fft_direct(value00, value12, value34);

        buffer.store_partial_lo_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 1);
        buffer.store_complex_v128(out[2], 3);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4 ,6, 8});

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

        write_complex_to_array_strided!(out_packed, buffer, 2, {0, 1, 2, 3, 4});
    }

    // length 5 fft of a, given as [x0, x0], [x1, x2], [x3, x4].
    // result is [[X0, Z], [X1, X2], [X3, X4]]
    // Note that Z should not be used.
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value00: v128,
        value12: v128,
        value34: v128,
    ) -> [v128; 3] {
        // This is a WasmSimd translation of the scalar 5-point butterfly
        let temp43 = reverse_complex_elements_f32(value34);
        let x1423p = f32x4_add(value12, temp43);
        let x1423n = f32x4_sub(value12, temp43);

        let x1414p = duplicate_lo_f32(x1423p);
        let x2323p = duplicate_hi_f32(x1423p);
        let x1414n = duplicate_lo_f32(x1423n);
        let x2323n = duplicate_hi_f32(x1423n);

        let temp_a1 = f32x4_mul(self.twiddle12re, x1414p);
        let temp_b1 = f32x4_mul(self.twiddle12im, x1414n);

        let temp_a = f32x4_add(temp_a1, f32x4_mul(self.twiddle21re, x2323p));
        let temp_a = f32x4_add(value00, temp_a);
        let temp_b = f32x4_add(temp_b1, f32x4_mul(self.twiddle21im, x2323n));

        let b_rot = self.rotate.rotate_both(temp_b);

        let x00 = f32x4_add(value00, f32x4_add(x1414p, x2323p));

        let x12 = f32x4_add(temp_a, b_rot);
        let x34 = reverse_complex_elements_f32(f32x4_sub(temp_a, b_rot));
        [x00, x12, x34]
    }

    // length 5 dual fft of x and y, given as (x0, y0), (x1, y1) ... (x4, y4).
    // result is [(X0, Y0), (X1, Y1) ... (X2, Y2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        value0: v128,
        value1: v128,
        value2: v128,
        value3: v128,
        value4: v128,
    ) -> [v128; 5] {
        // This is a WasmSimd translation of the scalar 3-point butterfly
        let x14p = f32x4_add(value1, value4);
        let x14n = f32x4_sub(value1, value4);
        let x23p = f32x4_add(value2, value3);
        let x23n = f32x4_sub(value2, value3);

        let temp_a1_1 = f32x4_mul(self.twiddle1re, x14p);
        let temp_a1_2 = f32x4_mul(self.twiddle2re, x23p);
        let temp_b1_1 = f32x4_mul(self.twiddle1im, x14n);
        let temp_b1_2 = f32x4_mul(self.twiddle2im, x23n);
        let temp_a2_1 = f32x4_mul(self.twiddle1re, x23p);
        let temp_a2_2 = f32x4_mul(self.twiddle2re, x14p);
        let temp_b2_1 = f32x4_mul(self.twiddle2im, x14n);
        let temp_b2_2 = f32x4_mul(self.twiddle1im, x23n);

        let temp_a1 = f32x4_add(value0, f32x4_add(temp_a1_1, temp_a1_2));
        let temp_b1 = f32x4_add(temp_b1_1, temp_b1_2);
        let temp_a2 = f32x4_add(value0, f32x4_add(temp_a2_1, temp_a2_2));
        let temp_b2 = f32x4_sub(temp_b2_1, temp_b2_2);

        [
            f32x4_add(value0, f32x4_add(x14p, x23p)),
            f32x4_add(temp_a1, self.rotate.rotate_both(temp_b1)),
            f32x4_add(temp_a2, self.rotate.rotate_both(temp_b2)),
            f32x4_sub(temp_a2, self.rotate.rotate_both(temp_b2)),
            f32x4_sub(temp_a1, self.rotate.rotate_both(temp_b1)),
        ]
    }
}

//   ____              __   _  _   _     _ _
//  | ___|            / /_ | || | | |__ (_) |_
//  |___ \    _____  | '_ \| || |_| '_ \| | __|
//   ___) |  |_____| | (_) |__   _| |_) | | |_
//  |____/            \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly5<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    rotate: Rotate90F64,
    twiddle1re: v128,
    twiddle1im: v128,
    twiddle2re: v128,
    twiddle2im: v128,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly5,
    5,
    |this: &WasmSimdF64Butterfly5<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly5,
    5,
    |this: &WasmSimdF64Butterfly5<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly5<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let rotate = Rotate90F64::new(true);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 5, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 5, direction);
        let twiddle1re = f64x2_splat(tw1.re);
        let twiddle1im = f64x2_splat(tw1.im);
        let twiddle2re = f64x2_splat(tw2.re);
        let twiddle2im = f64x2_splat(tw2.im);

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
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let value0 = buffer.load_complex_v128(0);
        let value1 = buffer.load_complex_v128(1);
        let value2 = buffer.load_complex_v128(2);
        let value3 = buffer.load_complex_v128(3);
        let value4 = buffer.load_complex_v128(4);

        let out = self.perform_fft_direct(value0, value1, value2, value3, value4);

        buffer.store_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 1);
        buffer.store_complex_v128(out[2], 2);
        buffer.store_complex_v128(out[3], 3);
        buffer.store_complex_v128(out[4], 4);
    }

    // length 5 fft of x, given as x0, x1, x2, x3, x4.
    // result is [X0, X1, X2, X3, X4]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: v128,
        value1: v128,
        value2: v128,
        value3: v128,
        value4: v128,
    ) -> [v128; 5] {
        // This is a WasmSimd translation of the scalar 5-point butterfly
        let x14p = f64x2_add(value1, value4);
        let x14n = f64x2_sub(value1, value4);
        let x23p = f64x2_add(value2, value3);
        let x23n = f64x2_sub(value2, value3);

        let temp_a1_1 = f64x2_mul(self.twiddle1re, x14p);
        let temp_a1_2 = f64x2_mul(self.twiddle2re, x23p);
        let temp_a2_1 = f64x2_mul(self.twiddle2re, x14p);
        let temp_a2_2 = f64x2_mul(self.twiddle1re, x23p);

        let temp_b1_1 = f64x2_mul(self.twiddle1im, x14n);
        let temp_b1_2 = f64x2_mul(self.twiddle2im, x23n);
        let temp_b2_1 = f64x2_mul(self.twiddle2im, x14n);
        let temp_b2_2 = f64x2_mul(self.twiddle1im, x23n);

        let temp_a1 = f64x2_add(value0, f64x2_add(temp_a1_1, temp_a1_2));
        let temp_a2 = f64x2_add(value0, f64x2_add(temp_a2_1, temp_a2_2));

        let temp_b1 = f64x2_add(temp_b1_1, temp_b1_2);
        let temp_b2 = f64x2_sub(temp_b2_1, temp_b2_2);

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        [
            f64x2_add(value0, f64x2_add(x14p, x23p)),
            f64x2_add(temp_a1, temp_b1_rot),
            f64x2_add(temp_a2, temp_b2_rot),
            f64x2_sub(temp_a2, temp_b2_rot),
            f64x2_sub(temp_a1, temp_b1_rot),
        ]
    }
}

//    __             _________  _     _ _
//   / /_           |___ /___ \| |__ (_) |_
//  | '_ \   _____    |_ \ __) | '_ \| | __|
//  | (_) | |_____|  ___) / __/| |_) | | |_
//   \___/          |____/_____|_.__/|_|\__|
//

pub struct WasmSimdF32Butterfly6<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF32Butterfly3<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly6,
    6,
    |this: &WasmSimdF32Butterfly6<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly6,
    6,
    |this: &WasmSimdF32Butterfly6<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = WasmSimdF32Butterfly3::new(direction);

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let value01 = buffer.load_complex_v128(0);
        let value23 = buffer.load_complex_v128(2);
        let value45 = buffer.load_complex_v128(4);

        let out = self.perform_fft_direct(value01, value23, value45);

        buffer.store_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 2);
        buffer.store_complex_v128(out[2], 4);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed = read_complex_to_array!(buffer,  {0, 2, 4, 6, 8, 10});

        let values = interleave_complex_f32!(input_packed, 3, {0, 1, 2});

        let out = self.perform_parallel_fft_direct(
            values[0], values[1], values[2], values[3], values[4], values[5],
        );

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4});
        write_complex_to_array_strided!(out_sorted, buffer, 2, {0, 1, 2, 3, 4, 5});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value01: v128,
        value23: v128,
        value45: v128,
    ) -> [v128; 3] {
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
        value0: v128,
        value1: v128,
        value2: v128,
        value3: v128,
        value4: v128,
        value5: v128,
    ) -> [v128; 6] {
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

pub struct WasmSimdF64Butterfly6<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF64Butterfly3<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly6,
    6,
    |this: &WasmSimdF64Butterfly6<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly6,
    6,
    |this: &WasmSimdF64Butterfly6<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = WasmSimdF64Butterfly3::new(direction);

        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let value0 = buffer.load_complex_v128(0);
        let value1 = buffer.load_complex_v128(1);
        let value2 = buffer.load_complex_v128(2);
        let value3 = buffer.load_complex_v128(3);
        let value4 = buffer.load_complex_v128(4);
        let value5 = buffer.load_complex_v128(5);

        let out = self.perform_fft_direct(value0, value1, value2, value3, value4, value5);

        buffer.store_complex_v128(out[0], 0);
        buffer.store_complex_v128(out[1], 1);
        buffer.store_complex_v128(out[2], 2);
        buffer.store_complex_v128(out[3], 3);
        buffer.store_complex_v128(out[4], 4);
        buffer.store_complex_v128(out[5], 5);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: v128,
        value1: v128,
        value2: v128,
        value3: v128,
        value4: v128,
        value5: v128,
    ) -> [v128; 6] {
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

pub struct WasmSimdF32Butterfly8<T> {
    root2: v128,
    root2_dual: v128,
    direction: FftDirection,
    bf4: WasmSimdF32Butterfly4<T>,
    rotate90: Rotate90F32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly8,
    8,
    |this: &WasmSimdF32Butterfly8<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly8,
    8,
    |this: &WasmSimdF32Butterfly8<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf4 = WasmSimdF32Butterfly4::new(direction);
        let root2 = f32x4(1.0, 1.0, 0.5f32.sqrt(), 0.5f32.sqrt());
        let root2_dual = f32x4_splat(0.5f32.sqrt());
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
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6});

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, buffer, 2, {0,1,2,3});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14});

        let values = interleave_complex_f32!(input_packed, 4, {0, 1, 2, 3});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6});

        write_complex_to_array_strided!(out_sorted, buffer, 2, {0,1,2,3,4,5,6,7});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, values: [v128; 4]) -> [v128; 4] {
        // we're going to hardcode a step of mixed radix
        // step 1: copy and reorder the input into the scratch
        let [in02, in13] = transpose_complex_2x2_f32(values[0], values[1]);
        let [in46, in57] = transpose_complex_2x2_f32(values[2], values[3]);

        // step 2: column FFTs
        let val0 = self.bf4.perform_fft_direct(in02, in46);
        let mut val2 = self.bf4.perform_fft_direct(in13, in57);

        // step 3: apply twiddle factors
        let val2b = self.rotate90.rotate_hi(val2[0]);
        let val2c = f32x4_add(val2b, val2[0]);
        let val2d = f32x4_mul(val2c, self.root2);
        val2[0] = extract_lo_hi_f32(val2[0], val2d);

        let val3b = self.rotate90.rotate_both(val2[1]);
        let val3c = f32x4_sub(val3b, val2[1]);
        let val3d = f32x4_mul(val3c, self.root2);
        val2[1] = extract_lo_hi_f32(val3b, val3d);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        let out0 = parallel_fft2_interleaved_f32(val0[0], val2[0]);
        let out1 = parallel_fft2_interleaved_f32(val0[1], val2[1]);

        // step 6: rearrange and copy to buffer
        [out0[0], out1[0], out0[1], out1[1]]
    }

    #[inline(always)]
    unsafe fn perform_parallel_fft_direct(&self, values: [v128; 8]) -> [v128; 8] {
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
        let val5c = f32x4_add(val5b, val47[1]);
        val47[1] = f32x4_mul(val5c, self.root2_dual);
        val47[2] = self.rotate90.rotate_both(val47[2]);
        let val7b = self.rotate90.rotate_both(val47[3]);
        let val7c = f32x4_sub(val7b, val47[3]);
        val47[3] = f32x4_mul(val7c, self.root2_dual);

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

pub struct WasmSimdF64Butterfly8<T> {
    root2: v128,
    direction: FftDirection,
    bf4: WasmSimdF64Butterfly4<T>,
    rotate90: Rotate90F64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly8,
    8,
    |this: &WasmSimdF64Butterfly8<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly8,
    8,
    |this: &WasmSimdF64Butterfly8<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf4 = WasmSimdF64Butterfly4::new(direction);
        let root2 = f64x2_splat(0.5f64.sqrt());
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
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, values: [v128; 8]) -> [v128; 8] {
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
        let val5c = f64x2_add(val5b, val47[1]);
        val47[1] = f64x2_mul(val5c, self.root2);
        val47[2] = self.rotate90.rotate(val47[2]);
        let val7b = self.rotate90.rotate(val47[3]);
        let val7c = f64x2_sub(val7b, val47[3]);
        val47[3] = f64x2_mul(val7c, self.root2);

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
pub struct WasmSimdF32Butterfly9<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF32Butterfly3<T>,
    twiddle1: v128,
    twiddle2: v128,
    twiddle4: v128,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly9,
    9,
    |this: &WasmSimdF32Butterfly9<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly9,
    9,
    |this: &WasmSimdF32Butterfly9<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly9<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = WasmSimdF32Butterfly3::new(direction);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 9, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 9, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 9, direction);
        let twiddle1 = f32x4(tw1.re, tw1.im, tw1.re, tw1.im);
        let twiddle2 = f32x4(tw2.re, tw2.im, tw2.re, tw2.im);
        let twiddle4 = f32x4(tw4.re, tw4.im, tw4.re, tw4.im);

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
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        // A single WasmSimd 9-point will need a lot of shuffling, let's just reuse the dual one
        let values = read_partial1_complex_to_array!(buffer, {0,1,2,3,4,5,6,7,8});

        let out = self.perform_parallel_fft_direct(values);

        for n in 0..9 {
            buffer.store_partial_lo_complex_v128(out[n], n);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16});

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

        write_complex_to_array_strided!(out_packed, buffer, 2, {0,1,2,3,4,5,6,7,8});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [v128; 9]) -> [v128; 9] {
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

pub struct WasmSimdF64Butterfly9<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF64Butterfly3<T>,
    twiddle1: v128,
    twiddle2: v128,
    twiddle4: v128,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly9,
    9,
    |this: &WasmSimdF64Butterfly9<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly9,
    9,
    |this: &WasmSimdF64Butterfly9<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly9<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = WasmSimdF64Butterfly3::new(direction);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 9, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 9, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 9, direction);
        let twiddle1 = f64x2(tw1.re, tw1.im);
        let twiddle2 = f64x2(tw2.re, tw2.im);
        let twiddle4 = f64x2(tw4.re, tw4.im);

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
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [v128; 9]) -> [v128; 9] {
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

pub struct WasmSimdF32Butterfly10<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf5: WasmSimdF32Butterfly5<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly10,
    10,
    |this: &WasmSimdF32Butterfly10<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly10,
    10,
    |this: &WasmSimdF32Butterfly10<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly10<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf5 = WasmSimdF32Butterfly5::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf5,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8});

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, buffer, 2, {0,1,2,3,4});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18});

        let values = interleave_complex_f32!(input_packed, 5, {0, 1, 2, 3, 4});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8});

        write_complex_to_array_strided!(out_sorted, buffer, 2, {0,1,2,3,4,5,6,7,8,9});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [v128; 5]) -> [v128; 5] {
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
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [v128; 10]) -> [v128; 10] {
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

pub struct WasmSimdF64Butterfly10<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf2: WasmSimdF64Butterfly2<T>,
    bf5: WasmSimdF64Butterfly5<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly10,
    10,
    |this: &WasmSimdF64Butterfly10<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly10,
    10,
    |this: &WasmSimdF64Butterfly10<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly10<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf2 = WasmSimdF64Butterfly2::new(direction);
        let bf5 = WasmSimdF64Butterfly5::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf2,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [v128; 10]) -> [v128; 10] {
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

pub struct WasmSimdF32Butterfly12<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF32Butterfly3<T>,
    bf4: WasmSimdF32Butterfly4<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly12,
    12,
    |this: &WasmSimdF32Butterfly12<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly12,
    12,
    |this: &WasmSimdF32Butterfly12<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly12<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = WasmSimdF32Butterfly3::new(direction);
        let bf4 = WasmSimdF32Butterfly4::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            bf4,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10 });

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, buffer, 2, {0,1,2,3,4,5});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed =
            read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});

        let values = interleave_complex_f32!(input_packed, 6, {0, 1, 2, 3, 4, 5});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8, 10});

        write_complex_to_array_strided!(out_sorted, buffer, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [v128; 6]) -> [v128; 6] {
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
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [v128; 12]) -> [v128; 12] {
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

pub struct WasmSimdF64Butterfly12<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF64Butterfly3<T>,
    bf4: WasmSimdF64Butterfly4<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly12,
    12,
    |this: &WasmSimdF64Butterfly12<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly12,
    12,
    |this: &WasmSimdF64Butterfly12<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly12<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = WasmSimdF64Butterfly3::new(direction);
        let bf4 = WasmSimdF64Butterfly4::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            bf4,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [v128; 12]) -> [v128; 12] {
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
pub struct WasmSimdF32Butterfly15<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF32Butterfly3<T>,
    bf5: WasmSimdF32Butterfly5<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly15,
    15,
    |this: &WasmSimdF32Butterfly15<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly15,
    15,
    |this: &WasmSimdF32Butterfly15<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly15<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = WasmSimdF32Butterfly3::new(direction);
        let bf5 = WasmSimdF32Butterfly5::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        // A single WasmSimd 15-point will need a lot of shuffling, let's just reuse the dual one
        let values = read_partial1_complex_to_array!(buffer, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14});

        let out = self.perform_parallel_fft_direct(values);

        for n in 0..15 {
            buffer.store_partial_lo_complex_v128(out[n], n);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed =
            read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28});

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

        write_complex_to_array_strided!(out_packed, buffer, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11, 12, 13, 14});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [v128; 15]) -> [v128; 15] {
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

pub struct WasmSimdF64Butterfly15<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF64Butterfly3<T>,
    bf5: WasmSimdF64Butterfly5<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly15,
    15,
    |this: &WasmSimdF64Butterfly15<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly15,
    15,
    |this: &WasmSimdF64Butterfly15<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly15<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = WasmSimdF64Butterfly3::new(direction);
        let bf5 = WasmSimdF64Butterfly5::new(direction);
        Self {
            direction,
            _phantom: std::marker::PhantomData,
            bf3,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let values =
            read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [v128; 15]) -> [v128; 15] {
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

pub struct WasmSimdF32Butterfly16<T> {
    direction: FftDirection,
    bf4: WasmSimdF32Butterfly4<T>,
    bf8: WasmSimdF32Butterfly8<T>,
    rotate90: Rotate90F32,
    twiddle01: v128,
    twiddle23: v128,
    twiddle01conj: v128,
    twiddle23conj: v128,
    twiddle1: v128,
    twiddle2: v128,
    twiddle3: v128,
    twiddle1c: v128,
    twiddle2c: v128,
    twiddle3c: v128,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly16,
    16,
    |this: &WasmSimdF32Butterfly16<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly16,
    16,
    |this: &WasmSimdF32Butterfly16<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf8 = WasmSimdF32Butterfly8::new(direction);
        let bf4 = WasmSimdF32Butterfly4::new(direction);
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F32::new(true)
        } else {
            Rotate90F32::new(false)
        };
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 16, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 16, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 16, direction);
        let twiddle01 = f32x4(1.0, 0.0, tw1.re, tw1.im);
        let twiddle23 = f32x4(tw2.re, tw2.im, tw3.re, tw3.im);
        let twiddle01conj = f32x4(1.0, 0.0, tw1.re, -tw1.im);
        let twiddle23conj = f32x4(tw2.re, -tw2.im, tw3.re, -tw3.im);
        let twiddle1 = f32x4(tw1.re, tw1.im, tw1.re, tw1.im);
        let twiddle2 = f32x4(tw2.re, tw2.im, tw2.re, tw2.im);
        let twiddle3 = f32x4(tw3.re, tw3.im, tw3.re, tw3.im);
        let twiddle1c = f32x4(tw1.re, -tw1.im, tw1.re, -tw1.im);
        let twiddle2c = f32x4(tw2.re, -tw2.im, tw2.re, -tw2.im);
        let twiddle3c = f32x4(tw3.re, -tw3.im, tw3.re, -tw3.im);
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
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14 });

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, buffer, 2, {0,1,2,3,4,5,6,7});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30});

        let values = interleave_complex_f32!(input_packed, 8, {0, 1, 2, 3 ,4 ,5 ,6 ,7});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8, 10, 12, 14});

        write_complex_to_array_strided!(out_sorted, buffer, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11,12,13,14, 15});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [v128; 8]) -> [v128; 8] {
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
            f32x4_add(evens[0], temp0[0]),
            f32x4_add(evens[1], temp1[0]),
            f32x4_add(evens[2], temp0[1]),
            f32x4_add(evens[3], temp1[1]),
            f32x4_sub(evens[0], temp0[0]),
            f32x4_sub(evens[1], temp1[0]),
            f32x4_sub(evens[2], temp0[1]),
            f32x4_sub(evens[3], temp1[1]),
        ]
    }

    #[inline(always)]
    unsafe fn perform_parallel_fft_direct(&self, input: [v128; 16]) -> [v128; 16] {
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
            f32x4_add(evens[0], temp0[0]),
            f32x4_add(evens[1], temp1[0]),
            f32x4_add(evens[2], temp2[0]),
            f32x4_add(evens[3], temp3[0]),
            f32x4_add(evens[4], temp0[1]),
            f32x4_add(evens[5], temp1[1]),
            f32x4_add(evens[6], temp2[1]),
            f32x4_add(evens[7], temp3[1]),
            f32x4_sub(evens[0], temp0[0]),
            f32x4_sub(evens[1], temp1[0]),
            f32x4_sub(evens[2], temp2[0]),
            f32x4_sub(evens[3], temp3[0]),
            f32x4_sub(evens[4], temp0[1]),
            f32x4_sub(evens[5], temp1[1]),
            f32x4_sub(evens[6], temp2[1]),
            f32x4_sub(evens[7], temp3[1]),
        ]
    }
}

//   _  __              __   _  _   _     _ _
//  / |/ /_            / /_ | || | | |__ (_) |_
//  | | '_ \   _____  | '_ \| || |_| '_ \| | __|
//  | | (_) | |_____| | (_) |__   _| |_) | | |_
//  |_|\___/           \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly16<T> {
    direction: FftDirection,
    bf4: WasmSimdF64Butterfly4<T>,
    bf8: WasmSimdF64Butterfly8<T>,
    rotate90: Rotate90F64,
    twiddle1: v128,
    twiddle2: v128,
    twiddle3: v128,
    twiddle1c: v128,
    twiddle2c: v128,
    twiddle3c: v128,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly16,
    16,
    |this: &WasmSimdF64Butterfly16<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly16,
    16,
    |this: &WasmSimdF64Butterfly16<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf8 = WasmSimdF64Butterfly8::new(direction);
        let bf4 = WasmSimdF64Butterfly4::new(direction);
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F64::new(true)
        } else {
            Rotate90F64::new(false)
        };
        let twiddle1 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(1, 16, direction) as *const _ as *const v128,
            )
        };
        let twiddle2 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(2, 16, direction) as *const _ as *const v128,
            )
        };
        let twiddle3 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(3, 16, direction) as *const _ as *const v128,
            )
        };
        let twiddle1c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(1, 16, direction).conj() as *const _
                    as *const v128,
            )
        };
        let twiddle2c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(2, 16, direction).conj() as *const _
                    as *const v128,
            )
        };
        let twiddle3c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(3, 16, direction).conj() as *const _
                    as *const v128,
            )
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
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let values =
            read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [v128; 16]) -> [v128; 16] {
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
            f64x2_add(evens[0], temp0[0]),
            f64x2_add(evens[1], temp1[0]),
            f64x2_add(evens[2], temp2[0]),
            f64x2_add(evens[3], temp3[0]),
            f64x2_add(evens[4], temp0[1]),
            f64x2_add(evens[5], temp1[1]),
            f64x2_add(evens[6], temp2[1]),
            f64x2_add(evens[7], temp3[1]),
            f64x2_sub(evens[0], temp0[0]),
            f64x2_sub(evens[1], temp1[0]),
            f64x2_sub(evens[2], temp2[0]),
            f64x2_sub(evens[3], temp3[0]),
            f64x2_sub(evens[4], temp0[1]),
            f64x2_sub(evens[5], temp1[1]),
            f64x2_sub(evens[6], temp2[1]),
            f64x2_sub(evens[7], temp3[1]),
        ]
    }
}

//   _________            _________  _     _ _
//  |___ /___ \          |___ /___ \| |__ (_) |_
//    |_ \ __) |  _____    |_ \ __) | '_ \| | __|
//   ___) / __/  |_____|  ___) / __/| |_) | | |_
//  |____/_____|         |____/_____|_.__/|_|\__|
//

pub struct WasmSimdF32Butterfly32<T> {
    direction: FftDirection,
    bf8: WasmSimdF32Butterfly8<T>,
    bf16: WasmSimdF32Butterfly16<T>,
    rotate90: Rotate90F32,
    twiddle01: v128,
    twiddle23: v128,
    twiddle45: v128,
    twiddle67: v128,
    twiddle01conj: v128,
    twiddle23conj: v128,
    twiddle45conj: v128,
    twiddle67conj: v128,
    twiddle1: v128,
    twiddle2: v128,
    twiddle3: v128,
    twiddle4: v128,
    twiddle5: v128,
    twiddle6: v128,
    twiddle7: v128,
    twiddle1c: v128,
    twiddle2c: v128,
    twiddle3c: v128,
    twiddle4c: v128,
    twiddle5c: v128,
    twiddle6c: v128,
    twiddle7c: v128,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly32,
    32,
    |this: &WasmSimdF32Butterfly32<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF32Butterfly32,
    32,
    |this: &WasmSimdF32Butterfly32<_>| this.direction
);
impl<T: FftNum> WasmSimdF32Butterfly32<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf8 = WasmSimdF32Butterfly8::new(direction);
        let bf16 = WasmSimdF32Butterfly16::new(direction);
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
        let twiddle01 = f32x4(1.0, 0.0, tw1.re, tw1.im);
        let twiddle23 = f32x4(tw2.re, tw2.im, tw3.re, tw3.im);
        let twiddle45 = f32x4(tw4.re, tw4.im, tw5.re, tw5.im);
        let twiddle67 = f32x4(tw6.re, tw6.im, tw7.re, tw7.im);
        let twiddle01conj = f32x4(1.0, 0.0, tw1.re, -tw1.im);
        let twiddle23conj = f32x4(tw2.re, -tw2.im, tw3.re, -tw3.im);
        let twiddle45conj = f32x4(tw4.re, -tw4.im, tw5.re, -tw5.im);
        let twiddle67conj = f32x4(tw6.re, -tw6.im, tw7.re, -tw7.im);
        let twiddle1 = f32x4(tw1.re, tw1.im, tw1.re, tw1.im);
        let twiddle2 = f32x4(tw2.re, tw2.im, tw2.re, tw2.im);
        let twiddle3 = f32x4(tw3.re, tw3.im, tw3.re, tw3.im);
        let twiddle4 = f32x4(tw4.re, tw4.im, tw4.re, tw4.im);
        let twiddle5 = f32x4(tw5.re, tw5.im, tw5.re, tw5.im);
        let twiddle6 = f32x4(tw6.re, tw6.im, tw6.re, tw6.im);
        let twiddle7 = f32x4(tw7.re, tw7.im, tw7.re, tw7.im);
        let twiddle1c = f32x4(tw1.re, -tw1.im, tw1.re, -tw1.im);
        let twiddle2c = f32x4(tw2.re, -tw2.im, tw2.re, -tw2.im);
        let twiddle3c = f32x4(tw3.re, -tw3.im, tw3.re, -tw3.im);
        let twiddle4c = f32x4(tw4.re, -tw4.im, tw4.re, -tw4.im);
        let twiddle5c = f32x4(tw5.re, -tw5.im, tw5.re, -tw5.im);
        let twiddle6c = f32x4(tw6.re, -tw6.im, tw6.re, -tw6.im);
        let twiddle7c = f32x4(tw7.re, -tw7.im, tw7.re, -tw7.im);
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
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f32>) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 });

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, buffer, 2, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl WasmSimdArrayMut<f32>,
    ) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62});

        let values = interleave_complex_f32!(input_packed, 16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30});

        write_complex_to_array_strided!(out_sorted, buffer, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 });
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [v128; 16]) -> [v128; 16] {
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
            f32x4_add(evens[0], temp0[0]),
            f32x4_add(evens[1], temp1[0]),
            f32x4_add(evens[2], temp2[0]),
            f32x4_add(evens[3], temp3[0]),
            f32x4_add(evens[4], temp0[1]),
            f32x4_add(evens[5], temp1[1]),
            f32x4_add(evens[6], temp2[1]),
            f32x4_add(evens[7], temp3[1]),
            f32x4_sub(evens[0], temp0[0]),
            f32x4_sub(evens[1], temp1[0]),
            f32x4_sub(evens[2], temp2[0]),
            f32x4_sub(evens[3], temp3[0]),
            f32x4_sub(evens[4], temp0[1]),
            f32x4_sub(evens[5], temp1[1]),
            f32x4_sub(evens[6], temp2[1]),
            f32x4_sub(evens[7], temp3[1]),
        ]
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, input: [v128; 32]) -> [v128; 32] {
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
            f32x4_add(evens[0], temp0[0]),
            f32x4_add(evens[1], temp1[0]),
            f32x4_add(evens[2], temp2[0]),
            f32x4_add(evens[3], temp3[0]),
            f32x4_add(evens[4], temp4[0]),
            f32x4_add(evens[5], temp5[0]),
            f32x4_add(evens[6], temp6[0]),
            f32x4_add(evens[7], temp7[0]),
            f32x4_add(evens[8], temp0[1]),
            f32x4_add(evens[9], temp1[1]),
            f32x4_add(evens[10], temp2[1]),
            f32x4_add(evens[11], temp3[1]),
            f32x4_add(evens[12], temp4[1]),
            f32x4_add(evens[13], temp5[1]),
            f32x4_add(evens[14], temp6[1]),
            f32x4_add(evens[15], temp7[1]),
            f32x4_sub(evens[0], temp0[0]),
            f32x4_sub(evens[1], temp1[0]),
            f32x4_sub(evens[2], temp2[0]),
            f32x4_sub(evens[3], temp3[0]),
            f32x4_sub(evens[4], temp4[0]),
            f32x4_sub(evens[5], temp5[0]),
            f32x4_sub(evens[6], temp6[0]),
            f32x4_sub(evens[7], temp7[0]),
            f32x4_sub(evens[8], temp0[1]),
            f32x4_sub(evens[9], temp1[1]),
            f32x4_sub(evens[10], temp2[1]),
            f32x4_sub(evens[11], temp3[1]),
            f32x4_sub(evens[12], temp4[1]),
            f32x4_sub(evens[13], temp5[1]),
            f32x4_sub(evens[14], temp6[1]),
            f32x4_sub(evens[15], temp7[1]),
        ]
    }
}

//   _________             __   _  _   _     _ _
//  |___ /___ \           / /_ | || | | |__ (_) |_
//    |_ \ __) |  _____  | '_ \| || |_| '_ \| | __|
//   ___) / __/  |_____| | (_) |__   _| |_) | | |_
//  |____/_____|          \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly32<T> {
    direction: FftDirection,
    bf8: WasmSimdF64Butterfly8<T>,
    bf16: WasmSimdF64Butterfly16<T>,
    rotate90: Rotate90F64,
    twiddle1: v128,
    twiddle2: v128,
    twiddle3: v128,
    twiddle4: v128,
    twiddle5: v128,
    twiddle6: v128,
    twiddle7: v128,
    twiddle1c: v128,
    twiddle2c: v128,
    twiddle3c: v128,
    twiddle4c: v128,
    twiddle5c: v128,
    twiddle6c: v128,
    twiddle7c: v128,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly32,
    32,
    |this: &WasmSimdF64Butterfly32<_>| this.direction
);
boilerplate_fft_wasm_simd_common_butterfly!(
    WasmSimdF64Butterfly32,
    32,
    |this: &WasmSimdF64Butterfly32<_>| this.direction
);
impl<T: FftNum> WasmSimdF64Butterfly32<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf8 = WasmSimdF64Butterfly8::new(direction);
        let bf16 = WasmSimdF64Butterfly16::new(direction);
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F64::new(true)
        } else {
            Rotate90F64::new(false)
        };
        let twiddle1 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(1, 32, direction) as *const _ as *const v128,
            )
        };
        let twiddle2 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(2, 32, direction) as *const _ as *const v128,
            )
        };
        let twiddle3 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(3, 32, direction) as *const _ as *const v128,
            )
        };
        let twiddle4 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(4, 32, direction) as *const _ as *const v128,
            )
        };
        let twiddle5 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(5, 32, direction) as *const _ as *const v128,
            )
        };
        let twiddle6 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(6, 32, direction) as *const _ as *const v128,
            )
        };
        let twiddle7 = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(7, 32, direction) as *const _ as *const v128,
            )
        };
        let twiddle1c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(1, 32, direction).conj() as *const _
                    as *const v128,
            )
        };
        let twiddle2c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(2, 32, direction).conj() as *const _
                    as *const v128,
            )
        };
        let twiddle3c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(3, 32, direction).conj() as *const _
                    as *const v128,
            )
        };
        let twiddle4c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(4, 32, direction).conj() as *const _
                    as *const v128,
            )
        };
        let twiddle5c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(5, 32, direction).conj() as *const _
                    as *const v128,
            )
        };
        let twiddle6c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(6, 32, direction).conj() as *const _
                    as *const v128,
            )
        };
        let twiddle7c = unsafe {
            v128_load(
                &twiddles::compute_twiddle::<f64>(7, 32, direction).conj() as *const _
                    as *const v128,
            )
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
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl WasmSimdArrayMut<f64>) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, input: [v128; 32]) -> [v128; 32] {
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
            f64x2_add(evens[0], temp0[0]),
            f64x2_add(evens[1], temp1[0]),
            f64x2_add(evens[2], temp2[0]),
            f64x2_add(evens[3], temp3[0]),
            f64x2_add(evens[4], temp4[0]),
            f64x2_add(evens[5], temp5[0]),
            f64x2_add(evens[6], temp6[0]),
            f64x2_add(evens[7], temp7[0]),
            f64x2_add(evens[8], temp0[1]),
            f64x2_add(evens[9], temp1[1]),
            f64x2_add(evens[10], temp2[1]),
            f64x2_add(evens[11], temp3[1]),
            f64x2_add(evens[12], temp4[1]),
            f64x2_add(evens[13], temp5[1]),
            f64x2_add(evens[14], temp6[1]),
            f64x2_add(evens[15], temp7[1]),
            f64x2_sub(evens[0], temp0[0]),
            f64x2_sub(evens[1], temp1[0]),
            f64x2_sub(evens[2], temp2[0]),
            f64x2_sub(evens[3], temp3[0]),
            f64x2_sub(evens[4], temp4[0]),
            f64x2_sub(evens[5], temp5[0]),
            f64x2_sub(evens[6], temp6[0]),
            f64x2_sub(evens[7], temp7[0]),
            f64x2_sub(evens[8], temp0[1]),
            f64x2_sub(evens[9], temp1[1]),
            f64x2_sub(evens[10], temp2[1]),
            f64x2_sub(evens[11], temp3[1]),
            f64x2_sub(evens[12], temp4[1]),
            f64x2_sub(evens[13], temp5[1]),
            f64x2_sub(evens[14], temp6[1]),
            f64x2_sub(evens[15], temp7[1]),
        ]
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::algorithm::Dft;
    use crate::test_utils::{check_fft_algorithm, compare_vectors};
    use wasm_bindgen_test::wasm_bindgen_test;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_32_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[wasm_bindgen_test]
            fn $test_name() {
                let butterfly = $struct_name::new(FftDirection::Forward);
                check_fft_algorithm::<f32>(&butterfly, $size, FftDirection::Forward);

                let butterfly_direction = $struct_name::new(FftDirection::Inverse);
                check_fft_algorithm::<f32>(&butterfly_direction, $size, FftDirection::Inverse);
            }
        };
    }
    test_butterfly_32_func!(test_wasm_simdf32_butterfly1, WasmSimdF32Butterfly1, 1);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly2, WasmSimdF32Butterfly2, 2);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly3, WasmSimdF32Butterfly3, 3);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly4, WasmSimdF32Butterfly4, 4);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly5, WasmSimdF32Butterfly5, 5);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly6, WasmSimdF32Butterfly6, 6);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly8, WasmSimdF32Butterfly8, 8);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly9, WasmSimdF32Butterfly9, 9);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly10, WasmSimdF32Butterfly10, 10);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly12, WasmSimdF32Butterfly12, 12);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly15, WasmSimdF32Butterfly15, 15);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly16, WasmSimdF32Butterfly16, 16);
    test_butterfly_32_func!(test_wasm_simdf32_butterfly32, WasmSimdF32Butterfly32, 32);

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_64_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[wasm_bindgen_test]
            fn $test_name() {
                let butterfly = $struct_name::new(FftDirection::Forward);
                check_fft_algorithm::<f64>(&butterfly, $size, FftDirection::Forward);

                let butterfly_direction = $struct_name::new(FftDirection::Inverse);
                check_fft_algorithm::<f64>(&butterfly_direction, $size, FftDirection::Inverse);
            }
        };
    }
    test_butterfly_64_func!(test_wasm_simdf64_butterfly1, WasmSimdF64Butterfly1, 1);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly2, WasmSimdF64Butterfly2, 2);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly3, WasmSimdF64Butterfly3, 3);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly4, WasmSimdF64Butterfly4, 4);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly5, WasmSimdF64Butterfly5, 5);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly6, WasmSimdF64Butterfly6, 6);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly8, WasmSimdF64Butterfly8, 8);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly9, WasmSimdF64Butterfly9, 9);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly10, WasmSimdF64Butterfly10, 10);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly12, WasmSimdF64Butterfly12, 12);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly15, WasmSimdF64Butterfly15, 15);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly16, WasmSimdF64Butterfly16, 16);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly32, WasmSimdF64Butterfly32, 32);

    #[wasm_bindgen_test]
    fn test_solo_fft2_32() {
        unsafe {
            let val1 = Complex::<f32>::new(1.0, 2.5);
            let val2 = Complex::<f32>::new(3.2, 4.2);

            let mut val = vec![val1, val2];

            let in_packed = v128_load(val.as_ptr() as *const v128);

            let dft = Dft::new(2, FftDirection::Forward);

            let bf2 = WasmSimdF32Butterfly2::<f32>::new(FftDirection::Forward);

            dft.process(&mut val);
            let res_packed = bf2.perform_fft_direct(in_packed);

            let res = std::mem::transmute::<v128, [Complex<f32>; 2]>(res_packed);
            assert_eq!(val[0], res[0]);
            assert_eq!(val[1], res[1]);
        }
    }

    #[wasm_bindgen_test]
    fn test_parallel_fft2_32() {
        unsafe {
            let val_a1 = Complex::<f32>::new(1.0, 2.5);
            let val_a2 = Complex::<f32>::new(3.2, 4.2);

            let val_b1 = Complex::<f32>::new(6.0, 24.5);
            let val_b2 = Complex::<f32>::new(4.3, 34.2);

            let mut val_a = vec![val_a1, val_a2];
            let mut val_b = vec![val_b1, val_b2];

            let p1 = v128_load(val_a.as_ptr() as *const v128);
            let p2 = v128_load(val_b.as_ptr() as *const v128);

            let dft = Dft::new(2, FftDirection::Forward);

            let bf2 = WasmSimdF32Butterfly2::<f32>::new(FftDirection::Forward);

            dft.process(&mut val_a);
            dft.process(&mut val_b);
            let res_both = bf2.perform_parallel_fft_direct(p1, p2);

            let res = std::mem::transmute::<[v128; 2], [Complex<f32>; 4]>(res_both);
            let wasmsimd_res_a = [res[0], res[2]];
            let wasmsimd_res_b = [res[1], res[3]];
            assert!(compare_vectors(&val_a, &wasmsimd_res_a));
            assert!(compare_vectors(&val_b, &wasmsimd_res_b));
        }
    }
}
