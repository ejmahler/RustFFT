use core::arch::wasm32::*;
use num_complex::Complex;

use crate::{common::FftNum, FftDirection};

use crate::array_utils::DoubleBuf;
use crate::twiddles;
use crate::{Direction, Fft, Length};

use crate::fft_helper::{
    fft_helper_immut, fft_helper_immut_unroll2x, fft_helper_inplace, fft_helper_inplace_unroll2x,
};

use crate::simd::simd_array::SimdComplexArrayMut;
use crate::simd::simd_vector::SimdVector;

use super::wasm_simd_common::{assert_f32, assert_f64};
use super::wasm_simd_utils::*;
use super::wasm_simd_vector::{WasmVector32, WasmVector64};

#[inline(always)]
fn pack_32(a: Complex<f32>, b: Complex<f32>) -> WasmVector32 {
    WasmVector32(f32x4(a.re, a.im, b.re, b.im))
}
#[inline(always)]
fn pack_64(a: Complex<f64>) -> WasmVector64 {
    WasmVector64(f64x2(a.re, a.im))
}

#[allow(unused)]
macro_rules! boilerplate_fft_wasm_simd_f32_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> Fft<T> for $struct_name<T> {
            #[target_feature(enable = "simd128")]
            fn process_immutable_with_scratch(
                &self,
                input: &[Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                unsafe {
                    let simd_input = crate::array_utils::workaround_transmute(input);
                    let simd_output = crate::array_utils::workaround_transmute_mut(output);
                    fft_helper_immut_unroll2x(
                        simd_input,
                        simd_output,
                        self.len(),
                        |input, output| {
                            self.perform_parallel_fft_contiguous(DoubleBuf { input, output })
                        },
                        |input, output| self.perform_fft_contiguous(DoubleBuf { input, output }),
                    )
                }
            }
            #[target_feature(enable = "simd128")]
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                self.process_immutable_with_scratch(input, output, scratch);
            }
            #[target_feature(enable = "simd128")]
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                unsafe {
                    let simd_buffer = crate::array_utils::workaround_transmute_mut(buffer);
                    fft_helper_inplace_unroll2x(
                        simd_buffer,
                        self.len(),
                        |chunk| self.perform_parallel_fft_contiguous(chunk),
                        |chunk| self.perform_fft_contiguous(chunk),
                    )
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
            #[inline(always)]
            fn get_immutable_scratch_len(&self) -> usize {
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

macro_rules! boilerplate_fft_wasm_simd_f64_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> Fft<T> for $struct_name<T> {
            #[target_feature(enable = "simd128")]
            fn process_immutable_with_scratch(
                &self,
                input: &[Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                unsafe {
                    let simd_input = crate::array_utils::workaround_transmute(input);
                    let simd_output = crate::array_utils::workaround_transmute_mut(output);
                    fft_helper_immut(
                        simd_input,
                        simd_output,
                        &mut [],
                        self.len(),
                        0,
                        |input, output, _| self.perform_fft_contiguous(DoubleBuf { input, output }),
                    )
                }
            }
            #[target_feature(enable = "simd128")]
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                self.process_immutable_with_scratch(input, output, scratch);
            }
            #[target_feature(enable = "simd128")]
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                unsafe {
                    let simd_buffer = crate::array_utils::workaround_transmute_mut(buffer);
                    fft_helper_inplace(simd_buffer, &mut [], self.len(), 0, |chunk, _| {
                        self.perform_fft_contiguous(chunk)
                    })
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
            #[inline(always)]
            fn get_immutable_scratch_len(&self) -> usize {
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
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let value = buffer.load1_lo(0);
        buffer.store1_lo(value, 0);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let value = buffer.load(0);
        buffer.store(value, 0);
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
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let value = buffer.load(0);
        buffer.store(value, 0);
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
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let values = buffer.load(0);

        let temp = self.perform_fft_direct(values);

        buffer.store(temp, 0);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let values_a = buffer.load(0);
        let values_b = buffer.load(2);

        let out = self.perform_parallel_fft_direct(values_a, values_b);

        let [out02, out13] = transpose_complex_2x2_f32(out[0], out[1]);

        buffer.store(out02, 0);
        buffer.store(out13, 2);
    }

    // length 2 fft of x, given as [x0, x1]
    // result is [X0, X1]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: WasmVector32) -> WasmVector32 {
        solo_fft2_f32(values)
    }

    // dual length 2 fft of x and y, given as [x0, x1], [y0, y1]
    // result is [X0, Y0], [X1, Y1]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        values_x: WasmVector32,
        values_y: WasmVector32,
    ) -> [WasmVector32; 2] {
        parallel_fft2_contiguous_f32(values_x, values_y)
    }
}

#[inline(always)]
pub unsafe fn wasm_column_butterfly2<V: SimdVector>(rows: [V; 2]) -> [V; 2] {
    [V::add(rows[0], rows[1]), V::sub(rows[0], rows[1])]
}

// double lenth 2 fft of a and b, given as [x0, x1], [y0, y1]
// result is [X0, Y0], [X1, Y1]
#[inline(always)]
unsafe fn parallel_fft2_contiguous_f32(
    left: WasmVector32,
    right: WasmVector32,
) -> [WasmVector32; 2] {
    let [temp02, temp13] = transpose_complex_2x2_f32(left, right);
    wasm_column_butterfly2([temp02, temp13])
}

// length 2 fft of x, given as [x0, x1]
// result is [X0, X1]
#[inline(always)]
unsafe fn solo_fft2_f32(values: WasmVector32) -> WasmVector32 {
    let high = WasmVector32(u64x2_shuffle::<0, 0>(values.0, values.0));
    let low = WasmVector32(u64x2_shuffle::<1, 1>(values.0, values.0));
    let low = SimdVector::mul(low, WasmVector32(f32x4(1.0, 1.0, -1.0, -1.0)));
    SimdVector::add(high, low)
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
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let value0 = buffer.load(0);
        let value1 = buffer.load(1);

        let out = self.perform_fft_direct(value0, value1);

        buffer.store(out[0], 0);
        buffer.store(out[1], 1);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: WasmVector64,
        value1: WasmVector64,
    ) -> [WasmVector64; 2] {
        solo_fft2_f64(value0, value1)
    }
}

#[inline(always)]
pub(crate) unsafe fn solo_fft2_f64(left: WasmVector64, right: WasmVector64) -> [WasmVector64; 2] {
    let temp0 = SimdVector::add(left, right);
    let temp1 = SimdVector::sub(left, right);
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
    twiddle: WasmVector32,
    twiddle1re: WasmVector32,
    twiddle1im: WasmVector32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
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
        let twiddle = WasmVector32(f32x4(tw1.re, tw1.re, -tw1.im, -tw1.im));
        let twiddle1re = unsafe { SimdVector::broadcast_scalar(tw1.re) };
        let twiddle1im = unsafe { SimdVector::broadcast_scalar(tw1.im) };
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
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let value0x = buffer.load1_lo(0);
        let value12 = buffer.load(1);

        let out = self.perform_fft_direct(value0x, value12);

        buffer.store1_lo(out[0], 0);
        buffer.store(out[1], 1);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let valuea0a1 = buffer.load(0);
        let valuea2b0 = buffer.load(2);
        let valueb1b2 = buffer.load(4);

        let value0 = extract_lo_hi_f32(valuea0a1, valuea2b0);
        let value1 = extract_hi_lo_f32(valuea0a1, valueb1b2);
        let value2 = extract_lo_hi_f32(valuea2b0, valueb1b2);

        let out = self.perform_parallel_fft_direct(value0, value1, value2);

        let out0 = extract_lo_lo_f32(out[0], out[1]);
        let out1 = extract_lo_hi_f32(out[2], out[0]);
        let out2 = extract_hi_hi_f32(out[1], out[2]);

        buffer.store(out0, 0);
        buffer.store(out1, 2);
        buffer.store(out2, 4);
    }

    // length 3 fft of a, given as [x0, 0.0], [x1, x2]
    // result is [X0, Z], [X1, X2]
    // The value Z should be discarded.
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0x: WasmVector32,
        value12: WasmVector32,
    ) -> [WasmVector32; 2] {
        // This is a WasmSimd translation of the scalar 3-point butterfly
        let rev12 = reverse_complex_and_negate_hi_f32(value12);
        let temp12pn = self.rotate.rotate_hi(SimdVector::add(value12, rev12));
        let twiddled = SimdVector::mul(temp12pn, self.twiddle);
        let temp = SimdVector::add(value0x, twiddled);

        let out12 = solo_fft2_f32(temp);
        let out0x = SimdVector::add(value0x, temp12pn);
        [out0x, out12]
    }

    // length 3 dual fft of a, given as (x0, y0), (x1, y1), (x2, y2).
    // result is [(X0, Y0), (X1, Y1), (X2, Y2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        value0: WasmVector32,
        value1: WasmVector32,
        value2: WasmVector32,
    ) -> [WasmVector32; 3] {
        // This is a WasmSimd translation of the scalar 3-point butterfly
        let x12p = SimdVector::add(value1, value2);
        let x12n = SimdVector::sub(value1, value2);
        let sum = SimdVector::add(value0, x12p);

        let temp_a = SimdVector::mul(self.twiddle1re, x12p);
        let temp_a = SimdVector::add(temp_a, value0);

        let n_rot = self.rotate.rotate_both(x12n);
        let temp_b = SimdVector::mul(self.twiddle1im, n_rot);

        let x1 = SimdVector::add(temp_a, temp_b);
        let x2 = SimdVector::sub(temp_a, temp_b);
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
    twiddle1re: WasmVector64,
    twiddle1im: WasmVector64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
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
        let twiddle1re = unsafe { SimdVector::broadcast_scalar(tw1.re) };
        let twiddle1im = unsafe { SimdVector::broadcast_scalar(tw1.im) };

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
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let value0 = buffer.load(0);
        let value1 = buffer.load(1);
        let value2 = buffer.load(2);

        let out = self.perform_fft_direct(value0, value1, value2);

        buffer.store(out[0], 0);
        buffer.store(out[1], 1);
        buffer.store(out[2], 2);
    }

    // length 3 fft of x, given as x0, x1, x2.
    // result is [X0, X1, X2]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: WasmVector64,
        value1: WasmVector64,
        value2: WasmVector64,
    ) -> [WasmVector64; 3] {
        // This is a WasmSimd translation of the scalar 3-point butterfly
        let x12p = SimdVector::add(value1, value2);
        let x12n = SimdVector::sub(value1, value2);
        let sum = SimdVector::add(value0, x12p);

        // let temp_a = vfmaq_f64(value0, self.twiddle1re, x12p);
        let temp_a = SimdVector::add(value0, SimdVector::mul(self.twiddle1re, x12p));

        let n_rot = self.rotate.rotate(x12n);
        let temp_b = SimdVector::mul(self.twiddle1im, n_rot);

        let x1 = SimdVector::add(temp_a, temp_b);
        let x2 = SimdVector::sub(temp_a, temp_b);
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
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let value01 = buffer.load(0);
        let value23 = buffer.load(2);

        let out = self.perform_fft_direct(value01, value23);

        buffer.store(out[0], 0);
        buffer.store(out[1], 2);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let value01a = buffer.load(0);
        let value23a = buffer.load(2);
        let value01b = buffer.load(4);
        let value23b = buffer.load(6);

        let [value0ab, value1ab] = transpose_complex_2x2_f32(value01a, value01b);
        let [value2ab, value3ab] = transpose_complex_2x2_f32(value23a, value23b);

        let out = self.perform_parallel_fft_direct([value0ab, value1ab, value2ab, value3ab]);

        let [out0, out1] = transpose_complex_2x2_f32(out[0], out[1]);
        let [out2, out3] = transpose_complex_2x2_f32(out[2], out[3]);

        buffer.store(out0, 0);
        buffer.store(out1, 4);
        buffer.store(out2, 2);
        buffer.store(out3, 6);
    }

    // length 4 fft of a, given as [x0, x1], [x2, x3]
    // result is [[X0, X1], [X2, X3]]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value01: WasmVector32,
        value23: WasmVector32,
    ) -> [WasmVector32; 2] {
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose
        // and
        // step 2: column FFTs
        let mut temp = wasm_column_butterfly2([value01, value23]);

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
        values: [WasmVector32; 4],
    ) -> [WasmVector32; 4] {
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose
        // and
        // step 2: column FFTs
        let temp0 = wasm_column_butterfly2([values[0], values[2]]);
        let mut temp1 = wasm_column_butterfly2([values[1], values[3]]);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        temp1[1] = self.rotate.rotate_both(temp1[1]);

        // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

        // step 5: row FFTs
        let out0 = wasm_column_butterfly2([temp0[0], temp1[0]]);
        let out2 = wasm_column_butterfly2([temp0[1], temp1[1]]);

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
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let value0 = buffer.load(0);
        let value1 = buffer.load(1);
        let value2 = buffer.load(2);
        let value3 = buffer.load(3);

        let out = self.perform_fft_direct([value0, value1, value2, value3]);

        buffer.store(out[0], 0);
        buffer.store(out[1], 1);
        buffer.store(out[2], 2);
        buffer.store(out[3], 3);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [WasmVector64; 4]) -> [WasmVector64; 4] {
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose
        // and
        // step 2: column FFTs
        let temp0 = solo_fft2_f64(values[0], values[2]);
        let mut temp1 = solo_fft2_f64(values[1], values[3]);

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

#[inline(always)]
pub unsafe fn wasm_column_butterfly4<V: SimdVector>(rows: [V; 4], rotation: V::Rotation) -> [V; 4] {
    // Algorithm: 2x2 mixed radix

    // Perform the first set of size-2 FFTs.
    let [mid0, mid2] = wasm_column_butterfly2([rows[0], rows[2]]);
    let [mid1, mid3] = wasm_column_butterfly2([rows[1], rows[3]]);

    // Apply twiddle factors (in this case just a rotation)
    let mid3_rotated = V::apply_rotate90(rotation, mid3);

    // Transpose the data and do size-2 FFTs down the columns
    let [output0, output1] = wasm_column_butterfly2([mid0, mid1]);
    let [output2, output3] = wasm_column_butterfly2([mid2, mid3_rotated]);

    // Swap outputs 1 and 2 in the output to do a square transpose
    [output0, output2, output1, output3]
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
    twiddle12re: WasmVector32,
    twiddle21re: WasmVector32,
    twiddle12im: WasmVector32,
    twiddle21im: WasmVector32,
    twiddle1re: WasmVector32,
    twiddle1im: WasmVector32,
    twiddle2re: WasmVector32,
    twiddle2im: WasmVector32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
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
        let twiddle12re = WasmVector32(f32x4(tw1.re, tw1.re, tw2.re, tw2.re));
        let twiddle21re = WasmVector32(f32x4(tw2.re, tw2.re, tw1.re, tw1.re));
        let twiddle12im = WasmVector32(f32x4(tw1.im, tw1.im, tw2.im, tw2.im));
        let twiddle21im = WasmVector32(f32x4(tw2.im, tw2.im, -tw1.im, -tw1.im));
        let twiddle1re = WasmVector32(f32x4_splat(tw1.re));
        let twiddle1im = WasmVector32(f32x4_splat(tw1.im));
        let twiddle2re = WasmVector32(f32x4_splat(tw2.re));
        let twiddle2im = WasmVector32(f32x4_splat(tw2.im));

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
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let value00 = buffer.load1_lo_broadcast(0);
        let value12 = buffer.load(1);
        let value34 = buffer.load(3);

        let out = self.perform_fft_direct(value00, value12, value34);

        buffer.store1_lo(out[0], 0);
        buffer.store(out[1], 1);
        buffer.store(out[2], 3);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
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
        value00: WasmVector32,
        value12: WasmVector32,
        value34: WasmVector32,
    ) -> [WasmVector32; 3] {
        // This is a WasmSimd translation of the scalar 5-point butterfly
        let temp43 = reverse_complex_elements_f32(value34);
        let x1423p = SimdVector::add(value12, temp43);
        let x1423n = SimdVector::sub(value12, temp43);

        let x1414p = duplicate_lo_f32(x1423p);
        let x2323p = duplicate_hi_f32(x1423p);
        let x1414n = duplicate_lo_f32(x1423n);
        let x2323n = duplicate_hi_f32(x1423n);

        let temp_a1 = SimdVector::mul(self.twiddle12re, x1414p);
        let temp_b1 = SimdVector::mul(self.twiddle12im, x1414n);

        let temp_a = SimdVector::add(temp_a1, SimdVector::mul(self.twiddle21re, x2323p));
        let temp_a = SimdVector::add(value00, temp_a);
        let temp_b = SimdVector::add(temp_b1, SimdVector::mul(self.twiddle21im, x2323n));

        let b_rot = self.rotate.rotate_both(temp_b);

        let x00 = SimdVector::add(value00, SimdVector::add(x1414p, x2323p));

        let x12 = SimdVector::add(temp_a, b_rot);
        let x34 = reverse_complex_elements_f32(SimdVector::sub(temp_a, b_rot));
        [x00, x12, x34]
    }

    // length 5 dual fft of x and y, given as (x0, y0), (x1, y1) ... (x4, y4).
    // result is [(X0, Y0), (X1, Y1) ... (X2, Y2)]
    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        value0: WasmVector32,
        value1: WasmVector32,
        value2: WasmVector32,
        value3: WasmVector32,
        value4: WasmVector32,
    ) -> [WasmVector32; 5] {
        // This is a WasmSimd translation of the scalar 3-point butterfly
        let x14p = SimdVector::add(value1, value4);
        let x14n = SimdVector::sub(value1, value4);
        let x23p = SimdVector::add(value2, value3);
        let x23n = SimdVector::sub(value2, value3);

        let temp_a1_1 = SimdVector::mul(self.twiddle1re, x14p);
        let temp_a1_2 = SimdVector::mul(self.twiddle2re, x23p);
        let temp_b1_1 = SimdVector::mul(self.twiddle1im, x14n);
        let temp_b1_2 = SimdVector::mul(self.twiddle2im, x23n);
        let temp_a2_1 = SimdVector::mul(self.twiddle1re, x23p);
        let temp_a2_2 = SimdVector::mul(self.twiddle2re, x14p);
        let temp_b2_1 = SimdVector::mul(self.twiddle2im, x14n);
        let temp_b2_2 = SimdVector::mul(self.twiddle1im, x23n);

        let temp_a1 = SimdVector::add(value0, SimdVector::add(temp_a1_1, temp_a1_2));
        let temp_b1 = SimdVector::add(temp_b1_1, temp_b1_2);
        let temp_a2 = SimdVector::add(value0, SimdVector::add(temp_a2_1, temp_a2_2));
        let temp_b2 = SimdVector::sub(temp_b2_1, temp_b2_2);

        [
            SimdVector::add(value0, SimdVector::add(x14p, x23p)),
            SimdVector::add(temp_a1, self.rotate.rotate_both(temp_b1)),
            SimdVector::add(temp_a2, self.rotate.rotate_both(temp_b2)),
            SimdVector::sub(temp_a2, self.rotate.rotate_both(temp_b2)),
            SimdVector::sub(temp_a1, self.rotate.rotate_both(temp_b1)),
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
    twiddle1re: WasmVector64,
    twiddle1im: WasmVector64,
    twiddle2re: WasmVector64,
    twiddle2im: WasmVector64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
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
        let twiddle1re = unsafe { SimdVector::broadcast_scalar(tw1.re) };
        let twiddle1im = unsafe { SimdVector::broadcast_scalar(tw1.im) };
        let twiddle2re = unsafe { SimdVector::broadcast_scalar(tw2.re) };
        let twiddle2im = unsafe { SimdVector::broadcast_scalar(tw2.im) };

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
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let value0 = buffer.load(0);
        let value1 = buffer.load(1);
        let value2 = buffer.load(2);
        let value3 = buffer.load(3);
        let value4 = buffer.load(4);

        let out = self.perform_fft_direct(value0, value1, value2, value3, value4);

        buffer.store(out[0], 0);
        buffer.store(out[1], 1);
        buffer.store(out[2], 2);
        buffer.store(out[3], 3);
        buffer.store(out[4], 4);
    }

    // length 5 fft of x, given as x0, x1, x2, x3, x4.
    // result is [X0, X1, X2, X3, X4]
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        value0: WasmVector64,
        value1: WasmVector64,
        value2: WasmVector64,
        value3: WasmVector64,
        value4: WasmVector64,
    ) -> [WasmVector64; 5] {
        // This is a WasmSimd translation of the scalar 5-point butterfly
        let x14p = SimdVector::add(value1, value4);
        let x14n = SimdVector::sub(value1, value4);
        let x23p = SimdVector::add(value2, value3);
        let x23n = SimdVector::sub(value2, value3);

        let temp_a1_1 = SimdVector::mul(self.twiddle1re, x14p);
        let temp_a1_2 = SimdVector::mul(self.twiddle2re, x23p);
        let temp_a2_1 = SimdVector::mul(self.twiddle2re, x14p);
        let temp_a2_2 = SimdVector::mul(self.twiddle1re, x23p);

        let temp_b1_1 = SimdVector::mul(self.twiddle1im, x14n);
        let temp_b1_2 = SimdVector::mul(self.twiddle2im, x23n);
        let temp_b2_1 = SimdVector::mul(self.twiddle2im, x14n);
        let temp_b2_2 = SimdVector::mul(self.twiddle1im, x23n);

        let temp_a1 = SimdVector::add(value0, SimdVector::add(temp_a1_1, temp_a1_2));
        let temp_a2 = SimdVector::add(value0, SimdVector::add(temp_a2_1, temp_a2_2));

        let temp_b1 = SimdVector::add(temp_b1_1, temp_b1_2);
        let temp_b2 = SimdVector::sub(temp_b2_1, temp_b2_2);

        let temp_b1_rot = self.rotate.rotate(temp_b1);
        let temp_b2_rot = self.rotate.rotate(temp_b2);
        [
            SimdVector::add(value0, SimdVector::add(x14p, x23p)),
            SimdVector::add(temp_a1, temp_b1_rot),
            SimdVector::add(temp_a2, temp_b2_rot),
            SimdVector::sub(temp_a2, temp_b2_rot),
            SimdVector::sub(temp_a1, temp_b1_rot),
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
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF32Butterfly3<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly6,
    6,
    |this: &WasmSimdF32Butterfly6<_>| this.bf3.direction
);
impl<T: FftNum> WasmSimdF32Butterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = WasmSimdF32Butterfly3::new(direction);

        Self {
            _phantom: std::marker::PhantomData,
            bf3,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let value01 = buffer.load(0);
        let value23 = buffer.load(2);
        let value45 = buffer.load(4);

        let out = self.perform_fft_direct(value01, value23, value45);

        buffer.store(out[0], 0);
        buffer.store(out[1], 2);
        buffer.store(out[2], 4);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
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
        value01: WasmVector32,
        value23: WasmVector32,
        value45: WasmVector32,
    ) -> [WasmVector32; 3] {
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
        value0: WasmVector32,
        value1: WasmVector32,
        value2: WasmVector32,
        value3: WasmVector32,
        value4: WasmVector32,
        value5: WasmVector32,
    ) -> [WasmVector32; 6] {
        // Algorithm: 3x2 good-thomas

        // Size-3 FFTs down the columns of our reordered array
        let mid0 = self.bf3.perform_parallel_fft_direct(value0, value2, value4);
        let mid1 = self.bf3.perform_parallel_fft_direct(value3, value5, value1);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = wasm_column_butterfly2([mid0[0], mid1[0]]);
        let [output2, output3] = wasm_column_butterfly2([mid0[1], mid1[1]]);
        let [output4, output5] = wasm_column_butterfly2([mid0[2], mid1[2]]);

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
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF64Butterfly3<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly6,
    6,
    |this: &WasmSimdF64Butterfly6<_>| this.bf3.direction
);
impl<T: FftNum> WasmSimdF64Butterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = WasmSimdF64Butterfly3::new(direction);

        Self {
            _phantom: std::marker::PhantomData,
            bf3,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let value0 = buffer.load(0);
        let value1 = buffer.load(1);
        let value2 = buffer.load(2);
        let value3 = buffer.load(3);
        let value4 = buffer.load(4);
        let value5 = buffer.load(5);

        let out = self.perform_fft_direct([value0, value1, value2, value3, value4, value5]);

        buffer.store(out[0], 0);
        buffer.store(out[1], 1);
        buffer.store(out[2], 2);
        buffer.store(out[3], 3);
        buffer.store(out[4], 4);
        buffer.store(out[5], 5);
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [WasmVector64; 6]) -> [WasmVector64; 6] {
        // Algorithm: 3x2 good-thomas

        // Size-3 FFTs down the columns of our reordered array
        let mid0 = self.bf3.perform_fft_direct(values[0], values[2], values[4]);
        let mid1 = self.bf3.perform_fft_direct(values[3], values[5], values[1]);

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
    root2: WasmVector32,
    root2_dual: WasmVector32,
    bf4: WasmSimdF32Butterfly4<T>,
    rotate90: Rotate90F32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly8,
    8,
    |this: &WasmSimdF32Butterfly8<_>| this.bf4.direction
);
impl<T: FftNum> WasmSimdF32Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf4 = WasmSimdF32Butterfly4::new(direction);
        let root2 = WasmVector32(f32x4(1.0, 1.0, 0.5f32.sqrt(), 0.5f32.sqrt()));
        let root2_dual = WasmVector32(f32x4_splat(0.5f32.sqrt()));
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F32::new(true)
        } else {
            Rotate90F32::new(false)
        };
        Self {
            root2,
            root2_dual,
            bf4,
            rotate90,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector32>) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6});

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, buffer, 2, {0,1,2,3});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14});

        let values = interleave_complex_f32!(input_packed, 4, {0, 1, 2, 3});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6});

        write_complex_to_array_strided!(out_sorted, buffer, 2, {0,1,2,3,4,5,6,7});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, values: [WasmVector32; 4]) -> [WasmVector32; 4] {
        // we're going to hardcode a step of mixed radix
        // step 1: copy and reorder the input into the scratch
        let [in02, in13] = transpose_complex_2x2_f32(values[0], values[1]);
        let [in46, in57] = transpose_complex_2x2_f32(values[2], values[3]);

        // step 2: column FFTs
        let val0 = self.bf4.perform_fft_direct(in02, in46);
        let mut val2 = self.bf4.perform_fft_direct(in13, in57);

        // step 3: apply twiddle factors
        let val2b = self.rotate90.rotate_hi(val2[0]);
        let val2c = SimdVector::add(val2b, val2[0]);
        let val2d = SimdVector::mul(val2c, self.root2);
        val2[0] = extract_lo_hi_f32(val2[0], val2d);

        let val3b = self.rotate90.rotate_both(val2[1]);
        let val3c = SimdVector::sub(val3b, val2[1]);
        let val3d = SimdVector::mul(val3c, self.root2);
        val2[1] = extract_lo_hi_f32(val3b, val3d);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        let out0 = wasm_column_butterfly2([val0[0], val2[0]]);
        let out1 = wasm_column_butterfly2([val0[1], val2[1]]);

        // step 6: rearrange and copy to buffer
        [out0[0], out1[0], out0[1], out1[1]]
    }

    #[inline(always)]
    unsafe fn perform_parallel_fft_direct(&self, values: [WasmVector32; 8]) -> [WasmVector32; 8] {
        // we're going to hardcode a step of mixed radix
        // step 1: copy and reorder the input into the scratch
        // and
        // step 2: column FFTs
        let val03 = self
            .bf4
            .perform_parallel_fft_direct([values[0], values[2], values[4], values[6]]);
        let mut val47 = self
            .bf4
            .perform_parallel_fft_direct([values[1], values[3], values[5], values[7]]);

        // step 3: apply twiddle factors
        let val5b = self.rotate90.rotate_both(val47[1]);
        let val5c = SimdVector::add(val5b, val47[1]);
        val47[1] = SimdVector::mul(val5c, self.root2_dual);
        val47[2] = self.rotate90.rotate_both(val47[2]);
        let val7b = self.rotate90.rotate_both(val47[3]);
        let val7c = SimdVector::sub(val7b, val47[3]);
        val47[3] = SimdVector::mul(val7c, self.root2_dual);

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        let out0 = wasm_column_butterfly2([val03[0], val47[0]]);
        let out1 = wasm_column_butterfly2([val03[1], val47[1]]);
        let out2 = wasm_column_butterfly2([val03[2], val47[2]]);
        let out3 = wasm_column_butterfly2([val03[3], val47[3]]);

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
    root2: WasmVector64,
    bf4: WasmSimdF64Butterfly4<T>,
    rotate90: Rotate90F64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly8,
    8,
    |this: &WasmSimdF64Butterfly8<_>| this.bf4.direction
);
impl<T: FftNum> WasmSimdF64Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf4 = WasmSimdF64Butterfly4::new(direction);
        let root2 = WasmVector64(f64x2_splat(0.5f64.sqrt()));
        let rotate90 = if direction == FftDirection::Inverse {
            Rotate90F64::new(true)
        } else {
            Rotate90F64::new(false)
        };
        Self {
            root2,
            bf4,
            rotate90,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector64>) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7});
    }

    #[inline(always)]
    unsafe fn perform_fft_direct(&self, values: [WasmVector64; 8]) -> [WasmVector64; 8] {
        // we're going to hardcode a step of mixed radix
        // step 1: copy and reorder the input into the scratch
        // and
        // step 2: column FFTs
        let val03 = self
            .bf4
            .perform_fft_direct([values[0], values[2], values[4], values[6]]);
        let mut val47 = self
            .bf4
            .perform_fft_direct([values[1], values[3], values[5], values[7]]);

        // step 3: apply twiddle factors
        let val5b = self.rotate90.rotate(val47[1]);
        let val5c = SimdVector::add(val5b, val47[1]);
        val47[1] = SimdVector::mul(val5c, self.root2);
        val47[2] = self.rotate90.rotate(val47[2]);
        let val7b = self.rotate90.rotate(val47[3]);
        let val7c = SimdVector::sub(val7b, val47[3]);
        val47[3] = SimdVector::mul(val7c, self.root2);

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
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF32Butterfly3<T>,
    twiddle1: WasmVector32,
    twiddle2: WasmVector32,
    twiddle4: WasmVector32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly9,
    9,
    |this: &WasmSimdF32Butterfly9<_>| this.bf3.direction
);
impl<T: FftNum> WasmSimdF32Butterfly9<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = WasmSimdF32Butterfly3::new(direction);
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 9, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 9, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 9, direction);

        Self {
            _phantom: std::marker::PhantomData,
            bf3,
            twiddle1: pack_32(tw1, tw1),
            twiddle2: pack_32(tw2, tw2),
            twiddle4: pack_32(tw4, tw4),
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        // A single WasmSimd 9-point will need a lot of shuffling, let's just reuse the dual one
        let values = read_partial1_complex_to_array!(buffer, {0,1,2,3,4,5,6,7,8});

        let out = self.perform_parallel_fft_direct(values);

        for n in 0..9 {
            buffer.store1_lo(out[n], n);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
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
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        values: [WasmVector32; 9],
    ) -> [WasmVector32; 9] {
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
        mid1[1] = SimdVector::mul_complex(self.twiddle1, mid1[1]);
        mid1[2] = SimdVector::mul_complex(self.twiddle2, mid1[2]);
        mid2[1] = SimdVector::mul_complex(self.twiddle2, mid2[1]);
        mid2[2] = SimdVector::mul_complex(self.twiddle4, mid2[2]);

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
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF64Butterfly3<T>,
    twiddle1: WasmVector64,
    twiddle2: WasmVector64,
    twiddle4: WasmVector64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly9,
    9,
    |this: &WasmSimdF64Butterfly9<_>| this.bf3.direction
);
impl<T: FftNum> WasmSimdF64Butterfly9<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = WasmSimdF64Butterfly3::new(direction);
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 9, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 9, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 9, direction);

        Self {
            _phantom: std::marker::PhantomData,
            bf3,
            twiddle1: pack_64(tw1),
            twiddle2: pack_64(tw2),
            twiddle4: pack_64(tw4),
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [WasmVector64; 9]) -> [WasmVector64; 9] {
        // Algorithm: 3x3 mixed radix

        // Size-3 FFTs down the columns
        let mid0 = self.bf3.perform_fft_direct(values[0], values[3], values[6]);
        let mut mid1 = self.bf3.perform_fft_direct(values[1], values[4], values[7]);
        let mut mid2 = self.bf3.perform_fft_direct(values[2], values[5], values[8]);

        // Apply twiddle factors. Note that we're re-using twiddle2
        mid1[1] = SimdVector::mul_complex(self.twiddle1, mid1[1]);
        mid1[2] = SimdVector::mul_complex(self.twiddle2, mid1[2]);
        mid2[1] = SimdVector::mul_complex(self.twiddle2, mid2[1]);
        mid2[2] = SimdVector::mul_complex(self.twiddle4, mid2[2]);

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
    _phantom: std::marker::PhantomData<T>,
    bf5: WasmSimdF32Butterfly5<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly10,
    10,
    |this: &WasmSimdF32Butterfly10<_>| this.bf5.direction
);
impl<T: FftNum> WasmSimdF32Butterfly10<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf5 = WasmSimdF32Butterfly5::new(direction);
        Self {
            _phantom: std::marker::PhantomData,
            bf5,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector32>) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8});

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, buffer, 2, {0,1,2,3,4});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18});

        let values = interleave_complex_f32!(input_packed, 5, {0, 1, 2, 3, 4});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8});

        write_complex_to_array_strided!(out_sorted, buffer, 2, {0,1,2,3,4,5,6,7,8,9});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [WasmVector32; 5]) -> [WasmVector32; 5] {
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
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        values: [WasmVector32; 10],
    ) -> [WasmVector32; 10] {
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
        let [output0, output1] = wasm_column_butterfly2([mid0[0], mid1[0]]);
        let [output2, output3] = wasm_column_butterfly2([mid0[1], mid1[1]]);
        let [output4, output5] = wasm_column_butterfly2([mid0[2], mid1[2]]);
        let [output6, output7] = wasm_column_butterfly2([mid0[3], mid1[3]]);
        let [output8, output9] = wasm_column_butterfly2([mid0[4], mid1[4]]);

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
    _phantom: std::marker::PhantomData<T>,
    bf2: WasmSimdF64Butterfly2<T>,
    bf5: WasmSimdF64Butterfly5<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly10,
    10,
    |this: &WasmSimdF64Butterfly10<_>| this.bf5.direction
);
impl<T: FftNum> WasmSimdF64Butterfly10<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf2 = WasmSimdF64Butterfly2::new(direction);
        let bf5 = WasmSimdF64Butterfly5::new(direction);
        Self {
            _phantom: std::marker::PhantomData,
            bf2,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [WasmVector64; 10],
    ) -> [WasmVector64; 10] {
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
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF32Butterfly3<T>,
    bf4: WasmSimdF32Butterfly4<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly12,
    12,
    |this: &WasmSimdF32Butterfly12<_>| this.bf4.direction
);
impl<T: FftNum> WasmSimdF32Butterfly12<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = WasmSimdF32Butterfly3::new(direction);
        let bf4 = WasmSimdF32Butterfly4::new(direction);
        Self {
            _phantom: std::marker::PhantomData,
            bf3,
            bf4,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector32>) {
        let input_packed = read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10 });

        let out = self.perform_fft_direct(input_packed);

        write_complex_to_array_strided!(out, buffer, 2, {0,1,2,3,4,5});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        let input_packed =
            read_complex_to_array!(buffer, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});

        let values = interleave_complex_f32!(input_packed, 6, {0, 1, 2, 3, 4, 5});

        let out = self.perform_parallel_fft_direct(values);

        let out_sorted = separate_interleaved_complex_f32!(out, {0, 2, 4, 6, 8, 10});

        write_complex_to_array_strided!(out_sorted, buffer, 2, {0,1,2,3,4,5,6,7,8,9, 10, 11});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [WasmVector32; 6]) -> [WasmVector32; 6] {
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
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        values: [WasmVector32; 12],
    ) -> [WasmVector32; 12] {
        // Algorithm: 4x3 good-thomas

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = self
            .bf4
            .perform_parallel_fft_direct([values[0], values[3], values[6], values[9]]);
        let mid1 = self
            .bf4
            .perform_parallel_fft_direct([values[4], values[7], values[10], values[1]]);
        let mid2 = self
            .bf4
            .perform_parallel_fft_direct([values[8], values[11], values[2], values[5]]);

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
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF64Butterfly3<T>,
    bf4: WasmSimdF64Butterfly4<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly12,
    12,
    |this: &WasmSimdF64Butterfly12<_>| this.bf4.direction
);
impl<T: FftNum> WasmSimdF64Butterfly12<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = WasmSimdF64Butterfly3::new(direction);
        let bf4 = WasmSimdF64Butterfly4::new(direction);
        Self {
            _phantom: std::marker::PhantomData,
            bf3,
            bf4,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let values = read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [WasmVector64; 12],
    ) -> [WasmVector64; 12] {
        // Algorithm: 4x3 good-thomas

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = self
            .bf4
            .perform_fft_direct([values[0], values[3], values[6], values[9]]);
        let mid1 = self
            .bf4
            .perform_fft_direct([values[4], values[7], values[10], values[1]]);
        let mid2 = self
            .bf4
            .perform_fft_direct([values[8], values[11], values[2], values[5]]);

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
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF32Butterfly3<T>,
    bf5: WasmSimdF32Butterfly5<T>,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly15,
    15,
    |this: &WasmSimdF32Butterfly15<_>| this.bf3.direction
);
impl<T: FftNum> WasmSimdF32Butterfly15<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let bf3 = WasmSimdF32Butterfly3::new(direction);
        let bf5 = WasmSimdF32Butterfly5::new(direction);
        Self {
            _phantom: std::marker::PhantomData,
            bf3,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        // A single WasmSimd 15-point will need a lot of shuffling, let's just reuse the dual one
        let values = read_partial1_complex_to_array!(buffer, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14});

        let out = self.perform_parallel_fft_direct(values);

        for n in 0..15 {
            buffer.store1_lo(out[n], n);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
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
    pub(crate) unsafe fn perform_parallel_fft_direct(
        &self,
        values: [WasmVector32; 15],
    ) -> [WasmVector32; 15] {
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
    _phantom: std::marker::PhantomData<T>,
    bf3: WasmSimdF64Butterfly3<T>,
    bf5: WasmSimdF64Butterfly5<T>,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly15,
    15,
    |this: &WasmSimdF64Butterfly15<_>| this.bf3.direction
);
impl<T: FftNum> WasmSimdF64Butterfly15<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let bf3 = WasmSimdF64Butterfly3::new(direction);
        let bf5 = WasmSimdF64Butterfly5::new(direction);
        Self {
            _phantom: std::marker::PhantomData,
            bf3,
            bf5,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector64>,
    ) {
        let values =
            read_complex_to_array!(buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(
        &self,
        values: [WasmVector64; 15],
    ) -> [WasmVector64; 15] {
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
    bf4: WasmSimdF32Butterfly4<T>,
    twiddles_packed: [WasmVector32; 6],
    twiddle1: WasmVector32,
    twiddle3: WasmVector32,
    twiddle9: WasmVector32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly16,
    16,
    |this: &WasmSimdF32Butterfly16<_>| this.bf4.direction
);
impl<T: FftNum> WasmSimdF32Butterfly16<T> {
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let tw0: Complex<f32> = Complex { re: 1.0, im: 0.0 };
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 16, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 16, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 16, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 16, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 16, direction);
        let tw9: Complex<f32> = twiddles::compute_twiddle(9, 16, direction);

        Self {
            bf4: WasmSimdF32Butterfly4::new(direction),
            twiddles_packed: [
                pack_32(tw0, tw1),
                pack_32(tw0, tw2),
                pack_32(tw0, tw3),
                pack_32(tw2, tw3),
                pack_32(tw4, tw6),
                pack_32(tw6, tw9),
            ],
            twiddle1: pack_32(tw1, tw1),
            twiddle3: pack_32(tw3, tw3),
            twiddle9: pack_32(tw9, tw9),
        }
    }

    #[inline(always)]
    unsafe fn load_chunk(
        buffer: &impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
    ) -> [WasmVector32; 4] {
        [
            buffer.load(i),
            buffer.load(i + 4),
            buffer.load(i + 8),
            buffer.load(i + 12),
        ]
    }

    #[inline(always)]
    unsafe fn store_chunk(
        buffer: &mut impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
        vectors: [WasmVector32; 4],
    ) {
        buffer.store(vectors[0], i + 0);
        buffer.store(vectors[1], i + 4);
        buffer.store(vectors[2], i + 8);
        buffer.store(vectors[3], i + 12);
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector32>) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 4x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-4 FFTs again
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end

        // For each pair of columns: load the data, apply our size-4 FFT, apply twiddle factors, and transpose
        let mut tmp0 = self
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 0));
        tmp0[1] = SimdVector::mul_complex(tmp0[1], self.twiddles_packed[0]);
        tmp0[2] = SimdVector::mul_complex(tmp0[2], self.twiddles_packed[1]);
        tmp0[3] = SimdVector::mul_complex(tmp0[3], self.twiddles_packed[2]);
        let [mid0, mid1] = transpose_complex_2x2_f32(tmp0[0], tmp0[1]);
        let [mid4, mid5] = transpose_complex_2x2_f32(tmp0[2], tmp0[3]);

        let mut tmp1 = self
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 2));
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddles_packed[3]);
        tmp1[2] = SimdVector::mul_complex(tmp1[2], self.twiddles_packed[4]);
        tmp1[3] = SimdVector::mul_complex(tmp1[3], self.twiddles_packed[5]);
        let [mid2, mid3] = transpose_complex_2x2_f32(tmp1[0], tmp1[1]);
        let [mid6, mid7] = transpose_complex_2x2_f32(tmp1[2], tmp1[3]);

        // Size-4 FFTs down each pair of transposed columns, storing them as soon as we're done with them
        let out0 = self
            .bf4
            .perform_parallel_fft_direct([mid0, mid1, mid2, mid3]);
        Self::store_chunk(&mut buffer, 0, out0);

        let out1 = self
            .bf4
            .perform_parallel_fft_direct([mid4, mid5, mid6, mid7]);
        Self::store_chunk(&mut buffer, 2, out1);
    }

    #[inline(always)]
    unsafe fn load_parallel_chunk(
        buffer: &impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
    ) -> [[WasmVector32; 4]; 2] {
        let [a0, a1] = transpose_complex_2x2_f32(buffer.load(i + 0), buffer.load(i + 16));
        let [b0, b1] = transpose_complex_2x2_f32(buffer.load(i + 4), buffer.load(i + 20));
        let [c0, c1] = transpose_complex_2x2_f32(buffer.load(i + 8), buffer.load(i + 24));
        let [d0, d1] = transpose_complex_2x2_f32(buffer.load(i + 12), buffer.load(i + 28));
        [[a0, b0, c0, d0], [a1, b1, c1, d1]]
    }

    #[inline(always)]
    unsafe fn store_parallel_chunk(
        buffer: &mut impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
        values_a: [WasmVector32; 4],
        values_b: [WasmVector32; 4],
    ) {
        for n in 0..4 {
            let [a, b] = transpose_complex_2x2_f32(values_a[n], values_b[n]);
            buffer.store(a, i + n * 4);
            buffer.store(b, i + n * 4 + 16);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 4x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-4 FFTs again
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end

        // For each pair of columns: load the data, apply our size-4 FFT, apply twiddle factors
        let [in2, in3] = Self::load_parallel_chunk(&buffer, 2);
        let mut tmp2 = self.bf4.perform_parallel_fft_direct(in2);
        let mut tmp3 = self.bf4.perform_parallel_fft_direct(in3);
        tmp2[1] = self.bf4.rotate.rotate_both_45(tmp2[1]);
        tmp2[2] = self.bf4.rotate.rotate_both(tmp2[2]);
        tmp2[3] = self.bf4.rotate.rotate_both_135(tmp2[3]);
        tmp3[1] = SimdVector::mul_complex(tmp3[1], self.twiddle3);
        tmp3[2] = self.bf4.rotate.rotate_both_135(tmp3[2]);
        tmp3[3] = SimdVector::mul_complex(tmp3[3], self.twiddle9);

        // Do these last, because fewer twiddles means fewer temporaries forcing the above data to spill
        let [in0, in1] = Self::load_parallel_chunk(&buffer, 0);
        let tmp0 = self.bf4.perform_parallel_fft_direct(in0);
        let mut tmp1 = self.bf4.perform_parallel_fft_direct(in1);
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddle1);
        tmp1[2] = self.bf4.rotate.rotate_both_45(tmp1[2]);
        tmp1[3] = SimdVector::mul_complex(tmp1[3], self.twiddle3);

        // Size-4 FFTs down each pair of transposed columns, storing them as soon as we're done with them
        let out0 = self
            .bf4
            .perform_parallel_fft_direct([tmp0[0], tmp1[0], tmp2[0], tmp3[0]]);
        let out1 = self
            .bf4
            .perform_parallel_fft_direct([tmp0[1], tmp1[1], tmp2[1], tmp3[1]]);
        Self::store_parallel_chunk(&mut buffer, 0, out0, out1);

        let out2 = self
            .bf4
            .perform_parallel_fft_direct([tmp0[2], tmp1[2], tmp2[2], tmp3[2]]);
        let out3 = self
            .bf4
            .perform_parallel_fft_direct([tmp0[3], tmp1[3], tmp2[3], tmp3[3]]);
        Self::store_parallel_chunk(&mut buffer, 2, out2, out3);
    }
}
//   _  __              __   _  _   _     _ _
//  / |/ /_            / /_ | || | | |__ (_) |_
//  | | '_ \   _____  | '_ \| || |_| '_ \| | __|
//  | | (_) | |_____| | (_) |__   _| |_) | | |_
//  |_|\___/           \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly16<T> {
    bf4: WasmSimdF64Butterfly4<T>,
    twiddle1: WasmVector64,
    twiddle3: WasmVector64,
    twiddle9: WasmVector64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly16,
    16,
    |this: &WasmSimdF64Butterfly16<_>| this.bf4.direction
);
impl<T: FftNum> WasmSimdF64Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 16, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 16, direction);
        let tw9: Complex<f64> = twiddles::compute_twiddle(9, 16, direction);

        Self {
            bf4: WasmSimdF64Butterfly4::new(direction),
            twiddle1: pack_64(tw1),
            twiddle3: pack_64(tw3),
            twiddle9: pack_64(tw9),
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector64>) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 4x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-4 FFTs again
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end
        let load = |i| {
            [
                buffer.load(i),
                buffer.load(i + 4),
                buffer.load(i + 8),
                buffer.load(i + 12),
            ]
        };

        // For each column: load the data, apply our size-4 FFT, apply twiddle factors
        let mut tmp1 = self.bf4.perform_fft_direct(load(1));
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddle1);
        tmp1[2] = self.bf4.rotate.rotate_45(tmp1[2]);
        tmp1[3] = SimdVector::mul_complex(tmp1[3], self.twiddle3);

        let mut tmp3 = self.bf4.perform_fft_direct(load(3));
        tmp3[1] = SimdVector::mul_complex(tmp3[1], self.twiddle3);
        tmp3[2] = self.bf4.rotate.rotate_135(tmp3[2]);
        tmp3[3] = SimdVector::mul_complex(tmp3[3], self.twiddle9);

        let mut tmp2 = self.bf4.perform_fft_direct(load(2));
        tmp2[1] = self.bf4.rotate.rotate_45(tmp2[1]);
        tmp2[2] = self.bf4.rotate.rotate(tmp2[2]);
        tmp2[3] = self.bf4.rotate.rotate_135(tmp2[3]);

        // Do the first column last, because no twiddles means fewer temporaries forcing the above data to spill
        let tmp0 = self.bf4.perform_fft_direct(load(0));

        ////////////////////////////////////////////////////////////
        let mut store = |i: usize, vectors: [WasmVector64; 4]| {
            buffer.store(vectors[0], i + 0);
            buffer.store(vectors[1], i + 4);
            buffer.store(vectors[2], i + 8);
            buffer.store(vectors[3], i + 12);
        };

        // Size-4 FFTs down each of our transposed columns, storing them as soon as we're done with them
        let out0 = self
            .bf4
            .perform_fft_direct([tmp0[0], tmp1[0], tmp2[0], tmp3[0]]);
        store(0, out0);

        let out1 = self
            .bf4
            .perform_fft_direct([tmp0[1], tmp1[1], tmp2[1], tmp3[1]]);
        store(1, out1);

        let out2 = self
            .bf4
            .perform_fft_direct([tmp0[2], tmp1[2], tmp2[2], tmp3[2]]);
        store(2, out2);

        let out3 = self
            .bf4
            .perform_fft_direct([tmp0[3], tmp1[3], tmp2[3], tmp3[3]]);
        store(3, out3);
    }
}

//    ___ _  _             _________  _     _ _
//   |__ \ || |           |___ /___ \| |__ (_) |_
//    __) ||| |_   _____    |_ \ __) | '_ \| | __|
//   / __/__   _| |_____|  ___) / __/| |_) | | |_
//  |____}  |_|           |____/_____|_.__/|_|\__|
//

pub struct WasmSimdF32Butterfly24<T> {
    bf4: WasmSimdF32Butterfly4<T>,
    bf6: WasmSimdF32Butterfly6<T>,
    twiddles_packed: [WasmVector32; 9],
    twiddle1: WasmVector32,
    twiddle2: WasmVector32,
    twiddle4: WasmVector32,
    twiddle5: WasmVector32,
    twiddle8: WasmVector32,
    twiddle10: WasmVector32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly24,
    24,
    |this: &WasmSimdF32Butterfly24<_>| this.bf4.direction
);
impl<T: FftNum> WasmSimdF32Butterfly24<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let tw0: Complex<f32> = Complex { re: 1.0, im: 0.0 };
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 24, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 24, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 24, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 24, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 24, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 24, direction);
        let tw8: Complex<f32> = twiddles::compute_twiddle(8, 24, direction);
        let tw9: Complex<f32> = twiddles::compute_twiddle(9, 24, direction);
        let tw10: Complex<f32> = twiddles::compute_twiddle(10, 24, direction);
        let tw12: Complex<f32> = twiddles::compute_twiddle(12, 24, direction);
        let tw15: Complex<f32> = twiddles::compute_twiddle(15, 24, direction);
        Self {
            bf4: WasmSimdF32Butterfly4::new(direction),
            bf6: WasmSimdF32Butterfly6::new(direction),
            twiddles_packed: [
                pack_32(tw0, tw1),
                pack_32(tw0, tw2),
                pack_32(tw0, tw3),
                pack_32(tw2, tw3),
                pack_32(tw4, tw6),
                pack_32(tw6, tw9),
                pack_32(tw4, tw5),
                pack_32(tw8, tw10),
                pack_32(tw12, tw15),
            ],
            twiddle1: pack_32(tw1, tw1),
            twiddle2: pack_32(tw2, tw2),
            twiddle4: pack_32(tw4, tw4),
            twiddle5: pack_32(tw5, tw5),
            twiddle8: pack_32(tw8, tw8),
            twiddle10: pack_32(tw10, tw10),
        }
    }

    #[inline(always)]
    unsafe fn load_chunk(
        buffer: &impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
    ) -> [WasmVector32; 4] {
        [
            buffer.load(i),
            buffer.load(i + 6),
            buffer.load(i + 12),
            buffer.load(i + 18),
        ]
    }

    #[inline(always)]
    unsafe fn store_chunk(
        buffer: &mut impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
        vectors: [WasmVector32; 6],
    ) {
        buffer.store(vectors[0], i + 0);
        buffer.store(vectors[1], i + 4);
        buffer.store(vectors[2], i + 8);
        buffer.store(vectors[3], i + 12);
        buffer.store(vectors[4], i + 16);
        buffer.store(vectors[5], i + 20);
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector32>) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 6x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-6 FFTs
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end

        // For each pair of columns: load the data, apply our size-4 FFT, apply twiddle factors, transpose
        let mut tmp1 = self
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 2));
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddles_packed[3]);
        tmp1[2] = SimdVector::mul_complex(tmp1[2], self.twiddles_packed[4]);
        tmp1[3] = SimdVector::mul_complex(tmp1[3], self.twiddles_packed[5]);
        let [mid2, mid3] = transpose_complex_2x2_f32(tmp1[0], tmp1[1]);
        let [mid8, mid9] = transpose_complex_2x2_f32(tmp1[2], tmp1[3]);

        let mut tmp2 = self
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 4));
        tmp2[1] = SimdVector::mul_complex(tmp2[1], self.twiddles_packed[6]);
        tmp2[2] = SimdVector::mul_complex(tmp2[2], self.twiddles_packed[7]);
        tmp2[3] = SimdVector::mul_complex(tmp2[3], self.twiddles_packed[8]);
        let [mid4, mid5] = transpose_complex_2x2_f32(tmp2[0], tmp2[1]);
        let [mid10, mid11] = transpose_complex_2x2_f32(tmp2[2], tmp2[3]);

        let mut tmp0 = self
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 0));
        tmp0[1] = SimdVector::mul_complex(tmp0[1], self.twiddles_packed[0]);
        tmp0[2] = SimdVector::mul_complex(tmp0[2], self.twiddles_packed[1]);
        tmp0[3] = SimdVector::mul_complex(tmp0[3], self.twiddles_packed[2]);
        let [mid0, mid1] = transpose_complex_2x2_f32(tmp0[0], tmp0[1]);
        let [mid6, mid7] = transpose_complex_2x2_f32(tmp0[2], tmp0[3]);

        // Size-6 FFTs down each pair of transposed columns, storing them as soon as we're done with them
        let out0 = self
            .bf6
            .perform_parallel_fft_direct(mid0, mid1, mid2, mid3, mid4, mid5);
        Self::store_chunk(&mut buffer, 0, out0);

        let out1 = self
            .bf6
            .perform_parallel_fft_direct(mid6, mid7, mid8, mid9, mid10, mid11);
        Self::store_chunk(&mut buffer, 2, out1);
    }

    #[inline(always)]
    unsafe fn load_parallel_chunk(
        buffer: &impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
    ) -> [[WasmVector32; 4]; 2] {
        let [a0, a1] = transpose_complex_2x2_f32(buffer.load(i + 0), buffer.load(i + 24));
        let [b0, b1] = transpose_complex_2x2_f32(buffer.load(i + 6), buffer.load(i + 30));
        let [c0, c1] = transpose_complex_2x2_f32(buffer.load(i + 12), buffer.load(i + 36));
        let [d0, d1] = transpose_complex_2x2_f32(buffer.load(i + 18), buffer.load(i + 42));
        [[a0, b0, c0, d0], [a1, b1, c1, d1]]
    }

    #[inline(always)]
    unsafe fn store_parallel_chunk(
        buffer: &mut impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
        values_a: [WasmVector32; 6],
        values_b: [WasmVector32; 6],
    ) {
        for n in 0..6 {
            let [a, b] = transpose_complex_2x2_f32(values_a[n], values_b[n]);
            buffer.store(a, i + n * 4);
            buffer.store(b, i + n * 4 + 24);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 6x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-6 FFTs
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end

        // For each pair of columns: load the data, apply our size-4 FFT, apply twiddle factors
        let [in0, in1] = Self::load_parallel_chunk(&buffer, 0);
        let tmp0 = self.bf4.perform_parallel_fft_direct(in0);
        let mut tmp1 = self.bf4.perform_parallel_fft_direct(in1);
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddle1);
        tmp1[2] = SimdVector::mul_complex(tmp1[2], self.twiddle2);
        tmp1[3] = self.bf4.rotate.rotate_both_45(tmp1[3]);

        let [in2, in3] = Self::load_parallel_chunk(&buffer, 2);
        let mut tmp2 = self.bf4.perform_parallel_fft_direct(in2);
        let mut tmp3 = self.bf4.perform_parallel_fft_direct(in3);
        tmp2[1] = SimdVector::mul_complex(tmp2[1], self.twiddle2);
        tmp2[2] = SimdVector::mul_complex(tmp2[2], self.twiddle4);
        tmp2[3] = self.bf4.rotate.rotate_both(tmp2[3]);
        tmp3[1] = self.bf4.rotate.rotate_both_45(tmp3[1]);
        tmp3[2] = self.bf4.rotate.rotate_both(tmp3[2]);
        tmp3[3] = self.bf4.rotate.rotate_both_135(tmp3[3]);

        let [in4, in5] = Self::load_parallel_chunk(&buffer, 4);
        let mut tmp4 = self.bf4.perform_parallel_fft_direct(in4);
        let mut tmp5 = self.bf4.perform_parallel_fft_direct(in5);
        tmp4[1] = SimdVector::mul_complex(tmp4[1], self.twiddle4);
        tmp4[2] = SimdVector::mul_complex(tmp4[2], self.twiddle8);
        tmp4[3] = SimdVector::neg(tmp4[3]);
        tmp5[1] = SimdVector::mul_complex(tmp5[1], self.twiddle5);
        tmp5[2] = SimdVector::mul_complex(tmp5[2], self.twiddle10);
        tmp5[3] = self.bf4.rotate.rotate_both_225(tmp5[3]);

        // Size-6 FFTs down each pair of transposed columns, storing them as soon as we're done with them
        let out0 = self
            .bf6
            .perform_parallel_fft_direct(tmp0[0], tmp1[0], tmp2[0], tmp3[0], tmp4[0], tmp5[0]);
        let out1 = self
            .bf6
            .perform_parallel_fft_direct(tmp0[1], tmp1[1], tmp2[1], tmp3[1], tmp4[1], tmp5[1]);
        Self::store_parallel_chunk(&mut buffer, 0, out0, out1);

        let out2 = self
            .bf6
            .perform_parallel_fft_direct(tmp0[2], tmp1[2], tmp2[2], tmp3[2], tmp4[2], tmp5[2]);
        let out3 = self
            .bf6
            .perform_parallel_fft_direct(tmp0[3], tmp1[3], tmp2[3], tmp3[3], tmp4[3], tmp5[3]);
        Self::store_parallel_chunk(&mut buffer, 2, out2, out3);
    }
}

//    ___ _  _              __   _  _   _     _ _
//   |__ \ || |            / /_ | || | | |__ (_) |_
//    __) ||| |_   _____  | '_ \| || |_| '_ \| | __|
//   / __/__   _| |_____| | (_) |__   _| |_) | | |_
//  |____}  |_|            \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly24<T> {
    bf4: WasmSimdF64Butterfly4<T>,
    bf6: WasmSimdF64Butterfly6<T>,
    twiddle1: WasmVector64,
    twiddle2: WasmVector64,
    twiddle4: WasmVector64,
    twiddle5: WasmVector64,
    twiddle8: WasmVector64,
    twiddle10: WasmVector64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly24,
    24,
    |this: &WasmSimdF64Butterfly24<_>| this.bf4.direction
);
impl<T: FftNum> WasmSimdF64Butterfly24<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 24, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 24, direction);
        let tw4: Complex<f64> = twiddles::compute_twiddle(4, 24, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 24, direction);
        let tw8: Complex<f64> = twiddles::compute_twiddle(8, 24, direction);
        let tw10: Complex<f64> = twiddles::compute_twiddle(10, 24, direction);

        Self {
            bf4: WasmSimdF64Butterfly4::new(direction),
            bf6: WasmSimdF64Butterfly6::new(direction),
            twiddle1: pack_64(tw1),
            twiddle2: pack_64(tw2),
            twiddle4: pack_64(tw4),
            twiddle5: pack_64(tw5),
            twiddle8: pack_64(tw8),
            twiddle10: pack_64(tw10),
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector64>) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 6x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-6 FFTs
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end
        let load = |i| {
            [
                buffer.load(i),
                buffer.load(i + 6),
                buffer.load(i + 12),
                buffer.load(i + 18),
            ]
        };

        // For each column: load the data, apply our size-4 FFT, apply twiddle factors
        let mut tmp1 = self.bf4.perform_fft_direct(load(1));
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddle1);
        tmp1[2] = SimdVector::mul_complex(tmp1[2], self.twiddle2);
        tmp1[3] = self.bf4.rotate.rotate_45(tmp1[3]);

        let mut tmp2 = self.bf4.perform_fft_direct(load(2));
        tmp2[1] = SimdVector::mul_complex(tmp2[1], self.twiddle2);
        tmp2[2] = SimdVector::mul_complex(tmp2[2], self.twiddle4);
        tmp2[3] = self.bf4.rotate.rotate(tmp2[3]);

        let mut tmp4 = self.bf4.perform_fft_direct(load(4));
        tmp4[1] = SimdVector::mul_complex(tmp4[1], self.twiddle4);
        tmp4[2] = SimdVector::mul_complex(tmp4[2], self.twiddle8);
        tmp4[3] = SimdVector::neg(tmp4[3]);

        let mut tmp5 = self.bf4.perform_fft_direct(load(5));
        tmp5[1] = SimdVector::mul_complex(tmp5[1], self.twiddle5);
        tmp5[2] = SimdVector::mul_complex(tmp5[2], self.twiddle10);
        tmp5[3] = self.bf4.rotate.rotate_225(tmp5[3]);

        let mut tmp3 = self.bf4.perform_fft_direct(load(3));
        tmp3[1] = self.bf4.rotate.rotate_45(tmp3[1]);
        tmp3[2] = self.bf4.rotate.rotate(tmp3[2]);
        tmp3[3] = self.bf4.rotate.rotate_135(tmp3[3]);

        // Do the first column last, because no twiddles means fewer temporaries forcing the above data to spill
        let tmp0 = self.bf4.perform_fft_direct(load(0));

        ////////////////////////////////////////////////////////////
        let mut store = |i, vectors: [WasmVector64; 6]| {
            buffer.store(vectors[0], i);
            buffer.store(vectors[1], i + 4);
            buffer.store(vectors[2], i + 8);
            buffer.store(vectors[3], i + 12);
            buffer.store(vectors[4], i + 16);
            buffer.store(vectors[5], i + 20);
        };

        // Size-6 FFTs down each of our transposed columns, storing them as soon as we're done with them
        let out0 = self
            .bf6
            .perform_fft_direct([tmp0[0], tmp1[0], tmp2[0], tmp3[0], tmp4[0], tmp5[0]]);
        store(0, out0);

        let out1 = self
            .bf6
            .perform_fft_direct([tmp0[1], tmp1[1], tmp2[1], tmp3[1], tmp4[1], tmp5[1]]);
        store(1, out1);

        let out2 = self
            .bf6
            .perform_fft_direct([tmp0[2], tmp1[2], tmp2[2], tmp3[2], tmp4[2], tmp5[2]]);
        store(2, out2);

        let out3 = self
            .bf6
            .perform_fft_direct([tmp0[3], tmp1[3], tmp2[3], tmp3[3], tmp4[3], tmp5[3]]);
        store(3, out3);
    }
}

//   _________            _________  _     _ _
//  |___ /___ \          |___ /___ \| |__ (_) |_
//    |_ \ __) |  _____    |_ \ __) | '_ \| | __|
//   ___) / __/  |_____|  ___) / __/| |_) | | |_
//  |____/_____|         |____/_____|_.__/|_|\__|
//

pub struct WasmSimdF32Butterfly32<T> {
    bf8: WasmSimdF32Butterfly8<T>,
    twiddles_packed: [WasmVector32; 12],
    twiddle1: WasmVector32,
    twiddle2: WasmVector32,
    twiddle3: WasmVector32,
    twiddle5: WasmVector32,
    twiddle6: WasmVector32,
    twiddle7: WasmVector32,
    twiddle9: WasmVector32,
    twiddle10: WasmVector32,
    twiddle14: WasmVector32,
    twiddle15: WasmVector32,
    twiddle18: WasmVector32,
    twiddle21: WasmVector32,
}

boilerplate_fft_wasm_simd_f32_butterfly!(
    WasmSimdF32Butterfly32,
    32,
    |this: &WasmSimdF32Butterfly32<_>| this.bf8.bf4.direction
);
impl<T: FftNum> WasmSimdF32Butterfly32<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let tw0: Complex<f32> = Complex { re: 1.0, im: 0.0 };
        let tw1: Complex<f32> = twiddles::compute_twiddle(1, 32, direction);
        let tw2: Complex<f32> = twiddles::compute_twiddle(2, 32, direction);
        let tw3: Complex<f32> = twiddles::compute_twiddle(3, 32, direction);
        let tw4: Complex<f32> = twiddles::compute_twiddle(4, 32, direction);
        let tw5: Complex<f32> = twiddles::compute_twiddle(5, 32, direction);
        let tw6: Complex<f32> = twiddles::compute_twiddle(6, 32, direction);
        let tw7: Complex<f32> = twiddles::compute_twiddle(7, 32, direction);
        let tw8: Complex<f32> = twiddles::compute_twiddle(8, 32, direction);
        let tw9: Complex<f32> = twiddles::compute_twiddle(9, 32, direction);
        let tw10: Complex<f32> = twiddles::compute_twiddle(10, 32, direction);
        let tw12: Complex<f32> = twiddles::compute_twiddle(12, 32, direction);
        let tw14: Complex<f32> = twiddles::compute_twiddle(14, 32, direction);
        let tw15: Complex<f32> = twiddles::compute_twiddle(15, 32, direction);
        let tw18: Complex<f32> = twiddles::compute_twiddle(18, 32, direction);
        let tw21: Complex<f32> = twiddles::compute_twiddle(21, 32, direction);

        Self {
            bf8: WasmSimdF32Butterfly8::new(direction),
            twiddles_packed: [
                pack_32(tw0, tw1),
                pack_32(tw0, tw2),
                pack_32(tw0, tw3),
                pack_32(tw2, tw3),
                pack_32(tw4, tw6),
                pack_32(tw6, tw9),
                pack_32(tw4, tw5),
                pack_32(tw8, tw10),
                pack_32(tw12, tw15),
                pack_32(tw6, tw7),
                pack_32(tw12, tw14),
                pack_32(tw18, tw21),
            ],
            twiddle1: pack_32(tw1, tw1),
            twiddle2: pack_32(tw2, tw2),
            twiddle3: pack_32(tw3, tw3),
            twiddle5: pack_32(tw5, tw5),
            twiddle6: pack_32(tw6, tw6),
            twiddle7: pack_32(tw7, tw7),
            twiddle9: pack_32(tw9, tw9),
            twiddle10: pack_32(tw10, tw10),
            twiddle14: pack_32(tw14, tw14),
            twiddle15: pack_32(tw15, tw15),
            twiddle18: pack_32(tw18, tw18),
            twiddle21: pack_32(tw21, tw21),
        }
    }

    #[inline(always)]
    unsafe fn load_chunk(
        buffer: &impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
    ) -> [WasmVector32; 4] {
        [
            buffer.load(i),
            buffer.load(i + 8),
            buffer.load(i + 16),
            buffer.load(i + 24),
        ]
    }

    #[inline(always)]
    unsafe fn store_chunk(
        buffer: &mut impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
        vectors: [WasmVector32; 8],
    ) {
        buffer.store(vectors[0], i + 0);
        buffer.store(vectors[1], i + 4);
        buffer.store(vectors[2], i + 8);
        buffer.store(vectors[3], i + 12);
        buffer.store(vectors[4], i + 16);
        buffer.store(vectors[5], i + 20);
        buffer.store(vectors[6], i + 24);
        buffer.store(vectors[7], i + 28);
    }

    #[target_feature(enable = "simd128")]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector32>) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 8x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-8 FFTs
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end

        // For each pair of columns: load the data, apply our size-4 FFT, apply twiddle factors
        let mut tmp0 = self
            .bf8
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 0));
        tmp0[1] = SimdVector::mul_complex(tmp0[1], self.twiddles_packed[0]);
        tmp0[2] = SimdVector::mul_complex(tmp0[2], self.twiddles_packed[1]);
        tmp0[3] = SimdVector::mul_complex(tmp0[3], self.twiddles_packed[2]);
        let [mid0, mid1] = transpose_complex_2x2_f32(tmp0[0], tmp0[1]);
        let [mid8, mid9] = transpose_complex_2x2_f32(tmp0[2], tmp0[3]);

        let mut tmp1 = self
            .bf8
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 2));
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddles_packed[3]);
        tmp1[2] = SimdVector::mul_complex(tmp1[2], self.twiddles_packed[4]);
        tmp1[3] = SimdVector::mul_complex(tmp1[3], self.twiddles_packed[5]);
        let [mid2, mid3] = transpose_complex_2x2_f32(tmp1[0], tmp1[1]);
        let [mid10, mid11] = transpose_complex_2x2_f32(tmp1[2], tmp1[3]);

        let mut tmp2 = self
            .bf8
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 4));
        tmp2[1] = SimdVector::mul_complex(tmp2[1], self.twiddles_packed[6]);
        tmp2[2] = SimdVector::mul_complex(tmp2[2], self.twiddles_packed[7]);
        tmp2[3] = SimdVector::mul_complex(tmp2[3], self.twiddles_packed[8]);
        let [mid4, mid5] = transpose_complex_2x2_f32(tmp2[0], tmp2[1]);
        let [mid12, mid13] = transpose_complex_2x2_f32(tmp2[2], tmp2[3]);

        let mut tmp3 = self
            .bf8
            .bf4
            .perform_parallel_fft_direct(Self::load_chunk(&buffer, 6));
        tmp3[1] = SimdVector::mul_complex(tmp3[1], self.twiddles_packed[9]);
        tmp3[2] = SimdVector::mul_complex(tmp3[2], self.twiddles_packed[10]);
        tmp3[3] = SimdVector::mul_complex(tmp3[3], self.twiddles_packed[11]);
        let [mid6, mid7] = transpose_complex_2x2_f32(tmp3[0], tmp3[1]);
        let [mid14, mid15] = transpose_complex_2x2_f32(tmp3[2], tmp3[3]);

        // Size-8 FFTs down each pair of transposed columns, storing them as soon as we're done with them
        let out0 = self
            .bf8
            .perform_parallel_fft_direct([mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7]);
        Self::store_chunk(&mut buffer, 0, out0);

        let out1 = self
            .bf8
            .perform_parallel_fft_direct([mid8, mid9, mid10, mid11, mid12, mid13, mid14, mid15]);
        Self::store_chunk(&mut buffer, 2, out1);
    }

    #[inline(always)]
    unsafe fn load_parallel_chunk(
        buffer: &impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
    ) -> [[WasmVector32; 4]; 2] {
        let [a0, a1] = transpose_complex_2x2_f32(buffer.load(i + 0), buffer.load(i + 32));
        let [b0, b1] = transpose_complex_2x2_f32(buffer.load(i + 8), buffer.load(i + 40));
        let [c0, c1] = transpose_complex_2x2_f32(buffer.load(i + 16), buffer.load(i + 48));
        let [d0, d1] = transpose_complex_2x2_f32(buffer.load(i + 24), buffer.load(i + 56));
        [[a0, b0, c0, d0], [a1, b1, c1, d1]]
    }

    #[inline(always)]
    unsafe fn store_parallel_chunk(
        buffer: &mut impl SimdComplexArrayMut<WasmVector32>,
        i: usize,
        values_a: [WasmVector32; 8],
        values_b: [WasmVector32; 8],
    ) {
        for n in 0..8 {
            let [a, b] = transpose_complex_2x2_f32(values_a[n], values_b[n]);
            buffer.store(a, i + n * 4);
            buffer.store(b, i + n * 4 + 32);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(
        &self,
        mut buffer: impl SimdComplexArrayMut<WasmVector32>,
    ) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 8x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-8 FFTs
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end

        // For each pair of columns: load the data, apply our size-4 FFT, apply twiddle factors
        let [in0, in1] = Self::load_parallel_chunk(&buffer, 0);
        let tmp0 = self.bf8.bf4.perform_parallel_fft_direct(in0);
        let mut tmp1 = self.bf8.bf4.perform_parallel_fft_direct(in1);
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddle1);
        tmp1[2] = SimdVector::mul_complex(tmp1[2], self.twiddle2);
        tmp1[3] = SimdVector::mul_complex(tmp1[3], self.twiddle3);

        let [in2, in3] = Self::load_parallel_chunk(&buffer, 2);
        let mut tmp2 = self.bf8.bf4.perform_parallel_fft_direct(in2);
        let mut tmp3 = self.bf8.bf4.perform_parallel_fft_direct(in3);
        tmp2[1] = SimdVector::mul_complex(tmp2[1], self.twiddle2);
        tmp2[2] = self.bf8.bf4.rotate.rotate_both_45(tmp2[2]);
        tmp2[3] = SimdVector::mul_complex(tmp2[3], self.twiddle6);
        tmp3[1] = SimdVector::mul_complex(tmp3[1], self.twiddle3);
        tmp3[2] = SimdVector::mul_complex(tmp3[2], self.twiddle6);
        tmp3[3] = SimdVector::mul_complex(tmp3[3], self.twiddle9);

        let [in4, in5] = Self::load_parallel_chunk(&buffer, 4);
        let mut tmp4 = self.bf8.bf4.perform_parallel_fft_direct(in4);
        let mut tmp5 = self.bf8.bf4.perform_parallel_fft_direct(in5);
        tmp4[1] = self.bf8.bf4.rotate.rotate_both_45(tmp4[1]);
        tmp4[2] = self.bf8.bf4.rotate.rotate_both(tmp4[2]);
        tmp4[3] = self.bf8.bf4.rotate.rotate_both_135(tmp4[3]);
        tmp5[1] = SimdVector::mul_complex(tmp5[1], self.twiddle5);
        tmp5[2] = SimdVector::mul_complex(tmp5[2], self.twiddle10);
        tmp5[3] = SimdVector::mul_complex(tmp5[3], self.twiddle15);

        let [in6, in7] = Self::load_parallel_chunk(&buffer, 6);
        let mut tmp6 = self.bf8.bf4.perform_parallel_fft_direct(in6);
        let mut tmp7 = self.bf8.bf4.perform_parallel_fft_direct(in7);
        tmp6[1] = SimdVector::mul_complex(tmp6[1], self.twiddle6);
        tmp6[2] = self.bf8.bf4.rotate.rotate_both_135(tmp6[2]);
        tmp6[3] = SimdVector::mul_complex(tmp6[3], self.twiddle18);
        tmp7[1] = SimdVector::mul_complex(tmp7[1], self.twiddle7);
        tmp7[2] = SimdVector::mul_complex(tmp7[2], self.twiddle14);
        tmp7[3] = SimdVector::mul_complex(tmp7[3], self.twiddle21);

        // Size-8 FFTs down each pair of transposed columns, storing them as soon as we're done with them
        let out0 = self.bf8.perform_parallel_fft_direct([
            tmp0[0], tmp1[0], tmp2[0], tmp3[0], tmp4[0], tmp5[0], tmp6[0], tmp7[0],
        ]);
        let out1 = self.bf8.perform_parallel_fft_direct([
            tmp0[1], tmp1[1], tmp2[1], tmp3[1], tmp4[1], tmp5[1], tmp6[1], tmp7[1],
        ]);
        Self::store_parallel_chunk(&mut buffer, 0, out0, out1);

        let out2 = self.bf8.perform_parallel_fft_direct([
            tmp0[2], tmp1[2], tmp2[2], tmp3[2], tmp4[2], tmp5[2], tmp6[2], tmp7[2],
        ]);
        let out3 = self.bf8.perform_parallel_fft_direct([
            tmp0[3], tmp1[3], tmp2[3], tmp3[3], tmp4[3], tmp5[3], tmp6[3], tmp7[3],
        ]);
        Self::store_parallel_chunk(&mut buffer, 2, out2, out3);
    }
}

//   _________             __   _  _   _     _ _
//  |___ /___ \           / /_ | || | | |__ (_) |_
//    |_ \ __) |  _____  | '_ \| || |_| '_ \| | __|
//   ___) / __/  |_____| | (_) |__   _| |_) | | |_
//  |____/_____|          \___/   |_| |_.__/|_|\__|
//

pub struct WasmSimdF64Butterfly32<T> {
    bf8: WasmSimdF64Butterfly8<T>,
    twiddle1: WasmVector64,
    twiddle2: WasmVector64,
    twiddle3: WasmVector64,
    twiddle5: WasmVector64,
    twiddle6: WasmVector64,
    twiddle7: WasmVector64,
    twiddle9: WasmVector64,
    twiddle10: WasmVector64,
    twiddle14: WasmVector64,
    twiddle15: WasmVector64,
    twiddle18: WasmVector64,
    twiddle21: WasmVector64,
}

boilerplate_fft_wasm_simd_f64_butterfly!(
    WasmSimdF64Butterfly32,
    32,
    |this: &WasmSimdF64Butterfly32<_>| this.bf8.bf4.direction
);
impl<T: FftNum> WasmSimdF64Butterfly32<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let tw1: Complex<f64> = twiddles::compute_twiddle(1, 32, direction);
        let tw2: Complex<f64> = twiddles::compute_twiddle(2, 32, direction);
        let tw3: Complex<f64> = twiddles::compute_twiddle(3, 32, direction);
        let tw5: Complex<f64> = twiddles::compute_twiddle(5, 32, direction);
        let tw6: Complex<f64> = twiddles::compute_twiddle(6, 32, direction);
        let tw7: Complex<f64> = twiddles::compute_twiddle(7, 32, direction);
        let tw9: Complex<f64> = twiddles::compute_twiddle(9, 32, direction);
        let tw10: Complex<f64> = twiddles::compute_twiddle(10, 32, direction);
        let tw14: Complex<f64> = twiddles::compute_twiddle(14, 32, direction);
        let tw15: Complex<f64> = twiddles::compute_twiddle(15, 32, direction);
        let tw18: Complex<f64> = twiddles::compute_twiddle(18, 32, direction);
        let tw21: Complex<f64> = twiddles::compute_twiddle(21, 32, direction);

        Self {
            bf8: WasmSimdF64Butterfly8::new(direction),
            twiddle1: pack_64(tw1),
            twiddle2: pack_64(tw2),
            twiddle3: pack_64(tw3),
            twiddle5: pack_64(tw5),
            twiddle6: pack_64(tw6),
            twiddle7: pack_64(tw7),
            twiddle9: pack_64(tw9),
            twiddle10: pack_64(tw10),
            twiddle14: pack_64(tw14),
            twiddle15: pack_64(tw15),
            twiddle18: pack_64(tw18),
            twiddle21: pack_64(tw21),
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl SimdComplexArrayMut<WasmVector64>) {
        // To make the best possible use of registers, we're going to write this algorithm in an unusual way
        // It's 8x4 mixed radix, so we're going to do the usual steps of size-4 FFTs down the columns, apply twiddle factors, then transpose and do size-8 FFTs
        // But to reduce the number of times registers get spilled, we have these optimizations:
        // 1: Load data as late as possible, not upfront
        // 2: Once we're working with a piece of data, make as much progress as possible before moving on
        //      IE, once we load a column, we should do the FFT down the column, do twiddle factors, and do the pieces of the transpose for that column, all before starting on the next column
        // 3: Store data as soon as we're finished with it, rather than waiting for the end
        let load = |i| {
            [
                buffer.load(i),
                buffer.load(i + 8),
                buffer.load(i + 16),
                buffer.load(i + 24),
            ]
        };

        // For each column: load the data, apply our size-4 FFT, apply twiddle factors
        let mut tmp1 = self.bf8.bf4.perform_fft_direct(load(1));
        tmp1[1] = SimdVector::mul_complex(tmp1[1], self.twiddle1);
        tmp1[2] = SimdVector::mul_complex(tmp1[2], self.twiddle2);
        tmp1[3] = SimdVector::mul_complex(tmp1[3], self.twiddle3);

        let mut tmp2 = self.bf8.bf4.perform_fft_direct(load(2));
        tmp2[1] = SimdVector::mul_complex(tmp2[1], self.twiddle2);
        tmp2[2] = self.bf8.bf4.rotate.rotate_45(tmp2[2]);
        tmp2[3] = SimdVector::mul_complex(tmp2[3], self.twiddle6);

        let mut tmp3 = self.bf8.bf4.perform_fft_direct(load(3));
        tmp3[1] = SimdVector::mul_complex(tmp3[1], self.twiddle3);
        tmp3[2] = SimdVector::mul_complex(tmp3[2], self.twiddle6);
        tmp3[3] = SimdVector::mul_complex(tmp3[3], self.twiddle9);

        let mut tmp5 = self.bf8.bf4.perform_fft_direct(load(5));
        tmp5[1] = SimdVector::mul_complex(tmp5[1], self.twiddle5);
        tmp5[2] = SimdVector::mul_complex(tmp5[2], self.twiddle10);
        tmp5[3] = SimdVector::mul_complex(tmp5[3], self.twiddle15);

        let mut tmp6 = self.bf8.bf4.perform_fft_direct(load(6));
        tmp6[1] = SimdVector::mul_complex(tmp6[1], self.twiddle6);
        tmp6[2] = self.bf8.bf4.rotate.rotate_135(tmp6[2]);
        tmp6[3] = SimdVector::mul_complex(tmp6[3], self.twiddle18);

        let mut tmp7 = self.bf8.bf4.perform_fft_direct(load(7));
        tmp7[1] = SimdVector::mul_complex(tmp7[1], self.twiddle7);
        tmp7[2] = SimdVector::mul_complex(tmp7[2], self.twiddle14);
        tmp7[3] = SimdVector::mul_complex(tmp7[3], self.twiddle21);

        let mut tmp4 = self.bf8.bf4.perform_fft_direct(load(4));
        tmp4[1] = self.bf8.bf4.rotate.rotate_45(tmp4[1]);
        tmp4[2] = self.bf8.bf4.rotate.rotate(tmp4[2]);
        tmp4[3] = self.bf8.bf4.rotate.rotate_135(tmp4[3]);

        // Do the first column last, because no twiddles means fewer temporaries forcing the above data to spill
        let tmp0 = self.bf8.bf4.perform_fft_direct(load(0));

        ////////////////////////////////////////////////////////////
        let mut store = |i, vectors: [WasmVector64; 8]| {
            buffer.store(vectors[0], i);
            buffer.store(vectors[1], i + 4);
            buffer.store(vectors[2], i + 8);
            buffer.store(vectors[3], i + 12);
            buffer.store(vectors[4], i + 16);
            buffer.store(vectors[5], i + 20);
            buffer.store(vectors[6], i + 24);
            buffer.store(vectors[7], i + 28);
        };

        // Size-8 FFTs down each of our transposed columns, storing them as soon as we're done with them
        let out0 = self.bf8.perform_fft_direct([
            tmp0[0], tmp1[0], tmp2[0], tmp3[0], tmp4[0], tmp5[0], tmp6[0], tmp7[0],
        ]);
        store(0, out0);

        let out1 = self.bf8.perform_fft_direct([
            tmp0[1], tmp1[1], tmp2[1], tmp3[1], tmp4[1], tmp5[1], tmp6[1], tmp7[1],
        ]);
        store(1, out1);

        let out2 = self.bf8.perform_fft_direct([
            tmp0[2], tmp1[2], tmp2[2], tmp3[2], tmp4[2], tmp5[2], tmp6[2], tmp7[2],
        ]);
        store(2, out2);

        let out3 = self.bf8.perform_fft_direct([
            tmp0[3], tmp1[3], tmp2[3], tmp3[3], tmp4[3], tmp5[3], tmp6[3], tmp7[3],
        ]);
        store(3, out3);
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
    test_butterfly_32_func!(test_wasm_simdf32_butterfly24, WasmSimdF32Butterfly24, 24);
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
    test_butterfly_64_func!(test_wasm_simdf64_butterfly24, WasmSimdF64Butterfly24, 24);
    test_butterfly_64_func!(test_wasm_simdf64_butterfly32, WasmSimdF64Butterfly32, 32);

    #[wasm_bindgen_test]
    fn test_solo_fft2_32() {
        unsafe {
            let val1 = Complex::<f32>::new(1.0, 2.5);
            let val2 = Complex::<f32>::new(3.2, 4.2);

            let mut val = vec![val1, val2];

            let in_packed = WasmVector32(v128_load(val.as_ptr() as *const v128));

            let dft = Dft::new(2, FftDirection::Forward);

            let bf2 = WasmSimdF32Butterfly2::<f32>::new(FftDirection::Forward);

            dft.process(&mut val);
            let res_packed = bf2.perform_fft_direct(in_packed);

            let res = std::mem::transmute::<WasmVector32, [Complex<f32>; 2]>(res_packed);
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

            let p1 = WasmVector32(v128_load(val_a.as_ptr() as *const v128));
            let p2 = WasmVector32(v128_load(val_b.as_ptr() as *const v128));

            let dft = Dft::new(2, FftDirection::Forward);

            let bf2 = WasmSimdF32Butterfly2::<f32>::new(FftDirection::Forward);

            dft.process(&mut val_a);
            dft.process(&mut val_b);
            let res_both = bf2.perform_parallel_fft_direct(p1, p2);

            let res = std::mem::transmute::<[WasmVector32; 2], [Complex<f32>; 4]>(res_both);
            let wasmsimd_res_a = [res[0], res[2]];
            let wasmsimd_res_b = [res[1], res[3]];
            assert!(compare_vectors(&val_a, &wasmsimd_res_a));
            assert!(compare_vectors(&val_b, &wasmsimd_res_b));
        }
    }
}
