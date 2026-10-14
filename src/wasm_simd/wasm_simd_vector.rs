use core::arch::wasm32::*;
use num_complex::Complex;
use num_traits::Zero;
use std::fmt::Debug;

use crate::{twiddles, FftDirection};

use crate::simd::simd_array::SimdComplexArray;
use crate::simd::simd_vector::SimdVector;
use crate::wasm_simd::wasm_simd_butterflies::*;
use crate::wasm_simd::wasm_simd_prime_butterflies::*;

/// Read these indexes from an WasmSimdArray and build an array of simd vectors.
/// Takes a name of a vector to read from, and a list of indexes to read.
/// This statement:
///
/// let values = read_complex_to_array!(input, {0, 1, 2, 3});
///
/// is equivalent to:
///
/// let values = [
///     input.load(0),
///     input.load(1),
///     input.load(2),
///     input.load(3),
/// ];
macro_rules! read_complex_to_array {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            $input.load($idx),
        )*
        ]
    }
}

/// Read these indexes from an WasmSimdArray and build an array or partially filled simd vectors.
/// Takes a name of a vector to read from, and a list of indexes to read.
/// This statement:
///
/// let values = read_partial1_complex_to_array!(input, {0, 1, 2, 3});
///
/// is equivalent to:
///
/// let values = [
///     input.load1_lo(0),
///     input.load1_lo(1),
///     input.load1_lo(2),
///     input.load1_lo(3),
/// ];
///
macro_rules! read_partial1_complex_to_array {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            $input.load1_lo($idx),
        )*
        ]
    }
}

/// Write these indexes of an array of simd vectors to the same indexes of an WasmSimdArray.
/// Takes a name of a vector to read from, one to write to, and a list of indexes.
/// This statement:
///
/// let values = write_complex_to_array!(input, output, {0, 1, 2, 3});
///
/// is equivalent to:
///
/// let values = [
///     output.store(input[0], 0),
///     output.store(input[1], 1),
///     output.store(input[2], 2),
///     output.store(input[3], 3),
/// ];
///
macro_rules! write_complex_to_array {
    ($input:ident, $output:ident, { $($idx:literal),* }) => {
        $(
            $output.store($input[$idx], $idx);
        )*
    }
}

/// Write the low half of these indexes of an array of simd vectors to the same indexes of an WasmSimdArray.
/// Takes a name of a vector to read from, one to write to, and a list of indexes.
/// This statement:
///
/// let values = write_partial_lo_complex_to_array!(input, output, {0, 1, 2, 3});
///
/// is equivalent to:
///
/// let values = [
///     output.store1_lo(input[0], 0),
///     output.store1_lo(input[1], 1),
///     output.store1_lo(input[2], 2),
///     output.store1_lo(input[3], 3),
/// ];
///
macro_rules! write_partial_lo_complex_to_array {
    ($input:ident, $output:ident, { $($idx:literal),* }) => {
        $(
            $output.store1_lo($input[$idx], $idx);
        )*
    }
}

/// Write these indexes of an array of simd vectors to the same indexes, multiplied by a stride, of an WasmSimdArray.
/// Takes a name of a vector to read from, one to write to, an integer stride, and a list of indexes.
/// This statement:
///
/// let values = write_complex_to_array_strided!(input, output, {0, 1, 2, 3});
///
/// is equivalent to:
///
/// let values = [
///     output.store(input[0], 0),
///     output.store(input[1], 2),
///     output.store(input[2], 4),
///     output.store(input[3], 6),
/// ];
///
macro_rules! write_complex_to_array_strided {
    ($input:ident, $output:ident, $stride:literal, { $($idx:literal),* }) => {
        $(
            $output.store($input[$idx], $idx*$stride);
        )*
    }
}

// We need newtypes for wasm vectors, since they don't have different vector types for f32 vs f64
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct WasmVector32(pub v128);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct WasmVector64(pub v128);

#[derive(Copy, Clone)]
pub struct Rotation90<V: SimdVector>(V);

impl crate::simd::simd_vector::SimdVector for WasmVector64 {
    const COMPLEX_PER_VECTOR: usize = 1;
    const RADIXN_CROSS_LAYER_UNROLL: bool = true;

    type ScalarType = f64;
    type Rotation = Rotation90<Self>;

    type Butterfly2 = WasmSimdF64ColumnButterfly2;
    type Butterfly3 = WasmSimdF64ColumnButterfly3;
    type Butterfly4 = WasmSimdF64ColumnButterfly4;
    type Butterfly5 = WasmSimdF64ColumnButterfly5;
    type Butterfly6 = WasmSimdF64ColumnButterfly6;
    type Butterfly7 = WasmSimdF64ColumnButterfly7;

    #[inline(always)]
    unsafe fn zero_vector() -> Self {
        Self(f64x2(0.0, 0.0))
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        Self(v128_load(ptr as *const v128))
    }

    #[inline(always)]
    unsafe fn load1_lo_complex(_ptr: *const Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }

    #[inline(always)]
    unsafe fn load1_dup_complex(_ptr: *const Complex<Self::ScalarType>) -> Self {
        unimplemented!("Unused: a dup load of one complex f64 is just load_complex");
    }

    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        v128_store(ptr as *mut v128, data.0);
    }

    #[inline(always)]
    unsafe fn store1_lo_complex(_ptr: *mut Complex<Self::ScalarType>, _data: Self) {
        unimplemented!("Impossible to do a partial store of complex f64's");
    }

    #[inline(always)]
    unsafe fn store1_hi_complex(_ptr: *mut Complex<Self::ScalarType>, _data: Self) {
        unimplemented!("Impossible to do a partial store of complex f64's");
    }

    #[inline(always)]
    unsafe fn neg(a: Self) -> Self {
        Self(f64x2_neg(a.0))
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        Self(f64x2_add(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        Self(f64x2_sub(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        Self(f64x2_mul(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn fmadd(acc: Self, a: Self, b: Self) -> Self {
        Self(f64x2_add(acc.0, f64x2_mul(a.0, b.0)))
    }
    #[inline(always)]
    unsafe fn nmadd(acc: Self, a: Self, b: Self) -> Self {
        Self(f64x2_sub(acc.0, f64x2_mul(a.0, b.0)))
    }

    #[inline(always)]
    unsafe fn broadcast_scalar(value: Self::ScalarType) -> Self {
        Self(f64x2_splat(value))
    }

    #[inline(always)]
    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: FftDirection,
    ) -> Self {
        let mut twiddle_chunk = [Complex::<f64>::zero(); Self::COMPLEX_PER_VECTOR];
        for i in 0..Self::COMPLEX_PER_VECTOR {
            twiddle_chunk[i] = twiddles::compute_twiddle(y * (x + i), len, direction);
        }

        twiddle_chunk.as_slice().load(0)
    }

    #[inline(always)]
    unsafe fn mul_complex(left: Self, right: Self) -> Self {
        const NEGATE_LEFT: v128 = f64x2(-0.0, 0.0);
        let temp = v128_xor(u64x2_shuffle::<1, 0>(left.0, left.0), NEGATE_LEFT);
        let sum = f64x2_mul(left.0, u64x2_shuffle::<0, 0>(right.0, right.0));
        Self(f64x2_add(
            sum,
            f64x2_mul(temp, u64x2_shuffle::<1, 1>(right.0, right.0)),
        ))
    }

    #[inline(always)]
    unsafe fn make_butterfly2(direction: FftDirection) -> Self::Butterfly2 {
        Self::Butterfly2::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        Self::Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly4(direction: FftDirection) -> Self::Butterfly4 {
        Self::Butterfly4::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        Self::Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        Self::Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        Self::Butterfly7::new(direction)
    }
}

impl crate::simd::simd_vector::SimdVector for WasmVector32 {
    const COMPLEX_PER_VECTOR: usize = 2;
    const RADIXN_CROSS_LAYER_UNROLL: bool = true;

    type ScalarType = f32;
    type Rotation = Rotation90<Self>;

    type Butterfly2 = WasmSimdF32ColumnButterfly2;
    type Butterfly3 = WasmSimdF32ColumnButterfly3;
    type Butterfly4 = WasmSimdF32ColumnButterfly4;
    type Butterfly5 = WasmSimdF32ColumnButterfly5;
    type Butterfly6 = WasmSimdF32ColumnButterfly6;
    type Butterfly7 = WasmSimdF32ColumnButterfly7;

    #[inline(always)]
    unsafe fn zero_vector() -> Self {
        Self(f32x4(0.0, 0.0, 0.0, 0.0))
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        Self(v128_load(ptr as *const v128))
    }

    #[inline(always)]
    unsafe fn load1_lo_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        Self(v128_load64_lane::<0>(f32x4_splat(0.0), ptr as *const u64))
    }

    #[inline(always)]
    unsafe fn load1_dup_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        Self(v128_load64_splat(ptr as *const u64))
    }

    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        v128_store(ptr as *mut v128, data.0);
    }

    #[inline(always)]
    unsafe fn store1_lo_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        v128_store64_lane::<0>(data.0, ptr as *mut u64);
    }

    #[inline(always)]
    unsafe fn store1_hi_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        v128_store64_lane::<1>(data.0, ptr as *mut u64);
    }

    #[inline(always)]
    unsafe fn neg(a: Self) -> Self {
        Self(f32x4_neg(a.0))
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        Self(f32x4_add(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        Self(f32x4_sub(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        Self(f32x4_mul(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn fmadd(acc: Self, a: Self, b: Self) -> Self {
        Self(f32x4_add(acc.0, f32x4_mul(a.0, b.0)))
    }
    #[inline(always)]
    unsafe fn nmadd(acc: Self, a: Self, b: Self) -> Self {
        Self(f32x4_sub(acc.0, f32x4_mul(a.0, b.0)))
    }

    #[inline(always)]
    unsafe fn broadcast_scalar(value: Self::ScalarType) -> Self {
        Self(f32x4_splat(value))
    }

    #[inline(always)]
    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: FftDirection,
    ) -> Self {
        let mut twiddle_chunk = [Complex::<f32>::zero(); Self::COMPLEX_PER_VECTOR];
        for i in 0..Self::COMPLEX_PER_VECTOR {
            twiddle_chunk[i] = twiddles::compute_twiddle(y * (x + i), len, direction);
        }

        twiddle_chunk.as_slice().load(0)
    }

    #[inline(always)]
    unsafe fn mul_complex(left: Self, right: Self) -> Self {
        let temp1 = u32x4_shuffle::<0, 4, 2, 6>(right.0, right.0);
        let temp2 = u32x4_shuffle::<1, 5, 3, 7>(right.0, f32x4_neg(right.0));
        let temp3 = f32x4_mul(temp2, left.0);
        let temp4 = u32x4_shuffle::<1, 0, 3, 2>(temp3, temp3);
        let temp5 = f32x4_mul(temp1, left.0);
        WasmVector32(f32x4_add(temp4, temp5))
    }

    #[inline(always)]
    unsafe fn make_butterfly2(direction: FftDirection) -> Self::Butterfly2 {
        Self::Butterfly2::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        Self::Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly4(direction: FftDirection) -> Self::Butterfly4 {
        Self::Butterfly4::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        Self::Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        Self::Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        Self::Butterfly7::new(direction)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    use crate::simd::simd_array::{SimdComplexArray, SimdComplexArrayMut};
    use num_complex::Complex;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_load_f64() {
        unsafe {
            let val1: Complex<f64> = Complex::new(1.0, 2.0);
            let val2: Complex<f64> = Complex::new(3.0, 4.0);
            let val3: Complex<f64> = Complex::new(5.0, 6.0);
            let val4: Complex<f64> = Complex::new(7.0, 8.0);
            let values = vec![val1, val2, val3, val4];
            let slice = values.as_slice();
            let load1 = slice.load(0);
            let load2 = slice.load(1);
            let load3 = slice.load(2);
            let load4 = slice.load(3);
            assert_eq!(
                val1,
                std::mem::transmute::<WasmVector64, Complex<f64>>(load1)
            );
            assert_eq!(
                val2,
                std::mem::transmute::<WasmVector64, Complex<f64>>(load2)
            );
            assert_eq!(
                val3,
                std::mem::transmute::<WasmVector64, Complex<f64>>(load3)
            );
            assert_eq!(
                val4,
                std::mem::transmute::<WasmVector64, Complex<f64>>(load4)
            );
        }
    }

    #[wasm_bindgen_test]
    fn test_store_f64() {
        unsafe {
            let val1: Complex<f64> = Complex::new(1.0, 2.0);
            let val2: Complex<f64> = Complex::new(3.0, 4.0);
            let val3: Complex<f64> = Complex::new(5.0, 6.0);
            let val4: Complex<f64> = Complex::new(7.0, 8.0);

            let nbr1 = WasmVector64(v128_load(&val1 as *const _ as *const v128));
            let nbr2 = WasmVector64(v128_load(&val2 as *const _ as *const v128));
            let nbr3 = WasmVector64(v128_load(&val3 as *const _ as *const v128));
            let nbr4 = WasmVector64(v128_load(&val4 as *const _ as *const v128));

            let mut values: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 4];
            let mut slice = values.as_mut_slice();
            slice.store(nbr1, 0);
            slice.store(nbr2, 1);
            slice.store(nbr3, 2);
            slice.store(nbr4, 3);
            assert_eq!(val1, values[0]);
            assert_eq!(val2, values[1]);
            assert_eq!(val3, values[2]);
            assert_eq!(val4, values[3]);
        }
    }
}
