use num_complex::Complex;
use num_traits::Zero;
use std::arch::x86_64::*;

use crate::simd::simd_array::SimdComplexArray;
use crate::simd::simd_vector::SimdVector;
use crate::sse::sse_butterflies::*;
use crate::sse::sse_prime_butterflies::*;
use crate::{twiddles, FftDirection};

// Read these indexes from an SseArray and build an array of simd vectors.
// Takes a name of a vector to read from, and a list of indexes to read.
// This statement:
// ```
// let values = read_complex_to_array!(input, {0, 1, 2, 3});
// ```
// is equivalent to:
// ```
// let values = [
//     input.load(0),
//     input.load(1),
//     input.load(2),
//     input.load(3),
// ];
// ```
macro_rules! read_complex_to_array {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            $input.load($idx),
        )*
        ]
    }
}

// Read these indexes from an SseArray and build an array or partially filled simd vectors.
// Takes a name of a vector to read from, and a list of indexes to read.
// This statement:
// ```
// let values = read_partial1_complex_to_array!(input, {0, 1, 2, 3});
// ```
// is equivalent to:
// ```
// let values = [
//     input.load1_lo(0),
//     input.load1_lo(1),
//     input.load1_lo(2),
//     input.load1_lo(3),
// ];
// ```
macro_rules! read_partial1_complex_to_array {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            $input.load1_lo($idx),
        )*
        ]
    }
}

// Write these indexes of an array of simd vectors to the same indexes of an SseArray.
// Takes a name of a vector to read from, one to write to, and a list of indexes.
// This statement:
// ```
// let values = write_complex_to_array!(input, output, {0, 1, 2, 3});
// ```
// is equivalent to:
// ```
// let values = [
//     output.store(input[0], 0),
//     output.store(input[1], 1),
//     output.store(input[2], 2),
//     output.store(input[3], 3),
// ];
// ```
macro_rules! write_complex_to_array {
    ($input:ident, $output:ident, { $($idx:literal),* }) => {
        $(
            $output.store($input[$idx], $idx);
        )*
    }
}

// Write the low half of these indexes of an array of simd vectors to the same indexes of an SseArray.
// Takes a name of a vector to read from, one to write to, and a list of indexes.
// This statement:
// ```
// let values = write_partial_lo_complex_to_array!(input, output, {0, 1, 2, 3});
// ```
// is equivalent to:
// ```
// let values = [
//     output.store1_lo(input[0], 0),
//     output.store1_lo(input[1], 1),
//     output.store1_lo(input[2], 2),
//     output.store1_lo(input[3], 3),
// ];
// ```
macro_rules! write_partial_lo_complex_to_array {
    ($input:ident, $output:ident, { $($idx:literal),* }) => {
        $(
            $output.store1_lo($input[$idx], $idx);
        )*
    }
}

// Write these indexes of an array of simd vectors to the same indexes, multiplied by a stride, of an SseArray.
// Takes a name of a vector to read from, one to write to, an integer stride, and a list of indexes.
// This statement:
// ```
// let values = write_complex_to_array_separate!(input, output, {0, 1, 2, 3});
// ```
// is equivalent to:
// ```
// let values = [
//     output.store(input[0], 0),
//     output.store(input[1], 2),
//     output.store(input[2], 4),
//     output.store(input[3], 6),
// ];
// ```
macro_rules! write_complex_to_array_strided {
    ($input:ident, $output:ident, $stride:literal, { $($idx:literal),* }) => {
        $(
            $output.store($input[$idx], $idx*$stride);
        )*
    }
}

#[derive(Copy, Clone)]
pub struct Rotation90<V: SimdVector>(V);

impl crate::simd::simd_vector::SimdVector for __m128d {
    const COMPLEX_PER_VECTOR: usize = 1;
    const RADIXN_CROSS_LAYER_UNROLL: bool = true;

    type ScalarType = f64;
    type Rotation = Rotation90<Self>;

    type Butterfly2 = SseF64ColumnButterfly2;
    type Butterfly3 = SseF64ColumnButterfly3;
    type Butterfly4 = SseF64ColumnButterfly4;
    type Butterfly5 = SseF64ColumnButterfly5;
    type Butterfly6 = SseF64ColumnButterfly6;
    type Butterfly7 = SseF64ColumnButterfly7;

    #[inline(always)]
    unsafe fn zero_vector() -> Self {
        _mm_setzero_pd()
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm_loadu_pd(ptr as *const f64)
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
        _mm_storeu_pd(ptr as *mut f64, data);
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
        _mm_xor_pd(a, _mm_set1_pd(-0.0))
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        _mm_add_pd(a, b)
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        _mm_sub_pd(a, b)
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        _mm_mul_pd(a, b)
    }
    #[inline(always)]
    unsafe fn fmadd(acc: Self, a: Self, b: Self) -> Self {
        _mm_add_pd(acc, _mm_mul_pd(a, b))
    }
    #[inline(always)]
    unsafe fn nmadd(acc: Self, a: Self, b: Self) -> Self {
        _mm_sub_pd(acc, _mm_mul_pd(a, b))
    }

    #[inline(always)]
    unsafe fn broadcast_scalar(value: Self::ScalarType) -> Self {
        _mm_set1_pd(value)
    }

    #[inline(always)]
    unsafe fn mul_complex(left: Self, right: Self) -> Self {
        // SSE3, taken from Intel performance manual
        let mut temp1 = _mm_unpacklo_pd(right, right);
        let mut temp2 = _mm_unpackhi_pd(right, right);
        temp1 = _mm_mul_pd(temp1, left);
        temp2 = _mm_mul_pd(temp2, left);
        temp2 = _mm_shuffle_pd(temp2, temp2, 0x01);
        _mm_addsub_pd(temp1, temp2)
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

impl crate::simd::simd_vector::SimdVector for __m128 {
    const COMPLEX_PER_VECTOR: usize = 2;
    const RADIXN_CROSS_LAYER_UNROLL: bool = true;

    type ScalarType = f32;
    type Rotation = Rotation90<Self>;

    type Butterfly2 = SseF32ColumnButterfly2;
    type Butterfly3 = SseF32ColumnButterfly3;
    type Butterfly4 = SseF32ColumnButterfly4;
    type Butterfly5 = SseF32ColumnButterfly5;
    type Butterfly6 = SseF32ColumnButterfly6;
    type Butterfly7 = SseF32ColumnButterfly7;

    #[inline(always)]
    unsafe fn zero_vector() -> Self {
        _mm_setzero_ps()
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm_loadu_ps(ptr as *const f32)
    }

    #[inline(always)]
    unsafe fn load1_lo_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm_castpd_ps(_mm_load_sd(ptr as *const f64))
    }

    #[inline(always)]
    unsafe fn load1_dup_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm_castpd_ps(_mm_load1_pd(ptr as *const f64))
    }

    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm_storeu_ps(ptr as *mut f32, data);
    }

    #[inline(always)]
    unsafe fn store1_lo_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm_storel_pd(ptr as *mut f64, _mm_castps_pd(data));
    }

    #[inline(always)]
    unsafe fn store1_hi_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm_storeh_pd(ptr as *mut f64, _mm_castps_pd(data));
    }

    #[inline(always)]
    unsafe fn neg(a: Self) -> Self {
        _mm_xor_ps(a, _mm_set1_ps(-0.0))
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        _mm_add_ps(a, b)
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        _mm_sub_ps(a, b)
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        _mm_mul_ps(a, b)
    }
    #[inline(always)]
    unsafe fn fmadd(acc: Self, a: Self, b: Self) -> Self {
        _mm_add_ps(acc, _mm_mul_ps(a, b))
    }
    #[inline(always)]
    unsafe fn nmadd(acc: Self, a: Self, b: Self) -> Self {
        _mm_sub_ps(acc, _mm_mul_ps(a, b))
    }

    #[inline(always)]
    unsafe fn broadcast_scalar(value: Self::ScalarType) -> Self {
        _mm_set1_ps(value)
    }

    #[inline(always)]
    unsafe fn mul_complex(left: Self, right: Self) -> Self {
        //SSE3, taken from Intel performance manual
        let mut temp1 = _mm_shuffle_ps(right, right, 0xA0);
        let mut temp2 = _mm_shuffle_ps(right, right, 0xF5);
        temp1 = _mm_mul_ps(temp1, left);
        temp2 = _mm_mul_ps(temp2, left);
        temp2 = _mm_shuffle_ps(temp2, temp2, 0xB1);
        _mm_addsub_ps(temp1, temp2)
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
