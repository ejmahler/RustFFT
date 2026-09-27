use num_complex::Complex;
use num_traits::Zero;
use std::arch::x86_64::*;

use crate::simd::simd_array::SimdComplexArray;
use crate::simd::simd_vector::SimdVector;
use crate::{twiddles, FftDirection};

use super::sse_butterflies::{
    SseF32Butterfly3, SseF32Butterfly5, SseF32Butterfly6, SseF64Butterfly3, SseF64Butterfly5,
    SseF64Butterfly6,
};
use super::sse_prime_butterflies::{SseF32Butterfly7, SseF64Butterfly7};

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

// The `SimdVector` impls, which let this backend use the algorithms in `src/simd`. The trait is
// named by path instead of imported, because importing it would make methods like
// `Self::column_butterfly2` ambiguous with the backend's own vector trait.

// The `SimdVector::fft_helper_*` methods, which are the same forwarding calls for every SSE
// vector type: they hand the chunk loop to the target-feature-enabled wrappers in
// `sse_common.rs`.
macro_rules! sse_vector_fft_helpers {
    () => {
        #[inline(always)]
        unsafe fn fft_helper_immut<E>(
            input: &[E],
            output: &mut [E],
            scratch: &mut [E],
            chunk_size: usize,
            required_scratch: usize,
            chunk_fn: impl FnMut(&[E], &mut [E], &mut [E]),
        ) {
            super::sse_common::sse_fft_helper_immut(
                input,
                output,
                scratch,
                chunk_size,
                required_scratch,
                chunk_fn,
            )
        }
        #[inline(always)]
        unsafe fn fft_helper_outofplace<E>(
            input: &mut [E],
            output: &mut [E],
            scratch: &mut [E],
            chunk_size: usize,
            required_scratch: usize,
            chunk_fn: impl FnMut(&mut [E], &mut [E], &mut [E]),
        ) {
            super::sse_common::sse_fft_helper_outofplace(
                input,
                output,
                scratch,
                chunk_size,
                required_scratch,
                chunk_fn,
            )
        }
        #[inline(always)]
        unsafe fn fft_helper_inplace<E>(
            buffer: &mut [E],
            scratch: &mut [E],
            chunk_size: usize,
            required_scratch: usize,
            chunk_fn: impl FnMut(&mut [E], &mut [E]),
        ) {
            super::sse_common::sse_fft_helper_inplace(
                buffer,
                scratch,
                chunk_size,
                required_scratch,
                chunk_fn,
            )
        }
    };
}

impl crate::simd::simd_vector::SimdVector for __m128d {
    const COMPLEX_PER_VECTOR: usize = 1;

    type ScalarType = f64;
    type Rotation = Rotation90<Self>;

    type Butterfly3 = SseF64Butterfly3<f64>;
    type Butterfly5 = SseF64Butterfly5<f64>;
    type Butterfly6 = SseF64Butterfly6<f64>;
    type Butterfly7 = SseF64Butterfly7<f64>;

    #[inline(always)]
    unsafe fn zero() -> Self {
        _mm_setzero_pd()
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm_loadu_pd(ptr as *const f64)
    }

    #[inline(always)]
    unsafe fn load1_lo_complex(_ptr: *const Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a load store of complex f64's");
    }

    #[inline(always)]
    unsafe fn load1_lo_broadcast_complex(_ptr: *const Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a load store of complex f64's");
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
    unsafe fn make_rotate90(direction: FftDirection) -> Self::Rotation {
        Rotation90(match direction {
            FftDirection::Forward => _mm_set_pd(-0.0, 0.0),
            FftDirection::Inverse => _mm_set_pd(0.0, -0.0),
        })
    }

    #[inline(always)]
    unsafe fn apply_rotate90(direction: Self::Rotation, values: Self) -> Self {
        let temp = _mm_shuffle_pd(values, values, 0x01);
        _mm_xor_pd(temp, direction.0)
    }
    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        SseF64Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        SseF64Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        SseF64Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        SseF64Butterfly7::new(direction)
    }

    #[inline(always)]
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        sse_column_butterfly2(rows)
    }
    #[inline(always)]
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        bf.perform_fft_direct(rows[0], rows[1], rows[2])
    }
    #[inline(always)]
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Self::Rotation) -> [Self; 4] {
        sse_column_butterfly4(rows, rotation)
    }
    #[inline(always)]
    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5] {
        bf.perform_fft_direct(rows[0], rows[1], rows[2], rows[3], rows[4])
    }
    #[inline(always)]
    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6] {
        bf.perform_fft_direct(rows)
    }
    #[inline(always)]
    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7] {
        bf.perform_fft_direct(rows)
    }

    simd_vector_cross_layer!(#[target_feature(enable = "sse4.1")]);
    sse_vector_fft_helpers!();
}

impl crate::simd::simd_vector::SimdVector for __m128 {
    const COMPLEX_PER_VECTOR: usize = 2;

    type ScalarType = f32;
    type Rotation = Rotation90<Self>;

    type Butterfly3 = SseF32Butterfly3<f32>;
    type Butterfly5 = SseF32Butterfly5<f32>;
    type Butterfly6 = SseF32Butterfly6<f32>;
    type Butterfly7 = SseF32Butterfly7<f32>;

    #[inline(always)]
    unsafe fn zero() -> Self {
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
    unsafe fn load1_lo_broadcast_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
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
    unsafe fn make_rotate90(direction: FftDirection) -> Self::Rotation {
        Rotation90(match direction {
            FftDirection::Forward => _mm_set_ps(-0.0, 0.0, -0.0, 0.0),
            FftDirection::Inverse => _mm_set_ps(0.0, -0.0, 0.0, -0.0),
        })
    }

    #[inline(always)]
    unsafe fn apply_rotate90(direction: Self::Rotation, values: Self) -> Self {
        let temp = _mm_shuffle_ps(values, values, 0xB1);
        _mm_xor_ps(temp, direction.0)
    }
    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        SseF32Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        SseF32Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        SseF32Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        SseF32Butterfly7::new(direction)
    }

    #[inline(always)]
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        sse_column_butterfly2(rows)
    }
    #[inline(always)]
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        bf.perform_parallel_fft_direct(rows[0], rows[1], rows[2])
    }
    #[inline(always)]
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Self::Rotation) -> [Self; 4] {
        sse_column_butterfly4(rows, rotation)
    }
    #[inline(always)]
    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5] {
        bf.perform_parallel_fft_direct(rows[0], rows[1], rows[2], rows[3], rows[4])
    }
    #[inline(always)]
    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6] {
        bf.perform_parallel_fft_direct(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5])
    }
    #[inline(always)]
    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7] {
        bf.perform_parallel_fft_direct(rows)
    }

    simd_vector_cross_layer!(#[target_feature(enable = "sse4.1")]);
    sse_vector_fft_helpers!();
}

#[inline(always)]
unsafe fn sse_column_butterfly2<V: SimdVector>(rows: [V; 2]) -> [V; 2] {
    [
        SimdVector::add(rows[0], rows[1]),
        SimdVector::sub(rows[0], rows[1]),
    ]
}

#[inline(always)]
unsafe fn sse_column_butterfly4<V: SimdVector>(rows: [V; 4], rotation: V::Rotation) -> [V; 4] {
    // Algorithm: 2x2 mixed radix

    // Perform the first set of size-2 FFTs.
    let [mid0, mid2] = sse_column_butterfly2([rows[0], rows[2]]);
    let [mid1, mid3] = sse_column_butterfly2([rows[1], rows[3]]);

    // Apply twiddle factors (in this case just a rotation)
    let mid3_rotated = V::apply_rotate90(rotation, mid3);

    // Transpose the data and do size-2 FFTs down the columns
    let [output0, output1] = sse_column_butterfly2([mid0, mid1]);
    let [output2, output3] = sse_column_butterfly2([mid2, mid3_rotated]);

    // Swap outputs 1 and 2 in the output to do a square transpose
    [output0, output2, output1, output3]
}
