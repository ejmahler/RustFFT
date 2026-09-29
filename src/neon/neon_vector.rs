use core::arch::aarch64::*;
use num_complex::Complex;
use num_traits::Zero;

use crate::simd::simd_array::SimdComplexArray;
use crate::simd::simd_vector::SimdVector;
use crate::{twiddles, FftDirection};

use super::neon_butterflies::{
    NeonF32Butterfly3, NeonF32Butterfly5, NeonF32Butterfly6, NeonF64Butterfly3, NeonF64Butterfly5,
    NeonF64Butterfly6,
};
use super::neon_prime_butterflies::{NeonF32Butterfly7, NeonF64Butterfly7};

// Read these indexes from an NeonArray and build an array of simd vectors.
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

// Read these indexes from an NeonArray and build an array or partially filled simd vectors.
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

// Write these indexes of an array of simd vectors to the same indexes of an NeonArray.
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

// Write the low half of these indexes of an array of simd vectors to the same indexes of an NeonArray.
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

// Write these indexes of an array of simd vectors to the same indexes, multiplied by a stride, of an NeonArray.
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

// The `SimdVector::fft_helper_*` methods, which are the same forwarding calls for every NEON
// vector type: they hand the chunk loop to the target-feature-enabled wrappers in
// `neon_common.rs`.
macro_rules! neon_vector_fft_helpers {
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
            super::neon_common::neon_fft_helper_immut(
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
            super::neon_common::neon_fft_helper_outofplace(
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
            super::neon_common::neon_fft_helper_inplace(
                buffer,
                scratch,
                chunk_size,
                required_scratch,
                chunk_fn,
            )
        }
    };
}

impl crate::simd::simd_vector::SimdVector for float64x2_t {
    const COMPLEX_PER_VECTOR: usize = 1;
    const RADIXN_CROSS_LAYER_UNROLL: bool = true;

    type ScalarType = f64;
    type Rotation = Rotation90<Self>;

    type Butterfly3 = NeonF64Butterfly3<f64>;
    type Butterfly5 = NeonF64Butterfly5<f64>;
    type Butterfly6 = NeonF64Butterfly6<f64>;
    type Butterfly7 = NeonF64Butterfly7<f64>;

    #[inline(always)]
    unsafe fn zero_vector() -> Self {
        vdupq_n_f64(0.0)
    }
    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        vld1q_f64(ptr as *const f64)
    }

    #[inline(always)]
    unsafe fn load1_lo_complex(_ptr: *const Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }

    #[inline(always)]
    unsafe fn load1_dup_complex(_ptr: *const Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }

    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        vst1q_f64(ptr as *mut f64, data);
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
        vnegq_f64(a)
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        vaddq_f64(a, b)
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        vsubq_f64(a, b)
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        vmulq_f64(a, b)
    }
    #[inline(always)]
    unsafe fn fmadd(acc: Self, a: Self, b: Self) -> Self {
        vfmaq_f64(acc, a, b)
    }
    #[inline(always)]
    unsafe fn nmadd(acc: Self, a: Self, b: Self) -> Self {
        vfmaq_f64(acc, a, vnegq_f64(b))
    }

    #[inline(always)]
    unsafe fn broadcast_scalar(value: Self::ScalarType) -> Self {
        vmovq_n_f64(value)
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
        // ARMv8.2-A introduced vcmulq_f64 and vcmlaq_f64 for complex multiplication, these intrinsics are not yet available.
        let temp = vcombine_f64(vneg_f64(vget_high_f64(left)), vget_low_f64(left));
        let sum = vmulq_laneq_f64::<0>(left, right);
        vfmaq_laneq_f64::<1>(sum, temp, right)
    }

    #[inline(always)]
    unsafe fn make_rotate90(direction: FftDirection) -> Rotation90<Self> {
        Rotation90(match direction {
            FftDirection::Forward => vld1q_f64([0.0, -0.0].as_ptr()),
            FftDirection::Inverse => vld1q_f64([-0.0, 0.0].as_ptr()),
        })
    }

    #[inline(always)]
    unsafe fn apply_rotate90(direction: Rotation90<Self>, values: Self) -> Self {
        let temp = vcombine_f64(vget_high_f64(values), vget_low_f64(values));
        vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(temp),
            vreinterpretq_u64_f64(direction.0),
        ))
    }
    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        NeonF64Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        NeonF64Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        NeonF64Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        NeonF64Butterfly7::new(direction)
    }

    #[inline(always)]
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        neon_column_butterfly2(rows)
    }
    #[inline(always)]
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        bf.perform_fft_direct(rows[0], rows[1], rows[2])
    }
    #[inline(always)]
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Self::Rotation) -> [Self; 4] {
        neon_column_butterfly4(rows, rotation)
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

    simd_vector_cross_layer!(#[inline(always)]);
    neon_vector_fft_helpers!();
}

impl crate::simd::simd_vector::SimdVector for float32x4_t {
    const COMPLEX_PER_VECTOR: usize = 2;
    const RADIXN_CROSS_LAYER_UNROLL: bool = true;

    type ScalarType = f32;
    type Rotation = Rotation90<Self>;

    type Butterfly3 = NeonF32Butterfly3<f32>;
    type Butterfly5 = NeonF32Butterfly5<f32>;
    type Butterfly6 = NeonF32Butterfly6<f32>;
    type Butterfly7 = NeonF32Butterfly7<f32>;

    #[inline(always)]
    unsafe fn zero_vector() -> Self {
        vdupq_n_f32(0.0)
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        vld1q_f32(ptr as *const f32)
    }

    #[inline(always)]
    unsafe fn load1_lo_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        let temp = vmovq_n_f32(0.0);
        vreinterpretq_f32_u64(vld1q_lane_u64::<0>(
            ptr as *const u64,
            vreinterpretq_u64_f32(temp),
        ))
    }

    #[inline(always)]
    unsafe fn load1_dup_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        vreinterpretq_f32_u64(vld1q_dup_u64(ptr as *const u64))
    }

    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        vst1q_f32(ptr as *mut f32, data);
    }

    #[inline(always)]
    unsafe fn store1_lo_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        let low = vget_low_f32(data);
        vst1_f32(ptr as *mut f32, low);
    }

    #[inline(always)]
    unsafe fn store1_hi_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        let high = vget_high_f32(data);
        vst1_f32(ptr as *mut f32, high);
    }

    #[inline(always)]
    unsafe fn neg(a: Self) -> Self {
        vnegq_f32(a)
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        vaddq_f32(a, b)
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        vsubq_f32(a, b)
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        vmulq_f32(a, b)
    }
    #[inline(always)]
    unsafe fn fmadd(acc: Self, a: Self, b: Self) -> Self {
        vfmaq_f32(acc, a, b)
    }
    #[inline(always)]
    unsafe fn nmadd(acc: Self, a: Self, b: Self) -> Self {
        vfmaq_f32(acc, a, vnegq_f32(b))
    }

    #[inline(always)]
    unsafe fn broadcast_scalar(value: Self::ScalarType) -> Self {
        vmovq_n_f32(value)
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
        // ARMv8.2-A introduced vcmulq_f32 and vcmlaq_f32 for complex multiplication, these intrinsics are not yet available.
        let temp1 = vtrn1q_f32(right, right);
        let temp2 = vtrn2q_f32(right, vnegq_f32(right));
        let temp3 = vmulq_f32(temp2, left);
        let temp4 = vrev64q_f32(temp3);
        vfmaq_f32(temp4, temp1, left)
    }

    #[inline(always)]
    unsafe fn make_rotate90(direction: FftDirection) -> Rotation90<Self> {
        Rotation90(match direction {
            FftDirection::Forward => vld1q_f32([0.0, -0.0, 0.0, -0.0].as_ptr()),
            FftDirection::Inverse => vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr()),
        })
    }

    #[inline(always)]
    unsafe fn apply_rotate90(direction: Rotation90<Self>, values: Self) -> Self {
        let temp = vrev64q_f32(values);
        vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(temp),
            vreinterpretq_u32_f32(direction.0),
        ))
    }

    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        NeonF32Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        NeonF32Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        NeonF32Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        NeonF32Butterfly7::new(direction)
    }

    #[inline(always)]
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        neon_column_butterfly2(rows)
    }
    #[inline(always)]
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        bf.perform_parallel_fft_direct(rows[0], rows[1], rows[2])
    }
    #[inline(always)]
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Self::Rotation) -> [Self; 4] {
        neon_column_butterfly4(rows, rotation)
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

    simd_vector_cross_layer!(#[inline(always)]);
    neon_vector_fft_helpers!();
}

#[inline(always)]
unsafe fn neon_column_butterfly2<V: SimdVector>(rows: [V; 2]) -> [V; 2] {
    [
        SimdVector::add(rows[0], rows[1]),
        SimdVector::sub(rows[0], rows[1]),
    ]
}

#[inline(always)]
unsafe fn neon_column_butterfly4<V: SimdVector>(rows: [V; 4], rotation: V::Rotation) -> [V; 4] {
    // Algorithm: 2x2 mixed radix

    // Perform the first set of size-2 FFTs.
    let [mid0, mid2] = neon_column_butterfly2([rows[0], rows[2]]);
    let [mid1, mid3] = neon_column_butterfly2([rows[1], rows[3]]);

    // Apply twiddle factors (in this case just a rotation)
    let mid3_rotated = V::apply_rotate90(rotation, mid3);

    // Transpose the data and do size-2 FFTs down the columns
    let [output0, output1] = neon_column_butterfly2([mid0, mid1]);
    let [output2, output3] = neon_column_butterfly2([mid2, mid3_rotated]);

    // Swap outputs 1 and 2 in the output to do a square transpose
    [output0, output2, output1, output3]
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::simd::simd_array::SimdComplexArrayMut;

    use num_complex::Complex;

    #[test]
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
                std::mem::transmute::<float64x2_t, Complex<f64>>(load1)
            );
            assert_eq!(
                val2,
                std::mem::transmute::<float64x2_t, Complex<f64>>(load2)
            );
            assert_eq!(
                val3,
                std::mem::transmute::<float64x2_t, Complex<f64>>(load3)
            );
            assert_eq!(
                val4,
                std::mem::transmute::<float64x2_t, Complex<f64>>(load4)
            );
        }
    }

    #[test]
    fn test_store_f64() {
        unsafe {
            let val1: Complex<f64> = Complex::new(1.0, 2.0);
            let val2: Complex<f64> = Complex::new(3.0, 4.0);
            let val3: Complex<f64> = Complex::new(5.0, 6.0);
            let val4: Complex<f64> = Complex::new(7.0, 8.0);

            let nbr1 = vld1q_f64(&val1 as *const _ as *const f64);
            let nbr2 = vld1q_f64(&val2 as *const _ as *const f64);
            let nbr3 = vld1q_f64(&val3 as *const _ as *const f64);
            let nbr4 = vld1q_f64(&val4 as *const _ as *const f64);

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
