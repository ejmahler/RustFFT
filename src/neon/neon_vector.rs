use core::arch::aarch64::*;
use num_complex::Complex;

use crate::array_utils::{RawSlice, RawSliceMut};
 
// Read these indexes from an SseArray and build an array of simd vectors.
// Takes a name of a vector to read from, and a list of indexes to read.
// This statement:
// ```
// let values = read_complex_to_array!(input, {0, 1, 2, 3});
// ```
// is equivalent to:
// ```
// let values = [
//     input.load_complex(0),
//     input.load_complex(1),
//     input.load_complex(2),
//     input.load_complex(3),
// ];
// ```
macro_rules! read_complex_to_array {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            $input.load_complex($idx),
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
//     input.load1_complex(0),
//     input.load1_complex(1),
//     input.load1_complex(2),
//     input.load1_complex(3),
// ];
// ```
macro_rules! read_partial1_complex_to_array {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            $input.load1_complex($idx),
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
//     output.store_complex(input[0], 0),
//     output.store_complex(input[1], 1),
//     output.store_complex(input[2], 2),
//     output.store_complex(input[3], 3),
// ];
// ```
macro_rules! write_complex_to_array {
    ($input:ident, $output:ident, { $($idx:literal),* }) => {
        $(
            $output.store_complex($input[$idx], $idx);
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
//     output.store_partial_lo_complex(input[0], 0),
//     output.store_partial_lo_complex(input[1], 1),
//     output.store_partial_lo_complex(input[2], 2),
//     output.store_partial_lo_complex(input[3], 3),
// ];
// ```
macro_rules! write_partial_lo_complex_to_array {
    ($input:ident, $output:ident, { $($idx:literal),* }) => {
        $(
            $output.store_partial_lo_complex($input[$idx], $idx);
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
//     output.store_complex(input[0], 0),
//     output.store_complex(input[1], 2),
//     output.store_complex(input[2], 4),
//     output.store_complex(input[3], 6),
// ];
// ```
macro_rules! write_complex_to_array_strided {
    ($input:ident, $output:ident, $stride:literal, { $($idx:literal),* }) => {
        $(
            $output.store_complex($input[$idx], $idx*$stride);
        )*
    }
}

// A trait to handle reading from an array of complex floats into SSE vectors.
// SSE works with 128-bit vectors, meaning a vector can hold two complex f32,
// or a single complex f64.
pub trait NeonArray {
    type VectorType;
    const COMPLEX_PER_VECTOR: usize;
    // Load complex numbers from the array to fill a SSE vector.
    unsafe fn load_complex(&self, index: usize) -> Self::VectorType;
    // Load a single complex number from the array into a SSE vector, setting the unused elements to zero.
    unsafe fn load_partial1_complex(&self, index: usize) -> Self::VectorType;
    // Load a single complex number from the array, and copy it to all elements of a SSE vector.
    unsafe fn load1_complex(&self, index: usize) -> Self::VectorType;
}

impl NeonArray for RawSlice<Complex<f32>> {
    type VectorType = float32x4_t;
    const COMPLEX_PER_VECTOR: usize = 2;

    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> Self::VectorType {
        debug_assert!(self.len() >= index + Self::COMPLEX_PER_VECTOR);
        vld1q_f32(self.as_ptr().add(index) as *const f32)
    }

    #[inline(always)]
    unsafe fn load_partial1_complex(&self, index: usize) -> Self::VectorType {
        debug_assert!(self.len() >= index + 1);
        let temp = vmovq_n_f32(0.0);
        vreinterpretq_f32_u64(vld1q_lane_u64::<0>(self.as_ptr().add(index) as *const u64, vreinterpretq_u64_f32(temp)))
    }

    #[inline(always)]
    unsafe fn load1_complex(&self, index: usize) -> Self::VectorType {
        debug_assert!(self.len() >= index + 1);
        vreinterpretq_f32_u64(vld1q_dup_u64(self.as_ptr().add(index) as *const u64))
    }
}

impl NeonArray for RawSlice<Complex<f64>> {
    type VectorType = float64x2_t;
    const COMPLEX_PER_VECTOR: usize = 1;

    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> Self::VectorType {
        debug_assert!(self.len() >= index + Self::COMPLEX_PER_VECTOR);
        vld1q_f64(self.as_ptr().add(index) as *const f64)
    }

    #[inline(always)]
    unsafe fn load_partial1_complex(&self, _index: usize) -> Self::VectorType {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }

    #[inline(always)]
    unsafe fn load1_complex(&self, _index: usize) -> Self::VectorType {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }
}


// A trait to handle writing to an array of complex floats from SSE vectors.
// SSE works with 128-bit vectors, meaning a vector can hold two complex f32,
// or a single complex f64.
pub trait NeonArrayMut {
    type VectorType;
    const COMPLEX_PER_VECTOR: usize;
    // Store all complex numbers from a SSE vector to the array.
    unsafe fn store_complex(&self, vector: Self::VectorType, index: usize);
    // Store the low complex number from a SSE vector to the array.
    unsafe fn store_partial_lo_complex(&self, vector: Self::VectorType, index: usize);
    // Store the high complex number from a SSE vector to the array.
    unsafe fn store_partial_hi_complex(&self, vector: Self::VectorType, index: usize);
}


impl NeonArrayMut for RawSliceMut<Complex<f32>> {
    type VectorType = float32x4_t;
    const COMPLEX_PER_VECTOR: usize = 2;

    #[inline(always)]
    unsafe fn store_complex(&self, vector: Self::VectorType, index: usize) {
        debug_assert!(self.len() >= index + Self::COMPLEX_PER_VECTOR);
        vst1q_f32(self.as_mut_ptr().add(index) as *mut f32, vector);
    }

    #[inline(always)]
    unsafe fn store_partial_hi_complex(&self, vector: Self::VectorType, index: usize) {
        debug_assert!(self.len() >= index + 1);
        let high = vget_high_f32(vector);
        vst1_f32(self.as_mut_ptr().add(index) as *mut f32, high);
    }

    #[inline(always)]
    unsafe fn store_partial_lo_complex(&self, vector: Self::VectorType, index: usize) {
        debug_assert!(self.len() >= index + 1);
        let low = vget_low_f32(vector);
        vst1_f32(self.as_mut_ptr().add(index) as *mut f32, low);
    }
}


impl NeonArrayMut for RawSliceMut<Complex<f64>> {
    type VectorType = float64x2_t;
    const COMPLEX_PER_VECTOR: usize = 1;

    #[inline(always)]
    unsafe fn store_complex(&self, vector: Self::VectorType, index: usize) {
        debug_assert!(self.len() >= index + Self::COMPLEX_PER_VECTOR);
        vst1q_f64(self.as_mut_ptr().add(index) as *mut f64, vector);
    }

    #[inline(always)]
    unsafe fn store_partial_hi_complex(&self, _vector: Self::VectorType, _index: usize) {
        unimplemented!("Impossible to do a partial store of complex f64's");
    }
    #[inline(always)]
    unsafe fn store_partial_lo_complex(&self, _vector: Self::VectorType, _index: usize) {
        unimplemented!("Impossible to do a partial store of complex f64's");
    }
}
