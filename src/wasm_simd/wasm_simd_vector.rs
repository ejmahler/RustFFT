// TODO: update docs
use core::arch::wasm32::*;
use num_complex::Complex;
use std::ops::{Deref, DerefMut};

use crate::array_utils::DoubleBuf;

/// Read these indexes from an NeonArray and build an array of simd vectors.
/// Takes a name of a vector to read from, and a list of indexes to read.
/// This statement:
/// ```
/// let values = read_complex_to_array!(input, {0, 1, 2, 3});
/// ```
/// is equivalent to:
/// ```
/// let values = [
///     input.load_complex(0),
///     input.load_complex(1),
///     input.load_complex(2),
///     input.load_complex(3),
/// ];
/// ```
macro_rules! read_complex_to_array {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            $input.load_complex($idx),
        )*
        ]
    }
}

/// Read these indexes from an NeonArray and build an array or partially filled simd vectors.
/// Takes a name of a vector to read from, and a list of indexes to read.
/// This statement:
/// ```
/// let values = read_partial1_complex_to_array!(input, {0, 1, 2, 3});
/// ```
/// is equivalent to:
/// ```
/// let values = [
///     input.load1_complex(0),
///     input.load1_complex(1),
///     input.load1_complex(2),
///     input.load1_complex(3),
/// ];
/// ```
macro_rules! read_partial1_complex_to_array {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            $input.load1_complex($idx),
        )*
        ]
    }
}

/// Write these indexes of an array of simd vectors to the same indexes of an NeonArray.
/// Takes a name of a vector to read from, one to write to, and a list of indexes.
/// This statement:
/// ```
/// let values = write_complex_to_array!(input, output, {0, 1, 2, 3});
/// ```
/// is equivalent to:
/// ```
/// let values = [
///     output.store_complex(input[0], 0),
///     output.store_complex(input[1], 1),
///     output.store_complex(input[2], 2),
///     output.store_complex(input[3], 3),
/// ];
/// ```
macro_rules! write_complex_to_array {
    ($input:ident, $output:ident, { $($idx:literal),* }) => {
        $(
            $output.store_complex($input[$idx], $idx);
        )*
    }
}

/// Write the low half of these indexes of an array of simd vectors to the same indexes of an NeonArray.
/// Takes a name of a vector to read from, one to write to, and a list of indexes.
/// This statement:
/// ```
/// let values = write_partial_lo_complex_to_array!(input, output, {0, 1, 2, 3});
/// ```
/// is equivalent to:
/// ```
/// let values = [
///     output.store_partial_lo_complex(input[0], 0),
///     output.store_partial_lo_complex(input[1], 1),
///     output.store_partial_lo_complex(input[2], 2),
///     output.store_partial_lo_complex(input[3], 3),
/// ];
/// ```
macro_rules! write_partial_lo_complex_to_array {
    ($input:ident, $output:ident, { $($idx:literal),* }) => {
        $(
            $output.store_partial_lo_complex($input[$idx], $idx);
        )*
    }
}

/// Write these indexes of an array of simd vectors to the same indexes, multiplied by a stride, of an NeonArray.
/// Takes a name of a vector to read from, one to write to, an integer stride, and a list of indexes.
/// This statement:
/// ```
/// let values = write_complex_to_array_separate!(input, output, {0, 1, 2, 3});
/// ```
/// is equivalent to:
/// ```
/// let values = [
///     output.store_complex(input[0], 0),
///     output.store_complex(input[1], 2),
///     output.store_complex(input[2], 4),
///     output.store_complex(input[3], 6),
/// ];
/// ```
macro_rules! write_complex_to_array_strided {
    ($input:ident, $output:ident, $stride:literal, { $($idx:literal),* }) => {
        $(
            $output.store_complex($input[$idx], $idx*$stride);
        )*
    }
}

pub trait WasmSimdNum {
    type VectorType;
    const COMPLEX_PER_VECTOR: usize;
}
impl WasmSimdNum for f32 {
    type VectorType = v128;
    const COMPLEX_PER_VECTOR: usize = 2;
}
impl WasmSimdNum for f64 {
    type VectorType = v128;
    const COMPLEX_PER_VECTOR: usize = 1;
}

/// A trait to handle reading from an array of complex floats into Neon vectors.
/// Neon works with 128-bit vectors, meaning a vector can hold two complex f32,
/// or a single complex f64.
pub trait WasmSimdArray<T: WasmSimdNum>: Deref {
    /// Load complex numbers from the array to fill a Neon vector.
    unsafe fn load_complex(&self, index: usize) -> T::VectorType;
    /// Load a single complex number from the array into a Neon vector, setting the unused elements to zero.
    unsafe fn load_partial1_complex(&self, index: usize) -> T::VectorType;
    /// Load a single complex number from the array, and copy it to all elements of a Neon vector.
    unsafe fn load1_complex(&self, index: usize) -> T::VectorType;
}

impl WasmSimdArray<f32> for &[Complex<f32>] {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> <f32 as WasmSimdNum>::VectorType {
        debug_assert!(self.len() >= index + <f32 as WasmSimdNum>::COMPLEX_PER_VECTOR);
        let Complex { re: re1, im: im1 } = self[index];
        let Complex { re: re2, im: im2 } = self[index + 1];
        f32x4(re1, im1, re2, im2)
    }

    #[inline(always)]
    unsafe fn load_partial1_complex(&self, index: usize) -> <f32 as WasmSimdNum>::VectorType {
        debug_assert!(self.len() >= index + 1);
        let Complex { re, im } = self[index];
        f32x4(re, im, 0.0, 0.0)
    }

    #[inline(always)]
    unsafe fn load1_complex(&self, index: usize) -> <f32 as WasmSimdNum>::VectorType {
        debug_assert!(self.len() >= index + 1);

        let Complex { re, im } = self[index];
        f32x4(re, im, re, im)
    }
}

impl WasmSimdArray<f32> for &mut [Complex<f32>] {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> <f32 as WasmSimdNum>::VectorType {
        debug_assert!(self.len() >= index + <f32 as WasmSimdNum>::COMPLEX_PER_VECTOR);
        let Complex { re: re1, im: im1 } = self[index];
        let Complex { re: re2, im: im2 } = self[index + 1];
        f32x4(re1, im1, re2, im2)
    }

    #[inline(always)]
    unsafe fn load_partial1_complex(&self, index: usize) -> <f32 as WasmSimdNum>::VectorType {
        debug_assert!(self.len() >= index + 1);
        let Complex { re, im } = self[index];
        f32x4(re, im, 0.0, 0.0)
    }

    #[inline(always)]
    unsafe fn load1_complex(&self, index: usize) -> <f32 as WasmSimdNum>::VectorType {
        debug_assert!(self.len() >= index + 1);
        let Complex { re, im } = self[index];
        f32x4(re, im, re, im)
    }
}

impl WasmSimdArray<f64> for &[Complex<f64>] {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> <f64 as WasmSimdNum>::VectorType {
        debug_assert!(self.len() >= index + <f64 as WasmSimdNum>::COMPLEX_PER_VECTOR);
        let Complex { re, im } = self[index];
        f64x2(re, im)
    }

    #[inline(always)]
    unsafe fn load_partial1_complex(&self, _index: usize) -> <f64 as WasmSimdNum>::VectorType {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }

    #[inline(always)]
    unsafe fn load1_complex(&self, _index: usize) -> <f64 as WasmSimdNum>::VectorType {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }
}

impl WasmSimdArray<f64> for &mut [Complex<f64>] {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> <f64 as WasmSimdNum>::VectorType {
        debug_assert!(self.len() >= index + <f64 as WasmSimdNum>::COMPLEX_PER_VECTOR);
        let Complex { re, im } = self[index];
        f64x2(re, im)
    }

    #[inline(always)]
    unsafe fn load_partial1_complex(&self, _index: usize) -> <f64 as WasmSimdNum>::VectorType {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }

    #[inline(always)]
    unsafe fn load1_complex(&self, _index: usize) -> <f64 as WasmSimdNum>::VectorType {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }
}

impl<'a, T: WasmSimdNum> WasmSimdArray<T> for DoubleBuf<'a, T>
where
    &'a [Complex<T>]: WasmSimdArray<T>,
{
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> T::VectorType {
        self.input.load_complex(index)
    }
    #[inline(always)]
    unsafe fn load_partial1_complex(&self, index: usize) -> T::VectorType {
        self.input.load_partial1_complex(index)
    }
    #[inline(always)]
    unsafe fn load1_complex(&self, index: usize) -> T::VectorType {
        self.input.load1_complex(index)
    }
}

/// A trait to handle writing to an array of complex floats from Neon vectors.
/// Neon works with 128-bit vectors, meaning a vector can hold two complex f32,
/// or a single complex f64.
pub trait WasmSimdArrayMut<T: WasmSimdNum>: WasmSimdArray<T> + DerefMut {
    /// Store all complex numbers from a Neon vector to the array.
    unsafe fn store_complex(&mut self, vector: T::VectorType, index: usize);
    /// Store the low complex number from a Neon vector to the array.
    unsafe fn store_partial_lo_complex(&mut self, vector: T::VectorType, index: usize);
    /// Store the high complex number from a Neon vector to the array.
    unsafe fn store_partial_hi_complex(&mut self, vector: T::VectorType, index: usize);
}

impl WasmSimdArrayMut<f32> for &mut [Complex<f32>] {
    #[inline(always)]
    unsafe fn store_complex(&mut self, vector: <f32 as WasmSimdNum>::VectorType, index: usize) {
        debug_assert!(self.len() >= index + <f32 as WasmSimdNum>::COMPLEX_PER_VECTOR);
        // vst1q_f32(self.as_mut_ptr().add(index) as *mut f32, vector);
        v128_store64_lane::<0>(vector, self.as_mut_ptr().add(index) as *mut u64);
        v128_store64_lane::<1>(vector, self.as_mut_ptr().add(index + 1) as *mut u64);
    }

    #[inline(always)]
    unsafe fn store_partial_hi_complex(
        &mut self,
        vector: <f32 as WasmSimdNum>::VectorType,
        index: usize,
    ) {
        debug_assert!(self.len() >= index + 1);
        v128_store64_lane::<1>(vector, self.as_mut_ptr().add(index) as *mut u64);
    }

    #[inline(always)]
    unsafe fn store_partial_lo_complex(
        &mut self,
        vector: <f32 as WasmSimdNum>::VectorType,
        index: usize,
    ) {
        debug_assert!(self.len() >= index + 1);
        v128_store64_lane::<0>(vector, self.as_mut_ptr().add(index) as *mut u64);
    }
}

impl WasmSimdArrayMut<f64> for &mut [Complex<f64>] {
    #[inline(always)]
    unsafe fn store_complex(&mut self, vector: <f64 as WasmSimdNum>::VectorType, index: usize) {
        debug_assert!(self.len() >= index + <f64 as WasmSimdNum>::COMPLEX_PER_VECTOR);
        v128_store(self.as_mut_ptr().add(index) as *mut v128, vector);
    }

    #[inline(always)]
    unsafe fn store_partial_hi_complex(
        &mut self,
        _vector: <f64 as WasmSimdNum>::VectorType,
        _index: usize,
    ) {
        unimplemented!("Impossible to do a partial store of complex f64's");
    }
    #[inline(always)]
    unsafe fn store_partial_lo_complex(
        &mut self,
        _vector: <f64 as WasmSimdNum>::VectorType,
        _index: usize,
    ) {
        unimplemented!("Impossible to do a partial store of complex f64's");
    }
}

impl<'a, T: WasmSimdNum> WasmSimdArrayMut<T> for DoubleBuf<'a, T>
where
    Self: WasmSimdArray<T>,
    &'a mut [Complex<T>]: WasmSimdArrayMut<T>,
{
    #[inline(always)]
    unsafe fn store_complex(&mut self, vector: T::VectorType, index: usize) {
        self.output.store_complex(vector, index);
    }
    #[inline(always)]
    unsafe fn store_partial_hi_complex(&mut self, vector: T::VectorType, index: usize) {
        self.output.store_partial_hi_complex(vector, index);
    }
    #[inline(always)]
    unsafe fn store_partial_lo_complex(&mut self, vector: T::VectorType, index: usize) {
        self.output.store_partial_lo_complex(vector, index);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

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
            let load1 = slice.load_complex(0);
            let load2 = slice.load_complex(1);
            let load3 = slice.load_complex(2);
            let load4 = slice.load_complex(3);
            assert_eq!(val1, std::mem::transmute::<v128, Complex<f64>>(load1));
            assert_eq!(val2, std::mem::transmute::<v128, Complex<f64>>(load2));
            assert_eq!(val3, std::mem::transmute::<v128, Complex<f64>>(load3));
            assert_eq!(val4, std::mem::transmute::<v128, Complex<f64>>(load4));
        }
    }

    #[wasm_bindgen_test]
    fn test_store_f64() {
        unsafe {
            let val1: Complex<f64> = Complex::new(1.0, 2.0);
            let val2: Complex<f64> = Complex::new(3.0, 4.0);
            let val3: Complex<f64> = Complex::new(5.0, 6.0);
            let val4: Complex<f64> = Complex::new(7.0, 8.0);

            let nbr1 = v128_load(&val1 as *const _ as *const v128);
            let nbr2 = v128_load(&val2 as *const _ as *const v128);
            let nbr3 = v128_load(&val3 as *const _ as *const v128);
            let nbr4 = v128_load(&val4 as *const _ as *const v128);

            let mut values: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 4];
            let mut slice = values.as_mut_slice();
            slice.store_complex(nbr1, 0);
            slice.store_complex(nbr2, 1);
            slice.store_complex(nbr3, 2);
            slice.store_complex(nbr4, 3);
            assert_eq!(val1, values[0]);
            assert_eq!(val2, values[1]);
            assert_eq!(val3, values[2]);
            assert_eq!(val4, values[3]);
        }
    }
}
