use core::arch::x86_64::*;
use num_complex::Complex;

use crate::array_utils::{RawSlice, RawSliceMut};


// Read these indexes from an SseArray and build an array of simd vectors.
// Takes a name of a vector to read from, and a list of indexes to read.
// This statement:
// ```
// let values = read_complex_to_array(input, {0, 1, 2, 3});
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
// let values = read_partial1_complex_to_array(input, {0, 1, 2, 3});
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

// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
macro_rules! interleave_complex_f32 {
    ($input:ident, $offset:literal, { $($idx:literal),* }) => {
        [
        $(
            pack_1st_f32($input[$idx], $input[$idx+$offset]),
            pack_2nd_f32($input[$idx], $input[$idx+$offset]),
        )*
        ]
    }
}


pub trait SseArray {
    type VectorType;
    const COMPLEX_PER_VECTOR: usize;
    unsafe fn load_complex(&self, index: usize) -> Self::VectorType;
    unsafe fn load_partial1_complex(
        &self,
        index: usize,
    ) -> Self::VectorType;
    unsafe fn load1_complex(
        &self,
        index: usize,
    ) -> Self::VectorType;
}

impl SseArray for RawSlice<Complex<f32>> {
    type VectorType = __m128;
    const COMPLEX_PER_VECTOR: usize = 2;
    
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> Self::VectorType {
        debug_assert!(self.len() >= index + Self::COMPLEX_PER_VECTOR);
        _mm_loadu_ps(self.as_ptr().add(index) as *const f32)
    }

    #[inline(always)]
    unsafe fn load_partial1_complex(
        &self,
        index: usize,
    ) -> Self::VectorType {
        debug_assert!(self.len() >= index + 1);
        _mm_castpd_ps(_mm_load_sd(self.as_ptr().add(index) as *const f64))
    }

    #[inline(always)]
    unsafe fn load1_complex(
        &self,
        index: usize,
    ) -> Self::VectorType {
        debug_assert!(self.len() >= index + 1);
        _mm_castpd_ps(_mm_load1_pd(self.as_ptr().add(index) as *const f64))
    }
}

impl SseArray for RawSlice<Complex<f64>> {
    type VectorType = __m128d;
    const COMPLEX_PER_VECTOR: usize = 1;
    
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> Self::VectorType {
        debug_assert!(self.len() >= index + Self::COMPLEX_PER_VECTOR);
        _mm_loadu_pd(self.as_ptr().add(index) as *const f64)
    }

    #[inline(always)]
    unsafe fn load_partial1_complex(
        &self,
        _index: usize,
    ) -> Self::VectorType {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }

    #[inline(always)]
    unsafe fn load1_complex(
        &self,
        _index: usize,
    ) -> Self::VectorType {
        unimplemented!("Impossible to do a partial load of complex f64's");
    }
}


pub trait SseArrayMut {
    type VectorType;
    const COMPLEX_PER_VECTOR: usize;
    unsafe fn store_complex(&self, vector: Self::VectorType, index: usize);
    unsafe fn store_partial_lo_complex(
        &self,
        vector: Self::VectorType,
        index: usize,
    );
    unsafe fn store_partial_hi_complex(
        &self,
        vector: Self::VectorType,
        index: usize,
    );
}

impl SseArrayMut for RawSliceMut<Complex<f32>> {
    type VectorType = __m128;
    const COMPLEX_PER_VECTOR: usize = 2;
    
    #[inline(always)]
    unsafe fn store_complex(&self, vector: Self::VectorType, index: usize) {
        debug_assert!(self.len() >= index + Self::COMPLEX_PER_VECTOR);
        _mm_storeu_ps(self.as_mut_ptr().add(index) as *mut f32, vector);
    }

    #[inline(always)]
    unsafe fn store_partial_hi_complex(
        &self,
        vector: Self::VectorType,
        index: usize,
    ) {
        debug_assert!(self.len() >= index + 1);
        _mm_storeh_pd(self.as_mut_ptr().add(index) as *mut f64, _mm_castps_pd(vector));
    }
    #[inline(always)]
    unsafe fn store_partial_lo_complex(
        &self,
        vector: Self::VectorType,
        index: usize,
    ) {
        debug_assert!(self.len() >= index + 1);
        _mm_storel_pd(self.as_mut_ptr().add(index) as *mut f64, _mm_castps_pd(vector));
    }
}

impl SseArrayMut for RawSliceMut<Complex<f64>> {
    type VectorType = __m128d;
    const COMPLEX_PER_VECTOR: usize = 1;
    
    #[inline(always)]
    unsafe fn store_complex(&self, vector: Self::VectorType, index: usize) {
        debug_assert!(self.len() >= index + Self::COMPLEX_PER_VECTOR);
        _mm_storeu_pd(self.as_mut_ptr().add(index) as *mut f64, vector);
    }

    #[inline(always)]
    unsafe fn store_partial_hi_complex(
        &self,
        _vector: Self::VectorType,
        _index: usize,
    ) {
        unimplemented!("Impossible to do a partial store of complex f64's");
    }
    #[inline(always)]
    unsafe fn store_partial_lo_complex(
        &self,
        _vector: Self::VectorType,
        _index: usize,
    ) {
        unimplemented!("Impossible to do a partial store of complex f64's");
    }
}


// RawSlice<Complex<f32>>

/*
impl<T: FftNum> SseArray<T> for RawSlice<Complex<f32>> {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> T::VectorType {
        debug_assert!(self.len() >= index + T::COMPLEX_PER_VECTOR);
        T::VectorType::load_complex(self.as_ptr().add(index))
    }
    #[inline(always)]
    unsafe fn load_partial1_complex(
        &self,
        index: usize,
    ) -> <T::VectorType as AvxVector256>::HalfVector {
        debug_assert!(self.len() >= index + 1);
        T::VectorType::load_partial1_complex(self.as_ptr().add(index))
    }
}
impl<T: AvxNum> AvxArray<T> for RawSlice<Complex<T>> {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> T::VectorType {
        debug_assert!(self.len() >= index + T::VectorType::COMPLEX_PER_VECTOR);
        T::VectorType::load_complex(self.as_ptr().add(index))
    }
    #[inline(always)]
    unsafe fn load_partial1_complex(
        &self,
        index: usize,
    ) -> <T::VectorType as AvxVector256>::HalfVector {
        debug_assert!(self.len() >= index + 1);
        T::VectorType::load_partial1_complex(self.as_ptr().add(index))
    }
}

impl<T: AvxNum> AvxArrayMut<T> for [Complex<T>] {
    #[inline(always)]
    unsafe fn store_complex(&mut self, data: T::VectorType, index: usize) {
        debug_assert!(self.len() >= index + T::VectorType::COMPLEX_PER_VECTOR);
        T::VectorType::store_complex(self.as_mut_ptr().add(index), data);
    }
    #[inline(always)]
    unsafe fn store_partial1_complex(
        &mut self,
        data: <T::VectorType as AvxVector256>::HalfVector,
        index: usize,
    ) {
        debug_assert!(self.len() >= index + 1);
        T::VectorType::store_partial1_complex(self.as_mut_ptr().add(index), data)
    }
}
impl<T: AvxNum> AvxArrayMut<T> for RawSliceMut<Complex<T>> {
    #[inline(always)]
    unsafe fn store_complex(&mut self, data: T::VectorType, index: usize) {
        debug_assert!(self.len() >= index + T::VectorType::COMPLEX_PER_VECTOR);
        T::VectorType::store_complex(self.as_mut_ptr().add(index), data);
    }
    #[inline(always)]
    unsafe fn store_partial1_complex(
        &mut self,
        data: <T::VectorType as AvxVector256>::HalfVector,
        index: usize,
    ) {
        debug_assert!(self.len() >= index + 1);
        T::VectorType::store_partial1_complex(self.as_mut_ptr().add(index), data)
    }
}
*/

/*
    #[inline(always)]
    unsafe fn load_partial1_complex(ptr: *const Complex<Self::ScalarType>) -> Self::HalfVector {
        let data = _mm_load_sd(ptr as *const f64);
        _mm_castpd_ps(data)
    }
    #[inline(always)]
    unsafe fn load_partial2_complex(ptr: *const Complex<Self::ScalarType>) -> Self::HalfVector {
        _mm_loadu_ps(ptr as *const f32)
    }
    #[inline(always)]
    unsafe fn load_partial3_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        let lo = Self::load_partial2_complex(ptr);
        let hi = Self::load_partial1_complex(ptr.add(2));
        Self::merge(lo, hi)
    }
    #[inline(always)]
    unsafe fn store_partial1_complex(ptr: *mut Complex<Self::ScalarType>, data: Self::HalfVector) {
        _mm_store_sd(ptr as *mut f64, _mm_castps_pd(data));
    }
    #[inline(always)]
    unsafe fn store_partial2_complex(ptr: *mut Complex<Self::ScalarType>, data: Self::HalfVector) {
        _mm_storeu_ps(ptr as *mut f32, data);
    }
    #[inline(always)]
    unsafe fn store_partial3_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        Self::store_partial2_complex(ptr, data.lo());
        Self::store_partial1_complex(ptr.add(2), data.hi());
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm256_loadu_ps(ptr as *const Self::ScalarType)
    }
    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm256_storeu_ps(ptr as *mut Self::ScalarType, data)
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm256_loadu_pd(ptr as *const Self::ScalarType)
    }
    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm256_storeu_pd(ptr as *mut Self::ScalarType, data)
    }

    */
