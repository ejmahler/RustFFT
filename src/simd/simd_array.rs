use std::ops::{Deref, DerefMut};

use num_complex::Complex;

use crate::{array_utils::DoubleBuf, simd::simd_vector::SimdVector};

// A trait to handle reading from an array of Complex<T> into SIMD vectors.
// Our SIMD platforms work with 128-bit vectors, meaning a vector can hold two complex f32,
// or a single complex f64.
pub trait SimdComplexArray<V: SimdVector>: Deref {
    // Load complex numbers from the array to fill a SIMD vector.
    unsafe fn load(&self, index: usize) -> V;
    // Load a single complex number from the array into a SIMD vector, setting the unused elements to zero.
    unsafe fn load1_lo(&self, index: usize) -> V;
    // Load a single complex number from the array, and copy it to all elements of a SIMD vector.
    unsafe fn load1_lo_broadcast(&self, index: usize) -> V;
}

impl<V: SimdVector> SimdComplexArray<V> for &[Complex<V::ScalarType>] {
    #[inline(always)]
    unsafe fn load(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + V::COMPLEX_PER_VECTOR);
        V::load_complex(self.as_ptr().add(index))
    }

    #[inline(always)]
    unsafe fn load1_lo(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + 1);
        V::load1_lo_complex(self.as_ptr().add(index))
    }

    #[inline(always)]
    unsafe fn load1_lo_broadcast(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + 1);
        V::load1_lo_broadcast_complex(self.as_ptr().add(index))
    }
}
impl<V: SimdVector> SimdComplexArray<V> for &mut [Complex<V::ScalarType>] {
    #[inline(always)]
    unsafe fn load(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + V::COMPLEX_PER_VECTOR);
        V::load_complex(self.as_ptr().add(index))
    }

    #[inline(always)]
    unsafe fn load1_lo(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + 1);
        V::load1_lo_complex(self.as_ptr().add(index))
    }

    #[inline(always)]
    unsafe fn load1_lo_broadcast(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + 1);
        V::load1_lo_broadcast_complex(self.as_ptr().add(index))
    }
}

impl<'a, V: SimdVector> SimdComplexArray<V> for DoubleBuf<'a, V::ScalarType>
where
    &'a [Complex<V::ScalarType>]: SimdComplexArray<V>,
{
    #[inline(always)]
    unsafe fn load(&self, index: usize) -> V {
        self.input.load(index)
    }
    #[inline(always)]
    unsafe fn load1_lo(&self, index: usize) -> V {
        self.input.load1_lo(index)
    }
    #[inline(always)]
    unsafe fn load1_lo_broadcast(&self, index: usize) -> V {
        self.input.load1_lo_broadcast(index)
    }
}

// A trait to handle writing to an array of Complex<T> from SIMD vectors.
// Our SIMD platforms work with 128-bit vectors, meaning a vector can hold two complex f32,
// or a single complex f64.
pub trait SimdComplexArrayMut<V: SimdVector>: SimdComplexArray<V> + DerefMut {
    // Store all complex numbers from a SIMD vector to the array.
    unsafe fn store(&mut self, vector: V, index: usize);
    // Store the low complex number from a SIMD vector to the array.
    unsafe fn store1_lo(&mut self, vector: V, index: usize);
}

impl<V: SimdVector> SimdComplexArrayMut<V> for &mut [Complex<V::ScalarType>] {
    #[inline(always)]
    unsafe fn store(&mut self, vector: V, index: usize) {
        debug_assert!(self.len() >= index + V::COMPLEX_PER_VECTOR);
        V::store_complex(self.as_mut_ptr().add(index), vector)
    }
    #[inline(always)]
    unsafe fn store1_lo(&mut self, vector: V, index: usize) {
        debug_assert!(self.len() >= index + 1);
        V::store1_lo_complex(self.as_mut_ptr().add(index), vector)
    }
}

impl<'a, V: SimdVector> SimdComplexArrayMut<V> for DoubleBuf<'a, V::ScalarType>
where
    Self: SimdComplexArray<V>,
    &'a mut [Complex<V::ScalarType>]: SimdComplexArrayMut<V>,
{
    #[inline(always)]
    unsafe fn store(&mut self, vector: V, index: usize) {
        self.output.store(vector, index);
    }
    #[inline(always)]
    unsafe fn store1_lo(&mut self, vector: V, index: usize) {
        self.output.store1_lo(vector, index);
    }
}
