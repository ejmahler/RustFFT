use std::arch::x86_64::*;
use num_complex::Complex;

pub trait AvxComplexArrayf32 {
    unsafe fn load_complex_f32(&self, index: usize) -> __m256;
    unsafe fn store_complex_f32(&mut self, index: usize, data: __m256);
}

impl AvxComplexArrayf32 for [Complex<f32>] {
    #[inline(always)]
    unsafe fn load_complex_f32(&self, index: usize) -> __m256 {
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f32;
        _mm256_loadu_ps(float_ptr)
    }
    #[inline(always)]
    unsafe fn store_complex_f32(&mut self, index: usize, data: __m256) {
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        _mm256_storeu_ps(float_ptr, data);
    }
}

#[inline(always)]
pub unsafe fn broadcast_complex_f32(value: Complex<f32>) -> __m256 {
    _mm256_set_ps(value.im, value.re, value.im, value.re, value.im, value.re, value.im, value.re)
}