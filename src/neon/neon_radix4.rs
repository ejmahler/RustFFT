//! The NEON side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `neon_vector.rs`. All that is left here is the wrapper type
//! and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_radix4::SimdRadix4, simd_vector::SimdVector},
};
pub struct NeonRadix4<V: SimdVector, T: FftNum>(SimdRadix4<V, T>);
impl<V: SimdVector, T: FftNum> NeonRadix4<V, T> {
    pub fn new(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadix4::new(k, base_fft))
    }
}
boilerplate_simd_fft!(NeonRadix4, #[target_feature(enable = "neon")]);

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radix4::test_bodies;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        k: u32,
        base_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(NeonRadix4::<V, V::ScalarType>::new(k, base_fft))
    }

    #[test]
    fn test_neon_radix4_replacement_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<float64x2_t>(&[1, 2, 3, 4, 5, 6], &new_fn::<float64x2_t>);
    }

    #[test]
    fn test_neon_radix4_replacement_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<float32x4_t>(&[1, 2, 3, 4, 5, 6], &new_fn::<float32x4_t>);
    }

    #[test]
    fn test_neon_radix4_replacement_composite_base() {
        test_bodies::composite_base::<float32x4_t, float64x2_t>(
            &new_fn::<float32x4_t>,
            &new_fn::<float64x2_t>,
        );
    }
}
