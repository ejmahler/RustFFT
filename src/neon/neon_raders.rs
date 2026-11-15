//! The NEON side of `SimdRaders`.
//!
//! The algorithm itself lives in `src/simd/simd_raders.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `neon_vector.rs`. All that is left here is the wrapper type
//! and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_raders::SimdRaders, simd_vector::SimdVector},
};

pub struct NeonRaders<V: SimdVector, T: FftNum>(SimdRaders<V, T>);

impl<V: SimdVector, T: FftNum> NeonRaders<V, T> {
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRaders::new(inner_fft))
    }
}
boilerplate_simd_fft!(NeonRaders, #[target_feature(enable = "neon")]);

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_raders::test_bodies;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        inner_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(NeonRaders::<V, V::ScalarType>::new(inner_fft))
    }

    #[test]
    fn test_neon_raders_f64() {
        test_bodies::prime_lengths::<float64x2_t>(&new_fn::<float64x2_t>);
    }

    #[test]
    fn test_neon_raders_f32() {
        test_bodies::prime_lengths::<float32x4_t>(&new_fn::<float32x4_t>);
    }
}
