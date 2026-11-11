//! The NEON side of `SimdBluesteins`.
//!
//! The algorithm itself lives in `src/simd/simd_bluesteins.rs`, shared by every SIMD backend, and
//! the `SimdVector` impls it runs on are in `neon_vector.rs`. All that is left here is the wrapper
//! type and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_bluesteins::SimdBluesteins, simd_vector::SimdVector},
};

pub struct NeonBluesteins<V: SimdVector, T: FftNum>(SimdBluesteins<V, T>);

impl<V: SimdVector, T: FftNum> NeonBluesteins<V, T> {
    pub fn new(len: usize, inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdBluesteins::new(len, inner_fft))
    }
}
boilerplate_simd_fft!(NeonBluesteins, #[inline(always)]);

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_bluesteins::test_bodies;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        len: usize,
        inner_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(NeonBluesteins::<V, V::ScalarType>::new(len, inner_fft))
    }

    #[test]
    fn test_neon_bluesteins_f64() {
        test_bodies::inner_lengths::<float64x2_t>(&new_fn::<float64x2_t>);
    }

    #[test]
    fn test_neon_bluesteins_f32() {
        // f32 fits two complex per vector, so an odd length means the multiply loops end with a
        // partial load and store
        test_bodies::inner_lengths::<float32x4_t>(&new_fn::<float32x4_t>);
    }
}
