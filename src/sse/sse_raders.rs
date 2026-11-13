//! The SSE side of `SimdRaders`.
//!
//! The algorithm itself lives in `src/simd/simd_raders.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `sse_vector.rs`. All that is left here is the wrapper type
//! and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_raders::SimdRaders, simd_vector::SimdVector},
};

pub struct SseRaders<V: SimdVector, T: FftNum>(SimdRaders<V, T>);

impl<V: SimdVector, T: FftNum> SseRaders<V, T> {
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRaders::new(inner_fft))
    }
}
boilerplate_simd_fft!(SseRaders, #[target_feature(enable = "sse4.1")]);

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_raders::test_bodies;
    use std::arch::x86_64::{__m128, __m128d};

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        inner_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(SseRaders::<V, V::ScalarType>::new(inner_fft))
    }

    #[test]
    fn test_sse_raders_f64() {
        test_bodies::prime_lengths::<__m128d>(&new_fn::<__m128d>);
    }

    #[test]
    fn test_sse_raders_f32() {
        test_bodies::prime_lengths::<__m128>(&new_fn::<__m128>);
    }
}
