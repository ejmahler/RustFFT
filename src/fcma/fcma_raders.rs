//! The FCMA side of `SimdRaders`.
//!
//! The algorithm itself lives in `src/simd/simd_raders.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `fcma_vector.rs`. All that is left here is the wrapper type
//! and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_raders::SimdRaders, simd_vector::SimdVector},
};

pub struct FcmaRaders<V: SimdVector, T: FftNum>(SimdRaders<V, T>);

impl<V: SimdVector, T: FftNum> FcmaRaders<V, T> {
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRaders::new(inner_fft))
    }
}
boilerplate_simd_fft!(FcmaRaders, #[target_feature(enable = "neon,fcma")]);

#[cfg(test)]
mod unit_tests {
    use super::super::fcma_vector::{FcmaVector32, FcmaVector64};
    use crate::simd::simd_raders::test_bodies;

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        inner_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(FcmaRaders::<V, V::ScalarType>::new(inner_fft))
    }

    #[test]
    fn test_fcma_raders_f64() {
        test_bodies::prime_lengths::<FcmaVector64>(&new_fn::<FcmaVector64>);
    }

    #[test]
    fn test_fcma_raders_f32() {
        test_bodies::prime_lengths::<FcmaVector32>(&new_fn::<FcmaVector32>);
    }
}
