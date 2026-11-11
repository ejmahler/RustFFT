//! The FCMA side of `SimdBluesteins`.
//!
//! The algorithm itself lives in `src/simd/simd_bluesteins.rs`, shared by every SIMD backend, and
//! the `SimdVector` impls it runs on are in `fcma_vector.rs`. All that is left here is the wrapper
//! type and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_bluesteins::SimdBluesteins, simd_vector::SimdVector},
};

pub struct FcmaBluesteins<V: SimdVector, T: FftNum>(SimdBluesteins<V, T>);

impl<V: SimdVector, T: FftNum> FcmaBluesteins<V, T> {
    pub fn new(len: usize, inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdBluesteins::new(len, inner_fft))
    }
}
boilerplate_simd_fft!(FcmaBluesteins, #[target_feature(enable = "neon,fcma")]);

#[cfg(test)]
mod unit_tests {
    use super::super::fcma_vector::{FcmaVector32, FcmaVector64};
    use crate::simd::simd_bluesteins::test_bodies;

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        len: usize,
        inner_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(FcmaBluesteins::<V, V::ScalarType>::new(len, inner_fft))
    }

    #[test]
    fn test_fcma_bluesteins_f64() {
        test_bodies::inner_lengths::<FcmaVector64>(&new_fn::<FcmaVector64>);
    }

    #[test]
    fn test_fcma_bluesteins_f32() {
        // f32 fits two complex per vector, so an odd length means the multiply loops end with a
        // partial load and store
        test_bodies::inner_lengths::<FcmaVector32>(&new_fn::<FcmaVector32>);
    }
}
