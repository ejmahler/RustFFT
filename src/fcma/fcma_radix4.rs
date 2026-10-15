//! The FCMA side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `fcma_vector.rs`. All that is left here is the wrapper type
//! and tests.

use std::sync::Arc;

use crate::{
    simd::{simd_radix4::SimdRadix4, simd_vector::SimdVector},
    Fft, FftNum,
};

pub struct FcmaRadix4<V: SimdVector, T: FftNum>(SimdRadix4<V, T>);
impl<V: SimdVector, T: FftNum> FcmaRadix4<V, T> {
    pub fn new(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadix4::new(k, base_fft))
    }
}
boilerplate_simd_fft!(FcmaRadix4, #[target_feature(enable = "neon,fcma")]);
#[cfg(test)]
mod unit_tests {
    use super::super::fcma_vector::{FcmaVector32, FcmaVector64};
    use crate::simd::simd_radix4::test_bodies;

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        k: u32,
        base_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(FcmaRadix4::<V, V::ScalarType>::new(k, base_fft))
    }

    #[test]
    fn test_fcma_radix4_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<FcmaVector64>(&[1, 2, 3, 4, 5, 6], &new_fn::<FcmaVector64>);
    }

    #[test]
    fn test_fcma_radix4_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<FcmaVector32>(&[1, 2, 3, 4, 5, 6], &new_fn::<FcmaVector32>);
    }

    #[test]
    fn test_fcma_radix4_composite_base() {
        test_bodies::composite_base::<FcmaVector32, FcmaVector64>(
            &new_fn::<FcmaVector32>,
            &new_fn::<FcmaVector64>,
        );
    }
}
