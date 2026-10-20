//! The FCMA side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `fcma_vector.rs`.  All that is left here is the wrapper type
//! and tests

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    common::RadixFactor,
    simd::{simd_radixn::SimdRadixN, simd_vector::SimdVector},
};

pub struct FcmaRadixN<V: SimdVector, T: FftNum>(SimdRadixN<V, T>);

impl<V: SimdVector, T: FftNum> FcmaRadixN<V, T> {
    #[allow(dead_code)]
    pub fn new(factors: &[RadixFactor], base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadixN::new(factors, base_fft))
    }
}
boilerplate_simd_fft!(FcmaRadixN, #[target_feature(enable = "neon,fcma")]);

#[cfg(test)]
mod unit_tests {
    use super::super::fcma_vector::{FcmaVector32, FcmaVector64};
    use crate::Fft;
    use crate::common::RadixFactor;
    use crate::simd::simd_radixn::test_bodies;

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        factors: &[RadixFactor],
        base_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(FcmaRadixN::<V, V::ScalarType>::new(factors, base_fft))
    }

    #[test]
    fn test_fcma_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<FcmaVector64>(&[1, 2, 3, 4, 5, 6], &new_fn::<FcmaVector64>);
    }

    #[test]
    fn test_fcma_radixn_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<FcmaVector32>(&[1, 2, 3, 4, 5, 6], &new_fn::<FcmaVector32>);
    }

    #[test]
    fn test_fcma_radixn_composite_base() {
        test_bodies::composite_base::<FcmaVector32, FcmaVector64>(
            &new_fn::<FcmaVector32>,
            &new_fn::<FcmaVector64>,
        );
    }

    #[test]
    fn test_fcma_radixn_large_recipes() {
        test_bodies::large_recipes::<FcmaVector32, FcmaVector64>(
            &new_fn::<FcmaVector32>,
            &new_fn::<FcmaVector64>,
        );
    }

    #[test]
    #[ignore]
    fn test_fcma_radixn_six_layers() {
        test_bodies::six_layers::<FcmaVector32, FcmaVector64>(
            &new_fn::<FcmaVector32>,
            &new_fn::<FcmaVector64>,
        );
    }
}
