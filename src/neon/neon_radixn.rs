//! The NEON side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `neon_vector.rs`. All that is left here is the wrapper type
//! and tests

use std::sync::Arc;

use num_complex::Complex;

use crate::{
    Fft, FftDirection, FftNum,
    array_utils::ArcSlice,
    simd::{
        simd_radixn::{
            cross_layer::{
                CrossLayer, LayerFactory, LayerInstance, boilerplate_simd_radixn,
                boilerplate_simd_radixn_layer,
            },
            radix_factor::RadixFactor,
            radixn::SimdRadixN,
        },
        simd_vector::SimdVector,
    },
};

pub struct NeonRadixN<V: SimdVector, T: FftNum>(SimdRadixN<V, T>);

impl<V: SimdVector, T: FftNum> NeonRadixN<V, T> {
    #[allow(dead_code)]
    #[target_feature(enable = "neon")]
    pub fn new(factors: &[RadixFactor], base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadixN::new(factors, base_fft, NeonLayerFactory))
    }
}
boilerplate_simd_fft!(NeonRadixN, #[target_feature(enable = "neon")]);
boilerplate_simd_radixn!(NeonLayerFactory, #[target_feature(enable = "neon")]);

#[cfg(test)]
mod unit_tests {
    use crate::Fft;
    use crate::simd::simd_radixn::radix_factor::RadixFactor;
    use crate::simd::simd_radixn::radixn::test_bodies;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        factors: &[RadixFactor],
        base_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        unsafe { Arc::new(NeonRadixN::<V, V::ScalarType>::new(factors, base_fft)) }
    }

    #[test]
    fn test_neon_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<float64x2_t>(&[1, 2, 3, 4, 5, 6], &new_fn::<float64x2_t>);
    }

    #[test]
    fn test_neon_radixn_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<float32x4_t>(&[1, 2, 3, 4, 5, 6], &new_fn::<float32x4_t>);
    }

    #[test]
    fn test_neon_radixn_composite_base() {
        test_bodies::composite_base::<float32x4_t, float64x2_t>(
            &new_fn::<float32x4_t>,
            &new_fn::<float64x2_t>,
        );
    }

    #[test]
    fn test_neon_radixn_large_recipes() {
        test_bodies::large_recipes::<float32x4_t, float64x2_t>(
            &new_fn::<float32x4_t>,
            &new_fn::<float64x2_t>,
        );
    }

    #[test]
    #[ignore]
    fn test_neon_radixn_six_layers() {
        test_bodies::six_layers::<float32x4_t, float64x2_t>(
            &new_fn::<float32x4_t>,
            &new_fn::<float64x2_t>,
        );
    }
}
