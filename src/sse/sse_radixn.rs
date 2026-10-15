//! The SSE side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `sse_vector.rs`. All that is left here is the wrapper type
//! and tests

use std::sync::Arc;

use crate::{
    common::RadixFactor,
    simd::{simd_radixn::SimdRadixN, simd_vector::SimdVector},
    Fft, FftNum,
};

pub struct SseRadixN<V: SimdVector, T: FftNum>(SimdRadixN<V, T>);

impl<V: SimdVector, T: FftNum> SseRadixN<V, T> {
    #[allow(dead_code)]
    pub fn new(factors: &[RadixFactor], base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadixN::new(factors, base_fft))
    }
}
boilerplate_simd_fft!(SseRadixN, #[inline(always)]);

#[cfg(test)]
mod unit_tests {
    use crate::{common::RadixFactor, simd::simd_radixn::test_bodies, Fft};
    use std::{
        arch::x86_64::{__m128, __m128d},
        sync::Arc,
    };

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        factors: &[RadixFactor],
        base_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(SseRadixN::<V, V::ScalarType>::new(factors, base_fft))
    }

    #[test]
    fn test_sse_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<__m128d>(&[1, 2, 3, 4, 5, 6], &new_fn::<__m128d>);
    }

    #[test]
    fn test_sse_radixn_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<__m128>(&[1, 2, 3, 4, 5, 6], &new_fn::<__m128>);
    }

    #[test]
    fn test_sse_radixn_composite_base() {
        test_bodies::composite_base::<__m128, __m128d>(&new_fn::<__m128>, &new_fn::<__m128d>);
    }

    #[test]
    fn test_sse_radixn_large_recipes() {
        test_bodies::large_recipes::<__m128, __m128d>(&new_fn::<__m128>, &new_fn::<__m128d>);
    }

    #[test]
    #[ignore]
    fn test_sse_radixn_six_layers() {
        test_bodies::six_layers::<__m128, __m128d>(&new_fn::<__m128>, &new_fn::<__m128d>);
    }
}
