//! The SSE side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `sse_vector.rs`. All that is left here is the type alias
//! and the tests.

use crate::simd::simd_radixn::SimdRadixN;

use super::SseNum;

/// FFT algorithm for lengths that factor into small radixes, SSE accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub type SseRadixN<S, T> = SimdRadixN<<S as SseNum>::VectorType, T>;

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radixn::test_bodies;
    use std::arch::x86_64::{__m128, __m128d};

    #[test]
    fn test_sse_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<__m128d>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_sse_radixn_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<__m128>(&[2, 4, 6]);
    }

    #[test]
    fn test_sse_radixn_composite_base() {
        test_bodies::composite_base::<__m128, __m128d>();
    }

    #[test]
    fn test_sse_radixn_large_recipes() {
        test_bodies::large_recipes::<__m128, __m128d>();
    }

    #[test]
    #[ignore]
    fn test_sse_radixn_six_layers() {
        test_bodies::six_layers::<__m128, __m128d>();
    }
}
