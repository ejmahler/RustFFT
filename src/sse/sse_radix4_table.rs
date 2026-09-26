//! The SSE side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `sse_vector.rs`. All that is left here is the type alias
//! and the tests.

use crate::simd::simd_radix4::SimdRadix4;

use super::SseNum;

/// FFT algorithm for lengths that factor into powers of 2, SSE accelerated version.
/// This is designed to be used via a Planner, and not created directly.
#[allow(dead_code)]
pub type SseRadix4Table<S, T> = SimdRadix4<<S as SseNum>::VectorType, T>;

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radix4::test_bodies;
    use std::arch::x86_64::{__m128, __m128d};

    #[test]
    fn test_sse_radix4_replacement_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<__m128d>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_sse_radix4_replacement_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<__m128>(&[2, 4, 6]);
    }

    #[test]
    fn test_sse_radix4_replacement_composite_base() {
        test_bodies::composite_base::<__m128, __m128d>();
    }
}
