//! The SSE side of `SimdRaders`.
//!
//! The algorithm itself lives in `src/simd/simd_raders.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `sse_vector.rs`. All that is left here is the tests.

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_raders::test_bodies;
    use std::arch::x86_64::{__m128, __m128d};

    #[test]
    fn test_sse_raders_f64() {
        test_bodies::prime_lengths::<__m128d>();
    }

    #[test]
    fn test_sse_raders_f32() {
        test_bodies::prime_lengths::<__m128>();
    }
}
