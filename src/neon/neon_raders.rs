//! The NEON side of `SimdRaders`.
//!
//! The algorithm itself lives in `src/simd/simd_raders.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `neon_vector.rs`. All that is left here is the tests.

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_raders::test_bodies;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    #[test]
    fn test_neon_raders_f64() {
        test_bodies::prime_lengths::<float64x2_t>();
    }

    #[test]
    fn test_neon_raders_f32() {
        test_bodies::prime_lengths::<float32x4_t>();
    }
}
