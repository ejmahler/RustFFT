//! The NEON side of `SimdBluesteins`.
//!
//! The algorithm itself lives in `src/simd/simd_bluesteins.rs`, shared by every SIMD backend, and
//! the `SimdVector` impls it runs on are in `neon_vector.rs`. All that is left here is the tests.

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_bluesteins::test_bodies;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    #[test]
    fn test_neon_bluesteins_f64() {
        test_bodies::inner_lengths::<float64x2_t>();
    }

    #[test]
    fn test_neon_bluesteins_f32() {
        // f32 fits two complex per vector, so an odd length means the multiply loops end with a
        // partial load and store
        test_bodies::inner_lengths::<float32x4_t>();
    }
}
