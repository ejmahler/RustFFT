//! The NEON side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `neon_vector.rs`. All that is left here is the type alias
//! and the tests.

use crate::simd::simd_radixn::SimdRadixN;

use super::NeonNum;

/// FFT algorithm for lengths that factor into small radixes, NEON accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub type NeonRadixN<N, T> = SimdRadixN<<N as NeonNum>::VectorType, T>;

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radixn::test_bodies;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    #[test]
    fn test_neon_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<float64x2_t>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_neon_radixn_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<float32x4_t>(&[2, 4, 6]);
    }

    #[test]
    fn test_neon_radixn_composite_base() {
        test_bodies::composite_base::<float32x4_t, float64x2_t>();
    }

    #[test]
    fn test_neon_radixn_large_recipes() {
        test_bodies::large_recipes::<float32x4_t, float64x2_t>();
    }

    #[test]
    #[ignore]
    fn test_neon_radixn_six_layers() {
        test_bodies::six_layers::<float32x4_t, float64x2_t>();
    }
}
