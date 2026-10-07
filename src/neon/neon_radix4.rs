//! The NEON side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `neon_vector.rs`. All that is left here is he tests.

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radix4::test_bodies;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    #[test]
    fn test_neon_radix4_replacement_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<float64x2_t>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_neon_radix4_replacement_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<float32x4_t>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_neon_radix4_replacement_composite_base() {
        test_bodies::composite_base::<float32x4_t, float64x2_t>();
    }
}
