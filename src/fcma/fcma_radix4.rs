//! The FCMA side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `fcma_vector.rs`. All that is left here is the tests.

#[cfg(test)]
mod unit_tests {
    use super::super::fcma_vector::{FcmaVector32, FcmaVector64};
    use crate::simd::simd_radix4::test_bodies;

    #[test]
    fn test_fcma_radix4_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<FcmaVector64>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_fcma_radix4_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<FcmaVector32>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_fcma_radix4_composite_base() {
        test_bodies::composite_base::<FcmaVector32, FcmaVector64>();
    }
}
