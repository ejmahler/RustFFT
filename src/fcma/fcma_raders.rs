//! The FCMA side of `SimdRaders`.
//!
//! The algorithm itself lives in `src/simd/simd_raders.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `fcma_vector.rs`. All that is left here is the tests.

#[cfg(test)]
mod unit_tests {
    use super::super::fcma_vector::{FcmaVector32, FcmaVector64};
    use crate::simd::simd_raders::test_bodies;

    #[test]
    fn test_fcma_raders_f64() {
        test_bodies::prime_lengths::<FcmaVector64>();
    }

    #[test]
    fn test_fcma_raders_f32() {
        test_bodies::prime_lengths::<FcmaVector32>();
    }
}
