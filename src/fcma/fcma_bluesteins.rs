//! The FCMA side of `SimdBluesteins`.
//!
//! The algorithm itself lives in `src/simd/simd_bluesteins.rs`, shared by every SIMD backend, and
//! the `SimdVector` impls it runs on are in `fcma_vector.rs`. All that is left here is the tests.

#[cfg(test)]
mod unit_tests {
    use super::super::fcma_vector::{FcmaVector32, FcmaVector64};
    use crate::simd::simd_bluesteins::test_bodies;

    #[test]
    fn test_fcma_bluesteins_f64() {
        test_bodies::inner_lengths::<FcmaVector64>();
    }

    #[test]
    fn test_fcma_bluesteins_f32() {
        // f32 fits two complex per vector, so an odd length means the multiply loops end with a
        // partial load and store
        test_bodies::inner_lengths::<FcmaVector32>();
    }
}
