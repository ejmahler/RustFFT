//! The FCMA side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `fcma_vector.rs`. All that is left here is the type alias
//! and the tests.

use crate::simd::simd_radixn::SimdRadixN;

use super::FcmaNum;

/// FFT algorithm for lengths that factor into small radixes, FCMA accelerated version.
/// This is designed to be used via a Planner, and not created directly.
#[allow(dead_code)]
pub type FcmaRadixN<N, T> = SimdRadixN<<N as FcmaNum>::SimdVectorType, T>;

#[cfg(test)]
mod unit_tests {
    use super::super::fcma_vector::{FcmaSimdVector32, FcmaSimdVector64};
    use crate::simd::simd_radixn::test_bodies;

    #[test]
    fn test_fcma_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<FcmaSimdVector64>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_fcma_radixn_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<FcmaSimdVector32>(&[2, 4, 6]);
    }

    #[test]
    fn test_fcma_radixn_composite_base() {
        test_bodies::composite_base::<FcmaSimdVector32, FcmaSimdVector64>();
    }

    #[test]
    fn test_fcma_radixn_large_recipes() {
        test_bodies::large_recipes::<FcmaSimdVector32, FcmaSimdVector64>();
    }

    #[test]
    #[ignore]
    fn test_fcma_radixn_six_layers() {
        test_bodies::six_layers::<FcmaSimdVector32, FcmaSimdVector64>();
    }
}
