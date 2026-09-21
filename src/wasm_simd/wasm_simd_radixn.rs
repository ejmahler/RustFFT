//! The WASM SIMD side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `wasm_simd_vector.rs`. All that is left here is the type
//! alias and the tests.

use crate::simd::simd_radixn::SimdRadixN;

use super::WasmNum;

/// FFT algorithm for lengths that factor into small radixes, WASM SIMD accelerated version.
/// This is designed to be used via a Planner, and not created directly.
#[allow(dead_code)]
pub type WasmSimdRadixN<S, T> = SimdRadixN<<S as WasmNum>::VectorType, T>;

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radixn::test_bodies;
    use crate::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_wasm_simd_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<WasmVector64>(&[1, 2, 3, 4, 5, 6]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radixn_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<WasmVector32>(&[2, 4, 6]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radixn_composite_base() {
        test_bodies::composite_base::<WasmVector32, WasmVector64>();
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radixn_large_recipes() {
        test_bodies::large_recipes::<WasmVector32, WasmVector64>();
    }

    #[wasm_bindgen_test]
    #[ignore]
    fn test_wasm_simd_radixn_six_layers() {
        test_bodies::six_layers::<WasmVector32, WasmVector64>();
    }
}
