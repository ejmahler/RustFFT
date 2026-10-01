//! The WASM SIMD side of `SimdRaders`.
//!
//! The algorithm itself lives in `src/simd/simd_raders.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `wasm_simd_vector.rs`. All that is left here is the tests.

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_raders::test_bodies;
    use crate::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_wasm_simd_raders_f64() {
        test_bodies::prime_lengths::<WasmVector64>();
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_raders_f32() {
        test_bodies::prime_lengths::<WasmVector32>();
    }
}
