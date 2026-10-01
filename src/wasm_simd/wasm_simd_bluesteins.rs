//! The WASM SIMD side of `SimdBluesteins`.
//!
//! The algorithm itself lives in `src/simd/simd_bluesteins.rs`, shared by every SIMD backend, and
//! the `SimdVector` impls it runs on are in `wasm_simd_vector.rs`. All that is left here is the
//! tests.

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_bluesteins::test_bodies;
    use crate::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_wasm_simd_bluesteins_f64() {
        test_bodies::inner_lengths::<WasmVector64>();
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_bluesteins_f32() {
        // f32 fits two complex per vector, so an odd length means the multiply loops end with a
        // partial load and store
        test_bodies::inner_lengths::<WasmVector32>();
    }
}
