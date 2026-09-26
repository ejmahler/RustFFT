//! The WASM SIMD side of `SimdRadix4`.
//!
//! The algorithm itself lives in `src/simd/simd_radix4.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `wasm_simd_vector.rs`. All that is left here is the type alias
//! and the tests.

use crate::simd::simd_radix4::SimdRadix4;

use super::WasmNum;

/// FFT algorithm for lengths that factor into powers of 2, Wasm SIMD accelerated version.
/// This is designed to be used via a Planner, and not created directly.
#[allow(dead_code)]
pub type WasmSimdRadix4Table<S, T> = SimdRadix4<<S as WasmNum>::VectorType, T>;

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radix4::test_bodies;
    use crate::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_wasm_simd_radix4_replacement_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<WasmVector64>(&[1, 2, 3, 4, 5, 6]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radix4_replacement_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<WasmVector32>(&[2, 4, 6]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radix4_replacement_composite_base() {
        test_bodies::composite_base::<WasmVector32, WasmVector64>();
    }
}
