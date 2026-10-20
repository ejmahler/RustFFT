//! The WASM SIMD side of `SimdRadix4`.
//!
//! The algorithm itself lives in `src/simd/simd_radix4.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `wasm_simd_vector.rs`. All that is left here is the wrapper type
//! and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_radix4::SimdRadix4, simd_vector::SimdVector},
};

pub struct WasmSimdRadix4<V: SimdVector, T: FftNum>(SimdRadix4<V, T>);
impl<V: SimdVector, T: FftNum> WasmSimdRadix4<V, T> {
    pub fn new(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadix4::new(k, base_fft))
    }
}
boilerplate_simd_fft!(WasmSimdRadix4, #[target_feature(enable = "simd128")]);

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radix4::test_bodies;
    use crate::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        k: u32,
        base_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(WasmSimdRadix4::<V, V::ScalarType>::new(k, base_fft))
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radix4_replacement_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<WasmVector64>(&[1, 2, 3, 4, 5, 6], &new_fn::<WasmVector64>);
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radix4_replacement_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<WasmVector32>(&[1, 2, 3, 4, 5, 6], &new_fn::<WasmVector32>);
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radix4_replacement_composite_base() {
        test_bodies::composite_base::<WasmVector32, WasmVector64>(
            &new_fn::<WasmVector32>,
            &new_fn::<WasmVector64>,
        );
    }
}
