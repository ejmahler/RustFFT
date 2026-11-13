//! The WASM SIMD side of `SimdRaders`.
//!
//! The algorithm itself lives in `src/simd/simd_raders.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `wasm_simd_vector.rs`. All that is left here is the wrapper
//! type and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_raders::SimdRaders, simd_vector::SimdVector},
};

pub struct WasmSimdRaders<V: SimdVector, T: FftNum>(SimdRaders<V, T>);

impl<V: SimdVector, T: FftNum> WasmSimdRaders<V, T> {
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRaders::new(inner_fft))
    }
}
boilerplate_simd_fft!(WasmSimdRaders, #[target_feature(enable = "simd128")]);

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_raders::test_bodies;
    use crate::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        inner_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(WasmSimdRaders::<V, V::ScalarType>::new(inner_fft))
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_raders_f64() {
        test_bodies::prime_lengths::<WasmVector64>(&new_fn::<WasmVector64>);
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_raders_f32() {
        test_bodies::prime_lengths::<WasmVector32>(&new_fn::<WasmVector32>);
    }
}
