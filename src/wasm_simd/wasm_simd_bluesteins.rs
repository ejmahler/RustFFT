//! The WASM SIMD side of `SimdBluesteins`.
//!
//! The algorithm itself lives in `src/simd/simd_bluesteins.rs`, shared by every SIMD backend, and
//! the `SimdVector` impls it runs on are in `wasm_simd_vector.rs`. All that is left here is the
//! wrapper type and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_bluesteins::SimdBluesteins, simd_vector::SimdVector},
};

pub struct WasmSimdBluesteins<V: SimdVector, T: FftNum>(SimdBluesteins<V, T>);

impl<V: SimdVector, T: FftNum> WasmSimdBluesteins<V, T> {
    pub fn new(len: usize, inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdBluesteins::new(len, inner_fft))
    }
}
boilerplate_simd_fft!(WasmSimdBluesteins, #[target_feature(enable = "simd128")]);

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_bluesteins::test_bodies;
    use crate::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        len: usize,
        inner_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(WasmSimdBluesteins::<V, V::ScalarType>::new(len, inner_fft))
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_bluesteins_f64() {
        test_bodies::inner_lengths::<WasmVector64>(&new_fn::<WasmVector64>);
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_bluesteins_f32() {
        // f32 fits two complex per vector, so an odd length means the multiply loops end with a
        // partial load and store
        test_bodies::inner_lengths::<WasmVector32>(&new_fn::<WasmVector32>);
    }
}
