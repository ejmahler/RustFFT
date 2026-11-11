//! The WasmSimd side of `SimdStockham`.

use crate::simd::simd_stockham::{SimdStockham, unfused_radixes};
use crate::simd::simd_vector::SimdVector;
use crate::{Fft, FftDirection, FftNum};

pub struct WasmSimdStockham<V: SimdVector, T: FftNum>(SimdStockham<V, T>);

impl<V: SimdVector, T: FftNum> WasmSimdStockham<V, T> {
    pub fn new(len: usize, direction: FftDirection) -> Self {
        Self(SimdStockham::new(len, direction, &unfused_radixes(len)))
    }
}
boilerplate_simd_fft!(WasmSimdStockham, #[target_feature(enable = "simd128")]);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;
    use crate::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};

    #[test]
    fn test_wasm_simd_stockham() {
        for bits in 4..=13 {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let fft = WasmSimdStockham::<WasmVector32, f32>::new(1 << bits, direction);
                check_fft_algorithm::<f32>(&fft, 1 << bits, direction);
                let fft = WasmSimdStockham::<WasmVector64, f64>::new(1 << bits, direction);
                check_fft_algorithm::<f64>(&fft, 1 << bits, direction);
            }
        }
    }
}
