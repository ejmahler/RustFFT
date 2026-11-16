//! The NEON side of `SimdStockham`.

use crate::simd::simd_stockham::{SimdStockham, unfused_radixes};
use crate::simd::simd_vector::SimdVector;
use crate::{Fft, FftDirection, FftNum};

pub struct SseStockham<V: SimdVector, T: FftNum>(SimdStockham<V, T>);

impl<V: SimdVector, T: FftNum> SseStockham<V, T> {
    #[allow(dead_code)]
    pub fn new(len: usize, direction: FftDirection) -> Self {
        Self(SimdStockham::new(
            len,
            direction,
            &unfused_radixes(len))
        )
    }
}
boilerplate_simd_fft!(SseStockham, #[target_feature(enable = "sse4.1")]);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;
    use std::arch::x86_64::{__m128, __m128d};

    #[test]
    fn test_sse_stockham() {
        for bits in 4..=13 {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let fft = SseStockham::<__m128, f32>::new(1 << bits, direction);
                check_fft_algorithm::<f32>(&fft, 1 << bits, direction);
                let fft = SseStockham::<__m128d, f64>::new(1 << bits, direction);
                check_fft_algorithm::<f64>(&fft, 1 << bits, direction);
            }
        }
    }
}
