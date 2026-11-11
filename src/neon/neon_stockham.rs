//! The NEON side of `SimdStockham`.

use crate::simd::simd_stockham::{SimdStockham, neon_radixes};
use crate::simd::simd_vector::SimdVector;
use crate::{Fft, FftDirection, FftNum};

pub struct NeonStockham<V: SimdVector, T: FftNum>(SimdStockham<V, T>);

impl<V: SimdVector, T: FftNum> NeonStockham<V, T> {
    pub fn new(len: usize, direction: FftDirection) -> Self {
        Self(SimdStockham::new(
            len,
            direction,
            &neon_radixes(len, V::COMPLEX_PER_VECTOR),
        ))
    }
}
boilerplate_simd_fft!(NeonStockham, #[inline(always)]);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;
    use std::arch::aarch64::{float32x4_t, float64x2_t};

    #[test]
    fn test_neon_stockham() {
        for bits in 4..=13 {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let fft = NeonStockham::<float32x4_t, f32>::new(1 << bits, direction);
                check_fft_algorithm::<f32>(&fft, 1 << bits, direction);
                let fft = NeonStockham::<float64x2_t, f64>::new(1 << bits, direction);
                check_fft_algorithm::<f64>(&fft, 1 << bits, direction);
            }
        }
    }
}
