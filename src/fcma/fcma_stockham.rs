//! The FCMA side of `SimdStockham`.

use crate::simd::simd_stockham::{SimdStockham, fused_radixes};
use crate::simd::simd_vector::SimdVector;
use crate::{Fft, FftDirection, FftNum};

pub struct FcmaStockham<V: SimdVector, T: FftNum>(SimdStockham<V, T>);

impl<V: SimdVector, T: FftNum> FcmaStockham<V, T> {
    pub fn new(len: usize, direction: FftDirection) -> Self {
        Self(SimdStockham::new(
            len,
            direction,
            &fused_radixes(len, V::COMPLEX_PER_VECTOR),
        ))
    }
}
boilerplate_simd_fft!(FcmaStockham, #[target_feature(enable = "neon,fcma")]);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::fcma::fcma_vector::{FcmaVector32, FcmaVector64};
    use crate::test_utils::check_fft_algorithm;

    #[test]
    fn test_fcma_stockham() {
        for bits in 4..=13 {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let fft = FcmaStockham::<FcmaVector32, f32>::new(1 << bits, direction);
                check_fft_algorithm::<f32>(&fft, 1 << bits, direction);
                let fft = FcmaStockham::<FcmaVector64, f64>::new(1 << bits, direction);
                check_fft_algorithm::<f64>(&fft, 1 << bits, direction);
            }
        }
    }
}
