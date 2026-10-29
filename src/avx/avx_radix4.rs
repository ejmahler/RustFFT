//! The AVX side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `sse_vector.rs`. All that is left here is the wrapper type
//! and tests.

use std::sync::Arc;

use crate::{
    Fft, FftNum,
    simd::{simd_radix4::SimdRadix4, simd_vector::SimdVector},
};
#[allow(dead_code)]
pub struct AvxRadix4<V: SimdVector, T: FftNum>(SimdRadix4<V, T>);
impl<V: SimdVector, T: FftNum> AvxRadix4<V, T> {
    #[allow(dead_code)]
    pub fn new(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadix4::new(k, base_fft))
    }
}
boilerplate_simd_fft!(AvxRadix4, #[target_feature(enable = "avx,fma")]);

#[cfg(test)]
mod unit_tests {
    use crate::{simd::simd_radix4::test_bodies, test_utils::BigScratchAlgorithm};
    use std::arch::x86_64::{__m256, __m256d};

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        k: u32,
        base_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(AvxRadix4::<V, V::ScalarType>::new(k, base_fft))
    }

    #[test]
    fn test_avx_radix4_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<__m256d>(&[1, 2, 3, 4, 5, 6], &new_fn::<__m256d>);
    }

    #[test]
    fn test_avx_radix4_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<__m256>(&[1, 2, 3, 4, 5, 6], &new_fn::<__m256>);
    }

    #[test]
    fn test_avx_radix4_composite_base() {
        test_bodies::composite_base::<__m256, __m256d>(&new_fn::<__m256>, &new_fn::<__m256d>);
    }

    #[test]
    fn test_avx_radix4_cross() {
        let base_fft = Arc::new(BigScratchAlgorithm{ len: 32, inplace_scratch: 0, outofplace_scratch: 0, immut_scratch: 0, direction: crate::FftDirection::Forward });
        let _radix4 = AvxRadix4::<__m256, f32>::new(10, base_fft);
    }
}
