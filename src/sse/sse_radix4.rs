//! The SSE side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd/simd_radixn.rs`, shared by every SIMD backend, and the
//! `SimdVector` impls it runs on are in `sse_vector.rs`. All that is left here is the wrapper type
//! and tests.

use std::sync::Arc;

use crate::{
    Fft, FftDirection, FftNum,
    array_utils::ArcSlice,
    simd::{
        simd_radix4::{Radix4ChildFft, Radix4Layer, SimdRadix4, SimdRadix4Layer},
        simd_vector::SimdVector,
    },
};

struct SseRadix4Layer<V: SimdVector, T: FftNum> {
    simd_layer: SimdRadix4Layer<V, T, SseRadix4Layer<V, T>, V::Butterfly4>,
}
impl<V: SimdVector, T: FftNum> SseRadix4Layer<V, T> {
    pub fn new(
        twiddles: ArcSlice<V>,
        child: Radix4ChildFft<V, T, SseRadix4Layer<V, T>>,
        direction: FftDirection,
    ) -> Self {
        Self {
            simd_layer: SimdRadix4Layer::new(twiddles, child, unsafe {
                V::make_butterfly4(direction)
            }),
        }
    }
}
impl<V: SimdVector, T: FftNum> Radix4Layer<V, T> for SseRadix4Layer<V, T> {
    #[target_feature(enable = "sse4.1")]
    unsafe fn process_layer(
        &self,
        buffer: &mut [num_complex::Complex<T>],
        base_scratch: &mut [num_complex::Complex<T>],
    ) {
        unsafe { self.simd_layer.process_layer(buffer, base_scratch) }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn process_layer_outofplace(
        &self,
        input: &mut [num_complex::Complex<T>],
        output: &mut [num_complex::Complex<T>],
        base_scratch: &mut [num_complex::Complex<T>],
    ) {
        unsafe {
            self.simd_layer
                .process_layer_outofplace(input, output, base_scratch)
        }
    }
}

pub struct SseRadix4<V: SimdVector, T: FftNum>(SimdRadix4<V, T, SseRadix4Layer<V, T>>);
impl<V: SimdVector, T: FftNum> SseRadix4<V, T> {
    pub fn new(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadix4::new(k, base_fft, SseRadix4Layer::new))
    }
}
boilerplate_simd_fft!(SseRadix4, #[target_feature(enable = "sse4.1")]);

#[cfg(test)]
mod unit_tests {
    use crate::simd::simd_radix4::test_bodies;
    use std::arch::x86_64::{__m128, __m128d};

    use super::*;

    fn new_fn<V: SimdVector + 'static>(
        k: u32,
        base_fft: Arc<dyn Fft<V::ScalarType>>,
    ) -> Arc<dyn Fft<V::ScalarType>> {
        Arc::new(SseRadix4::<V, V::ScalarType>::new(k, base_fft))
    }

    #[test]
    fn test_sse_radix4_replacement_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<__m128d>(&[1, 2, 3, 4, 5, 6], &new_fn::<__m128d>);
    }

    #[test]
    fn test_sse_radix4_replacement_f32() {
        // f32 fits two complex per vector, so an odd base length means every layer ends with a
        // partial column
        test_bodies::factor_pairs::<__m128>(&[1, 2, 3, 4, 5, 6], &new_fn::<__m128>);
    }

    #[test]
    fn test_sse_radix4_replacement_composite_base() {
        test_bodies::composite_base::<__m128, __m128d>(&new_fn::<__m128>, &new_fn::<__m128d>);
    }
}
