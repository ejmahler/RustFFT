use std::sync::Arc;

use num_complex::Complex;

use crate::{common::RadixFactor, simd::simd_radixn::SimdRadixN, Fft, FftNum};

pub struct RadixN<T: FftNum>(SimdRadixN<Complex<T>, T>);

impl<T: FftNum> RadixN<T> {
    pub fn new(factors: &[RadixFactor], base_fft: Arc<dyn Fft<T>>) -> Self {
        Self(SimdRadixN::new(factors, base_fft))
    }
}
boilerplate_scalar_fft!(RadixN, #[inline(always)]);

#[cfg(test)]
mod unit_tests {
    use std::sync::Arc;

    use num_complex::Complex;

    use crate::{simd::simd_radixn::test_bodies, Fft, FftNum};

    use super::*;

    fn new_fn<T: FftNum>(factors: &[RadixFactor], base_fft: Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
        Arc::new(RadixN::new(factors, base_fft))
    }

    #[test]
    fn test_scalar_radixn_f64() {
        // a scalar vector is a single complex number, so every base length is legal
        test_bodies::factor_pairs::<Complex<f64>>(&[1, 2, 3, 4, 5, 6], &new_fn);
    }

    #[test]
    fn test_scalar_radixn_f32() {
        // the scalar vector is one complex number whatever the precision, so f32 is no different
        test_bodies::factor_pairs::<Complex<f32>>(&[1, 2, 3, 4, 5, 6], &new_fn);
    }

    #[test]
    fn test_scalar_radixn_composite_base() {
        test_bodies::composite_base::<Complex<f32>, Complex<f64>>(&new_fn, &new_fn);
    }

    #[test]
    fn test_scalar_radixn_large_recipes() {
        test_bodies::large_recipes::<Complex<f32>, Complex<f64>>(&new_fn, &new_fn);
    }

    #[test]
    #[ignore]
    fn test_scalar_radixn_six_layers() {
        test_bodies::six_layers::<Complex<f32>, Complex<f64>>(&new_fn, &new_fn);
    }
}
