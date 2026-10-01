use num_complex::Complex;

use crate::simd::simd_radixn::SimdRadixN;

pub type RadixN<T> = SimdRadixN<Complex<T>, T>;

#[cfg(test)]
mod unit_tests {
    use num_complex::Complex;

    use crate::simd::simd_radixn::test_bodies;

    #[test]
    fn test_scalar_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<Complex<f64>>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_scalar_radixn_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<Complex<f32>>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_scalar_radixn_composite_base() {
        test_bodies::composite_base::<Complex<f32>, Complex<f64>>();
    }

    #[test]
    fn test_scalar_radixn_large_recipes() {
        test_bodies::large_recipes::<Complex<f32>, Complex<f64>>();
    }

    #[test]
    #[ignore]
    fn test_scalar_radixn_six_layers() {
        test_bodies::six_layers::<Complex<f32>, Complex<f64>>();
    }
}
