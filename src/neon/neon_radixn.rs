//! The NEON side of `SimdRadixN`.
//!
//! The algorithm itself lives in `src/simd_radixn.rs`, shared by every SIMD backend. All that is
//! left here is the `RadixNVector` impl for each vector type: the NEON loads, stores and vector
//! math, and the element-type-specific butterfly structs for radix 3, 5, 6 and 7.

use std::arch::aarch64::{float32x4_t, float64x2_t};

use num_complex::Complex;

use crate::simd_radixn::{RadixNVector, SimdRadixN};
use crate::FftDirection;

use super::neon_butterflies::{
    NeonF32Butterfly3, NeonF32Butterfly5, NeonF32Butterfly6, NeonF64Butterfly3, NeonF64Butterfly5,
    NeonF64Butterfly6,
};
use super::neon_prime_butterflies::{NeonF32Butterfly7, NeonF64Butterfly7};
use super::neon_vector::{NeonArray, NeonArrayMut, NeonVector, Rotation90};
use super::NeonNum;

/// FFT algorithm for lengths that factor into small radixes, NEON accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub type NeonRadixN<N, T> = SimdRadixN<<N as NeonNum>::VectorType, T>;

impl RadixNVector for float64x2_t {
    const COMPLEX_PER_VECTOR: usize = 1;

    type ScalarType = f64;
    type Rotation = Rotation90<Self>;

    type Butterfly3 = NeonF64Butterfly3<f64>;
    type Butterfly5 = NeonF64Butterfly5<f64>;
    type Butterfly6 = NeonF64Butterfly6<f64>;
    type Butterfly7 = NeonF64Butterfly7<f64>;

    #[inline(always)]
    unsafe fn load(data: &[Complex<f64>], index: usize) -> Self {
        data.load_complex(index)
    }
    #[inline(always)]
    unsafe fn store(mut data: &mut [Complex<f64>], value: Self, index: usize) {
        data.store_complex(value, index)
    }

    #[inline(always)]
    unsafe fn mul_complex(left: Self, right: Self) -> Self {
        NeonVector::mul_complex(left, right)
    }
    #[inline(always)]
    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: FftDirection,
    ) -> Self {
        NeonVector::make_mixedradix_twiddle_chunk(x, y, len, direction)
    }

    #[inline(always)]
    unsafe fn make_rotate90(direction: FftDirection) -> Self::Rotation {
        NeonVector::make_rotate90(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        NeonF64Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        NeonF64Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        NeonF64Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        NeonF64Butterfly7::new(direction)
    }

    #[inline(always)]
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        NeonVector::column_butterfly2(rows)
    }
    #[inline(always)]
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        bf.perform_fft_direct(rows[0], rows[1], rows[2])
    }
    #[inline(always)]
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Self::Rotation) -> [Self; 4] {
        NeonVector::column_butterfly4(rows, rotation)
    }
    #[inline(always)]
    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5] {
        bf.perform_fft_direct(rows[0], rows[1], rows[2], rows[3], rows[4])
    }
    #[inline(always)]
    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6] {
        bf.perform_fft_direct(rows)
    }
    #[inline(always)]
    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7] {
        bf.perform_fft_direct(rows)
    }

    neon_radixn_fft_helpers!();
}

impl RadixNVector for float32x4_t {
    const COMPLEX_PER_VECTOR: usize = 2;

    type ScalarType = f32;
    type Rotation = Rotation90<Self>;

    type Butterfly3 = NeonF32Butterfly3<f32>;
    type Butterfly5 = NeonF32Butterfly5<f32>;
    type Butterfly6 = NeonF32Butterfly6<f32>;
    type Butterfly7 = NeonF32Butterfly7<f32>;

    #[inline(always)]
    unsafe fn load(data: &[Complex<f32>], index: usize) -> Self {
        data.load_complex(index)
    }
    #[inline(always)]
    unsafe fn store(mut data: &mut [Complex<f32>], value: Self, index: usize) {
        data.store_complex(value, index)
    }

    #[inline(always)]
    unsafe fn mul_complex(left: Self, right: Self) -> Self {
        NeonVector::mul_complex(left, right)
    }
    #[inline(always)]
    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: FftDirection,
    ) -> Self {
        NeonVector::make_mixedradix_twiddle_chunk(x, y, len, direction)
    }

    #[inline(always)]
    unsafe fn make_rotate90(direction: FftDirection) -> Self::Rotation {
        NeonVector::make_rotate90(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        NeonF32Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        NeonF32Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        NeonF32Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        NeonF32Butterfly7::new(direction)
    }

    #[inline(always)]
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        NeonVector::column_butterfly2(rows)
    }
    #[inline(always)]
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        bf.perform_parallel_fft_direct(rows[0], rows[1], rows[2])
    }
    #[inline(always)]
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Self::Rotation) -> [Self; 4] {
        NeonVector::column_butterfly4(rows, rotation)
    }
    #[inline(always)]
    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5] {
        bf.perform_parallel_fft_direct(rows[0], rows[1], rows[2], rows[3], rows[4])
    }
    #[inline(always)]
    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6] {
        bf.perform_parallel_fft_direct(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5])
    }
    #[inline(always)]
    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7] {
        bf.perform_parallel_fft_direct(rows)
    }

    neon_radixn_fft_helpers!();
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::simd_radixn::test_bodies;

    #[test]
    fn test_neon_radixn_f64() {
        // f64 fits one complex per vector, so every base length is legal
        test_bodies::factor_pairs::<float64x2_t>(&[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_neon_radixn_f32() {
        // f32 fits two complex per vector, so the base length has to be even
        test_bodies::factor_pairs::<float32x4_t>(&[2, 4, 6]);
    }

    #[test]
    fn test_neon_radixn_composite_base() {
        test_bodies::composite_base::<float32x4_t, float64x2_t>();
    }

    #[test]
    fn test_neon_radixn_large_recipes() {
        test_bodies::large_recipes::<float32x4_t, float64x2_t>();
    }

    #[test]
    #[ignore]
    fn test_neon_radixn_six_layers() {
        test_bodies::six_layers::<float32x4_t, float64x2_t>();
    }
}
