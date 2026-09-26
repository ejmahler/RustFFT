#[macro_use]
mod fcma_common;
#[macro_use]
mod fcma_vector;

#[macro_use]
pub mod fcma_butterflies;
pub mod fcma_prime_butterflies;
pub mod fcma_radix4;
pub mod fcma_radixn;

mod fcma_utils;

pub mod fcma_planner;

use std::arch::aarch64::{float32x4_t, float64x2_t};

use crate::simd::simd_vector::SimdVector;
use crate::FftNum;
use fcma_vector::{FcmaSimdVector32, FcmaSimdVector64, FcmaVector};

pub trait FcmaNum: FftNum {
    type VectorType: FcmaVector<ScalarType = Self>;

    /// The same vector, wrapped in the newtype that carries this backend's `SimdVector` impl.
    /// See [`FcmaSimdVector64`] for why the wrapper is needed.
    type SimdVectorType: SimdVector<ScalarType = Self>;
}

impl FcmaNum for f32 {
    type VectorType = float32x4_t;
    type SimdVectorType = FcmaSimdVector32;
}
impl FcmaNum for f64 {
    type VectorType = float64x2_t;
    type SimdVectorType = FcmaSimdVector64;
}
