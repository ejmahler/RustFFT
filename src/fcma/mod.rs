#[macro_use]
mod fcma_common;
#[macro_use]
mod fcma_vector;

#[macro_use]
pub mod fcma_butterflies;
pub mod fcma_prime_butterflies;
pub mod fcma_radix4;

mod fcma_utils;

pub mod fcma_planner;

use std::arch::aarch64::{float32x4_t, float64x2_t};

use crate::FftNum;
use fcma_vector::FcmaVector;

pub trait FcmaNum: FftNum {
    type VectorType: FcmaVector<ScalarType = Self>;
}

impl FcmaNum for f32 {
    type VectorType = float32x4_t;
}
impl FcmaNum for f64 {
    type VectorType = float64x2_t;
}
