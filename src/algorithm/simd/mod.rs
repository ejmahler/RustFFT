// todo: cfg for the correct platform
mod avx_utils;
mod avx_split_radix;
mod avx_butterflies;

pub use self::avx_split_radix::SplitRadixAvx;
pub use self::avx_butterflies::{MixedRadixAvx4x2, MixedRadixAvx4x4, MixedRadixAvx4x8, MixedRadixAvx8x8};