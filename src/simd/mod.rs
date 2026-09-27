pub mod simd_array;

// SimdRadixN isn't used by the planner yet, so don't throw warnings about unused code
#[allow(dead_code)]
pub mod simd_radix4;
#[allow(dead_code)]
pub mod simd_radixn;
#[macro_use]
pub mod simd_vector;
