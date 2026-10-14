use crate::simd::simd_vector::SimdVector;

// A trait for computing "parallel" in a SIMD environment.
//
// The input is interpreted as a LEN x V::COMPLEX_PER_VECTOR row-major 2d array,
// and V::COMPLEX_PER_VECTOR parallel FFTs are computed down the columns of that 2D array.
//
// Each butterfly struct is implemented for exactly one LEN parameter
pub trait ParallelFft<V: SimdVector, const LEN: usize>: Send + Sync {
    // Compute V::COMPLEX_PER_VECTOR parallel FFTs on `data`
    fn parallel_fft(&self, data: [V; LEN]) -> [V; LEN];
}
