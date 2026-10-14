use crate::simd::simd_vector::SimdVector;

// A trait for computing "column FFTs", aka the cross FFTs in RadixN/Radix4.
// These FFTs are computed in-place and strided, hence they're treated as columns of a row-major array.
// This arrangement is inherently well-suited for SIMD, and this trait helps compute those cross FFTs via SIMD
// Each butterfly struct is implemented for exactly one LEN parameter
pub trait ColumnFft<V: SimdVector, const LEN: usize>: Send + Sync {
    // Compute V::COMPLEX_PER_VECTOR parallel FFTs on `data`
    fn column_fft(&self, data: [V; LEN]) -> [V; LEN];

    // Compute a single FFT on `data`, and assumes the remaining lanes are empty. Possibly faster than column_butterfly()
    #[inline(always)]
    #[allow(unused)]
    fn column_fft_lo1(&self, data: [V; LEN]) -> [V; LEN] {
        self.column_fft(data)
    }
}
