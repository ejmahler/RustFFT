use std::arch::x86_64::*;
use num_complex::Complex;

use ::array_utils::{RawSlice, RawSliceMut};
use super::avx32_utils::Rotate90Config;

pub trait AvxComplexArray64 {
    unsafe fn load_complex_f64(&self, index: usize) -> __m256d;
    unsafe fn load_complex_f64_lo(&self, index: usize) -> __m128d;
}
pub trait AvxComplexArrayMut64 {
    // Store the 4 complex numbers contained in `data` at the given memory addresses
    unsafe fn store_complex_f64(&mut self, data: __m256d, index: usize);

    // Store the first 2 of 4 complex numbers in `data` at the given memory addresses
    unsafe fn store_complex_f64_lo(&mut self, data: __m128d, index: usize);
}

impl AvxComplexArray64 for [Complex<f64>] {
    #[inline(always)]
    unsafe fn load_complex_f64(&self, index: usize) -> __m256d {
        debug_assert!(self.len() >= index + 2);
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f64;
        _mm256_loadu_pd(float_ptr)
    }
    #[inline(always)]
    unsafe fn load_complex_f64_lo(&self, index: usize) -> __m128d {
        debug_assert!(self.len() > index);
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f64;
        _mm_loadu_pd(float_ptr)
    }
}
impl AvxComplexArrayMut64 for [Complex<f64>] {
    #[inline(always)]
    unsafe fn store_complex_f64(&mut self, data: __m256d, index: usize) {
        debug_assert!(self.len() >= index + 2);
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f64;
        _mm256_storeu_pd(float_ptr, data);
    }
    #[inline(always)]
    unsafe fn store_complex_f64_lo(&mut self, data: __m128d, index: usize) {
        debug_assert!(self.len() > index);
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f64;
        _mm_storeu_pd(float_ptr, data);
    }
}
impl AvxComplexArray64 for RawSlice<Complex<f64>> {
    #[inline(always)]
    unsafe fn load_complex_f64(&self, index: usize) -> __m256d {
        debug_assert!(self.len() >= index + 2);
        let float_ptr  = self.as_ptr().add(index) as *const f64;
        _mm256_loadu_pd(float_ptr)
    }
    #[inline(always)]
    unsafe fn load_complex_f64_lo(&self, index: usize) -> __m128d {
        debug_assert!(self.len() > index);
        let float_ptr  = self.as_ptr().add(index) as *const f64;
        _mm_loadu_pd(float_ptr)
    }
}
impl AvxComplexArrayMut64 for RawSliceMut<Complex<f64>> {
    #[inline(always)]
    unsafe fn store_complex_f64(&mut self, data: __m256d, index: usize) {
        debug_assert!(self.len() >= index + 2);
        let float_ptr = self.as_mut_ptr().add(index) as *mut f64;
        _mm256_storeu_pd(float_ptr, data);
    }
    #[inline(always)]
    unsafe fn store_complex_f64_lo(&mut self, data: __m128d, index: usize) {
        debug_assert!(self.len() > index);
        let float_ptr = self.as_mut_ptr().add(index) as *mut f64;
        _mm_storeu_pd(float_ptr, data);
    }
}

// Fills an AVX register by repeating the given complex number over and over
#[inline(always)]
pub unsafe fn broadcast_complex_f64(value: Complex<f64>) -> __m256d {
    _mm256_set_pd(value.im, value.re, value.im, value.re)
}

// Compute a single parallel butterfly 2 using SSE instructions
// rowN contains the nth element of the FFT
#[inline(always)]
pub unsafe fn column_butterfly2_f64_lo(rows: [__m128d; 2]) -> [__m128d; 2] {
    let output0 = _mm_add_pd(rows[0], rows[1]);
    let output1 = _mm_sub_pd(rows[0], rows[1]);

    [output0, output1]
}

// Compute 2 parallel butterfly 2's using AVX instructions
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly2_f64(row0: __m256d, row1: __m256d) -> (__m256d, __m256d) {
    let output0 = _mm256_add_pd(row0, row1);
    let output1 = _mm256_sub_pd(row0, row1);

    (output0, output1)
}

// Compute 2 parallel butterfly 2's using AVX instructions
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly2_array_f64(rows: [__m256d; 2]) -> [__m256d; 2] {
    let output0 = _mm256_add_pd(rows[0], rows[1]);
    let output1 = _mm256_sub_pd(rows[0], rows[1]);

    [output0, output1]
}

// Compute 2 parallel butterfly 4's using AVX instructions
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly4_f64(rows: [__m256d; 4], twiddle_config: Rotate90Config<__m256d>) -> [__m256d; 4] {
    // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
    let (mid0, mid2) = column_butterfly2_f64(rows[0], rows[2]);
    let (mid1, mid3) = column_butterfly2_f64(rows[1], rows[3]);

    // Apply element 3 inner twiddle factor
    let mid3_rotated = twiddle_config.rotate90(mid3);

    // Perform the second set of size-2 FFTs
    let (output0, output1) = column_butterfly2_f64(mid0, mid1);
    let (output2, output3) = column_butterfly2_f64(mid2, mid3_rotated);

    // Swap outputs 1 and 2 in the output to do a square transpose
    [output0, output2, output1, output3]
}

// Compute 2 parallel butterfly 4's using AVX instructions
// rowN contains the nth element of each parallel FFT
// this variant will roll a negation of row 3 wihout adding any new instructions
#[inline(always)]
pub unsafe fn column_butterfly4_negaterow3_f64(rows: [__m256d; 4], twiddle_config: Rotate90Config<__m256d>) -> [__m256d; 4] {
    // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
    let (mid0, mid2) = column_butterfly2_f64(rows[0], rows[2]);
    let (mid1, mid3) = (_mm256_sub_pd(rows[1], rows[3]), _mm256_add_pd(rows[1], rows[3])); // to negate row 3, swap add and sub in the butterfly 2

    // Apply element 3 inner twiddle factor
    let mid3_rotated = twiddle_config.rotate90(mid3);

    // Perform the second set of size-2 FFTs
    let (output0, output1) = column_butterfly2_f64(mid0, mid1);
    let (output2, output3) = column_butterfly2_f64(mid2, mid3_rotated);

    // Swap outputs 1 and 2 in the output to do a square transpose
    [output0, output2, output1, output3]
}

// Compute 1 butterfly 4 using AVX instructions
// rowN contains the nth element of the FFT
// this variant will roll a negation of row 3 wihout adding any new instructions
#[inline(always)]
pub unsafe fn column_butterfly4_negaterow3_f64_lo(rows: [__m128d; 4], twiddle_config: Rotate90Config<__m256d>) -> [__m128d; 4] {
    // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
    let mid0 = column_butterfly2_f64_lo([rows[0], rows[2]]);
    let mid1 = [_mm_sub_pd(rows[1], rows[3]), _mm_add_pd(rows[1], rows[3])]; // to negate row 3, swap add and sub in the butterfly 2

    // Apply element 3 inner twiddle factor
    let mid3_rotated = twiddle_config.rotate90_lo(mid1[1]);

    // Perform the second set of size-2 FFTs
    let output0 = column_butterfly2_f64_lo([mid0[0], mid1[0]]);
    let output1 = column_butterfly2_f64_lo([mid0[1], mid3_rotated]);

    // Swap outputs 1 and 2 in the output to do a square transpose
    [output0[0], output1[0], output0[1], output1[1]]
}

// Compute 1 butterfly 4 using AVX instructions
// rowN contains the nth element of the FFT
#[inline(always)]
pub unsafe fn column_butterfly4_f64_lo(rows: [__m128d; 4], twiddle_config: Rotate90Config<__m256d>) -> [__m128d; 4] {
    // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
    let mid0 = column_butterfly2_f64_lo([rows[0], rows[2]]);
    let mid1 = column_butterfly2_f64_lo([rows[1], rows[3]]);

    // Apply element 3 inner twiddle factor
    let mid3_rotated = twiddle_config.rotate90_lo(mid1[1]);

    // Perform the second set of size-2 FFTs
    let output0 = column_butterfly2_f64_lo([mid0[0], mid1[0]]);
    let output1 = column_butterfly2_f64_lo([mid0[1], mid3_rotated]);

    // Swap outputs 1 and 2 in the output to do a square transpose
    [output0[0], output1[0], output0[1], output1[1]]
}


// Treat the input like the rows of a 2x2 array, and transpose said rows to the columns
#[inline(always)]
pub unsafe fn transpose_2x2_f64(rows: [__m256d; 2]) -> [__m256d; 2] {
    let col0 = _mm256_permute2f128_pd(rows[0], rows[1], 0x20);
    let col1 = _mm256_permute2f128_pd(rows[0], rows[1], 0x31);

    [col0, col1]
}

// Treat the input like the rows of a 4x2 array, and transpose it to a 2x4 array
#[inline(always)]
pub unsafe fn transpose_4x2_to_2x4_f64(rows0: [__m256d; 2], rows1: [__m256d; 2]) -> [__m256d; 4] {
    let output00 = transpose_2x2_f64(rows0);
    let output01 = transpose_2x2_f64(rows1);

    [output00[0], output00[1], output01[0], output01[1]]
}


// Treat the input like the rows of a 2x3 array, and transpose it to a 3x2 array
// But instead of storing "columns" of registers as separate arrays for further processing, pack them all into one array
#[inline(always)]
pub unsafe fn transpose_2x3_to_3x2_packed_f64(rows: [__m256d; 3]) -> [__m256d; 3] {
    [
        _mm256_permute2f128_pd(rows[0], rows[1], 0x20),
        _mm256_permute2f128_pd(rows[2], rows[0], 0x30),
        _mm256_permute2f128_pd(rows[1], rows[2], 0x31),
    ]
}

// Treat the input like the rows of a 2x4 array, and transpose it to a 4x2 array
// But instead of storing "columns" of registers as separate arrays for further processing, pack them all into one array
#[inline(always)]
pub unsafe fn transpose_2x4_to_4x2_packed_f64(rows: [__m256d; 4]) -> [__m256d; 4] {
    let chunk0 = [rows[0], rows[1]];
    let chunk1 = [rows[2], rows[3]];

    let output0 = transpose_2x2_f64(chunk0);
    let output1 = transpose_2x2_f64(chunk1);

    [output0[0], output1[0], output0[1], output1[1]]
}

// Treat the input like the rows of a 2x8 array, and transpose it to a 8x2 array.
// But instead of storing "columns" of registers as separate arrays for further processing, pack them all into one array
#[inline(always)]
pub unsafe fn transpose_2x6_to_6x2_packed_f64(rows: [__m256d; 6]) -> [__m256d; 6] {
    let chunk0 = [rows[0], rows[1]];
    let chunk1 = [rows[2], rows[3]];
    let chunk2 = [rows[4], rows[5]];

    let output0 = transpose_2x2_f64(chunk0);
    let output1 = transpose_2x2_f64(chunk1);
    let output2 = transpose_2x2_f64(chunk2);

    [output0[0], output1[0], output2[0], output0[1], output1[1], output2[1]] 
}

// Treat the input like the rows of a 2x8 array, and transpose it to a 8x2 array.
// But instead of storing "columns" of registers as separate arrays for further processing, pack them all into one array
#[inline(always)]
pub unsafe fn transpose_2x8_to_8x2_packed_f64(rows: [__m256d; 8]) -> [__m256d; 8] {
    let chunk0 = [rows[0], rows[1]];
    let chunk1 = [rows[2], rows[3]];
    let chunk2 = [rows[4], rows[5]];
    let chunk3 = [rows[6], rows[7]];

    let output0 = transpose_2x2_f64(chunk0);
    let output1 = transpose_2x2_f64(chunk1);
    let output2 = transpose_2x2_f64(chunk2);
    let output3 = transpose_2x2_f64(chunk3);

    [output0[0], output1[0], output2[0], output3[0], output0[1], output1[1], output2[1], output3[1]] 
}


// Treat the input like the rows of a 2x9 array, and transpose it to a 9x2 array
// But instead of storing "columns" of registers as separate arrays for further processing, pack them all into one array
#[inline(always)]
pub unsafe fn transpose_2x9_to_9x2_packed_f64(rows: [__m256d; 9]) -> [__m256d; 9] {
    [
        _mm256_permute2f128_pd(rows[0], rows[1], 0x20),
        _mm256_permute2f128_pd(rows[2], rows[3], 0x20),
        _mm256_permute2f128_pd(rows[4], rows[5], 0x20),
        _mm256_permute2f128_pd(rows[6], rows[7], 0x20),
        _mm256_permute2f128_pd(rows[8], rows[0], 0x30),
        _mm256_permute2f128_pd(rows[1], rows[2], 0x31),
        _mm256_permute2f128_pd(rows[3], rows[4], 0x31),
        _mm256_permute2f128_pd(rows[5], rows[6], 0x31),
        _mm256_permute2f128_pd(rows[7], rows[8], 0x31),
    ]
}

// Treat the input like the rows of a 2x12 array, and transpose it to a 12x2 array.
// But instead of storing "columns" of registers as separate arrays for further processing, pack them all into one array
#[inline(always)]
pub unsafe fn transpose_2x12_to_12x2_packed_f64(rows: [__m256d; 12]) -> [__m256d; 12] {
    let chunk0 = [rows[0], rows[1]];
    let chunk1 = [rows[2], rows[3]];
    let chunk2 = [rows[4], rows[5]];
    let chunk3 = [rows[6], rows[7]];
    let chunk4 = [rows[8], rows[9]];
    let chunk5 = [rows[10], rows[11]];

    let output0 = transpose_2x2_f64(chunk0);
    let output1 = transpose_2x2_f64(chunk1);
    let output2 = transpose_2x2_f64(chunk2);
    let output3 = transpose_2x2_f64(chunk3);
    let output4 = transpose_2x2_f64(chunk4);
    let output5 = transpose_2x2_f64(chunk5);

    [output0[0], output1[0], output2[0], output3[0], output4[0], output5[0], output0[1], output1[1], output2[1], output3[1], output4[1], output5[1]] 
}

// Treat the input like the rows of a 2x16 array, and transpose it to a 16x2 array.
// But instead of storing "columns" of registers as separate arrays for further processing, pack them all into one array
#[inline(always)]
pub unsafe fn transpose_2x16_to_16x2_packed_f64(rows: [__m256d; 16]) -> [__m256d; 16] {
    let chunk0 = [rows[0], rows[1]];
    let chunk1 = [rows[2], rows[3]];
    let chunk2 = [rows[4], rows[5]];
    let chunk3 = [rows[6], rows[7]];
    let chunk4 = [rows[8], rows[9]];
    let chunk5 = [rows[10], rows[11]];
    let chunk6 = [rows[12], rows[13]];
    let chunk7 = [rows[14], rows[15]];

    let output0 = transpose_2x2_f64(chunk0);
    let output1 = transpose_2x2_f64(chunk1);
    let output2 = transpose_2x2_f64(chunk2);
    let output3 = transpose_2x2_f64(chunk3);
    let output4 = transpose_2x2_f64(chunk4);
    let output5 = transpose_2x2_f64(chunk5);
    let output6 = transpose_2x2_f64(chunk6);
    let output7 = transpose_2x2_f64(chunk7);

    [output0[0], output1[0], output2[0], output3[0], output4[0], output5[0], output6[0], output7[0], output0[1], output1[1], output2[1], output3[1], output4[1], output5[1], output6[1], output7[1]] 
}

// Treat the input like the rows of a 3x3 array, and transpose it
// The assumption here is that it's very likely that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_3x3_f64(rows0: [__m128d; 3], rows1: [__m256d; 3]) -> ([__m128d; 3], [__m256d; 3]) {
    // the first column of output will be made up of the first row of input
    let output0 = [
        rows0[0],
        _mm256_castpd256_pd128(rows1[0]),
        _mm256_extractf128_pd(rows1[0], 1),
    ];

    // the second column of output will be made of the second 2 rows of input
    let output10 = _mm256_permute2f128_pd(_mm256_castpd128_pd256(rows0[1]), _mm256_castpd128_pd256(rows0[2]), 0x20);
    let lower_chunk = [rows1[1], rows1[2]];
    let lower_transposed = transpose_2x2_f64(lower_chunk);
    let output1 = [
        output10,
        lower_transposed[0],
        lower_transposed[1],
    ];

    (output0, output1)
}



// Treat the input like the rows of a 3x4 array, and transpose it to a 4x3 array
// The assumption here is that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_3x4_to_4x3_f64(rows0: [__m128d; 4], rows1: [__m256d; 4]) -> ([__m256d; 3], [__m256d; 3]) {
    // the top row of each output array will come from the first column, and the second 2 rows will come from 2x2 transposing the rows1 array
    let merged0 = _mm256_permute2f128_pd(_mm256_castpd128_pd256(rows0[0]), _mm256_castpd128_pd256(rows0[1]), 0x20);
    let merged1 = _mm256_permute2f128_pd(_mm256_castpd128_pd256(rows0[2]), _mm256_castpd128_pd256(rows0[3]), 0x20);

    let chunk0 = [rows1[0], rows1[1]];
    let chunk1 = [rows1[2], rows1[3]];

    let lower0 = transpose_2x2_f64(chunk0);
    let lower1 = transpose_2x2_f64(chunk1);

    (
        [merged0, lower0[0], lower0[1]],
        [merged1, lower1[0], lower1[1]],
    )
}

// Treat the input like the rows of a 3x6 array, and transpose it to a 6x3 array
// The assumption here is that caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_3x6_to_6x3_f64(rows0: [__m128d; 6], rows1: [__m256d; 6]) -> ([__m256d; 3], [__m256d; 3], [__m256d; 3]) {
    let chunk0 = [rows1[0], rows1[1]];
    let chunk1 = [rows1[2], rows1[3]];
    let chunk2 = [rows1[4], rows1[5]];

    let transposed0 = transpose_2x2_f64(chunk0);
    let transposed1 = transpose_2x2_f64(chunk1);
    let transposed2 = transpose_2x2_f64(chunk2);

    
    let output0 = [
        _mm256_insertf128_pd(_mm256_castpd128_pd256(rows0[0]), rows0[1], 1),
        transposed0[0],
        transposed0[1],
    ];
    let output1 = [
        _mm256_insertf128_pd(_mm256_castpd128_pd256(rows0[2]), rows0[3], 1),
        transposed1[0],
        transposed1[1],
    ];
    let output2 = [
        _mm256_insertf128_pd(_mm256_castpd128_pd256(rows0[4]), rows0[5], 1),
        transposed2[0],
        transposed2[1],
    ];

    (output0, output1, output2)
}

// Treat the input like the rows of a 9x3 array, and transpose it to a 3x9 array
// The assumption here is that caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_9x3_to_3x9_f64(rows0: [__m128d; 3], rows1: [__m256d; 3], rows2: [__m256d; 3], rows3: [__m256d; 3], rows4: [__m256d; 3]) -> ([__m128d; 9], [__m256d; 9]) {
    let chunk1 = [rows1[1], rows1[2]];
    let chunk2 = [rows2[1], rows2[2]];
    let chunk3 = [rows3[1], rows3[2]];
    let chunk4 = [rows4[1], rows4[2]];

    let transposed1 = transpose_2x2_f64(chunk1);
    let transposed2 = transpose_2x2_f64(chunk2);
    let transposed3 = transpose_2x2_f64(chunk3);
    let transposed4 = transpose_2x2_f64(chunk4);

    
    let output0 = [
        rows0[0],
        _mm256_castpd256_pd128(rows1[0]),
        _mm256_extractf128_pd(rows1[0], 1),
        _mm256_castpd256_pd128(rows2[0]),
        _mm256_extractf128_pd(rows2[0], 1),
        _mm256_castpd256_pd128(rows3[0]),
        _mm256_extractf128_pd(rows3[0], 1),
        _mm256_castpd256_pd128(rows4[0]),
        _mm256_extractf128_pd(rows4[0], 1),
    ];
    let output1 = [
        _mm256_insertf128_pd(_mm256_castpd128_pd256(rows0[1]), rows0[2], 1),
        transposed1[0],
        transposed1[1],
        transposed2[0],
        transposed2[1],
        transposed3[0],
        transposed3[1],
        transposed4[0],
        transposed4[1],
    ];

    (output0, output1)
}


// Treat the input like the rows of a 4x4 array, and transpose said rows to the columns
// The assumption here is that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_4x4_f64(rows0: [__m256d; 4], rows1: [__m256d; 4]) -> ([__m256d; 4], [__m256d; 4]) {
    let chunk00 = [rows0[0], rows0[1]];
    let chunk01 = [rows0[2], rows0[3]];
    let chunk10 = [rows1[0], rows1[1]];
    let chunk11 = [rows1[2], rows1[3]];

    let output00 = transpose_2x2_f64(chunk00);
    let output01 = transpose_2x2_f64(chunk10);
    let output10 = transpose_2x2_f64(chunk01);
    let output11 = transpose_2x2_f64(chunk11);

    ([output00[0], output00[1], output01[0], output01[1]], [output10[0], output10[1], output11[0], output11[1]])
}

// Treat the input like the rows of a 6x6 array, and transpose said rows to the columns
// The assumption here is that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_6x6_f64(rows0: [__m256d; 6], rows1: [__m256d; 6], rows2: [__m256d; 6]) -> ([__m256d; 6], [__m256d; 6], [__m256d; 6]) {
    let chunk00 = [rows0[0], rows0[1]];
    let chunk01 = [rows0[2], rows0[3]];
    let chunk02 = [rows0[4], rows0[5]];
    let chunk10 = [rows1[0], rows1[1]];
    let chunk11 = [rows1[2], rows1[3]];
    let chunk12 = [rows1[4], rows1[5]];
    let chunk20 = [rows2[0], rows2[1]];
    let chunk21 = [rows2[2], rows2[3]];
    let chunk22 = [rows2[4], rows2[5]];

    let output00 = transpose_2x2_f64(chunk00);
    let output01 = transpose_2x2_f64(chunk10);
    let output02 = transpose_2x2_f64(chunk20);
    let output10 = transpose_2x2_f64(chunk01);
    let output11 = transpose_2x2_f64(chunk11);
    let output12 = transpose_2x2_f64(chunk21);
    let output20 = transpose_2x2_f64(chunk02);
    let output21 = transpose_2x2_f64(chunk12);
    let output22 = transpose_2x2_f64(chunk22);

    (
        [output00[0], output00[1], output01[0], output01[1], output02[0], output02[1]],
        [output10[0], output10[1], output11[0], output11[1], output12[0], output12[1]],
        [output20[0], output20[1], output21[0], output21[1], output22[0], output22[1]],
    )
}


// Treat the input like the rows of a 6x4 array, and transpose said rows to the columns
// The assumption here is that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_6x4_to_4x6_f64(rows0: [__m256d; 4], rows1: [__m256d; 4], rows2: [__m256d; 4]) -> ([__m256d; 6], [__m256d; 6]) {
    let chunk00 = [rows0[0], rows0[1]];
    let chunk01 = [rows0[2], rows0[3]];
    let chunk10 = [rows1[0], rows1[1]];
    let chunk11 = [rows1[2], rows1[3]];
    let chunk20 = [rows2[0], rows2[1]];
    let chunk21 = [rows2[2], rows2[3]];

    let output00 = transpose_2x2_f64(chunk00);
    let output01 = transpose_2x2_f64(chunk10);
    let output02 = transpose_2x2_f64(chunk20);
    let output10 = transpose_2x2_f64(chunk01);
    let output11 = transpose_2x2_f64(chunk11);
    let output12 = transpose_2x2_f64(chunk21);

    ([output00[0], output00[1], output01[0], output01[1], output02[0], output02[1]], [output10[0], output10[1], output11[0], output11[1], output12[0], output12[1]])
}


// Treat the input like the rows of a 8x4 array, and transpose it to a 4x8 array
// The assumption here is that it's very likely that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_8x4_to_4x8_f64(rows0: [__m256d;4], rows1: [__m256d; 4], rows2: [__m256d; 4], rows3: [__m256d; 4]) -> ([__m256d;8], [__m256d;8]) {
    let chunk00 = [rows0[0], rows0[1]];
    let chunk01 = [rows0[2], rows0[3]];
    let chunk10 = [rows1[0], rows1[1]];
    let chunk11 = [rows1[2], rows1[3]];
    let chunk20 = [rows2[0], rows2[1]];
    let chunk21 = [rows2[2], rows2[3]];
    let chunk30 = [rows3[0], rows3[1]];
    let chunk31 = [rows3[2], rows3[3]];

    let output00 = transpose_2x2_f64(chunk00);
    let output01 = transpose_2x2_f64(chunk10);
    let output02 = transpose_2x2_f64(chunk20);
    let output03 = transpose_2x2_f64(chunk30);
    let output10 = transpose_2x2_f64(chunk01);
    let output11 = transpose_2x2_f64(chunk11);
    let output12 = transpose_2x2_f64(chunk21);
    let output13 = transpose_2x2_f64(chunk31);

    ([output00[0], output00[1], output01[0], output01[1], output02[0], output02[1], output03[0], output03[1]], [output10[0], output10[1], output11[0], output11[1], output12[0], output12[1], output13[0], output13[1]])
}

// Functions in the "FMA" sub-module require the fma instruction set in addition to AVX
pub mod fma {
    use super::*;

    // Multiply the complex numbers in `left` by the complex numbers in `right`, using FMA instructions where possible
    #[inline(always)]
    pub unsafe fn complex_multiply_f64(left: __m256d, right: __m256d) -> __m256d {
        // Extract the real and imaginary components from left into 2 separate registers
        let left_real = _mm256_movedup_pd(left);
        let left_imag = _mm256_permute_pd(left, 0x0F); // apparently the avx deigners just skipped movehdup f64?? So use a permute instruction to take its place, copying the imaginaries int othe reals

        // create a shuffled version of right where the imaginary values are swapped with the reals
        let right_shuffled = _mm256_permute_pd(right, 0x05);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = _mm256_mul_pd(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then alternatingly add and subtract the left side from the right
        _mm256_fmaddsub_pd(left_real, right, output_right)
    }

    // Multiply the complex number in `left` by the complex numbers in `right`, using FMA instructions where possible
    #[inline(always)]
    pub unsafe fn complex_multiply_f64_lo(left: __m128d, right: __m128d) -> __m128d {
        // Extract the real and imaginary components from left into 2 separate registers
        let left_real = _mm_movedup_pd(left);
        let left_imag = _mm_permute_pd(left, 0x0F); // apparently the avx deigners just skipped movehdup f64?? So use a permute instruction to take its place, copying the imaginaries int othe reals

        // create a shuffled version of right where the imaginary values are swapped with the reals
        let right_shuffled = _mm_permute_pd(right, 0x05);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = _mm_mul_pd(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then alternatingly add and subtract the left side from the right
        _mm_fmaddsub_pd(left_real, right, output_right)
    }

    // Multiply the complex numbers in `left` by the complex numbers in `right`, using FMA instructions where possible
    // This variant assumes that `left` should be conjugated before multiplying (IE, the imaginary numbers in `left` should be negated)
    // Thankfully, it is straightforward to roll this into existing instructions. Namely, we can get away with replacing "fmaddsub" with "fmsubadd"
    #[inline(always)]
    pub unsafe fn complex_conjugated_multiply_f64(left: __m256d, right: __m256d) -> __m256d {
        // Extract the real and imaginary components from left into 2 separate registers
        let left_real = _mm256_movedup_pd(left);
        let left_imag = _mm256_permute_pd(left, 0x0f);

        // create a shuffled version of right where the imaginary values are swapped with the reals
        let right_shuffled = _mm256_permute_pd(right, 0x05);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = _mm256_mul_pd(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then alternatingly add and subtract the left side from the right
        _mm256_fmsubadd_pd(left_real, right, output_right)
    }

    // Multiply the complex number in `left` by the complex number in `right`, using FMA instructions where possible
    // This variant assumes that `left` should be conjugated before multiplying (IE, the imaginary numbers in `left` should be negated)
    // Thankfully, it is straightforward to roll this into existing instructions. Namely, we can get away with replacing "fmaddsub" with "fmsubadd"
    #[inline(always)]
    pub unsafe fn complex_conjugated_multiply_f64_lo(left: __m128d, right: __m128d) -> __m128d {
        // Extract the real and imaginary components from left into 2 separate registers
        let left_real = _mm_movedup_pd(left);
        let left_imag = _mm_permute_pd(left, 0x0f);

        // create a shuffled version of right where the imaginary values are swapped with the reals
        let right_shuffled = _mm_permute_pd(right, 0x05);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = _mm_mul_pd(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then alternatingly add and subtract the left side from the right
        _mm_fmsubadd_pd(left_real, right, output_right)
    }

    // compute buffer[i] = buffer[i].conj() * multiplier[i] pairwise complex multiplication for each element.
    // This is kind of usage-specific, because 'b' is stored as pre-loaded AVX registers, but 'a' is stored as loose complex numbers
    #[target_feature(enable = "avx", enable = "fma")]
    pub unsafe fn pairwise_complex_multiply_conjugated(buffer: &mut [Complex<f64>], multiplier: &[__m256d]) {
        for (i, twiddle) in multiplier.iter().enumerate() {
            let inner_vector = buffer.load_complex_f64(i*2);
            let product_vector = complex_conjugated_multiply_f64(inner_vector, *twiddle);
            buffer.store_complex_f64(product_vector, i*2);
        }
    }

    // Compute 2 parallel butterfly 3's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly3_f64(rows: [__m256d; 3], twiddles: __m256d) -> [__m256d; 3] {
        let (mid1_pretwiddle, mid2_pretwiddle) = column_butterfly2_f64(rows[1], rows[2]);
        let output0 = _mm256_add_pd(rows[0], mid1_pretwiddle);

        let twiddle_real = _mm256_movedup_pd(twiddles);
        
        let mid1 = _mm256_fmadd_pd(mid1_pretwiddle, twiddle_real, rows[0]);

        let mid2_rotated = Rotate90Config::new_f64(true).rotate90(mid2_pretwiddle);

        // combine the twiddle with the subsequent butterfly 2 via FMA instructions. 3 instructions instead of two, and removes a layer of latency
        let twiddle_imag = _mm256_permute_pd(twiddles, 0x0F); // apparently the avx deigners just skipped movehdup f64?? So use a permute instruction to take its place, copying the imaginaries into the reals
        let output1 = _mm256_fmadd_pd(mid2_rotated, twiddle_imag, mid1);
        let output2 = _mm256_fnmadd_pd(mid2_rotated, twiddle_imag, mid1);

        [output0, output1, output2]
    }

    // Compute 2 parallel butterfly 3's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly3_f64_lo(rows: [__m128d; 3], twiddles: __m256d) -> [__m128d; 3] {
        let [mid1_pretwiddle, mid2_pretwiddle] = column_butterfly2_f64_lo([rows[1], rows[2]]);
        let output0 = _mm_add_pd(rows[0], mid1_pretwiddle);

        let twiddles_lo = _mm256_castpd256_pd128(twiddles);
        let twiddle_real = _mm_movedup_pd(twiddles_lo);
        
        let mid1 = _mm_fmadd_pd(mid1_pretwiddle, twiddle_real, rows[0]);

        let mid2_rotated = Rotate90Config::new_f64(true).rotate90_lo(mid2_pretwiddle);

        // combine the twiddle with the subsequent butterfly 2 via FMA instructions. 3 instructions instead of two, and removes a layer of latency
        let twiddle_imag = _mm_permute_pd(twiddles_lo, 0x0F); // apparently the avx deigners just skipped movehdup f64?? So use a permute instruction to take its place, copying the imaginaries into the reals
        let output1 = _mm_fmadd_pd(mid2_rotated, twiddle_imag, mid1);
        let output2 = _mm_fnmadd_pd(mid2_rotated, twiddle_imag, mid1);

        [output0, output1, output2]
    }

    // Compute 2 parallel butterfly 6's using AVX instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly6_f64(rows: [__m256d; 6], butterfly3_twiddles: __m256d) -> [__m256d; 6] {
        // We're going good-thomas algorithm. We can reorder the inputs and outputs in such a way that we don't need twiddle factors!
        let mid0 = column_butterfly3_f64([rows[0], rows[2], rows[4]], butterfly3_twiddles);
        let mid1 = column_butterfly3_f64([rows[3], rows[5], rows[1]], butterfly3_twiddles);

        // transpose the data and do butterfly 2's
        let (output0, output1) = column_butterfly2_f64(mid0[0], mid1[0]);
        let (output2, output3) = column_butterfly2_f64(mid0[1], mid1[1]);
        let (output4, output5) = column_butterfly2_f64(mid0[2], mid1[2]);

        // reorder into output
        [output0, output3, output4, output1, output2, output5]
    }

    // Compute 2 parallel butterfly 6's using AVX instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly6_f64_lo(rows: [__m128d; 6], butterfly3_twiddles: __m256d) -> [__m128d; 6] {
        
        // We're going good-thomas algorithm. We can reorder the inputs and outputs in such a way that we don't need twiddle factors!
        // Since we're only dealing with f128s, we can merge the two butterfly 3's we have to compute into a single butterfly3. 
        // Not a huge benefit but every little bit helps, and if we're loading right before this, the compiler is smart enough to generate instuctions that load directly from the inserts
        let merged = [
            _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[0]), rows[3], 1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[2]), rows[5], 1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[4]), rows[1], 1),
        ];
        let mid = column_butterfly3_f64(merged, butterfly3_twiddles);

        // extract our merged arrays
        let mid0 = [
            _mm256_castpd256_pd128(mid[0]),
            _mm256_castpd256_pd128(mid[1]),
            _mm256_castpd256_pd128(mid[2]),
        ];
        let mid1 = [
            _mm256_extractf128_pd(mid[0], 1),
            _mm256_extractf128_pd(mid[1], 1),
            _mm256_extractf128_pd(mid[2], 1),
        ];

        // transpose the data and do butterfly 2's
        let [output0, output1] = column_butterfly2_f64_lo([mid0[0], mid1[0]]);
        let [output2, output3] = column_butterfly2_f64_lo([mid0[1], mid1[1]]);
        let [output4, output5] = column_butterfly2_f64_lo([mid0[2], mid1[2]]);

        // reorder into output
        [output0, output3, output4, output1, output2, output5]
    }

    #[inline(always)]
    pub unsafe fn apply_butterfly8_twiddle1(input: __m256d, twiddle_config: Rotate90Config<__m256d>) -> __m256d {
        // note: we're computing a square root here, but checking the assembly says the compiler is smart enough to turn this into a constant
        let root2 = 2.0f64.sqrt() * 0.5;
        let root2_vector = _mm256_broadcast_sd(&root2);
        let rotated = twiddle_config.rotate90(input);
        let combined = _mm256_add_pd(rotated, input);
        _mm256_mul_pd(root2_vector, combined)
    }
    #[inline(always)]
    pub unsafe fn apply_butterfly8_twiddle3(input: __m256d, twiddle_config: Rotate90Config<__m256d>) -> __m256d {
        // note: we're computing a square root here, but checking the assembly says the compiler is smart enough to turn this into a constant
        let root2 = 2.0f64.sqrt() * 0.5;
        let root2_vector = _mm256_broadcast_sd(&root2);
        let rotated = twiddle_config.rotate90(input);
        let combined = _mm256_sub_pd(rotated, input);
        _mm256_mul_pd(root2_vector, combined)
    }

    #[inline(always)]
    pub unsafe fn apply_butterfly8_twiddle1_lo(input: __m128d, twiddle_config: Rotate90Config<__m256d>) -> __m128d {
        // note: we're computing a square root here, but checking the assembly says the compiler is smart enough to turn this into a constant
        let root2 = 2.0f64.sqrt() * 0.5;
        let root2_vector = _mm_load1_pd(&root2);
        let rotated = twiddle_config.rotate90_lo(input);
        let combined = _mm_add_pd(rotated, input);
        _mm_mul_pd(root2_vector, combined)
    }
    #[inline(always)]
    pub unsafe fn apply_butterfly8_twiddle3_lo(input: __m128d, twiddle_config: Rotate90Config<__m256d>) -> __m128d {
        // note: we're computing a square root here, but checking the assembly says the compiler is smart enough to turn this into a constant
        let root2 = 2.0f64.sqrt() * 0.5;
        let root2_vector = _mm_load1_pd(&root2);
        let rotated = twiddle_config.rotate90_lo(input);
        let combined = _mm_sub_pd(rotated, input);
        _mm_mul_pd(root2_vector, combined)
    }

    // Compute 2 parallel butterfly 8's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly8_f64(rows: [__m256d; 8], twiddle_config: Rotate90Config<__m256d>) -> [__m256d; 8] {
        // Treat our butterfly-8 as a 2x4 array. first, do butterfly 4's down the columns
        let mid0     = column_butterfly4_f64([rows[0], rows[2], rows[4], rows[6]], twiddle_config);
        let mut mid1 = column_butterfly4_f64([rows[1], rows[3], rows[5], rows[7]], twiddle_config);

        // Apply twiddle factors
        mid1[1] = apply_butterfly8_twiddle1(mid1[1], twiddle_config);
        mid1[2] = twiddle_config.rotate90(mid1[2]);
        mid1[3] = apply_butterfly8_twiddle3(mid1[3], twiddle_config);

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transposE" and thne apply butterfly 2's across the columns of our 4x2 array
        let (final0, final1) = column_butterfly2_f64(mid0[0], mid1[0]);
        let (final2, final3) = column_butterfly2_f64(mid0[1], mid1[1]);
        let (final4, final5) = column_butterfly2_f64(mid0[2], mid1[2]);
        let (final6, final7) = column_butterfly2_f64(mid0[3], mid1[3]);

        [final0, final2, final4, final6, final1, final3, final5, final7]
    }

    // Compute 2 parallel butterfly 8's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly8_f64_lo(rows: [__m128d; 8], twiddle_config: Rotate90Config<__m256d>) -> [__m128d; 8] {
        // Treat our butterfly-8 as a 2x4 array. first, do butterfly 4's down the columns
        let mid0     = column_butterfly4_f64_lo([rows[0], rows[2], rows[4], rows[6]], twiddle_config);
        let mut mid1 = column_butterfly4_f64_lo([rows[1], rows[3], rows[5], rows[7]], twiddle_config);

        // Apply twiddle factors
        mid1[1] = apply_butterfly8_twiddle1_lo(mid1[1], twiddle_config);
        mid1[2] = twiddle_config.rotate90_lo(mid1[2]);
        mid1[3] = apply_butterfly8_twiddle3_lo(mid1[3], twiddle_config);

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transpose" and then apply butterfly 2's across the columns of our 4x2 array
        let final0 = column_butterfly2_f64_lo([mid0[0], mid1[0]]);
        let final1 = column_butterfly2_f64_lo([mid0[1], mid1[1]]);
        let final2 = column_butterfly2_f64_lo([mid0[2], mid1[2]]);
        let final3 = column_butterfly2_f64_lo([mid0[3], mid1[3]]);

        [final0[0], final1[0], final2[0], final3[0], final0[1], final1[1], final2[1], final3[1]]
    }

    // Compute 2 parallel butterfly 9's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly9_f64(rows: [__m256d; 9], twiddles: [__m256d;3], butterfly3_twiddles: __m256d) -> [__m256d; 9] {
        let mid0 = column_butterfly3_f64([rows[0], rows[3], rows[6]], butterfly3_twiddles);
        let mut mid1 = column_butterfly3_f64([rows[1], rows[4], rows[7]], butterfly3_twiddles);
        let mut mid2 = column_butterfly3_f64([rows[2], rows[5], rows[8]], butterfly3_twiddles);

        // Apply twiddle factors. Note that we're re-using twiddles[1]
        mid1[1] = complex_multiply_f64(twiddles[0], mid1[1]);
        mid1[2] = complex_multiply_f64(twiddles[1], mid1[2]);
        mid2[1] = complex_multiply_f64(twiddles[1], mid2[1]);
        mid2[2] = complex_multiply_f64(twiddles[2], mid2[2]);

        let [output0, output1, output2] = column_butterfly3_f64([mid0[0], mid1[0], mid2[0]], butterfly3_twiddles);
        let [output3, output4, output5] = column_butterfly3_f64([mid0[1], mid1[1], mid2[1]], butterfly3_twiddles);
        let [output6, output7, output8] = column_butterfly3_f64([mid0[2], mid1[2], mid2[2]], butterfly3_twiddles);

        [output0, output3, output6, output1, output4, output7, output2, output5, output8]
    }

    // Compute 2 parallel butterfly 9's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly9_f64_lo(rows: [__m128d; 9], twiddles_merged: [__m256d;2], butterfly3_twiddles: __m256d) -> [__m128d; 9] {
        // since this is a "lo" step, each column is only half full. if we merge some registers, we can do a single column butterfly3 on YMM registers instead of 2 on XMM registers
        let rows12 = _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[1]), rows[2], 0x1);
        let rows45 = _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[4]), rows[5], 0x1);
        let rows78 = _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[7]), rows[8], 0x1);

        let mid0 = column_butterfly3_f64_lo([rows[0], rows[3], rows[6]], butterfly3_twiddles);
        let mut mid12 = column_butterfly3_f64([rows12, rows45, rows78], butterfly3_twiddles);

        // Apply twiddle factors. we're applying them on the merged set of vectors, so we need slightly different twiddle factors
        mid12[1] = complex_multiply_f64(twiddles_merged[0], mid12[1]);
        mid12[2] = complex_multiply_f64(twiddles_merged[1], mid12[2]);

        // we can't use our merged columns anymore. we also want to merge some columns of our next set of FFTs, so extract/re-merge as necessary
        let mid1_0 = _mm256_castpd256_pd128(mid12[0]);
        let mid2_0 = _mm256_extractf128_pd(mid12[0], 0x1);
        
        let transposed12 = _mm256_insertf128_pd(_mm256_castpd128_pd256(mid0[1]), mid0[2], 0x1);
        let transposed45 = _mm256_permute2f128_pd(mid12[1], mid12[2], 0x20);
        let transposed78 = _mm256_permute2f128_pd(mid12[1], mid12[2], 0x31);

        let [output0, output1, output2] = column_butterfly3_f64_lo([mid0[0], mid1_0, mid2_0], butterfly3_twiddles);
        let [output36, output47, output58] = column_butterfly3_f64([transposed12, transposed45, transposed78], butterfly3_twiddles);

        // finally, extract our second set of merged columns
        let output3 = _mm256_castpd256_pd128(output36);
        let output6 = _mm256_extractf128_pd(output36, 0x1);
        let output4 = _mm256_castpd256_pd128(output47);
        let output7 = _mm256_extractf128_pd(output47, 0x1);
        let output5 = _mm256_castpd256_pd128(output58);
        let output8 = _mm256_extractf128_pd(output58, 0x1);

        [output0, output3, output6, output1, output4, output7, output2, output5, output8]
    }

    // Compute 2 parallel butterfly 12's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly12_f64(rows: [__m256d; 12], butterfly3_twiddles: __m256d, twiddle_config: Rotate90Config<__m256d>) -> [__m256d; 12] {
        // Compute this as a 4x3 FFT. since 4 and 3 are coprime, we can use the good-thomas algorithm. That means crazy reordering of our inputs and outputs, but it also means no twiddle factors
        let mid0 = column_butterfly4_f64([rows[0], rows[3], rows[6], rows[9]], twiddle_config);
        let mid1 = column_butterfly4_f64([rows[4], rows[7], rows[10],rows[1]], twiddle_config);
        let mid2 = column_butterfly4_f64([rows[8], rows[11],rows[2], rows[5]], twiddle_config);

        let [output0, output1, output2] = column_butterfly3_f64([mid0[0], mid1[0], mid2[0]], butterfly3_twiddles);
        let [output3, output4, output5] = column_butterfly3_f64([mid0[1], mid1[1], mid2[1]], butterfly3_twiddles);
        let [output6, output7, output8] = column_butterfly3_f64([mid0[2], mid1[2], mid2[2]], butterfly3_twiddles);
        let [output9, output10,output11]= column_butterfly3_f64([mid0[3], mid1[3], mid2[3]], butterfly3_twiddles);

        [output0, output4, output8, output9, output1, output5, output6, output10, output2, output3, output7, output11]
    }

    // Compute 2 parallel butterfly 12's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly12_f64_lo(rows: [__m128d; 12], butterfly3_twiddles: __m256d, twiddle_config: Rotate90Config<__m256d>) -> [__m128d; 12] {
        // Compute this as a 4x3 FFT. since 4 and 3 are coprime, we can use the good-thomas algorithm. That means crazy reordering of our inputs and outputs, but it also means no twiddle factors
        let merged12 = [
            _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[4]), rows[8], 0x1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[7]), rows[11], 0x1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[10]), rows[2], 0x1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(rows[1]), rows[5], 0x1),
        ];
        let mid0 = column_butterfly4_f64_lo([rows[0], rows[3], rows[6], rows[9]], twiddle_config);
        let mid12 = column_butterfly4_f64(merged12, twiddle_config);

        // extract our merged data
        let mid1 = [
            _mm256_castpd256_pd128(mid12[0]),
            _mm256_castpd256_pd128(mid12[1]),
            _mm256_castpd256_pd128(mid12[2]),
            _mm256_castpd256_pd128(mid12[3]),
        ];
        let mid2 = [
            _mm256_extractf128_pd(mid12[0], 0x1),
            _mm256_extractf128_pd(mid12[1], 0x1),
            _mm256_extractf128_pd(mid12[2], 0x1),
            _mm256_extractf128_pd(mid12[3], 0x1),
        ];

        // merge our half-registers into half as many full registers, so we can do 2 butterfly 3's instead of 3
        let merged0 = [
            _mm256_insertf128_pd(_mm256_castpd128_pd256(mid0[0]), mid0[1], 0x1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(mid1[0]), mid1[1], 0x1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(mid2[0]), mid2[1], 0x1),
        ];
        let merged1 = [
            _mm256_insertf128_pd(_mm256_castpd128_pd256(mid0[2]), mid0[3], 0x1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(mid1[2]), mid1[3], 0x1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(mid2[2]), mid2[3], 0x1),
        ];

        let packed0 = column_butterfly3_f64(merged0, butterfly3_twiddles);
        let packed1 = column_butterfly3_f64(merged1, butterfly3_twiddles);

        // extract our merged data
        let output0 = _mm256_castpd256_pd128(packed0[0]);
        let output1 = _mm256_castpd256_pd128(packed0[1]);
        let output2 = _mm256_castpd256_pd128(packed0[2]);
        let output3 = _mm256_extractf128_pd(packed0[0], 0x1);
        let output4 = _mm256_extractf128_pd(packed0[1], 0x1);
        let output5 = _mm256_extractf128_pd(packed0[2], 0x1);
        let output6 = _mm256_castpd256_pd128(packed1[0]);
        let output7 = _mm256_castpd256_pd128(packed1[1]);
        let output8 = _mm256_castpd256_pd128(packed1[2]);
        let output9 = _mm256_extractf128_pd(packed1[0], 0x1);
        let output10= _mm256_extractf128_pd(packed1[1], 0x1);
        let output11= _mm256_extractf128_pd(packed1[2], 0x1);

        [output0, output4, output8, output9, output1, output5, output6, output10, output2, output3, output7, output11]
    }

    // Compute 2 parallel butterfly 16's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly16_f64( rows: [__m256d; 16], twiddles: [__m256d; 2], twiddle_config: Rotate90Config<__m256d>) -> [__m256d; 16] {
        // Treat our butterfly-16 as a 4x4 array. first, do butterfly 4's down the columns
        let mid0     = column_butterfly4_f64([rows[0], rows[4], rows[8],  rows[12]], twiddle_config);
        let mut mid1 = column_butterfly4_f64([rows[1], rows[5], rows[9],  rows[13]], twiddle_config);
        let mut mid2 = column_butterfly4_f64([rows[2], rows[6], rows[10], rows[14]], twiddle_config);
        let mut mid3 = column_butterfly4_f64([rows[3], rows[7], rows[11], rows[15]], twiddle_config);

        // Apply twiddle factors
        mid1[1] = complex_multiply_f64(mid1[1], twiddles[0]);

        // for twiddle(2, 16), we can use the butterfly8 twiddle1 instead, which takes fewer instructions and fewer multiplies
        mid2[1] = apply_butterfly8_twiddle1(mid2[1], twiddle_config);
        mid1[2] = apply_butterfly8_twiddle1(mid1[2], twiddle_config);

        // for twiddle(3,16), we can use twiddle(1,16), sort of, but we'd need a branch, and at this point it's easier to just have another vector
        mid3[1] = complex_multiply_f64(mid3[1], twiddles[1]);
        mid1[3] = complex_multiply_f64(mid1[3], twiddles[1]);

        // twiddle(4,16) is just a rotate
        mid2[2] = twiddle_config.rotate90(mid2[2]);

        // for twiddle(6, 16), we can use the butterfly8 twiddle3 instead, which takes fewer instructions and fewer multiplies
        mid3[2] = apply_butterfly8_twiddle3(mid3[2], twiddle_config);
        mid2[3] = apply_butterfly8_twiddle3(mid2[3], twiddle_config);

        // twiddle(9, 16) is twiddle (1,16) negated. we're just going to use the same twiddle as (1,16) for now, and apply the negation as a part of our subsequent butterfly 4's
        mid3[3] = complex_multiply_f64(mid3[3], twiddles[0]);

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transpose" and thne apply butterfly 4's across the columns of our 4x4 array
        let output0 = column_butterfly4_f64([mid0[0], mid1[0], mid2[0], mid3[0]], twiddle_config);
        let output1 = column_butterfly4_f64([mid0[1], mid1[1], mid2[1], mid3[1]], twiddle_config);
        let output2 = column_butterfly4_f64([mid0[2], mid1[2], mid2[2], mid3[2]], twiddle_config);
        let output3 = column_butterfly4_negaterow3_f64([mid0[3], mid1[3], mid2[3], mid3[3]], twiddle_config);

        // finally, one more transpose
        [output0[0], output1[0], output2[0], output3[0], output0[1], output1[1], output2[1], output3[1], output0[2], output1[2], output2[2], output3[2], output0[3], output1[3], output2[3], output3[3]]
    }

    // Compute 2 parallel butterfly 16's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly16_f64_lo(rows: [__m128d; 16], twiddles: [__m256d; 2], twiddle_config: Rotate90Config<__m256d>) -> [__m128d; 16] {
        // Treat our butterfly-16 as a 4x4 array. first, do butterfly 4's down the columns
        let mid0     = column_butterfly4_f64_lo([rows[0], rows[4], rows[8],  rows[12]], twiddle_config);
        let mut mid1 = column_butterfly4_f64_lo([rows[1], rows[5], rows[9],  rows[13]], twiddle_config);
        let mut mid2 = column_butterfly4_f64_lo([rows[2], rows[6], rows[10], rows[14]], twiddle_config);
        let mut mid3 = column_butterfly4_f64_lo([rows[3], rows[7], rows[11], rows[15]], twiddle_config);

        // Apply twiddle factors. Note that we're re-using a couple twiddles!
        mid1[1] = complex_multiply_f64_lo(mid1[1], _mm256_castpd256_pd128(twiddles[0]));

        // for twiddle(2, 16), we can use the butterfly8 twiddle1 instead, which takes fewer instructions and fewer multiplies
        mid2[1] = apply_butterfly8_twiddle1_lo(mid2[1], twiddle_config);
        mid1[2] = apply_butterfly8_twiddle1_lo(mid1[2], twiddle_config);

        // for twiddle(3,16), we can use twiddle(1,16), sort of, but we'd need a branch, and at this point it's easier to just have another vector
        mid3[1] = complex_multiply_f64_lo(mid3[1], _mm256_castpd256_pd128(twiddles[1]));
        mid1[3] = complex_multiply_f64_lo(mid1[3], _mm256_castpd256_pd128(twiddles[1]));

        // twiddle(4,16) is just a rotate
        mid2[2] = twiddle_config.rotate90_lo(mid2[2]);

        // for twiddle(6, 16), we can use the butterfly8 twiddle3 instead, which takes fewer instructions and fewer multiplies
        mid3[2] = apply_butterfly8_twiddle3_lo(mid3[2], twiddle_config);
        mid2[3] = apply_butterfly8_twiddle3_lo(mid2[3], twiddle_config);

        // twiddle(9, 16) is twiddle (1,16) negated. we're just going to use the same twiddle as (1,16) for now, and apply the negation as a part of our subsequent butterfly 4's
        mid3[3] = complex_multiply_f64_lo(mid3[3], _mm256_castpd256_pd128(twiddles[0]));

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transpose" and thne apply butterfly 4's across the columns of our 4x4 array
        let output0 = column_butterfly4_f64_lo([mid0[0], mid1[0], mid2[0], mid3[0]], twiddle_config);
        let output1 = column_butterfly4_f64_lo([mid0[1], mid1[1], mid2[1], mid3[1]], twiddle_config);
        let output2 = column_butterfly4_f64_lo([mid0[2], mid1[2], mid2[2], mid3[2]], twiddle_config);
        let output3 = column_butterfly4_negaterow3_f64_lo([mid0[3], mid1[3], mid2[3], mid3[3]], twiddle_config);

        // finally, one more transpose
        [output0[0], output1[0], output2[0], output3[0], output0[1], output1[1], output2[1], output3[1], output0[2], output1[2], output2[2], output3[2], output0[3], output1[3], output2[3], output3[3]]
    }
}