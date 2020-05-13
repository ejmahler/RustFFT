use std::arch::x86_64::*;
use num_complex::Complex;

use crate::array_utils::{RawSlice, RawSliceMut};

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
}