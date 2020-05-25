use std::arch::x86_64::*;
use num_complex::Complex;

use crate::array_utils::{RawSlice, RawSliceMut};

pub trait AvxComplexArrayf32 {
    unsafe fn load_complex_f32(&self, index: usize) -> __m256;

    unsafe fn load_complex_remainder1_f32(&self, index: usize) -> __m128;
    unsafe fn load_complex_f32_lo(&self, index: usize) -> __m128;
    unsafe fn load_complex_remainder3_f32(&self, index: usize) -> __m256 {
        let lo = self.load_complex_f32_lo(index);
        let hi = self.load_complex_remainder1_f32(index + 2);
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1)
    }


    unsafe fn load_complex_remainder_f32(&self, remainder_mask: RemainderMask, index: usize) -> __m256;
    unsafe fn load_complex_remainder_f32_lo(&self, remainder_mask: RemainderMask, index: usize) -> __m128;
}
pub trait AvxComplexArrayMutf32 {
    // Store the 4 complex numbers contained in `data` at the given memory addresses
    unsafe fn store_complex_f32(&mut self, index: usize, data: __m256);

    // Store the first 2 of 4 complex numbers in `data` at the given memory addresses
    unsafe fn store_complex_remainder1_f32(&mut self, data: __m128, index: usize);
    unsafe fn store_complex_f32_lo(&mut self, data: __m128, index: usize);
    unsafe fn store_complex_remainder3_f32(&mut self, data: __m256, index: usize) {
        self.store_complex_f32_lo(_mm256_castps256_ps128(data), index);
        self.store_complex_remainder1_f32(_mm256_extractf128_ps(data, 1), index + 2);
    }

    // Store some of the 4 complex numbers in `data at the given memory address, using remainder_mask to decide which elements to write
    unsafe fn store_complex_remainder_f32(&mut self, remainder_mask: RemainderMask,  data: __m256, index: usize);
    unsafe fn store_complex_remainder_f32_lo(&mut self, remainder_mask: RemainderMask,  data: __m128, index: usize);
}

impl AvxComplexArrayf32 for [Complex<f32>] {
    #[inline(always)]
    unsafe fn load_complex_f32(&self, index: usize) -> __m256 {
        debug_assert!(self.len() >= index + 4);
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f32;
        _mm256_loadu_ps(float_ptr)
    }
    #[inline(always)]
    unsafe fn load_complex_remainder1_f32(&self, index: usize) -> __m128 {
        debug_assert!(self.len() >= index + 2);
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f32;
        _mm_castpd_ps(_mm_load_sd(float_ptr as *const f64))
    }
    #[inline(always)]
    unsafe fn load_complex_f32_lo(&self, index: usize) -> __m128 {
        debug_assert!(self.len() >= index + 2);
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f32;
        _mm_loadu_ps(float_ptr)
    }
    #[inline(always)]
    unsafe fn load_complex_remainder_f32(&self, remainder_mask: RemainderMask, index: usize) -> __m256 {
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f32;
        _mm256_maskload_ps(float_ptr, remainder_mask.0)
    }
    #[inline(always)]
    unsafe fn load_complex_remainder_f32_lo(&self, remainder_mask: RemainderMask, index: usize) -> __m128 {
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f32;
        _mm_maskload_ps(float_ptr, _mm256_castsi256_si128(remainder_mask.0))
    }
}
impl AvxComplexArrayMutf32 for [Complex<f32>] {
    #[inline(always)]
    unsafe fn store_complex_f32(&mut self, index: usize, data: __m256) {
        debug_assert!(self.len() >= index + 4);
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        _mm256_storeu_ps(float_ptr, data);
    }
    #[inline(always)]
    unsafe fn store_complex_f32_lo(&mut self, data: __m128, index: usize) {
        debug_assert!(self.len() >= index + 2);
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        _mm_storeu_ps(float_ptr, data);
    }
    #[inline(always)]
    unsafe fn store_complex_remainder_f32(&mut self, remainder_mask: RemainderMask, data: __m256, index: usize) {
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        _mm256_maskstore_ps(float_ptr, remainder_mask.0, data);
    }
    #[inline(always)]
    unsafe fn store_complex_remainder1_f32(&mut self, data: __m128, index: usize) {
        debug_assert!(self.len() >= index + 1);

        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        _mm_store_sd(float_ptr as *mut f64, _mm_castps_pd(data));
    }
    #[inline(always)]
    unsafe fn store_complex_remainder_f32_lo(&mut self, remainder_mask: RemainderMask, data: __m128, index: usize) {
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        _mm_maskstore_ps(float_ptr, _mm256_castsi256_si128(remainder_mask.0), data);
    }
}
impl AvxComplexArrayf32 for RawSlice<Complex<f32>> {
    #[inline(always)]
    unsafe fn load_complex_f32(&self, index: usize) -> __m256 {
        debug_assert!(self.len() >= index + 4);
        let float_ptr  = self.as_ptr().add(index) as *const f32;
        _mm256_loadu_ps(float_ptr)
    }
    #[inline(always)]
    unsafe fn load_complex_remainder1_f32(&self, index: usize) -> __m128 {
        debug_assert!(self.len() >= index + 2);
        let float_ptr  = self.as_ptr().add(index) as *const f32;
        _mm_castpd_ps(_mm_load_sd(float_ptr as *const f64))
    }
    #[inline(always)]
    unsafe fn load_complex_f32_lo(&self, index: usize) -> __m128 {
        debug_assert!(self.len() >= index + 2);
        let float_ptr  = self.as_ptr().add(index) as *const f32;
        _mm_loadu_ps(float_ptr)
    }
    #[inline(always)]
    unsafe fn load_complex_remainder_f32(&self, remainder_mask: RemainderMask, index: usize) -> __m256 {
        let float_ptr  = self.as_ptr().add(index) as *const f32;
        _mm256_maskload_ps(float_ptr, remainder_mask.0)
    }
    #[inline(always)]
    unsafe fn load_complex_remainder_f32_lo(&self, remainder_mask: RemainderMask, index: usize) -> __m128 {
        let float_ptr  = self.as_ptr().add(index) as *const f32;
        _mm_maskload_ps(float_ptr, _mm256_castsi256_si128(remainder_mask.0))
    }
}
impl AvxComplexArrayMutf32 for RawSliceMut<Complex<f32>> {
    #[inline(always)]
    unsafe fn store_complex_f32(&mut self, index: usize, data: __m256) {
        debug_assert!(self.len() >= index + 4);
        let float_ptr = self.as_mut_ptr().add(index) as *mut f32;
        _mm256_storeu_ps(float_ptr, data);
    }
    #[inline(always)]
    unsafe fn store_complex_f32_lo(&mut self, data: __m128, index: usize) {
        debug_assert!(self.len() >= index + 2);
        let float_ptr = self.as_mut_ptr().add(index) as *mut f32;
        _mm_storeu_ps(float_ptr, data);
    }
    #[inline(always)]
    unsafe fn store_complex_remainder_f32(&mut self, remainder_mask: RemainderMask, data: __m256, index: usize) {
        let float_ptr = self.as_mut_ptr().add(index) as *mut f32;
        _mm256_maskstore_ps(float_ptr, remainder_mask.0, data);
    }
    #[inline(always)]
    unsafe fn store_complex_remainder1_f32(&mut self, data: __m128, index: usize) {
        debug_assert!(self.len() >= index + 1);

        let float_ptr = self.as_mut_ptr().add(index) as *mut f32;
        _mm_store_sd(float_ptr as *mut f64, _mm_castps_pd(data));
    }
    #[inline(always)]
    unsafe fn store_complex_remainder_f32_lo(&mut self, remainder_mask: RemainderMask, data: __m128, index: usize) {
        let float_ptr = self.as_mut_ptr().add(index) as *mut f32;
        _mm_maskstore_ps(float_ptr, _mm256_castsi256_si128(remainder_mask.0), data);
    }
}

// Struct that encapsulates the process of storing/loading "remainders" for FFT buffers that are not multiples of 4.
// Use with load_remainder_complex_f32 and store_remainder_complex_f32 beloe
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct RemainderMask(__m256i);
impl RemainderMask {
    #[inline(always)]
    pub unsafe fn new_f32(remainder: usize) -> Self {
        let mut mask_array = [0u64; 4];
        for i in 0..remainder {
            mask_array[i] = std::u64::MAX;
        }
        Self(_mm256_lddqu_si256(std::mem::transmute::<*const u64, *const __m256i>(mask_array.as_ptr())))
    }
}

// given a number of elements (assumed to be complex<f32>), divides the elements into chunks that will fit in AVX registers, with a remainder that may not completely fill an AVX register
// returns (num_chunks, remainder) with remainder being in the range [1,4] (if len is 0, num_chunks and remainder will both be 0)
// the intention is that this is used to create a main loop, and then an unconditional remainder after the loop
#[inline(always)]
pub fn compute_chunk_count_complex_f32(len: usize) -> (usize, usize) {
    let quotient = len / 4;
    let naive_remainder = len % 4;

    if quotient > 0 && naive_remainder == 0 {
        (quotient - 1, naive_remainder + 4)
    } else {
        (quotient, naive_remainder)
    }
}

// Fills an AVX register by repeating the given complex number over and over
#[inline(always)]
pub unsafe fn broadcast_complex_f32(value: Complex<f32>) -> __m256 {
    _mm256_set_ps(value.im, value.re, value.im, value.re, value.im, value.re, value.im, value.re)
}

// Does the equivalent of "unpackhi" and "unpacklo" but for complex numbers
#[inline(always)]
pub unsafe fn unpack_complex_f32(row0: __m256, row1: __m256) -> (__m256, __m256) {
    // these two intrinsics compile down to nothing! they're basically transmutes
    let row0_double = _mm256_castps_pd(row0);
    let row1_double = _mm256_castps_pd(row1);

    // unpack as doubles
    let unpacked0 = _mm256_unpacklo_pd(row0_double, row1_double);
    let unpacked1 = _mm256_unpackhi_pd(row0_double, row1_double);

    // re-cast to floats. again, just a transmute, so this compilesdown ot nothing
    let output0 = _mm256_castpd_ps(unpacked0);
    let output1 = _mm256_castpd_ps(unpacked1);

    (output0, output1)
}

// Does the equivalent of "unpackhi" and "unpacklo" but for complex numbers
#[inline(always)]
pub unsafe fn unpack_complex_f32_lo(row0: __m128, row1: __m128) -> (__m128, __m128) {
    // these two intrinsics compile down to nothing! they're basically transmutes
    let row0_double = _mm_castps_pd(row0);
    let row1_double = _mm_castps_pd(row1);

    // unpack as doubles
    let unpacked0 = _mm_unpacklo_pd(row0_double, row1_double);
    let unpacked1 = _mm_unpackhi_pd(row0_double, row1_double);

    // re-cast to floats. again, just a transmute, so this compilesdown ot nothing
    let output0 = _mm_castpd_ps(unpacked0);
    let output1 = _mm_castpd_ps(unpacked1);

    (output0, output1)
}

// Treat the input like the rows of a 4x4 array, and transpose said rows to the columns
#[inline(always)]
pub unsafe fn transpose_4x4_f32(rows: [__m256;4]) -> [__m256;4] {

    let permute0 = _mm256_permute2f128_ps(rows[0], rows[2], 0x20);
    let permute1 = _mm256_permute2f128_ps(rows[1], rows[3], 0x20);
    let permute2 = _mm256_permute2f128_ps(rows[0], rows[2], 0x31);
    let permute3 = _mm256_permute2f128_ps(rows[1], rows[3], 0x31);

    let (unpacked0, unpacked1) = unpack_complex_f32(permute0, permute1);
    let (unpacked2, unpacked3) = unpack_complex_f32(permute2, permute3);

    [unpacked0, unpacked1, unpacked2, unpacked3]
}

// Treat the input like the rows of a 4x8 array, and transpose it to a 8x4 array, where each array of 4 is one set of 4 columns
// The assumption here is that it's very likely that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
// The second array only has two columns of valid data. TODO: make them __m128 instead
#[inline(always)]
pub unsafe fn transpose_4x6_to_6x4_f32(rows: [__m256;6]) -> ([__m256;4], [__m256;4]) {
    let chunk0 = [rows[0], rows[1], rows[2], rows[3]];
    let chunk1 = [rows[4], rows[5], _mm256_setzero_ps(), _mm256_setzero_ps()];

    let output0 = transpose_4x4_f32(chunk0);
    let output1 = transpose_4x4_f32(chunk1);

    (output0, output1)
}

// Treat the input like the rows of a 8x4 array, and transpose it to a 4x8 array
#[inline(always)]
pub unsafe fn transpose_8x4_to_4x8_f32(rows0: [__m256;4], rows1: [__m256;4]) -> [__m256;8] {
    let transposed0 = transpose_4x4_f32(rows0);
    let transposed1 = transpose_4x4_f32(rows1);

    [transposed0[0], transposed0[1], transposed0[2], transposed0[3], transposed1[0], transposed1[1], transposed1[2], transposed1[3]]
}

// Treat the input like the rows of a 9x3 array, and transpose it to a 3x9 array. 
// our parameters are technically 10 columns, not 9 -- we're going to discard the second element of row0
#[inline(always)]
pub unsafe fn transpose_9x3_to_3x9_emptycolumn1_f32(rows0: [__m128;3], rows1: [__m256;3], rows2: [__m256;3]) -> [__m256;9] {

    // the first row of the output will be the first column of the input
    let (unpacked0, _) = unpack_complex_f32_lo(rows0[0], rows0[1]);
    let (unpacked1, _) = unpack_complex_f32_lo(rows0[2], _mm_setzero_ps());
    let output0 = _mm256_insertf128_ps(_mm256_castps128_ps256(unpacked0), unpacked1, 0x1);

    let transposed0 = transpose_4x4_f32([rows1[0], rows1[1], rows1[2], _mm256_setzero_ps()]);
    let transposed1 = transpose_4x4_f32([rows2[0], rows2[1], rows2[2], _mm256_setzero_ps()]);

    [output0, transposed0[0], transposed0[1], transposed0[2], transposed0[3], transposed1[0], transposed1[1], transposed1[2], transposed1[3]]
}


// Treat the input like the rows of a 9x4 array, and transpose it to a 4x9 array. 
// our parameters are technically 10 columns, not 9 -- we're going to discard the second element of row0
#[inline(always)]
pub unsafe fn transpose_9x4_to_4x9_emptycolumn1_f32(rows0: [__m128;4], rows1: [__m256;4], rows2: [__m256;4]) -> [__m256;9] {

    // the first row of the output will be the first column of the input
    let (unpacked0, _) = unpack_complex_f32_lo(rows0[0], rows0[1]);
    let (unpacked1, _) = unpack_complex_f32_lo(rows0[2], rows0[3]);
    let output0 = _mm256_insertf128_ps(_mm256_castps128_ps256(unpacked0), unpacked1, 0x1);

    let transposed0 = transpose_4x4_f32([rows1[0], rows1[1], rows1[2], rows1[3]]);
    let transposed1 = transpose_4x4_f32([rows2[0], rows2[1], rows2[2], rows2[3]]);

    [output0, transposed0[0], transposed0[1], transposed0[2], transposed0[3], transposed1[0], transposed1[1], transposed1[2], transposed1[3]]
}

// Treat the input like the rows of a 9x4 array, and transpose it to a 4x9 array. 
// our parameters are technically 10 columns, not 9 -- we're going to discard the second element of row0
#[inline(always)]
pub unsafe fn transpose_9x6_to_6x9_emptycolumn1_f32(rows0: [__m128;6], rows1: [__m256;6], rows2: [__m256;6]) -> ([__m256;9], [__m128;9])  {

    // the first row of the output will be the first column of the input
    let (unpacked0, _) = unpack_complex_f32_lo(rows0[0], rows0[1]);
    let (unpacked1, _) = unpack_complex_f32_lo(rows0[2], rows0[3]);
    let (unpacked2, _) = unpack_complex_f32_lo(rows0[4], rows0[5]);
    let output0 = _mm256_insertf128_ps(_mm256_castps128_ps256(unpacked0), unpacked1, 0x1);

    let transposed_hi0 = transpose_4x4_f32([rows1[0], rows1[1], rows1[2], rows1[3]]);
    let transposed_hi1 = transpose_4x4_f32([rows2[0], rows2[1], rows2[2], rows2[3]]);

    let (unpacked_bottom0, unpacked_bottom1) = unpack_complex_f32(rows1[4], rows1[5]);
    let (unpacked_bottom2, unpacked_bottom3) = unpack_complex_f32(rows2[4], rows2[5]);

    let transposed_lo = [
        unpacked2,
        _mm256_castps256_ps128(unpacked_bottom0),
        _mm256_castps256_ps128(unpacked_bottom1),
        _mm256_extractf128_ps(unpacked_bottom0, 0x1),
        _mm256_extractf128_ps(unpacked_bottom1, 0x1),
        _mm256_castps256_ps128(unpacked_bottom2),
        _mm256_castps256_ps128(unpacked_bottom3),
        _mm256_extractf128_ps(unpacked_bottom2, 0x1),
        _mm256_extractf128_ps(unpacked_bottom3, 0x1),
    ];

    (
        [output0, transposed_hi0[0], transposed_hi0[1], transposed_hi0[2], transposed_hi0[3], transposed_hi1[0], transposed_hi1[1], transposed_hi1[2], transposed_hi1[3]],
        transposed_lo
    )
}

// Treat the input like the rows of a 12x4 array, and transpose it to a 4x12 array
// The assumption here is that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_12x4_to_4x12_f32(rows0: [__m256;4], rows1: [__m256;4], rows2: [__m256;4]) -> [__m256;12] {
    let transposed0 = transpose_4x4_f32(rows0);
    let transposed1 = transpose_4x4_f32(rows1);
    let transposed2 = transpose_4x4_f32(rows2);

    [transposed0[0], transposed0[1], transposed0[2], transposed0[3], transposed1[0], transposed1[1], transposed1[2], transposed1[3], transposed2[0], transposed2[1], transposed2[2], transposed2[3]]
}

// Treat the input like the rows of a 12x6 array, and transpose it to a 6x12 array
// The assumption here is that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_12x6_to_6x12_f32(rows0: [__m256;6], rows1: [__m256;6], rows2: [__m256;6]) -> ([__m128;12], [__m256;12]) {
    let (unpacked0, unpacked1) = unpack_complex_f32(rows0[0], rows0[1]);
    let (unpacked2, unpacked3) = unpack_complex_f32(rows1[0], rows1[1]);
    let (unpacked4, unpacked5) = unpack_complex_f32(rows2[0], rows2[1]);

    let output0 = [
        _mm256_castps256_ps128(unpacked0),
        _mm256_castps256_ps128(unpacked1),
        _mm256_extractf128_ps(unpacked0, 1),
        _mm256_extractf128_ps(unpacked1, 1),
        _mm256_castps256_ps128(unpacked2),
        _mm256_castps256_ps128(unpacked3),
        _mm256_extractf128_ps(unpacked2, 1),
        _mm256_extractf128_ps(unpacked3, 1),
        _mm256_castps256_ps128(unpacked4),
        _mm256_castps256_ps128(unpacked5),
        _mm256_extractf128_ps(unpacked4, 1),
        _mm256_extractf128_ps(unpacked5, 1),
    ];
    let transposed0 = transpose_4x4_f32([rows0[2], rows0[3], rows0[4], rows0[5]]);
    let transposed1 = transpose_4x4_f32([rows1[2], rows1[3], rows1[4], rows1[5]]);
    let transposed2 = transpose_4x4_f32([rows2[2], rows2[3], rows2[4], rows2[5]]);

    let output1 = [
        transposed0[0],
        transposed0[1],
        transposed0[2],
        transposed0[3],
        transposed1[0],
        transposed1[1],
        transposed1[2],
        transposed1[3],
        transposed2[0],
        transposed2[1],
        transposed2[2],
        transposed2[3],
    ];

    (output0, output1)
}

// Treat the input like the rows of a 8x8 array, and transpose said rows to the columns
// The assumption here is that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_8x8_f32(rows0: [__m256;8], rows1: [__m256;8]) -> ([__m256;8], [__m256;8]) {
    let chunk00 = [rows0[0],  rows0[1],  rows0[2],  rows0[3]];
    let chunk01 = [rows0[4],  rows0[5],  rows0[6],  rows0[7]];
    let chunk10 = [rows1[0],  rows1[1],  rows1[2],  rows1[3]];
    let chunk11 = [rows1[4],  rows1[5],  rows1[6],  rows1[7]];

    let transposed00 = transpose_4x4_f32(chunk00);
    let transposed01 = transpose_4x4_f32(chunk10);
    let transposed10 = transpose_4x4_f32(chunk01);
    let transposed11 = transpose_4x4_f32(chunk11);

    let output0 = [transposed00[0], transposed00[1], transposed00[2], transposed00[3], transposed01[0], transposed01[1], transposed01[2], transposed01[3]];
    let output1 = [transposed10[0], transposed10[1], transposed10[2], transposed10[3], transposed11[0], transposed11[1], transposed11[2], transposed11[3]];

    (output0, output1)
}

// utility for packing a 3x4 array into just 3 registers, preserving order
#[inline(always)]
pub unsafe fn pack_3x4_4x3_f32(rows: [__m256;4]) -> [__m256; 3] {
    let (unpacked_lo, _) = unpack_complex_f32(rows[2], rows[1]); // 5 8 3 6

    // copy the lower half of row 1 into the upper half, then swap even and od values
    let row1_duplicated = _mm256_insertf128_ps(rows[1], _mm256_castps256_ps128(rows[1]), 0x1); // 4 3 4 3
    let row1_dupswap = _mm256_permute_ps(row1_duplicated, 0x4E); // 3 4 3 4
    
    // both lanes of row 2 have some data they want to swap to the other side
    let row2_swapped = _mm256_permute2f128_ps(rows[2], unpacked_lo, 0x03); // 7 6 5 8
    
    // none of the data in row 3 is in the right position, and it'll take several instructions to get there
    let permuted3 = _mm256_permute_ps(rows[3], 0x4E); // swap even and odd complex numbers // 11 _ 9 10
    let intermediate3 = _mm256_insertf128_ps(row2_swapped, _mm256_castps256_ps128(permuted3), 0x1); // 9 10 5 8
    
    let output0 = _mm256_blend_ps(rows[0], row1_dupswap, 0xC0); // 3 2 1 0
    let output1 = _mm256_blend_ps(row2_swapped, row1_dupswap, 0x03); // 7 6 5 4
    let output2 = _mm256_blend_ps(permuted3, intermediate3, 0x33); // 11 10 9 8

    [output0, output1, output2]
}

// Functions in the "FMA" sub-module require the fma instruction set in addition to AVX
pub mod fma {
    use super::*;

    // Multiply the complex numbers in `left` by the complex numbers in `right`, using FMA instructions where possible
    #[inline(always)]
    pub unsafe fn complex_multiply_f32(left: __m256, right: __m256) -> __m256 {
        // Extract the real and imaginary components from left into 2 separate registers
        let left_real = _mm256_moveldup_ps(left);
        let left_imag = _mm256_movehdup_ps(left);

        // create a shuffled version of right where the imaginary values are swapped with the reals
        let right_shuffled = _mm256_permute_ps(right, 0xB1);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = _mm256_mul_ps(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then alternatingly add and subtract the left side from the right
        _mm256_fmaddsub_ps(left_real, right, output_right)
    }

    // Multiply the complex numbers in `left` by the complex numbers in `right`, using FMA instructions where possible
    // This variant assumes that `left` should be conjugated before multiplying (IE, the imaginary numbers in `left` should be negated)
    // Thankfully, it is straightforward to roll this into existing instructions. Namely, we can get away with replacing "fmaddsub" with "fmsubadd"
    #[inline(always)]
    pub unsafe fn complex_conjugated_multiply_f32(left: __m256, right: __m256) -> __m256 {
        // Extract the real and imaginary components from left into 2 separate registers
        let left_real = _mm256_moveldup_ps(left);
        let left_imag = _mm256_movehdup_ps(left);

        // create a shuffled version of right where the imaginary values are swapped with the reals
        let right_shuffled = _mm256_permute_ps(right, 0xB1);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = _mm256_mul_ps(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then alternatingly add and subtract the left side from the right
        _mm256_fmsubadd_ps(left_real, right, output_right)
    }

    // compute buffer[i] = buffer[i].conj() * multiplier[i] pairwise complex multiplication for each element.
    // This is kind of usage-specific, because 'b' is stored as pre-loaded AVX registers, but 'a' is stored as loose complex numbers
    #[target_feature(enable = "avx", enable = "fma")]
    pub unsafe fn pairwise_complex_multiply_conjugated(buffer: &mut [Complex<f32>], multiplier: &[__m256]) {
        for (i, twiddle) in multiplier.iter().enumerate() {
            let inner_vector = buffer.load_complex_f32(i*4);
            let product_vector = complex_conjugated_multiply_f32(inner_vector, *twiddle);
            buffer.store_complex_f32(i*4, product_vector);
        }
    }
}