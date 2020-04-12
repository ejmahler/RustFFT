use std::arch::x86_64::*;
use num_complex::Complex;

use ::array_utils::{RawSlice, RawSliceMut};

pub trait AvxComplexArrayf32 {
    unsafe fn load_complex_f32(&self, index: usize) -> __m256;
    unsafe fn load_complex_remainder_f32(&self, remainder_mask: RemainderMask, index: usize) -> __m256;
}
pub trait AvxComplexArrayMutf32 {
    // Store the 4 complex numbers contained in `data` at the given memory addresses
    unsafe fn store_complex_f32(&mut self, index: usize, data: __m256);

    // Store the first 2 of 4 complex numbers in `data` at the given memory addresses
    unsafe fn store_complex_f32_lo(&mut self, data: __m256, index: usize);

    // Store some of the 4 complex numbers in `data at the given memory address, using remainder_mask to decide which elements to write
    unsafe fn store_complex_remainder_f32(&mut self, remainder_mask: RemainderMask,  data: __m256, index: usize);
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
    unsafe fn load_complex_remainder_f32(&self, remainder_mask: RemainderMask, index: usize) -> __m256 {
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f32;
        _mm256_maskload_ps(float_ptr, remainder_mask.0)
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
    unsafe fn store_complex_f32_lo(&mut self, data: __m256, index: usize) {
        debug_assert!(self.len() >= index + 2);
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        let half_data = _mm256_castps256_ps128(data);
        _mm_storeu_ps(float_ptr, half_data);
    }
    #[inline(always)]
    unsafe fn store_complex_remainder_f32(&mut self, remainder_mask: RemainderMask, data: __m256, index: usize) {
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        _mm256_maskstore_ps(float_ptr, remainder_mask.0, data);
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
    unsafe fn load_complex_remainder_f32(&self, remainder_mask: RemainderMask, index: usize) -> __m256 {
        let float_ptr  = self.as_ptr().add(index) as *const f32;
        _mm256_maskload_ps(float_ptr, remainder_mask.0)
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
    unsafe fn store_complex_f32_lo(&mut self, data: __m256, index: usize) {
        debug_assert!(self.len() >= index + 2);
        let float_ptr = self.as_mut_ptr().add(index) as *mut f32;
        let half_data = _mm256_castps256_ps128(data);
        _mm_storeu_ps(float_ptr, half_data);
    }
    #[inline(always)]
    unsafe fn store_complex_remainder_f32(&mut self, remainder_mask: RemainderMask, data: __m256, index: usize) {
        let float_ptr = self.as_mut_ptr().add(index) as *mut f32;
        _mm256_maskstore_ps(float_ptr, remainder_mask.0, data);
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

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Rotate90Config<V>(V);
impl Rotate90Config<__m256> {
    #[inline(always)]
    pub unsafe fn new_f32(is_inverse: bool) -> Self {
        if is_inverse { 
            Self(broadcast_complex_f32(Complex::new(-0.0, 0.0)))
        } else {
            Self(broadcast_complex_f32(Complex::new(0.0, -0.0)))
        }
    }

    // Apply a multiplication by (0, i) or (0, -i), based on the value of rotation_config. Much faster than an actual multiplication.
    #[inline(always)]
    pub unsafe fn rotate90(&self, elements: __m256) -> __m256 {
        // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an inverse or not
        let elements_swapped = _mm256_permute_ps(elements, 0xB1);

        // We can negate the elements we want by xoring the row with a pre-set vector
        _mm256_xor_ps(elements_swapped, self.0)
    }
}
use super::avx64_utils::broadcast_complex_f64;
impl Rotate90Config<__m256d> {
    #[inline(always)]
    pub unsafe fn new_f64(is_inverse: bool) -> Self {
        if is_inverse { 
            Self(broadcast_complex_f64(Complex::new(-0.0, 0.0)))
        } else {
            Self(broadcast_complex_f64(Complex::new(0.0, -0.0)))
        }
    }

    // Apply a multiplication by (0, i) or (0, -i), based on the value of rotation_config. Much faster than an actual multiplication.
    #[inline(always)]
    pub unsafe fn rotate90(&self, elements: __m256d) -> __m256d {
        // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an inverse or not
        let elements_swapped = _mm256_permute_pd(elements, 0x05);

        // We can negate the elements we want by xoring the row with a pre-set vector
        _mm256_xor_pd(elements_swapped, self.0)
    }
}
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Rotate90OddConfig(__m256);
impl Rotate90OddConfig {
    #[inline(always)]
    pub unsafe fn get_from_inverse(is_inverse: bool) -> Self {
        if is_inverse { 
            Self([Complex::new(0.0, 0.0), Complex::new(-0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(-0.0, 0.0)].load_complex_f32(0))
        } else {
            Self([Complex::new(0.0, 0.0), Complex::new(0.0, -0.0), Complex::new(0.0, 0.0), Complex::new(0.0, -0.0)].load_complex_f32(0))
        }
    }

    // Apply a multiplication by (0, i) or (0, -i), based on the value of rotation_config. Much faster than an actual multiplication.
    // This variant only applies the rotation to elements 1 and 3, leaving the other two untouched
    #[inline(always)]
    pub unsafe fn rotate90_oddelements(&self, elements: __m256) -> __m256 {
        // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an inverse or not
        let elements_swapped = _mm256_permute_ps(elements, 0xB4);

        // We can negate the elements we want by xoring the row with a pre-set vector
        _mm256_xor_ps(elements_swapped, self.0)
    }
}

// Compute 4 parallel butterfly 2's using AVX instructions
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly2_f32(row0: __m256, row1: __m256) -> (__m256, __m256) {
    let output0 = _mm256_add_ps(row0, row1);
    let output1 = _mm256_sub_ps(row0, row1);

    (output0, output1)
}

// Compute 4 parallel butterfly 2's using AVX instructions. This variant rolls in a negation of row 1
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly2_f32_negaterow1(row0: __m256, row1: __m256) -> (__m256, __m256) {
    let output0 = _mm256_sub_ps(row0, row1);
    let output1 = _mm256_add_ps(row0, row1);

    (output0, output1)
}

// Compute 4 parallel butterfly 4's using AVX instructions
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly4_f32(rows: [__m256;4], twiddle_config: Rotate90Config<__m256>) -> [__m256;4] {
    // Perform the first set of size-2 FFTs.
    let (mid0, mid2) = column_butterfly2_f32(rows[0], rows[2]);
    let (mid1, mid3) = column_butterfly2_f32(rows[1], rows[3]);

    // Apply element 3 inner twiddle factor
    let mid3_rotated = twiddle_config.rotate90(mid3);

    // Perform the second set of size-2 FFTs
    let (output0, output1) = column_butterfly2_f32(mid0, mid1);
    let (output2, output3) = column_butterfly2_f32(mid2, mid3_rotated);

    // Swap outputs 1 and 2 in the output to do a square transpose
    [output0, output2, output1, output3]
}

// Treat the input like the rows of a 4x3 array, and transpose said rows to the columns.
// But, since the output has 3 columns while AVX registers have 4 columns, we shift elements around so that they're stored contiguously in 3 registers, hence the word "packed"
#[allow(unused)]
#[inline(always)]
pub unsafe fn transpose_4x3_packed_f32(row0: __m256, row1: __m256, row2: __m256) -> (__m256, __m256, __m256) {
    let (unpacked0, _) = unpack_complex_f32(row0, row1);
    let (_, unpacked2) = unpack_complex_f32(row1, row2);
    
    // output0 and output2 each need to swap some elements. thankfully we can blend those elements into the same intermediate value, and then do a permute 128 from there
    let blended = _mm256_blend_ps(row0, row2, 0x33);
    
    let output1 = _mm256_permute2f128_ps(unpacked0, unpacked2, 0x12);
    
    let output0 = _mm256_permute2f128_ps(unpacked0, blended, 0x20);
    let output2 = _mm256_permute2f128_ps(unpacked2, blended, 0x13);

    (output0, output1, output2)
}

// Treat the input like the rows of a 4x4 array, and transpose said rows to the columns
#[inline(always)]
pub unsafe fn transpose_4x4_f32(rows: [__m256;4]) -> [__m256;4] {
    let (unpacked0, unpacked1) = unpack_complex_f32(rows[0], rows[1]);
    let (unpacked2, unpacked3) = unpack_complex_f32(rows[2], rows[3]);

    let col0 = _mm256_permute2f128_ps(unpacked0, unpacked2, 0x20);
    let col1 = _mm256_permute2f128_ps(unpacked1, unpacked3, 0x20);
    let col2 = _mm256_permute2f128_ps(unpacked0, unpacked2, 0x31);
    let col3 = _mm256_permute2f128_ps(unpacked1, unpacked3, 0x31);

    [col0, col1, col2, col3]
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

// Treat the input like the rows of a 4x8 array, and transpose it to a 8x4 array, where each array of 4 is one set of 4 columns
// The assumption here is that it's very likely that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_4x8_to_8x4_f32(rows: [__m256;8]) -> ([__m256;4], [__m256;4]) {
    let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3]];
    let chunk1 = [rows[4],  rows[5],  rows[6],  rows[7]];

    let output0 = transpose_4x4_f32(chunk0);
    let output1 = transpose_4x4_f32(chunk1);

    (output0, output1)
}

// Treat the input like the rows of a 12x4 array, and transpose it to a 4x12 array
#[inline(always)]
pub unsafe fn transpose_8x4_to_4x8_f32(rows0: [__m256;4], rows1: [__m256;4]) -> [__m256;8] {
    let transposed0 = transpose_4x4_f32(rows0);
    let transposed1 = transpose_4x4_f32(rows1);

    [transposed0[0], transposed0[1], transposed0[2], transposed0[3], transposed1[0], transposed1[1], transposed1[2], transposed1[3]]
}

// Treat the input like the rows of a 12x4 array, and transpose it to a 4x12 array
#[inline(always)]
pub unsafe fn transpose_12x4_to_4x12_f32(rows0: [__m256;4], rows1: [__m256;4], rows2: [__m256;4]) -> [__m256;12] {
    let transposed0 = transpose_4x4_f32(rows0);
    let transposed1 = transpose_4x4_f32(rows1);
    let transposed2 = transpose_4x4_f32(rows2);

    [transposed0[0], transposed0[1], transposed0[2], transposed0[3], transposed1[0], transposed1[1], transposed1[2], transposed1[3], transposed2[0], transposed2[1], transposed2[2], transposed2[3]]
}

// Treat the input like the rows of a 8x8 array, and transpose said rows to the columns
// The assumption here is that it's very likely that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
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

// Treat the input like the rows of a 4x16 array, and transpose it to a 16x4 array, where each array of 4 is one set of 4 columns
// The assumption here is that it's very likely that the caller wants to do some more AVX operations on the columns of the transposed array, so the output is arranged to make that more convenient
#[inline(always)]
pub unsafe fn transpose_4x16_to_16x4_f32(rows: [__m256;16]) -> ([__m256;4], [__m256;4], [__m256;4], [__m256;4]) {
    let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3]];
    let chunk1 = [rows[4],  rows[5],  rows[6],  rows[7]];
    let chunk2 = [rows[8],  rows[9],  rows[10], rows[11]];
    let chunk3 = [rows[12], rows[13], rows[14], rows[15]];

    let output0 = transpose_4x4_f32(chunk0);
    let output1 = transpose_4x4_f32(chunk1);
    let output2 = transpose_4x4_f32(chunk2);
    let output3 = transpose_4x4_f32(chunk3);

    (output0, output1, output2, output3)
}

// Split the array into evens and odds
#[inline(always)]
pub unsafe fn split_evens_odds_f32(row0: __m256, row1: __m256) -> (__m256, __m256) {
    let permuted0 = _mm256_permute2f128_ps(row0, row1, 0x20);
    let permuted1 = _mm256_permute2f128_ps(row0, row1, 0x31);
    let unpacked1 = _mm256_unpackhi_ps(permuted0, permuted1);
    let unpacked0 = _mm256_unpacklo_ps(permuted0, permuted1);
    let output1 = _mm256_permute_ps(unpacked1, 0xD8);
    let output0 = _mm256_permute_ps(unpacked0, 0xD8);

    (output0, output1)
}

// Interleave even elements and odd elements into a single array
#[inline(always)]
pub unsafe fn interleave_evens_odds_f32(row0: __m256, row1: __m256) -> (__m256, __m256) {
    let unpacked0 = _mm256_unpacklo_ps(row0, row1);
    let unpacked1 = _mm256_unpackhi_ps(row0, row1);
    let permuted0 = _mm256_permute_ps(unpacked0, 0xD8);
    let permuted1 = _mm256_permute_ps(unpacked1, 0xD8);
    let output0 = _mm256_permute2f128_ps(permuted0, permuted1, 0x20);
    let output1 = _mm256_permute2f128_ps(permuted0, permuted1, 0x31);
    
    (output0, output1)
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

    // Compute 4 parallel butterfly 3's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly3_f32(row0: __m256, row1: __m256, row2: __m256, twiddles: __m256) -> (__m256, __m256, __m256) {
        let (mid1_pretwiddle, mid2_pretwiddle) = column_butterfly2_f32(row1, row2);
        let output0 = _mm256_add_ps(row0, mid1_pretwiddle);

        let twiddle_real = _mm256_moveldup_ps(twiddles);
        let twiddle_imag = _mm256_movehdup_ps(twiddles);
        
        let mid1 = _mm256_fmadd_ps(mid1_pretwiddle, twiddle_real, row0);

        let mid2_rotated = Rotate90Config::new_f32(true).rotate90(mid2_pretwiddle);
        let mid2 = _mm256_mul_ps(mid2_rotated, twiddle_imag);

        let (output1, output2) = column_butterfly2_f32(mid1, mid2);

        (output0, output1, output2)
    }

    // Compute 4 parallel butterfly 6's using AVX instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly6_f32(rows: [__m256; 6], butterfly3_twiddles: __m256) -> [__m256; 6] {
        // We're going good-thomas algorithm. We can reorder the inputs and outputs in such a way that we don't need twiddle factors!
        let (mid0, mid2, mid4) = column_butterfly3_f32(rows[0], rows[2], rows[4], butterfly3_twiddles);
        let (mid1, mid3, mid5) = column_butterfly3_f32(rows[3], rows[5], rows[1], butterfly3_twiddles);

        // transpose the data and do butterfly 2's
        let (output0, output1) = column_butterfly2_f32(mid0, mid1);
        let (output2, output3) = column_butterfly2_f32(mid2, mid3);
        let (output4, output5) = column_butterfly2_f32(mid4, mid5);

        // reorder into output
        [output0, output3, output4, output1, output2, output5]
    }

    // Compute 4 parallel butterfly 8's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly8_f32(rows: [__m256;8], twiddles: __m256, twiddle_config: Rotate90Config<__m256>)  -> [__m256;8] {
        // Treat our butterfly-8 as a 2x4 array. first, do butterfly 4's down the columns
        let mid0     = column_butterfly4_f32([rows[0], rows[2], rows[4], rows[6]], twiddle_config);
        let mut mid1 = column_butterfly4_f32([rows[1], rows[3], rows[5], rows[7]], twiddle_config);

        // Apply twiddle factors
        // We want to negate the reals of the twiddles when multiplying mid7, but it's easier to conjugate the twiddles (Ie negate the imaginaries)
        // Negating the reals before amultiplication is equivalent to negating the imaginaries before the multiplication and then negatign the entire result
        // And we can "negate the entire result" by rollign that operation into the subsequent butterfly 2's
        mid1[1] = complex_multiply_f32(twiddles, mid1[1]);
        mid1[2] = twiddle_config.rotate90(mid1[2]);
        mid1[3] = complex_conjugated_multiply_f32(twiddles, mid1[3]);

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transpose" and thne apply butterfly 2's across the columns of our 4x2 array
        let (output0, output1) = column_butterfly2_f32(mid0[0], mid1[0]);
        let (output2, output3) = column_butterfly2_f32(mid0[1], mid1[1]);
        let (output4, output5) = column_butterfly2_f32(mid0[2], mid1[2]);
        let (output6, output7) = column_butterfly2_f32_negaterow1(mid0[3], mid1[3]);// Finish applying the negation from our twiddles by calling a different butterfly 2 function

        [output0, output2, output4, output6, output1, output3, output5, output7]
    }

    // Compute 4 parallel butterfly 12's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly12_f32(rows: [__m256; 12], butterfly3_twiddles: __m256, twiddle_config: Rotate90Config<__m256>) -> [__m256; 12] {
        // Compute this as a 4x3 FFT. since 4 and 3 are coprime, we can use the good-thomas algorithm. That means crazy reordering of our inputs and outputs, but it also means no twiddle factors
        let mid0 = column_butterfly4_f32([rows[0], rows[3], rows[6], rows[9]], twiddle_config);
        let mid1 = column_butterfly4_f32([rows[4], rows[7], rows[10],rows[1]], twiddle_config);
        let mid2 = column_butterfly4_f32([rows[8], rows[11],rows[2], rows[5]], twiddle_config);

        let (output0, output1, output2) = column_butterfly3_f32(mid0[0], mid1[0], mid2[0], butterfly3_twiddles);
        let (output3, output4, output5) = column_butterfly3_f32(mid0[1], mid1[1], mid2[1], butterfly3_twiddles);
        let (output6, output7, output8) = column_butterfly3_f32(mid0[2], mid1[2], mid2[2], butterfly3_twiddles);
        let (output9, output10,output11)= column_butterfly3_f32(mid0[3], mid1[3], mid2[3], butterfly3_twiddles);

        [output0, output4, output8, output9, output1, output5, output6, output10, output2, output3, output7, output11]
    }

    // Compute 4 parallel butterfly 16's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly16_f32(rows: [__m256; 16], twiddles: [__m256; 6], twiddle_config: Rotate90Config<__m256>) -> [__m256; 16] {
        // Treat our butterfly-16 as a 4x4 array. first, do butterfly 4's down the columns
        let mid0     = column_butterfly4_f32([rows[0], rows[4], rows[8],  rows[12]], twiddle_config);
        let mut mid1 = column_butterfly4_f32([rows[1], rows[5], rows[9],  rows[13]], twiddle_config);
        let mut mid2 = column_butterfly4_f32([rows[2], rows[6], rows[10], rows[14]], twiddle_config);
        let mut mid3 = column_butterfly4_f32([rows[3], rows[7], rows[11], rows[15]], twiddle_config);

        // Apply twiddle factors. Note that we're re-using a couple twiddles!
        mid1[1] = complex_multiply_f32(twiddles[0], mid1[1]);
        mid2[1] = complex_multiply_f32(twiddles[1], mid2[1]);
        mid1[2] = complex_multiply_f32(twiddles[1], mid1[2]);
        mid3[1] = complex_multiply_f32(twiddles[2], mid3[1]);
        mid1[3] = complex_multiply_f32(twiddles[2], mid1[3]);
        mid2[2] = complex_multiply_f32(twiddles[3], mid2[2]);
        mid3[2] = complex_multiply_f32(twiddles[4], mid3[2]);
        mid2[3] = complex_multiply_f32(twiddles[4], mid2[3]);
        mid3[3] = complex_multiply_f32(twiddles[5], mid3[3]);

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transpose" and thne apply butterfly 4's across the columns of our 4x4 array
        let output0 = column_butterfly4_f32([mid0[0], mid1[0], mid2[0], mid3[0]], twiddle_config);
        let output1 = column_butterfly4_f32([mid0[1], mid1[1], mid2[1], mid3[1]], twiddle_config);
        let output2 = column_butterfly4_f32([mid0[2], mid1[2], mid2[2], mid3[2]], twiddle_config);
        let output3 = column_butterfly4_f32([mid0[3], mid1[3], mid2[3], mid3[3]], twiddle_config);

        // finally, one more transpose
        [output0[0], output1[0], output2[0], output3[0], output0[1], output1[1], output2[1], output3[1], output0[2], output1[2], output2[2], output3[2], output0[3], output1[3], output2[3], output3[3]]
    }
}