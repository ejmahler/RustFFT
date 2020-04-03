use std::arch::x86_64::*;
use num_complex::Complex;

use ::array_utils::{RawSlice, RawSliceMut};

pub trait AvxComplexArrayf32 {
    unsafe fn load_complex_f32(&self, index: usize) -> __m256;
    unsafe fn load_complex_remainder_f32(&self, remainder_mask: RemainderMask, index: usize) -> __m256;
}
pub trait AvxComplexArrayMutf32 {
    unsafe fn store_complex_f32(&mut self, index: usize, data: __m256);
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
    unsafe fn store_complex_remainder_f32(&mut self, remainder_mask: RemainderMask, data: __m256, index: usize) {
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f32;
        _mm256_maskstore_ps(float_ptr, remainder_mask.0, data);
    }
}
/// A RawSlice is a normal slice, but aliasable. Its functionality is severely limited.

impl AvxComplexArrayf32 for RawSlice<Complex<f32>> {
    #[inline(always)]
    unsafe fn load_complex_f32(&self, index: usize) -> __m256 {
        debug_assert!(index + 4 <= self.len());
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
        debug_assert!(index + 4 <= self.len());
        let float_ptr = self.as_mut_ptr().add(index) as *mut f32;
        _mm256_storeu_ps(float_ptr, data);
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

// Multiply the complex numbers in `left` by the complex numbers in `right`, using FMA instructions where possible
#[inline(always)]
pub unsafe fn complex_multiply_fma_f32(left: __m256, right: __m256) -> __m256 {
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
pub unsafe fn complex_conjugated_multiply_fma_f32(left: __m256, right: __m256) -> __m256 {
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

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Rotate90Config(__m256);
impl Rotate90Config {
    #[inline(always)]
    pub unsafe fn get_from_inverse(is_inverse: bool) -> Self {
        if is_inverse { 
            Self(broadcast_complex_f32(Complex::new(-0.0, 0.0)))
        } else {
            Self(broadcast_complex_f32(Complex::new(0.0, -0.0)))
        }
    }
}

// Apply a multiplication by (0, i) or (0, -i), based on the value of rotation_config. Much faster than an actual multiplication.
#[inline(always)]
pub unsafe fn rotate90_f32(elements: __m256, rotation_config: Rotate90Config) -> __m256 {
    // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an inverse or not
    let elements_swapped = _mm256_permute_ps(elements, 0xB1);

    // We can negate the elements we want by xoring the row with a pre-set vector
    _mm256_xor_ps(elements_swapped, rotation_config.0)
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
}

// Apply a multiplication by (0, i) or (0, -i), based on the value of rotation_config. Much faster than an actual multiplication.
// This variant only applies the rotation to elements 1 and 3, leaving the other two untouched
#[inline(always)]
pub unsafe fn rotate90_oddelements_f32(elements: __m256, rotation_config: Rotate90OddConfig) -> __m256 {
    // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an inverse or not
    let elements_swapped = _mm256_permute_ps(elements, 0xB4);

    // We can negate the elements we want by xoring the row with a pre-set vector
    _mm256_xor_ps(elements_swapped, rotation_config.0)
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
pub unsafe fn column_butterfly4_f32(row0: __m256, row1: __m256, row2: __m256, row3: __m256, twiddle_config: Rotate90Config) -> (__m256, __m256, __m256, __m256) {
    // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
    let (mid0, mid2) = column_butterfly2_f32(row0, row2);
    let (mid1, mid3_pretwiddle) = column_butterfly2_f32(row1, row3);

    // Apply element 3 inner twiddle factor
    let mid3 = rotate90_f32(mid3_pretwiddle, twiddle_config);

    // Perform the second set of size-2 FFTs
    let (output0, output1) = column_butterfly2_f32(mid0, mid1);
    let (output2, output3) = column_butterfly2_f32(mid2, mid3);

    // Swap outputs 1 and 2 in the output to do a square transpose
    (output0, output2, output1, output3)
}

// Compute 4 parallel butterfly 8's using AVX and FMA instructions
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly8_fma_f32(row0: __m256, row1: __m256, row2: __m256, row3: __m256, row4: __m256, row5: __m256, row6: __m256, row7: __m256, twiddles: __m256, twiddle_config: Rotate90Config)  -> (__m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256) {
    // Treat our butterfly-8 as a 2x4 array. first, do butterfly 4's down the columns
    let (mid0, mid2, mid4, mid6) = column_butterfly4_f32(row0, row2, row4, row6, twiddle_config);
    let (mid1, mid3, mid5, mid7) = column_butterfly4_f32(row1, row3, row5, row7, twiddle_config);

    // Apply twiddle factors
    // We want to negate the reals of the twiddles when multiplying mid7, but it's easier to conjugate the twiddles (Ie negate the imaginaries)
    // Negating the reals before amultiplication is equivalent to negating the imaginaries before the multiplication and then negatign the entire result
    // And we can "negate the entire result" by rollign that operation into the subsequent butterfly 2's
    let mid3_twiddled       = complex_multiply_fma_f32(twiddles, mid3);
    let mid5_twiddled       = rotate90_f32(mid5, twiddle_config);
    let mid7_twiddled_neg   = complex_conjugated_multiply_fma_f32(twiddles, mid7);

    // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
    // "transposE" and thne apply butterfly 2's across the columns of our 4x2 array
    let (final0, final1) = column_butterfly2_f32(mid0, mid1);
    let (final2, final3) = column_butterfly2_f32(mid2, mid3_twiddled);
    let (final4, final5) = column_butterfly2_f32(mid4, mid5_twiddled);
    let (final6, final7) = column_butterfly2_f32_negaterow1(mid6, mid7_twiddled_neg); // Finish applying the negation from our twiddles by calling a different butterfly 2 function

    (final0, final2, final4, final6, final1, final3, final5, final7)
}

// Compute 4 parallel butterfly 16's using AVX and FMA instructions
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly16_fma_f32(
    row0: __m256, row1: __m256, row2: __m256, row3: __m256, row4: __m256, row5: __m256, row6: __m256, row7: __m256,
    row8: __m256, row9: __m256, row10: __m256, row11: __m256, row12: __m256, row13: __m256, row14: __m256, row15: __m256,
    twiddles: [__m256; 6], twiddle_config: Rotate90Config) 
-> (__m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256, __m256) 
{
    // Treat our butterfly-16 as a 4x4 array. first, do butterfly 4's down the columns
    let (mid0, mid4, mid8,  mid12) = column_butterfly4_f32(row0, row4, row8,  row12, twiddle_config);
    let (mid1, mid5, mid9,  mid13) = column_butterfly4_f32(row1, row5, row9,  row13, twiddle_config);
    let (mid2, mid6, mid10, mid14) = column_butterfly4_f32(row2, row6, row10, row14, twiddle_config);
    let (mid3, mid7, mid11, mid15) = column_butterfly4_f32(row3, row7, row11, row15, twiddle_config);

    // Apply twiddle factors. Note that we're re-using a couple twiddles!
    let mid5_twiddled   = complex_multiply_fma_f32(twiddles[0], mid5);
    let mid6_twiddled   = complex_multiply_fma_f32(twiddles[1], mid6);
    let mid9_twiddled   = complex_multiply_fma_f32(twiddles[1], mid9);
    let mid7_twiddled   = complex_multiply_fma_f32(twiddles[2], mid7);
    let mid13_twiddled  = complex_multiply_fma_f32(twiddles[2], mid13);
    let mid10_twiddled  = complex_multiply_fma_f32(twiddles[3], mid10);
    let mid11_twiddled  = complex_multiply_fma_f32(twiddles[4], mid11);
    let mid14_twiddled  = complex_multiply_fma_f32(twiddles[4], mid14);

    // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
    // "transposE" and thne apply butterfly 4's across the columns of our 4x4 array
    let (final0,  final1,  final2,  final3)  = column_butterfly4_f32(mid0,  mid1, mid2, mid3, twiddle_config);
    let (final4,  final5,  final6,  final7)  = column_butterfly4_f32(mid4,  mid5_twiddled, mid6_twiddled, mid7_twiddled, twiddle_config);
    let (final8,  final9,  final10, final11) = column_butterfly4_f32(mid8,  mid9_twiddled, mid10_twiddled, mid11_twiddled, twiddle_config);
    let mid15_twiddled  = complex_multiply_fma_f32(twiddles[5], mid15);
    let (final12, final13, final14, final15) = column_butterfly4_f32(mid12, mid13_twiddled, mid14_twiddled, mid15_twiddled, twiddle_config);

    // finally, one more transpose
    (final0, final4, final8, final12, final1, final5, final9, final13, final2, final6, final10, final14, final3, final7, final11, final15)
}


// Treat the input like the rows of a 4x4 array, and transpose said rows to the columns
#[inline(always)]
pub unsafe fn transpose_4x4_f32(row0: __m256, row1: __m256, row2: __m256, row3: __m256) -> (__m256, __m256, __m256, __m256) {
    let unpacked0 = _mm256_unpacklo_ps(row0, row1);
    let unpacked1 = _mm256_unpackhi_ps(row0, row1);
    let unpacked2 = _mm256_unpacklo_ps(row2, row3);
    let unpacked3 = _mm256_unpackhi_ps(row2, row3);

    let swapped0 = _mm256_permute_ps(unpacked0, 0xD8);
    let swapped1 = _mm256_permute_ps(unpacked1, 0xD8);
    let swapped2 = _mm256_permute_ps(unpacked2, 0xD8);
    let swapped3 = _mm256_permute_ps(unpacked3, 0xD8);

    let col0 = _mm256_permute2f128_ps(swapped0, swapped2, 0x20);
    let col1 = _mm256_permute2f128_ps(swapped1, swapped3, 0x20);
    let col2 = _mm256_permute2f128_ps(swapped0, swapped2, 0x31);
    let col3 = _mm256_permute2f128_ps(swapped1, swapped3, 0x31);

    (col0, col1, col2, col3)
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