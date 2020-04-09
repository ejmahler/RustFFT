use std::arch::x86_64::*;
use num_complex::Complex;

use ::array_utils::{RawSlice, RawSliceMut};
use super::avx32_utils::Rotate90Config;

pub trait AvxComplexArray64 {
    unsafe fn load_complex_f64(&self, index: usize) -> __m256d;
    unsafe fn load_complex_f64_lo(&self, index: usize) -> __m256d;
}
pub trait AvxComplexArrayMut64 {
    // Store the 4 complex numbers contained in `data` at the given memory addresses
    unsafe fn store_complex_f64(&mut self, data: __m256d, index: usize);

    // Store the first 2 of 4 complex numbers in `data` at the given memory addresses
    unsafe fn store_complex_f64_lo(&mut self, data: __m256d, index: usize);
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
    unsafe fn load_complex_f64_lo(&self, index: usize) -> __m256d {
        debug_assert!(self.len() > index);
        let complex_ref = self.get_unchecked(index);
        let float_ptr  = (&complex_ref.re) as *const f64;
        let lo = _mm_loadu_pd(float_ptr);
        _mm256_zextpd128_pd256(lo)
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
    unsafe fn store_complex_f64_lo(&mut self, data: __m256d, index: usize) {
        debug_assert!(self.len() > index);
        let complex_ref = self.get_unchecked_mut(index);
        let float_ptr = (&mut complex_ref.re) as *mut f64;
        let half_data = _mm256_castpd256_pd128(data);
        _mm_storeu_pd(float_ptr, half_data);
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
    unsafe fn load_complex_f64_lo(&self, index: usize) -> __m256d {
        debug_assert!(self.len() > index);
        let float_ptr  = self.as_ptr().add(index) as *const f64;
        let lo = _mm_loadu_pd(float_ptr);
        _mm256_zextpd128_pd256(lo)
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
    unsafe fn store_complex_f64_lo(&mut self, data: __m256d, index: usize) {
        debug_assert!(self.len() > index);
        let float_ptr = self.as_mut_ptr().add(index) as *mut f64;
        let half_data = _mm256_castpd256_pd128(data);
        _mm_storeu_pd(float_ptr, half_data);
    }
}

// Fills an AVX register by repeating the given complex number over and over
#[inline(always)]
pub unsafe fn broadcast_complex_f64(value: Complex<f64>) -> __m256d {
    _mm256_set_pd(value.im, value.re, value.im, value.re)
}

// Compute 2 parallel butterfly 2's using AVX instructions
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly2_f64(row0: __m256d, row1: __m256d) -> (__m256d, __m256d) {
    let output0 = _mm256_add_pd(row0, row1);
    let output1 = _mm256_sub_pd(row0, row1);

    (output0, output1)
}

// Compute 2 parallel butterfly 2's using AVX instructions. This variant rolls in a negation of row 1
// rowN contains the nth element of each parallel FFT
#[inline(always)]
pub unsafe fn column_butterfly2_f64_negaterow1(row0: __m256d, row1: __m256d) -> (__m256d, __m256d) {
    let output0 = _mm256_sub_pd(row0, row1);
    let output1 = _mm256_add_pd(row0, row1);

    (output0, output1)
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
#[inline(always)]
pub unsafe fn column_butterfly4_split_f64(row0_real: __m256d, row0_imag: __m256d, row1_real: __m256d, row1_imag: __m256d, row2_real: __m256d, row2_imag: __m256d, row3_real: __m256d, row3_imag: __m256d) 
-> (__m256d, __m256d, __m256d, __m256d, __m256d, __m256d, __m256d, __m256d) {
    // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
    let (mid0_real, mid2_real) = column_butterfly2_f64(row0_real, row2_real);
    let (mid1_real, mid3_real) = column_butterfly2_f64(row1_real, row3_real);
    let (mid0_imag, mid2_imag) = column_butterfly2_f64(row0_imag, row2_imag);
    let (mid1_imag, mid3_imag) = column_butterfly2_f64(row1_imag, row3_imag);

    // Apply element 3 inner twiddle factor. we do this by swapping the reals and imaginaries of element 3
    // then negating the imaginaries. But we'll roll the negation int the following instructions.
    let (mid3_real, mid3_imag_neg) = (mid3_imag, mid3_real);

    // Perform the second set of size-2 FFTs
    let (output0_real, output1_real) = column_butterfly2_f64(mid0_real, mid1_real);
    let (output2_real, output3_real) = column_butterfly2_f64(mid2_real, mid3_real);
    let (output0_imag, output1_imag) = column_butterfly2_f64(mid0_imag, mid1_imag);
    let (output2_imag, output3_imag) = column_butterfly2_f64_negaterow1(mid2_imag, mid3_imag_neg);

    // Swap outputs 1 and 2 in the output to do a square transpose
    (output0_real, output0_imag, output2_real, output2_imag, output1_real, output1_imag, output3_real, output3_imag)
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

// Treat the input like the rows of a 4x2 array, and transpose it to a 2x4 array
#[inline(always)]
pub unsafe fn transpose_2x4_to_4x2_f64(rows0: [__m256d; 4]) -> ([__m256d; 2], [__m256d; 2]) {
    let chunk00 = [rows0[0], rows0[1]];
    let chunk01 = [rows0[2], rows0[3]];

    let output00 = transpose_2x2_f64(chunk00);
    let output10 = transpose_2x2_f64(chunk01);

    (output00, output10)
}


// Treat the input like the rows of a 2x2 array, and transpose said rows to the columns
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

// Treat the input like the rows of a 4x4 array of real numbers (not complex numbers), and transpose said rows to the columns
#[inline(always)]
pub unsafe fn transpose_4x4_real_f64(row0: __m256d, row1: __m256d, row2: __m256d, row3: __m256d) -> (__m256d, __m256d, __m256d, __m256d) {
    let unpacked0 = _mm256_unpacklo_pd(row0, row1);
    let unpacked1 = _mm256_unpackhi_pd(row0, row1);
    let unpacked2 = _mm256_unpacklo_pd(row2, row3);
    let unpacked3 = _mm256_unpackhi_pd(row2, row3);

    let col0 = _mm256_permute2f128_pd(unpacked0, unpacked2, 0x20);
    let col1 = _mm256_permute2f128_pd(unpacked1, unpacked3, 0x20);
    let col2 = _mm256_permute2f128_pd(unpacked0, unpacked2, 0x31);
    let col3 = _mm256_permute2f128_pd(unpacked1, unpacked3, 0x31);

    (col0, col1, col2, col3)
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

    // Multiply the complex numbers in `left` by the complex numbers in `right`, using FMA instructions where possible. 
    #[inline(always)]
    pub unsafe fn complex_multiply_split_f64(left_real: __m256d, left_imag: __m256d, right_real: __m256d, right_imag: __m256d) -> (__m256d, __m256d) {
        let intermediate_real = _mm256_mul_pd(left_imag, right_imag);
        let intermediate_imag = _mm256_mul_pd(left_real, right_imag);

        let output_real = _mm256_fmsub_pd(left_real, right_real, intermediate_real);
        let output_imag = _mm256_fmadd_pd(left_imag, right_real, intermediate_imag);

        (output_real, output_imag)
    }

    // Multiply the complex numbers in `left` by the complex numbers in `right`, using FMA instructions where possible
    // This variant assumes that `left` should be conjugated before multiplying (IE, the imaginary numbers in `left` should be negated)
    // Thankfully, it is straightforward to roll this into existing instructions. Namely, we can get away with replacing "fmaddsub" with "fmsubadd"
    #[allow(unused)]
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

    // Compute 2 parallel butterfly 8's using AVX and FMA instructions
    // rowN contains the nth element of each parallel FFT
    #[inline(always)]
    pub unsafe fn column_butterfly8_f64(rows: [__m256d; 8], twiddles: __m256d, twiddle_config: Rotate90Config<__m256d>) -> [__m256d; 8] {
        // Treat our butterfly-8 as a 2x4 array. first, do butterfly 4's down the columns
        let mid0     = column_butterfly4_f64([rows[0], rows[2], rows[4], rows[6]], twiddle_config);
        let mut mid1 = column_butterfly4_f64([rows[1], rows[3], rows[5], rows[7]], twiddle_config);

        // Apply twiddle factors
        // We want to negate the reals of the twiddles when multiplying mid7, but it's easier to conjugate the twiddles (Ie negate the imaginaries)
        // Negating the reals before amultiplication is equivalent to negating the imaginaries before the multiplication and then negatign the entire result
        // And we can "negate the entire result" by rollign that operation into the subsequent butterfly 2's
        mid1[1] = complex_multiply_f64(twiddles, mid1[1]);
        mid1[2] = twiddle_config.rotate90(mid1[2]);
        mid1[3] = complex_conjugated_multiply_f64(twiddles, mid1[3]);

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transposE" and thne apply butterfly 2's across the columns of our 4x2 array
        let (final0, final1) = column_butterfly2_f64(mid0[0], mid1[0]);
        let (final2, final3) = column_butterfly2_f64(mid0[1], mid1[1]);
        let (final4, final5) = column_butterfly2_f64(mid0[2], mid1[2]);
        let (final6, final7) = column_butterfly2_f64_negaterow1(mid0[3], mid1[3]); // Finish applying the negation from our twiddles by calling a different butterfly 2 function

        [final0, final2, final4, final6, final1, final3, final5, final7]
    }
}