use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, IsInverse, FFT};
use twiddles;

use algorithm::butterflies::FFTButterfly;

use std::marker::PhantomData;

/// FFT algorithm optimized for power-of-two sizes
///
/// ~~~
/// // Computes a forward FFT of size 4096
/// use rustfft::algorithm::SplitRadix;
/// use rustfft::FFT;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 4096];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 4096];
///
/// let fft = SplitRadix::new(4096, false);
/// fft.process(&mut input, &mut output);
/// ~~~

pub struct SplitRadix<T> {
    twiddles: Box<[Complex<T>]>,
    fft_half: Arc<FFT<T>>,
    fft_quarter: Arc<FFT<T>>,
    len: usize,
    inverse: bool,
}

impl<T: FFTnum> SplitRadix<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(fft_half: Arc<FFT<T>>, fft_quarter: Arc<FFT<T>>) -> Self {
        assert_eq!(
            fft_half.is_inverse(), fft_quarter.is_inverse(), 
            "fft_half and fft_quarter must both be inverse, or neither. got fft_half inverse={}, fft_quarter inverse={}",
            fft_half.is_inverse(), fft_quarter.is_inverse());

        assert_eq!(
            fft_half.len(), fft_quarter.len() * 2, 
            "fft_half must be 2x the len of fft_quarter. got fft_half len={}, fft_quarter len={}",
            fft_half.len(), fft_quarter.len())
;
        let inverse = fft_quarter.is_inverse();
        let quarter_len = fft_quarter.len();
        let len = quarter_len * 4;

        let twiddles : Vec<_> = (0..quarter_len).map(|i| twiddles::single_twiddle(i, len, inverse)).collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
            fft_half,
            fft_quarter,
            len,
            inverse,
        }
    }

    unsafe fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        let half_len = self.len / 2;
        let quarter_len = self.len / 4;

        // Split the output into the buffers we'll use for the sub-FFT input
        let (inner_half_input, inner_quarter_input) = output.split_at_mut(half_len);
        let (inner_quarter1_input, inner_quarter3_input) = inner_quarter_input.split_at_mut(quarter_len);

        *inner_half_input.get_unchecked_mut(0)     = *input.get_unchecked(0);
        *inner_quarter1_input.get_unchecked_mut(0) = *input.get_unchecked(1);
        *inner_half_input.get_unchecked_mut(1)     = *input.get_unchecked(2);
        for i in 1..quarter_len {
            *inner_quarter3_input.get_unchecked_mut(i)     = *input.get_unchecked(i * 4 - 1);
            *inner_half_input.get_unchecked_mut(i * 2)     = *input.get_unchecked(i * 4);
            *inner_quarter1_input.get_unchecked_mut(i)     = *input.get_unchecked(i * 4 + 1);
            *inner_half_input.get_unchecked_mut(i * 2 + 1) = *input.get_unchecked(i * 4 + 2);
        }
        *inner_quarter3_input.get_unchecked_mut(0) = *input.get_unchecked(input.len() - 1);

        // Split the input into the buffers we'll use for the sub-FFT output
        let (inner_half_output, inner_quarter_output) = input.split_at_mut(half_len);
        let (inner_quarter1_output, inner_quarter3_output) = inner_quarter_output.split_at_mut(quarter_len);

        // Execute the inner FFTs
        self.fft_half.process(inner_half_input, inner_half_output);
        self.fft_quarter.process(inner_quarter1_input, inner_quarter1_output);
        self.fft_quarter.process(inner_quarter3_input, inner_quarter3_output);

        // Recombine into a single buffer
        for i in 0..quarter_len {
            let twiddle = *self.twiddles.get_unchecked(i);

            let twiddled_quarter1 = twiddle * inner_quarter1_output[i];
            let twiddled_quarter3 = twiddle.conj() * inner_quarter3_output[i];
            let quarter_sum  = twiddled_quarter1 + twiddled_quarter3;
            let quarter_diff = twiddled_quarter1 - twiddled_quarter3;

            *output.get_unchecked_mut(i)            = *inner_half_output.get_unchecked(i) + quarter_sum;
            *output.get_unchecked_mut(i + half_len) = *inner_half_output.get_unchecked(i) - quarter_sum;
            if self.is_inverse() {
                *output.get_unchecked_mut(i + quarter_len)     = *inner_half_output.get_unchecked(i + quarter_len) + twiddles::rotate_90(quarter_diff, true);
                *output.get_unchecked_mut(i + quarter_len * 3) = *inner_half_output.get_unchecked(i + quarter_len) + twiddles::rotate_90(quarter_diff, false);
            } else {
                *output.get_unchecked_mut(i + quarter_len)     = *inner_half_output.get_unchecked(i + quarter_len) + twiddles::rotate_90(quarter_diff, false);
                *output.get_unchecked_mut(i + quarter_len * 3) = *inner_half_output.get_unchecked(i + quarter_len) + twiddles::rotate_90(quarter_diff, true);
            }
        }
    }
}

impl<T: FFTnum> FFT<T> for SplitRadix<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

         unsafe {self.perform_fft(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            unsafe {self.perform_fft(in_chunk, out_chunk) };
        }
    }
}
impl<T> Length for SplitRadix<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for SplitRadix<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}


pub struct SplitRadixAvx<T> {
    twiddles: Box<[Complex<T>]>,
    fft_half: Arc<FFT<T>>,
    fft_quarter: Arc<FFT<T>>,
    len: usize,
    inverse: bool,
}

impl<T: FFTnum> SplitRadixAvx<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(fft_half: Arc<FFT<T>>, fft_quarter: Arc<FFT<T>>) -> Self {
        assert_eq!(
            fft_half.is_inverse(), fft_quarter.is_inverse(), 
            "fft_half and fft_quarter must both be inverse, or neither. got fft_half inverse={}, fft_quarter inverse={}",
            fft_half.is_inverse(), fft_quarter.is_inverse());

        assert_eq!(
            fft_half.len(), fft_quarter.len() * 2, 
            "fft_half must be 2x the len of fft_quarter. got fft_half len={}, fft_quarter len={}",
            fft_half.len(), fft_quarter.len());

        let inverse = fft_quarter.is_inverse();
        let quarter_len = fft_quarter.len();
        let len = quarter_len * 4;

        let twiddles : Vec<_> = (0..quarter_len).map(|i| twiddles::single_twiddle(i, len, inverse)).collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
            fft_half,
            fft_quarter,
            len,
            inverse,
        }
    }
}


fn to_float_ptr<T: FFTnum>(item: &Complex<T>) -> *const T {
    unsafe { std::mem::transmute::<*const Complex<T>, *const T>(item as *const Complex<T>) }
}

fn to_float_mut_ptr<T: FFTnum>(item: &mut Complex<T>) -> *mut T {
    unsafe { std::mem::transmute::<*mut Complex<T>, *mut T>(item as *mut Complex<T>) }
}

use std::arch::x86_64::*;

macro_rules! load_avx_f32 {
    ($array: expr, $index: expr) => {{
        _mm256_loadu_ps(to_float_ptr($array.get_unchecked($index)))
    }};
}

macro_rules! store_avx_f32 {
    ($array: expr, $index: expr, $source: expr) => {{
        _mm256_storeu_ps(to_float_mut_ptr($array.get_unchecked_mut($index)), $source);
    }};
}

// Split the array into evens and odds. IE, do a 2xN -> Nx2 transpose
macro_rules! split_evens_f32 {
    ($row0: expr, $row1: expr) => {{
        let permuted0 = _mm256_permute2f128_ps($row0, $row1, 0x20);
        let permuted1 = _mm256_permute2f128_ps($row0, $row1, 0x31);
        let unpacked1 = _mm256_unpackhi_ps(permuted0, permuted1);
        let unpacked0 = _mm256_unpacklo_ps(permuted0, permuted1);
        let output1 = _mm256_permute_ps(unpacked1, 0xD8);
        let output0 = _mm256_permute_ps(unpacked0, 0xD8);

        (output0, output1)
    }};
}



macro_rules! complex_multiply_f32 {
    ($left: expr, $right: expr) => {{
        // Extract the real and imaginary components from $left into 2 separate registers, using duplicate instructions
        let left_real = _mm256_moveldup_ps($left);
        let left_imag = _mm256_movehdup_ps($left);

        // create a shuffled version of $right where the imaginary values are swapped with the reals
        let right_shuffled = _mm256_permute_ps($right, 0xB1);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = _mm256_mul_ps(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then  alternatingly add and subtract the left side fro mthe right
        _mm256_fmaddsub_ps(left_real, $right, output_right)
    }};
}

// This variant assumes that `left` should be conjugated before multiplying
// Thankfully, it is straightforward to roll this into existing instructions. Namely, we can get away with replacing "fmaddsub" with "fmsubadd"
macro_rules! complex_conj_multiply_f32 {
    ($left: expr, $right: expr) => {{
        // Extract the real and imaginary components from $left into 2 separate registers, using duplicate instructions
        let left_real = _mm256_moveldup_ps($left);
        let left_imag = _mm256_movehdup_ps($left);

        // create a shuffled version of $right where the imaginary values are swapped with the reals
        let right_shuffled = _mm256_permute_ps($right, 0xB1);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = _mm256_mul_ps(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then  alternatingly add and subtract the left side fro mthe right
        _mm256_fmsubadd_ps(left_real, $right, output_right)
    }};
}


// Compute 4 parallel *column* butterfly 2's using AVX instructions
macro_rules! column_butterfly2_avx_f32 {
    ($row0: expr, $row1: expr) => {{
        let output0 = _mm256_add_ps($row0, $row1);
        let output1 = _mm256_sub_ps($row0, $row1);

        (output0, output1)
    }};
}

// Compute 4 parallel *column* butterfly 2's using AVX instructions
// This variant will flip the sign of $row1 before each operation
macro_rules! column_butterfly2_negaterow1_avx_f32 {
    ($row0: expr, $row1: expr) => {{
        let output0 = _mm256_sub_ps($row0, $row1);
        let output1 = _mm256_add_ps($row0, $row1);

        (output0, output1)
    }};
}


// Apply the butterfly 4's twiddle factor (IE, rotate 90 degrees) to an entire row of elements
macro_rules! butterfly4_twiddle_avx_f32 {
    ($elements: expr, $inverse: expr) => {{
        // As usual, it's easiest to swap all the reals with imaginaries, create a negated copy, then blend in the specific values we want to negate
        let elements_swapped = _mm256_permute_ps($elements, 0xB1);
        let elements_negated = _mm256_xor_ps(elements_swapped, _mm256_set1_ps(-0.0));

        // Which negative we blend in depends on whether we're forward or inverse
        let result = if $inverse {
            _mm256_blend_ps(elements_swapped, elements_negated, 0x55)
        } else {
            _mm256_blend_ps(elements_swapped, elements_negated, 0xAA)
        };

        result
    }};
}

// Apply the butterfly 4's twiddle factor (IE, rotate 90 degrees) to the odd elements of a row
macro_rules! butterfly4_twiddle_alternating_avx_f32 {
    ($elements: expr, $inverse: expr) => {{
        // As usual, it's easiest to swap all the reals with imaginaries, create a negated copy, then blend in the specific values we want to negate
        let elements_swapped = _mm256_permute_ps($elements, 0xB4);
        let elements_negated = _mm256_xor_ps(elements_swapped, _mm256_set1_ps(-0.0));

        // Which negative we blend in depends on whether we're forward or inverse
        let result = if $inverse {
            _mm256_blend_ps(elements_swapped, elements_negated, 0x44)
        } else {
            _mm256_blend_ps(elements_swapped, elements_negated, 0x88)
        };

        result
    }};
}

// Compute 2 butterfly 4's across the pair of rows
macro_rules! butterfly4_pair_avx_f32 {
    ($row0: expr, $row1: expr, $inverse: expr) => {{
        // Rearrange the data before we do our butterfly 4s. This swaps the last 2 elements of butterfly0 with the first two elements of butterfly1
        // The result is that all elements will be in place for us to do the first butterfly-2
        let permuted0 = _mm256_permute2f128_ps($row0, $row1, 0x20);
        let permuted1 = _mm256_permute2f128_ps($row0, $row1, 0x31);

        // Do the first set of butterfly 2's
        let (postbutterfly0, postbutterfly1_pretwiddle) = column_butterfly2_avx_f32!(permuted0, permuted1);

        // Which negative we blend in depends on whether we're forward or inverse
        let postbutterfly1 = butterfly4_twiddle_alternating_avx_f32!(postbutterfly1_pretwiddle, $inverse);

        // use unpack instructions. conveniently, this will put everything in place for our second butterfly 2
        let unpacked0 = _mm256_unpacklo_ps(postbutterfly0, postbutterfly1);
        let unpacked1 = _mm256_unpackhi_ps(postbutterfly0, postbutterfly1);
        let swapped0 = _mm256_permute_ps(unpacked0, 0xD8);
        let swapped1 = _mm256_permute_ps(unpacked1, 0xD8);

        // last set of butterfly 2's
        let (final0, final1) = column_butterfly2_avx_f32!(swapped0, swapped1);

        // Undo the permute that we did initially. The previous permute, followed by the unpack, will end up putting everything in its correct output location.
        let unpermuted0 = _mm256_permute2f128_ps(final0, final1, 0x20);
        let unpermuted1 = _mm256_permute2f128_ps(final0, final1, 0x31);

        (unpermuted0, unpermuted1)
    }};
}

// Compute a single butterfly8 using AVX instructions, where $row0 is an AVX register containing the first 4 complex elements,
// $row1 is an AVX register containing the last 4 elements, and $twiddles is an AVX register containing the twiddle factors for a 2x4 cooley tukey step
// TODO: try a split radix version of this? Seems like it would be hard, because there would only be 2 twiddle factors.
macro_rules! butterfly8_avx_f32 {
    ($row0: expr, $row1: expr, $twiddles: expr, $inverse: expr) => {{

        // Do our butterfly 2's down the columns
        let (intermediate0, intermediate1_pretwiddle) = column_butterfly2_avx_f32!($row0, $row1);

        // Apply the size-8 twiddle factors
        let intermediate1 = complex_multiply_f32!(intermediate1_pretwiddle, $twiddles);

        // Rearrange the data before we do our butterfly 4s. This swaps the last 2 elements of butterfly0 with the first two elements of butterfly1
        // The result is that we can then do a 4x butterfly 2, apply twiddles, use unpack instructions to transpose to the final output, then do another 4x butterfly 2
        let permuted0 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x20);
        let permuted1 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x31);

        // Do the first set of butterfly 2's
        let (postbutterfly0, postbutterfly1_pretwiddle) = column_butterfly2_avx_f32!(permuted0, permuted1);

        // Which negative we blend in depends on whether we're forward or inverse
        let postbutterfly1 = butterfly4_twiddle_alternating_avx_f32!(postbutterfly1_pretwiddle, $inverse);

        // use unpack instructions to transpose, and to prepare for the final butterfly 2's
        let unpermuted0 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x20);
        let unpermuted1 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x31);
        let unpacked0 = _mm256_unpacklo_ps(unpermuted0, unpermuted1);
        let unpacked1 = _mm256_unpackhi_ps(unpermuted0, unpermuted1);
        let swapped0 = _mm256_permute_ps(unpacked0, 0xD8);
        let swapped1 = _mm256_permute_ps(unpacked1, 0xD8);

        let (output0, output1) = column_butterfly2_avx_f32!(swapped0, swapped1);

        (output0, output1)
    }};
}

// Compute 4 parallel butterfly 4's , with $row containing the first element of each butterfly 4, $row1 containignthe second element of each, etc
macro_rules! column_butterfly4_avx_f32 {
    ($row0: expr, $row1: expr, $row2: expr, $row3: expr, $inverse: expr) => {{
        // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
        let (mid0, mid2) = column_butterfly2_avx_f32!($row0, $row2);
        let (mid1, mid3_pretwiddle) = column_butterfly2_avx_f32!($row1, $row3);

        // Apply element 3 inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
        let mid3 = butterfly4_twiddle_avx_f32!(mid3_pretwiddle, $inverse);

        // Perform the second set of size-2 FFTs
        let (output0, output1) = column_butterfly2_avx_f32!(mid0, mid1);
        let (output2, output3) = column_butterfly2_avx_f32!(mid2, mid3);

        // Swap outputs 1 and 2 in the output to do a square transpose
        (output0, output2, output1, output3)
    }};
}

// Compute 4 parallel butterfly 8's , with $row containing the first element of each butterfly 4, $row1 containignthe second element of each, etc
macro_rules! column_butterfly8_avx_f32 {
    ($row0: expr, $row1: expr, $row2: expr, $row3: expr, $row4: expr, $row5: expr, $row6: expr, $row7: expr, $twiddles: expr, $inverse: expr) => {{

        // Treat our butterfly-8 as a 2x4 array. first, do butterfly 4's down the columns
        let (mid0, mid2, mid4, mid6) = column_butterfly4_avx_f32!($row0, $row2, $row4, $row6, $inverse);
        let (mid1, mid3, mid5, mid7) = column_butterfly4_avx_f32!($row1, $row3, $row5, $row7, $inverse);

        // Apply twiddle factors
        // We want to negate the reals of the twiddles when multiplying mid7, but it's easier to conjugate the twiddles (Ie negate the imaginaries)
        // Negating the reals before amultiplication is equivalent to negating the imaginaries before the multiplication and then negatign the entire result
        // And we can "negate the entire result" by rollign that operation into the subsequent butterfly 2's
        let mid3_twiddled = complex_multiply_f32!($twiddles, mid3);
        let mid5_twiddled = butterfly4_twiddle_avx_f32!(mid5, $inverse);
        let mid7_twiddled_neg = complex_conj_multiply_f32!($twiddles, mid7);

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transposE" and thne apply butterfly 2's across the columns of our 4x2 array
        let (final0, final1) = column_butterfly2_avx_f32!(mid0, mid1);
        let (final2, final3) = column_butterfly2_avx_f32!(mid2, mid3_twiddled);
        let (final4, final5) = column_butterfly2_avx_f32!(mid4, mid5_twiddled);
        let (final6, final7) = column_butterfly2_negaterow1_avx_f32!(mid6, mid7_twiddled_neg); // Finish applying the negation from our twiddles by calling a different butterfly 2 function

        (final0, final2, final4, final6, final1, final3, final5, final7)
    }};
}


macro_rules! transpose_4x4_f32 {
    ($col0: expr, $col1: expr, $col2: expr, $col3: expr) => {{
        let unpacked_0 = _mm256_unpacklo_ps($col0, $col1);
        let unpacked_1 = _mm256_unpackhi_ps($col0, $col1);
        let unpacked_2 = _mm256_unpacklo_ps($col2, $col3);
        let unpacked_3 = _mm256_unpackhi_ps($col2, $col3);

        let swapped_0 = _mm256_permute_ps(unpacked_0, 0xD8);
        let swapped_1 = _mm256_permute_ps(unpacked_1, 0xD8);
        let swapped_2 = _mm256_permute_ps(unpacked_2, 0xD8);
        let swapped_3 = _mm256_permute_ps(unpacked_3, 0xD8);

        let output_0 = _mm256_permute2f128_ps(swapped_0, swapped_2, 0x20);
        let output_1 = _mm256_permute2f128_ps(swapped_1, swapped_3, 0x20);
        let output_2 = _mm256_permute2f128_ps(swapped_0, swapped_2, 0x31);
        let output_3 = _mm256_permute2f128_ps(swapped_1, swapped_3, 0x31);

        (output_0, output_1, output_2, output_3)
    }};
}


// Compute a single butterfly16 using AVX instructions, where $row0 is an AVX register containing the first 4 complex elements,
// $row1 is an AVX register containing the last 4 elements, etc and $twiddles1 through twiddles3 contain the twiddle factors for a 4x4 mixed radix step
// TODO: try a split radix version of this? Would result in much fewer twiddle factors
macro_rules! butterfly16_avx_f32 {
    ($row0: expr, $row1: expr, $row2: expr, $row3: expr, $twiddle1: expr, $twiddle2: expr, $twiddle3: expr, $inverse: expr) => {{

        // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
        let (element_0_middle, element_2_middle) = column_butterfly2_avx_f32!($row0, $row2);
        let (element_1_middle, element_3_middle_pretwiddle) = column_butterfly2_avx_f32!($row1, $row3);

        // Apply element 3 inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
        let element_3_middle = butterfly4_twiddle_avx_f32!(element_3_middle_pretwiddle, $inverse);

        // Perform the second set of size-2 FFTs
        let (element_0_final, element_1_final_pretwiddle) = column_butterfly2_avx_f32!(element_0_middle, element_1_middle);
        let (element_2_final_pretwiddle, element_3_final_pretwiddle) = column_butterfly2_avx_f32!(element_2_middle, element_3_middle);

        let element_1_final = complex_multiply_f32!(element_2_final_pretwiddle, $twiddle1);
        let element_2_final = complex_multiply_f32!(element_1_final_pretwiddle, $twiddle2);
        let element_3_final = complex_multiply_f32!(element_3_final_pretwiddle, $twiddle3);

        // The first set of 4 butterlies is done. Now, transpose the columns to the rows.
        let (row_0, row_1, row_2, row_3) = transpose_4x4_f32!(element_0_final, element_1_final, element_2_final, element_3_final);

        // We've transposed. Now, process size 4 butterflies again.
        // Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
        let (element_0_middle, element_2_middle) = column_butterfly2_avx_f32!(row_0, row_2);
        let (element_1_middle, element_3_middle_pretwiddle) = column_butterfly2_avx_f32!(row_1, row_3);

        // Apply element 3 inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
        let element_3_middle = butterfly4_twiddle_avx_f32!(element_3_middle_pretwiddle, $inverse);

        // Perform the second set of size-2 FFTs
        let (element_0_final, element_1_final) = column_butterfly2_avx_f32!(element_0_middle, element_1_middle);
        let (element_2_final, element_3_final) = column_butterfly2_avx_f32!(element_2_middle, element_3_middle);

        (element_0_final, element_1_final, element_2_final, element_3_final)
    }};
}


impl SplitRadixAvx<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        let half_len = self.len / 2;
        let quarter_len = self.len / 4;
        let three_quarter_len = half_len + quarter_len;
        let sixteenth_len = self.len / 16;

        let scratch = &mut output[0..half_len];
        let (scratch_quarter1, scratch_quarter3) = scratch.split_at_mut(quarter_len);

        for i in 0..sixteenth_len {
            let chunk0 = load_avx_f32!(input, i*16);
            let chunk1 = load_avx_f32!(input, i*16 + 4);
            let (even0, odd0) = split_evens_f32!(chunk0, chunk1);

            let chunk2 = load_avx_f32!(input, i*16 + 8);
            let chunk3 = load_avx_f32!(input, i*16 + 12);
            let (even1, odd1) = split_evens_f32!(chunk2, chunk3);

            let (quarter1, quarter3) = split_evens_f32!(odd0, odd1);

            store_avx_f32!(input, i*8, even0);
            store_avx_f32!(input, i*8 + 4, even1);
            store_avx_f32!(scratch_quarter1, i*4, quarter1);
            store_avx_f32!(scratch_quarter3, i*4, quarter3);
        }

        // Split the input into the buffers we'll use for the sub-FFT input
        let (inner_half_input, inner_quarter_input) = input.split_at_mut(half_len);
        let (inner_quarter1_input, inner_quarter3_input) = inner_quarter_input.split_at_mut(quarter_len);

        // Copy the quarter data from the scratch back to the split input
        inner_quarter1_input.copy_from_slice(scratch_quarter1);
        let (scratch_quarter3_last, scratch_quarter3) = scratch_quarter3.split_last_mut().unwrap();
        let (inner_quarter3_first, inner_quarter3_remaining) = inner_quarter3_input.split_first_mut().unwrap();

        // When copying the quarter3 slice, we actually want to rotate the last element of the scratch to the first element of the input
        inner_quarter3_remaining.copy_from_slice(scratch_quarter3);
        *inner_quarter3_first = *scratch_quarter3_last;

        // Split the output into the buffers we'll use for the sub-FFT output
        let (inner_half_output, inner_quarter_output) = output.split_at_mut(half_len);
        let (inner_quarter1_output, inner_quarter3_output) = inner_quarter_output.split_at_mut(quarter_len);

        // Execute the inner FFTs
        self.fft_half.process(inner_half_input, inner_half_output);
        self.fft_quarter.process(inner_quarter1_input, inner_quarter1_output);
        self.fft_quarter.process(inner_quarter3_input, inner_quarter3_output);

        // Recombine into a single buffer
        for i in 0..sixteenth_len {
            let inner_even0_entry = load_avx_f32!(output, i * 4);
            let inner_even1_entry = load_avx_f32!(output, quarter_len + i * 4);
            let inner_quarter1_entry = load_avx_f32!(output, half_len + i * 4);
            let inner_quarter3_entry = load_avx_f32!(output, three_quarter_len + i * 4);

            let twiddle = _mm256_loadu_ps(to_float_ptr(self.twiddles.get_unchecked(i * 4)));

            let twiddled_quarter1 = complex_multiply_f32!(twiddle, inner_quarter1_entry);
            let twiddled_quarter3 = complex_conj_multiply_f32!(twiddle, inner_quarter3_entry);
            let (quarter_sum, quarter_diff) = column_butterfly2_avx_f32!(twiddled_quarter1, twiddled_quarter3);

            let (output_i, output_i_half) = column_butterfly2_avx_f32!(inner_even0_entry, quarter_sum);
            store_avx_f32!(output, i * 4, output_i);
            store_avx_f32!(output, i * 4 + half_len, output_i_half);
            
            // compute the twiddle for quarter diff by rotating it
            let quarter_diff_rotated = butterfly4_twiddle_avx_f32!(quarter_diff, false);

            let (output_quarter1_preswap, output_quarter3_preswap) = column_butterfly2_avx_f32!(inner_even1_entry, quarter_diff_rotated);
            let (output_quarter1, output_quarter3) = if self.is_inverse() {
                (output_quarter3_preswap, output_quarter1_preswap)
            } else {
                (output_quarter1_preswap, output_quarter3_preswap)
            };
            store_avx_f32!(output, quarter_len + i * 4, output_quarter1);
            store_avx_f32!(output, three_quarter_len + i * 4, output_quarter3);
        }
    }
}
default impl<T: FFTnum> FFT<T> for SplitRadixAvx<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for SplitRadixAvx<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());

        unsafe { self.perform_fft_f32(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            unsafe { self.perform_fft_f32(in_chunk, out_chunk) };
        }
    }
}

impl<T> Length for SplitRadixAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for SplitRadixAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}


pub struct MixedRadixAvx4x2<T> {
    twiddles: [Complex<T>; 4],
    inverse: bool,
}
impl<T: FFTnum> MixedRadixAvx4x2<T>
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        assert!(has_avx && has_fma, "The MixedRadixAvx4x2 algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

        Self {
            twiddles: [
                Complex{ re: T::one(), im: T::zero() },
                twiddles::single_twiddle(1, 8, inverse),
                twiddles::single_twiddle(2, 8, inverse),
                twiddles::single_twiddle(3, 8, inverse),
            ],
            inverse: inverse,
        }
    }
}

impl MixedRadixAvx4x2<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn process_butterfly8_f32(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {

        let twiddle = _mm256_loadu_ps(to_float_ptr(self.twiddles.get_unchecked(0)));
        let row0 = _mm256_loadu_ps(to_float_ptr(input.get_unchecked(0)));
        let row1 = _mm256_loadu_ps(to_float_ptr(input.get_unchecked(4)));

        let (output0, output1) = butterfly8_avx_f32!(row0, row1, twiddle, self.inverse);

        _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(0)), output0);
        _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(4)), output1);
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn process_butterfly8_f32_inplace(&self, buffer: &mut [Complex<f32>]) {

        let twiddle = _mm256_loadu_ps(to_float_ptr(self.twiddles.get_unchecked(0)));
        let row0 = _mm256_loadu_ps(to_float_ptr(buffer.get_unchecked(0)));
        let row1 = _mm256_loadu_ps(to_float_ptr(buffer.get_unchecked(4)));

        let (output0, output1) = butterfly8_avx_f32!(row0, row1, twiddle, self.inverse);

        _mm256_storeu_ps(to_float_mut_ptr(buffer.get_unchecked_mut(0)), output0);
        _mm256_storeu_ps(to_float_mut_ptr(buffer.get_unchecked_mut(4)), output1);
    }
}
default impl<T: FFTnum> FFT<T> for MixedRadixAvx4x2<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for MixedRadixAvx4x2<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());

        unsafe { self.process_butterfly8_f32(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            unsafe { self.process_butterfly8_f32(in_chunk, out_chunk) };
        }
    }
}
default impl<T: FFTnum> FFTButterfly<T> for MixedRadixAvx4x2<T> {
    #[inline(always)]
    unsafe fn process_inplace(&self, _buffer: &mut [Complex<T>]) {
        unimplemented!();
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, _buffer: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFTButterfly<f32> for MixedRadixAvx4x2<f32> {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<f32>]) {
        self.process_butterfly8_f32_inplace(buffer);
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<f32>]) {
        for chunk in buffer.chunks_exact_mut(self.len()) {
            self.process_butterfly8_f32_inplace(chunk);
        }
    }
}
impl<T> Length for MixedRadixAvx4x2<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        8
    }
}
impl<T> IsInverse for MixedRadixAvx4x2<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct SplitRadixAvxButterfly16<T> {
    twiddles: __m256,
    twiddles_butterfly8: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl SplitRadixAvxButterfly16<f32>
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        assert!(has_avx && has_fma, "The SplitRadixAvxButterfly16 algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(1, 16, inverse),
            twiddles::single_twiddle(2, 16, inverse),
            twiddles::single_twiddle(3, 16, inverse),
        ];
        let twiddles_butterfly8 = [
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(1, 8, inverse),
            twiddles::single_twiddle(2, 8, inverse),
            twiddles::single_twiddle(3, 8, inverse),
        ];

        Self {
            twiddles: unsafe { load_avx_f32!(twiddles, 0) },
            twiddles_butterfly8: unsafe {load_avx_f32!(twiddles_butterfly8, 0) }, 
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
}

impl SplitRadixAvxButterfly16<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn process_butterfly16_f32(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {

        // Load the input and transpose it into evens/odds
        let input0 = load_avx_f32!(input, 0);
        let input1 = load_avx_f32!(input, 1 * 4);
        let (even0, odd0) = split_evens_f32!(input0, input1);

        let input2 = load_avx_f32!(input, 2 * 4);
        let input3 = load_avx_f32!(input, 3 * 4);
        let (even1, odd1) = split_evens_f32!(input2, input3);

        let (quarter1_input, quarter3_input) = split_evens_f32!(odd0, odd1);

        // In our quarter3 array, we need to rotate the last element around to the first. The easiest way to do that without AVX2 instructions is to spill them out to memory and reload them
        // TODO: measure using AVX2 permute8xf32 instead
        let mut spilled_quarter_3 = [Complex::zero();5];
        store_avx_f32!(spilled_quarter_3, 1, quarter3_input);
        spilled_quarter_3[0] = spilled_quarter_3[4]; // do the rotation here. We wpilled to an index of 1, but next we're going to read from an index of 0
        let quarter3_input = load_avx_f32!(spilled_quarter_3, 0);

        // Compute our butterfly 4's
        let (quarter1_output, quarter3_output) = butterfly4_pair_avx_f32!(quarter1_input, quarter3_input, self.inverse);
        // compute our butterfly 8
        let (even_output0, even_output1) = butterfly8_avx_f32!(even0, even1, self.twiddles_butterfly8, self.inverse);

        // Recombine into a single buffer
        {
            let twiddled_quarter1 = complex_multiply_f32!(self.twiddles, quarter1_output);
            let twiddled_quarter3 = complex_conj_multiply_f32!(self.twiddles, quarter3_output);

            let (quarter_sum, quarter_diff) = column_butterfly2_avx_f32!(twiddled_quarter1, twiddled_quarter3);

            let (output_i, output_i_half) = column_butterfly2_avx_f32!(even_output0, quarter_sum);
            store_avx_f32!(output, 0, output_i);
            store_avx_f32!(output, 8, output_i_half);
            
            // compute the twiddle for quarter diff by rotating it
            let quarter_diff_rotated = butterfly4_twiddle_avx_f32!(quarter_diff, false);

            let (output_quarter1, output_quarter3) = column_butterfly2_avx_f32!(even_output1, quarter_diff_rotated);
            let (output_quarter1, output_quarter3) = if self.is_inverse() {
                (output_quarter3, output_quarter1)
            } else {
                (output_quarter1, output_quarter3)
            };
            store_avx_f32!(output, 4, output_quarter1);
            store_avx_f32!(output, 12, output_quarter3);
        }
    }
}
default impl<T: FFTnum> FFT<T> for SplitRadixAvxButterfly16<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for SplitRadixAvxButterfly16<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());

        unsafe { self.process_butterfly16_f32(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            unsafe { self.process_butterfly16_f32(in_chunk, out_chunk) };
        }
    }
}
impl<T> Length for SplitRadixAvxButterfly16<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        16
    }
}
impl<T> IsInverse for SplitRadixAvxButterfly16<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}




pub struct SplitRadixAvxButterfly32<T> {
    twiddles: [__m256; 2],
    twiddles_butterfly8: __m256,
    twiddles_butterfly16: [__m256; 3],
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FFTnum> SplitRadixAvxButterfly32<T>
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        assert!(has_avx && has_fma, "The SplitRadixAvxButterfly32 algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(1, 32, inverse),
            twiddles::single_twiddle(2, 32, inverse),
            twiddles::single_twiddle(3, 32, inverse),
            twiddles::single_twiddle(4, 32, inverse),
            twiddles::single_twiddle(5, 32, inverse),
            twiddles::single_twiddle(6, 32, inverse),
            twiddles::single_twiddle(7, 32, inverse),
        ];
        let twiddles_butterfly8 = [
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(1, 8, inverse),
            twiddles::single_twiddle(2, 8, inverse),
            twiddles::single_twiddle(3, 8, inverse),
        ];
        let twiddles_butterfly16 = [
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(1, 16, inverse),
            twiddles::single_twiddle(2, 16, inverse),
            twiddles::single_twiddle(3, 16, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(2, 16, inverse),
            twiddles::single_twiddle(4, 16, inverse),
            twiddles::single_twiddle(6, 16, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(3, 16, inverse),
            twiddles::single_twiddle(6, 16, inverse),
            twiddles::single_twiddle(9, 16, inverse),
        ];

        Self {
            twiddles: unsafe { [load_avx_f32!(twiddles, 0), load_avx_f32!(twiddles, 4)] },
            twiddles_butterfly8: unsafe { load_avx_f32!(twiddles_butterfly8, 0) }, 
            twiddles_butterfly16: unsafe { [load_avx_f32!(twiddles_butterfly16, 0), load_avx_f32!(twiddles_butterfly16, 4), load_avx_f32!(twiddles_butterfly16, 8)] },
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
}

impl SplitRadixAvxButterfly32<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn process_butterfly32_f32(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {

        // Load the input and transpose it into evens/odds
        let input0 = load_avx_f32!(input, 0);
        let input1 = load_avx_f32!(input, 1 * 4);
        let (even0, odd0) = split_evens_f32!(input0, input1);

        let input2 = load_avx_f32!(input, 2 * 4);
        let input3 = load_avx_f32!(input, 3 * 4);
        let (even1, odd1) = split_evens_f32!(input2, input3);

        let (quarter1_input0, quarter3_input0) = split_evens_f32!(odd0, odd1);

        let input4 = load_avx_f32!(input, 4 * 4);
        let input5 = load_avx_f32!(input, 5 * 4);
        let (even2, odd2) = split_evens_f32!(input4, input5);

        let input6 = load_avx_f32!(input, 6 * 4);
        let input7 = load_avx_f32!(input, 7 * 4);
        let (even3, odd3) = split_evens_f32!(input6, input7);

        let (quarter1_input1, quarter3_input1) = split_evens_f32!(odd2, odd3);

        // In our quarter3 array, we need to rotate the last element around to the first. The easiest way to do that without AVX2 instructions is to spill them out to memory and reload them
        // TODO: measure using AVX2 permute8xf32 instead
        let mut spilled_quarter_3 = [Complex::zero();9];
        store_avx_f32!(spilled_quarter_3, 1, quarter3_input0);
        store_avx_f32!(spilled_quarter_3, 5, quarter3_input1);
        spilled_quarter_3[0] = spilled_quarter_3[8]; // do the rotation here. We wpilled to an index of 1, but next we're going to read from an index of 0
        let quarter3_input0 = load_avx_f32!(spilled_quarter_3, 0);
        let quarter3_input1 = load_avx_f32!(spilled_quarter_3, 4);

        // Compute our butterfly 8's
        let (quarter1_output0, quarter1_output1) = butterfly8_avx_f32!(quarter1_input0, quarter1_input1, self.twiddles_butterfly8, self.inverse);
        let (quarter3_output0, quarter3_output1) = butterfly8_avx_f32!(quarter3_input0, quarter3_input1, self.twiddles_butterfly8, self.inverse);

        // compute our butterfly 16
        let (even_output0, even_output1, even_output2, even_output3) = butterfly16_avx_f32!(even0, even1, even2, even3, self.twiddles_butterfly16[0], self.twiddles_butterfly16[1], self.twiddles_butterfly16[2], self.inverse);

        // Recombine into a single buffer
        {
            let i = 0;
            let twiddled_quarter1 = complex_multiply_f32!(self.twiddles[0], quarter1_output0);
            let twiddled_quarter3 = complex_conj_multiply_f32!(self.twiddles[0], quarter3_output0);

            let (quarter_sum, quarter_diff) = column_butterfly2_avx_f32!(twiddled_quarter1, twiddled_quarter3);

            let (output_i, output_i_half) = column_butterfly2_avx_f32!(even_output0, quarter_sum);
            store_avx_f32!(output, i * 4, output_i);
            store_avx_f32!(output, i * 4 + 16, output_i_half);
            
            // compute the twiddle for quarter diff by rotating it
            let quarter_diff_rotated = butterfly4_twiddle_avx_f32!(quarter_diff, false);

            let (output_quarter1, output_quarter3) = column_butterfly2_avx_f32!(even_output1, quarter_diff_rotated);
            let (output_quarter1, output_quarter3) = if self.is_inverse() {
                (output_quarter3, output_quarter1)
            } else {
                (output_quarter1, output_quarter3)
            };
            store_avx_f32!(output, i * 4 + 8, output_quarter1);
            store_avx_f32!(output, i * 4 + 24, output_quarter3);
        }
        // Recombine into a single buffer
        {
            let i = 1;
            let twiddled_quarter1 = complex_multiply_f32!(self.twiddles[1], quarter1_output1);
            let twiddled_quarter3 = complex_conj_multiply_f32!(self.twiddles[1], quarter3_output1);

            let (quarter_sum, quarter_diff) = column_butterfly2_avx_f32!(twiddled_quarter1, twiddled_quarter3);

            let (output_i, output_i_half) = column_butterfly2_avx_f32!(even_output2, quarter_sum);
            store_avx_f32!(output, i * 4, output_i);
            store_avx_f32!(output, i * 4 + 16, output_i_half);
            
            // compute the twiddle for quarter diff by rotating it
            let quarter_diff_rotated = butterfly4_twiddle_avx_f32!(quarter_diff, false);

            let (output_quarter1, output_quarter3) = column_butterfly2_avx_f32!(even_output3, quarter_diff_rotated);
            let (output_quarter1, output_quarter3) = if self.is_inverse() {
                (output_quarter3, output_quarter1)
            } else {
                (output_quarter1, output_quarter3)
            };
            store_avx_f32!(output, i * 4 + 8, output_quarter1);
            store_avx_f32!(output, i * 4 + 24, output_quarter3);
        }
    }
}
default impl<T: FFTnum> FFT<T> for SplitRadixAvxButterfly32<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for SplitRadixAvxButterfly32<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());

        unsafe { self.process_butterfly32_f32(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            unsafe { self.process_butterfly32_f32(in_chunk, out_chunk) };
        }
    }
}
impl<T> Length for SplitRadixAvxButterfly32<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        32
    }
}
impl<T> IsInverse for SplitRadixAvxButterfly32<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}


pub struct MixedRadixAvx4x4<T> {
    twiddles: [__m256; 3],
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FFTnum> MixedRadixAvx4x4<T>
{
    #[inline(never)]
    pub fn new(inverse: bool) -> Self {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        assert!(has_avx && has_fma, "The MixedRadixAvx4x4 algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(1, 16, inverse),
            twiddles::single_twiddle(2, 16, inverse),
            twiddles::single_twiddle(3, 16, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(2, 16, inverse),
            twiddles::single_twiddle(4, 16, inverse),
            twiddles::single_twiddle(6, 16, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(3, 16, inverse),
            twiddles::single_twiddle(6, 16, inverse),
            twiddles::single_twiddle(9, 16, inverse),
        ];

        Self {
            twiddles: unsafe { [
                load_avx_f32!(twiddles, 0),
                load_avx_f32!(twiddles, 4),
                load_avx_f32!(twiddles, 8),
                ]
            },
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
}

impl MixedRadixAvx4x4<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn process_butterfly16_f32(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let input0 = load_avx_f32!(input, 0);
        let input1 = load_avx_f32!(input, 1 * 4);
        let input2 = load_avx_f32!(input, 2 * 4);
        let input3 = load_avx_f32!(input, 3 * 4);

        // We're going to treat our input as a 3x4 2d array. First, do 3 butterfly 4's down the columns of that array.
        let (mid0, mid1, mid2, mid3) = column_butterfly4_avx_f32!(input0, input1, input2, input3, self.inverse);

        // Multiply in our twiddle factors
        let mid1_twiddled = complex_multiply_f32!(mid1, self.twiddles[0]);
        let mid2_twiddled = complex_multiply_f32!(mid2, self.twiddles[1]);
        let mid3_twiddled = complex_multiply_f32!(mid3, self.twiddles[2]);

        // Transpose out 4x4 array
        let (transposed0, transposed1, transposed2, transposed3) = transpose_4x4_f32!(mid0, mid1_twiddled, mid2_twiddled, mid3_twiddled);

        // Do 4 butterfly 8's down the columns of our transpsed array
        let (output0, output1, output2, output3) = column_butterfly4_avx_f32!(transposed0, transposed1, transposed2, transposed3, self.inverse);

        store_avx_f32!(output, 0, output0);
        store_avx_f32!(output, 1 * 4, output1);
        store_avx_f32!(output, 2 * 4, output2);
        store_avx_f32!(output, 3 * 4, output3);
    }
}
default impl<T: FFTnum> FFT<T> for MixedRadixAvx4x4<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for MixedRadixAvx4x4<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());

        unsafe { self.process_butterfly16_f32(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            unsafe { self.process_butterfly16_f32(in_chunk, out_chunk) };
        }
    }
}
impl<T> Length for MixedRadixAvx4x4<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        16
    }
}
impl<T> IsInverse for MixedRadixAvx4x4<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct MixedRadixAvx4x8<T> {
    twiddles: [__m256; 6],
    twiddles_butterfly8: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FFTnum> MixedRadixAvx4x8<T>
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        assert!(has_avx && has_fma, "The MixedRadixAvx4x8 algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(1, 32, inverse),
            twiddles::single_twiddle(2, 32, inverse),
            twiddles::single_twiddle(3, 32, inverse),
            twiddles::single_twiddle(4, 32, inverse),
            twiddles::single_twiddle(5, 32, inverse),
            twiddles::single_twiddle(6, 32, inverse),
            twiddles::single_twiddle(7, 32, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(2, 32, inverse),
            twiddles::single_twiddle(4, 32, inverse),
            twiddles::single_twiddle(6, 32, inverse),
            twiddles::single_twiddle(8, 32, inverse),
            twiddles::single_twiddle(10, 32, inverse),
            twiddles::single_twiddle(12, 32, inverse),
            twiddles::single_twiddle(14, 32, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(3, 32, inverse),
            twiddles::single_twiddle(6, 32, inverse),
            twiddles::single_twiddle(9, 32, inverse),
            twiddles::single_twiddle(12, 32, inverse),
            twiddles::single_twiddle(15, 32, inverse),
            twiddles::single_twiddle(18, 32, inverse),
            twiddles::single_twiddle(21, 32, inverse),
        ];

        let butterfly8_twiddle5 = twiddles::single_twiddle(1, 8, inverse);

        Self {
            twiddles: unsafe { [
                load_avx_f32!(twiddles, 0),
                load_avx_f32!(twiddles, 4),
                load_avx_f32!(twiddles, 8),
                load_avx_f32!(twiddles, 12),
                load_avx_f32!(twiddles, 16),
                load_avx_f32!(twiddles, 20),
                ]
            },
            twiddles_butterfly8: unsafe { 
                _mm256_set_ps(butterfly8_twiddle5.im, butterfly8_twiddle5.re, butterfly8_twiddle5.im, butterfly8_twiddle5.re, butterfly8_twiddle5.im, butterfly8_twiddle5.re, butterfly8_twiddle5.im, butterfly8_twiddle5.re)
            },
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
}

impl MixedRadixAvx4x8<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn process_butterfly32_f32(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let input0 = load_avx_f32!(input, 0);
        let input1 = load_avx_f32!(input, 1 * 4);
        let input2 = load_avx_f32!(input, 2 * 4);
        let input3 = load_avx_f32!(input, 3 * 4);
        let input4 = load_avx_f32!(input, 4 * 4);
        let input5 = load_avx_f32!(input, 5 * 4);
        let input6 = load_avx_f32!(input, 6 * 4);
        let input7 = load_avx_f32!(input, 7 * 4);

        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        let (mid0, mid2, mid4, mid6) = column_butterfly4_avx_f32!(input0, input2, input4, input6, self.inverse);
        let (mid1, mid3, mid5, mid7) = column_butterfly4_avx_f32!(input1, input3, input5, input7, self.inverse);


        // Multiply in our twiddle factors
        let mid2_twiddled = complex_multiply_f32!(mid2, self.twiddles[0]);
        let mid3_twiddled = complex_multiply_f32!(mid3, self.twiddles[1]);
        let mid4_twiddled = complex_multiply_f32!(mid4, self.twiddles[2]);
        let mid5_twiddled = complex_multiply_f32!(mid5, self.twiddles[3]);
        let mid6_twiddled = complex_multiply_f32!(mid6, self.twiddles[4]);
        let mid7_twiddled = complex_multiply_f32!(mid7, self.twiddles[5]);

        // Transpose our 8x4 array to an 8x4 array. Thankfully we can just do 2 4x4 transposes, which are only 4 instructions each!
        let (transposed0, transposed1, transposed2, transposed3) = transpose_4x4_f32!(mid0, mid2_twiddled, mid4_twiddled, mid6_twiddled);
        let (transposed4, transposed5, transposed6, transposed7) = transpose_4x4_f32!(mid1, mid3_twiddled, mid5_twiddled, mid7_twiddled);

        // Do 4 butterfly 8's down the columns of our transpsed array
        let (output0, output1, output2, output3, output4, output5, output6, output7) = column_butterfly8_avx_f32!(transposed0, transposed1, transposed2, transposed3, transposed4, transposed5, transposed6, transposed7, self.twiddles_butterfly8, self.inverse);

        store_avx_f32!(output, 0, output0);
        store_avx_f32!(output, 1 * 4, output1);
        store_avx_f32!(output, 2 * 4, output2);
        store_avx_f32!(output, 3 * 4, output3);
        store_avx_f32!(output, 4 * 4, output4);
        store_avx_f32!(output, 5 * 4, output5);
        store_avx_f32!(output, 6 * 4, output6);
        store_avx_f32!(output, 7 * 4, output7);
    }
}
default impl<T: FFTnum> FFT<T> for MixedRadixAvx4x8<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for MixedRadixAvx4x8<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());

        unsafe { self.process_butterfly32_f32(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            unsafe { self.process_butterfly32_f32(in_chunk, out_chunk) };
        }
    }
}
impl<T> Length for MixedRadixAvx4x8<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        32
    }
}
impl<T> IsInverse for MixedRadixAvx4x8<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct MixedRadixAvx8x8<T> {
    twiddles: [__m256; 14],
    twiddles_butterfly8: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FFTnum> MixedRadixAvx8x8<T>
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        assert!(has_avx && has_fma, "The MixedRadixAvx8x8 algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(1, 64, inverse),
            twiddles::single_twiddle(2, 64, inverse),
            twiddles::single_twiddle(3, 64, inverse),
            twiddles::single_twiddle(4, 64, inverse),
            twiddles::single_twiddle(5, 64, inverse),
            twiddles::single_twiddle(6, 64, inverse),
            twiddles::single_twiddle(7, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(2, 64, inverse),
            twiddles::single_twiddle(4, 64, inverse),
            twiddles::single_twiddle(6, 64, inverse),
            twiddles::single_twiddle(8, 64, inverse),
            twiddles::single_twiddle(10, 64, inverse),
            twiddles::single_twiddle(12, 64, inverse),
            twiddles::single_twiddle(14, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(3, 64, inverse),
            twiddles::single_twiddle(6, 64, inverse),
            twiddles::single_twiddle(9, 64, inverse),
            twiddles::single_twiddle(12, 64, inverse),
            twiddles::single_twiddle(15, 64, inverse),
            twiddles::single_twiddle(18, 64, inverse),
            twiddles::single_twiddle(21, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(4, 64, inverse),
            twiddles::single_twiddle(8, 64, inverse),
            twiddles::single_twiddle(12, 64, inverse),
            twiddles::single_twiddle(16, 64, inverse),
            twiddles::single_twiddle(20, 64, inverse),
            twiddles::single_twiddle(24, 64, inverse),
            twiddles::single_twiddle(28, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(5, 64, inverse),
            twiddles::single_twiddle(10, 64, inverse),
            twiddles::single_twiddle(15, 64, inverse),
            twiddles::single_twiddle(20, 64, inverse),
            twiddles::single_twiddle(25, 64, inverse),
            twiddles::single_twiddle(30, 64, inverse),
            twiddles::single_twiddle(35, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(6, 64, inverse),
            twiddles::single_twiddle(12, 64, inverse),
            twiddles::single_twiddle(18, 64, inverse),
            twiddles::single_twiddle(24, 64, inverse),
            twiddles::single_twiddle(30, 64, inverse),
            twiddles::single_twiddle(36, 64, inverse),
            twiddles::single_twiddle(42, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            twiddles::single_twiddle(7, 64, inverse),
            twiddles::single_twiddle(14, 64, inverse),
            twiddles::single_twiddle(21, 64, inverse),
            twiddles::single_twiddle(28, 64, inverse),
            twiddles::single_twiddle(35, 64, inverse),
            twiddles::single_twiddle(42, 64, inverse),
            twiddles::single_twiddle(49, 64, inverse),
        ];

        Self {
            twiddles: unsafe { [
                load_avx_f32!(twiddles, 0),
                load_avx_f32!(twiddles, 4),
                load_avx_f32!(twiddles, 8),
                load_avx_f32!(twiddles, 12),
                load_avx_f32!(twiddles, 16),
                load_avx_f32!(twiddles, 20),
                load_avx_f32!(twiddles, 24),
                load_avx_f32!(twiddles, 28),
                load_avx_f32!(twiddles, 32),
                load_avx_f32!(twiddles, 36),
                load_avx_f32!(twiddles, 40),
                load_avx_f32!(twiddles, 44),
                load_avx_f32!(twiddles, 48),
                load_avx_f32!(twiddles, 52),
                ]
            },
            twiddles_butterfly8: unsafe { 
                let twiddle = twiddles::single_twiddle(1, 8, inverse);
                _mm256_set_ps(twiddle.im, twiddle.re, twiddle.im, twiddle.re, twiddle.im, twiddle.re, twiddle.im, twiddle.re)
            },
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
}

impl MixedRadixAvx8x8<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn process_butterfly32_f32(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let input0 = load_avx_f32!(input, 0);
        let input2 = load_avx_f32!(input, 2 * 4);
        let input4 = load_avx_f32!(input, 4 * 4);
        let input6 = load_avx_f32!(input, 6 * 4);
        let input8 = load_avx_f32!(input, 8 * 4);
        let input10 = load_avx_f32!(input, 10 * 4);
        let input12 = load_avx_f32!(input, 12 * 4);
        let input14 = load_avx_f32!(input, 14 * 4);

        // We're going to treat our input as a 8x8 2d array. First, do 8 butterfly 8's down the columns of that array.
        let (mid0, mid2, mid4, mid6, mid8, mid10, mid12, mid14) = column_butterfly8_avx_f32!(input0, input2, input4, input6, input8, input10, input12, input14, self.twiddles_butterfly8, self.inverse);

        // Apply twiddle factors to the first half of our data
        let mid2_twiddled =  complex_multiply_f32!(mid2,  self.twiddles[0]);
        let mid4_twiddled =  complex_multiply_f32!(mid4,  self.twiddles[2]);
        let mid6_twiddled =  complex_multiply_f32!(mid6,  self.twiddles[4]);
        let mid8_twiddled =  complex_multiply_f32!(mid8,  self.twiddles[6]);
        let mid10_twiddled = complex_multiply_f32!(mid10, self.twiddles[8]);
        let mid12_twiddled = complex_multiply_f32!(mid12, self.twiddles[10]);
        let mid14_twiddled = complex_multiply_f32!(mid14, self.twiddles[12]);

        // Transpose the first half of this. After this, the compiler can spill this stuff, because it won't be needed until after the loads+butterfly8s+twiddles below are done
        let (transposed0, transposed2,  transposed4,  transposed6)  = transpose_4x4_f32!(mid0, mid2_twiddled, mid4_twiddled, mid6_twiddled);
        let (transposed1, transposed3,  transposed5,  transposed7)  = transpose_4x4_f32!(mid8_twiddled, mid10_twiddled, mid12_twiddled, mid14_twiddled);

        // Now that the first half of our data has been transposed, the compiler is free to spill those registers to make room for the other half
        let input1 = load_avx_f32!(input, 1 * 4);
        let input3 = load_avx_f32!(input, 3 * 4);
        let input5 = load_avx_f32!(input, 5 * 4);
        let input7 = load_avx_f32!(input, 7 * 4);
        let input9 = load_avx_f32!(input, 9 * 4);
        let input11 = load_avx_f32!(input, 11 * 4);
        let input13 = load_avx_f32!(input, 13 * 4);
        let input15 = load_avx_f32!(input, 15 * 4);
        let (mid1, mid3, mid5, mid7, mid9, mid11, mid13, mid15) = column_butterfly8_avx_f32!(input1, input3, input5, input7, input9, input11, input13, input15, self.twiddles_butterfly8, self.inverse);

        // Apply twiddle factors to the second half of our data
        let mid3_twiddled =  complex_multiply_f32!(mid3,  self.twiddles[1]);
        let mid5_twiddled =  complex_multiply_f32!(mid5,  self.twiddles[3]);
        let mid7_twiddled =  complex_multiply_f32!(mid7,  self.twiddles[5]);
        let mid9_twiddled =  complex_multiply_f32!(mid9,  self.twiddles[7]);
        let mid11_twiddled = complex_multiply_f32!(mid11, self.twiddles[9]);
        let mid13_twiddled = complex_multiply_f32!(mid13, self.twiddles[11]);
        let mid15_twiddled = complex_multiply_f32!(mid15, self.twiddles[13]);

        // Transpose the second half of our 8x8
        let (transposed8, transposed10, transposed12, transposed14) = transpose_4x4_f32!(mid1, mid3_twiddled, mid5_twiddled, mid7_twiddled);
        let (transposed9, transposed11, transposed13, transposed15) = transpose_4x4_f32!(mid9_twiddled, mid11_twiddled, mid13_twiddled, mid15_twiddled);

        // Do 4 butterfly 8's down the columns of our transposed array, and store the results
        let (output0, output2, output4, output6, output8, output10, output12, output14) = column_butterfly8_avx_f32!(transposed0, transposed2, transposed4, transposed6, transposed8, transposed10, transposed12, transposed14, self.twiddles_butterfly8, self.inverse);
        store_avx_f32!(output, 0, output0);
        store_avx_f32!(output, 2 * 4, output2);
        store_avx_f32!(output, 4 * 4, output4);
        store_avx_f32!(output, 6 * 4, output6);
        store_avx_f32!(output, 8 * 4, output8);
        store_avx_f32!(output, 10 * 4, output10);
        store_avx_f32!(output, 12 * 4, output12);
        store_avx_f32!(output, 14 * 4, output14);

        // We freed up a bunch of registers, so we should easily have enough room to compute+store the other half of our butterfly 8s
        let (output1, output3, output5, output7, output9, output11, output13, output15) = column_butterfly8_avx_f32!(transposed1, transposed3, transposed5, transposed7, transposed9, transposed11, transposed13, transposed15, self.twiddles_butterfly8, self.inverse);
        store_avx_f32!(output, 1 * 4, output1);
        store_avx_f32!(output, 3 * 4, output3);
        store_avx_f32!(output, 5 * 4, output5);
        store_avx_f32!(output, 7 * 4, output7);
        store_avx_f32!(output, 9 * 4, output9);
        store_avx_f32!(output, 11 * 4, output11);
        store_avx_f32!(output, 13 * 4, output13);
        store_avx_f32!(output, 15 * 4, output15);
    }
}
default impl<T: FFTnum> FFT<T> for MixedRadixAvx8x8<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for MixedRadixAvx8x8<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());

        unsafe { self.process_butterfly32_f32(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            unsafe { self.process_butterfly32_f32(in_chunk, out_chunk) };
        }
    }
}
impl<T> Length for MixedRadixAvx8x8<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        64
    }
}
impl<T> IsInverse for MixedRadixAvx8x8<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}



#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_fft_algorithm;
    use std::sync::Arc;
    use algorithm::*;

    #[test]
    fn test_splitradix() {
        for pow in 3..8 {
            let len = 1 << pow;
            test_splitradix_with_length(len, false);
            test_splitradix_with_length(len, true);
        }
    }

    fn test_splitradix_with_length(len: usize, inverse: bool) {
        let quarter = Arc::new(DFT::new(len / 4, inverse));
        let half = Arc::new(DFT::new(len / 2, inverse));
        let fft = SplitRadix::new(half, quarter);

        check_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_avx_splitradix() {
        for pow in 4..8 {
            let len = 1 << pow;
            test_avx_splitradix_with_length(len, false);
            test_avx_splitradix_with_length(len, true);
        }
    }

    fn test_avx_splitradix_with_length(len: usize, inverse: bool) {
        let quarter = Arc::new(DFT::new(len / 4, inverse)) as Arc<dyn FFT<f32>>;
        let half = Arc::new(DFT::new(len / 2, inverse)) as Arc<dyn FFT<f32>>;
        let fft = SplitRadixAvx::new(half, quarter);

        check_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_avx_splitradix_butterfly8() {
        let fft_forward = MixedRadixAvx4x2::new(false);
        check_fft_algorithm(&fft_forward, 8, false);

        let fft_inverse = MixedRadixAvx4x2::new(true);
        check_fft_algorithm(&fft_inverse, 8, true);
    }

    #[test]
    fn test_avx_mixedradix4x4() {
        let fft_forward = MixedRadixAvx4x4::new(false);
        check_fft_algorithm(&fft_forward, 16, false);

        let fft_inverse = MixedRadixAvx4x4::new(true);
        check_fft_algorithm(&fft_inverse, 16, true);
    }

    #[test]
    fn test_avx_mixedradix4x8() {
        let fft_forward = MixedRadixAvx4x8::new(false);
        check_fft_algorithm(&fft_forward, 32, false);

        let fft_inverse = MixedRadixAvx4x8::new(true);
        check_fft_algorithm(&fft_inverse, 32, true);
    }

    #[test]
    fn test_avx_mixedradix8x8() {
        let fft_forward = MixedRadixAvx8x8::new(false);
        check_fft_algorithm(&fft_forward, 64, false);

        let fft_inverse = MixedRadixAvx8x8::new(true);
        check_fft_algorithm(&fft_inverse, 64, true);
    }

    #[test]
    fn test_avx_splitradix_butterfly16() {
        let dft_8 = Arc::new(DFT::new(8, false));
        let dft_4 = Arc::new(DFT::new(4, false));
        let control = SplitRadix::new(dft_8, dft_4);

        check_fft_algorithm(&control, 16, false);

        let fft_forward = SplitRadixAvxButterfly16::new(false);
        check_fft_algorithm(&fft_forward, 16, false);

        let fft_inverse = SplitRadixAvxButterfly16::new(true);
        check_fft_algorithm(&fft_inverse, 16, true);
    }

    #[test]
    fn test_avx_splitradix_butterfly32() {
        let fft_forward = SplitRadixAvxButterfly32::new(false);
        check_fft_algorithm(&fft_forward, 32, false);

        let fft_inverse = SplitRadixAvxButterfly32::new(true);
        check_fft_algorithm(&fft_inverse, 32, true);
    }
}
