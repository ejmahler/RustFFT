use std::sync::Arc;

use num_complex::Complex;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, IsInverse, FFT};
use twiddles;

use super::mixed_radix_cxn::MixedRadix4x4Avx;
use super::butterflies::Butterfly8;
use algorithm::butterflies::FFTButterfly;

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

// In practice, the compiler is smart enough to convert these 4 instructions into just _mm256_unpacklo_pd and _mm256_unpackhi_pd
macro_rules! transpose_4x2_to_2x4_f32 {
    ($row0: expr, $row1: expr) => {{
        let unpacked_0 = _mm256_unpacklo_ps($row0, $row1);
        let unpacked_1 = _mm256_unpackhi_ps($row0, $row1);

        

        (output_0, output_1)
    }};
}

// the given row contains 4 complex numbers, and this does horizontal butterfly 2's between adjacent pairs
macro_rules! butterfly2_horizontal_f32 {
    ($row: expr) => {{
        row
    }};
}


// This variany assumes that `left` should be conjugated before multiplying
// Thankfully, it is straightforward to roll this into existing instructions. Namely, we can get away with replacing "fmaddsub" with "fmsubadd"
macro_rules! butterfly8_f32 {
    ($row0: expr, $row1: expr, $twiddles: expr) => {{
        // First, do the butterfly 2's across the rows
        let intermediate0 = _mm256_add_ps($row0, row1);
        let intermediate1_pretwiddle = _mm256_sub_ps($row0, intermediate1_pretwiddle);

        // Apply the size-8 twiddle factors
        let intermediate1 = complex_multiply_f32!(intermediate1_pretwiddle, $twiddles);

        // Swap the middle elements of each row. Our goal is to enable the butterfly4s
        // The first 2 elements of row0 will be the first 2 elements of the first butterfly 4, and the other two will be the first 2 elements of the other butterfly 4
        let (butterfly4_row0, butterfly4_row1) = transpose_4x2_to_2x4_f32(intermediate0, intermediate1);

        // Do horizontal butterflies across our transposed data -- in addition, apply the butterfly 4 twiddle rotation to the last element
        let row01 = butterfly2_horizontal_f32!(butterfly4_row0);
        let row23 = butterfly2_horizontal_f32!(butterfly4_row1);

        // Rotate the 2 odd complex elements of row23 to accomplish the butterlfy 4 twiddle factors

        // Do vertical butterflies
        let final01 = _mm256_add_ps(row01, row23);
        let final23 = _mm256_sub_ps(row01, row23);

        (final01, final23)
    }};
}


impl SplitRadixAvx<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        let half_len = self.len / 2;
        let quarter_len = self.len / 4;
        let sixteenth_len = self.len / 16;

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
        for i in 0..sixteenth_len {
            let twiddle = _mm256_loadu_ps(to_float_ptr(self.twiddles.get_unchecked(i * 4)));
            let inner_quarter1_entry = _mm256_loadu_ps(to_float_ptr(inner_quarter1_output.get_unchecked(i * 4)));
            let inner_quarter3_entry = _mm256_loadu_ps(to_float_ptr(inner_quarter3_output.get_unchecked(i * 4)));

            let twiddled_quarter1 = complex_multiply_f32!(twiddle, inner_quarter1_entry);
            let twiddled_quarter3 = complex_conj_multiply_f32!(twiddle, inner_quarter3_entry);
            let quarter_sum  = _mm256_add_ps(twiddled_quarter1, twiddled_quarter3);
            let quarter_diff = _mm256_sub_ps(twiddled_quarter1, twiddled_quarter3);

            let inner_half_entry = _mm256_loadu_ps(to_float_ptr(inner_half_output.get_unchecked(i * 4)));

            let output_i = _mm256_add_ps(inner_half_entry, quarter_sum);
            let output_i_half = _mm256_sub_ps(inner_half_entry, quarter_sum);

            _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(i * 4)), output_i);
            _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(i * 4 + half_len)), output_i_half);
            
            // compute the twiddle for quarter diff by rotating it
            // Apply quarter diff inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
            let quarter_diff_swapped = _mm256_permute_ps(quarter_diff, 0xB1);

            // negate ALL elements in quarter_diff, then blend in only the negated odd ones
            // TODO: See if we can roll this into downstream operations? for example, output_quarter1_preswap can possibly use addsub instead of add
            let quarter_diff_negated = _mm256_xor_ps(quarter_diff_swapped, _mm256_set1_ps(-0.0));
            let quarter_diff_rotated = _mm256_blend_ps(quarter_diff_swapped, quarter_diff_negated, 0xAA);

            let inner_half3_entry = _mm256_loadu_ps(to_float_ptr(inner_half_output.get_unchecked(i * 4 + quarter_len)));

            let output_quarter1_preswap = _mm256_add_ps(inner_half3_entry, quarter_diff_rotated);
            let output_quarter3_preswap = _mm256_sub_ps(inner_half3_entry, quarter_diff_rotated);

            let (output_quarter1, output_quarter3) = if self.is_inverse() {
                (output_quarter3_preswap, output_quarter1_preswap)
            } else {
                (output_quarter1_preswap, output_quarter3_preswap)
            };
            _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(i * 4 + quarter_len)), output_quarter1);
            _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(i * 4 + quarter_len * 3)), output_quarter3);
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



pub struct SplitRadixAvxButterfly32<T> {
    butterfly16: MixedRadix4x4Avx<T>,
    butterfly8: Butterfly8<T>,
    twiddles: [Complex<T>; 8],
    inverse: bool,
}
impl<T: FFTnum> SplitRadixAvxButterfly32<T>
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        assert!(has_avx && has_fma, "The SplitRadixAvxButterfly32 algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

        Self {
            butterfly16: MixedRadix4x4Avx::new(inverse),
            butterfly8: Butterfly8::new(inverse),
            twiddles: [
                Complex{ re: T::one(), im: T::zero() },
                twiddles::single_twiddle(1, 32, inverse),
                twiddles::single_twiddle(2, 32, inverse),
                twiddles::single_twiddle(3, 32, inverse),
                twiddles::single_twiddle(4, 32, inverse),
                twiddles::single_twiddle(5, 32, inverse),
                twiddles::single_twiddle(6, 32, inverse),
                twiddles::single_twiddle(7, 32, inverse),
            ],
            inverse: inverse,
        }
    }
}

impl SplitRadixAvxButterfly32<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn process_f32(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {

        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        let mut scratch_evens = [
            *input.get_unchecked(0),
            *input.get_unchecked(2),
            *input.get_unchecked(4),
            *input.get_unchecked(6),
            *input.get_unchecked(8),
            *input.get_unchecked(10),
            *input.get_unchecked(12),
            *input.get_unchecked(14),
            *input.get_unchecked(16),
            *input.get_unchecked(18),
            *input.get_unchecked(20),
            *input.get_unchecked(22),
            *input.get_unchecked(24),
            *input.get_unchecked(26),
            *input.get_unchecked(28),
            *input.get_unchecked(30),
        ];

        let mut scratch_odds_n1 = [
            *input.get_unchecked(1),
            *input.get_unchecked(5),
            *input.get_unchecked(9),
            *input.get_unchecked(13),
            *input.get_unchecked(17),
            *input.get_unchecked(21),
            *input.get_unchecked(25),
            *input.get_unchecked(29),
        ];
        let mut scratch_odds_n3 = [
            *input.get_unchecked(31),
            *input.get_unchecked(3),
            *input.get_unchecked(7),
            *input.get_unchecked(11),
            *input.get_unchecked(15),
            *input.get_unchecked(19),
            *input.get_unchecked(23),
            *input.get_unchecked(27),
        ];

        // step 2: column FFTs
        self.butterfly16.process_inplace(&mut scratch_evens);
        self.butterfly8.process_inplace(&mut scratch_odds_n1);
        self.butterfly8.process_inplace(&mut scratch_odds_n3);

        // Recombine into a single buffer
        for i in 0..2 {
            let twiddle = _mm256_loadu_ps(to_float_ptr(self.twiddles.get_unchecked(i * 4)));
            let inner_quarter1_entry = _mm256_loadu_ps(to_float_ptr(scratch_odds_n1.get_unchecked(i * 4)));
            let inner_quarter3_entry = _mm256_loadu_ps(to_float_ptr(scratch_odds_n3.get_unchecked(i * 4)));

            let twiddled_quarter1 = complex_multiply_f32!(twiddle, inner_quarter1_entry);
            let twiddled_quarter3 = complex_conj_multiply_f32!(twiddle, inner_quarter3_entry);
            let quarter_sum  = _mm256_add_ps(twiddled_quarter1, twiddled_quarter3);
            let quarter_diff = _mm256_sub_ps(twiddled_quarter1, twiddled_quarter3);

            let inner_half_entry = _mm256_loadu_ps(to_float_ptr(scratch_evens.get_unchecked(i * 4)));

            let output_i = _mm256_add_ps(inner_half_entry, quarter_sum);
            let output_i_half = _mm256_sub_ps(inner_half_entry, quarter_sum);

            _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(i * 4)), output_i);
            _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(i * 4 + 16)), output_i_half);
            
            // compute the twiddle for quarter diff by rotating it
            // Apply quarter diff inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
            let quarter_diff_swapped = _mm256_permute_ps(quarter_diff, 0xB1);

            // negate ALL elements in quarter_diff, then blend in only the negated odd ones
            // TODO: See if we can roll this into downstream operations? for example, output_quarter1_preswap can possibly use addsub instead of add
            let quarter_diff_negated = _mm256_xor_ps(quarter_diff_swapped, _mm256_set1_ps(-0.0));
            let quarter_diff_rotated = _mm256_blend_ps(quarter_diff_swapped, quarter_diff_negated, 0xAA);

            let inner_half3_entry = _mm256_loadu_ps(to_float_ptr(scratch_evens.get_unchecked(i * 4 + 8)));

            let output_quarter1_preswap = _mm256_add_ps(inner_half3_entry, quarter_diff_rotated);
            let output_quarter3_preswap = _mm256_sub_ps(inner_half3_entry, quarter_diff_rotated);

            let (output_quarter1, output_quarter3) = if self.is_inverse() {
                (output_quarter3_preswap, output_quarter1_preswap)
            } else {
                (output_quarter1_preswap, output_quarter3_preswap)
            };
            _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(i * 4 + 8)), output_quarter1);
            _mm256_storeu_ps(to_float_mut_ptr(output.get_unchecked_mut(i * 4 + 24)), output_quarter3);
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

        unsafe { self.process_f32(input, output) };
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            unsafe { self.process_f32(in_chunk, out_chunk) };
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




#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_fft_algorithm;
    use std::sync::Arc;
    use algorithm::DFT;

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
        let quarter = Arc::new(DFT::new(len / 4, inverse));
        let half = Arc::new(DFT::new(len / 2, inverse));
        let fft = SplitRadixAvx::new(half, quarter);

        check_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_avx_splitradix_butterfly32() {
        let fft_forward = SplitRadixAvxButterfly32::new(false);
        check_fft_algorithm(&fft_forward, 32, false);

        let fft_forward = SplitRadixAvxButterfly32::new(true);
        check_fft_algorithm(&fft_forward, 32, true);
    }
}
