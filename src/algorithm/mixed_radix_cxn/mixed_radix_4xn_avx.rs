use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;
use transpose;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, IsInverse, FFT};
use algorithm::butterflies::FFTButterfly;
use twiddles;

use std::arch::x86_64::*;

use super::ColumnChunksExactMut4;
use super::mixed_radix_4xn::{process_butterfly4_chunks_naive, column_butterfly4s_inverse};

const CHUNKS_PER_SUPERCHUNK : usize = 1;
const CHUNK_SIZE : usize = 4;
const FLOATS_PER_CHUNK : usize = CHUNK_SIZE * 2;

fn to_float_ptr<T: FFTnum>(slice: &[Complex<T>]) -> *const T {
	unsafe { std::mem::transmute::<*const Complex<T>, *const T>(slice.as_ptr()) }
}

fn to_float_mut_ptr<T: FFTnum>(slice: &mut [Complex<T>]) -> *mut T {
	unsafe { std::mem::transmute::<*mut Complex<T>, *mut T>(slice.as_mut_ptr()) }
}



/// Specialized implementation of the Mixed-Radix FFT algorithm where the FFT size is a multiple of 4. Uses AVX instructions.
///
/// This algorithm factors a FFT into 4 * N, computes several inner FFTs of size 4 and N, then combines the 
/// results to get the final answer. N must greater than or equal to 4.
///
/// ~~~
/// // Computes a forward FFT of size 1200, using the Mixed-Radix 2xN Algorithm
/// use rustfft::algorithm::MixedRadix2xN;
/// use rustfft::{FFT, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1200];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1200];
///
/// // we need to find a N such that 4 * N == 1200
/// // N = 300 satisfies this
/// let mut planner = FFTplanner::new(false);
/// let inner_fft = planner.plan_fft(300);
///
/// // the mixed radix FFT length will be 4 * inner_fft.len() = 1200
/// let fft = MixedRadix4xnAvx::new(inner_fft);
/// fft.process(&mut input, &mut output);
/// ~~~

pub struct MixedRadix4xnAvx<T> {
    inner_fft: Arc<FFT<T>>,
    twiddles: Box<[Complex<T>]>,

    chunk_size: usize,
    inner_len: usize,

    inverse: bool,
}

impl<T: FFTnum> MixedRadix4xnAvx<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `4 * inner_fft.len()`
    pub fn new(inner_fft: Arc<FFT<T>>) -> Self {
    	let has_avx = is_x86_feature_detected!("avx");
    	let has_fma = is_x86_feature_detected!("fma");
    	assert!(has_avx && has_fma, "The MixedRadix4xnAvx algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

    	let inner_len = inner_fft.len();
        let len = 4 * inner_len;

        let chunk_size = Self::get_chunk_size();
        let chunk_count = inner_len / chunk_size;

        let inverse = inner_fft.is_inverse();

        let mut twiddles = Vec::with_capacity(len / 4 * 3);

        // Chunk widdles
        for x_chunk in 0..chunk_count {
        	for y in 1..4 {
        		for x_idx in 0..chunk_size {
        			let x = x_chunk * chunk_size + x_idx;
	                twiddles.push(twiddles::single_twiddle(x * y, len, inverse));
	            }
	        }
        }

        // Remainder twiddles
        for y in 1..4 {
        	for x in chunk_count*chunk_size..inner_len {
        		twiddles.push(twiddles::single_twiddle(x * y, len, inverse));
        	}
        }

        Self {
            inner_fft,
            twiddles: twiddles.into_boxed_slice(),
            chunk_size,
            inner_len,
            inverse,
        }
    }

    // When doing our FFTs of size 4, we're actually going to do them in chunks of this size.
    // The simplest thing to do would be a chunk size of 1, but benchmarking shows that more is decidedly faster
    // We will store our twiddles in rows of this chunk size for easy SIMD layout
    fn get_chunk_size() -> usize {
    	CHUNKS_PER_SUPERCHUNK * CHUNK_SIZE
    }
}

impl MixedRadix4xnAvx<f32> {
	fn perform_fft_f32(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        // six step FFT, just like the main mixed radix implementation. But swe know the eact size of one of the two FFTs, it's pretty safe to skip the transposes, and we can be a little smarter with our twiddle factors 

        // STEP 1: Transpose -- skipped because we will do the first set of FFTs non-contiguously

        // STEP 2 and 3 combined: Perform FFTs of size 4 and apply our twiddle factors.
        if !self.inverse {
	        unsafe { column_butterfly4s_forward_avx_f32(input, &self.twiddles, self.chunk_size) };
	    } else {
	    	column_butterfly4s_inverse(input, &self.twiddles, self.chunk_size);
	    }
	    output.copy_from_slice(input);
       
       	// STEP 4: Transpose -- skipped because we will do the first set of FFTs non-contiguously

        // STEP 5: perform FFTs of size `inner_len`
        self.inner_fft.process_multi(output, input);

        // STEP 6: transpose again
        transpose::transpose(input, output, self.inner_len, 4);
    }
}
default impl<T: FFTnum> FFT<T> for MixedRadix4xnAvx<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for MixedRadix4xnAvx<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());

        self.perform_fft_f32(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
            self.perform_fft_f32(in_chunk, out_chunk);
        }
    }
}
impl<T> Length for MixedRadix4xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len() / 3 * 4
    }
}
impl<T> IsInverse for MixedRadix4xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
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

#[target_feature(enable = "avx", enable = "fma")]
unsafe fn column_butterfly4s_forward_avx_f32(buffer: &mut [Complex<f32>], outer_twiddles: &[Complex<f32>], chunk_size: usize) {
	assert_eq!(buffer.len() / 4 * 3, outer_twiddles.len());

	let mut column_chunk_iter = ColumnChunksExactMut4::new(buffer, chunk_size);
	let mut outer_twiddle_chunks = outer_twiddles.chunks_exact(chunk_size * 3);	

	for (column_chunks, outer_twiddle_chunk) in column_chunk_iter.by_ref().zip(outer_twiddle_chunks.by_ref()) {
		let (twiddle1_row, twiddle23_row) = outer_twiddle_chunk.split_at(chunk_size);
		let (twiddle2_row, twiddle3_row) = twiddle23_row.split_at(chunk_size);

		for i in 0..CHUNKS_PER_SUPERCHUNK {
			// Load rows of FFT data from our column arrays 
			let element_0 = _mm256_loadu_ps(to_float_ptr(column_chunks[0]).add(FLOATS_PER_CHUNK * i));
			let element_1 = _mm256_loadu_ps(to_float_ptr(column_chunks[1]).add(FLOATS_PER_CHUNK * i));
			let element_2 = _mm256_loadu_ps(to_float_ptr(column_chunks[2]).add(FLOATS_PER_CHUNK * i));
			let element_3 = _mm256_loadu_ps(to_float_ptr(column_chunks[3]).add(FLOATS_PER_CHUNK * i));

			// Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
			let element_0_middle = _mm256_add_ps(element_0, element_2);
			let element_1_middle = _mm256_add_ps(element_1, element_3);
			let element_2_middle = _mm256_sub_ps(element_0, element_2);
			let element_3_middle_pretwiddle = _mm256_sub_ps(element_1, element_3);

			// Apply element 3 inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
			let element_3_swapped = _mm256_permute_ps(element_3_middle_pretwiddle, 0xB1);

			// negate ALL elements in element 3, then blend in only the negated odd ones
			// TODO: See if we can roll this into downstream operations? for example, element_2_final_pretwiddle can possibly use addsub instead of add
			let element_3_negated = _mm256_xor_ps(element_3_swapped, _mm256_set1_ps(-0.0));
			let element_3_middle = _mm256_blend_ps(element_3_swapped, element_3_negated, 0xAA);

			// Perform the second set of size-2 FFTs
			let element_0_final = _mm256_add_ps(element_0_middle, element_1_middle);
			let element_1_final_pretwiddle = _mm256_sub_ps(element_0_middle, element_1_middle);
			let element_2_final_pretwiddle = _mm256_add_ps(element_2_middle, element_3_middle);
			let element_3_final_pretwiddle = _mm256_sub_ps(element_2_middle, element_3_middle);

			let twiddle1 = _mm256_loadu_ps(to_float_ptr(twiddle1_row).add(FLOATS_PER_CHUNK * i));
			let twiddle2 = _mm256_loadu_ps(to_float_ptr(twiddle2_row).add(FLOATS_PER_CHUNK * i));
			let twiddle3 = _mm256_loadu_ps(to_float_ptr(twiddle3_row).add(FLOATS_PER_CHUNK * i));
			let element_1_final = complex_multiply_f32!(element_2_final_pretwiddle, twiddle1);
			let element_2_final = complex_multiply_f32!(element_1_final_pretwiddle, twiddle2);
			let element_3_final = complex_multiply_f32!(element_3_final_pretwiddle, twiddle3);

			// Write back, and swap elements 1 and 2 as we do for the transpose
			_mm256_storeu_ps(to_float_mut_ptr(column_chunks[0]).add(FLOATS_PER_CHUNK * i), element_0_final);
			_mm256_storeu_ps(to_float_mut_ptr(column_chunks[1]).add(FLOATS_PER_CHUNK * i), element_1_final);
			_mm256_storeu_ps(to_float_mut_ptr(column_chunks[2]).add(FLOATS_PER_CHUNK * i), element_2_final);
			_mm256_storeu_ps(to_float_mut_ptr(column_chunks[3]).add(FLOATS_PER_CHUNK * i), element_3_final);
		}
	}

	let column_chunks = column_chunk_iter.into_remainder();
	let outer_twiddle_chunk = outer_twiddle_chunks.remainder();
	let remainder_len = column_chunks[0].len();
	let twiddle_fn = |c: Complex<f32>| Complex{ re: c.im, im: -c.re };
	process_butterfly4_chunks_naive(column_chunks, outer_twiddle_chunk, remainder_len, twiddle_fn);
}

/// Specialized implementation of the Mixed-Radix FFT algorithm for size 16. Uses AVX instructions.
pub struct MixedRadix4x4Avx<T> {
    twiddles: [Complex<T>;12],
    inverse: bool,
}

impl<T: FFTnum> MixedRadix4x4Avx<T> {
    /// Creates a FFT instance which will process inputs/outputs of size 16.
    pub fn new(inverse: bool) -> Self {
    	let has_avx = is_x86_feature_detected!("avx");
    	let has_fma = is_x86_feature_detected!("fma");
    	assert!(has_avx && has_fma, "The MixedRadix4xnAvx algorithm requires the 'avx' and 'fma' to exist on this machine. avx detected: {}, fma fetected: {}", has_avx, has_fma);

    	let mut result = Self {
            twiddles: [Complex::zero(); 12],
            inverse,
        };

        // Remainder twiddles
        for y in 1..4 {
        	for x in 0..4 {
        		let index = (y - 1) * 4 + x;
        		result.twiddles[index] = twiddles::single_twiddle(x * y, 16, inverse);
        	}
        }

        result
    }
}

impl MixedRadix4x4Avx<f32> {
	#[inline(always)]
	fn process_f32(&self, buffer: &mut [Complex<f32>]) {
		if self.inverse {
	        unsafe { mixed_radix_4x4_avx_inverse_f32(buffer, &self.twiddles) };
	    }  else {
	    	unsafe { mixed_radix_4x4_avx_forward_f32(buffer, &self.twiddles) };
	    }
	}
}

default impl<T: FFTnum> FFTButterfly<T> for MixedRadix4x4Avx<T> {
    #[inline(always)]
    unsafe fn process_inplace(&self, _buffer: &mut [Complex<T>]) {
        unimplemented!();
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, _buffer: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFTButterfly<f32> for MixedRadix4x4Avx<f32> {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<f32>]) {
        self.process_f32(buffer);
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<f32>]) {
        for chunk in buffer.chunks_exact_mut(self.len()) {
        	self.process_f32(chunk);
        }
    }
}
default impl<T: FFTnum> FFT<T> for MixedRadix4x4Avx<T> {
    fn process(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>]) {
        unimplemented!();
    }
}
impl FFT<f32> for MixedRadix4x4Avx<f32> {
    fn process(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);
      	self.process_f32(output);
    }
    fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);
        for chunk in output.chunks_exact_mut(self.len()) {
        	self.process_f32(chunk);
        }
    }
}
impl<T> Length for MixedRadix4x4Avx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        16
    }
}
impl<T> IsInverse for MixedRadix4x4Avx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
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

#[target_feature(enable = "avx", enable = "fma")]
unsafe fn mixed_radix_4x4_avx_forward_f32(buffer: &mut [Complex<f32>], outer_twiddles: &[Complex<f32>; 12]) {
	let buffer_ptr = to_float_ptr(buffer);
	
	let element_0 = _mm256_loadu_ps(buffer_ptr);
	let element_1 = _mm256_loadu_ps(buffer_ptr.add(FLOATS_PER_CHUNK));
	let element_2 = _mm256_loadu_ps(buffer_ptr.add(FLOATS_PER_CHUNK * 2));
	let element_3 = _mm256_loadu_ps(buffer_ptr.add(FLOATS_PER_CHUNK * 3));

	// First, process the columns
	// Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
	let element_0_middle = _mm256_add_ps(element_0, element_2);
	let element_1_middle = _mm256_add_ps(element_1, element_3);
	let element_2_middle = _mm256_sub_ps(element_0, element_2);
	let element_3_middle_pretwiddle = _mm256_sub_ps(element_1, element_3);

	// Apply element 3 inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
	let element_3_swapped = _mm256_permute_ps(element_3_middle_pretwiddle, 0xB1);

	// negate ALL elements in element 3, then blend in only the negated odd ones
	// TODO: See if we can roll this into downstream operations? for example, element_2_final_pretwiddle can possibly use addsub instead of add
	let element_3_negated = _mm256_xor_ps(element_3_swapped, _mm256_set1_ps(-0.0));
	let element_3_middle = _mm256_blend_ps(element_3_swapped, element_3_negated, 0xAA);

	// Perform the second set of size-2 FFTs
	let element_0_final = _mm256_add_ps(element_0_middle, element_1_middle);
	let element_1_final_pretwiddle = _mm256_sub_ps(element_0_middle, element_1_middle);
	let element_2_final_pretwiddle = _mm256_add_ps(element_2_middle, element_3_middle);
	let element_3_final_pretwiddle = _mm256_sub_ps(element_2_middle, element_3_middle);

	let twiddle1 = _mm256_loadu_ps(to_float_ptr(&outer_twiddles[..]));
	let twiddle2 = _mm256_loadu_ps(to_float_ptr(&outer_twiddles[4..]));
	let twiddle3 = _mm256_loadu_ps(to_float_ptr(&outer_twiddles[8..]));
	let element_1_final = complex_multiply_f32!(element_2_final_pretwiddle, twiddle1);
	let element_2_final = complex_multiply_f32!(element_1_final_pretwiddle, twiddle2);
	let element_3_final = complex_multiply_f32!(element_3_final_pretwiddle, twiddle3);

	// The first set of 4 butterlies is done. Now, transpose the columns to the rows.
	let (row_0, row_1, row_2, row_3) = transpose_4x4_f32!(element_0_final, element_1_final, element_2_final, element_3_final);

	// We've transposed. Now, process size 4 butterflies again.
	// Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
	let element_0_middle = _mm256_add_ps(row_0, row_2);
	let element_1_middle = _mm256_add_ps(row_1, row_3);
	let element_2_middle = _mm256_sub_ps(row_0, row_2);
	let element_3_middle_pretwiddle = _mm256_sub_ps(row_1, row_3);

	// Apply element 3 inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
	let element_3_swapped = _mm256_permute_ps(element_3_middle_pretwiddle, 0xB1);

	// negate ALL elements in element 3, then blend in only the negated odd ones
	// TODO: See if we can roll this into downstream operations? for example, element_2_final_pretwiddle can possibly use addsub instead of add
	let element_3_negated = _mm256_xor_ps(element_3_swapped, _mm256_set1_ps(-0.0));
	let element_3_middle = _mm256_blend_ps(element_3_swapped, element_3_negated, 0xAA);

	// Perform the second set of size-2 FFTs
	let element_0_final = _mm256_add_ps(element_0_middle, element_1_middle);
	let element_1_final = _mm256_sub_ps(element_0_middle, element_1_middle);
	let element_2_final = _mm256_add_ps(element_2_middle, element_3_middle);
	let element_3_final = _mm256_sub_ps(element_2_middle, element_3_middle);

	// Write back, and swap elements 1 and 2 as we do for the transpose
	let buffer_mut_ptr = to_float_mut_ptr(buffer);
	_mm256_storeu_ps(buffer_mut_ptr, element_0_final);
	_mm256_storeu_ps(buffer_mut_ptr.add(FLOATS_PER_CHUNK), element_2_final);
	_mm256_storeu_ps(buffer_mut_ptr.add(FLOATS_PER_CHUNK * 2), element_1_final);
	_mm256_storeu_ps(buffer_mut_ptr.add(FLOATS_PER_CHUNK * 3), element_3_final);
}


#[target_feature(enable = "avx", enable = "fma")]
unsafe fn mixed_radix_4x4_avx_inverse_f32(buffer: &mut [Complex<f32>], outer_twiddles: &[Complex<f32>; 12]) {
	let buffer_ptr = to_float_ptr(buffer);
	
	let element_0 = _mm256_loadu_ps(buffer_ptr);
	let element_1 = _mm256_loadu_ps(buffer_ptr.add(FLOATS_PER_CHUNK));
	let element_2 = _mm256_loadu_ps(buffer_ptr.add(FLOATS_PER_CHUNK * 2));
	let element_3 = _mm256_loadu_ps(buffer_ptr.add(FLOATS_PER_CHUNK * 3));

	// First, process the columns
	// Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
	let element_0_middle = _mm256_add_ps(element_0, element_2);
	let element_1_middle = _mm256_add_ps(element_1, element_3);
	let element_2_middle = _mm256_sub_ps(element_0, element_2);
	let element_3_middle_pretwiddle = _mm256_sub_ps(element_1, element_3);

	// Apply element 3 inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
	let element_3_swapped = _mm256_permute_ps(element_3_middle_pretwiddle, 0xB1);

	// negate ALL elements in element 3, then blend in only the negated odd ones
	// TODO: See if we can roll this into downstream operations? for example, element_2_final_pretwiddle can possibly use addsub instead of add
	let element_3_negated = _mm256_xor_ps(element_3_swapped, _mm256_set1_ps(-0.0));
	let element_3_middle = _mm256_blend_ps(element_3_swapped, element_3_negated, 0x55);

	// Perform the second set of size-2 FFTs
	let element_0_final = _mm256_add_ps(element_0_middle, element_1_middle);
	let element_1_final_pretwiddle = _mm256_sub_ps(element_0_middle, element_1_middle);
	let element_2_final_pretwiddle = _mm256_add_ps(element_2_middle, element_3_middle);
	let element_3_final_pretwiddle = _mm256_sub_ps(element_2_middle, element_3_middle);

	let twiddle1 = _mm256_loadu_ps(to_float_ptr(&outer_twiddles[..]));
	let twiddle2 = _mm256_loadu_ps(to_float_ptr(&outer_twiddles[4..]));
	let twiddle3 = _mm256_loadu_ps(to_float_ptr(&outer_twiddles[8..]));
	let element_1_final = complex_multiply_f32!(element_2_final_pretwiddle, twiddle1);
	let element_2_final = complex_multiply_f32!(element_1_final_pretwiddle, twiddle2);
	let element_3_final = complex_multiply_f32!(element_3_final_pretwiddle, twiddle3);

	// The first set of 4 butterlies is done. Now, transpose the columns to the rows.
	let (row_0, row_1, row_2, row_3) = transpose_4x4_f32!(element_0_final, element_1_final, element_2_final, element_3_final);

	// We've transposed. Now, process size 4 butterflies again.
	// Perform the first set of size-2 FFTs. Make sure to apply the twiddle factor to element 3.
	let element_0_middle = _mm256_add_ps(row_0, row_2);
	let element_1_middle = _mm256_add_ps(row_1, row_3);
	let element_2_middle = _mm256_sub_ps(row_0, row_2);
	let element_3_middle_pretwiddle = _mm256_sub_ps(row_1, row_3);

	// Apply element 3 inner twiddle factor by swapping reals with imaginaries, then negating the imaginaries
	let element_3_swapped = _mm256_permute_ps(element_3_middle_pretwiddle, 0xB1);

	// negate ALL elements in element 3, then blend in only the negated odd ones
	// TODO: See if we can roll this into downstream operations? for example, element_2_final_pretwiddle can possibly use addsub instead of add
	let element_3_negated = _mm256_xor_ps(element_3_swapped, _mm256_set1_ps(-0.0));
	let element_3_middle = _mm256_blend_ps(element_3_swapped, element_3_negated, 0x55);

	// Perform the second set of size-2 FFTs
	let element_0_final = _mm256_add_ps(element_0_middle, element_1_middle);
	let element_1_final = _mm256_sub_ps(element_0_middle, element_1_middle);
	let element_2_final = _mm256_add_ps(element_2_middle, element_3_middle);
	let element_3_final = _mm256_sub_ps(element_2_middle, element_3_middle);

	// Write back, and swap elements 1 and 2 as we do for the transpose
	let buffer_mut_ptr = to_float_mut_ptr(buffer);
	_mm256_storeu_ps(buffer_mut_ptr, element_0_final);
	_mm256_storeu_ps(buffer_mut_ptr.add(FLOATS_PER_CHUNK), element_2_final);
	_mm256_storeu_ps(buffer_mut_ptr.add(FLOATS_PER_CHUNK * 2), element_1_final);
	_mm256_storeu_ps(buffer_mut_ptr.add(FLOATS_PER_CHUNK * 3), element_3_final);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use test_utils::check_fft_algorithm;
    use algorithm::DFT;

    #[test]
    fn test_mixedx4avx_radix() {
        for width in 8..9 {
            test_mixed_radix_with_length(width, false);
            test_mixed_radix_with_length(width, true);
        }
    }

    fn test_mixed_radix_with_length(inner_len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(inner_len, inverse)) as Arc<FFT<f32>>;

        let fft = MixedRadix4xnAvx::new(inner_fft);

        check_fft_algorithm(&fft, inner_len * 4, inverse);
    }

     #[test]
    fn test_mixed_4x4_avx() {
    	let forward = MixedRadix4x4Avx::new(false);
    	check_fft_algorithm(&forward, 16, false);

    	let inverse = MixedRadix4x4Avx::new(true);
    	check_fft_algorithm(&inverse, 16, true);
    }
}
