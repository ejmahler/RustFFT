use std::sync::Arc;
use std::arch::x86_64::*;
use std::cmp::min;

use num_complex::Complex;
use num_traits::Zero;

use common::FFTnum;

use ::{Length, IsInverse, Fft};

use super::avx32_utils::{AvxComplexArrayf32, AvxComplexArrayMutf32};
use super::avx32_utils;

pub struct MixedRadix2xnAvx<T> {
    twiddles: Box<[__m256]>,
    inner_fft: Arc<Fft<T>>,
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}
impl MixedRadix2xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<Fft<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let half_len = inner_fft.len();
        let len = half_len * 2;

        assert_eq!(len % 2, 0, "MixedRadix2xnAvx requires its FFT length to be an even number. Got {}", len);

        let quotient = half_len / 4;
        let remainder = half_len % 4;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let twiddles: Vec<_> = (0..num_twiddle_columns).map(|x| {
            let chunk_size = if x == quotient { remainder } else { 4 };
            let mut twiddle_chunk = [Complex::zero(); 4];
            for i in 0..chunk_size {
                twiddle_chunk[i] = f32::generate_twiddle_factor(x*4 + i, len, inverse);
            }
            twiddle_chunk.load_complex_f32(0)
        }).collect();

        let inner_outofplace_scratch = inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = inner_fft.get_inplace_scratch_len();

        Self {
            twiddles: twiddles.into_boxed_slice(),
            inplace_scratch_len: len + inner_outofplace_scratch,
            outofplace_scratch_len: if inner_inplace_scratch > len { inner_inplace_scratch } else { 0 },
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f32>]) {
        let half_len = self.len() / 2;
        
        let chunk_count = half_len / 4;
        let remainder = half_len % 4;

        // process the column FFTs
        for i in 0..chunk_count {
        	let input0 = buffer.load_complex_f32(i*4); 
        	let input1 = buffer.load_complex_f32(i*4 + half_len);

        	let (output0, output1_pretwiddle) = avx32_utils::column_butterfly2_f32(input0, input1);
            buffer.store_complex_f32(i*4, output0);
            
        	let output1 = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(i), output1_pretwiddle);
        	buffer.store_complex_f32(i*4 + half_len, output1);
        }

        // process the remainder
        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            let input0 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + half_len);

            let (output0, output1_pretwiddle) = avx32_utils::column_butterfly2_f32(input0, input1);
            buffer.store_complex_remainder_f32(remainder_mask, output0, chunk_count*4);

            let output1 = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(chunk_count), output1_pretwiddle);
            buffer.store_complex_remainder_f32(remainder_mask, output1, chunk_count*4 + half_len);
        }
    }

    // Transpose the input (treated as a nx2 array) into the output (as a 2xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let half_len = self.len() / 2;
        
        let chunk_count = half_len / 4;
        let remainder = half_len % 4;

        // transpose the scratch as a nx2 array into the buffer as an 2xn array
        for i in 0..chunk_count {
            let input0 = input.load_complex_f32(i*4); 
            let input1 = input.load_complex_f32(i*4 + half_len);

            // We loaded data from 2 separate arrays. inteleave the two arrays
            let (transposed0, transposed1) = avx32_utils::interleave_evens_odds_f32(input0, input1);

            // store the interleaved array contiguously
            output.store_complex_f32(i*8, transposed0);
            output.store_complex_f32(i*8 + 4, transposed1);
        }

        // transpose the remainder
        if remainder > 0 {
            let load_remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            let input0 = input.load_complex_remainder_f32(load_remainder_mask, chunk_count*4); 
            let input1 = input.load_complex_remainder_f32(load_remainder_mask, chunk_count*4 + half_len);

            // We loaded data from 2 separate arrays. inteleave the two arrays
            let (transposed0, transposed1) = avx32_utils::interleave_evens_odds_f32(input0, input1);

            // store the interleaved array contiguously
            let store_remainder_mask_0 = avx32_utils::RemainderMask::new_f32(min(remainder * 2, 4));
            output.store_complex_remainder_f32(store_remainder_mask_0, transposed0, chunk_count*8);
            if remainder > 2 {
                let store_remainder_mask_1 = avx32_utils::RemainderMask::new_f32(remainder * 2 - 4);
                output.store_complex_remainder_f32(store_remainder_mask_1, transposed1, chunk_count*8 + 4);
            }
        }
    }

    #[inline]
    fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        // Perform the column FFTs
        // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.perform_column_butterflies(buffer) };

        // process the row FFTs
        let (scratch, inner_scratch) = scratch.split_at_mut(self.len());
        self.inner_fft.process_multi(buffer, scratch, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(scratch, buffer) };
    }

    #[inline]
    fn perform_fft_out_of_place_f32(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        // Perform the column FFTs
        // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't avaiable
        unsafe { self.perform_column_butterflies(input) };

        // process the row FFTs. If extra scratch was provided, pass it in. Otherwise, use the output.
        let inner_scratch = if scratch.len() > 0 { scratch } else { &mut output[..] };
        self.inner_fft.process_inplace_multi(input, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(input, output) };
    }
}
boilerplate_fft_simd_f32!(MixedRadix2xnAvx, 
    |this:&MixedRadix2xnAvx<_>| this.len,
    |this:&MixedRadix2xnAvx<_>| this.inplace_scratch_len,
    |this:&MixedRadix2xnAvx<_>| this.outofplace_scratch_len
);

pub struct MixedRadix4xnAvx<T> {
    twiddle_config: avx32_utils::Rotate90Config,
    twiddles: Box<[__m256]>,
    inner_fft: Arc<Fft<T>>,
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}
impl MixedRadix4xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<Fft<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let quarter_len = inner_fft.len();
        let len = quarter_len * 4;

        assert_eq!(len % 4, 0, "MixedRadix4xnAvx requires its FFT length to be a multiple of 4. Got {}", len);

        let quotient = quarter_len / 4;
        let remainder = quarter_len % 4;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 3);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { 4 };

        	for y in 1..4 {
                let mut twiddle_chunk = [Complex::zero(); 4];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x*4 + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        let inner_outofplace_scratch = inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = inner_fft.get_inplace_scratch_len();

        Self {
            twiddle_config: avx32_utils::Rotate90Config::get_from_inverse(inverse),
            twiddles: twiddles.into_boxed_slice(),
            inplace_scratch_len: len + inner_outofplace_scratch,
            outofplace_scratch_len: if inner_inplace_scratch > len { inner_inplace_scratch } else { 0 },
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f32>]) {
        let quarter_len = self.len() / 4;

        let chunk_count = quarter_len / 4;
        let remainder = quarter_len % 4;

        // process the column FFTs
        for i in 0..chunk_count {
        	let input0 = buffer.load_complex_f32(i*4); 
        	let input1 = buffer.load_complex_f32(i*4 + quarter_len);
        	let input2 = buffer.load_complex_f32(i*4 + quarter_len*2);
        	let input3 = buffer.load_complex_f32(i*4 + quarter_len*3);

        	let (output0, output1_pretwiddle, output2_pretwiddle, output3_pretwiddle) = avx32_utils::column_butterfly4_f32(input0, input1, input2, input3, self.twiddle_config);

            buffer.store_complex_f32(i*4, output0);
            let output1 = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(i*3), output1_pretwiddle);
            buffer.store_complex_f32(i*4 + quarter_len, output1);
            let output2 = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(i*3+1), output2_pretwiddle);
            buffer.store_complex_f32(i*4 + quarter_len*2, output2);
            let output3 = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(i*3+2), output3_pretwiddle);
        	buffer.store_complex_f32(i*4 + quarter_len*3, output3);
        }

        // process the remainder
        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            let input0 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len);
            let input2 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*2);
            let input3 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*3);

            let (output0, output1_pretwiddle, output2_pretwiddle, output3_pretwiddle) = avx32_utils::column_butterfly4_f32(input0, input1, input2, input3, self.twiddle_config);

            buffer.store_complex_remainder_f32(remainder_mask, output0, chunk_count*4);
            let output1 = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(chunk_count*3), output1_pretwiddle);
            buffer.store_complex_remainder_f32(remainder_mask, output1, chunk_count*4 + quarter_len);
            let output2 = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(chunk_count*3+1), output2_pretwiddle);
            buffer.store_complex_remainder_f32(remainder_mask, output2, chunk_count*4 + quarter_len*2);
            let output3 = avx32_utils::fma::complex_multiply_f32(*self.twiddles.get_unchecked(chunk_count*3+2), output3_pretwiddle);
            buffer.store_complex_remainder_f32(remainder_mask, output3, chunk_count*4 + quarter_len*3);
        }
    }

    // Transpose the input (treated as a nx4 array) into the output (as a 4xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        let quarter_len = self.len() / 4;

        let chunk_count = quarter_len / 4;
        let remainder = quarter_len % 4;

        // transpose the scratch as a nx4 array into the buffer as an 4xn array
        for i in 0..chunk_count {
            let input0 = input.load_complex_f32(i*4); 
            let input1 = input.load_complex_f32(i*4 + quarter_len);
            let input2 = input.load_complex_f32(i*4 + quarter_len*2);
            let input3 = input.load_complex_f32(i*4 + quarter_len*3);

            // Transpose the 8x4 array and scatter them
            let (transposed0, transposed1, transposed2, transposed3) = avx32_utils::transpose_4x4_f32(input0, input1, input2, input3);

            // store the first chunk directly back into 
            output.store_complex_f32(i*16, transposed0);
            output.store_complex_f32(i*16 + 4, transposed1);
            output.store_complex_f32(i*16 + 4*2, transposed2);
            output.store_complex_f32(i*16 + 4*3, transposed3);
        }

        // transpose the remainder
        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            let input0 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len);
            let input2 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*2);
            let input3 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*3);

            // Transpose the 8x4 array and scatter them
            let (transposed0, transposed1, transposed2, _transposed3) = avx32_utils::transpose_4x4_f32(input0, input1, input2, input3);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            output.store_complex_f32(chunk_count*16, transposed0);
            if remainder >= 2 {
                output.store_complex_f32(chunk_count*16 + 4, transposed1);
                if remainder >= 3 {
                    output.store_complex_f32(chunk_count*16 + 4*2, transposed2);
                    // the remainder will never be 4 - because if it was 4, it would have been handled in the loop above
                    // so we can just not use the final 2 values. thankfully, the optimizer is smart enough to never even generate the last 2 values
                }
            }
        }
    }

    #[inline]
    fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        // Perform the column FFTs
        // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.perform_column_butterflies(buffer) };

        // process the row FFTs
        let (scratch, inner_scratch) = scratch.split_at_mut(self.len());
        self.inner_fft.process_multi(buffer, scratch, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(scratch, buffer) };
    }

    #[inline]
    fn perform_fft_out_of_place_f32(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        // Perform the column FFTs
        // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't avaiable
        unsafe { self.perform_column_butterflies(input) };

        // process the row FFTs. If extra scratch was provided, pass it in. Otherwise, use the output.
        let inner_scratch = if scratch.len() > 0 { scratch } else { &mut output[..] };
        self.inner_fft.process_inplace_multi(input, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(input, output) };
    }
}
boilerplate_fft_simd_f32!(MixedRadix4xnAvx, 
    |this:&MixedRadix4xnAvx<_>| this.len,
    |this:&MixedRadix4xnAvx<_>| this.inplace_scratch_len,
    |this:&MixedRadix4xnAvx<_>| this.outofplace_scratch_len
);

pub struct MixedRadix8xnAvx<T> {
    twiddle_config: avx32_utils::Rotate90Config,
    twiddles_butterfly8: __m256,
    twiddles: Box<[__m256]>,
    inner_fft: Arc<Fft<T>>,
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}
impl MixedRadix8xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<Fft<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let eigth_len = inner_fft.len();
        let len = eigth_len * 8;

        assert_eq!(len % 8, 0, "MixedRadix8xnAvx requires its FFT length to be a multiple of 8. Got {}", len);

        let quotient = eigth_len / 4;
        let remainder = eigth_len % 4;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 7);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { 4 };

        	for y in 1..8 {
                let mut twiddle_chunk = [Complex::zero(); 4];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x*4 + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        let inner_outofplace_scratch = inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = inner_fft.get_inplace_scratch_len();

        Self {
        	twiddle_config: avx32_utils::Rotate90Config::get_from_inverse(inverse),
        	twiddles_butterfly8: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 8, inverse)),
            twiddles: twiddles.into_boxed_slice(),
            inplace_scratch_len: len + inner_outofplace_scratch,
            outofplace_scratch_len: if inner_inplace_scratch > len { inner_inplace_scratch } else { 0 },
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f32>]) {
        let eigth_len = self.len() / 8;

        let chunk_count = eigth_len / 4;
        let remainder = eigth_len % 4;

        // process the column FFTs
        for i in 0..chunk_count {
            // Load 4 columns at once
            let mut columns = [_mm256_setzero_ps(); 8];
            for n in 0..8 {
                columns[n] = buffer.load_complex_f32(i*4 + eigth_len*n);
            }

            // Perform 4 parallel butterfly 8's on the columns
        	let processed_columns = avx32_utils::fma::column_butterfly8_array_f32(columns, self.twiddles_butterfly8, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
            debug_assert!(self.twiddles.len() >= (i+1) * 7);
        	buffer.store_complex_f32(i * 4, processed_columns[0]);
            for n in 1..8 {
                let twiddle = *self.twiddles.get_unchecked(i*7 + n - 1);
                let output = avx32_utils::fma::complex_multiply_f32(twiddle,  processed_columns[n]);
        	    buffer.store_complex_f32(i*4 + eigth_len*n, output);
            }
        }

        // process the remainder, if there is a remainder to process
        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            // Load (up to) 4 columns at once, based on our remainder
            let mut columns = [_mm256_setzero_ps(); 8];
            for n in 0..8 {
                columns[n] = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*n);
            }

            // Perform (up to) 4 parallel butterfly 8's on the columns
        	let processed_columns = avx32_utils::fma::column_butterfly8_array_f32(columns, self.twiddles_butterfly8, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
        	debug_assert!(self.twiddles.len() >= (chunk_count+1) * 7);
            buffer.store_complex_remainder_f32(remainder_mask, processed_columns[0], chunk_count*4);
            for n in 1..8 {
                let twiddle = *self.twiddles.get_unchecked(chunk_count*7 + n - 1);
                let output = avx32_utils::fma::complex_multiply_f32(twiddle,  processed_columns[n]);
                buffer.store_complex_remainder_f32(remainder_mask, output, chunk_count*4 + eigth_len*n);
            }
        }
    }

    // Transpose the input (treated as a nx8 array) into the output (as a 8xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
        let eigth_len = self.len() / 8;

        let chunk_count = eigth_len / 4;
        let remainder = eigth_len % 4;

        // transpose the scratch as a nx8 array into the buffer as an 8xn array
        for i in 0..chunk_count {
            let input0 = input.load_complex_f32(i * 4); 
            let input1 = input.load_complex_f32(i * 4 + eigth_len);
            let input2 = input.load_complex_f32(i * 4 + eigth_len*2);
            let input3 = input.load_complex_f32(i * 4 + eigth_len*3);
            let input4 = input.load_complex_f32(i * 4 + eigth_len*4);
            let input5 = input.load_complex_f32(i * 4 + eigth_len*5);
            let input6 = input.load_complex_f32(i * 4 + eigth_len*6);
            let input7 = input.load_complex_f32(i * 4 + eigth_len*7);

            // Transpose the 8x4 array and scatter them
            let (transposed0, transposed1, transposed2, transposed3) = avx32_utils::transpose_4x4_f32(input0, input1, input2, input3);
            let (transposed4, transposed5, transposed6, transposed7) = avx32_utils::transpose_4x4_f32(input4, input5, input6, input7);

            // store the first chunk directly back into 
            output.store_complex_f32(i * 32, transposed0);
            output.store_complex_f32(i * 32 + 4, transposed4);
            output.store_complex_f32(i * 32 + 4*2, transposed1);
            output.store_complex_f32(i * 32 + 4*3, transposed5);
            output.store_complex_f32(i * 32 + 4*4, transposed2);
            output.store_complex_f32(i * 32 + 4*5, transposed6);
            output.store_complex_f32(i * 32 + 4*6, transposed3);
            output.store_complex_f32(i * 32 + 4*7, transposed7);
        }

        // transpose the remainder, if there is a remainder to process
        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            let input0 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len);
            let input2 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*2);
            let input3 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*3);
            let input4 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*4);
            let input5 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*5);
            let input6 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*6);
            let input7 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*7);

            // Transpose the 8x4 array and scatter them
            let (transposed0, transposed1, transposed2, _transposed3) = avx32_utils::transpose_4x4_f32(input0, input1, input2, input3);
            let (transposed4, transposed5, transposed6, _transposed7) = avx32_utils::transpose_4x4_f32(input4, input5, input6, input7);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            output.store_complex_f32(chunk_count*32, transposed0);
            output.store_complex_f32(chunk_count*32 + 4, transposed4);
            if remainder >= 2 {
                output.store_complex_f32(chunk_count*32 + 4*2, transposed1);
                output.store_complex_f32(chunk_count*32 + 4*3, transposed5);
                if remainder >= 3 {
                    output.store_complex_f32(chunk_count*32 + 4*4, transposed2);
                    output.store_complex_f32(chunk_count*32 + 4*5, transposed6);
                    // the remainder will never be 4 - because if it was 4, it would have been handled in the loop above
                    // so we can just not use the final 2 values. thankfully, the optimizer is smart enough to never even generate the last 2 values
                }
            }
        }
    }

    #[inline]
    fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        // Perform the column FFTs
        // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.perform_column_butterflies(buffer) };

        // process the row FFTs
        let (scratch, inner_scratch) = scratch.split_at_mut(self.len());
        self.inner_fft.process_multi(buffer, scratch, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(scratch, buffer) };
    }

    #[inline]
    fn perform_fft_out_of_place_f32(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        // Perform the column FFTs
        // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't avaiable
        unsafe { self.perform_column_butterflies(input) };

        // process the row FFTs. If extra scratch was provided, pass it in. Otherwise, use the output.
        let inner_scratch = if scratch.len() > 0 { scratch } else { &mut output[..] };
        self.inner_fft.process_inplace_multi(input, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(input, output) };
    }
}
boilerplate_fft_simd_f32!(MixedRadix8xnAvx, 
    |this:&MixedRadix8xnAvx<_>| this.len,
    |this:&MixedRadix8xnAvx<_>| this.inplace_scratch_len,
    |this:&MixedRadix8xnAvx<_>| this.outofplace_scratch_len
);

pub struct MixedRadix16xnAvx<T> {
    twiddle_config: avx32_utils::Rotate90Config,
    twiddles_butterfly16: [__m256; 6],
    twiddles: Box<[__m256]>,
    inner_fft: Arc<Fft<T>>,
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}
impl MixedRadix16xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<Fft<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let sixteenth_len = inner_fft.len();
        let len = sixteenth_len * 16;

        assert_eq!(len % 16, 0, "MixedRadix16xnAvx requires its FFT length to be a multiple of 16. Got {}", len);

        let quotient = sixteenth_len / 4;
        let remainder = sixteenth_len % 4;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 15);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { 4 };

        	for y in 1..16 {
                let mut twiddle_chunk = [Complex::zero(); 4];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x*4 + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        let inner_outofplace_scratch = inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = inner_fft.get_inplace_scratch_len();

        Self {
        	twiddle_config: avx32_utils::Rotate90Config::get_from_inverse(inverse),
        	twiddles_butterfly16: [
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 16, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(2, 16, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(3, 16, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(4, 16, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(6, 16, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(9, 16, inverse)),
            ],
            twiddles: twiddles.into_boxed_slice(),
            inplace_scratch_len: len + inner_outofplace_scratch,
            outofplace_scratch_len: if inner_inplace_scratch > len { inner_inplace_scratch } else { 0 },
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f32>]) {
        let sixteenth_len = self.len() / 16;

        let chunk_count = sixteenth_len / 4;
        let remainder = sixteenth_len % 4;

        // process the column FFTs
        for i in 0..chunk_count {
            // Load 4 columns at once
            let mut columns = [_mm256_setzero_ps(); 16];
            for n in 0..16 {
                columns[n] = buffer.load_complex_f32(i*4 + sixteenth_len*n);
            }

            // Perform 4 parallel butterfly 16's on the columns
        	let processed_columns = avx32_utils::fma::column_butterfly16_f32(columns, self.twiddles_butterfly16, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
            debug_assert!(self.twiddles.len() >= (i+1) * 15);
        	buffer.store_complex_f32(i * 4, processed_columns[0]);
            for n in 1..16 {
                let twiddle = *self.twiddles.get_unchecked(i*15 + n - 1);
                let output = avx32_utils::fma::complex_multiply_f32(twiddle,  processed_columns[n]);
        	    buffer.store_complex_f32(i * 4 + sixteenth_len * n, output);
            }
        }

        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            // Load (up to) 4 columns at once, based on our remainder
            let mut columns = [_mm256_setzero_ps(); 16];
            for n in 0..16 {
                columns[n] = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*n);
            }

            // Perform (up to) 4 parallel butterfly 16's on the columns
        	let processed_columns = avx32_utils::fma::column_butterfly16_f32(columns, self.twiddles_butterfly16, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
        	debug_assert!(self.twiddles.len() >= (chunk_count+1) * 15);
            buffer.store_complex_remainder_f32(remainder_mask, processed_columns[0], chunk_count*4);
            for n in 1..16 {
                let twiddle = *self.twiddles.get_unchecked(chunk_count*15 + n - 1);
                let output = avx32_utils::fma::complex_multiply_f32(twiddle,  processed_columns[n]);
                buffer.store_complex_remainder_f32(remainder_mask, output, chunk_count*4 + sixteenth_len*n);
            }
        }
    }

    // Transpose the input (treated as a nx16 array) into the output (as a 16xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let sixteenth_len = self.len() / 16;

        let chunk_count = sixteenth_len / 4;
        let remainder = sixteenth_len % 4;

        for i in 0..chunk_count {
            let input0  = input.load_complex_f32(i * 4); 
            let input1  = input.load_complex_f32(i * 4 + sixteenth_len);
            let input2  = input.load_complex_f32(i * 4 + sixteenth_len*2);
            let input3  = input.load_complex_f32(i * 4 + sixteenth_len*3);
            let input4  = input.load_complex_f32(i * 4 + sixteenth_len*4);
            let input5  = input.load_complex_f32(i * 4 + sixteenth_len*5);
            let input6  = input.load_complex_f32(i * 4 + sixteenth_len*6);
            let input7  = input.load_complex_f32(i * 4 + sixteenth_len*7);
            let input8  = input.load_complex_f32(i * 4 + sixteenth_len*8); 
            let input9  = input.load_complex_f32(i * 4 + sixteenth_len*9);
            let input10 = input.load_complex_f32(i * 4 + sixteenth_len*10);
            let input11 = input.load_complex_f32(i * 4 + sixteenth_len*11);

            // Transpose the 8x4 array and scatter them
            let (transposed0,  transposed1,  transposed2,  transposed3)  = avx32_utils::transpose_4x4_f32(input0, input1, input2, input3);
            output.store_complex_f32(i * 64, transposed0);
            let (transposed4,  transposed5,  transposed6,  transposed7)  = avx32_utils::transpose_4x4_f32(input4, input5, input6, input7);
            output.store_complex_f32(i * 64 + 4, transposed4);
            let (transposed8,  transposed9,  transposed10, transposed11) = avx32_utils::transpose_4x4_f32(input8, input9, input10, input11);
            output.store_complex_f32(i * 64 + 4*2, transposed8);
            let input12 = input.load_complex_f32(i * 4 + sixteenth_len*12);
            let input13 = input.load_complex_f32(i * 4 + sixteenth_len*13);
            let input14 = input.load_complex_f32(i * 4 + sixteenth_len*14);
            let input15 = input.load_complex_f32(i * 4 + sixteenth_len*15);
            let (transposed12, transposed13, transposed14, transposed15) = avx32_utils::transpose_4x4_f32(input12, input13, input14, input15);

            // store the first chunk directly back into 
            output.store_complex_f32(i * 64 + 4*3, transposed12);
            output.store_complex_f32(i * 64 + 4*4, transposed1);
            output.store_complex_f32(i * 64 + 4*5, transposed5);
            output.store_complex_f32(i * 64 + 4*6, transposed9);
            output.store_complex_f32(i * 64 + 4*7, transposed13);
            output.store_complex_f32(i * 64 + 4*8, transposed2);
            output.store_complex_f32(i * 64 + 4*9, transposed6);
            output.store_complex_f32(i * 64 + 4*10, transposed10);
            output.store_complex_f32(i * 64 + 4*11, transposed14);
            output.store_complex_f32(i * 64 + 4*12, transposed3);
            output.store_complex_f32(i * 64 + 4*13, transposed7);
            output.store_complex_f32(i * 64 + 4*14, transposed11);
            output.store_complex_f32(i * 64 + 4*15, transposed15);
        }

        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            let input0  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len);
            let input2  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*2);
            let input3  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*3);
            let input4  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*4);
            let input5  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*5);
            let input6  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*6);
            let input7  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*7);
            let input8  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*8); 
            let input9  = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*9);
            let input10 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*10);
            let input11 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*11);
            let input12 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*12);
            let input13 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*13);
            let input14 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*14);
            let input15 = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*15);

            // Transpose the 8x4 array and scatter them
            let (transposed0,  transposed1,  transposed2,  _transposed3)  = avx32_utils::transpose_4x4_f32(input0, input1, input2, input3);
            output.store_complex_f32(chunk_count*64, transposed0);
            let (transposed4,  transposed5,  transposed6,  _transposed7)  = avx32_utils::transpose_4x4_f32(input4, input5, input6, input7);
            output.store_complex_f32(chunk_count*64 + 4, transposed4);
            let (transposed8,  transposed9,  transposed10, _transposed11) = avx32_utils::transpose_4x4_f32(input8, input9, input10, input11);
            output.store_complex_f32(chunk_count*64 + 4*2, transposed8);
            let (transposed12, transposed13, transposed14, _transposed15) = avx32_utils::transpose_4x4_f32(input12, input13, input14, input15);
            output.store_complex_f32(chunk_count*64 + 4*3, transposed12);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            if remainder >= 2 {
                output.store_complex_f32(chunk_count*64 + 4*4, transposed1);
                output.store_complex_f32(chunk_count*64 + 4*5, transposed5);
                output.store_complex_f32(chunk_count*64 + 4*6, transposed9);
                output.store_complex_f32(chunk_count*64 + 4*7, transposed13);
                if remainder >= 3 {
                    output.store_complex_f32(chunk_count*64 + 4*8, transposed2);
                    output.store_complex_f32(chunk_count*64 + 4*9, transposed6);
                    output.store_complex_f32(chunk_count*64 + 4*10, transposed10);
                    output.store_complex_f32(chunk_count*64 + 4*11, transposed14);
                    // the remainder will never be 4 - because if it was 4, it would have been handled in the loop above
                    // so we can just not use the final 2 values. thankfully, the optimizer is smart enough to never even generate the last 2 values
                }
            }
        }
    }

    #[inline]
    fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        // Perform the column FFTs
        // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.perform_column_butterflies(buffer) };

        // process the row FFTs
        let (scratch, inner_scratch) = scratch.split_at_mut(self.len());
        self.inner_fft.process_multi(buffer, scratch, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(scratch, buffer) };
    }

    #[inline]
    fn perform_fft_out_of_place_f32(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        // Perform the column FFTs
        // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't avaiable
        unsafe { self.perform_column_butterflies(input) };

        // process the row FFTs. If extra scratch was provided, pass it in. Otherwise, use the output.
        let inner_scratch = if scratch.len() > 0 { scratch } else { &mut output[..] };
        self.inner_fft.process_inplace_multi(input, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(input, output) };
    }
}
boilerplate_fft_simd_f32!(MixedRadix16xnAvx, 
    |this:&MixedRadix16xnAvx<_>| this.len,
    |this:&MixedRadix16xnAvx<_>| this.inplace_scratch_len,
    |this:&MixedRadix16xnAvx<_>| this.outofplace_scratch_len
);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_fft_algorithm;
    use std::sync::Arc;
    use algorithm::*;

    #[test]
    fn test_mixedradix_2xn_avx() {
        for pow in 2..8 {
            for remainder in 0..4 {
                let len = (1 << pow) + 2 * remainder;
                test_mixedradix_2xn_avx_with_length(len, false);
                test_mixedradix_2xn_avx_with_length(len, true);
            }
        }
    }

    fn test_mixedradix_2xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 2, inverse)) as Arc<dyn Fft<f32>>;
        let fft = MixedRadix2xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_mixedradix_4xn_avx() {
        for pow in 2..8 {
            for remainder in 0..4 {
                let len = (1 << pow) + 4 * remainder;
                test_mixedradix_4xn_avx_with_length(len, false);
                test_mixedradix_4xn_avx_with_length(len, true);
            }
        }
    }

    fn test_mixedradix_4xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 4, inverse)) as Arc<dyn Fft<f32>>;
        let fft = MixedRadix4xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_mixedradix_8xn_avx() {
        for pow in 3..9 {
            for remainder in 0..4 {
                let len = (1 << pow) + remainder * 8;
                test_mixedradix_8xn_avx_with_length(len, false);
                test_mixedradix_8xn_avx_with_length(len, true);
            }
        }
    }

    fn test_mixedradix_8xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 8, inverse)) as Arc<dyn Fft<f32>>;
        let fft = MixedRadix8xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_mixedradix_16xn_avx() {
        for pow in 4..10 {
            for remainder in 0..4 {
                let len = (1 << pow) + remainder * 16;
                test_mixedradix_16xn_avx_with_length(len, false);
                test_mixedradix_16xn_avx_with_length(len, true);
            }
        }
    }

    fn test_mixedradix_16xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 16, inverse)) as Arc<dyn Fft<f32>>;
        let fft = MixedRadix16xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_fft_algorithm(&fft, len, inverse);
    }
}