use std::sync::Arc;
use std::arch::x86_64::*;
use std::cmp::min;

use num_complex::Complex;
use num_traits::Zero;

use common::FFTnum;

use ::{Length, IsInverse, Fft};

use super::avx32_utils::{AvxComplexArrayf32, AvxComplexArrayMutf32};
use super::avx32_utils;
use super::avx64_utils::{AvxComplexArray64, AvxComplexArrayMut64};
use super::avx64_utils;

macro_rules! mixed_radix_boilerplate_f32{ () => (
    #[allow(unused)]
    const CHUNK_SIZE : usize = 4;
    
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new_f32(inner_fft: Arc<Fft<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
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
)}
macro_rules! mixed_radix_boilerplate_f64{() => (
    #[allow(unused)]
    const CHUNK_SIZE : usize = 2;

    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new_f64(inner_fft: Arc<Fft<f64>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we just checked that it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[inline]
    fn perform_fft_inplace_f64(&self, buffer: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
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
    fn perform_fft_out_of_place_f64(&self, input: &mut [Complex<f64>], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
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
)}

pub struct MixedRadix2xnAvx<T, V> {
    twiddles: Box<[V]>,
    inner_fft: Arc<Fft<T>>,
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}
boilerplate_fft_simd!(MixedRadix2xnAvx, |this:&MixedRadix2xnAvx<_,_>| this.len);

impl MixedRadix2xnAvx<f32, __m256> {
    mixed_radix_boilerplate_f32!();
    
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
}
impl MixedRadix2xnAvx<f64, __m256d> {
    mixed_radix_boilerplate_f64!();
    
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f64>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let half_len = inner_fft.len();
        let len = half_len * 2;

        assert_eq!(len % 2, 0, "MixedRadix2xnAvx requires its FFT length to be an even number. Got {}", len);

        let quotient = half_len / 2;
        let remainder = half_len % 2;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let twiddles: Vec<_> = (0..num_twiddle_columns).map(|x| {
            let chunk_size = if x == quotient { remainder } else { 2 };
            let mut twiddle_chunk = [Complex::zero(); 2];
            for i in 0..chunk_size {
                twiddle_chunk[i] = f64::generate_twiddle_factor(x*2 + i, len, inverse);
            }
            twiddle_chunk.load_complex_f64(0)
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
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f64>]) {
        let half_len = self.len() / 2;
        
        let chunk_count = half_len / 2;
        let remainder = half_len % 2;

        // process the column FFTs
        for i in 0..chunk_count {
        	let input0 = buffer.load_complex_f64(i*2); 
        	let input1 = buffer.load_complex_f64(i*2 + half_len);

        	let (output0, output1_pretwiddle) = avx64_utils::column_butterfly2_f64(input0, input1);
            buffer.store_complex_f64(output0, i*2);
            
        	let output1 = avx64_utils::fma::complex_multiply_f64(*self.twiddles.get_unchecked(i), output1_pretwiddle);
        	buffer.store_complex_f64(output1, i*2 + half_len);
        }

        // process the remainder
        if remainder > 0 {
            let input0 = buffer.load_complex_f64_lo(chunk_count*2); 
            let input1 = buffer.load_complex_f64_lo(chunk_count*2 + half_len);

            let (output0, output1_pretwiddle) = avx64_utils::column_butterfly2_f64_lo(input0, input1);
            buffer.store_complex_f64_lo(output0, chunk_count*2);

            let output1 = avx64_utils::fma::complex_multiply_f64_lo(_mm256_castpd256_pd128(*self.twiddles.get_unchecked(chunk_count)), output1_pretwiddle);
            buffer.store_complex_f64_lo(output1, chunk_count*2 + half_len);
        }
    }

    // Transpose the input (treated as a nx2 array) into the output (as a 2xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let half_len = self.len() / 2;
        
        let chunk_count = half_len / 2;
        let remainder = half_len % 2;

        // transpose the scratch as a nx2 array into the buffer as an 2xn array
        for i in 0..chunk_count {
            let input0 = input.load_complex_f64(i*2); 
            let input1 = input.load_complex_f64(i*2 + half_len);

            // We loaded data from 2 separate arrays. inteleave the two arrays
            let transposed = avx64_utils::transpose_2x2_f64([input0, input1]);

            // store the interleaved array contiguously
            output.store_complex_f64(transposed[0], i*4);
            output.store_complex_f64(transposed[1], i*4 + 2);
        }

        // transpose the remainder
        if remainder > 0 {
            // since we only have a single column, we don't need to do any transposing, just copying
            let input0 = input.load_complex_f64_lo(chunk_count*2); 
            let input1 = input.load_complex_f64_lo(chunk_count*2 + half_len);

            // store the interleaved array contiguously
            output.store_complex_f64_lo(input0, chunk_count*4);
            output.store_complex_f64_lo(input1, chunk_count*4 + 1);
        }
    }
}

















pub struct MixedRadix4xnAvx<T, V> {
    twiddle_config: avx32_utils::Rotate90Config<V>,
    twiddles: Box<[V]>,
    inner_fft: Arc<Fft<T>>,
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}
boilerplate_fft_simd!(MixedRadix4xnAvx, |this:&MixedRadix4xnAvx<_,_>| this.len);

impl MixedRadix4xnAvx<f32, __m256> {
    mixed_radix_boilerplate_f32!();

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
            twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
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
            let mut rows = [_mm256_setzero_ps(); 4];
            for n in 0..4 {
                rows[n] = buffer.load_complex_f32(i*4 + quarter_len*n);
            }

        	let processed_rows = avx32_utils::column_butterfly4_f32(rows, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
            debug_assert!(self.twiddles.len() >= (i+1) * 3);
        	buffer.store_complex_f32(i*4, processed_rows[0]);
            for n in 1..4 {
                let twiddle = *self.twiddles.get_unchecked(i*3 + n - 1);
                let output = avx32_utils::fma::complex_multiply_f32(twiddle,  processed_rows[n]);
        	    buffer.store_complex_f32(i*4 + quarter_len*n, output);
            }
        }

        // process the remainder
        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            // Load (up to) 4 columns at once, based on our remainder
            let mut rows = [_mm256_setzero_ps(); 4];
            for n in 0..4 {
                rows[n] = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*n);
            }

            // Perform (up to) 4 parallel butterfly 8's on the columns
        	let processed_rows = avx32_utils::column_butterfly4_f32(rows, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
        	debug_assert!(self.twiddles.len() >= (chunk_count+1) * 3);
            buffer.store_complex_remainder_f32(remainder_mask, processed_rows[0], chunk_count*4);
            for n in 1..4 {
                let twiddle = *self.twiddles.get_unchecked(chunk_count*3 + n - 1);
                let output = avx32_utils::fma::complex_multiply_f32(twiddle,  processed_rows[n]);
                buffer.store_complex_remainder_f32(remainder_mask, output, chunk_count*4 + quarter_len*n);
            }
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
            // Load 4 columns at once, giving us a 4x4 array
            let mut rows = [_mm256_setzero_ps(); 4];
            for n in 0..4 {
                rows[n] = input.load_complex_f32(i*4 + quarter_len*n);
            }

            // Transpose the 4x4 array
            let transposed = avx32_utils::transpose_4x4_f32(rows);

            // store each row of our transposed array contiguously. Manually unroll this loop because as of nightly april 8 2020,
            // it generates horrible code wherei t dumps every ymm register to the stack and then immediatelyreloads it
            output.store_complex_f32(i*16, transposed[0]);
            output.store_complex_f32(i*16 + 4, transposed[1]);
            output.store_complex_f32(i*16 + 2*4, transposed[2]);
            output.store_complex_f32(i*16 + 3*4, transposed[3]);
        }

        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            // Load (up to) 4 columns at once, giving us a 4x48 array
            let mut rows = [_mm256_setzero_ps(); 4];
            for n in 0..4 {
                rows[n] = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*n);
            }

            // Transpose the 4x4 array
            let transposed = avx32_utils::transpose_4x4_f32(rows);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            for n in 0..remainder {
                output.store_complex_f32(chunk_count*16 + n*4, transposed[n]);
            }
        }
    }
}
impl MixedRadix4xnAvx<f64, __m256d> {
    mixed_radix_boilerplate_f64!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f64>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let quarter_len = inner_fft.len();
        let len = quarter_len * 4;

        assert_eq!(len % 4, 0, "MixedRadix4xnAvx requires its FFT length to be a multiple of 4. Got {}", len);

        let quotient = quarter_len / Self::CHUNK_SIZE;
        let remainder = quarter_len % Self::CHUNK_SIZE;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 3);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { Self::CHUNK_SIZE };

        	for y in 1..4 {
                let mut twiddle_chunk = [Complex::zero(); 4];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f64::generate_twiddle_factor(y*(x*Self::CHUNK_SIZE + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f64(0));
        	}
        }

        let inner_outofplace_scratch = inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = inner_fft.get_inplace_scratch_len();

        Self {
            twiddle_config: avx32_utils::Rotate90Config::new_f64(inverse),
            twiddles: twiddles.into_boxed_slice(),
            inplace_scratch_len: len + inner_outofplace_scratch,
            outofplace_scratch_len: if inner_inplace_scratch > len { inner_inplace_scratch } else { 0 },
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f64>]) {
        let quarter_len = self.len() / 4;

        let chunk_count = quarter_len / Self::CHUNK_SIZE;
        let remainder = quarter_len % Self::CHUNK_SIZE;

        // process the column FFTs
        for i in 0..chunk_count {
            let mut rows = [_mm256_setzero_pd(); 4];
            for n in 0..4 {
                rows[n] = buffer.load_complex_f64(i*Self::CHUNK_SIZE + quarter_len*n);
            }

        	let processed_rows = avx64_utils::column_butterfly4_f64(rows, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
            debug_assert!(self.twiddles.len() >= (i+1) * 3);
        	buffer.store_complex_f64(processed_rows[0], i*Self::CHUNK_SIZE);
            for n in 1..4 {
                let twiddle = *self.twiddles.get_unchecked(i*3 + n - 1);
                let output = avx64_utils::fma::complex_multiply_f64(twiddle,  processed_rows[n]);
        	    buffer.store_complex_f64(output, i*Self::CHUNK_SIZE + quarter_len*n);
            }
        }

        // process the remainder
        if remainder > 0 {
            // Load (up to) 4 columns at once, based on our remainder
            let mut rows = [_mm_setzero_pd(); 4];
            for n in 0..4 {
                rows[n] = buffer.load_complex_f64_lo(chunk_count*Self::CHUNK_SIZE + quarter_len*n);
            }

            // Perform (up to) 4 parallel butterfly 8's on the columns
        	let processed_rows = avx64_utils::column_butterfly4_f64_lo(rows, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
        	debug_assert!(self.twiddles.len() >= (chunk_count+1) * 3);
            buffer.store_complex_f64_lo(processed_rows[0], chunk_count*Self::CHUNK_SIZE);
            for n in 1..4 {
                let twiddle = _mm256_castpd256_pd128(*self.twiddles.get_unchecked(chunk_count*3 + n - 1));
                let output = avx64_utils::fma::complex_multiply_f64_lo(twiddle,  processed_rows[n]);
                buffer.store_complex_f64_lo(output, chunk_count*Self::CHUNK_SIZE + quarter_len*n);
            }
        }
    }

    // Transpose the input (treated as a nx4 array) into the output (as a 4xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let quarter_len = self.len() / 4;

        let chunk_count = quarter_len / Self::CHUNK_SIZE;
        let remainder = quarter_len % Self::CHUNK_SIZE;

        // transpose the scratch as a nx4 array into the buffer as an 4xn array
        for i in 0..chunk_count {
            // Load 4 columns at once, giving us a 4x4 array
            let mut rows = [_mm256_setzero_pd(); 4];
            for n in 0..4 {
                rows[n] = input.load_complex_f64(i*Self::CHUNK_SIZE + quarter_len*n);
            }

            // Transpose the 4x4 array
            let (transposed0, transposed1) = avx64_utils::transpose_2x4_to_4x2_f64(rows);

            // store each row of our transposed array contiguously. Manually unroll this loop because as of nightly april 8 2020,
            // it generates horrible code wherei t dumps every ymm register to the stack and then immediatelyreloads it
            output.store_complex_f64(transposed0[0], i*8);
            output.store_complex_f64(transposed1[0], i*8 + Self::CHUNK_SIZE);
            output.store_complex_f64(transposed0[1], i*8 + Self::CHUNK_SIZE*2);
            output.store_complex_f64(transposed1[1], i*8 + Self::CHUNK_SIZE*3);
        }

        if remainder > 0 {
            // since we only have a single column, we don't need to do any transposing, just copying
            for n in 0..4 {
                let row = input.load_complex_f64_lo(chunk_count*Self::CHUNK_SIZE + quarter_len*n);
                output.store_complex_f64_lo(row, chunk_count*8 + n);
            }
        }
    }
}










pub struct MixedRadix8xnAvx<T, V> {
    twiddle_config: avx32_utils::Rotate90Config<V>,
    twiddles_butterfly8: V,
    twiddles: Box<[V]>,
    inner_fft: Arc<Fft<T>>,
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}
boilerplate_fft_simd!(MixedRadix8xnAvx, |this:&MixedRadix8xnAvx<_,_>| this.len);

impl MixedRadix8xnAvx<f32, __m256> {
    mixed_radix_boilerplate_f32!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let eigth_len = inner_fft.len();
        let len = eigth_len * 8;

        assert_eq!(len % 8, 0, "MixedRadix8xnAvx requires its FFT length to be a multiple of 8. Got {}", len);

        let quotient = eigth_len / Self::CHUNK_SIZE;
        let remainder = eigth_len % Self::CHUNK_SIZE;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 7);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { Self::CHUNK_SIZE };

        	for y in 1..8 {
                let mut twiddle_chunk = [Complex::zero(); Self::CHUNK_SIZE];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x*Self::CHUNK_SIZE + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        let inner_outofplace_scratch = inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = inner_fft.get_inplace_scratch_len();

        Self {
        	twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
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
        	let processed_columns = avx32_utils::fma::column_butterfly8_f32(columns, self.twiddles_butterfly8, self.twiddle_config);

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
        	let processed_columns = avx32_utils::fma::column_butterfly8_f32(columns, self.twiddles_butterfly8, self.twiddle_config);

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
            // Load 4 columns at once, giving us a 4x8 array
            let mut rows = [_mm256_setzero_ps(); 8];
            for n in 0..8 {
                rows[n] = input.load_complex_f32(i*4 + eigth_len*n);
            }

            // Transpose the 4x8 array to a 8x4 array
            let (chunk0, chunk1) = avx32_utils::transpose_4x8_to_8x4_f32(rows);

            // store each row of our transposed array contiguously
            for n in 0..4 {
                output.store_complex_f32(i*32 + n*8, chunk0[n]);
                output.store_complex_f32(i*32 + n*8 + 4, chunk1[n]);
            }
        }

        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            // Load (up to) 4 columns at once, giving us a 4x8 array
            let mut rows = [_mm256_setzero_ps(); 8];
            for n in 0..8 {
                rows[n] = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*n);
            }

            // Transpose the 4x8 array to a 8x4 array
            let (chunk0, chunk1) = avx32_utils::transpose_4x8_to_8x4_f32(rows);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            for n in 0..remainder {
                output.store_complex_f32(chunk_count*32 + n*8, chunk0[n]);
                output.store_complex_f32(chunk_count*32 + n*8 + 4, chunk1[n]);
            }
        }
    }
}
impl MixedRadix8xnAvx<f64, __m256d> {
    mixed_radix_boilerplate_f64!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f64>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let eigth_len = inner_fft.len();
        let len = eigth_len * 8;

        assert_eq!(len % 8, 0, "MixedRadix8xnAvx requires its FFT length to be a multiple of 8. Got {}", len);

        let quotient = eigth_len / Self::CHUNK_SIZE;
        let remainder = eigth_len % Self::CHUNK_SIZE;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 7);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { Self::CHUNK_SIZE };

        	for y in 1..8 {
                let mut twiddle_chunk = [Complex::zero(); Self::CHUNK_SIZE];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f64::generate_twiddle_factor(y*(x*Self::CHUNK_SIZE + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f64(0));
        	}
        }

        let inner_outofplace_scratch = inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = inner_fft.get_inplace_scratch_len();

        Self {
        	twiddle_config: avx32_utils::Rotate90Config::new_f64(inverse),
        	twiddles_butterfly8: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 8, inverse)),
            twiddles: twiddles.into_boxed_slice(),
            inplace_scratch_len: len + inner_outofplace_scratch,
            outofplace_scratch_len: if inner_inplace_scratch > len { inner_inplace_scratch } else { 0 },
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f64>]) {
        let eigth_len = self.len() / 8;

        let chunk_count = eigth_len / 2;
        let remainder = eigth_len % 2;

        // process the column FFTs
        for i in 0..chunk_count {
            // Load 4 columns at once
            let mut columns = [_mm256_setzero_pd(); 8];
            for n in 0..8 {
                columns[n] = buffer.load_complex_f64(i*2 + eigth_len*n);
            }

            // Perform 4 parallel butterfly 8's on the columns
        	let processed_columns = avx64_utils::fma::column_butterfly8_f64(columns, self.twiddles_butterfly8, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
            debug_assert!(self.twiddles.len() >= (i+1) * 7);
        	buffer.store_complex_f64(processed_columns[0], i*2);
            for n in 1..8 {
                let twiddle = *self.twiddles.get_unchecked(i*7 + n - 1);
                let output = avx64_utils::fma::complex_multiply_f64(twiddle,  processed_columns[n]);
        	    buffer.store_complex_f64(output, i*2 + eigth_len*n);
            }
        }

        // process the remainder, if there is a remainder to process
        if remainder > 0 {
            // Load 1 column instead of the usual 2, based on our remainder
            let mut columns = [_mm_setzero_pd(); 8];
            for n in 0..8 {
                columns[n] = buffer.load_complex_f64_lo(chunk_count*2 + eigth_len*n);
            }

            // Perform (up to) 4 parallel butterfly 8's on the columns
        	let processed_columns = avx64_utils::fma::column_butterfly8_f64_lo(columns, _mm256_castpd256_pd128(self.twiddles_butterfly8), self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
        	debug_assert!(self.twiddles.len() >= (chunk_count+1) * 7);
            buffer.store_complex_f64_lo(processed_columns[0], chunk_count*2);
            for n in 1..8 {
                let twiddle = _mm256_castpd256_pd128(*self.twiddles.get_unchecked(chunk_count*7 + n - 1));
                let output = avx64_utils::fma::complex_multiply_f64_lo(twiddle,  processed_columns[n]);
                buffer.store_complex_f64_lo(output, chunk_count*2 + eigth_len*n);
            }
        }
    }

    // Transpose the input (treated as a nx8 array) into the output (as a 8xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &mut [Complex<f64>], output: &mut [Complex<f64>]) {
        let eigth_len = self.len() / 8;

        let chunk_count = eigth_len / 2;
        let remainder = eigth_len % 2;

        // transpose the scratch as a nx8 array into the buffer as an 8xn array
        for i in 0..chunk_count {
            // Load 2 columns at once, giving us a 2x8 array
            let mut rows = [_mm256_setzero_pd(); 8];
            for n in 0..8 {
                rows[n] = input.load_complex_f64(i*2 + eigth_len*n);
            }

            // Transpose the 2x8 array to a 8x2 array
            let transposed = avx64_utils::transpose_2x8_to_8x2_packed_f64(rows);

            // store each row of our transposed array contiguously
            // manually unroll
            output.store_complex_f64(transposed[0], i*16);
            output.store_complex_f64(transposed[1], i*16 + 2);
            output.store_complex_f64(transposed[2], i*16 + 4);
            output.store_complex_f64(transposed[3], i*16 + 6);
            output.store_complex_f64(transposed[4], i*16 + 8);
            output.store_complex_f64(transposed[5], i*16 + 10);
            output.store_complex_f64(transposed[6], i*16 + 12);
            output.store_complex_f64(transposed[7], i*16 + 14);
        }

        if remainder > 0 {
            // since we only have a single column, we don't need to do any transposing, just copying
            for n in 0..8 {
                let row = input.load_complex_f64_lo(chunk_count*2 + eigth_len*n);
                output.store_complex_f64_lo(row, chunk_count*16 + n);
            }
        }
    }
}






pub struct MixedRadix16xnAvx<T, V> {
    twiddle_config: avx32_utils::Rotate90Config<V>,
    twiddles_butterfly16: [V; 6],
    twiddles: Box<[V]>,
    inner_fft: Arc<Fft<T>>,
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}
boilerplate_fft_simd!(MixedRadix16xnAvx, |this:&MixedRadix16xnAvx<_,_>| this.len);

impl MixedRadix16xnAvx<f32, __m256> {
    mixed_radix_boilerplate_f32!();

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
        	twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
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
            // Load 4 columns at once, giving us a 4x16 array
            let mut rows = [_mm256_setzero_ps(); 16];
            for n in 0..16 {
                rows[n] = input.load_complex_f32(i*4 + sixteenth_len*n);
            }

            // Transpose the 4x16 array to a 16x4 array
            let (chunk0, chunk1, chunk2, chunk3) = avx32_utils::transpose_4x16_to_16x4_f32(rows);

            // store each row of our transposed array contiguously
            for n in 0..4 {
                output.store_complex_f32(i*64 + n*16, chunk0[n]);
                output.store_complex_f32(i*64 + n*16 + 4, chunk1[n]);
                output.store_complex_f32(i*64 + n*16 + 8, chunk2[n]);
                output.store_complex_f32(i*64 + n*16 + 12, chunk3[n]);
            }
        }

        if remainder > 0 {
            let remainder_mask = avx32_utils::RemainderMask::new_f32(remainder);

            // Load (up to) 4 columns at once, giving us a 4x16 array
            let mut rows = [_mm256_setzero_ps(); 16];
            for n in 0..16 {
                rows[n] = input.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*n);
            }

            // Transpose the 4x16 array to a 16x4 array
            let (chunk0, chunk1, chunk2, chunk3) = avx32_utils::transpose_4x16_to_16x4_f32(rows);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            for n in 0..remainder {
                output.store_complex_f32(chunk_count*64 + n*16, chunk0[n]);
                output.store_complex_f32(chunk_count*64 + n*16 + 4, chunk1[n]);
                output.store_complex_f32(chunk_count*64 + n*16 + 8, chunk2[n]);
                output.store_complex_f32(chunk_count*64 + n*16 + 12, chunk3[n]);
            }
        }
    }
}
impl MixedRadix16xnAvx<f64, __m256d> {
    mixed_radix_boilerplate_f64!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<Fft<f64>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let sixteenth_len = inner_fft.len();
        let len = sixteenth_len * 16;

        assert_eq!(len % 16, 0, "MixedRadix16xnAvx requires its FFT length to be a multiple of 16. Got {}", len);

        let quotient = sixteenth_len / 2;
        let remainder = sixteenth_len % 2;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 15);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { 2 };

        	for y in 1..16 {
                let mut twiddle_chunk = [Complex::zero(); 2];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f64::generate_twiddle_factor(y*(x*2 + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f64(0));
        	}
        }

        let inner_outofplace_scratch = inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = inner_fft.get_inplace_scratch_len();

        Self {
        	twiddle_config: avx32_utils::Rotate90Config::new_f64(inverse),
        	twiddles_butterfly16: [
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 16, inverse)),
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(2, 16, inverse)),
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(3, 16, inverse)),
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(4, 16, inverse)),
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(6, 16, inverse)),
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(9, 16, inverse)),
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
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f64>]) {
        let sixteenth_len = self.len() / 16;

        let chunk_count = sixteenth_len / 2;
        let remainder = sixteenth_len % 2;

        // process the column FFTs
        for i in 0..chunk_count {
            // Load 4 columns at once
            let mut columns = [_mm256_setzero_pd(); 16];
            for n in 0..16 {
                columns[n] = buffer.load_complex_f64(i*2 + sixteenth_len*n);
            }

            // Perform 4 parallel butterfly 16's on the columns
        	let processed_columns = avx64_utils::fma::column_butterfly16_f64(columns, self.twiddles_butterfly16, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
            debug_assert!(self.twiddles.len() >= (i+1) * 15);
        	buffer.store_complex_f64(processed_columns[0], i*2);
            for n in 1..16 {
                let twiddle = *self.twiddles.get_unchecked(i*15 + n - 1);
                let output = avx64_utils::fma::complex_multiply_f64(twiddle,  processed_columns[n]);
        	    buffer.store_complex_f64(output, i*2 + sixteenth_len*n);
            }
        }

        if remainder > 0 {
            // Load half a column of data
            let mut columns = [_mm256_setzero_pd(); 16];
            for n in 0..16 {
                columns[n] = _mm256_zextpd128_pd256(buffer.load_complex_f64_lo(chunk_count*2 + sixteenth_len*n));
            }

            // Perform (up to) 4 parallel butterfly 16's on the columns
        	let processed_columns = avx64_utils::fma::column_butterfly16_f64(columns, self.twiddles_butterfly16, self.twiddle_config);

            // Apply twiddle factors to the column and store them where they came from
        	debug_assert!(self.twiddles.len() >= (chunk_count+1) * 15);
            buffer.store_complex_f64_lo(_mm256_castpd256_pd128(processed_columns[0]), chunk_count*2);
            for n in 1..16 {
                let twiddle = _mm256_castpd256_pd128(*self.twiddles.get_unchecked(chunk_count*15 + n - 1));
                let output = avx64_utils::fma::complex_multiply_f64_lo(twiddle,  _mm256_castpd256_pd128(processed_columns[n]));
                buffer.store_complex_f64_lo(output, chunk_count*2 + sixteenth_len*n);
            }
        }
    }

    // Transpose the input (treated as a nx16 array) into the output (as a 16xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let sixteenth_len = self.len() / 16;

        let chunk_count = sixteenth_len / 2;
        let remainder = sixteenth_len % 2;

        for i in 0..chunk_count {
            // Load 2 columns at once, giving us a 2x16 array
            let mut rows = [_mm256_setzero_pd(); 16];
            for n in 0..16 {
                rows[n] = input.load_complex_f64(i*2 + sixteenth_len*n);
            }

            // Transpose the 2x16 array to a 16x2 array
            let transposed = avx64_utils::transpose_2x16_to_16x2_packed_f64(rows);

            // store each row of our transposed array contiguously
            // Manually unroll this loop because otherwise rustc generates slow code :(
            // https://github.com/rust-lang/rust/issues/71025
            output.store_complex_f64(transposed[0], i*32 + 0);
            output.store_complex_f64(transposed[1], i*32 + 2);
            output.store_complex_f64(transposed[2], i*32 + 4);
            output.store_complex_f64(transposed[3], i*32 + 6);
            output.store_complex_f64(transposed[4], i*32 + 8);
            output.store_complex_f64(transposed[5], i*32 + 10);
            output.store_complex_f64(transposed[6], i*32 + 12);
            output.store_complex_f64(transposed[7], i*32 + 14);
            output.store_complex_f64(transposed[8], i*32 + 16);
            output.store_complex_f64(transposed[9], i*32 + 18);
            output.store_complex_f64(transposed[10], i*32 + 20);
            output.store_complex_f64(transposed[11], i*32 + 22);
            output.store_complex_f64(transposed[12], i*32 + 24);
            output.store_complex_f64(transposed[13], i*32 + 26);
            output.store_complex_f64(transposed[14], i*32 + 28);
            output.store_complex_f64(transposed[15], i*32 + 30);
        }

        if remainder > 0 {
            // since we only have a single column, we don't need to do any transposing, just copying
            for n in 0..16 {
                let row = input.load_complex_f64_lo(chunk_count*2 + n*sixteenth_len);
                output.store_complex_f64_lo(row, chunk_count*32 + n);
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_fft_algorithm;
    use std::sync::Arc;
    use algorithm::*;

    macro_rules! test_avx_mixed_radix {
        ($f32_test_name:ident, $f64_test_name:ident, $struct_name:ident, $pow_range:expr, $inner_len:expr) => (
            #[test]
            fn $f32_test_name() {
                for pow in $pow_range {
                    for remainder in 0..4 {
                        let len = (1 << pow) + $inner_len * remainder;

                        let inner_fft_forward = Arc::new(DFT::new(len / $inner_len, false)) as Arc<dyn Fft<f32>>;
                        let fft_forward = $struct_name::new_f32(inner_fft_forward).expect("Can't run test because this machine doesn't have the required instruction sets");
                        check_fft_algorithm(&fft_forward, len, false);

                        let inner_fft_inverse = Arc::new(DFT::new(len / $inner_len, true)) as Arc<dyn Fft<f32>>;
                        let fft_inverse = $struct_name::new_f32(inner_fft_inverse).expect("Can't run test because this machine doesn't have the required instruction sets");
                        check_fft_algorithm(&fft_inverse, len, true);
                    }
                }
            }
            #[test]
            fn $f64_test_name() {
                for pow in $pow_range {
                    for remainder in 0..2 {
                        let len = (1 << pow) + $inner_len * remainder;

                        let inner_fft_forward = Arc::new(DFT::new(len / $inner_len, false)) as Arc<dyn Fft<f64>>;
                        let fft_forward = $struct_name::new_f64(inner_fft_forward).expect("Can't run test because this machine doesn't have the required instruction sets");
                        check_fft_algorithm(&fft_forward, len, false);

                        let inner_fft_inverse = Arc::new(DFT::new(len / $inner_len, true)) as Arc<dyn Fft<f64>>;
                        let fft_inverse = $struct_name::new_f64(inner_fft_inverse).expect("Can't run test because this machine doesn't have the required instruction sets");
                        check_fft_algorithm(&fft_inverse, len, true);
                    }
                }
            }
        )
    }

    test_avx_mixed_radix!(test_mixedradix_2xn_avx_f32, test_mixedradix_2xn_avx_f64, MixedRadix2xnAvx, 1..5, 2);
    test_avx_mixed_radix!(test_mixedradix_4xn_avx_f32, test_mixedradix_4xn_avx_f64, MixedRadix4xnAvx, 2..6, 4);
    test_avx_mixed_radix!(test_mixedradix_8xn_avx_f32, test_mixedradix_8xn_avx_f64, MixedRadix8xnAvx, 3..7, 8);
    test_avx_mixed_radix!(test_mixedradix_16xn_avx_f32, test_mixedradix_16xn_avx_f64, MixedRadix16xnAvx, 4..8, 16);
}