use std::sync::Arc;
use std::arch::x86_64::*;

use num_complex::Complex;
use num_traits::Zero;

use crate::common::FFTnum;

use crate::{Length, IsInverse, Fft};

use super::avx32_utils::{AvxComplexArrayf32, AvxComplexArrayMutf32};
use super::avx32_utils;
use super::avx64_utils::{AvxComplexArray64, AvxComplexArrayMut64};
use super::avx64_utils;
use super::CommonSimdData;
use super::avx_vector::{AvxVector, AvxVector128, AvxVector256, Rotation90};

// Take the ceiling of dividing a by b
// Ie, if the inputs are a=3, b=5, the return will be 1. if the inputs are a=12 and b=5, the return will be 3
fn div_ceil(a: usize, b: usize) -> usize {
    a / b + if a % b == 0 { 0 } else { 1 }
}

macro_rules! mixedradix_boilerplate_f32{ () => (
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new_f32(inner_fft: Arc<dyn Fft<f32>>) -> Result<Self, ()> {
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
        self.common_data.inner_fft.process_multi(buffer, scratch, inner_scratch);

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
        self.common_data.inner_fft.process_inplace_multi(input, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(input, output) };
    }
)}
macro_rules! mixedradix_boilerplate_f64{() => (
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new_f64(inner_fft: Arc<dyn Fft<f64>>) -> Result<Self, ()> {
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
        self.common_data.inner_fft.process_multi(buffer, scratch, inner_scratch);

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
        self.common_data.inner_fft.process_inplace_multi(input, inner_scratch);

        // Transpose
        // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
        unsafe { self.transpose(input, output) };
    }
)}

macro_rules! mixedradix_gen_data_f32 {
    ($row_count: expr, $inner_fft:expr) => {{
        // Important constants
        const ROW_COUNT : usize = $row_count;
        const CHUNK_SIZE : usize = 4;
        const TWIDDLES_PER_COLUMN : usize = ROW_COUNT - 1;

        // derive some info from our inner FFT
        let inverse = $inner_fft.is_inverse();
        let len_per_row = $inner_fft.len();
        let len = len_per_row * ROW_COUNT;

        // We're going to process each row of the FFT one AVX register at a time. We need to know how many AVX registers each row can fit,
        // and if the last register in each row going to have partial data (ie a remainder)
        let quotient = len_per_row / CHUNK_SIZE;
        let remainder = len_per_row % CHUNK_SIZE;

        let num_twiddle_columns = quotient + div_ceil(remainder, CHUNK_SIZE);

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * TWIDDLES_PER_COLUMN);
        for x in 0..num_twiddle_columns {
            for y in 1..ROW_COUNT {
                let mut twiddle_chunk = [Complex::zero(); CHUNK_SIZE];
                for i in 0..CHUNK_SIZE {
                    twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x*CHUNK_SIZE + i), len, inverse);
                }

                twiddles.push(twiddle_chunk.load_complex_f32(0));
            }
        }

        let inner_outofplace_scratch = $inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = $inner_fft.get_inplace_scratch_len();

        CommonSimdData {
            twiddles: twiddles.into_boxed_slice(),
            inplace_scratch_len: len + inner_outofplace_scratch,
            outofplace_scratch_len: if inner_inplace_scratch > len { inner_inplace_scratch } else { 0 },
            inner_fft: $inner_fft,
            len,
            inverse,
        }
    }}
}


macro_rules! mixedradix_gen_data_f64 {
    ($row_count: expr, $inner_fft:expr) => {{
        // Important constants
        const ROW_COUNT : usize = $row_count;
        const CHUNK_SIZE : usize = 2;
        const TWIDDLES_PER_COLUMN : usize = ROW_COUNT - 1;

        // derive some info from our inner FFT
        let inverse = $inner_fft.is_inverse();
        let len_per_row = $inner_fft.len();
        let len = len_per_row * ROW_COUNT;

        // We're going to process each row of the FFT one AVX register at a time. We need to know how many AVX registers each row can fit,
        // and if the last register in each row going to have partial data (ie a remainder)
        let quotient = len_per_row / CHUNK_SIZE;
        let remainder = len_per_row % CHUNK_SIZE;

        let num_twiddle_columns = quotient + div_ceil(remainder, CHUNK_SIZE);

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * TWIDDLES_PER_COLUMN);
        for x in 0..num_twiddle_columns {
            for y in 1..ROW_COUNT {
                let mut twiddle_chunk = [Complex::zero(); CHUNK_SIZE];
                for i in 0..CHUNK_SIZE {
                    twiddle_chunk[i] = f64::generate_twiddle_factor(y*(x*CHUNK_SIZE + i), len, inverse);
                }

                twiddles.push(twiddle_chunk.load_complex_f64(0));
            }
        }

        let inner_outofplace_scratch = $inner_fft.get_out_of_place_scratch_len();
        let inner_inplace_scratch = $inner_fft.get_inplace_scratch_len();

        CommonSimdData {
            twiddles: twiddles.into_boxed_slice(),
            inplace_scratch_len: len + inner_outofplace_scratch,
            outofplace_scratch_len: if inner_inplace_scratch > len { inner_inplace_scratch } else { 0 },
            inner_fft: $inner_fft,
            len,
            inverse,
        }
    }}
}


macro_rules! mixedradix_column_butterflies_f32{
    ($row_count: expr, $butterfly_fn: expr, $butterfly_fn_lo: expr) => (

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f32>]) {
        // How many complex numbers fit in a single register
        const CHUNK_SIZE : usize = 4;

        // How many rows this FFT has, ie 2 for 2xn, 4 for 4xn, etc
        const ROW_COUNT : usize = $row_count;
        const TWIDDLES_PER_COLUMN : usize = ROW_COUNT - 1;

        let len_per_row = self.len() / ROW_COUNT;
        let chunk_count = len_per_row / CHUNK_SIZE;

        // process the column FFTs
        for (c, twiddle_chunk) in self.common_data.twiddles.chunks_exact(TWIDDLES_PER_COLUMN).take(chunk_count).enumerate() {
            let index_base = c*CHUNK_SIZE;

            // Load columns from the buffer into registers
            let mut columns = [_mm256_setzero_ps(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                columns[i] = buffer.load_complex_f32(index_base + len_per_row*i);
            }

            // apply our butterfly function down the columns
            let output = $butterfly_fn(columns, self);

            // always write the first row directly back without twiddles
            buffer.store_complex_f32(index_base, output[0]);
            
            // for every other row, apply twiddle factors and then write back to memory
            for i in 1..ROW_COUNT {
                let twiddle = twiddle_chunk[i - 1];
                let output = avx32_utils::fma::complex_multiply_f32(twiddle, output[i]);
                buffer.store_complex_f32(index_base + len_per_row*i, output);
            }
        }

        // finally, we might have a single partial chunk.
        // Normally, we can fit 4 complex numbers into an AVX register, but we only have `partial_remainder` columns left, so we need special logic to handle these final columns
        let partial_remainder = len_per_row % CHUNK_SIZE;
        if partial_remainder > 0 {
            let partial_remainder_base = chunk_count * CHUNK_SIZE;
            let partial_remainder_twiddle_base = self.common_data.twiddles.len() - TWIDDLES_PER_COLUMN;
            let final_twiddle_chunk = &self.common_data.twiddles[partial_remainder_twiddle_base..];

            let remainder_mask = avx32_utils::RemainderMask::new_f32(partial_remainder);

            if partial_remainder > 2 {
                // Load partial columns for our final remainder, using RemainderMask to limit which memory locations we load
                let mut columns = [_mm256_setzero_ps(); ROW_COUNT];
                for i in 0..ROW_COUNT {
                    columns[i] = buffer.load_complex_remainder_f32(remainder_mask, partial_remainder_base + len_per_row*i);
                }

                // apply our butterfly function down the columns
                let output = $butterfly_fn(columns, self);

                // always write the first row without twiddles
                buffer.store_complex_remainder_f32(remainder_mask, output[0], partial_remainder_base);

                // for every other row, apply twiddle factors and then write back to memory, using RemainderMask to limit which memory locations to write to
                for i in 1..ROW_COUNT {
                    let twiddle = final_twiddle_chunk[i - 1];
                    let output = avx32_utils::fma::complex_multiply_f32(twiddle, output[i]);
                    buffer.store_complex_remainder_f32(remainder_mask, output, partial_remainder_base + len_per_row*i);
                }
            } else {
                // Load partial columns for our final remainder, using RemainderMask to limit which memory locations we load
                // our remainder will be only half-filled, so we can eke out a little speed by only doing a half-column
                let mut columns = [_mm_setzero_ps(); ROW_COUNT];
                for i in 0..ROW_COUNT {
                    columns[i] = buffer.load_complex_remainder_f32_lo(remainder_mask, partial_remainder_base + len_per_row*i);
                }

                // apply our butterfly function down the columns
                let output = $butterfly_fn_lo(columns, self);

                // always write the first row without twiddles
                buffer.store_complex_remainder_f32_lo(remainder_mask, output[0], partial_remainder_base);

                // for every other row, apply twiddle factors and then write back to memory, using RemainderMask to limit which memory locations to write to
                for i in 1..ROW_COUNT {
                    let twiddle = _mm256_castps256_ps128(final_twiddle_chunk[i - 1]);
                    let output = avx32_utils::fma::complex_multiply_f32_lo(twiddle, output[i]);
                    buffer.store_complex_remainder_f32_lo(remainder_mask, output, partial_remainder_base + len_per_row*i);
                }
            }
        }
    }
)}

macro_rules! mixedradix_transpose_f32{
    ($row_count: expr, $transpose_fn: path, $($unroll_workaround_index:expr);*) => (

    // Transpose the input (treated as a nx2 array) into the output (as a 2xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        const CHUNK_SIZE : usize = 4;

        const ROW_COUNT : usize = $row_count;

        let len_per_row = self.len() / ROW_COUNT;
        let chunk_count = len_per_row / CHUNK_SIZE;

        // transpose the scratch as a nx2 array into the buffer as an 2xn array
        for c in 0..chunk_count {
            let input_index_base = c*CHUNK_SIZE;
            let output_index_base = input_index_base * ROW_COUNT;

            // Load columns from the input into registers
            let mut columns = [_mm256_setzero_ps(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                columns[i] = input.load_complex_f32(input_index_base + len_per_row*i);
            }

            // transpose the columns to the rows
            let transposed = $transpose_fn(columns);

            // store the transposed rows contiguously
            // we are using a macro hack to manually unroll the loop, to work around this rustc bug:
            // https://github.com/rust-lang/rust/issues/71025
            
            // if we don't manually unroll the loop, the compiler will insert unnecessary writes+reads to the stack which tank performance
            // once the compiler bug is fixed, this can be replaced by a "for i in 0..ROW_COUNT" loop
            $( 
                output.store_complex_f32(output_index_base + CHUNK_SIZE * $unroll_workaround_index, transposed[$unroll_workaround_index]);
            )*
        }

        // transpose the remainder
        let partial_remainder = len_per_row % CHUNK_SIZE;
        if partial_remainder > 0 {
            let load_remainder_mask = avx32_utils::RemainderMask::new_f32(partial_remainder);

            let input_index_base = chunk_count * CHUNK_SIZE;
            let output_index_base = input_index_base * ROW_COUNT;

            let mut columns = [_mm256_setzero_ps(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                columns[i] = input.load_complex_remainder_f32(load_remainder_mask, input_index_base + len_per_row*i);
            }

            // transpose the columns to the rows
            let transposed = $transpose_fn(columns);

            // store the transposed rows contiguously, keeping in mind that because we're dealing with a remainder, we should only write a portion of the data
            // This last section is different per row count, so it's not somethign we can put inside the macro, so we're going to call back into the algorithm for this
            Self::write_partial_remainder(&mut output[output_index_base..], transposed, partial_remainder)
        }
    }
)}

macro_rules! mixedradix_column_butterflies_f64{
    ($row_count: expr, $butterfly_fn: expr, $butterfly_fn_lo: expr) => (

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<f64>]) {
        // How many complex numbers fit in a single register
        const CHUNK_SIZE : usize = 2;

        // How many rows this FFT has, ie 2 for 2xn, 4 for 4xn, etc
        const ROW_COUNT : usize = $row_count;
        const TWIDDLES_PER_COLUMN : usize = ROW_COUNT - 1;

        let len_per_row = self.len() / ROW_COUNT;
        let chunk_count = len_per_row / CHUNK_SIZE;

        // process the column FFTs
        for (c, twiddle_chunk) in self.common_data.twiddles.chunks_exact(TWIDDLES_PER_COLUMN).take(chunk_count).enumerate() {
            let index_base = c*CHUNK_SIZE;

            // Load columns from the buffer into registers
            let mut columns = [_mm256_setzero_pd(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                columns[i] = buffer.load_complex_f64(index_base + len_per_row*i);
            }

            // apply our butterfly function down the columns
            let output = $butterfly_fn(columns, self);

            // always write the first row directly back without twiddles
            buffer.store_complex_f64(output[0], index_base);
            
            // for every other row, apply twiddle factors and then write back to memory
            for i in 1..ROW_COUNT {
                let twiddle = twiddle_chunk[i - 1];
                let output = avx64_utils::fma::complex_multiply_f64(twiddle, output[i]);
                buffer.store_complex_f64(output, index_base + len_per_row*i);
            }
        }

        // finally, we might have a single partial chunk.
        // Normally, we can fit 4 complex numbers into an AVX register, but we only have `partial_remainder` columns left, so we need special logic to handle these final columns
        let partial_remainder = len_per_row % CHUNK_SIZE;
        if partial_remainder > 0 {
            let partial_remainder_base = chunk_count * CHUNK_SIZE;
            let partial_remainder_twiddle_base = self.common_data.twiddles.len() - TWIDDLES_PER_COLUMN;

            // Load partial columns for our final remainder, using RemainderMask to limit which memory locations we load
            let mut columns = [_mm_setzero_pd(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                columns[i] = buffer.load_complex_f64_lo(partial_remainder_base + len_per_row*i);
            }

            // apply our butterfly function down the columns
            let output = $butterfly_fn_lo(columns, self);

            // always write the first row without twiddles
            buffer.store_complex_f64_lo(output[0], partial_remainder_base);

            // for every other row, apply twiddle factors and then write back to memory, using RemainderMask to limit which memory locations to write to
            for i in 1..ROW_COUNT {
                let twiddle = _mm256_castpd256_pd128(self.common_data.twiddles[partial_remainder_twiddle_base + i - 1]);
                let output = avx64_utils::fma::complex_multiply_f64_lo(twiddle, output[i]);
                buffer.store_complex_f64_lo(output, partial_remainder_base + len_per_row*i);
            }
        }
    }
)}

macro_rules! mixedradix_transpose_f64{
    ($row_count: expr, $transpose_fn: path, $($unroll_workaround_index:expr);*) => (

    // Transpose the input (treated as a nx2 array) into the output (as a 2xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &[Complex<f64>], output: &mut [Complex<f64>]) {
        const CHUNK_SIZE : usize = 2;

        const ROW_COUNT : usize = $row_count;

        let len_per_row = self.len() / ROW_COUNT;
        let chunk_count = len_per_row / CHUNK_SIZE;

        // transpose the scratch as a nx2 array into the buffer as an 2xn array
        for c in 0..chunk_count {
            let input_index_base = c*CHUNK_SIZE;
            let output_index_base = input_index_base * ROW_COUNT;

            // Load columns from the input into registers
            let mut columns = [_mm256_setzero_pd(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                columns[i] = input.load_complex_f64(input_index_base + len_per_row*i);
            }

            // transpose the columns to the rows
            let transposed = $transpose_fn(columns);

            // store the transposed rows contiguously
            // we are using a macro hack to manually unroll the loop, to work around this rustc bug:
            // https://github.com/rust-lang/rust/issues/71025
            
            // if we don't manually unroll the loop, the compiler will insert unnecessary writes+reads to the stack which tank performance
            // once the compiler bug is fixed, this can be replaced by a "for i in 0..ROW_COUNT" loop
            $( 
                output.store_complex_f64(transposed[$unroll_workaround_index], output_index_base + CHUNK_SIZE * $unroll_workaround_index);
            )*
        }

        // transpose the remainder
        let partial_remainder = len_per_row % CHUNK_SIZE;
        if partial_remainder > 0 {
            let input_index_base = chunk_count * CHUNK_SIZE;
            let output_index_base = input_index_base * ROW_COUNT;

            // since we only have a single column, we don't need to do any transposing, just copying
            for n in 0..ROW_COUNT {
                let row = input.load_complex_f64_lo(input_index_base + len_per_row*n);
                output.store_complex_f64_lo(row, output_index_base + n);
            }
        }
    }
)}

pub struct MixedRadix2xnAvx<T, V> {
    common_data: CommonSimdData<T,V>
}
boilerplate_fft_commondata!(MixedRadix2xnAvx);

impl MixedRadix2xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();
    
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f32>>) -> Self {
        Self {
            common_data: mixedradix_gen_data_f32!(2, inner_fft),
        }
    }

    mixedradix_column_butterflies_f32!(2,
        |rows, _:_| AvxVector::column_butterfly2(rows),
        |rows, _:_| AvxVector::column_butterfly2(rows)
    );
    mixedradix_transpose_f32!(2, avx32_utils::interleave_evens_odds_f32, 0;1);

    // This is called by mixedradix_transpose_f32!() -- this one single section will be different for every mixed radix algorithm,
    // and even different from f32 to f64 of the same algorithm, so it has to go outside the macro
    #[inline(always)]
    unsafe fn write_partial_remainder(output: &mut[Complex<f32>], packed_data: [__m256; 2], partial_remainder: usize) {
        assert!(partial_remainder > 0 && partial_remainder < 4);
        if partial_remainder == 1 {
            output.store_complex_f32_lo(packed_data[0].lo(), 0);
        } else {
            output.store_complex_f32(0, packed_data[0]);

            if partial_remainder == 3 {
                output.store_complex_f32_lo(packed_data[1].lo(), 4);
            }
        }
    }
}
impl MixedRadix2xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();
    
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f64>>) -> Self {
        Self {
            common_data: mixedradix_gen_data_f64!(2, inner_fft),
        }
    }

    mixedradix_column_butterflies_f64!(2,
        |columns, _:_| AvxVector::column_butterfly2(columns),
        |columns, _:_| AvxVector::column_butterfly2(columns)
    );
    mixedradix_transpose_f64!(2, avx64_utils::transpose_2x2_f64, 0;1);
}

pub struct MixedRadix3xnAvx<T, V> {
    twiddles_butterfly3: V,
    common_data: CommonSimdData<T,V>
}
boilerplate_fft_commondata!(MixedRadix3xnAvx);

impl MixedRadix3xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();
    
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f32>>) -> Self {
        Self {
            twiddles_butterfly3: AvxVector::broadcast_complex_elements(f32::generate_twiddle_factor(1, 3, inner_fft.is_inverse())),
            common_data: mixedradix_gen_data_f32!(3, inner_fft),
        }
    }

    mixedradix_column_butterflies_f32!(3, 
        |columns, this: &Self| AvxVector::column_butterfly3(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector::column_butterfly3(columns, this.twiddles_butterfly3.lo())
    );
    mixedradix_transpose_f32!(3, avx32_utils::transpose_4x3_packed_f32, 0;1;2);

    // This is called by mixedradix_transpose_f32!() -- this one single section will be different for every mixed radix algorithm,
    // and even different from f32 to f64 of the same algorithm, so it has to go outside the macro
    #[inline(always)]
    unsafe fn write_partial_remainder(output: &mut[Complex<f32>], packed_data: [__m256; 3], partial_remainder: usize) {
        assert!(partial_remainder > 0 && partial_remainder < 4);
        if partial_remainder == 1 {
            output.store_complex_remainder3_f32(packed_data[0], 0);
        } else {
            output.store_complex_f32(0, packed_data[0]);

            if partial_remainder == 2 {
                output.store_complex_f32_lo(packed_data[1].lo(), 4);
            }
            if partial_remainder == 3 {
                output.store_complex_f32(4, packed_data[1]);
                output.store_complex_remainder1_f32(packed_data[2].lo(), 8);
            }
        }
    }
}
impl MixedRadix3xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();
    
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f64>>) -> Self {
        Self {
            twiddles_butterfly3: AvxVector::broadcast_complex_elements(f64::generate_twiddle_factor(1, 3, inner_fft.is_inverse())),
            common_data: mixedradix_gen_data_f64!(3, inner_fft),
        }
    }

    mixedradix_column_butterflies_f64!(3,
        |columns, this: &Self| AvxVector::column_butterfly3(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector::column_butterfly3(columns, this.twiddles_butterfly3.lo())
    );
    mixedradix_transpose_f64!(3, avx64_utils::transpose_2x3_to_3x2_packed_f64, 0;1;2);
}





pub struct MixedRadix4xnAvx<T, V> {
    twiddles_butterfly4: Rotation90<V>,
    common_data: CommonSimdData<T,V>,
}
boilerplate_fft_commondata!(MixedRadix4xnAvx);

impl MixedRadix4xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f32>>) -> Self {
        Self {
            twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
            common_data: mixedradix_gen_data_f32!(4, inner_fft),
        }
    }

    mixedradix_column_butterflies_f32!(4, 
        |columns, this: &Self| AvxVector::column_butterfly4(columns, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly4(columns, this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f32!(4, avx32_utils::transpose_4x4_f32, 0;1;2;3);

    // This is called by mixedradix_transpose_f32!() -- this one single section will be different for every mixed radix algorithm,
    // and even different from f32 to f64 of the same algorithm, so it has to go outside the macro
    #[inline(always)]
    unsafe fn write_partial_remainder(output: &mut[Complex<f32>], packed_data: [__m256; 4], partial_remainder: usize) {
        // We're manually unrolling this loop. if we don't, the compiler will insert unnecessary writes+reads to the stack which tank performance
        // see: https://github.com/rust-lang/rust/issues/71025
        // once the compiler bug is fixed, this can be replaced by a "for i in 0..partial_remainder" loop
        output.store_complex_f32(0, packed_data[0]);
        if partial_remainder > 1 {
            output.store_complex_f32(4, packed_data[1]);
            if partial_remainder > 2 {
                output.store_complex_f32(8, packed_data[2]);
            }
        }
    }
}
impl MixedRadix4xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f64>>) -> Self {
        Self {
            twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
            common_data: mixedradix_gen_data_f64!(4, inner_fft),
        }
    }

    mixedradix_column_butterflies_f64!(4,
        |columns, this: &Self| AvxVector::column_butterfly4(columns, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly4(columns, this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f64!(4, avx64_utils::transpose_2x4_to_4x2_packed_f64, 0;1;2;3);
}





pub struct MixedRadix6xnAvx<T, V> {
    twiddles_butterfly3: V,
    common_data: CommonSimdData<T,V>
}
boilerplate_fft_commondata!(MixedRadix6xnAvx);

impl MixedRadix6xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();
    
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f32>>) -> Self {
        Self {
            twiddles_butterfly3: AvxVector::broadcast_complex_elements(f32::generate_twiddle_factor(1, 3, inner_fft.is_inverse())),
            common_data: mixedradix_gen_data_f32!(6, inner_fft),
        }
    }

    mixedradix_column_butterflies_f32!(6,
        |columns, this: &Self| AvxVector::column_butterfly6(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector::column_butterfly6(columns, this.twiddles_butterfly3.lo())
    );
    mixedradix_transpose_f32!(6, avx32_utils::transpose_4x6_to_6x4_packed_f32, 0;1;2;3;4;5);

    // This is called by mixedradix_transpose_f32!() -- this one single section will be different for every mixed radix algorithm,
    // and even different from f32 to f64 of the same algorithm, so it has to go outside the macro
    #[inline(always)]
    unsafe fn write_partial_remainder(output: &mut[Complex<f32>], packed_data: [__m256; 6], partial_remainder: usize) {
        assert!(partial_remainder > 0 && partial_remainder < 4);
        output.store_complex_f32(0, packed_data[0]);
        if partial_remainder == 1 {
            output.store_complex_f32_lo(packed_data[1].lo(), 4);
        } else {
            output.store_complex_f32(4, packed_data[1]);
            output.store_complex_f32(8, packed_data[2]);

            if partial_remainder == 3 {
                output.store_complex_f32(12, packed_data[3]);
                output.store_complex_f32_lo(packed_data[4].lo(), 16);
            }
        }
    }
}
impl MixedRadix6xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();
    
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f64>>) -> Self {
        Self {
            twiddles_butterfly3: AvxVector::broadcast_complex_elements(f64::generate_twiddle_factor(1, 3, inner_fft.is_inverse())),
            common_data: mixedradix_gen_data_f64!(6, inner_fft),
        }
    }

    mixedradix_column_butterflies_f64!(6,
        |columns, this: &Self| AvxVector::column_butterfly6(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector::column_butterfly6(columns, this.twiddles_butterfly3.lo())
    );
    mixedradix_transpose_f64!(6, avx64_utils::transpose_2x6_to_6x2_packed_f64, 0;1;2;3;4;5);
}







pub struct MixedRadix8xnAvx<T, V> {
    twiddles_butterfly4: Rotation90<V>,
    common_data: CommonSimdData<T, V>,
}
boilerplate_fft_commondata!(MixedRadix8xnAvx);

impl MixedRadix8xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f32>>) -> Self {
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
            common_data: mixedradix_gen_data_f32!(8, inner_fft),
        }
    }

    mixedradix_column_butterflies_f32!(8,
        |columns, this: &Self| AvxVector::column_butterfly8(columns, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly8(columns, this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f32!(8, avx32_utils::transpose_4x8_to_8x4_packed_f32, 0;1;2;3;4;5;6;7);

    // This is called by mixedradix_transpose_f32!() -- this one single section will be different for every mixed radix algorithm,
    // and even different from f32 to f64 of the same algorithm, so it has to go outside the macro
    #[inline(always)]
    unsafe fn write_partial_remainder(output: &mut[Complex<f32>], packed_data: [__m256; 8], partial_remainder: usize) {
        // We're manually unrolling this loop. if we don't, the compiler will insert unnecessary writes+reads to the stack which tank performance
        // see: https://github.com/rust-lang/rust/issues/71025
        // once the compiler bug is fixed, this can be replaced by a "for i in 0..partial_remainder" loop
        output.store_complex_f32(0, packed_data[0]);
        output.store_complex_f32(4, packed_data[1]);
        if partial_remainder > 1 {
            output.store_complex_f32(8, packed_data[2]);
            output.store_complex_f32(12, packed_data[3]);
            if partial_remainder > 2 {
                output.store_complex_f32(16, packed_data[4]);
                output.store_complex_f32(20, packed_data[5]);
            }
        }
    }
}
impl MixedRadix8xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f64>>) -> Self {
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
            common_data: mixedradix_gen_data_f64!(8, inner_fft),
        }
    }

    mixedradix_column_butterflies_f64!(8,
        |columns, this: &Self| AvxVector::column_butterfly8(columns, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly8(columns, this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f64!(8, avx64_utils::transpose_2x8_to_8x2_packed_f64, 0;1;2;3;4;5;6;7);
}




pub struct MixedRadix9xnAvx<T, V> {
    twiddles_butterfly9: [V; 3],
    twiddles_butterfly9_lo: [V; 2],
    twiddles_butterfly3: V,
    common_data: CommonSimdData<T, V>,
}
boilerplate_fft_commondata!(MixedRadix9xnAvx);

impl MixedRadix9xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();

        let twiddles_butterfly9_lo = [
            f32::generate_twiddle_factor(1, 9, inverse),
            f32::generate_twiddle_factor(1, 9, inverse),
            f32::generate_twiddle_factor(2, 9, inverse),
            f32::generate_twiddle_factor(2, 9, inverse),
            f32::generate_twiddle_factor(2, 9, inverse),
            f32::generate_twiddle_factor(2, 9, inverse),
            f32::generate_twiddle_factor(4, 9, inverse),
            f32::generate_twiddle_factor(4, 9, inverse),
        ];
        Self {
        	twiddles_butterfly9: [
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 9, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(2, 9, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(4, 9, inverse)),
            ],
            twiddles_butterfly9_lo: [
                twiddles_butterfly9_lo.load_complex_f32(0),
                twiddles_butterfly9_lo.load_complex_f32(4),
            ],
        	twiddles_butterfly3: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            common_data: mixedradix_gen_data_f32!(9, inner_fft),
        }
    }

    mixedradix_column_butterflies_f32!(9, 
        |columns, this: &Self| AvxVector256::column_butterfly9(columns, this.twiddles_butterfly9, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector128::column_butterfly9(columns, this.twiddles_butterfly9_lo, this.twiddles_butterfly3)
    );
    mixedradix_transpose_f32!(9, avx32_utils::transpose_4x9_to_9x4_packed_f32, 0;1;2;3;4;5;6;7;8);

    // This is called by mixedradix_transpose_f32!() -- this one single section will be different for every mixed radix algorithm,
    // and even different from f32 to f64 of the same algorithm, so it has to go outside the macro
    #[inline(always)]
    unsafe fn write_partial_remainder(output: &mut[Complex<f32>], packed_data: [__m256; 9], partial_remainder: usize) {
        output.store_complex_f32(0, packed_data[0]);
        output.store_complex_f32(4, packed_data[1]);

        if partial_remainder == 1 {
            output.store_complex_remainder1_f32(_mm256_castps256_ps128(packed_data[2]), 8);
        } else {
            output.store_complex_f32(8,  packed_data[2]);
            output.store_complex_f32(12, packed_data[3]);

            if partial_remainder == 2 {
                output.store_complex_f32_lo(_mm256_castps256_ps128(packed_data[4]), 16);
            }
            else {
                output.store_complex_f32(16, packed_data[4]);
                output.store_complex_f32(20, packed_data[5]);
                output.store_complex_remainder3_f32(packed_data[6], 24);
            }
        }
    }
}
impl MixedRadix9xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f64>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let twiddles_butterfly9_lo = [
            f64::generate_twiddle_factor(1, 9, inverse),
            f64::generate_twiddle_factor(2, 9, inverse),
            f64::generate_twiddle_factor(2, 9, inverse),
            f64::generate_twiddle_factor(4, 9, inverse),
        ];
        Self {
        	twiddles_butterfly9: [
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 9, inverse)),
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(2, 9, inverse)),
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(4, 9, inverse)),
            ],
            twiddles_butterfly9_lo: [
                twiddles_butterfly9_lo.load_complex_f64(0),
                twiddles_butterfly9_lo.load_complex_f64(2),
            ],
        	twiddles_butterfly3: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 3, inverse)),
            common_data: mixedradix_gen_data_f64!(9, inner_fft),
        }
    }

    mixedradix_column_butterflies_f64!(9,
        |columns, this: &Self| AvxVector256::column_butterfly9(columns, this.twiddles_butterfly9, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector128::column_butterfly9(columns, this.twiddles_butterfly9_lo, this.twiddles_butterfly3)
    );
    mixedradix_transpose_f64!(9, avx64_utils::transpose_2x9_to_9x2_packed_f64, 0;1;2;3;4;5;6;7;8);
}







pub struct MixedRadix12xnAvx<T, V> {
    twiddles_butterfly4: Rotation90<V>,
    twiddles_butterfly3: V,
    common_data: CommonSimdData<T, V>,
}
boilerplate_fft_commondata!(MixedRadix12xnAvx);

impl MixedRadix12xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
        	twiddles_butterfly3: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            common_data: mixedradix_gen_data_f32!(12, inner_fft),
        }
    }

    mixedradix_column_butterflies_f32!(12, 
        |columns, this: &Self| AvxVector256::column_butterfly12(columns, this.twiddles_butterfly3, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector128::column_butterfly12(columns, this.twiddles_butterfly3, this.twiddles_butterfly4)
    );
    mixedradix_transpose_f32!(12, avx32_utils::transpose_4x12_to_12x4_packed_f32, 0;1;2;3;4;5;6;7;8;9;10;11);

    // This is called by mixedradix_transpose_f32!() -- this one single section will be different for every mixed radix algorithm,
    // and even different from f32 to f64 of the same algorithm, so it has to go outside the macro
    #[inline(always)]
    unsafe fn write_partial_remainder(output: &mut[Complex<f32>], packed_data: [__m256; 12], partial_remainder: usize) {
        // We're manually unrolling this loop. if we don't, the compiler will insert unnecessary writes+reads to the stack which tank performance
        // see: https://github.com/rust-lang/rust/issues/71025
        // once the compiler bug is fixed, this can be replaced by a "for i in 0..partial_remainder" loop
        output.store_complex_f32(0, packed_data[0]);
        output.store_complex_f32(4, packed_data[1]);
        output.store_complex_f32(8, packed_data[2]);
        if partial_remainder > 1 {
            output.store_complex_f32(12, packed_data[3]);
            output.store_complex_f32(16, packed_data[4]);
            output.store_complex_f32(20, packed_data[5]);
            if partial_remainder > 2 {
                output.store_complex_f32(24, packed_data[6]);
                output.store_complex_f32(28, packed_data[7]);
                output.store_complex_f32(32, packed_data[8]);
            }
        }
    }
}
impl MixedRadix12xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f64>>) -> Self {
        let inverse = inner_fft.is_inverse();
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
        	twiddles_butterfly3: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 3, inverse)),
            common_data: mixedradix_gen_data_f64!(12, inner_fft),
        }
    }

    mixedradix_column_butterflies_f64!(12,
        |columns, this: &Self| AvxVector256::column_butterfly12(columns, this.twiddles_butterfly3, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector128::column_butterfly12(columns, this.twiddles_butterfly3, this.twiddles_butterfly4)
    );
    mixedradix_transpose_f64!(12, avx64_utils::transpose_2x12_to_12x2_packed_f64, 0;1;2;3;4;5;6;7;8;9;10;11);
}






pub struct MixedRadix16xnAvx<T, V> {
    twiddles_butterfly4: Rotation90<V>,
    twiddles_butterfly16: [V; 2],
    common_data: CommonSimdData<T, V>,
}
boilerplate_fft_commondata!(MixedRadix16xnAvx);

impl MixedRadix16xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
        	twiddles_butterfly16: [
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 16, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(3, 16, inverse)),
            ],
            common_data: mixedradix_gen_data_f32!(16, inner_fft),
        }
    }

    mixedradix_column_butterflies_f32!(16, 
        |columns, this: &Self| AvxVector::column_butterfly16(columns, this.twiddles_butterfly16, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly16(columns, [this.twiddles_butterfly16[0].lo(), this.twiddles_butterfly16[1].lo()], this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f32!(16, avx32_utils::transpose_4x16_to_16x4_packed_f32, 0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15);

    // This is called by mixedradix_transpose_f32!() -- this one single section will be different for every mixed radix algorithm,
    // and even different from f32 to f64 of the same algorithm, so it has to go outside the macro
    #[inline(always)]
    unsafe fn write_partial_remainder(output: &mut[Complex<f32>], packed_data: [__m256; 16], partial_remainder: usize) {
        // We're manually unrolling this loop. if we don't, the compiler will insert unnecessary writes+reads to the stack which tank performance
        // see: https://github.com/rust-lang/rust/issues/71025
        // once the compiler bug is fixed, this can be replaced by a "for i in 0..partial_remainder" loop
        output.store_complex_f32(0, packed_data[0]);
        output.store_complex_f32(4, packed_data[1]);
        output.store_complex_f32(8, packed_data[2]);
        output.store_complex_f32(12, packed_data[3]);
        if partial_remainder > 1 {
            output.store_complex_f32(16, packed_data[4]);
            output.store_complex_f32(20, packed_data[5]);
            output.store_complex_f32(24, packed_data[6]);
            output.store_complex_f32(28, packed_data[7]);
            if partial_remainder > 2 {
                output.store_complex_f32(32, packed_data[8]);
                output.store_complex_f32(36, packed_data[9]);
                output.store_complex_f32(40, packed_data[10]);
                output.store_complex_f32(44, packed_data[11]);
            }
        }
    }
}
impl MixedRadix16xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<f64>>) -> Self {
        let inverse = inner_fft.is_inverse();
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
        	twiddles_butterfly16: [
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 16, inverse)),
                avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(3, 16, inverse)),
            ],
            common_data: mixedradix_gen_data_f64!(16, inner_fft),
        }
    }

    mixedradix_column_butterflies_f64!(16,
        |columns, this: &Self| AvxVector::column_butterfly16(columns, this.twiddles_butterfly16, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly16(columns, [this.twiddles_butterfly16[0].lo(), this.twiddles_butterfly16[1].lo()], this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f64!(16, avx64_utils::transpose_2x16_to_16x2_packed_f64, 0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;
    use std::sync::Arc;
    use crate::algorithm::*;

    macro_rules! test_avx_mixed_radix {
        ($f32_test_name:ident, $f64_test_name:ident, $struct_name:ident, $inner_count:expr) => (
            #[test]
            fn $f32_test_name() {
                for inner_fft_len in 1..32 {
                    let len = inner_fft_len * $inner_count;

                    let inner_fft_forward = Arc::new(DFT::new(inner_fft_len, false)) as Arc<dyn Fft<f32>>;
                    let fft_forward = $struct_name::new_f32(inner_fft_forward).expect("Can't run test because this machine doesn't have the required instruction sets");
                    check_fft_algorithm(&fft_forward, len, false);

                    let inner_fft_inverse = Arc::new(DFT::new(inner_fft_len, true)) as Arc<dyn Fft<f32>>;
                    let fft_inverse = $struct_name::new_f32(inner_fft_inverse).expect("Can't run test because this machine doesn't have the required instruction sets");
                    check_fft_algorithm(&fft_inverse, len, true);
                }
            }
            #[test]
            fn $f64_test_name() {
                for inner_fft_len in 1..32 {
                    let len = inner_fft_len * $inner_count;

                    let inner_fft_forward = Arc::new(DFT::new(inner_fft_len, false)) as Arc<dyn Fft<f64>>;
                    let fft_forward = $struct_name::new_f64(inner_fft_forward).expect("Can't run test because this machine doesn't have the required instruction sets");
                    check_fft_algorithm(&fft_forward, len, false);

                    let inner_fft_inverse = Arc::new(DFT::new(inner_fft_len, true)) as Arc<dyn Fft<f64>>;
                    let fft_inverse = $struct_name::new_f64(inner_fft_inverse).expect("Can't run test because this machine doesn't have the required instruction sets");
                    check_fft_algorithm(&fft_inverse, len, true);
                }
            }
        )
    }

    test_avx_mixed_radix!(test_mixedradix_2xn_avx_f32, test_mixedradix_2xn_avx_f64, MixedRadix2xnAvx, 2);
    test_avx_mixed_radix!(test_mixedradix_3xn_avx_f32, test_mixedradix_3xn_avx_f64, MixedRadix3xnAvx, 3);
    test_avx_mixed_radix!(test_mixedradix_4xn_avx_f32, test_mixedradix_4xn_avx_f64, MixedRadix4xnAvx, 4);
    test_avx_mixed_radix!(test_mixedradix_6xn_avx_f32, test_mixedradix_6xn_avx_f64, MixedRadix6xnAvx, 6);
    test_avx_mixed_radix!(test_mixedradix_8xn_avx_f32, test_mixedradix_8xn_avx_f64, MixedRadix8xnAvx, 8);
    test_avx_mixed_radix!(test_mixedradix_9xn_avx_f32, test_mixedradix_9xn_avx_f64, MixedRadix9xnAvx, 9);
    test_avx_mixed_radix!(test_mixedradix_12xn_avx_f32, test_mixedradix_12xn_avx_f64, MixedRadix12xnAvx, 12);
    test_avx_mixed_radix!(test_mixedradix_16xn_avx_f32, test_mixedradix_16xn_avx_f64, MixedRadix16xnAvx, 16);
}