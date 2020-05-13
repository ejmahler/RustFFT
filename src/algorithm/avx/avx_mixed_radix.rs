use std::sync::Arc;
use std::arch::x86_64::*;

use num_complex::Complex;

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

macro_rules! mixedradix_gen_data {
    ($row_count: expr, $inner_fft:expr) => {{
        // Important constants
        const ROW_COUNT : usize = $row_count;
        const TWIDDLES_PER_COLUMN : usize = ROW_COUNT - 1;

        // derive some info from our inner FFT
        let inverse = $inner_fft.is_inverse();
        let len_per_row = $inner_fft.len();
        let len = len_per_row * ROW_COUNT;

        // We're going to process each row of the FFT one AVX register at a time. We need to know how many AVX registers each row can fit,
        // and if the last register in each row going to have partial data (ie a remainder)
        let quotient = len_per_row / V::COMPLEX_PER_VECTOR;
        let remainder = len_per_row % V::COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + div_ceil(remainder, V::COMPLEX_PER_VECTOR);

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * TWIDDLES_PER_COLUMN);
        for x in 0..num_twiddle_columns {
            for y in 1..ROW_COUNT {
                twiddles.push(AvxVector::make_mixedradix_twiddle_chunk(x * V::COMPLEX_PER_VECTOR, y, len, inverse));
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
            // first, see how many full AVX vectors we have to write
            let full_vector_count = match partial_remainder {
                1 => {
                    let vector_count = ROW_COUNT / CHUNK_SIZE;
                    for i in 0..vector_count {
                        output.store_complex_f32(output_index_base + i * CHUNK_SIZE, transposed[i]);
                    }
                    vector_count
                },
                2 => {
                    let vector_count = 2*ROW_COUNT / CHUNK_SIZE;
                    for i in 0..vector_count {
                        output.store_complex_f32(output_index_base + i * CHUNK_SIZE, transposed[i]);
                    }
                    vector_count
                },
                3 => {
                    let vector_count = 3*ROW_COUNT / CHUNK_SIZE;
                    for i in 0..vector_count {
                        output.store_complex_f32(output_index_base + i * CHUNK_SIZE, transposed[i]);
                    }
                    vector_count
                },
                _ => unreachable!(),
            };

            // finally, transposed[full_vector_count] is only a partial vector, so write out only the elements we need to
            let final_remainder = (partial_remainder * ROW_COUNT) % CHUNK_SIZE;
            match final_remainder {
                0 => {},
                1 => output.store_complex_remainder1_f32(transposed[full_vector_count].lo(), output_index_base + full_vector_count * CHUNK_SIZE),
                2 => output.store_complex_f32_lo(transposed[full_vector_count].lo(), output_index_base + full_vector_count * CHUNK_SIZE),
                3 => output.store_complex_remainder3_f32(transposed[full_vector_count], output_index_base + full_vector_count * CHUNK_SIZE),
                _ => unreachable!(),
            }
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

impl<T: FFTnum, V: AvxVector> MixedRadix2xnAvx<T, V> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            common_data: mixedradix_gen_data!(2, inner_fft),
        }
    }
}

impl MixedRadix2xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    mixedradix_column_butterflies_f32!(2,
        |rows, _:_| AvxVector::column_butterfly2(rows),
        |rows, _:_| AvxVector::column_butterfly2(rows)
    );
    mixedradix_transpose_f32!(2, avx32_utils::interleave_evens_odds_f32, 0;1);
}
impl MixedRadix2xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

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

impl<T: FFTnum, V: AvxVector> MixedRadix3xnAvx<T, V> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            twiddles_butterfly3: V::broadcast_twiddle(1, 3, inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(3, inner_fft),
        }
    }
}

impl MixedRadix3xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    mixedradix_column_butterflies_f32!(3, 
        |columns, this: &Self| AvxVector::column_butterfly3(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector::column_butterfly3(columns, this.twiddles_butterfly3.lo())
    );
    mixedradix_transpose_f32!(3, avx32_utils::transpose_4x3_packed_f32, 0;1;2);
}
impl MixedRadix3xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

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

impl<T: FFTnum, V: AvxVector> MixedRadix4xnAvx<T, V> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            twiddles_butterfly4: V::make_rotation90(inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(4, inner_fft),
        }
    }
}

impl MixedRadix4xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    mixedradix_column_butterflies_f32!(4, 
        |columns, this: &Self| AvxVector::column_butterfly4(columns, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly4(columns, this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f32!(4, avx32_utils::transpose_4x4_f32, 0;1;2;3);
}
impl MixedRadix4xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

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

impl<T: FFTnum, V: AvxVector> MixedRadix6xnAvx<T, V> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            twiddles_butterfly3: V::broadcast_twiddle(1, 3, inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(6, inner_fft),
        }
    }
}

impl MixedRadix6xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    mixedradix_column_butterflies_f32!(6,
        |columns, this: &Self| AvxVector256::column_butterfly6(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector128::column_butterfly6(columns, this.twiddles_butterfly3)
    );
    mixedradix_transpose_f32!(6, avx32_utils::transpose_4x6_to_6x4_packed_f32, 0;1;2;3;4;5);
}
impl MixedRadix6xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

    mixedradix_column_butterflies_f64!(6,
        |columns, this: &Self| AvxVector256::column_butterfly6(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector128::column_butterfly6(columns, this.twiddles_butterfly3)
    );
    mixedradix_transpose_f64!(6, avx64_utils::transpose_2x6_to_6x2_packed_f64, 0;1;2;3;4;5);
}







pub struct MixedRadix8xnAvx<T, V> {
    twiddles_butterfly4: Rotation90<V>,
    common_data: CommonSimdData<T, V>,
}
boilerplate_fft_commondata!(MixedRadix8xnAvx);

impl<T: FFTnum, V: AvxVector> MixedRadix8xnAvx<T, V> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            twiddles_butterfly4: V::make_rotation90(inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(8, inner_fft),
        }
    }
}

impl MixedRadix8xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    mixedradix_column_butterflies_f32!(8,
        |columns, this: &Self| AvxVector::column_butterfly8(columns, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly8(columns, this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f32!(8, avx32_utils::transpose_4x8_to_8x4_packed_f32, 0;1;2;3;4;5;6;7);
}
impl MixedRadix8xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

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

impl<T: FFTnum, V: AvxVector256> MixedRadix9xnAvx<T, V> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inverse = inner_fft.is_inverse();

        let twiddle1 = V::HalfVector::broadcast_twiddle(1, 9, inner_fft.is_inverse());
        let twiddle2 = V::HalfVector::broadcast_twiddle(2, 9, inner_fft.is_inverse());
        let twiddle4 = V::HalfVector::broadcast_twiddle(4, 9, inner_fft.is_inverse());

        Self {
        	twiddles_butterfly9: [
                V::broadcast_twiddle(1, 9, inverse),
                V::broadcast_twiddle(2, 9, inverse),
                V::broadcast_twiddle(4, 9, inverse),
            ],
            twiddles_butterfly9_lo: [
                V::merge(twiddle1, twiddle2),
                V::merge(twiddle2, twiddle4),
            ],
        	twiddles_butterfly3: V::broadcast_twiddle(1, 3, inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(9, inner_fft),
        }
    }
}

impl MixedRadix9xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    mixedradix_column_butterflies_f32!(9, 
        |columns, this: &Self| AvxVector256::column_butterfly9(columns, this.twiddles_butterfly9, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector128::column_butterfly9(columns, this.twiddles_butterfly9_lo, this.twiddles_butterfly3)
    );
    mixedradix_transpose_f32!(9, avx32_utils::transpose_4x9_to_9x4_packed_f32, 0;1;2;3;4;5;6;7;8);
}
impl MixedRadix9xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

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

impl<T: FFTnum, V: AvxVector256> MixedRadix12xnAvx<T, V> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inverse = inner_fft.is_inverse();
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inverse),
        	twiddles_butterfly3: V::broadcast_twiddle(1, 3, inverse),
            common_data: mixedradix_gen_data!(12, inner_fft),
        }
    }
}

impl MixedRadix12xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    mixedradix_column_butterflies_f32!(12, 
        |columns, this: &Self| AvxVector256::column_butterfly12(columns, this.twiddles_butterfly3, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector128::column_butterfly12(columns, this.twiddles_butterfly3, this.twiddles_butterfly4)
    );
    mixedradix_transpose_f32!(12, avx32_utils::transpose_4x12_to_12x4_packed_f32, 0;1;2;3;4;5;6;7;8;9;10;11);
}
impl MixedRadix12xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

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

impl<T: FFTnum, V: AvxVector256> MixedRadix16xnAvx<T, V> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inverse = inner_fft.is_inverse();
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
        	twiddles_butterfly16: [
                V::broadcast_twiddle(1, 16, inverse),
                V::broadcast_twiddle(3, 16, inverse),
            ],
            common_data: mixedradix_gen_data!(16, inner_fft),
        }
    }
}

impl MixedRadix16xnAvx<f32, __m256> {
    mixedradix_boilerplate_f32!();

    mixedradix_column_butterflies_f32!(16, 
        |columns, this: &Self| AvxVector::column_butterfly16(columns, this.twiddles_butterfly16, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly16(columns, [this.twiddles_butterfly16[0].lo(), this.twiddles_butterfly16[1].lo()], this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose_f32!(16, avx32_utils::transpose_4x16_to_16x4_packed_f32, 0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15);
}
impl MixedRadix16xnAvx<f64, __m256d> {
    mixedradix_boilerplate_f64!();

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