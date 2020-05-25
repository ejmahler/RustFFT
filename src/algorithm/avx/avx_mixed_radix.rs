use std::sync::Arc;

use num_complex::Complex;

use crate::common::FFTnum;
use crate::math_utils::div_ceil;
use crate::{Length, IsInverse, Fft};

use super::CommonSimdData;
use super::avx_vector::{AvxVector, AvxVector128, AvxVector256, Rotation90, AvxArrayGeneric, AvxArrayMutGeneric};

macro_rules! boilerplate_mixedradix {
    () => (
        /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the FFT
        /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
        #[inline]
        pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Result<Self, ()> {
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
        fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
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
        fn perform_fft_out_of_place(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
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
    )
}

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
        let quotient = len_per_row / T::AvxType::COMPLEX_PER_VECTOR;
        let remainder = len_per_row % T::AvxType::COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + div_ceil(remainder, T::AvxType::COMPLEX_PER_VECTOR);

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * TWIDDLES_PER_COLUMN);
        for x in 0..num_twiddle_columns {
            for y in 1..ROW_COUNT {
                twiddles.push(AvxVector::make_mixedradix_twiddle_chunk(x * T::AvxType::COMPLEX_PER_VECTOR, y, len, inverse));
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

macro_rules! mixedradix_column_butterflies{
    ($row_count: expr, $butterfly_fn: expr, $butterfly_fn_lo: expr) => (

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_column_butterflies(&self, buffer: &mut [Complex<T>]) {
        // How many rows this FFT has, ie 2 for 2xn, 4 for 4xn, etc
        const ROW_COUNT : usize = $row_count;
        const TWIDDLES_PER_COLUMN : usize = ROW_COUNT - 1;

        let len_per_row = self.len() / ROW_COUNT;
        let chunk_count = len_per_row / T::AvxType::COMPLEX_PER_VECTOR;

        // process the column FFTs
        for (c, twiddle_chunk) in self.common_data.twiddles.chunks_exact(TWIDDLES_PER_COLUMN).take(chunk_count).enumerate() {
            let index_base = c*T::AvxType::COMPLEX_PER_VECTOR;

            // Load columns from the buffer into registers
            let mut columns = [AvxVector::zero(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                columns[i] = buffer.load_complex2(index_base + len_per_row*i);
            }

            // apply our butterfly function down the columns
            let output = $butterfly_fn(columns, self);

            // always write the first row directly back without twiddles
            buffer.store_complex2(output[0], index_base);
            
            // for every other row, apply twiddle factors and then write back to memory
            for i in 1..ROW_COUNT {
                let twiddle = twiddle_chunk[i - 1];
                let output = AvxVector::mul_complex(twiddle, output[i]);
                buffer.store_complex2(output, index_base + len_per_row*i);
            }
        }

        // finally, we might have a single partial chunk.
        // Normally, we can fit 4 complex numbers into an AVX register, but we only have `partial_remainder` columns left, so we need special logic to handle these final columns
        let partial_remainder = len_per_row % T::AvxType::COMPLEX_PER_VECTOR;
        if partial_remainder > 0 {
            let partial_remainder_base = chunk_count * T::AvxType::COMPLEX_PER_VECTOR;
            let partial_remainder_twiddle_base = self.common_data.twiddles.len() - TWIDDLES_PER_COLUMN;
            let final_twiddle_chunk = &self.common_data.twiddles[partial_remainder_twiddle_base..];

            if partial_remainder > 2 {
                // Load 3 columns into full AVX vectors to preocess our remainder
                let mut columns = [AvxVector::zero(); ROW_COUNT];
                for i in 0..ROW_COUNT {
                    columns[i] = buffer.load_partial3_complex2(partial_remainder_base + len_per_row*i);
                }

                // apply our butterfly function down the columns
                let mid = $butterfly_fn(columns, self);

                // always write the first row without twiddles
                buffer.store_partial3_complex2(mid[0], partial_remainder_base);

                // for the remaining rows, apply twiddle factors and then write back to memory
                for i in 1..ROW_COUNT {
                    let twiddle = final_twiddle_chunk[i - 1];
                    let output = AvxVector::mul_complex(twiddle, mid[i]);
                    buffer.store_partial3_complex2(output, partial_remainder_base + len_per_row*i);
                }
            } else {
                // Load 1 or 2 columns into half vectors to process our remainder. Thankfully, the compiler is smart enough to eliminate this branch on f64, since the partial remainder can only possibly be 1
                let mut columns = [AvxVector::zero(); ROW_COUNT];
                if partial_remainder == 1 {
                    for i in 0..ROW_COUNT {
                        columns[i] = AvxArrayGeneric::<T>::load_partial1_complex2(buffer, partial_remainder_base + len_per_row*i);
                    }
                } else {
                    for i in 0..ROW_COUNT {
                        columns[i] = AvxArrayGeneric::<T>::load_partial2_complex2(buffer, partial_remainder_base + len_per_row*i);
                    }
                }

                // apply our butterfly function down the columns
                let mut mid = $butterfly_fn_lo(columns, self);

                // apply twiddle factors
                for i in 1..ROW_COUNT {
                    mid[i] = AvxVector::mul_complex(final_twiddle_chunk[i - 1].lo(), mid[i]);
                }

                // store output
                if partial_remainder == 1 {
                    for i in 0..ROW_COUNT {
                        AvxArrayMutGeneric::<T>::store_partial1_complex2(buffer, mid[i], partial_remainder_base + len_per_row*i);
                    }
                } else {
                    for i in 0..ROW_COUNT {
                        AvxArrayMutGeneric::<T>::store_partial2_complex2(buffer, mid[i], partial_remainder_base + len_per_row*i);
                    }
                }
            }
        }
    }
)}

macro_rules! mixedradix_transpose{
    ($row_count: expr, $transpose_fn: path, $transpose_fn_lo: path, $($unroll_workaround_index:expr);*, $($remainder3_unroll_workaround_index:expr);*) => (

    // Transpose the input (treated as a nx2 array) into the output (as a 2xn array)
    #[target_feature(enable = "avx")]
    unsafe fn transpose(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        const ROW_COUNT : usize = $row_count;

        let len_per_row = self.len() / ROW_COUNT;
        let chunk_count = len_per_row / T::AvxType::COMPLEX_PER_VECTOR;

        // transpose the scratch as a nx2 array into the buffer as an 2xn array
        for c in 0..chunk_count {
            let input_index_base = c*T::AvxType::COMPLEX_PER_VECTOR;
            let output_index_base = input_index_base * ROW_COUNT;

            // Load rows from the input into registers
            let mut rows : [T::AvxType; ROW_COUNT] = [AvxVector::zero(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                rows[i] = input.load_complex2(input_index_base + len_per_row*i);
            }

            // transpose the rows to the columns
            let transposed = $transpose_fn(rows);

            // store the transposed rows contiguously
            // we are using a macro hack to manually unroll the loop, to work around this rustc bug:
            // https://github.com/rust-lang/rust/issues/71025
            
            // if we don't manually unroll the loop, the compiler will insert unnecessary writes+reads to the stack which tank performance
            // once the compiler bug is fixed, this can be replaced by a "for i in 0..ROW_COUNT" loop
            $( 
                output.store_complex2(transposed[$unroll_workaround_index], output_index_base + T::AvxType::COMPLEX_PER_VECTOR * $unroll_workaround_index);
            )*
        }

        // transpose the remainder
        let input_index_base = chunk_count * T::AvxType::COMPLEX_PER_VECTOR;
        let output_index_base = input_index_base * ROW_COUNT;

        let partial_remainder = len_per_row % T::AvxType::COMPLEX_PER_VECTOR;
        if partial_remainder == 1 {
            // If the partial remainder is 1, there's no transposing to do - just gather from across the rows and store contiguously
            for i in 0..ROW_COUNT {
                let input_cell = input.get_unchecked(input_index_base + len_per_row*i);
                let output_cell = output.get_unchecked_mut(output_index_base + i);
                *output_cell = *input_cell;
            }
        } else if partial_remainder == 2 {
            // If the partial remainder is 2, use the provided transpose_lo function to do a transpose on half-vectors
            let mut rows = [AvxVector::zero(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                rows[i] = AvxArrayGeneric::<T>::load_partial2_complex2(input, input_index_base + len_per_row*i);
            }

            let transposed = $transpose_fn_lo(rows);

            // use the same macro hack as above to unroll the loop
            $( 
                AvxArrayMutGeneric::<T>::store_partial2_complex2(output, transposed[$unroll_workaround_index], output_index_base + <T::AvxType as AvxVector256>::HalfVector::COMPLEX_PER_VECTOR * $unroll_workaround_index);
            )*
        }
        else if partial_remainder == 3 {
            // If the partial remainder is 3, we have to load full vectors, use the full transpose, and then write out a variable number of outputs
            let mut rows = [AvxVector::zero(); ROW_COUNT];
            for i in 0..ROW_COUNT {
                rows[i] = input.load_partial3_complex2(input_index_base + len_per_row*i);
            }

            // transpose the rows to the columns
            let transposed = $transpose_fn(rows);

            // We're going to write constant number of full vectors, and then some constant-sized partial vector
            // Sadly, because of rust limitations, we can't make full_vector_count a const, so we have to cross our fingers that the compiler optimizes it to a constant
            let element_count = 3*ROW_COUNT;
            let full_vector_count = element_count / T::AvxType::COMPLEX_PER_VECTOR;
            let final_remainder_count = element_count % T::AvxType::COMPLEX_PER_VECTOR;

            // write out our full vectors
            // we are using a macro hack to manually unroll the loop, to work around this rustc bug:
            // https://github.com/rust-lang/rust/issues/71025
            
            // if we don't manually unroll the loop, the compiler will insert unnecessary writes+reads to the stack which tank performance
            // once the compiler bug is fixed, this can be replaced by a "for i in 0..full_vector_count" loop
            $( 
                output.store_complex2(transposed[$remainder3_unroll_workaround_index], output_index_base + T::AvxType::COMPLEX_PER_VECTOR * $remainder3_unroll_workaround_index);
            )*

            // write out our partial vector. again, this is a compile-time constant, even if we can't represent that within rust yet
            match final_remainder_count {
                0 => {},
                1 => AvxArrayMutGeneric::<T>::store_partial1_complex2(output, transposed[full_vector_count].lo(), output_index_base + full_vector_count * T::AvxType::COMPLEX_PER_VECTOR),
                2 => AvxArrayMutGeneric::<T>::store_partial2_complex2(output, transposed[full_vector_count].lo(), output_index_base + full_vector_count * T::AvxType::COMPLEX_PER_VECTOR),
                3 => AvxArrayMutGeneric::<T>::store_partial3_complex2(output, transposed[full_vector_count], output_index_base + full_vector_count * T::AvxType::COMPLEX_PER_VECTOR),
                _ => unreachable!(),
            }
        }
    }
)}

pub struct MixedRadix2xnAvx<T: FFTnum> {
    common_data: CommonSimdData<T,T::AvxType>
}
boilerplate_fft_commondata!(MixedRadix2xnAvx);

impl<T: FFTnum> MixedRadix2xnAvx<T> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            common_data: mixedradix_gen_data!(2, inner_fft),
        }
    }
    mixedradix_column_butterflies!(2,
        |columns, _: _| AvxVector::column_butterfly2(columns),
        |columns, _: _| AvxVector::column_butterfly2(columns)
    );
    mixedradix_transpose!(2, 
        AvxVector::transpose2_packed,
        AvxVector::transpose2_packed,
        0;1, 0
    );
    boilerplate_mixedradix!();
}

pub struct MixedRadix3xnAvx<T: FFTnum> {
    twiddles_butterfly3: T::AvxType,
    common_data: CommonSimdData<T,T::AvxType>
}
boilerplate_fft_commondata!(MixedRadix3xnAvx);

impl<T: FFTnum> MixedRadix3xnAvx<T> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(3, inner_fft),
        }
    }
    mixedradix_column_butterflies!(3,
        |columns, this: &Self| AvxVector::column_butterfly3(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector::column_butterfly3(columns, this.twiddles_butterfly3.lo())
    );
    mixedradix_transpose!(3, 
        AvxVector::transpose3_packed,
        AvxVector::transpose3_packed,
        0;1;2, 0;1
    );
    boilerplate_mixedradix!();
}




pub struct MixedRadix4xnAvx<T: FFTnum> {
    twiddles_butterfly4: Rotation90<T::AvxType>,
    common_data: CommonSimdData<T,T::AvxType>,
}
boilerplate_fft_commondata!(MixedRadix4xnAvx);

impl<T: FFTnum> MixedRadix4xnAvx<T> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(4, inner_fft),
        }
    }
    mixedradix_column_butterflies!(4,
        |columns, this: &Self| AvxVector::column_butterfly4(columns, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly4(columns, this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose!(4, 
        AvxVector::transpose4_packed,
        AvxVector::transpose4_packed,
        0;1;2;3, 0;1;2
    );
    boilerplate_mixedradix!();
}




pub struct MixedRadix6xnAvx<T: FFTnum> {
    twiddles_butterfly3: T::AvxType,
    common_data: CommonSimdData<T,T::AvxType>
}
boilerplate_fft_commondata!(MixedRadix6xnAvx);

impl<T: FFTnum> MixedRadix6xnAvx<T> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(6, inner_fft),
        }
    }
    mixedradix_column_butterflies!(6,
        |columns, this: &Self| AvxVector256::column_butterfly6(columns, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector128::column_butterfly6(columns, this.twiddles_butterfly3)
    );
    mixedradix_transpose!(6,
        AvxVector::transpose6_packed,
        AvxVector::transpose6_packed,
        0;1;2;3;4;5, 0;1;2;3
    );
    boilerplate_mixedradix!();
}







pub struct MixedRadix8xnAvx<T: FFTnum> {
    twiddles_butterfly4: Rotation90<T::AvxType>,
    common_data: CommonSimdData<T, T::AvxType>,
}
boilerplate_fft_commondata!(MixedRadix8xnAvx);

impl<T: FFTnum> MixedRadix8xnAvx<T> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        Self {
            twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(8, inner_fft),
        }
    }

    mixedradix_column_butterflies!(8,
        |columns, this: &Self| AvxVector::column_butterfly8(columns, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly8(columns, this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose!(8,
        AvxVector::transpose8_packed,
        AvxVector::transpose8_packed,
        0;1;2;3;4;5;6;7, 0;1;2;3;4;5
    );
    boilerplate_mixedradix!();
}



pub struct MixedRadix9xnAvx<T: FFTnum> {
    twiddles_butterfly9: [T::AvxType; 3],
    twiddles_butterfly9_lo: [T::AvxType; 2],
    twiddles_butterfly3: T::AvxType,
    common_data: CommonSimdData<T, T::AvxType>,
}
boilerplate_fft_commondata!(MixedRadix9xnAvx);

impl<T: FFTnum> MixedRadix9xnAvx<T> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inverse = inner_fft.is_inverse();

        let twiddle1 = AvxVector::broadcast_twiddle(1, 9, inner_fft.is_inverse());
        let twiddle2 = AvxVector::broadcast_twiddle(2, 9, inner_fft.is_inverse());
        let twiddle4 = AvxVector::broadcast_twiddle(4, 9, inner_fft.is_inverse());

        Self {
        	twiddles_butterfly9: [
                AvxVector::broadcast_twiddle(1, 9, inverse),
                AvxVector::broadcast_twiddle(2, 9, inverse),
                AvxVector::broadcast_twiddle(4, 9, inverse),
            ],
            twiddles_butterfly9_lo: [
                AvxVector256::merge(twiddle1, twiddle2),
                AvxVector256::merge(twiddle2, twiddle4),
            ],
        	twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inner_fft.is_inverse()),
            common_data: mixedradix_gen_data!(9, inner_fft),
        }
    }

    mixedradix_column_butterflies!(9, 
        |columns, this: &Self| AvxVector256::column_butterfly9(columns, this.twiddles_butterfly9, this.twiddles_butterfly3),
        |columns, this: &Self| AvxVector128::column_butterfly9(columns, this.twiddles_butterfly9_lo, this.twiddles_butterfly3)
    );
    mixedradix_transpose!(9,
        AvxVector::transpose9_packed,
        AvxVector::transpose9_packed,
        0;1;2;3;4;5;6;7;8, 0;1;2;3;4;5
    );
    boilerplate_mixedradix!();
}






pub struct MixedRadix12xnAvx<T: FFTnum> {
    twiddles_butterfly4: Rotation90<T::AvxType>,
    twiddles_butterfly3: T::AvxType,
    common_data: CommonSimdData<T, T::AvxType>,
}
boilerplate_fft_commondata!(MixedRadix12xnAvx);

impl<T: FFTnum> MixedRadix12xnAvx<T> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inverse = inner_fft.is_inverse();
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inverse),
        	twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            common_data: mixedradix_gen_data!(12, inner_fft),
        }
    }

    mixedradix_column_butterflies!(12, 
        |columns, this: &Self| AvxVector256::column_butterfly12(columns, this.twiddles_butterfly3, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector128::column_butterfly12(columns, this.twiddles_butterfly3, this.twiddles_butterfly4)
    );
    mixedradix_transpose!(12,
        AvxVector::transpose12_packed,
        AvxVector::transpose12_packed,
        0;1;2;3;4;5;6;7;8;9;10;11, 0;1;2;3;4;5;6;7;8
    );
    boilerplate_mixedradix!();
}




pub struct MixedRadix16xnAvx<T: FFTnum> {
    twiddles_butterfly4: Rotation90<T::AvxType>,
    twiddles_butterfly16: [T::AvxType; 2],
    common_data: CommonSimdData<T, T::AvxType>,
}
boilerplate_fft_commondata!(MixedRadix16xnAvx);

impl<T: FFTnum> MixedRadix16xnAvx<T> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inverse = inner_fft.is_inverse();
        Self {
        	twiddles_butterfly4: AvxVector::make_rotation90(inner_fft.is_inverse()),
        	twiddles_butterfly16: [
                AvxVector::broadcast_twiddle(1, 16, inverse),
                AvxVector::broadcast_twiddle(3, 16, inverse),
            ],
            common_data: mixedradix_gen_data!(16, inner_fft),
        }
    }

    mixedradix_column_butterflies!(16, 
        |columns, this: &Self| AvxVector::column_butterfly16(columns, this.twiddles_butterfly16, this.twiddles_butterfly4),
        |columns, this: &Self| AvxVector::column_butterfly16(columns, [this.twiddles_butterfly16[0].lo(), this.twiddles_butterfly16[1].lo()], this.twiddles_butterfly4.lo())
    );
    mixedradix_transpose!(16,
        AvxVector::transpose16_packed,
        AvxVector::transpose16_packed,
        0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15, 0;1;2;3;4;5;6;7;8;9;10;11
    );
    boilerplate_mixedradix!();
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
                    let fft_forward = $struct_name::new(inner_fft_forward).expect("Can't run test because this machine doesn't have the required instruction sets");
                    check_fft_algorithm(&fft_forward, len, false);

                    let inner_fft_inverse = Arc::new(DFT::new(inner_fft_len, true)) as Arc<dyn Fft<f32>>;
                    let fft_inverse = $struct_name::new(inner_fft_inverse).expect("Can't run test because this machine doesn't have the required instruction sets");
                    check_fft_algorithm(&fft_inverse, len, true);
                }
            }
            #[test]
            fn $f64_test_name() {
                for inner_fft_len in 1..32 {
                    let len = inner_fft_len * $inner_count;

                    let inner_fft_forward = Arc::new(DFT::new(inner_fft_len, false)) as Arc<dyn Fft<f64>>;
                    let fft_forward = $struct_name::new(inner_fft_forward).expect("Can't run test because this machine doesn't have the required instruction sets");
                    check_fft_algorithm(&fft_forward, len, false);

                    let inner_fft_inverse = Arc::new(DFT::new(inner_fft_len, true)) as Arc<dyn Fft<f64>>;
                    let fft_inverse = $struct_name::new(inner_fft_inverse).expect("Can't run test because this machine doesn't have the required instruction sets");
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