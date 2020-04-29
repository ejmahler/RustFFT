use std::marker::PhantomData;
use std::arch::x86_64::*;

use num_complex::Complex;

use common::FFTnum;

use ::{Length, IsInverse, Fft};

use ::array_utils::{RawSlice, RawSliceMut};
use super::avx64_utils::{AvxComplexArray64, AvxComplexArrayMut64};
use super::avx64_utils;
use super::avx32_utils;

// Safety: This macro will call `self::perform_fft_f32()` which probably has a #[target_feature(enable = "...")] annotation on it.
// Calling functions with that annotation is unsafe, because it doesn't actually check if the CPU has the required features.
// Callers of this macro must guarantee that users can't even obtain an instance of $struct_name if their CPU doesn't have the required CPU features.
#[allow(unused)]
macro_rules! boilerplate_fft_simd_butterfly {
    ($struct_name:ident, $len:expr) => (
		default impl<T: FFTnum> Fft<T> for $struct_name<T> {
            fn process_inplace_with_scratch(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
				unimplemented!();
            }
            fn process_inplace_multi(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                unimplemented!();
			}
			fn process_with_scratch(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
				unimplemented!();
            }
            fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                unimplemented!();
            }
            fn get_inplace_scratch_len(&self) -> usize {
                unimplemented!();
            }
            fn get_out_of_place_scratch_len(&self) -> usize {
                unimplemented!();
            }
        }
        impl Fft<f64> for $struct_name<f64> {
            fn process_with_scratch(&self, input: &mut [Complex<f64>], output: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) {
                assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
		
				unsafe { self.perform_fft_f64(RawSlice::new(input), RawSliceMut::new(output)) };
            }
            fn process_multi(&self, input: &mut [Complex<f64>], output: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) {
                assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					unsafe { self.perform_fft_f64(RawSlice::new(in_chunk), RawSliceMut::new(out_chunk)) };
				}
            }
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());
        
                unsafe { self.perform_fft_f64(RawSlice::new(buffer), RawSliceMut::new(buffer)) };
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    unsafe { self.perform_fft_f64(RawSlice::new(chunk), RawSliceMut::new(chunk)) };
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
                0
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len
            }
        }
        impl<T> IsInverse for $struct_name<T> {
            #[inline(always)]
            fn is_inverse(&self) -> bool {
                self.inverse
            }
        }
    )
}


macro_rules! gen_butterfly_twiddles_interleaved_columns {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $inverse: expr) => {{
        const FFT_LEN : usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS : usize = $num_rows - 1;
        const TWIDDLE_COLS : usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS : usize = TWIDDLE_COLS / 2;
        const TWIDDLE_VECTOR_COUNT : usize = TWIDDLE_VECTOR_COLS*TWIDDLE_ROWS;
        let mut twiddles = [_mm256_setzero_pd(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index / TWIDDLE_VECTOR_COLS) + 1;
            let x = (index % TWIDDLE_VECTOR_COLS) * 2 + $skip_cols;

            let twiddle_chunk = [
                f64::generate_twiddle_factor(y*(x), FFT_LEN, $inverse),
                f64::generate_twiddle_factor(y*(x+1), FFT_LEN, $inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f64(0);
        }
        twiddles
    }}
}


macro_rules! gen_butterfly_twiddles_separated_columns {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $inverse: expr) => {{
        const FFT_LEN : usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS : usize = $num_rows - 1;
        const TWIDDLE_COLS : usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS : usize = TWIDDLE_COLS / 2;
        const TWIDDLE_VECTOR_COUNT : usize = TWIDDLE_VECTOR_COLS*TWIDDLE_ROWS;
        let mut twiddles = [_mm256_setzero_pd(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index % TWIDDLE_ROWS) + 1;
            let x = (index / TWIDDLE_ROWS) * 2 + $skip_cols;

            let twiddle_chunk = [
                f64::generate_twiddle_factor(y*(x), FFT_LEN, $inverse),
                f64::generate_twiddle_factor(y*(x+1), FFT_LEN, $inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f64(0);
        }
        twiddles
    }}
}



pub struct Butterfly5Avx64<T> {
    twiddles: [__m256d; 3],
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly5Avx64, 5);
impl Butterfly5Avx64<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        let twiddle1 = f64::generate_twiddle_factor(1, 5, inverse);
        let twiddle2 = f64::generate_twiddle_factor(2, 5, inverse);
        Self {
            twiddles: [
                _mm256_set_pd(twiddle1.im, twiddle1.im, twiddle1.re, twiddle1.re),
                _mm256_set_pd(twiddle2.im, twiddle2.im, twiddle2.re, twiddle2.re),
                _mm256_set_pd(-twiddle1.im, -twiddle1.im, twiddle1.re, twiddle1.re),
            ],
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let input0 = _mm256_loadu2_m128d(input.as_ptr() as *const f64, input.as_ptr() as *const f64);
        let input12 = input.load_complex_f64(1);
        let input34 = input.load_complex_f64(3);
        
        // swap elements for inputs 3 and 4
        let input43 = _mm256_permute2f128_pd(input34, input34, 0x01);

        // do some prep work before we can start applying twiddle factors
        let [sum12, diff43] = avx64_utils::column_butterfly2_array_f64([input12, input43]);
        let rotated43 = avx32_utils::Rotate90Config::new_f64(true).rotate90(diff43);

        let [mid14, mid23] = avx64_utils::transpose_2x2_f64([sum12, rotated43]);

        // to compute the first output, compute the sum of all elements. mid14[0] and mid23[0] already have the sum of 1+4 and 2+3 respectively, so if we add them, we'll get the sum of all 4
        let sum1234 = _mm_add_pd(_mm256_castpd256_pd128(mid14), _mm256_castpd256_pd128(mid23));
        let output0 = _mm_add_pd(_mm256_castpd256_pd128(input0), sum1234);
        
        // apply twiddle factors
        let twiddled_outer14 = _mm256_mul_pd(mid14, self.twiddles[0]);
        let twiddled_inner14 = _mm256_mul_pd(mid14, self.twiddles[1]);
        let twiddled14 = _mm256_fmadd_pd(mid23, self.twiddles[1], twiddled_outer14);
        let twiddled23 = _mm256_fmadd_pd(mid23, self.twiddles[2], twiddled_inner14);

        // unpack the data for the last butterfly 2
        let [twiddled12, twiddled43] = avx64_utils::transpose_2x2_f64([twiddled14, twiddled23]);
        let [output12, output43] = avx64_utils::column_butterfly2_array_f64([twiddled12, twiddled43]);

        // swap the elements in output43 before writing them out, and add the first input to everything
        let final12  = _mm256_add_pd(input0, output12);
        let output34 = _mm256_permute2f128_pd(output43, output43, 0x01);
        let final34  = _mm256_add_pd(input0, output34);

        output.store_complex_f64_lo(output0, 0);
        output.store_complex_f64(final12, 1);
        output.store_complex_f64(final34, 3);
    }
}




pub struct MixedRadix64Avx4x2<T> {
    twiddles: [__m256d; 2],
    twiddle_config: avx32_utils::Rotate90Config<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx4x2, 8);
impl MixedRadix64Avx4x2<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(2, 4, 0, inverse),
            twiddle_config: avx32_utils::Rotate90Config::new_f64(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let row0 = input.load_complex_f64(0);
        let row1 = input.load_complex_f64(2);
        let row2 = input.load_complex_f64(4);
        let row3 = input.load_complex_f64(6);

        // Do our butterfly 2's down the columns of a 4x2 array
        let (mid0, mid2) = avx64_utils::column_butterfly2_f64(row0, row2);
        let (mid1, mid3) = avx64_utils::column_butterfly2_f64(row1, row3);

        let mid2_twiddled = avx64_utils::fma::complex_multiply_f64(mid2, self.twiddles[0]);
        let mid3_twiddled = avx64_utils::fma::complex_multiply_f64(mid3, self.twiddles[1]);

        // transpose to a 2x4 array
        let transposed = avx64_utils::transpose_4x2_to_2x4_f64([mid0, mid2_twiddled], [mid1, mid3_twiddled]);

        // butterfly 4's down the transposed array
        let output_rows = avx64_utils::column_butterfly4_f64(transposed, self.twiddle_config);

        output.store_complex_f64(output_rows[0], 0);
        output.store_complex_f64(output_rows[1], 2);
        output.store_complex_f64(output_rows[2], 4);
        output.store_complex_f64(output_rows[3], 6);
    }
}

pub struct MixedRadix64Avx3x3<T> {
    twiddles: [__m256d; 2],
    twiddles_butterfly3: __m256d,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx3x3, 9);
impl MixedRadix64Avx3x3<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(3, 3, 1, inverse),
            twiddles_butterfly3: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 3, inverse)),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 3x4 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second column as full.
        let mut rows0 = [_mm_setzero_pd(); 3];
        let mut rows1 = [_mm256_setzero_pd(); 3];

        for r in 0..3 {
            rows0[r] = input.load_complex_f64_lo(3*r);
            rows1[r] = input.load_complex_f64(3*r+1);
        }

        // do butterfly 4's down the columns
        let mid0 = avx64_utils::fma::column_butterfly3_f64_lo(rows0, self.twiddles_butterfly3);
        let mut mid1 = avx64_utils::fma::column_butterfly3_f64(rows1, self.twiddles_butterfly3);

        // apply twiddle factors
        for n in 1..3 {
            mid1[n] = avx64_utils::fma::complex_multiply_f64(mid1[n], self.twiddles[n - 1]);
        }

        // transpose our 3x3 array
        let (transposed0, transposed1) = avx64_utils::transpose_3x3_f64(mid0, mid1);

        // apply butterfly 3's down the columns
        let output0 = avx64_utils::fma::column_butterfly3_f64_lo(transposed0, self.twiddles_butterfly3);
        let output1 = avx64_utils::fma::column_butterfly3_f64(transposed1, self.twiddles_butterfly3);

        for r in 0..3 {
            output.store_complex_f64_lo(output0[r], 3*r);
            output.store_complex_f64(output1[r], 3*r+1);
        }
    }
}



pub struct MixedRadix64Avx4x3<T> {
    twiddles: [__m256d; 3],
    twiddles_butterfly3: __m256d,
    twiddle_config: avx32_utils::Rotate90Config<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx4x3, 12);
impl MixedRadix64Avx4x3<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 3, 1, inverse),
            twiddles_butterfly3: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 3, inverse)),
            twiddle_config: avx32_utils::Rotate90Config::new_f64(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 3x4 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second column as full.
        let mut rows0 = [_mm_setzero_pd(); 4];
        let mut rows1 = [_mm256_setzero_pd(); 4];

        for n in 0..4 {
            rows0[n] = input.load_complex_f64_lo(n * 3);
            rows1[n] = input.load_complex_f64(n * 3 + 1);
        }

        // do butterfly 4's down the columns
        let mid0 = avx64_utils::column_butterfly4_f64_lo(rows0, self.twiddle_config);
        let mut mid1 = avx64_utils::column_butterfly4_f64(rows1, self.twiddle_config);

        // apply twiddle factors
        for n in 1..4 {
            mid1[n] = avx64_utils::fma::complex_multiply_f64(mid1[n], self.twiddles[n - 1]);
        }

        // transpose our 3x4 array to a 4x3 array
        let (transposed0, transposed1) = avx64_utils::transpose_3x4_to_4x3_f64(mid0, mid1);

        // apply butterfly 3's down the columns
        let output0 = avx64_utils::fma::column_butterfly3_f64(transposed0, self.twiddles_butterfly3);
        let output1 = avx64_utils::fma::column_butterfly3_f64(transposed1, self.twiddles_butterfly3);

        for r in 0..3 {
            output.store_complex_f64(output0[r], 4*r);
            output.store_complex_f64(output1[r], 4*r+2);
        }
    }
}


pub struct MixedRadix64Avx4x4<T> {
    twiddles: [__m256d; 6],
    twiddle_config: avx32_utils::Rotate90Config<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx4x4, 16);
impl MixedRadix64Avx4x4<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 4, 0, inverse),
            twiddle_config: avx32_utils::Rotate90Config::new_f64(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let mut rows0 = [_mm256_setzero_pd(); 4];
        let mut rows1 = [_mm256_setzero_pd(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex_f64(4*r);
            rows1[r] = input.load_complex_f64(4*r + 2);
        }

        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        let mut mid0 = avx64_utils::column_butterfly4_f64(rows0, self.twiddle_config);
        let mut mid1 = avx64_utils::column_butterfly4_f64(rows1, self.twiddle_config);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = avx64_utils::fma::complex_multiply_f64(mid0[r], self.twiddles[2*r - 2]);
            mid1[r] = avx64_utils::fma::complex_multiply_f64(mid1[r], self.twiddles[2*r - 1]);
        }

        // Transpose our 4x4 array
        let (transposed0, transposed1) = avx64_utils::transpose_4x4_f64(mid0, mid1);

        // Butterfly 4's down columns of the transposed array
        let output0 = avx64_utils::column_butterfly4_f64(transposed0, self.twiddle_config);
        let output1 = avx64_utils::column_butterfly4_f64(transposed1, self.twiddle_config);

        for r in 0..4 {
            output.store_complex_f64(output0[r], 4*r);
            output.store_complex_f64(output1[r], 4*r+2);
        }
    }
}

pub struct MixedRadix64Avx3x6<T> {
    twiddles: [__m256d; 5],
    twiddles_butterfly3: __m256d,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx3x6, 18);
impl MixedRadix64Avx3x6<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(6, 3, 1, inverse),
            twiddles_butterfly3: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 3, inverse)),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 3x6 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second column as full.
        let mut rows0 = [_mm_setzero_pd(); 6];
        let mut rows1 = [_mm256_setzero_pd(); 6];
        for n in 0..6 {
            rows0[n] = input.load_complex_f64_lo(n * 3);
            rows1[n] = input.load_complex_f64(n * 3 + 1);
        }

        // do butterfly 6's down the columns
        let mid0 = avx64_utils::fma::column_butterfly6_f64_lo(rows0, self.twiddles_butterfly3);
        let mut mid1 = avx64_utils::fma::column_butterfly6_f64(rows1, self.twiddles_butterfly3);

        // apply twiddle factors
        for n in 1..6 {
            mid1[n] = avx64_utils::fma::complex_multiply_f64(mid1[n], self.twiddles[n - 1]);
        }

        // transpose our 3x4 array to a 4x3 array
        let (transposed0, transposed1, transposed2) = avx64_utils::transpose_3x6_to_6x3_f64(mid0, mid1);

        // apply butterfly 3's down the columns
        let output0 = avx64_utils::fma::column_butterfly3_f64(transposed0, self.twiddles_butterfly3);
        let output1 = avx64_utils::fma::column_butterfly3_f64(transposed1, self.twiddles_butterfly3);
        let output2 = avx64_utils::fma::column_butterfly3_f64(transposed2, self.twiddles_butterfly3);

        for r in 0..3 {
            output.store_complex_f64(output0[r], 6*r);
            output.store_complex_f64(output1[r], 6*r+2);
            output.store_complex_f64(output2[r], 6*r+4);
        }
    }
}

pub struct MixedRadix64Avx4x6<T> {
    twiddles: [__m256d; 9],
    twiddles_butterfly3: __m256d,
    twiddle_config: avx32_utils::Rotate90Config<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx4x6, 24);
impl MixedRadix64Avx4x6<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 6, 0, inverse),
            twiddles_butterfly3: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 3, inverse)),
            twiddle_config: avx32_utils::Rotate90Config::new_f64(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let mut rows0 = [_mm256_setzero_pd(); 4];
        let mut rows1 = [_mm256_setzero_pd(); 4];
        let mut rows2 = [_mm256_setzero_pd(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex_f64(6*r);
            rows1[r] = input.load_complex_f64(6*r + 2);
            rows2[r] = input.load_complex_f64(6*r + 4);
        }

        // We're going to treat our input as a 6x4 2d array. First, do 6 butterfly 4's down the columns of that array.
        let mut mid0 = avx64_utils::column_butterfly4_f64(rows0, self.twiddle_config);
        let mut mid1 = avx64_utils::column_butterfly4_f64(rows1, self.twiddle_config);
        let mut mid2 = avx64_utils::column_butterfly4_f64(rows2, self.twiddle_config);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = avx64_utils::fma::complex_multiply_f64(mid0[r], self.twiddles[3*r - 3]);
            mid1[r] = avx64_utils::fma::complex_multiply_f64(mid1[r], self.twiddles[3*r - 2]);
            mid2[r] = avx64_utils::fma::complex_multiply_f64(mid2[r], self.twiddles[3*r - 1]);
        }
        
        // Transpose our 6x4 array
        let (transposed0, transposed1) = avx64_utils::transpose_6x4_to_4x6_f64(mid0, mid1, mid2);

        // Butterfly 6's down columns of the transposed array
        let output0 = avx64_utils::fma::column_butterfly6_f64(transposed0, self.twiddles_butterfly3);
        let output1 = avx64_utils::fma::column_butterfly6_f64(transposed1, self.twiddles_butterfly3);

        for r in 0..6 {
            output.store_complex_f64(output0[r], 4*r);
            output.store_complex_f64(output1[r], 4*r+2);
        }
    }
}

pub struct MixedRadix64Avx3x9<T> {
    twiddles: [__m256d; 8],
    twiddles_butterfly9: [__m256d; 3],
    twiddles_butterfly9_lo: [__m256d; 2],
    twiddles_butterfly3: __m256d,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx3x9, 27);
impl MixedRadix64Avx3x9<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        let twiddles_butterfly9_lo = [
            f64::generate_twiddle_factor(1, 9, inverse),
            f64::generate_twiddle_factor(2, 9, inverse),
            f64::generate_twiddle_factor(2, 9, inverse),
            f64::generate_twiddle_factor(4, 9, inverse),
        ];


        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(3, 9, 1, inverse),
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
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 3x6 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second 2 columns as full.
        let mut rows0 = [_mm_setzero_pd(); 3];
        for n in 0..3 {
            rows0[n] = input.load_complex_f64_lo(n*9);
        }
        let mid0 = avx64_utils::fma::column_butterfly3_f64_lo(rows0, self.twiddles_butterfly3);

        // we're going to load our input as a 3x6 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second 2 columns as full.
        let mut rows1 = [_mm256_setzero_pd(); 3];
        let mut rows2 = [_mm256_setzero_pd(); 3];
        for n in 0..3 {
            rows1[n] = input.load_complex_f64(n*9 + 1);
            rows2[n] = input.load_complex_f64(n*9 + 3);
        }
        let mut mid1 = avx64_utils::fma::column_butterfly3_f64(rows1, self.twiddles_butterfly3);
        let mut mid2 = avx64_utils::fma::column_butterfly3_f64(rows2, self.twiddles_butterfly3);
        for r in 1..3 {
            mid1[r] = avx64_utils::fma::complex_multiply_f64(mid1[r], self.twiddles[4*r - 4]);
            mid2[r] = avx64_utils::fma::complex_multiply_f64(mid2[r], self.twiddles[4*r - 3]);
        }

        // we're going to load our input as a 3x6 array. We have to load 3 columns, which is a little awkward
        let mut rows3 = [_mm256_setzero_pd(); 3];
        let mut rows4 = [_mm256_setzero_pd(); 3];
        for n in 0..3 {
            rows3[n] = input.load_complex_f64(n*9 + 5);
            rows4[n] = input.load_complex_f64(n*9 + 7);
        }
        let mut mid3 = avx64_utils::fma::column_butterfly3_f64(rows3, self.twiddles_butterfly3);
        let mut mid4 = avx64_utils::fma::column_butterfly3_f64(rows4, self.twiddles_butterfly3);
        for r in 1..3 {
            mid3[r] = avx64_utils::fma::complex_multiply_f64(mid3[r], self.twiddles[4*r - 2]);
            mid4[r] = avx64_utils::fma::complex_multiply_f64(mid4[r], self.twiddles[4*r - 1]);
        }

        // transpose our 9x3 array to a 3x9 array
        let (transposed0, transposed1) = avx64_utils::transpose_9x3_to_3x9_f64(mid0, mid1, mid2, mid3, mid4);

        // apply butterfly 9's down the columns
        let output0 = avx64_utils::fma::column_butterfly9_f64_lo(transposed0, self.twiddles_butterfly9_lo, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex_f64_lo(output0[r*3], 9*r);
            output.store_complex_f64_lo(output0[r*3+1], 9*r+3);
            output.store_complex_f64_lo(output0[r*3+2], 9*r+6);
        }

        let output1 = avx64_utils::fma::column_butterfly9_f64(transposed1, self.twiddles_butterfly9, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex_f64(output1[r*3], 9*r+1);
            output.store_complex_f64(output1[r*3+1], 9*r+4);
            output.store_complex_f64(output1[r*3+2], 9*r+7);
        }
    }
}



pub struct MixedRadix64Avx4x8<T> {
    twiddles: [__m256d; 12],
    twiddle_config: avx32_utils::Rotate90Config<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx4x8, 32);
impl MixedRadix64Avx4x8<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 8, 0, inverse),
            twiddle_config: avx32_utils::Rotate90Config::new_f64(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        // We can't fit the whole problem into AVX registers at once, so we'll have to spill some things.
        // By computing half of the problem and then not referencing any of it for a while, we're making it easy for the compiler to decide what to spill
        let mut rows0 = [_mm256_setzero_pd(); 4];
        let mut rows1 = [_mm256_setzero_pd(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex_f64(8*r);
            rows1[r] = input.load_complex_f64(8*r + 2);
        }
        let mut mid0 = avx64_utils::column_butterfly4_f64(rows0, self.twiddle_config);
        let mut mid1 = avx64_utils::column_butterfly4_f64(rows1, self.twiddle_config);
        for r in 1..4 {
            mid0[r] = avx64_utils::fma::complex_multiply_f64(mid0[r], self.twiddles[4 * r - 4]);
            mid1[r] = avx64_utils::fma::complex_multiply_f64(mid1[r], self.twiddles[4 * r - 3]);
        }

        // One half is done, so the compiler can spill everything above this. Now do the other set of columns
        let mut rows2 = [_mm256_setzero_pd(); 4];
        let mut rows3 = [_mm256_setzero_pd(); 4];
        for r in 0..4 {
            rows2[r] = input.load_complex_f64(8*r + 4);
            rows3[r] = input.load_complex_f64(8*r + 6);
        }
        let mut mid2 = avx64_utils::column_butterfly4_f64(rows2, self.twiddle_config);
        let mut mid3 = avx64_utils::column_butterfly4_f64(rows3, self.twiddle_config);
        for r in 1..4 {
            mid2[r] = avx64_utils::fma::complex_multiply_f64(mid2[r], self.twiddles[4 * r - 2]);
            mid3[r] = avx64_utils::fma::complex_multiply_f64(mid3[r], self.twiddles[4 * r - 1]);
        }

        // Transpose our 8x4 array to a 4x8 array
        let (transposed0, transposed1) = avx64_utils::transpose_8x4_to_4x8_f64(mid0, mid1, mid2, mid3);

        // Do 4 butterfly 8's down columns of the transposed array
        // Same thing as above - Do the half of the butterfly 8's separately to give the compiler a better hint about what to spill
        let output0 = avx64_utils::fma::column_butterfly8_f64(transposed0, self.twiddle_config);
        for r in 0..8 {
            output.store_complex_f64(output0[r], 4*r);
        }
        let output1 = avx64_utils::fma::column_butterfly8_f64(transposed1, self.twiddle_config);
        for r in 0..8 {
            output.store_complex_f64(output1[r], 4*r + 2);
        }
    }
}


pub struct MixedRadix64Avx6x6<T> {
    twiddles: [__m256d; 15],
    twiddles_butterfly3: __m256d,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx6x6, 36);
impl MixedRadix64Avx6x6<f64> {
    #[inline]
    pub fn new(inverse: bool) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inverse) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inverse: bool) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_separated_columns!(6, 6, 0, inverse),
            twiddles_butterfly3: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1, 3, inverse)),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 6x6 array
        let mut rows0 = [_mm256_setzero_pd(); 6];
        for n in 0..6 {
            rows0[n] = input.load_complex_f64(n*6);
        }
        let mut mid0 = avx64_utils::fma::column_butterfly6_f64(rows0, self.twiddles_butterfly3);
        for r in 1..6 {
            mid0[r] = avx64_utils::fma::complex_multiply_f64(mid0[r], self.twiddles[r - 1]);
        }

        // we're going to load our input as a 6x6 array
        let mut rows1 = [_mm256_setzero_pd(); 6];
        for n in 0..6 {
            rows1[n] = input.load_complex_f64(n*6+2);
        }
        let mut mid1 = avx64_utils::fma::column_butterfly6_f64(rows1, self.twiddles_butterfly3);
        for r in 1..6 {
            mid1[r] = avx64_utils::fma::complex_multiply_f64(mid1[r], self.twiddles[r + 4]);
        }

        // we're going to load our input as a 6x6 array
        let mut rows2 = [_mm256_setzero_pd(); 6];
        for n in 0..6 {
            rows2[n] = input.load_complex_f64(n*6+4);
        }
        let mut mid2 = avx64_utils::fma::column_butterfly6_f64(rows2, self.twiddles_butterfly3);
        for r in 1..6 {
            mid2[r] = avx64_utils::fma::complex_multiply_f64(mid2[r], self.twiddles[r + 9]);
        }


        // transpose our 6x6 array
        let (transposed0, transposed1, transposed2) = avx64_utils::transpose_6x6_f64(mid0, mid1, mid2);


        // apply butterfly 6's down the columns
        let output0 = avx64_utils::fma::column_butterfly6_f64(transposed0, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex_f64(output0[r*2], 12*r);
            output.store_complex_f64(output0[r*2+1], 12*r+6);
        }

        let output1 = avx64_utils::fma::column_butterfly6_f64(transposed1, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex_f64(output1[r*2], 12*r+2);
            output.store_complex_f64(output1[r*2+1], 12*r+8);
        }

        let output2 = avx64_utils::fma::column_butterfly6_f64(transposed2, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex_f64(output2[r*2], 12*r+4);
            output.store_complex_f64(output2[r*2+1], 12*r+10);
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_fft_algorithm;

    macro_rules! test_avx_butterfly {
        ($test_name:ident, $struct_name:ident, $size:expr) => (
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(false).expect("Can't run test because this machine doesn't have the required instruction sets");
                check_fft_algorithm(&butterfly, $size, false);

                let butterfly_inverse = $struct_name::new(true).expect("Can't run test because this machine doesn't have the required instruction sets");
                check_fft_algorithm(&butterfly_inverse, $size, true);
            }
        )
    }

    test_avx_butterfly!(test_avx_butterfly5_f64, Butterfly5Avx64, 5);
    test_avx_butterfly!(test_avx_mixedradix4x2_f64, MixedRadix64Avx4x2, 8);
    test_avx_butterfly!(test_avx_mixedradix3x3_f64, MixedRadix64Avx3x3, 9);
    test_avx_butterfly!(test_avx_mixedradix4x3_f64, MixedRadix64Avx4x3, 12);
    test_avx_butterfly!(test_avx_mixedradix4x4_f64, MixedRadix64Avx4x4, 16);
    test_avx_butterfly!(test_avx_mixedradix3x6_f64, MixedRadix64Avx3x6, 18);
    test_avx_butterfly!(test_avx_mixedradix4x6_f64, MixedRadix64Avx4x6, 24);
    test_avx_butterfly!(test_avx_mixedradix3x9_f64, MixedRadix64Avx3x9, 27);
    test_avx_butterfly!(test_avx_mixedradix4x8_f64, MixedRadix64Avx4x8, 32);
    test_avx_butterfly!(test_avx_mixedradix6x6_f64, MixedRadix64Avx6x6, 36);
}
