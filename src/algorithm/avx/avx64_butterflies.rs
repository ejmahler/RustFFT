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
        let twiddle_array = [
            Complex{ re: 1.0, im: 0.0 },
            f64::generate_twiddle_factor(1, 8, inverse),
            f64::generate_twiddle_factor(2, 8, inverse),
            f64::generate_twiddle_factor(3, 8, inverse),
        ];
        Self {
            twiddles: [twiddle_array.load_complex_f64(0),twiddle_array.load_complex_f64(2)],
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
        let twiddle_array = [
            f64::generate_twiddle_factor(1, 12, inverse),
            f64::generate_twiddle_factor(2, 12, inverse),
            f64::generate_twiddle_factor(2, 12, inverse),
            f64::generate_twiddle_factor(4, 12, inverse),
            f64::generate_twiddle_factor(3, 12, inverse),
            f64::generate_twiddle_factor(6, 12, inverse),
        ];
        Self {
            twiddles: [twiddle_array.load_complex_f64(0), twiddle_array.load_complex_f64(2), twiddle_array.load_complex_f64(4)],
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
        let mut twiddles = [_mm256_setzero_pd(); 6];
        for index in 0..6 {
            let y = (index / 2) + 1;
            let x = (index % 2) * 2;

            let twiddle_chunk = [
                f64::generate_twiddle_factor(y*(x), 16, inverse),
                f64::generate_twiddle_factor(y*(x+1), 16, inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f64(0);
        }
        Self {
            twiddles,
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
        let mut twiddles = [_mm256_setzero_pd(); 9];
        let mut index = 0;
        for y in 1..4 {
            for x in (0..6).step_by(2) {
                let twiddle_chunk = [
                    f64::generate_twiddle_factor(y*(x), 24, inverse),
                    f64::generate_twiddle_factor(y*(x+1), 24, inverse),
                ];
                twiddles[index] = twiddle_chunk.load_complex_f64(0);
                index += 1;
            }
        }
        Self {
            twiddles,
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

pub struct MixedRadix64Avx4x4SplitRealImaginary<T> {
    twiddles_real: [__m256d; 3],
    twiddles_imag: [__m256d; 3],
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadix64Avx4x4SplitRealImaginary, 16);
impl MixedRadix64Avx4x4SplitRealImaginary<f64> {
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
        let twiddle_array = [
            Complex{ re: 1.0, im: 0.0 },
            f64::generate_twiddle_factor(1, 16, inverse),
            f64::generate_twiddle_factor(2, 16, inverse),
            f64::generate_twiddle_factor(3, 16, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f64::generate_twiddle_factor(2, 16, inverse),
            f64::generate_twiddle_factor(4, 16, inverse),
            f64::generate_twiddle_factor(6, 16, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f64::generate_twiddle_factor(3, 16, inverse),
            f64::generate_twiddle_factor(6, 16, inverse),
            f64::generate_twiddle_factor(9, 16, inverse),
        ];

        let combined_twiddles = [
            twiddle_array.load_complex_f64(0),
            twiddle_array.load_complex_f64(2),
            twiddle_array.load_complex_f64(4),
            twiddle_array.load_complex_f64(6),
            twiddle_array.load_complex_f64(8),
            twiddle_array.load_complex_f64(10),
        ];
        Self {
            twiddles_real: [
                _mm256_unpacklo_pd(combined_twiddles[0], combined_twiddles[1]),
                _mm256_unpacklo_pd(combined_twiddles[2], combined_twiddles[3]),
                _mm256_unpacklo_pd(combined_twiddles[4], combined_twiddles[5]),
            ],
            twiddles_imag: [
                _mm256_unpackhi_pd(combined_twiddles[0], combined_twiddles[1]),
                _mm256_unpackhi_pd(combined_twiddles[2], combined_twiddles[3]),
                _mm256_unpackhi_pd(combined_twiddles[4], combined_twiddles[5]),
            ],
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
        let row4 = input.load_complex_f64(8);
        let row5 = input.load_complex_f64(10);
        let row6 = input.load_complex_f64(12);
        let row7 = input.load_complex_f64(14);

        let row0_real = _mm256_unpacklo_pd(row0, row1);
        let row0_imag = _mm256_unpackhi_pd(row0, row1);
        let row1_real = _mm256_unpacklo_pd(row2, row3);
        let row1_imag = _mm256_unpackhi_pd(row2, row3);
        let row2_real = _mm256_unpacklo_pd(row4, row5);
        let row2_imag = _mm256_unpackhi_pd(row4, row5);
        let row3_real = _mm256_unpacklo_pd(row6, row7);
        let row3_imag = _mm256_unpackhi_pd(row6, row7);

        // Do our butterfly 4's down the columns of a 4x4 array
        let (mid0_real, mid0_imag, mid1_real, mid1_imag, mid2_real, mid2_imag, mid3_real, mid3_imag) = 
            avx64_utils::column_butterfly4_split_f64(row0_real, row0_imag, row1_real, row1_imag, row2_real, row2_imag, row3_real, row3_imag);

        // Apply twiddle factors
        let (mid1_real_twiddled, mid1_imag_twiddled) = avx64_utils::fma::complex_multiply_split_f64(mid1_real, mid1_imag, self.twiddles_real[0], self.twiddles_imag[0]);
        let (mid2_real_twiddled, mid2_imag_twiddled) = avx64_utils::fma::complex_multiply_split_f64(mid2_real, mid2_imag, self.twiddles_real[1], self.twiddles_imag[1]);
        let (mid3_real_twiddled, mid3_imag_twiddled) = avx64_utils::fma::complex_multiply_split_f64(mid3_real, mid3_imag, self.twiddles_real[2], self.twiddles_imag[2]);

        // Transpose our 4x4 array. but our unpacks to split the reals and imaginaries left the data in a weird order - and we can partially fix the wrong order by passing 
        let (transposed0_real, transposed1_real, transposed2_real, transposed3_real) = avx64_utils::transpose_4x4_real_f64(mid0_real, mid1_real_twiddled, mid2_real_twiddled, mid3_real_twiddled);
        let (transposed0_imag, transposed1_imag, transposed2_imag, transposed3_imag) = avx64_utils::transpose_4x4_real_f64(mid0_imag, mid1_imag_twiddled, mid2_imag_twiddled, mid3_imag_twiddled);

        // Butterfly 4's down columns of the transposed array
        let (output0_real, output0_imag, output1_real, output1_imag, output2_real, output2_imag, output3_real, output3_imag) = 
            avx64_utils::column_butterfly4_split_f64(transposed0_real, transposed0_imag, transposed1_real, transposed1_imag, transposed2_real, transposed2_imag, transposed3_real, transposed3_imag);

        let packed0 = _mm256_unpacklo_pd(output0_real, output0_imag);
        let packed1 = _mm256_unpackhi_pd(output0_real, output0_imag);
        output.store_complex_f64(packed0, 0);
        output.store_complex_f64(packed1, 2);

        let packed2 = _mm256_unpacklo_pd(output1_real, output1_imag);
        let packed3 = _mm256_unpackhi_pd(output1_real, output1_imag);
        output.store_complex_f64(packed2, 4);
        output.store_complex_f64(packed3, 6);

        let packed4 = _mm256_unpacklo_pd(output2_real, output2_imag);
        let packed5 = _mm256_unpackhi_pd(output2_real, output2_imag);
        output.store_complex_f64(packed4, 8);
        output.store_complex_f64(packed5, 10);

        let packed6 = _mm256_unpacklo_pd(output3_real, output3_imag);
        let packed7 = _mm256_unpackhi_pd(output3_real, output3_imag);
        output.store_complex_f64(packed6, 12);
        output.store_complex_f64(packed7, 14);
    }
}

pub struct MixedRadix64Avx4x8<T> {
    twiddles: [__m256d; 12],
    twiddles_butterfly8: __m256d,
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
        let mut twiddles = [_mm256_setzero_pd(); 12];
        for index in 0..12 {
            let y = (index / 4) + 1;
            let x = (index % 4) * 2;

            let twiddle_chunk = [
                f64::generate_twiddle_factor(y*(x), 32, inverse),
                f64::generate_twiddle_factor(y*(x+1), 32, inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f64(0);
        }
        Self {
            twiddles,
            twiddles_butterfly8: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1,8,inverse)),
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
        let output0 = avx64_utils::fma::column_butterfly8_f64(transposed0, self.twiddles_butterfly8, self.twiddle_config);
        for r in 0..8 {
            output.store_complex_f64(output0[r], 4*r);
        }
        let output1 = avx64_utils::fma::column_butterfly8_f64(transposed1, self.twiddles_butterfly8, self.twiddle_config);
        for r in 0..8 {
            output.store_complex_f64(output1[r], 4*r + 2);
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
                use algorithm::MixedRadixSmall;
                use algorithm::DFT;
                use std::sync::Arc;

                let inner6 : Arc<dyn Fft<f32>> = Arc::new(DFT::new(6, false));
                let inner4 = Arc::new(DFT::new(4, false));

                let control = MixedRadixSmall::new(inner6, inner4);

                check_fft_algorithm(&control, 24, false);

                let butterfly = $struct_name::new(false).expect("Can't run test because this machine doesn't have the required instruction sets");
                check_fft_algorithm(&butterfly, $size, false);

                let butterfly_inverse = $struct_name::new(true).expect("Can't run test because this machine doesn't have the required instruction sets");
                check_fft_algorithm(&butterfly_inverse, $size, true);
            }
        )
    }

    test_avx_butterfly!(test_avx_mixedradix4x2_f64, MixedRadix64Avx4x2, 8);
    test_avx_butterfly!(test_avx_mixedradix4x3_f64, MixedRadix64Avx4x3, 12);
    test_avx_butterfly!(test_avx_mixedradix4x4_f64, MixedRadix64Avx4x4, 16);
    //test_avx_butterfly!(test_avx_mixedradix4x4_split_f64, MixedRadix64Avx4x4SplitRealImaginary, 16); // Currently broken -- saving code for reference on splitting reals/imaginaries
    test_avx_butterfly!(test_avx_mixedradix4x6_f64, MixedRadix64Avx4x6, 24);
    test_avx_butterfly!(test_avx_mixedradix4x8_f64, MixedRadix64Avx4x8, 32);
}
