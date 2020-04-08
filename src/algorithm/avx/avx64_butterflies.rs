use std::marker::PhantomData;
use std::arch::x86_64::*;

use num_complex::Complex;

use common::FFTnum;

use ::{Length, IsInverse, Fft};

use ::array_utils::{RawSlice, RawSliceMut};
use super::avx64_utils::{AvxComplexArray64, AvxComplexArrayMut64};
use super::avx64_utils;

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
    twiddle_config: avx64_utils::Rotate90Config,
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
            twiddle_config: avx64_utils::Rotate90Config::get_from_inverse(inverse),
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
        let (transposed0, transposed1) = avx64_utils::transpose_2x2_f64(mid0, mid2_twiddled);
        let (transposed2, transposed3) = avx64_utils::transpose_2x2_f64(mid1, mid3_twiddled);

        let (output0, output1, output2, output3) = avx64_utils::column_butterfly4_f64(transposed0,transposed1, transposed2, transposed3, self.twiddle_config);

        output.store_complex_f64(output0, 0);
        output.store_complex_f64(output1, 2);
        output.store_complex_f64(output2, 4);
        output.store_complex_f64(output3, 6);
    }
}

pub struct MixedRadix64Avx4x4<T> {
    twiddles: [__m256d; 6],
    twiddle_config: avx64_utils::Rotate90Config,
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
        Self {
            twiddles: [
                twiddle_array.load_complex_f64(0),
                twiddle_array.load_complex_f64(2),
                twiddle_array.load_complex_f64(4),
                twiddle_array.load_complex_f64(6),
                twiddle_array.load_complex_f64(8),
                twiddle_array.load_complex_f64(10),
                ],
            twiddle_config: avx64_utils::Rotate90Config::get_from_inverse(inverse),
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

        // Do our butterfly 4's down the columns of a 4x4 array
        let (mid0, mid2, mid4, mid6) = avx64_utils::column_butterfly4_f64(row0,row2, row4, row6, self.twiddle_config);
        let (mid1, mid3, mid5, mid7) = avx64_utils::column_butterfly4_f64(row1,row3, row5, row7, self.twiddle_config);

        // Apply twiddle factors
        let mid2_twiddled = avx64_utils::fma::complex_multiply_f64(mid2, self.twiddles[0]);
        let mid3_twiddled = avx64_utils::fma::complex_multiply_f64(mid3, self.twiddles[1]);
        let mid4_twiddled = avx64_utils::fma::complex_multiply_f64(mid4, self.twiddles[2]);
        let mid5_twiddled = avx64_utils::fma::complex_multiply_f64(mid5, self.twiddles[3]);
        let mid6_twiddled = avx64_utils::fma::complex_multiply_f64(mid6, self.twiddles[4]);
        let mid7_twiddled = avx64_utils::fma::complex_multiply_f64(mid7, self.twiddles[5]);

        // Transpose our 4x4 array
        let (transposed0, transposed2) = avx64_utils::transpose_2x2_f64(mid0, mid2_twiddled);
        let (transposed4, transposed6) = avx64_utils::transpose_2x2_f64(mid1, mid3_twiddled);
        let (transposed1, transposed3) = avx64_utils::transpose_2x2_f64(mid4_twiddled, mid6_twiddled);
        let (transposed5, transposed7) = avx64_utils::transpose_2x2_f64(mid5_twiddled, mid7_twiddled);

        // Butterfly 4's down columns of the transposed array
        let (output0, output2, output4, output6) = avx64_utils::column_butterfly4_f64(transposed0, transposed2, transposed4, transposed6, self.twiddle_config);
        let (output1, output3, output5, output7) = avx64_utils::column_butterfly4_f64(transposed1, transposed3, transposed5, transposed7, self.twiddle_config);

        output.store_complex_f64(output0, 0);
        output.store_complex_f64(output1, 2);
        output.store_complex_f64(output2, 4);
        output.store_complex_f64(output3, 6);
        output.store_complex_f64(output4, 8);
        output.store_complex_f64(output5, 10);
        output.store_complex_f64(output6, 12);
        output.store_complex_f64(output7, 14);
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
        let (transposed0_real, transposed1_real, transposed2_real, transposed3_real) = avx64_utils::transpose_4x4_f64(mid0_real, mid1_real_twiddled, mid2_real_twiddled, mid3_real_twiddled);
        let (transposed0_imag, transposed1_imag, transposed2_imag, transposed3_imag) = avx64_utils::transpose_4x4_f64(mid0_imag, mid1_imag_twiddled, mid2_imag_twiddled, mid3_imag_twiddled);

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
    twiddle_config: avx64_utils::Rotate90Config,
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
        let twiddle_array = [
            Complex{ re: 1.0, im: 0.0 },
            f64::generate_twiddle_factor(1, 32, inverse),
            f64::generate_twiddle_factor(2, 32, inverse),
            f64::generate_twiddle_factor(3, 32, inverse),
            f64::generate_twiddle_factor(4, 32, inverse),
            f64::generate_twiddle_factor(5, 32, inverse),
            f64::generate_twiddle_factor(6, 32, inverse),
            f64::generate_twiddle_factor(7, 32, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f64::generate_twiddle_factor(2, 32, inverse),
            f64::generate_twiddle_factor(4, 32, inverse),
            f64::generate_twiddle_factor(6, 32, inverse),
            f64::generate_twiddle_factor(8, 32, inverse),
            f64::generate_twiddle_factor(10, 32, inverse),
            f64::generate_twiddle_factor(12, 32, inverse),
            f64::generate_twiddle_factor(14, 32, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f64::generate_twiddle_factor(3, 32, inverse),
            f64::generate_twiddle_factor(6, 32, inverse),
            f64::generate_twiddle_factor(9, 32, inverse),
            f64::generate_twiddle_factor(12, 32, inverse),
            f64::generate_twiddle_factor(15, 32, inverse),
            f64::generate_twiddle_factor(18, 32, inverse),
            f64::generate_twiddle_factor(21, 32, inverse),
        ];
        Self {
            twiddles: [
                twiddle_array.load_complex_f64(0),
                twiddle_array.load_complex_f64(2),
                twiddle_array.load_complex_f64(4),
                twiddle_array.load_complex_f64(6),
                twiddle_array.load_complex_f64(8),
                twiddle_array.load_complex_f64(10),
                twiddle_array.load_complex_f64(12),
                twiddle_array.load_complex_f64(14),
                twiddle_array.load_complex_f64(16),
                twiddle_array.load_complex_f64(18),
                twiddle_array.load_complex_f64(20),
                twiddle_array.load_complex_f64(22),
            ],
            twiddles_butterfly8: avx64_utils::broadcast_complex_f64(f64::generate_twiddle_factor(1,8,inverse)),
            twiddle_config: avx64_utils::Rotate90Config::get_from_inverse(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let row00 = input.load_complex_f64(0);
        let row02 = input.load_complex_f64(2);
        let row04 = input.load_complex_f64(4);
        let row06 = input.load_complex_f64(6);
        let row10 = input.load_complex_f64(8);
        let row12 = input.load_complex_f64(10);
        let row14 = input.load_complex_f64(12);
        let row16 = input.load_complex_f64(14);
        let row20 = input.load_complex_f64(16);
        let row22 = input.load_complex_f64(18);
        let row24 = input.load_complex_f64(20);
        let row26 = input.load_complex_f64(22);
        let row30 = input.load_complex_f64(24);
        let row32 = input.load_complex_f64(26);
        let row34 = input.load_complex_f64(28);
        let row36 = input.load_complex_f64(30);

        // Do our butterfly 4's down the columns of a 8x4 array
        let (mid00, mid10, mid20, mid30) = avx64_utils::column_butterfly4_f64(row00,row10, row20, row30, self.twiddle_config);
        let (mid02, mid12, mid22, mid32) = avx64_utils::column_butterfly4_f64(row02,row12, row22, row32, self.twiddle_config);
        let (mid04, mid14, mid24, mid34) = avx64_utils::column_butterfly4_f64(row04,row14, row24, row34, self.twiddle_config);
        let (mid06, mid16, mid26, mid36) = avx64_utils::column_butterfly4_f64(row06,row16, row26, row36, self.twiddle_config);

        // Apply twiddle factors
        let mid10_twiddled = avx64_utils::fma::complex_multiply_f64(mid10, self.twiddles[0]);
        let mid12_twiddled = avx64_utils::fma::complex_multiply_f64(mid12, self.twiddles[1]);
        let mid14_twiddled = avx64_utils::fma::complex_multiply_f64(mid14, self.twiddles[2]);
        let mid16_twiddled = avx64_utils::fma::complex_multiply_f64(mid16, self.twiddles[3]);
        let mid20_twiddled = avx64_utils::fma::complex_multiply_f64(mid20, self.twiddles[4]);
        let mid22_twiddled = avx64_utils::fma::complex_multiply_f64(mid22, self.twiddles[5]);
        let mid24_twiddled = avx64_utils::fma::complex_multiply_f64(mid24, self.twiddles[6]);
        let mid26_twiddled = avx64_utils::fma::complex_multiply_f64(mid26, self.twiddles[7]);
        let mid30_twiddled = avx64_utils::fma::complex_multiply_f64(mid30, self.twiddles[8]);
        let mid32_twiddled = avx64_utils::fma::complex_multiply_f64(mid32, self.twiddles[9]);
        let mid34_twiddled = avx64_utils::fma::complex_multiply_f64(mid34, self.twiddles[10]);
        let mid36_twiddled = avx64_utils::fma::complex_multiply_f64(mid36, self.twiddles[11]);

        // Transpose our 8x4 array to a 4x8 array
        let (transposed00, transposed10) = avx64_utils::transpose_2x2_f64(mid00, mid10_twiddled);
        let (transposed20, transposed30) = avx64_utils::transpose_2x2_f64(mid02, mid12_twiddled);
        let (transposed40, transposed50) = avx64_utils::transpose_2x2_f64(mid04, mid14_twiddled);
        let (transposed60, transposed70) = avx64_utils::transpose_2x2_f64(mid06, mid16_twiddled);

        let (transposed01, transposed11) = avx64_utils::transpose_2x2_f64(mid20_twiddled, mid30_twiddled);
        let (transposed21, transposed31) = avx64_utils::transpose_2x2_f64(mid22_twiddled, mid32_twiddled);
        let (transposed41, transposed51) = avx64_utils::transpose_2x2_f64(mid24_twiddled, mid34_twiddled);
        let (transposed61, transposed71) = avx64_utils::transpose_2x2_f64(mid26_twiddled, mid36_twiddled);

        // Butterfly 8's down columns of the transposed array
        let (output00, output10, output20, output30, output40, output50, output60, output70) = avx64_utils::fma::column_butterfly8_f64(
            transposed00, transposed10, transposed20, transposed30, transposed40, transposed50, transposed60, transposed70, self.twiddles_butterfly8, self.twiddle_config
        );
        let (output01, output11, output21, output31, output41, output51, output61, output71) = avx64_utils::fma::column_butterfly8_f64(
            transposed01, transposed11, transposed21, transposed31, transposed41, transposed51, transposed61, transposed71, self.twiddles_butterfly8, self.twiddle_config
        );

        output.store_complex_f64(output00, 0);
        output.store_complex_f64(output01, 2);
        output.store_complex_f64(output10, 4);
        output.store_complex_f64(output11, 6);
        output.store_complex_f64(output20, 8);
        output.store_complex_f64(output21, 10);
        output.store_complex_f64(output30, 12);
        output.store_complex_f64(output31, 14);
        output.store_complex_f64(output40, 16);
        output.store_complex_f64(output41, 18);
        output.store_complex_f64(output50, 20);
        output.store_complex_f64(output51, 22);
        output.store_complex_f64(output60, 24);
        output.store_complex_f64(output61, 26);
        output.store_complex_f64(output70, 28);
        output.store_complex_f64(output71, 30);
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

                //let butterfly_inverse = $struct_name::new(true).expect("Can't run test because this machine doesn't have the required instruction sets");
                //check_fft_algorithm(&butterfly_inverse, $size, true);
            }
        )
    }

    test_avx_butterfly!(test_avx_mixedradix4x2_f64, MixedRadix64Avx4x2, 8);
    test_avx_butterfly!(test_avx_mixedradix4x4_f64, MixedRadix64Avx4x4, 16);
    test_avx_butterfly!(test_avx_mixedradix4x4_split_f64, MixedRadix64Avx4x4SplitRealImaginary, 16);
    test_avx_butterfly!(test_avx_mixedradix4x8_f64, MixedRadix64Avx4x8, 32);
}
