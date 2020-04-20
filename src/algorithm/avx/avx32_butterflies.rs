use std::marker::PhantomData;
use std::arch::x86_64::*;

use num_complex::Complex;
use num_traits::Zero;

use common::FFTnum;

use ::{Length, IsInverse, Fft};

use ::array_utils::{RawSlice, RawSliceMut};
use super::avx32_utils::{AvxComplexArrayf32, AvxComplexArrayMutf32};
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
        impl Fft<f32> for $struct_name<f32> {
            fn process_with_scratch(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], _scratch: &mut [Complex<f32>]) {
                assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
		
				unsafe { self.perform_fft_f32(RawSlice::new(input), RawSliceMut::new(output)) };
            }
            fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], _scratch: &mut [Complex<f32>]) {
                assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					unsafe { self.perform_fft_f32(RawSlice::new(in_chunk), RawSliceMut::new(out_chunk)) };
				}
            }
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<f32>], _scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());
        
                unsafe { self.perform_fft_f32(RawSlice::new(buffer), RawSliceMut::new(buffer)) };
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<f32>], _scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    unsafe { self.perform_fft_f32(RawSlice::new(chunk), RawSliceMut::new(chunk)) };
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


pub struct MixedRadixAvx4x2<T> {
    twiddles: __m256,
    twiddle_config: avx32_utils::Rotate90OddConfig,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x2, 8);
impl MixedRadixAvx4x2<f32> {
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
            f32::generate_twiddle_factor(1, 8, inverse),
            f32::generate_twiddle_factor(2, 8, inverse),
            f32::generate_twiddle_factor(3, 8, inverse),
        ];
        Self {
            twiddles: twiddle_array.load_complex_f32(0),
            twiddle_config: avx32_utils::Rotate90OddConfig::get_from_inverse(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let row0 = input.load_complex_f32(0);
        let row1 = input.load_complex_f32(4);

        // Do our butterfly 2's down the columns
        let (intermediate0, intermediate1_pretwiddle) = avx32_utils::column_butterfly2_f32(row0, row1);

        // Apply the size-8 twiddle factors
        let intermediate1 = avx32_utils::fma::complex_multiply_f32(intermediate1_pretwiddle, self.twiddles);

        // Rearrange the data before we do our butterfly 4s. This swaps the last 2 elements of butterfly0 with the first two elements of butterfly1
        // The result is that we can then do a 4x butterfly 2, apply twiddles, use unpack instructions to transpose to the final output, then do another 4x butterfly 2
        let permuted0 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x20);
        let permuted1 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x31);

        // Do the first set of butterfly 2's
        let (postbutterfly0, postbutterfly1_pretwiddle) = avx32_utils::column_butterfly2_f32(permuted0, permuted1);

        // Which negative we blend in depends on whether we're forward or inverse
        let postbutterfly1 = self.twiddle_config.rotate90_oddelements(postbutterfly1_pretwiddle);

        // use unpack instructions to transpose, and to prepare for the final butterfly 2's
        let unpermuted0 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x20);
        let unpermuted1 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x31);
        let unpacked0 = _mm256_unpacklo_ps(unpermuted0, unpermuted1);
        let unpacked1 = _mm256_unpackhi_ps(unpermuted0, unpermuted1);
        let swapped0 = _mm256_permute_ps(unpacked0, 0xD8);
        let swapped1 = _mm256_permute_ps(unpacked1, 0xD8);

        let (output0, output1) = avx32_utils::column_butterfly2_f32(swapped0, swapped1);

        output.store_complex_f32(0, output0);
        output.store_complex_f32(4, output1);
    }
}


pub struct MixedRadixAvx3x3<T> {
    twiddles: __m256,
    twiddles_butterfly3: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx3x3, 9);
impl MixedRadixAvx3x3<f32> {
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
        let twiddles = [
            f32::generate_twiddle_factor(1, 9, inverse),
            f32::generate_twiddle_factor(2, 9, inverse),
            f32::generate_twiddle_factor(2, 9, inverse),
            f32::generate_twiddle_factor(4, 9, inverse),
        ];
        Self {
            twiddles: twiddles.load_complex_f32(0),
            twiddles_butterfly3: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // we're going to load these elements in a peculiar way. instead of loading a row into the first 3 element of each register and leaving the last element empty
        // we're leaving the first element empty and putting the data in the last 3 elements. this will let us do 3 total complex multiplies instead of 4.

        let input0_lo = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let input0_hi = input.load_complex_f32_lo(1);
        let input0 = _mm256_insertf128_ps(_mm256_castps128_ps256(input0_lo), input0_hi, 0x1);
        let input1 = input.load_complex_f32(2);
        let input2 = input.load_complex_f32(5);

        // We're going to treat our input as a 3x4 2d array. First, do 3 butterfly 4's down the columns of that array.
        let [mid0, mid1, mid2] = avx32_utils::fma::column_butterfly3_f32([input0, input1, input2], self.twiddles_butterfly3);
        
        // merge the twiddle-able data into a single avx vector
        let twiddle_data = _mm256_permute2f128_ps(mid1, mid2, 0x31);
        let twiddled = avx32_utils::fma::complex_multiply_f32(twiddle_data, self.twiddles);

        // Transpose our 3x3 array. we're using the 4x4 transpose with an empty bottom row, which will result in an empty last column
        // but it turns out that it'll make our packing process later simpler if we duplicate the second row into the last row
        // which will result in duplicating the second column into the last column after the transpose
        let permute0 = _mm256_permute2f128_ps(mid0, mid2, 0x20);
        let permute1 = _mm256_permute2f128_ps(mid1, mid1, 0x20);
        let permute2 = _mm256_permute2f128_ps(mid0, twiddled, 0x31);
        let permute3 = _mm256_permute2f128_ps(twiddled, twiddled, 0x20);

        let (_, transposed0) = avx32_utils::unpack_complex_f32(permute0, permute1);
        let (transposed1, transposed2) = avx32_utils::unpack_complex_f32(permute2, permute3);

        // more size 3 buterflies
        let output_rows = avx32_utils::fma::column_butterfly3_f32([transposed0, transposed1, transposed2], self.twiddles_butterfly3);

        // the elements of row 1 are in pretty much the worst possible order, most of these instructions will be about getting them into the right order
        let swapped1 = _mm256_permute_ps(output_rows[1], 0x4E); // swap even and odd complex numbers
        let packed1 = _mm256_permute2f128_ps(swapped1, output_rows[2], 0x21);
        output.store_complex_f32(4, packed1);

        // merge just the high element of swapped_lo into the high element of row 0
        let empty = _mm256_setzero_ps();
        let swapped1_lo = _mm256_extractf128_ps(swapped1, 0x0);
        let zero_swapped1_lo = _mm256_insertf128_ps(empty, swapped1_lo, 0x1);
        let packed0 = _mm256_blend_ps(output_rows[0], zero_swapped1_lo, 0xC0);
        output.store_complex_f32(0, packed0);

        // rather than trying to rearrange the last element with AVX registers, just write the register to the stack and copy normally
        let mut dump_output2 = [Complex::zero(); 4];
        dump_output2.store_complex_f32(0, output_rows[2]);
        output.store(dump_output2[2], 8);
    }
}

pub struct MixedRadixAvx4x3<T> {
    twiddles: [__m256; 3],
    twiddles_butterfly3: __m256,
    twiddle_config: avx32_utils::Rotate90Config<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl MixedRadixAvx4x3<f32> {
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
        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(1, 12, inverse),
            f32::generate_twiddle_factor(2, 12, inverse),
            Complex{ re: 1.0, im: 0.0 },
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(2, 12, inverse),
            f32::generate_twiddle_factor(4, 12, inverse),
            Complex{ re: 1.0, im: 0.0 },
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(3, 12, inverse),
            f32::generate_twiddle_factor(6, 12, inverse),
        ];
        Self {
            twiddles: [
                twiddles.load_complex_f32(0),
                twiddles.load_complex_f32(4),
                twiddles.load_complex_f32(8),
            ],
            twiddles_butterfly3: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // we're going to load these elements in a peculiar way. instead of loading a row into the first 3 element of each register and leaving the last element empty
        // we're leaving the first element empty and putting the data in the last 3 elements. this will save us a complex multiply.
        
        // for everything but the first element, we can do overlapping reads. for the first element, an "overlapping read" would have us reading from index -1, so instead we have to shuffle some data around
        let input0_lo = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let input0_hi = input.load_complex_f32_lo(1);
        let input_rows = [
            _mm256_insertf128_ps(_mm256_castps128_ps256(input0_lo), input0_hi, 0x1),
            input.load_complex_f32(2),
            input.load_complex_f32(5),
            input.load_complex_f32(8),
        ];

        // butterfly 4's down the columns
        let mut mid = avx32_utils::column_butterfly4_f32(input_rows, self.twiddle_config);

        // Multiply in our twiddle factors. the first one will be normal, but for the second one, merge the twiddle-able parts of the vectors into a single vector, so that we can do one multiply instead of 2
        mid[1] = avx32_utils::fma::complex_multiply_f32(mid[1], self.twiddles[0]);
        mid[2] = avx32_utils::fma::complex_multiply_f32(mid[2], self.twiddles[1]);
        mid[3] = avx32_utils::fma::complex_multiply_f32(mid[3], self.twiddles[2]);

        // Transpose our 3x4 array into a 4x3
        let [_, transposed0, transposed1, transposed2] = avx32_utils::transpose_4x4_f32(mid);

        // Do 4 butterfly 4's down the columns of our transposed array
        let output_rows = avx32_utils::fma::column_butterfly3_f32([transposed0, transposed1, transposed2], self.twiddles_butterfly3); 

        output.store_complex_f32(0, output_rows[0]);
        output.store_complex_f32(1 * 4, output_rows[1]);
        output.store_complex_f32(2 * 4, output_rows[2]);
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x3, 12);

pub struct MixedRadixAvx4x4<T> {
    twiddles: [__m256; 3],
    twiddle_config: avx32_utils::Rotate90Config<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl MixedRadixAvx4x4<f32> {
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
        let mut twiddles = [_mm256_setzero_ps(); 3];
        for index in 0..3 {
            let y = index + 1;

            let twiddle_chunk = [
                Complex{ re: 1.0, im: 0.0 },
                f32::generate_twiddle_factor(y*1, 16, inverse),
                f32::generate_twiddle_factor(y*2, 16, inverse),
                f32::generate_twiddle_factor(y*3, 16, inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f32(0);
        }
        Self {
            twiddles,
            twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // Manually unrolling this loop because writing a "for r in 0..4" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        let rows = [
            input.load_complex_f32(0),
            input.load_complex_f32(4),
            input.load_complex_f32(8),
            input.load_complex_f32(12),
        ];

        // We're going to treat our input as a 4x4 2d array. First, do 4 butterfly 4's down the columns of that array.
        let mut mid = avx32_utils::column_butterfly4_f32(rows, self.twiddle_config);

        // apply twiddle factors
        for r in 1..4 {
            mid[r] = avx32_utils::fma::complex_multiply_f32(mid[r], self.twiddles[r - 1]);
        }

        // Transpose our 4x4 array
        let transposed = avx32_utils::transpose_4x4_f32(mid);

        // Do 4 butterfly 4's down the columns of our transposed array
        let output_rows = avx32_utils::column_butterfly4_f32(transposed, self.twiddle_config);

        // Manually unrolling this loop because writing a "for r in 0..4" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        output.store_complex_f32(0, output_rows[0]);
        output.store_complex_f32(4, output_rows[1]);
        output.store_complex_f32(8, output_rows[2]);
        output.store_complex_f32(12,output_rows[3]);
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x4, 16);

pub struct MixedRadixAvx4x6<T> {
    twiddles: [__m256; 5],
    twiddles_butterfly3: __m256,
    twiddle_config: avx32_utils::Rotate90Config<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl MixedRadixAvx4x6<f32> {
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
        let mut twiddles = [_mm256_setzero_ps(); 5];
        for index in 0..5 {
            let y = index + 1;

            let twiddle_chunk = [
                Complex{ re: 1.0, im: 0.0 },
                f32::generate_twiddle_factor(y*1, 24, inverse),
                f32::generate_twiddle_factor(y*2, 24, inverse),
                f32::generate_twiddle_factor(y*3, 24, inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f32(0);
        }
        Self {
            twiddles,
            twiddles_butterfly3: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // Manually unrolling this loop because writing a "for r in 0..6" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        let rows = [
            input.load_complex_f32(0),
            input.load_complex_f32(4),
            input.load_complex_f32(8),
            input.load_complex_f32(12),
            input.load_complex_f32(16),
            input.load_complex_f32(20),
        ];

        // We're going to treat our input as a 4x6 2d array. First, do 4 butterfly 6's down the columns of that array.
        let mut mid = avx32_utils::fma::column_butterfly6_f32(rows, self.twiddles_butterfly3);

        // apply twiddle factors
        for r in 1..6 {
            mid[r] = avx32_utils::fma::complex_multiply_f32(mid[r], self.twiddles[r - 1]);
        }

        // Transpose our 6x4 array into a 4x6. Sadly this will leave us with 2 garbage columns on the end
        let (transposed0, transposed1) = avx32_utils::transpose_4x6_to_6x4_f32(mid);

        // Do 4 butterfly 4's down the columns of our transposed array
        let output0 = avx32_utils::column_butterfly4_f32(transposed0, self.twiddle_config);
        let output1 = avx32_utils::column_butterfly4_f32(transposed1, self.twiddle_config);

        // the upper two elements of output1 are empty, so only store half the data for it
        for r in 0..4 {
            output.store_complex_f32(6*r, output0[r]);
            output.store_complex_f32_lo(output1[r], r*6 + 4);
        }
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x6, 24);

pub struct MixedRadixAvx3x9<T> {
    twiddles: [__m256; 4],
    twiddles_butterfly9: [__m256; 3],
    twiddles_butterfly3: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx3x9, 27);
impl MixedRadixAvx3x9<f32> {
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
        let mut twiddles = [_mm256_setzero_ps(); 4];
        for index in 0..4 {
            let y = (index / 2) + 1;
            let x = (index % 2) * 4 + 1;

            let twiddle_chunk = [
                f32::generate_twiddle_factor(y*(x), 27, inverse),
                f32::generate_twiddle_factor(y*(x+1), 27, inverse),
                f32::generate_twiddle_factor(y*(x+2), 27, inverse),
                f32::generate_twiddle_factor(y*(x+3), 27, inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f32(0);
        }
        Self {
            twiddles,
            twiddles_butterfly9: [
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 9, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(2, 9, inverse)),
                avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(4, 9, inverse)),
            ],
            twiddles_butterfly3: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // we're going to load our data in a peculiar way. we're going to load the first column on its own as a column of __m128.
        // it's faster to just load the first 2 columns into these m128s than trying to worry about masks, etc, so the second column will piggyback along and we just won't use it
        let mut rows0 = [_mm_setzero_ps(); 3];
        let mut rows1 = [_mm256_setzero_ps(); 3];
        let mut rows2 = [_mm256_setzero_ps(); 3];
        for r in 0..3 {
            rows0[r] = input.load_complex_f32_lo(r * 9);
            rows1[r] = input.load_complex_f32(r * 9 + 1);
            rows2[r] = input.load_complex_f32(r * 9 + 5);
        }

        // butterfly 3s down the columns
        let mid0 = avx32_utils::fma::column_butterfly3_f32_lo(rows0, self.twiddles_butterfly3);
        let mut mid1 = avx32_utils::fma::column_butterfly3_f32(rows1, self.twiddles_butterfly3);
        let mut mid2 = avx32_utils::fma::column_butterfly3_f32(rows2, self.twiddles_butterfly3);
        
        // apply twiddle factors
        mid1[1] = avx32_utils::fma::complex_multiply_f32(mid1[1], self.twiddles[0]);
        mid2[1] = avx32_utils::fma::complex_multiply_f32(mid2[1], self.twiddles[1]);
        mid1[2] = avx32_utils::fma::complex_multiply_f32(mid1[2], self.twiddles[2]);
        mid2[2] = avx32_utils::fma::complex_multiply_f32(mid2[2], self.twiddles[3]);

        // transpose 9x3 to 3x9. this will be a little awkward because of row0 containing garbage data
        let transposed = avx32_utils::transpose_9x3_to_3x9_emptycolumn1_f32(mid0, mid1, mid2);

        // butterfly 9s down the rows
        let output_rows = avx32_utils::fma::column_butterfly9_f32(transposed, self.twiddles_butterfly9, self.twiddles_butterfly3);

        // we can't directly write our data to the output, because it's in an unpacked state with the last column enpty. we have to pack the data in, which will involve a lot of reshuffling
        let final_mask = avx32_utils::RemainderMask::new_f32(3);

        let packed0 = avx32_utils::pack_3x4_4x3_f32([output_rows[0], output_rows[1], output_rows[2], output_rows[3]]);
        output.store_complex_f32(0, packed0[0]);
        output.store_complex_f32(4, packed0[1]);
        output.store_complex_f32(8, packed0[2]);

        let packed1 = avx32_utils::pack_3x4_4x3_f32([output_rows[4], output_rows[5], output_rows[6], output_rows[7]]);
        output.store_complex_f32(12, packed1[0]);
        output.store_complex_f32(16, packed1[1]);
        output.store_complex_f32(20, packed1[2]);

        output.store_complex_remainder_f32(final_mask, output_rows[8], 24);
    }
}

pub struct MixedRadixAvx4x8<T> {
    twiddles: [__m256; 6],
    twiddles_butterfly8: __m256,
    twiddle_config: avx32_utils::Rotate90Config<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl MixedRadixAvx4x8<f32> {
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
        let mut twiddles = [_mm256_setzero_ps(); 6];
        for index in 0..6 {
            let y = (index / 2) + 1;
            let x = (index % 2) * 4;

            let twiddle_chunk = [
                f32::generate_twiddle_factor(y*(x), 32, inverse),
                f32::generate_twiddle_factor(y*(x+1), 32, inverse),
                f32::generate_twiddle_factor(y*(x+2), 32, inverse),
                f32::generate_twiddle_factor(y*(x+3), 32, inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f32(0);
        }

        Self {
            twiddles,
            twiddles_butterfly8: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 8, inverse)),
            twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let mut rows0 = [_mm256_setzero_ps(); 4];
        let mut rows1 = [_mm256_setzero_ps(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex_f32(8*r);
            rows1[r] = input.load_complex_f32(8*r + 4);
        }

        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        let mut mid0 = avx32_utils::column_butterfly4_f32(rows0, self.twiddle_config);
        let mut mid1 = avx32_utils::column_butterfly4_f32(rows1, self.twiddle_config);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = avx32_utils::fma::complex_multiply_f32(mid0[r], self.twiddles[2*r - 2]);
            mid1[r] = avx32_utils::fma::complex_multiply_f32(mid1[r], self.twiddles[2*r - 1]);
        }

        // Transpose our 8x4 array to an 8x4 array
        let transposed = avx32_utils::transpose_8x4_to_4x8_f32(mid0, mid1);

        // Do 4 butterfly 8's down the columns of our transpsed array
        let output_rows = avx32_utils::fma::column_butterfly8_f32(transposed, self.twiddles_butterfly8, self.twiddle_config);

        // Manually unrolling this loop because writing a "for r in 0..8" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        output.store_complex_f32(0, output_rows[0]);
        output.store_complex_f32(1 * 4, output_rows[1]);
        output.store_complex_f32(2 * 4, output_rows[2]);
        output.store_complex_f32(3 * 4, output_rows[3]);
        output.store_complex_f32(4 * 4, output_rows[4]);
        output.store_complex_f32(5 * 4, output_rows[5]);
        output.store_complex_f32(6 * 4, output_rows[6]);
        output.store_complex_f32(7 * 4, output_rows[7]);
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x8, 32);

pub struct MixedRadixAvx4x12<T> {
    twiddles: [__m256; 9],
    twiddles_butterfly3: __m256,
    twiddle_config: avx32_utils::Rotate90Config<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl MixedRadixAvx4x12<f32> {
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
        let mut twiddles = [_mm256_setzero_ps(); 9];
        for index in 0..9 {
            let y = (index / 3) + 1;
            let x = (index % 3) * 4;

            let twiddle_chunk = [
                f32::generate_twiddle_factor(y*(x), 48, inverse),
                f32::generate_twiddle_factor(y*(x+1), 48, inverse),
                f32::generate_twiddle_factor(y*(x+2), 48, inverse),
                f32::generate_twiddle_factor(y*(x+3), 48, inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f32(0);
        }
        Self {
            twiddles,
            twiddles_butterfly3: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let mut rows0 = [_mm256_setzero_ps(); 4];
        let mut rows1 = [_mm256_setzero_ps(); 4];
        let mut rows2 = [_mm256_setzero_ps(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex_f32(12*r);
            rows1[r] = input.load_complex_f32(12*r + 4);
            rows2[r] = input.load_complex_f32(12*r + 8);
        }

        // We're going to treat our input as a 12x4 2d array. First, do 12 butterfly 4's down the columns of that array.
        let mut mid0 = avx32_utils::column_butterfly4_f32(rows0, self.twiddle_config);
        let mut mid1 = avx32_utils::column_butterfly4_f32(rows1, self.twiddle_config);
        let mut mid2 = avx32_utils::column_butterfly4_f32(rows2, self.twiddle_config);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = avx32_utils::fma::complex_multiply_f32(mid0[r], self.twiddles[3*r - 3]);
            mid1[r] = avx32_utils::fma::complex_multiply_f32(mid1[r], self.twiddles[3*r - 2]);
            mid2[r] = avx32_utils::fma::complex_multiply_f32(mid2[r], self.twiddles[3*r - 1]);
        }

        // Transpose our 12x4 array into a 4x12.
        let transposed = avx32_utils::transpose_12x4_to_4x12_f32(mid0, mid1, mid2);

        // Do 4 butterfly 12's down the columns of our transposed array
        let output_rows = avx32_utils::fma::column_butterfly12_f32(transposed, self.twiddles_butterfly3, self.twiddle_config);

        // Manually unrolling this loop because writing a "for r in 0..12" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        output.store_complex_f32(0, output_rows[0]);
        output.store_complex_f32(4, output_rows[1]);
        output.store_complex_f32(8, output_rows[2]);
        output.store_complex_f32(12, output_rows[3]);
        output.store_complex_f32(16, output_rows[4]);
        output.store_complex_f32(20, output_rows[5]);
        output.store_complex_f32(24, output_rows[6]);
        output.store_complex_f32(28, output_rows[7]);
        output.store_complex_f32(32, output_rows[8]);
        output.store_complex_f32(36, output_rows[9]);
        output.store_complex_f32(40, output_rows[10]);
        output.store_complex_f32(44, output_rows[11]);
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x12, 48);

pub struct MixedRadixAvx8x8<T> {
    twiddles: [__m256; 14],
    twiddles_butterfly8: __m256,
    twiddle_config: avx32_utils::Rotate90Config<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl MixedRadixAvx8x8<f32> {
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
        let mut twiddles = [_mm256_setzero_ps(); 14];
        for index in 0..14 {
            // we're going to do one entire column of AVX twiddles, then a second entire column
            let y = (index % 7) + 1;
            let x = (index / 7) * 4;

            let twiddle_chunk = [
                f32::generate_twiddle_factor(y*(x), 64, inverse),
                f32::generate_twiddle_factor(y*(x+1), 64, inverse),
                f32::generate_twiddle_factor(y*(x+2), 64, inverse),
                f32::generate_twiddle_factor(y*(x+3), 64, inverse),
            ];
            twiddles[index] = twiddle_chunk.load_complex_f32(0);
        }

        Self {
            twiddles,
            twiddles_butterfly8: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 8, inverse)),
            twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // We're going to treat our input as a 8x8 2d array. First, do 8 butterfly 8's down the columns of that array.
        // We can't fit the whole problem into AVX registers at once, so we'll have to spill some things.
        // By computing a sizeable chunk and not referencing any of it for a while, we're making it easy for the compiler to decide what to spill
        let mut rows0 = [_mm256_setzero_ps(); 8];
        for r in 0..8 {
            rows0[r] = input.load_complex_f32(8*r);
        }
        let mut mid0 = avx32_utils::fma::column_butterfly8_f32(rows0, self.twiddles_butterfly8, self.twiddle_config);
        for r in 1..8 {
            mid0[r] = avx32_utils::fma::complex_multiply_f32(mid0[r],  self.twiddles[r - 1]);
        }
        
        // One half is done, so the compiler can spill everything above this. Now do the other set of columns
        let mut rows1 = [_mm256_setzero_ps(); 8];
        for r in 0..8 {
            rows1[r] = input.load_complex_f32(8*r + 4);
        }
        let mut mid1 = avx32_utils::fma::column_butterfly8_f32(rows1, self.twiddles_butterfly8, self.twiddle_config);
        for r in 1..8 {
            mid1[r] = avx32_utils::fma::complex_multiply_f32(mid1[r],  self.twiddles[r - 1 + 7]);
        }

        // Transpose our 8x8 array
        let (transposed0, transposed1)  = avx32_utils::transpose_8x8_f32(mid0, mid1);

        // Do 8 butterfly 8's down the columns of our transposed array, and store the results
        // Same thing as above - Do the half of the butterfly 8's separately to give the compiler a better hint about what to spill
        let output0 = avx32_utils::fma::column_butterfly8_f32(transposed0, self.twiddles_butterfly8, self.twiddle_config);
        for r in 0..8 {
            output.store_complex_f32(8*r, output0[r]);
        }

        let output1 = avx32_utils::fma::column_butterfly8_f32(transposed1, self.twiddles_butterfly8, self.twiddle_config);
        for r in 0..8 {
            output.store_complex_f32(8*r + 4, output1[r]);
        }
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx8x8, 64);

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

    test_avx_butterfly!(test_avx_mixedradix4x2, MixedRadixAvx4x2, 8);
    test_avx_butterfly!(test_avx_mixedradix3x3, MixedRadixAvx3x3, 9);
    test_avx_butterfly!(test_avx_mixedradix4x3, MixedRadixAvx4x3, 12);
    test_avx_butterfly!(test_avx_mixedradix4x4, MixedRadixAvx4x4, 16);
    test_avx_butterfly!(test_avx_mixedradix4x6, MixedRadixAvx4x6, 24);
    test_avx_butterfly!(test_avx_mixedradix3x9, MixedRadixAvx3x9, 27);
    test_avx_butterfly!(test_avx_mixedradix4x8, MixedRadixAvx4x8, 32);
    test_avx_butterfly!(test_avx_mixedradix4x12,MixedRadixAvx4x12,48);
    test_avx_butterfly!(test_avx_mixedradix8x8, MixedRadixAvx8x8, 64);
}
