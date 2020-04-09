use std::marker::PhantomData;
use std::arch::x86_64::*;

use num_complex::Complex;

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
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x2, 8);

pub struct MixedRadixAvx4x3<T> {
    twiddles: [__m256; 2],
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
            f32::generate_twiddle_factor(1, 12, inverse),
            f32::generate_twiddle_factor(2, 12, inverse),
            f32::generate_twiddle_factor(3, 12, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(2, 12, inverse),
            f32::generate_twiddle_factor(4, 12, inverse),
            f32::generate_twiddle_factor(6, 12, inverse),
        ];
        Self {
            twiddles: [
                twiddles.load_complex_f32(0),
                twiddles.load_complex_f32(4),
            ],
            twiddles_butterfly3: avx32_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            twiddle_config: avx32_utils::Rotate90Config::new_f32(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let input0 = input.load_complex_f32(0);
        let input1 = input.load_complex_f32(1 * 4);
        let input2 = input.load_complex_f32(2 * 4);

        // We're going to treat our input as a 3x4 2d array. First, do 3 butterfly 4's down the columns of that array.
        let (mid0, mid1, mid2) = avx32_utils::fma::column_butterfly3_f32(input0, input1, input2, self.twiddles_butterfly3);

        // Multiply in our twiddle factors
        let mid1_twiddled = avx32_utils::fma::complex_multiply_f32(mid1, self.twiddles[0]);
        let mid2_twiddled = avx32_utils::fma::complex_multiply_f32(mid2, self.twiddles[1]);

        // Transpose our 4x3 array into a 3x4
        // this will leave us with an empty column. after the butterfly 4's, we'll need to do some data reshuffling,
        // and that reshuffling will be simpler if we duplicate the second column into the third, and shift the third to the fourth. we can do that for free by shifting the rows we pass to the transpose
        // the computed data and runtime cost will be the same either way, so why not?
        let transposed = avx32_utils::transpose_4x4_f32([mid0, mid1_twiddled, mid1_twiddled, mid2_twiddled]);

        // Do 4 butterfly 4's down the columns of our transposed array
        let output_rows = avx32_utils::column_butterfly4_f32(transposed, self.twiddle_config);

        // Theoretically, we could do masked writes of our 4 output registers directly to memory, but we'd be doing overlapping writes, and benchmarking shows that it's incredibly slow
        // instead, aided by our column duplication above, we can pack our data into just 3 non-overlapping registers

        // the data we want is scattered across 4 different registers, so these first 3 instructions bring them all into the same register, in the right order
        let intermediate0 = _mm256_permute2f128_ps(output_rows[0], output_rows[2], 0x13);
        let intermediate1 = _mm256_permute2f128_ps(output_rows[1], output_rows[3], 0x02);

        let shuffled = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(intermediate0), _mm256_castps_pd(intermediate1), 0x05));

        // Now that the "suffled" vector contains all the elements gathere from the various rows, we can pack everything together
        let packed1 = _mm256_permute2f128_ps(output_rows[1], output_rows[2], 0x21);
        let packed0 = _mm256_permute2f128_ps(output_rows[0], shuffled, 0x30); 
        let packed2 = _mm256_permute2f128_ps(shuffled, output_rows[3], 0x30); 

        // the last element in each column is empty. We're going to do partially overlapping writes to stomp over the previous value.
        // TODO: See if it's faster to rearrange the data back into 3 rows that we write non-overlapping. Might require AVX2
        output.store_complex_f32(0, packed0);
        output.store_complex_f32(1 * 4, packed1);
        output.store_complex_f32(2 * 4, packed2);
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
    test_avx_butterfly!(test_avx_mixedradix4x3, MixedRadixAvx4x3, 12);
    test_avx_butterfly!(test_avx_mixedradix4x4, MixedRadixAvx4x4, 16);
    test_avx_butterfly!(test_avx_mixedradix4x6, MixedRadixAvx4x6, 24);
    test_avx_butterfly!(test_avx_mixedradix4x8, MixedRadixAvx4x8, 32);
    test_avx_butterfly!(test_avx_mixedradix4x12,MixedRadixAvx4x12,48);
    test_avx_butterfly!(test_avx_mixedradix8x8, MixedRadixAvx8x8, 64);
}
