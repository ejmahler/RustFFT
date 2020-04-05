use std::marker::PhantomData;
use std::arch::x86_64::*;

use num_complex::Complex;

use common::FFTnum;

use ::{Length, IsInverse, Fft};

use ::array_utils::{RawSlice, RawSliceMut};
use super::avx_utils::{AvxComplexArrayf32, AvxComplexArrayMutf32};
use super::avx_utils;

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
    twiddle_config: avx_utils::Rotate90OddConfig,
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
            twiddle_config: avx_utils::Rotate90OddConfig::get_from_inverse(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let row0 = input.load_complex_f32(0);
        let row1 = input.load_complex_f32(4);

        // Do our butterfly 2's down the columns
        let (intermediate0, intermediate1_pretwiddle) = avx_utils::column_butterfly2_f32(row0, row1);

        // Apply the size-8 twiddle factors
        let intermediate1 = avx_utils::complex_multiply_fma_f32(intermediate1_pretwiddle, self.twiddles);

        // Rearrange the data before we do our butterfly 4s. This swaps the last 2 elements of butterfly0 with the first two elements of butterfly1
        // The result is that we can then do a 4x butterfly 2, apply twiddles, use unpack instructions to transpose to the final output, then do another 4x butterfly 2
        let permuted0 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x20);
        let permuted1 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x31);

        // Do the first set of butterfly 2's
        let (postbutterfly0, postbutterfly1_pretwiddle) = avx_utils::column_butterfly2_f32(permuted0, permuted1);

        // Which negative we blend in depends on whether we're forward or inverse
        let postbutterfly1 = avx_utils::rotate90_oddelements_f32(postbutterfly1_pretwiddle, self.twiddle_config);

        // use unpack instructions to transpose, and to prepare for the final butterfly 2's
        let unpermuted0 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x20);
        let unpermuted1 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x31);
        let unpacked0 = _mm256_unpacklo_ps(unpermuted0, unpermuted1);
        let unpacked1 = _mm256_unpackhi_ps(unpermuted0, unpermuted1);
        let swapped0 = _mm256_permute_ps(unpacked0, 0xD8);
        let swapped1 = _mm256_permute_ps(unpacked1, 0xD8);

        let (output0, output1) = avx_utils::column_butterfly2_f32(swapped0, swapped1);

        output.store_complex_f32(0, output0);
        output.store_complex_f32(4, output1);
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x2, 8);

pub struct MixedRadixAvx4x3<T> {
    twiddles: [__m256; 2],
    twiddles_butterfly3: __m256,
    twiddle_config: avx_utils::Rotate90Config,
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
            twiddles_butterfly3: avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
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
        let (mid0, mid1, mid2) = avx_utils::column_butterfly3_f32(input0, input1, input2, self.twiddles_butterfly3);

        // Multiply in our twiddle factors
        let mid1_twiddled = avx_utils::complex_multiply_fma_f32(mid1, self.twiddles[0]);
        let mid2_twiddled = avx_utils::complex_multiply_fma_f32(mid2, self.twiddles[1]);

        // Transpose our 4x3 array into a 3x4
        // this will leave us with an empty column. after the butterfly 4's, we'll need to do some data reshuffling,
        // and that reshuffling will be simpler if we duplicate the second column into the third, and shift the third to the fourth. we cand othat by accordingly shifting the rows we pass to the transpose
        // the computed data and runtime will be the same either way, so why not?
        let (transposed0, transposed1, transposed2, transposed3) = avx_utils::transpose_4x4_f32(mid0, mid1_twiddled, mid1_twiddled, mid2_twiddled);

        // Do 4 butterfly 4's down the columns of our transposed array
        let (output0, output1, output2, output3) = avx_utils::column_butterfly4_f32(transposed0, transposed1, transposed2, transposed3, self.twiddle_config);

        // Theoretically, we could do masked writes of our 4 output registers directly to memory, but we'd be doing overlapping writes, and benchmarking shows that it's incredibly slow
        // instead, aided by our column duplication above, we can pack our data into just 3 non-overlapping registers

        // the data we want is scattered across 4 different registers, so these first 3 instructions bring them all into the same register, in the right order
        let intermediate0 = _mm256_permute2f128_ps(output0, output2, 0x13);
        let intermediate1 = _mm256_permute2f128_ps(output1, output3, 0x02);

        let shuffled = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(intermediate0), _mm256_castps_pd(intermediate1), 0x05));

        // Now that the "suffled" vector contains all the elements gathere from the various rows, we can pack everything together
        let packed1 = _mm256_permute2f128_ps(output1, output2, 0x21);
        let packed0 = _mm256_permute2f128_ps(output0, shuffled, 0x30); 
        let packed2 = _mm256_permute2f128_ps(shuffled, output3, 0x30); 

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
    twiddle_config: avx_utils::Rotate90Config,
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
        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(1, 16, inverse),
            f32::generate_twiddle_factor(2, 16, inverse),
            f32::generate_twiddle_factor(3, 16, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(2, 16, inverse),
            f32::generate_twiddle_factor(4, 16, inverse),
            f32::generate_twiddle_factor(6, 16, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(3, 16, inverse),
            f32::generate_twiddle_factor(6, 16, inverse),
            f32::generate_twiddle_factor(9, 16, inverse),
        ];
        Self {
            twiddles: [
                twiddles.load_complex_f32(0),
                twiddles.load_complex_f32(4),
                twiddles.load_complex_f32(8),
            ],
            twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let input0 = input.load_complex_f32(0);
        let input1 = input.load_complex_f32(1 * 4);
        let input2 = input.load_complex_f32(2 * 4);
        let input3 = input.load_complex_f32(3 * 4);

        // We're going to treat our input as a 3x4 2d array. First, do 3 butterfly 4's down the columns of that array.
        let (mid0, mid1, mid2, mid3) = avx_utils::column_butterfly4_f32(input0, input1, input2, input3, self.twiddle_config);

        // Multiply in our twiddle factors
        let mid1_twiddled = avx_utils::complex_multiply_fma_f32(mid1, self.twiddles[0]);
        let mid2_twiddled = avx_utils::complex_multiply_fma_f32(mid2, self.twiddles[1]);
        let mid3_twiddled = avx_utils::complex_multiply_fma_f32(mid3, self.twiddles[2]);

        // Transpose our 4x4 array
        let (transposed0, transposed1, transposed2, transposed3) = avx_utils::transpose_4x4_f32(mid0, mid1_twiddled, mid2_twiddled, mid3_twiddled);

        // Do 4 butterfly 4's down the columns of our transposed array
        let (output0, output1, output2, output3) = avx_utils::column_butterfly4_f32(transposed0, transposed1, transposed2, transposed3, self.twiddle_config);

        output.store_complex_f32(0, output0);
        output.store_complex_f32(1 * 4, output1);
        output.store_complex_f32(2 * 4, output2);
        output.store_complex_f32(3 * 4, output3);
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x4, 16);

pub struct MixedRadixAvx4x6<T> {
    twiddles: [__m256; 5],
    twiddles_butterfly3: __m256,
    twiddle_config: avx_utils::Rotate90Config,
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
        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(1, 24, inverse),
            f32::generate_twiddle_factor(2, 24, inverse),
            f32::generate_twiddle_factor(3, 24, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(2, 24, inverse),
            f32::generate_twiddle_factor(4, 24, inverse),
            f32::generate_twiddle_factor(6, 24, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(3, 24, inverse),
            f32::generate_twiddle_factor(6, 24, inverse),
            f32::generate_twiddle_factor(9, 24, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(4, 24, inverse),
            f32::generate_twiddle_factor(8, 24, inverse),
            f32::generate_twiddle_factor(12, 24, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(5, 24, inverse),
            f32::generate_twiddle_factor(10, 24, inverse),
            f32::generate_twiddle_factor(15, 24, inverse),
        ];
        Self {
            twiddles: [
                twiddles.load_complex_f32(0),
                twiddles.load_complex_f32(4),
                twiddles.load_complex_f32(8),
                twiddles.load_complex_f32(12),
                twiddles.load_complex_f32(16),
            ],
            twiddles_butterfly3: avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 3, inverse)),
            twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let input0 = input.load_complex_f32(0);
        let input1 = input.load_complex_f32(1 * 4);
        let input2 = input.load_complex_f32(2 * 4);
        let input3 = input.load_complex_f32(3 * 4);
        let input4 = input.load_complex_f32(4 * 4);
        let input5 = input.load_complex_f32(5 * 4);

        // We're going to treat our input as a 4x6 2d array. First, do 4 butterfly 6's down the columns of that array.
        let (mid0, mid1, mid2, mid3, mid4, mid5) = avx_utils::column_butterfly6_f32(input0, input1, input2, input3, input4, input5, self.twiddles_butterfly3);

        // Multiply in our twiddle factors
        let mid1_twiddled = avx_utils::complex_multiply_fma_f32(mid1, self.twiddles[0]);
        let mid2_twiddled = avx_utils::complex_multiply_fma_f32(mid2, self.twiddles[1]);
        let mid3_twiddled = avx_utils::complex_multiply_fma_f32(mid3, self.twiddles[2]);
        let mid4_twiddled = avx_utils::complex_multiply_fma_f32(mid4, self.twiddles[3]);
        let mid5_twiddled = avx_utils::complex_multiply_fma_f32(mid5, self.twiddles[4]);


        // Transpose our 6x4 array into a 4x6. Sadly this will leave us with 2 garbage columns on the end
        let (transposed0, transposed1, transposed2, transposed3) = avx_utils::transpose_4x4_f32(mid0, mid1_twiddled, mid2_twiddled, mid3_twiddled);
        let (transposed4, transposed5, transposed6, transposed7) = avx_utils::transpose_4x4_f32(mid4_twiddled, mid5_twiddled, _mm256_setzero_ps(), _mm256_setzero_ps());

        // Do 4 butterfly 4's down the columns of our transposed array
        let (output0, output1, output2, output3) = avx_utils::column_butterfly4_f32(transposed0, transposed1, transposed2, transposed3, self.twiddle_config);
        let (output4, output5, output6, output7) = avx_utils::column_butterfly4_f32(transposed4, transposed5, transposed6, transposed7, self.twiddle_config);

        // the last two elements in each array are empty, so for every other element, we're only going to store half of the data
        output.store_complex_f32(0, output0);
        output.store_complex_f32_lo(output4, 4);

        output.store_complex_f32(1 * 6, output1);
        output.store_complex_f32_lo(output5, 1 * 6 + 4);

        output.store_complex_f32(2 * 6, output2);
        output.store_complex_f32_lo(output6, 2 * 6 + 4);

        output.store_complex_f32(3 * 6, output3);
        output.store_complex_f32_lo(output7, 3 * 6 + 4);
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x6, 24);

pub struct MixedRadixAvx4x8<T> {
    twiddles: [__m256; 6],
    twiddles_butterfly8: __m256,
    twiddle_config: avx_utils::Rotate90Config,
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
        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(1, 32, inverse),
            f32::generate_twiddle_factor(2, 32, inverse),
            f32::generate_twiddle_factor(3, 32, inverse),
            f32::generate_twiddle_factor(4, 32, inverse),
            f32::generate_twiddle_factor(5, 32, inverse),
            f32::generate_twiddle_factor(6, 32, inverse),
            f32::generate_twiddle_factor(7, 32, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(2, 32, inverse),
            f32::generate_twiddle_factor(4, 32, inverse),
            f32::generate_twiddle_factor(6, 32, inverse),
            f32::generate_twiddle_factor(8, 32, inverse),
            f32::generate_twiddle_factor(10, 32, inverse),
            f32::generate_twiddle_factor(12, 32, inverse),
            f32::generate_twiddle_factor(14, 32, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(3, 32, inverse),
            f32::generate_twiddle_factor(6, 32, inverse),
            f32::generate_twiddle_factor(9, 32, inverse),
            f32::generate_twiddle_factor(12, 32, inverse),
            f32::generate_twiddle_factor(15, 32, inverse),
            f32::generate_twiddle_factor(18, 32, inverse),
            f32::generate_twiddle_factor(21, 32, inverse),
        ];

        Self {
            twiddles: [
                twiddles.load_complex_f32(0),
                twiddles.load_complex_f32(4),
                twiddles.load_complex_f32(8),
                twiddles.load_complex_f32(12),
                twiddles.load_complex_f32(16),
                twiddles.load_complex_f32(20),
            ],
            twiddles_butterfly8: avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 8, inverse)),
            twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let input0 = input.load_complex_f32(0);
        let input1 = input.load_complex_f32(1 * 4);
        let input2 = input.load_complex_f32(2 * 4);
        let input3 = input.load_complex_f32(3 * 4);
        let input4 = input.load_complex_f32(4 * 4);
        let input5 = input.load_complex_f32(5 * 4);
        let input6 = input.load_complex_f32(6 * 4);
        let input7 = input.load_complex_f32(7 * 4);

        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        let (mid0, mid2, mid4, mid6) = avx_utils::column_butterfly4_f32(input0, input2, input4, input6, self.twiddle_config);
        let (mid1, mid3, mid5, mid7) = avx_utils::column_butterfly4_f32(input1, input3, input5, input7, self.twiddle_config);

        // Multiply in our twiddle factors
        let mid2_twiddled = avx_utils::complex_multiply_fma_f32(mid2, self.twiddles[0]);
        let mid3_twiddled = avx_utils::complex_multiply_fma_f32(mid3, self.twiddles[1]);
        let mid4_twiddled = avx_utils::complex_multiply_fma_f32(mid4, self.twiddles[2]);
        let mid5_twiddled = avx_utils::complex_multiply_fma_f32(mid5, self.twiddles[3]);
        let mid6_twiddled = avx_utils::complex_multiply_fma_f32(mid6, self.twiddles[4]);
        let mid7_twiddled = avx_utils::complex_multiply_fma_f32(mid7, self.twiddles[5]);

        // Transpose our 8x4 array to an 8x4 array. Thankfully we can just do 2 4x4 transposes, which are only 8 instructions each!
        let (transposed0, transposed1, transposed2, transposed3) = avx_utils::transpose_4x4_f32(mid0, mid2_twiddled, mid4_twiddled, mid6_twiddled);
        let (transposed4, transposed5, transposed6, transposed7) = avx_utils::transpose_4x4_f32(mid1, mid3_twiddled, mid5_twiddled, mid7_twiddled);

        // Do 4 butterfly 8's down the columns of our transpsed array
        let (output0, output1, output2, output3, output4, output5, output6, output7) = avx_utils::column_butterfly8_fma_f32(transposed0, transposed1, transposed2, transposed3, transposed4, transposed5, transposed6, transposed7, self.twiddles_butterfly8, self.twiddle_config);

        output.store_complex_f32(0, output0);
        output.store_complex_f32(1 * 4, output1);
        output.store_complex_f32(2 * 4, output2);
        output.store_complex_f32(3 * 4, output3);
        output.store_complex_f32(4 * 4, output4);
        output.store_complex_f32(5 * 4, output5);
        output.store_complex_f32(6 * 4, output6);
        output.store_complex_f32(7 * 4, output7);
    }
}
boilerplate_fft_simd_butterfly!(MixedRadixAvx4x8, 32);

pub struct MixedRadixAvx8x8<T> {
    twiddles: [__m256; 14],
    twiddles_butterfly8: __m256,
    twiddle_config: avx_utils::Rotate90Config,
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
        let twiddles = [
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(1, 64, inverse),
            f32::generate_twiddle_factor(2, 64, inverse),
            f32::generate_twiddle_factor(3, 64, inverse),
            f32::generate_twiddle_factor(4, 64, inverse),
            f32::generate_twiddle_factor(5, 64, inverse),
            f32::generate_twiddle_factor(6, 64, inverse),
            f32::generate_twiddle_factor(7, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(2, 64, inverse),
            f32::generate_twiddle_factor(4, 64, inverse),
            f32::generate_twiddle_factor(6, 64, inverse),
            f32::generate_twiddle_factor(8, 64, inverse),
            f32::generate_twiddle_factor(10, 64, inverse),
            f32::generate_twiddle_factor(12, 64, inverse),
            f32::generate_twiddle_factor(14, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(3, 64, inverse),
            f32::generate_twiddle_factor(6, 64, inverse),
            f32::generate_twiddle_factor(9, 64, inverse),
            f32::generate_twiddle_factor(12, 64, inverse),
            f32::generate_twiddle_factor(15, 64, inverse),
            f32::generate_twiddle_factor(18, 64, inverse),
            f32::generate_twiddle_factor(21, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(4, 64, inverse),
            f32::generate_twiddle_factor(8, 64, inverse),
            f32::generate_twiddle_factor(12, 64, inverse),
            f32::generate_twiddle_factor(16, 64, inverse),
            f32::generate_twiddle_factor(20, 64, inverse),
            f32::generate_twiddle_factor(24, 64, inverse),
            f32::generate_twiddle_factor(28, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(5, 64, inverse),
            f32::generate_twiddle_factor(10, 64, inverse),
            f32::generate_twiddle_factor(15, 64, inverse),
            f32::generate_twiddle_factor(20, 64, inverse),
            f32::generate_twiddle_factor(25, 64, inverse),
            f32::generate_twiddle_factor(30, 64, inverse),
            f32::generate_twiddle_factor(35, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(6, 64, inverse),
            f32::generate_twiddle_factor(12, 64, inverse),
            f32::generate_twiddle_factor(18, 64, inverse),
            f32::generate_twiddle_factor(24, 64, inverse),
            f32::generate_twiddle_factor(30, 64, inverse),
            f32::generate_twiddle_factor(36, 64, inverse),
            f32::generate_twiddle_factor(42, 64, inverse),
            Complex{ re: 1.0, im: 0.0 },
            f32::generate_twiddle_factor(7, 64, inverse),
            f32::generate_twiddle_factor(14, 64, inverse),
            f32::generate_twiddle_factor(21, 64, inverse),
            f32::generate_twiddle_factor(28, 64, inverse),
            f32::generate_twiddle_factor(35, 64, inverse),
            f32::generate_twiddle_factor(42, 64, inverse),
            f32::generate_twiddle_factor(49, 64, inverse),
        ];

        Self {
            twiddles: [
                twiddles.load_complex_f32(0),
                twiddles.load_complex_f32(4),
                twiddles.load_complex_f32(8),
                twiddles.load_complex_f32(12),
                twiddles.load_complex_f32(16),
                twiddles.load_complex_f32(20),
                twiddles.load_complex_f32(24),
                twiddles.load_complex_f32(28),
                twiddles.load_complex_f32(32),
                twiddles.load_complex_f32(36),
                twiddles.load_complex_f32(40),
                twiddles.load_complex_f32(44),
                twiddles.load_complex_f32(48),
                twiddles.load_complex_f32(52),
            ],
            twiddles_butterfly8: avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 8, inverse)),
            twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let input0 = input.load_complex_f32(0);
        let input2 = input.load_complex_f32(2 * 4);
        let input4 = input.load_complex_f32(4 * 4);
        let input6 = input.load_complex_f32(6 * 4);
        let input8 = input.load_complex_f32(8 * 4);
        let input10 = input.load_complex_f32(10 * 4);
        let input12 = input.load_complex_f32(12 * 4);
        let input14 = input.load_complex_f32(14 * 4);

        // We're going to treat our input as a 8x8 2d array. First, do 8 butterfly 8's down the columns of that array.
        let (mid0, mid2, mid4, mid6, mid8, mid10, mid12, mid14) = avx_utils::column_butterfly8_fma_f32(input0, input2, input4, input6, input8, input10, input12, input14, self.twiddles_butterfly8, self.twiddle_config);

        // Apply twiddle factors to the first half of our data
        let mid2_twiddled =  avx_utils::complex_multiply_fma_f32(mid2,  self.twiddles[0]);
        let mid4_twiddled =  avx_utils::complex_multiply_fma_f32(mid4,  self.twiddles[2]);
        let mid6_twiddled =  avx_utils::complex_multiply_fma_f32(mid6,  self.twiddles[4]);
        let mid8_twiddled =  avx_utils::complex_multiply_fma_f32(mid8,  self.twiddles[6]);
        let mid10_twiddled = avx_utils::complex_multiply_fma_f32(mid10, self.twiddles[8]);
        let mid12_twiddled = avx_utils::complex_multiply_fma_f32(mid12, self.twiddles[10]);
        let mid14_twiddled = avx_utils::complex_multiply_fma_f32(mid14, self.twiddles[12]);

        // Transpose the first half of this. After this, the compiler can spill this stuff, because it won't be needed until after the loads+butterfly8s+twiddles below are done
        let (transposed0, transposed2,  transposed4,  transposed6)  = avx_utils::transpose_4x4_f32(mid0, mid2_twiddled, mid4_twiddled, mid6_twiddled);
        let (transposed1, transposed3,  transposed5,  transposed7)  = avx_utils::transpose_4x4_f32(mid8_twiddled, mid10_twiddled, mid12_twiddled, mid14_twiddled);

        // Now that the first half of our data has been transposed, the compiler is free to spill those registers to make room for the other half
        let input1 = input.load_complex_f32(1 * 4);
        let input3 = input.load_complex_f32(3 * 4);
        let input5 = input.load_complex_f32(5 * 4);
        let input7 = input.load_complex_f32(7 * 4);
        let input9 = input.load_complex_f32(9 * 4);
        let input11 = input.load_complex_f32(11 * 4);
        let input13 = input.load_complex_f32(13 * 4);
        let input15 = input.load_complex_f32(15 * 4);
        let (mid1, mid3, mid5, mid7, mid9, mid11, mid13, mid15) = avx_utils::column_butterfly8_fma_f32(input1, input3, input5, input7, input9, input11, input13, input15, self.twiddles_butterfly8, self.twiddle_config);

        // Apply twiddle factors to the second half of our data
        let mid3_twiddled =  avx_utils::complex_multiply_fma_f32(mid3,  self.twiddles[1]);
        let mid5_twiddled =  avx_utils::complex_multiply_fma_f32(mid5,  self.twiddles[3]);
        let mid7_twiddled =  avx_utils::complex_multiply_fma_f32(mid7,  self.twiddles[5]);
        let mid9_twiddled =  avx_utils::complex_multiply_fma_f32(mid9,  self.twiddles[7]);
        let mid11_twiddled = avx_utils::complex_multiply_fma_f32(mid11, self.twiddles[9]);
        let mid13_twiddled = avx_utils::complex_multiply_fma_f32(mid13, self.twiddles[11]);
        let mid15_twiddled = avx_utils::complex_multiply_fma_f32(mid15, self.twiddles[13]);

        // Transpose the second half of our 8x8
        let (transposed8, transposed10, transposed12, transposed14) = avx_utils::transpose_4x4_f32(mid1, mid3_twiddled, mid5_twiddled, mid7_twiddled);
        let (transposed9, transposed11, transposed13, transposed15) = avx_utils::transpose_4x4_f32(mid9_twiddled, mid11_twiddled, mid13_twiddled, mid15_twiddled);

        // Do 4 butterfly 8's down the columns of our transposed array, and store the results
        let (output0, output2, output4, output6, output8, output10, output12, output14) = avx_utils::column_butterfly8_fma_f32(transposed0, transposed2, transposed4, transposed6, transposed8, transposed10, transposed12, transposed14, self.twiddles_butterfly8, self.twiddle_config);
        output.store_complex_f32(0, output0);
        output.store_complex_f32(2 * 4, output2);
        output.store_complex_f32(4 * 4, output4);
        output.store_complex_f32(6 * 4, output6);
        output.store_complex_f32(8 * 4, output8);
        output.store_complex_f32(10 * 4, output10);
        output.store_complex_f32(12 * 4, output12);
        output.store_complex_f32(14 * 4, output14);

        // We freed up a bunch of registers, so we should easily have enough room to compute+store the other half of our butterfly 8s
        let (output1, output3, output5, output7, output9, output11, output13, output15) = avx_utils::column_butterfly8_fma_f32(transposed1, transposed3, transposed5, transposed7, transposed9, transposed11, transposed13, transposed15, self.twiddles_butterfly8, self.twiddle_config);
        output.store_complex_f32(1 * 4, output1);
        output.store_complex_f32(3 * 4, output3);
        output.store_complex_f32(5 * 4, output5);
        output.store_complex_f32(7 * 4, output7);
        output.store_complex_f32(9 * 4, output9);
        output.store_complex_f32(11 * 4, output11);
        output.store_complex_f32(13 * 4, output13);
        output.store_complex_f32(15 * 4, output15);
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
    test_avx_butterfly!(test_avx_mixedradix8x8, MixedRadixAvx8x8, 64);

    use std::sync::Arc;
    use ::algorithm::butterflies::*;
    use ::algorithm::MixedRadixSmall;

    #[test]
    fn test_butterfly24() {
        let inner_width = Arc::new(Butterfly4::new(false));
        let inner_height = Arc::new(Butterfly6::new(false));

        let control_fft = MixedRadixSmall::new(inner_width, inner_height);
        check_fft_algorithm(&control_fft, 24, false);

        let butterfly = MixedRadixAvx4x6::new(false).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_fft_algorithm(&butterfly, 24, false);

        let butterfly_inverse = MixedRadixAvx4x6::new(true).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_fft_algorithm(&butterfly_inverse, 24, true);
    }
}
