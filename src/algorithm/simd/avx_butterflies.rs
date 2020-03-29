use std::marker::PhantomData;
use std::arch::x86_64::*;

use num_complex::Complex;

use common::FFTnum;

use ::{Length, IsInverse, FftInline};

use super::avx_utils::AvxComplexArrayf32;
use super::avx_utils;


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
    unsafe fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], _scratch: &[Complex<f32>]) {
        let row0 = buffer.load_complex_f32(0);
        let row1 = buffer.load_complex_f32(4);

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

        buffer.store_complex_f32(0, output0);
        buffer.store_complex_f32(4, output1);
    }
}
boilerplate_fft_simd_unsafe!(MixedRadixAvx4x2, 
    |_| 8,
    |_| 0
);

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
    unsafe fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], _scratch: &[Complex<f32>]) {
        let input0 = buffer.load_complex_f32(0);
        let input1 = buffer.load_complex_f32(1 * 4);
        let input2 = buffer.load_complex_f32(2 * 4);
        let input3 = buffer.load_complex_f32(3 * 4);

        // We're going to treat our input as a 3x4 2d array. First, do 3 butterfly 4's down the columns of that array.
        let (mid0, mid1, mid2, mid3) = avx_utils::column_butterfly4_f32(input0, input1, input2, input3, self.twiddle_config);

        // Multiply in our twiddle factors
        let mid1_twiddled = avx_utils::complex_multiply_fma_f32(mid1, self.twiddles[0]);
        let mid2_twiddled = avx_utils::complex_multiply_fma_f32(mid2, self.twiddles[1]);
        let mid3_twiddled = avx_utils::complex_multiply_fma_f32(mid3, self.twiddles[2]);

        // Transpose out 4x4 array
        let (transposed0, transposed1, transposed2, transposed3) = avx_utils::transpose_4x4_f32(mid0, mid1_twiddled, mid2_twiddled, mid3_twiddled);

        // Do 4 butterfly 8's down the columns of our transpsed array
        let (output0, output1, output2, output3) = avx_utils::column_butterfly4_f32(transposed0, transposed1, transposed2, transposed3, self.twiddle_config);

        buffer.store_complex_f32(0, output0);
        buffer.store_complex_f32(1 * 4, output1);
        buffer.store_complex_f32(2 * 4, output2);
        buffer.store_complex_f32(3 * 4, output3);
    }
}
boilerplate_fft_simd_unsafe!(MixedRadixAvx4x4, 
    |_| 16,
    |_| 0
);

pub struct MixedRadixAvx4x8<T> {
    twiddles: [__m256; 6],
    twiddles_butterfly8: __m256,
    twiddle_config: avx_utils::Rotate90Config,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FFTnum> MixedRadixAvx4x8<T> {
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
    unsafe fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], _scratch: &[Complex<f32>]) {
        let input0 = buffer.load_complex_f32(0);
        let input1 = buffer.load_complex_f32(1 * 4);
        let input2 = buffer.load_complex_f32(2 * 4);
        let input3 = buffer.load_complex_f32(3 * 4);
        let input4 = buffer.load_complex_f32(4 * 4);
        let input5 = buffer.load_complex_f32(5 * 4);
        let input6 = buffer.load_complex_f32(6 * 4);
        let input7 = buffer.load_complex_f32(7 * 4);

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

        buffer.store_complex_f32(0, output0);
        buffer.store_complex_f32(1 * 4, output1);
        buffer.store_complex_f32(2 * 4, output2);
        buffer.store_complex_f32(3 * 4, output3);
        buffer.store_complex_f32(4 * 4, output4);
        buffer.store_complex_f32(5 * 4, output5);
        buffer.store_complex_f32(6 * 4, output6);
        buffer.store_complex_f32(7 * 4, output7);
    }
}
boilerplate_fft_simd_unsafe!(MixedRadixAvx4x8, 
    |_| 32,
    |_| 0
);

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
}

impl MixedRadixAvx8x8<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_inplace_f32(&self, buffer: &mut [Complex<f32>], _scratch: &[Complex<f32>]) {
        let input0 = buffer.load_complex_f32(0);
        let input2 = buffer.load_complex_f32(2 * 4);
        let input4 = buffer.load_complex_f32(4 * 4);
        let input6 = buffer.load_complex_f32(6 * 4);
        let input8 = buffer.load_complex_f32(8 * 4);
        let input10 = buffer.load_complex_f32(10 * 4);
        let input12 = buffer.load_complex_f32(12 * 4);
        let input14 = buffer.load_complex_f32(14 * 4);

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
        let input1 = buffer.load_complex_f32(1 * 4);
        let input3 = buffer.load_complex_f32(3 * 4);
        let input5 = buffer.load_complex_f32(5 * 4);
        let input7 = buffer.load_complex_f32(7 * 4);
        let input9 = buffer.load_complex_f32(9 * 4);
        let input11 = buffer.load_complex_f32(11 * 4);
        let input13 = buffer.load_complex_f32(13 * 4);
        let input15 = buffer.load_complex_f32(15 * 4);
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
        buffer.store_complex_f32(0, output0);
        buffer.store_complex_f32(2 * 4, output2);
        buffer.store_complex_f32(4 * 4, output4);
        buffer.store_complex_f32(6 * 4, output6);
        buffer.store_complex_f32(8 * 4, output8);
        buffer.store_complex_f32(10 * 4, output10);
        buffer.store_complex_f32(12 * 4, output12);
        buffer.store_complex_f32(14 * 4, output14);

        // We freed up a bunch of registers, so we should easily have enough room to compute+store the other half of our butterfly 8s
        let (output1, output3, output5, output7, output9, output11, output13, output15) = avx_utils::column_butterfly8_fma_f32(transposed1, transposed3, transposed5, transposed7, transposed9, transposed11, transposed13, transposed15, self.twiddles_butterfly8, self.twiddle_config);
        buffer.store_complex_f32(1 * 4, output1);
        buffer.store_complex_f32(3 * 4, output3);
        buffer.store_complex_f32(5 * 4, output5);
        buffer.store_complex_f32(7 * 4, output7);
        buffer.store_complex_f32(9 * 4, output9);
        buffer.store_complex_f32(11 * 4, output11);
        buffer.store_complex_f32(13 * 4, output13);
        buffer.store_complex_f32(15 * 4, output15);
    }
}
boilerplate_fft_simd_unsafe!(MixedRadixAvx8x8, 
    |_| 64,
    |_| 0
);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::{check_inline_fft_algorithm};

    #[test]
    fn test_avx_mixedradix4x2() {
        let fft_forward = MixedRadixAvx4x2::new(false).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_inline_fft_algorithm(&fft_forward, 8, false);

        let fft_inverse = MixedRadixAvx4x2::new(true).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_inline_fft_algorithm(&fft_inverse, 8, true);
    }

    #[test]
    fn test_avx_mixedradix4x4() {
        let fft_forward = MixedRadixAvx4x4::new(false).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_inline_fft_algorithm(&fft_forward, 16, false);

        let fft_inverse = MixedRadixAvx4x4::new(true).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_inline_fft_algorithm(&fft_inverse, 16, true);
    }

    #[test]
    fn test_avx_mixedradix4x8() {
        let fft_forward = MixedRadixAvx4x8::new(false).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_inline_fft_algorithm(&fft_forward, 32, false);

        let fft_inverse = MixedRadixAvx4x8::new(true).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_inline_fft_algorithm(&fft_inverse, 32, true);
    }

    #[test]
    fn test_avx_mixedradix8x8() {
        let fft_forward = MixedRadixAvx8x8::new(false).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_inline_fft_algorithm(&fft_forward, 64, false);

        let fft_inverse = MixedRadixAvx8x8::new(true).expect("Can't run test because this machine doesn't have the required instruction sets");
        check_inline_fft_algorithm(&fft_inverse, 64, true);
    }
}
