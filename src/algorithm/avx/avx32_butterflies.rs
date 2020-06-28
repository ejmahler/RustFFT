use std::marker::PhantomData;
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

use num_complex::Complex;

use crate::common::FFTnum;

use crate::{Length, IsInverse, Fft};

use crate::array_utils::{RawSlice, RawSliceMut};
use super::avx32_utils;
use super::avx_vector::{AvxVector, AvxVector128, AvxVector256, Rotation90, AvxArray, AvxArrayMut};
use super::avx_vector;

// Safety: This macro will call `self::perform_fft_f32()` which probably has a #[target_feature(enable = "...")] annotation on it.
// Calling functions with that annotation is unsafe, because it doesn't actually check if the CPU has the required features.
// Callers of this macro must guarantee that users can't even obtain an instance of $struct_name if their CPU doesn't have the required CPU features.
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

// Safety: This macro will call `self::perform_fft_f32()` which probably has a #[target_feature(enable = "...")] annotation on it.
// Calling functions with that annotation is unsafe, because it doesn't actually check if the CPU has the required features.
// Callers of this macro must guarantee that users can't even obtain an instance of $struct_name if their CPU doesn't have the required CPU features.
macro_rules! boilerplate_fft_simd_butterfly_with_scratch {
    ($struct_name:ident, $len:expr) => (
        impl $struct_name<f32> {
            #[inline]
            fn perform_fft_inplace(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                // Perform the column FFTs
                // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't available
                unsafe { self.column_butterflies_and_transpose(buffer, scratch) };

                // process the row FFTs, and copy from the scratch back to the buffer as we go
                // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
                unsafe { self.row_butterflies(RawSlice::new(scratch), RawSliceMut::new(buffer)) };
            }

            #[inline]
            fn perform_fft_out_of_place(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>]) {
                // Perform the column FFTs
                // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't available
                unsafe { self.column_butterflies_and_transpose(input, output) };

                // process the row FFTs in-place in the output buffer
                // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
                unsafe { self.row_butterflies(RawSlice::new(output), RawSliceMut::new(output)) };
            }
        }
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
		
				self.perform_fft_out_of_place(input, output);
            }
            fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], _scratch: &mut [Complex<f32>]) {
                assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					self.perform_fft_out_of_place(in_chunk, out_chunk);
				}
            }
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                assert!(scratch.len() >= $len);
                let scratch = &mut scratch[..$len];
        
                self.perform_fft_inplace(buffer, scratch);
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                assert!(scratch.len() >= $len);
                let scratch = &mut scratch[..$len];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    self.perform_fft_inplace(chunk, scratch);
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                $len
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
        const TWIDDLE_VECTOR_COLS : usize = TWIDDLE_COLS / 4;
        const TWIDDLE_VECTOR_COUNT : usize = TWIDDLE_VECTOR_COLS*TWIDDLE_ROWS;
        let mut twiddles = [AvxVector::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index / TWIDDLE_VECTOR_COLS) + 1;
            let x = (index % TWIDDLE_VECTOR_COLS) * 4 + $skip_cols;
            
            twiddles[index] = AvxVector::make_mixedradix_twiddle_chunk(x, y, FFT_LEN, $inverse);
        }
        twiddles
    }}
}


macro_rules! gen_butterfly_twiddles_separated_columns {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $inverse: expr) => {{
        const FFT_LEN : usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS : usize = $num_rows - 1;
        const TWIDDLE_COLS : usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS : usize = TWIDDLE_COLS / 4;
        const TWIDDLE_VECTOR_COUNT : usize = TWIDDLE_VECTOR_COLS*TWIDDLE_ROWS;
        let mut twiddles = [AvxVector::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index % TWIDDLE_ROWS) + 1;
            let x = (index / TWIDDLE_ROWS) * 4 + $skip_cols;

            twiddles[index] = AvxVector::make_mixedradix_twiddle_chunk(x, y, FFT_LEN, $inverse);
        }
        twiddles
    }}
}



pub struct Butterfly5Avx<T> {
    twiddles: [__m128; 3],
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly5Avx, 5);
impl Butterfly5Avx<f32> {
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
        let twiddle1 = f32::generate_twiddle_factor(1, 5, inverse);
        let twiddle2 = f32::generate_twiddle_factor(2, 5, inverse);
        Self {
            twiddles: [
                _mm_set_ps(twiddle1.im, twiddle1.im, twiddle1.re, twiddle1.re),
                _mm_set_ps(twiddle2.im, twiddle2.im, twiddle2.re, twiddle2.re),
                _mm_set_ps(-twiddle1.im, -twiddle1.im, twiddle1.re, twiddle1.re),
            ],
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let input0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64)); // load the first element of the input, and duplicate it into both complex number slots of input0
        let input12 = input.load_partial2_complex(1);
        let input34 = input.load_partial2_complex(3);

        // swap elements for inputs 3 and 4
        let input43 = AvxVector::reverse_complex_elements(input34);

        // do some prep work before we can start applying twiddle factors
        let [sum12, diff43] = AvxVector::column_butterfly2([input12, input43]);
        
        let rotation = AvxVector::make_rotation90(true);
        let rotated43 = AvxVector::rotate90(diff43, rotation);

        let [mid14, mid23] = AvxVector::unpack_complex([sum12, rotated43]);

        // to compute the first output, compute the sum of all elements. mid14[0] and mid23[0] already have the sum of 1+4 and 2+3 respectively, so if we add them, we'll get the sum of all 4
        let sum1234 = AvxVector::add(mid14, mid23);
        let output0 = AvxVector::add(input0, sum1234);
        
        // apply twiddle factors
        let twiddled14_mid = AvxVector::mul(mid14, self.twiddles[0]);
        let twiddled23_mid = AvxVector::mul(mid14, self.twiddles[1]);
        let twiddled14 = AvxVector::fmadd(mid23, self.twiddles[1], twiddled14_mid);
        let twiddled23 = AvxVector::fmadd(mid23, self.twiddles[2], twiddled23_mid);

        // unpack the data for the last butterfly 2
        let [twiddled12, twiddled43] = AvxVector::unpack_complex([twiddled14, twiddled23]);
        let [output12, output43] = AvxVector::column_butterfly2([twiddled12, twiddled43]);

        // swap the elements in output43 before writing them out, and add the first input to everything
        let final12 = AvxVector::add(input0, output12);
        let output34 = AvxVector::reverse_complex_elements(output43);
        let final34 = AvxVector::add(input0, output34);

        output.store_partial1_complex(output0, 0);
        output.store_partial2_complex(final12, 1);
        output.store_partial2_complex(final34, 3);
    }
}


pub struct Butterfly7Avx<T> {
    twiddles: [__m128; 5],
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly7Avx, 7);
impl Butterfly7Avx<f32> {
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
        let twiddle1 = f32::generate_twiddle_factor(1, 7, inverse);
        let twiddle2 = f32::generate_twiddle_factor(2, 7, inverse);
        let twiddle3 = f32::generate_twiddle_factor(3, 7, inverse);
        Self {
            twiddles: [
                _mm_set_ps(twiddle1.im, twiddle1.im, twiddle1.re, twiddle1.re),
                _mm_set_ps(twiddle2.im, twiddle2.im, twiddle2.re, twiddle2.re),
                _mm_set_ps(twiddle3.im, twiddle3.im, twiddle3.re, twiddle3.re),
                _mm_set_ps(-twiddle3.im, -twiddle3.im, twiddle3.re, twiddle3.re),
                _mm_set_ps(-twiddle1.im, -twiddle1.im, twiddle1.re, twiddle1.re),
            ],
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let input0 = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64)); // load the first element of the input, and duplicate it into both complex number slots of input0
        // we want to load 3 elements into 123 and 3 elements into 456, but we can only load 4, so we're going to do slightly overlapping reads here
        // we have to reverse 456 immediately after loading, and that'll be easiest if we load the 456 into the latter 3 slots of the register, rather than the front 3 slots
        // as a bonus, that also means we don't need masked reads or anything
        let input123 = input.load_complex(1);
        let input456 = input.load_complex(3);

        // reverse the order of input456
        let input654 = AvxVector::reverse_complex_elements(input456);

        // do some prep work before we can start applying twiddle factors
        let [sum123, diff654] = AvxVector::column_butterfly2([input123, input654]);
        let rotation = AvxVector::make_rotation90(true);
        let rotated654 = AvxVector::rotate90(diff654, rotation);

        let [mid1634, mid25] = AvxVector::unpack_complex([sum123, rotated654]);

        let mid16 = mid1634.lo();
        let mid25 = mid25.lo();
        let mid34 = mid1634.hi();

        // to compute the first output, compute the sum of all elements. mid16[0], mid25[0], and mid34[0] already have the sum of 1+6, 2+5 and 3+4 respectively, so if we add them, we'll get 1+2+3+4+5+6
        let output0_left  = AvxVector::add(mid16, mid25);
        let output0_right = AvxVector::add(input0, mid34);
        let output0 = AvxVector::add(output0_left, output0_right);
        output.store_partial1_complex(output0, 0);

        _mm256_zeroupper();
        
        // apply twiddle factors
        let twiddled16_intermediate1 = AvxVector::mul(mid16, self.twiddles[0]);
        let twiddled25_intermediate1 = AvxVector::mul(mid16, self.twiddles[1]);
        let twiddled34_intermediate1 = AvxVector::mul(mid16, self.twiddles[2]);

        let twiddled16_intermediate2 = AvxVector::fmadd(mid25, self.twiddles[1], twiddled16_intermediate1);
        let twiddled25_intermediate2 = AvxVector::fmadd(mid25, self.twiddles[3], twiddled25_intermediate1);
        let twiddled34_intermediate2 = AvxVector::fmadd(mid25, self.twiddles[4], twiddled34_intermediate1);

        let twiddled16 = AvxVector::fmadd(mid34, self.twiddles[2], twiddled16_intermediate2);
        let twiddled25 = AvxVector::fmadd(mid34, self.twiddles[4], twiddled25_intermediate2);
        let twiddled34 = AvxVector::fmadd(mid34, self.twiddles[1], twiddled34_intermediate2);

        // unpack the data for the last butterfly 2
        let [twiddled12, twiddled65] = AvxVector::unpack_complex([twiddled16, twiddled25]);
        let [twiddled33, twiddled44] = AvxVector::unpack_complex([twiddled34, twiddled34]);

        // we can save one add if we add input0 to twiddled33 now. normally we'd add input0 to the final output, but the arrangement of data makes that a little awkward
        let twiddled033 = AvxVector::add(twiddled33, input0);

        let [output12, output65] = AvxVector::column_butterfly2([twiddled12, twiddled65]);
        let [output033, output044] = AvxVector::column_butterfly2([twiddled033, twiddled44]);
        let output56 = AvxVector::reverse_complex_elements(output65);
        
        output.store_partial2_complex(AvxVector::add(output12, input0), 1);
        output.store_partial1_complex(output033, 3);
        output.store_partial1_complex(output044, 4);
        output.store_partial2_complex(AvxVector::add(output56, input0), 5);
    }
}



pub struct Butterfly8Avx<T> {
    twiddles: __m256,
    twiddles_butterfly4: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly8Avx, 8);
impl Butterfly8Avx<f32> {
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
            twiddles: AvxVector::make_mixedradix_twiddle_chunk(0, 1, 8, inverse),
            twiddles_butterfly4: if inverse { 
                [Complex::new(0.0f32, 0.0), Complex::new(-0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(-0.0, 0.0)].load_complex(0)
            } else {
                [Complex::new(0.0f32, 0.0), Complex::new(0.0, -0.0), Complex::new(0.0, 0.0), Complex::new(0.0, -0.0)].load_complex(0)
            },
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let row0 = input.load_complex(0);
        let row1 = input.load_complex(4);

        // Do our butterfly 2's down the columns
        let [intermediate0, intermediate1_pretwiddle] = AvxVector::column_butterfly2([row0, row1]);

        // Apply the size-8 twiddle factors
        let intermediate1 = AvxVector::mul_complex(intermediate1_pretwiddle, self.twiddles);

        // Rearrange the data before we do our butterfly 4s. This swaps the last 2 elements of butterfly0 with the first two elements of butterfly1
        // The result is that we can then do a 4x butterfly 2, apply twiddles, use unpack instructions to transpose to the final output, then do another 4x butterfly 2
        let permuted0 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x20);
        let permuted1 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x31);

        // Do the first set of butterfly 2's
        let [postbutterfly0, postbutterfly1_pretwiddle] = AvxVector::column_butterfly2([permuted0, permuted1]);

        // Which negative we blend in depends on whether we're forward or inverse
        // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an inverse or not
        // but we can't use the AvxVector swap_complex_components function, because we only want to swap the odd reals with the odd imaginaries
        let elements_swapped = _mm256_permute_ps(postbutterfly1_pretwiddle, 0xB4);

        // We can negate the elements we want by xoring the row with a pre-set vector
        let postbutterfly1 = AvxVector::xor(elements_swapped, self.twiddles_butterfly4);

        // use unpack instructions to transpose, and to prepare for the final butterfly 2's
        let unpermuted0 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x20);
        let unpermuted1 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x31);
        let unpacked = AvxVector::unpack_complex([unpermuted0, unpermuted1]);

        let [output0, output1] = AvxVector::column_butterfly2(unpacked);

        output.store_complex(output0, 0);
        output.store_complex(output1, 4);
    }
}


pub struct Butterfly9Avx<T> {
    twiddles: __m256,
    twiddles_butterfly3: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly9Avx, 9);
impl Butterfly9Avx<f32> {
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
            twiddles: twiddles.load_complex(0),
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // we're going to load these elements in a peculiar way. instead of loading a row into the first 3 element of each register and leaving the last element empty
        // we're leaving the first element empty and putting the data in the last 3 elements. this will let us do 3 total complex multiplies instead of 4.

        let input0_lo = _mm_castpd_ps(_mm_load1_pd(input.as_ptr() as *const f64));
        let input0_hi = input.load_partial2_complex(1);
        let input0 = AvxVector256::merge(input0_lo, input0_hi);
        let input1 = input.load_complex(2);
        let input2 = input.load_complex(5);

        // We're going to treat our input as a 3x4 2d array. First, do 3 butterfly 4's down the columns of that array.
        let [mid0, mid1, mid2] = AvxVector::column_butterfly3([input0, input1, input2], self.twiddles_butterfly3);
        
        // merge the twiddle-able data into a single avx vector
        let twiddle_data = _mm256_permute2f128_ps(mid1, mid2, 0x31);
        let twiddled = AvxVector::mul_complex(twiddle_data, self.twiddles);

        // Transpose our 3x3 array. we're using the 4x4 transpose with an empty bottom row, which will result in an empty last column
        // but it turns out that it'll make our packing process later simpler if we duplicate the second row into the last row
        // which will result in duplicating the second column into the last column after the transpose
        let permute0 = _mm256_permute2f128_ps(mid0, mid2, 0x20);
        let permute1 = _mm256_permute2f128_ps(mid1, mid1, 0x20);
        let permute2 = _mm256_permute2f128_ps(mid0, twiddled, 0x31);
        let permute3 = _mm256_permute2f128_ps(twiddled, twiddled, 0x20);

        let transposed0 = AvxVector::unpackhi_complex([permute0, permute1]);
        let [transposed1, transposed2] = AvxVector::unpack_complex([permute2, permute3]);

        // more size 3 buterflies
        let output_rows = AvxVector::column_butterfly3([transposed0, transposed1, transposed2], self.twiddles_butterfly3);

        // the elements of row 1 are in pretty much the worst possible order, most of these instructions will be about getting them into the right order
        let swapped1 = _mm256_permute_ps(output_rows[1], 0x4E); // swap even and odd complex numbers
        let packed1 = _mm256_permute2f128_ps(swapped1, output_rows[2], 0x21);
        output.store_complex(packed1, 4);

        // merge just the high element of swapped_lo into the high element of row 0
        let zero_swapped1_lo = AvxVector256::merge(AvxVector::zero(), swapped1.lo());
        let packed0 = _mm256_blend_ps(output_rows[0], zero_swapped1_lo, 0xC0);
        output.store_complex(packed0, 0);

        // The last element can just be written on its own
        output.store_partial1_complex(output_rows[2].hi(), 8);
    }
}

pub struct Butterfly12Avx<T> {
    twiddles: [__m256; 2],
    twiddles_butterfly3: __m256,
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl Butterfly12Avx<f32> {
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
            f32::generate_twiddle_factor(2, 12, inverse),
            f32::generate_twiddle_factor(4, 12, inverse),

            // note that these twiddles are deliberately in a weird order, see perform_fft_f32 for why
            f32::generate_twiddle_factor(1, 12, inverse),
            f32::generate_twiddle_factor(2, 12, inverse),
            f32::generate_twiddle_factor(3, 12, inverse),
            f32::generate_twiddle_factor(6, 12, inverse),
        ];
        Self {
            twiddles: [
                twiddles.load_complex(0),
                twiddles.load_complex(4),
            ],
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
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
        let input0_hi = input.load_partial2_complex(1);
        let input_rows = [
            AvxVector256::merge(input0_lo, input0_hi),
            input.load_complex(2),
            input.load_complex(5),
            input.load_complex(8),
        ];

        // butterfly 4's down the columns
        let mut mid = AvxVector::column_butterfly4(input_rows, self.twiddles_butterfly4);

        // Multiply in our twiddle factors. mid2 will be normal, but for mid1 and mid3, we're going to merge the twiddle-able parts into a single vector,
        // and do a single complex multiply on it. this transformation saves a complex multiply and costs nothing,
        // because we needthe second halves of mid1 and mid3 in a single vector for the transpose afterward anyways, so we would have done this permute2f128 operation either way
        mid[2] = AvxVector::mul_complex(mid[2], self.twiddles[0]);
        let merged_mid13 = _mm256_permute2f128_ps(mid[1], mid[3], 0x31);
        let twiddled13 = AvxVector::mul_complex(self.twiddles[1], merged_mid13);
        
        // Transpose our 3x4 array into a 4x3. we're doing a custom transpose here because we have to re-distribute the merged twiddled23 back out, and we can roll that into the transpose to make it free
        let transposed = {
            let permute0 = _mm256_permute2f128_ps(mid[0], mid[2], 0x20);
            let permute1 = _mm256_permute2f128_ps(mid[1], mid[3], 0x20);
            let permute2 = _mm256_permute2f128_ps(mid[0], mid[2], 0x31);
            let permute3 = twiddled13; // normally we'd need to do a permute here, but we can skip it because we already did it for twiddle factors

            let unpacked1 = AvxVector::unpackhi_complex([permute0, permute1]);
            let [unpacked2, unpacked3] = AvxVector::unpack_complex([permute2, permute3]);

            [unpacked1, unpacked2, unpacked3]
        };

        // Do 4 butterfly 4's down the columns of our transposed array
        let output_rows = AvxVector::column_butterfly3(transposed, self.twiddles_butterfly3); 

        output.store_complex(output_rows[0], 0);
        output.store_complex(output_rows[1], 4);
        output.store_complex(output_rows[2], 8);
    }
}
boilerplate_fft_simd_butterfly!(Butterfly12Avx, 12);

pub struct Butterfly16Avx<T> {
    twiddles: [__m256; 3],
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl Butterfly16Avx<f32> {
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
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // Manually unrolling this loop because writing a "for r in 0..4" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        let rows = [
            input.load_complex(0),
            input.load_complex(4),
            input.load_complex(8),
            input.load_complex(12),
        ];

        // We're going to treat our input as a 4x4 2d array. First, do 4 butterfly 4's down the columns of that array.
        let mut mid = AvxVector::column_butterfly4(rows, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid[r] = AvxVector::mul_complex(mid[r], self.twiddles[r - 1]);
        }

        // Transpose our 4x4 array
        let transposed = avx32_utils::transpose_4x4_f32(mid);

        // Do 4 butterfly 4's down the columns of our transposed array
        let output_rows = AvxVector::column_butterfly4(transposed, self.twiddles_butterfly4);

        // Manually unrolling this loop because writing a "for r in 0..4" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        output.store_complex(output_rows[0], 0);
        output.store_complex(output_rows[1], 4);
        output.store_complex(output_rows[2], 8);
        output.store_complex(output_rows[3], 12);
    }
}
boilerplate_fft_simd_butterfly!(Butterfly16Avx, 16);

pub struct Butterfly24Avx<T> {
    twiddles: [__m256; 5],
    twiddles_butterfly3: __m256,
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl Butterfly24Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_interleaved_columns!(6, 4, 0, inverse),
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // Manually unrolling this loop because writing a "for r in 0..6" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        let rows = [
            input.load_complex(0),
            input.load_complex(4),
            input.load_complex(8),
            input.load_complex(12),
            input.load_complex(16),
            input.load_complex(20),
        ];

        // We're going to treat our input as a 4x6 2d array. First, do 4 butterfly 6's down the columns of that array.
        let mut mid = AvxVector256::column_butterfly6(rows, self.twiddles_butterfly3);

        // apply twiddle factors
        for r in 1..6 {
            mid[r] = AvxVector::mul_complex(mid[r], self.twiddles[r - 1]);
        }

        // Transpose our 6x4 array into a 4x6. Sadly this will leave us with 2 garbage columns on the end
        let (transposed0, transposed1) = avx32_utils::transpose_4x6_to_6x4_f32(mid);

        // Do 4 butterfly 4's down the columns of our transposed array
        let output0 = AvxVector::column_butterfly4(transposed0, self.twiddles_butterfly4);
        let output1 = AvxVector::column_butterfly4(transposed1, self.twiddles_butterfly4);

        // the upper two elements of output1 are empty, so only store half the data for it
        for r in 0..4 {
            output.store_complex(output0[r], 6*r);
            output.store_partial2_complex(output1[r].lo(), r*6 + 4);
        }
    }
}
boilerplate_fft_simd_butterfly!(Butterfly24Avx, 24);

pub struct Butterfly27Avx<T> {
    twiddles: [__m256; 4],
    twiddles_butterfly9: [__m256; 3],
    twiddles_butterfly3: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly27Avx, 27);
impl Butterfly27Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_interleaved_columns!(3, 9, 1, inverse),
            twiddles_butterfly9: [
                AvxVector::broadcast_twiddle(1, 9, inverse),
                AvxVector::broadcast_twiddle(2, 9, inverse),
                AvxVector::broadcast_twiddle(4, 9, inverse),
            ],
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // we're going to load our data in a peculiar way. we're going to load the first column on its own as a column of __m128.
        // it's faster to just load the first 2 columns into these m128s than trying to worry about masks, etc, so the second column will piggyback along and we just won't use it
        let mut rows0 = [AvxVector::zero(); 3];
        let mut rows1 = [AvxVector::zero(); 3];
        let mut rows2 = [AvxVector::zero(); 3];
        for r in 0..3 {
            rows0[r] = input.load_partial2_complex(r * 9);
            rows1[r] = input.load_complex(r * 9 + 1);
            rows2[r] = input.load_complex(r * 9 + 5);
        }

        // butterfly 3s down the columns
        let mid0 = AvxVector::column_butterfly3(rows0, self.twiddles_butterfly3.lo());
        let mut mid1 = AvxVector::column_butterfly3(rows1, self.twiddles_butterfly3);
        let mut mid2 = AvxVector::column_butterfly3(rows2, self.twiddles_butterfly3);
        
        // apply twiddle factors
        mid1[1] = AvxVector::mul_complex(mid1[1], self.twiddles[0]);
        mid2[1] = AvxVector::mul_complex(mid2[1], self.twiddles[1]);
        mid1[2] = AvxVector::mul_complex(mid1[2], self.twiddles[2]);
        mid2[2] = AvxVector::mul_complex(mid2[2], self.twiddles[3]);

        // transpose 9x3 to 3x9. this will be a little awkward because of row0 containing garbage data
        let transposed = avx32_utils::transpose_9x3_to_3x9_emptycolumn1_f32(mid0, mid1, mid2);

        // butterfly 9s down the rows
        let output_rows = AvxVector256::column_butterfly9(transposed, self.twiddles_butterfly9, self.twiddles_butterfly3);

        // Our last column is empty, so it's a bit awkward to write out to memory. We could pack it in fewer vectors, but benchmarking shows it's simpler and just as fast to just brute-force it with partial writes
        output.store_partial3_complex(output_rows[0], 0);
        output.store_partial3_complex(output_rows[1], 3);
        output.store_partial3_complex(output_rows[2], 6);
        output.store_partial3_complex(output_rows[3], 9);
        output.store_partial3_complex(output_rows[4], 12);
        output.store_partial3_complex(output_rows[5], 15);
        output.store_partial3_complex(output_rows[6], 18);
        output.store_partial3_complex(output_rows[7], 21);
        output.store_partial3_complex(output_rows[8], 24);
    }
}

pub struct Butterfly32Avx<T> {
    twiddles: [__m256; 6],
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly32Avx, 32);
impl Butterfly32Avx<f32> {
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
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex(8*r);
            rows1[r] = input.load_complex(8*r + 4);
        }

        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        let mut mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4);
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[2*r - 2]);
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[2*r - 1]);
        }

        // Transpose our 8x4 array to an 8x4 array
        let transposed = avx32_utils::transpose_8x4_to_4x8_f32(mid0, mid1);

        // Do 4 butterfly 8's down the columns of our transpsed array
        let output_rows = AvxVector::column_butterfly8(transposed, self.twiddles_butterfly4);

        // Manually unrolling this loop because writing a "for r in 0..8" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        output.store_complex(output_rows[0], 0);
        output.store_complex(output_rows[1], 1 * 4);
        output.store_complex(output_rows[2], 2 * 4);
        output.store_complex(output_rows[3], 3 * 4);
        output.store_complex(output_rows[4], 4 * 4);
        output.store_complex(output_rows[5], 5 * 4);
        output.store_complex(output_rows[6], 6 * 4);
        output.store_complex(output_rows[7], 7 * 4);
    }
}

pub struct Butterfly36Avx<T> {
    twiddles: [__m256; 6],
    twiddles_butterfly9: [__m256; 3],
    twiddles_butterfly3: __m256,
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly36Avx, 36);
impl Butterfly36Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 9, 1, inverse),
            twiddles_butterfly9: [
                AvxVector::broadcast_twiddle(1, 9, inverse),
                AvxVector::broadcast_twiddle(2, 9, inverse),
                AvxVector::broadcast_twiddle(4, 9, inverse),
            ],
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // we're going to load our data in a peculiar way. we're going to load the first column on its own as a column of __m128.
        // it's faster to just load the first 2 columns into these m128s than trying to worry about masks, etc, so the second column will piggyback along and we just won't use it
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        let mut rows2 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = input.load_partial2_complex(r * 9);
            rows1[r] = input.load_complex(r * 9 + 1);
            rows2[r] = input.load_complex(r * 9 + 5);
        }

        // butterfly 4s down the columns
        let mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4.lo());
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);
        let mut mid2 = AvxVector::column_butterfly4(rows2, self.twiddles_butterfly4);
        
        // apply twiddle factors
        for r in 1..4 {
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[2*r - 2]);
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[2*r - 1]);
        }

        // transpose 9x3 to 3x9. this will be a little awkward because of row0 containing garbage data
        let transposed = avx32_utils::transpose_9x4_to_4x9_emptycolumn1_f32(mid0, mid1, mid2);

        // butterfly 9s down the rows
        let output_rows = AvxVector256::column_butterfly9(transposed, self.twiddles_butterfly9, self.twiddles_butterfly3);

        for r in 0..3 {
            output.store_complex(output_rows[r*3], r*12);
            output.store_complex(output_rows[r*3+1], r*12 + 4);
            output.store_complex(output_rows[r*3+2], r*12 + 8);
        }
    }
}

pub struct Butterfly48Avx<T> {
    twiddles: [__m256; 9],
    twiddles_butterfly3: __m256,
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly48Avx, 48);
impl Butterfly48Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 12, 0, inverse),
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        let mut rows2 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex(12*r);
            rows1[r] = input.load_complex(12*r + 4);
            rows2[r] = input.load_complex(12*r + 8);
        }

        // We're going to treat our input as a 12x4 2d array. First, do 12 butterfly 4's down the columns of that array.
        let mut mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4);
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);
        let mut mid2 = AvxVector::column_butterfly4(rows2, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[3*r - 3]);
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[3*r - 2]);
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[3*r - 1]);
        }

        // Transpose our 12x4 array into a 4x12.
        let transposed = avx32_utils::transpose_12x4_to_4x12_f32(mid0, mid1, mid2);

        // Do 4 butterfly 12's down the columns of our transposed array
        let output_rows = AvxVector256::column_butterfly12(transposed, self.twiddles_butterfly3, self.twiddles_butterfly4);

        // Manually unrolling this loop because writing a "for r in 0..12" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        output.store_complex(output_rows[0], 0);
        output.store_complex(output_rows[1], 4);
        output.store_complex(output_rows[2], 8);
        output.store_complex(output_rows[3], 12);
        output.store_complex(output_rows[4], 16);
        output.store_complex(output_rows[5], 20);
        output.store_complex(output_rows[6], 24);
        output.store_complex(output_rows[7], 28);
        output.store_complex(output_rows[8], 32);
        output.store_complex(output_rows[9], 36);
        output.store_complex(output_rows[10], 40);
        output.store_complex(output_rows[11], 44);
    }
}

pub struct Butterfly54Avx<T> {
    twiddles: [__m256; 10],
    twiddles_butterfly9: [__m256; 3],
    twiddles_butterfly9_lo: [__m256; 2],
    twiddles_butterfly3: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly54Avx, 54);
impl Butterfly54Avx<f32> {
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
        let twiddle1 = __m128::broadcast_twiddle(1, 9, inverse);
        let twiddle2 = __m128::broadcast_twiddle(2, 9, inverse);
        let twiddle4 = __m128::broadcast_twiddle(4, 9, inverse);

        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(6, 9, 1, inverse),
            twiddles_butterfly9: [
                AvxVector::broadcast_twiddle(1, 9, inverse),
                AvxVector::broadcast_twiddle(2, 9, inverse),
                AvxVector::broadcast_twiddle(4, 9, inverse),
            ],
            twiddles_butterfly9_lo: [
                AvxVector256::merge(twiddle1, twiddle2),
                AvxVector256::merge(twiddle2, twiddle4),
            ],
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // we're going to load our data in a peculiar way. we're going to load the first column on its own as a column of __m128.
        // it's faster to just load the first 2 columns into these m128s than trying to worry about masks, etc, so the second column will piggyback along and we just won't use it
        // 
        // we have too much data to fit into registers all at once, so split up our data processing so that we entirely finish with one "rows_" array before moving to the next
        let mut rows0 = [AvxVector::zero(); 6];
        for r in 0..3 {
            rows0[r*2] = input.load_partial2_complex(r * 18);
            rows0[r*2+1] = input.load_partial2_complex(r * 18 + 9);
        }
        let mid0 = AvxVector128::column_butterfly6(rows0, self.twiddles_butterfly3);

        // next set of butterfly 6's
        let mut rows1 = [AvxVector::zero(); 6];
        for r in 0..3 {
            rows1[r*2] = input.load_complex(r * 18 + 1);
            rows1[r*2+1] = input.load_complex(r * 18 + 10);
        }
        let mut mid1 = AvxVector256::column_butterfly6(rows1, self.twiddles_butterfly3);
        for r in 1..6 {
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[2*r - 2]);
        }

        // final set of butterfly 6's
        let mut rows2 = [AvxVector::zero(); 6];
        for r in 0..3 {
            rows2[r*2] = input.load_complex(r * 18 + 5);
            rows2[r*2+1] = input.load_complex(r * 18 + 14);
        }
        let mut mid2 = AvxVector256::column_butterfly6(rows2, self.twiddles_butterfly3);
        for r in 1..6 {
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[2*r - 1]);
        }

        // transpose 9x3 to 3x9. this will be a little awkward because of row0 containing garbage data
        let (transposed0, transposed1) = avx32_utils::transpose_9x6_to_6x9_emptycolumn1_f32(mid0, mid1, mid2);

        // butterfly 9s down the rows
        // process the other half
        let output_rows1 = AvxVector128::column_butterfly9(transposed1, self.twiddles_butterfly9_lo, self.twiddles_butterfly3);
        for r in 0..9 {
            output.store_partial2_complex(output_rows1[r], r*6 + 4);
        }
        
        // we have too much data to fit into registers all at once, do one set of butterfly 9's and output them before even starting on the others, to make it easier for the compiler to figure out what to spill
        let output_rows0 = AvxVector256::column_butterfly9(transposed0, self.twiddles_butterfly9, self.twiddles_butterfly3);
        for r in 0..9 {
            output.store_complex(output_rows0[r], r*6);
        }
    }
}

pub struct Butterfly64Avx<T> {
    twiddles: [__m256; 14],
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly64Avx, 64);
impl Butterfly64Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_separated_columns!(8, 8, 0, inverse),
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // We're going to treat our input as a 8x8 2d array. First, do 8 butterfly 8's down the columns of that array.
        // We can't fit the whole problem into AVX registers at once, so we'll have to spill some things.
        // By computing a sizeable chunk and not referencing any of it for a while, we're making it easy for the compiler to decide what to spill
        let mut rows0 = [AvxVector::zero(); 8];
        for r in 0..8 {
            rows0[r] = input.load_complex(8*r);
        }
        let mut mid0 = AvxVector::column_butterfly8(rows0, self.twiddles_butterfly4);
        for r in 1..8 {
            mid0[r] = AvxVector::mul_complex(mid0[r],  self.twiddles[r - 1]);
        }
        
        // One half is done, so the compiler can spill everything above this. Now do the other set of columns
        let mut rows1 = [AvxVector::zero(); 8];
        for r in 0..8 {
            rows1[r] = input.load_complex(8*r + 4);
        }
        let mut mid1 = AvxVector::column_butterfly8(rows1, self.twiddles_butterfly4);
        for r in 1..8 {
            mid1[r] = AvxVector::mul_complex(mid1[r],  self.twiddles[r - 1 + 7]);
        }

        // Transpose our 8x8 array
        let (transposed0, transposed1)  = avx32_utils::transpose_8x8_f32(mid0, mid1);

        // Do 8 butterfly 8's down the columns of our transposed array, and store the results
        // Same thing as above - Do the half of the butterfly 8's separately to give the compiler a better hint about what to spill
        let output0 = AvxVector::column_butterfly8(transposed0, self.twiddles_butterfly4);
        for r in 0..8 {
            output.store_complex(output0[r], 8*r);
        }

        let output1 = AvxVector::column_butterfly8(transposed1, self.twiddles_butterfly4);
        for r in 0..8 {
            output.store_complex(output1[r], 8*r + 4);
        }
    }
}

pub struct Butterfly72Avx<T> {
    twiddles: [__m256; 15],
    twiddles_butterfly4: Rotation90<__m256>,
    twiddles_butterfly3: __m256,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly72Avx, 72);
impl Butterfly72Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_separated_columns!(6, 12, 0, inverse),
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // We're going to treat our input as a 12x6 2d array. First, do butterfly 6's down the columns of that array.
        // We can't fit the whole problem into AVX registers at once, so we'll have to spill some things.
        // By computing a sizeable chunk and not referencing any of it for a while, we're making it easy for the compiler to decide what to spill
        let mut rows0 = [AvxVector::zero(); 6];
        for r in 0..6 {
            rows0[r] = input.load_complex(12*r);
        }
        let mut mid0 = AvxVector256::column_butterfly6(rows0, self.twiddles_butterfly3);
        for r in 1..6 {
            mid0[r] = AvxVector::mul_complex(mid0[r],  self.twiddles[r - 1]);
        }
        
        // One third is done, so the compiler can spill everything above this. Now do the middle set of columns
        let mut rows1 = [AvxVector::zero(); 6];
        for r in 0..6 {
            rows1[r] = input.load_complex(12*r + 4);
        }
        let mut mid1 = AvxVector256::column_butterfly6(rows1, self.twiddles_butterfly3);
        for r in 1..6 {
            mid1[r] = AvxVector::mul_complex(mid1[r],  self.twiddles[r - 1 + 5]);
        }

        // two thirds are done, so the compiler can spill everything above this. Now do the final set of columns
        let mut rows2 = [AvxVector::zero(); 6];
        for r in 0..6 {
            rows2[r] = input.load_complex(12*r + 8);
        }
        let mut mid2 = AvxVector256::column_butterfly6(rows2, self.twiddles_butterfly3);
        for r in 1..6 {
            mid2[r] = AvxVector::mul_complex(mid2[r],  self.twiddles[r - 1 + 10]);
        }

        // Transpose our 12x6 array to 6x12 array
        let (transposed0, transposed1)  = avx32_utils::transpose_12x6_to_6x12_f32(mid0, mid1, mid2);

        // Do butterfly 12's down the columns of our transposed array, and store the results
        // Same thing as above - Do the half of the butterfly 12's separately to give the compiler a better hint about what to spill
        let output0 = AvxVector128::column_butterfly12(transposed0, self.twiddles_butterfly3, self.twiddles_butterfly4);
        for r in 0..12 {
            output.store_partial2_complex(output0[r], 6*r);
        }

        let output1 = AvxVector256::column_butterfly12(transposed1, self.twiddles_butterfly3, self.twiddles_butterfly4);
        for r in 0..12 {
            output.store_complex(output1[r], 6*r + 2);
        }
    }
}

#[allow(non_camel_case_types)]
pub struct Butterfly128Avx<T> {
    twiddles: [__m256; 28],
    twiddles_butterfly16: [__m256; 2],
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly_with_scratch!(Butterfly128Avx, 128);
impl Butterfly128Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_separated_columns!(8, 16, 0, inverse),
            twiddles_butterfly16: [
                AvxVector::broadcast_twiddle(1, 16, inverse),
                AvxVector::broadcast_twiddle(3, 16, inverse),
            ],
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterflies_and_transpose(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        // Treat this 128-length array like a 32x8 2D array, and do butterfly 8's down the columns
        // Then, apply twiddle factors, and finally transpose into the output
        for columnset in 0..4 {
            let mut rows = [AvxVector::zero(); 8];
            for r in 0..8 {
                rows[r] = input.load_complex(columnset*4 + 16*r);
            }
            // apply butterfly 8
            let mut mid = AvxVector::column_butterfly8(rows, self.twiddles_butterfly4);

            // apply twiddle factors
            for r in 1..8 {
                mid[r] = AvxVector::mul_complex(mid[r],  self.twiddles[r - 1 + 7*columnset]);
            }

            // transpose
            let transposed = AvxVector::transpose8_packed(mid);

            // write out
            for i in 0..4 {
                output.store_complex(transposed[i*2], columnset*32 + i*8);
                output.store_complex(transposed[i*2+1], columnset*32 + i*8 + 4);
            }
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn row_butterflies(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // butterfly 16's down the columns of our transposed array
        for columnset in 0usize..2 {
            column_butterfly16_loadfn!(
                |index: usize| input.load_complex(columnset*4 + index*8),
                |data, index| output.store_complex(data, columnset*4 + index*8),
                self.twiddles_butterfly16,
                self.twiddles_butterfly4
            );
        }
    }
}

#[allow(non_camel_case_types)]
pub struct Butterfly256Avx<T> {
    twiddles: [__m256; 56],
    twiddles_butterfly32: [__m256; 6],
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly_with_scratch!(Butterfly256Avx, 256);
impl Butterfly256Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_separated_columns!(8, 32, 0, inverse),
            twiddles_butterfly32: [
                AvxVector::broadcast_twiddle(1, 32, inverse),
                AvxVector::broadcast_twiddle(2, 32, inverse),
                AvxVector::broadcast_twiddle(3, 32, inverse),
                AvxVector::broadcast_twiddle(5, 32, inverse),
                AvxVector::broadcast_twiddle(6, 32, inverse),
                AvxVector::broadcast_twiddle(7, 32, inverse),
            ],
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterflies_and_transpose(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        // process columns
        for columnset in 0..8 {
            let mut rows = [AvxVector::zero(); 8];
            for r in 0..8 {
                rows[r] = input.load_complex(columnset*4 + 32*r);
            }
            let mut mid = AvxVector::column_butterfly8(rows, self.twiddles_butterfly4);
            for r in 1..8 {
                mid[r] = AvxVector::mul_complex(mid[r],  self.twiddles[r - 1 + 7*columnset]);
            }

            // Before writing to the scratch, transpose this chunk of the array
            let transposed = AvxVector::transpose8_packed(mid);

            for i in 0..4 {
                output.store_complex(transposed[i*2], columnset*32 + i*8);
                output.store_complex(transposed[i*2+1], columnset*32 + i*8 + 4);
            }
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn row_butterflies(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // butterfly 32's down the columns
        for columnset in 0..2 {
            column_butterfly32_loadfn!(
                |index: usize| input.load_complex(columnset*4 + index*8),
                |data, index| output.store_complex(data, columnset*4 + index*8),
                self.twiddles_butterfly32,
                self.twiddles_butterfly4
            );
        }
    }
}

pub struct Butterfly512Avx<T> {
    twiddles: [__m256; 120],
    twiddles_butterfly32: [__m256; 6],
    twiddles_butterfly16: [__m256; 2],
    twiddles_butterfly4: Rotation90<__m256>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly_with_scratch!(Butterfly512Avx, 512);
impl Butterfly512Avx<f32> {
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
            twiddles: gen_butterfly_twiddles_separated_columns!(16, 32, 0, inverse),
            twiddles_butterfly32: [
                AvxVector::broadcast_twiddle(1, 32, inverse),
                AvxVector::broadcast_twiddle(2, 32, inverse),
                AvxVector::broadcast_twiddle(3, 32, inverse),
                AvxVector::broadcast_twiddle(5, 32, inverse),
                AvxVector::broadcast_twiddle(6, 32, inverse),
                AvxVector::broadcast_twiddle(7, 32, inverse),
            ],
            twiddles_butterfly16: [
                AvxVector::broadcast_twiddle(1, 16, inverse),
                AvxVector::broadcast_twiddle(3, 16, inverse),
            ],
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterflies_and_transpose(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        // In this function, we're going to trat our input like a 32x16 2D array, and do butterfly-16's down the columns of that array
        // Then, we apply twiddle factors, then transpose into the output

        // Because we're pushing the limit of how much data can for into registers here,
        // we're including bespoke versions of butterfly 16 and transpose 16 that reorganize the operations with the goal of reducing spills to the stack.

        const TWIDDLES_PER_COLUMN : usize = 15;

        // Process FFTs of size 16 down the columns
        for (columnset, twiddle_chunk) in self.twiddles.chunks_exact(TWIDDLES_PER_COLUMN).enumerate() {
            // Sadly we have to use MaybeUninit here. If we init an array like normal with AvxVector::Zero(), the compiler can't seem to figure out that it can
            // eliminate the dead stores of zeroes to the stack. By using uninit here, we avoid those unnecessary writes
            let mut mid_uninit : [MaybeUninit::<__m256>; 16] = MaybeUninit::uninit_array();

            column_butterfly16_loadfn!(
                |index: usize| input.load_complex(columnset*4 + 32*index),
                |data, index| MaybeUninit::first_ptr_mut(&mut mid_uninit).add(index).write(data),
                self.twiddles_butterfly16,
                self.twiddles_butterfly4
            );
            let mid = MaybeUninit::slice_get_ref(&mid_uninit);


            // Apply twiddle factors, transpose, and store. Traditionally we apply all the twiddle factors at once and then do all the transposes at once,
            // But our data is pushing the limit of what we can store in registers, so the idea here is to get the data out the door with as few spills to the stack as possible
            for chunk in 0..4 {
                let twiddled = [
                    if chunk > 0 { AvxVector::mul_complex(mid[4*chunk    ],  twiddle_chunk[4*chunk - 1]) } else { mid[4*chunk] },
                    AvxVector::mul_complex(mid[4*chunk + 1],  twiddle_chunk[4*chunk    ]),
                    AvxVector::mul_complex(mid[4*chunk + 2],  twiddle_chunk[4*chunk + 1]),
                    AvxVector::mul_complex(mid[4*chunk + 3],  twiddle_chunk[4*chunk + 2]),
                ];

                let transposed = AvxVector::transpose4_packed(twiddled);

                output.store_complex(transposed[0], columnset * 64 + 0*16 + 4*chunk);
                output.store_complex(transposed[1], columnset * 64 + 1*16 + 4*chunk);
                output.store_complex(transposed[2], columnset * 64 + 2*16 + 4*chunk);
                output.store_complex(transposed[3], columnset * 64 + 3*16 + 4*chunk);
            }
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn row_butterflies(&self, input: RawSlice<Complex<f32>>, mut output: RawSliceMut<Complex<f32>>) {
        // butterfly 32's down the columns
        for columnset in 0..4 {
            column_butterfly32_loadfn!(
                |index: usize| input.load_complex(columnset*4 + index*16),
                |data, index| output.store_complex(data, columnset*4 + index*16),
                self.twiddles_butterfly32,
                self.twiddles_butterfly4
            );
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

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

    test_avx_butterfly!(test_avx_butterfly5, Butterfly5Avx, 5);
    test_avx_butterfly!(test_avx_butterfly7, Butterfly7Avx, 7);
    test_avx_butterfly!(test_avx_butterfly8, Butterfly8Avx, 8);
    test_avx_butterfly!(test_avx_butterfly9, Butterfly9Avx, 9);
    test_avx_butterfly!(test_avx_butterfly12, Butterfly12Avx, 12);
    test_avx_butterfly!(test_avx_butterfly16, Butterfly16Avx, 16);
    test_avx_butterfly!(test_avx_butterfly24, Butterfly24Avx, 24);
    test_avx_butterfly!(test_avx_butterfly27, Butterfly27Avx, 27);
    test_avx_butterfly!(test_avx_butterfly32, Butterfly32Avx, 32);
    test_avx_butterfly!(test_avx_butterfly36, Butterfly36Avx, 36);
    test_avx_butterfly!(test_avx_butterfly48, Butterfly48Avx, 48);
    test_avx_butterfly!(test_avx_butterfly54, Butterfly54Avx, 54);
    test_avx_butterfly!(test_avx_butterfly64, Butterfly64Avx, 64);
    test_avx_butterfly!(test_avx_butterfly72, Butterfly72Avx, 72);
    test_avx_butterfly!(test_avx_butterfly128, Butterfly128Avx, 128);
    test_avx_butterfly!(test_avx_butterfly256, Butterfly256Avx, 256);
    test_avx_butterfly!(test_avx_butterfly512, Butterfly512Avx, 512);
}
