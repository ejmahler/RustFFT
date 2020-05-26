use std::marker::PhantomData;
use std::arch::x86_64::*;

use num_complex::Complex;

use crate::common::FFTnum;

use crate::{Length, IsInverse, Fft};

use crate::array_utils::{RawSlice, RawSliceMut};
use super::avx64_utils;
use super::avx_vector::{AvxVector, AvxVector128, AvxVector256, Rotation90, AvxArray, AvxArrayMut};

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
        let mut twiddles = [AvxVector::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index / TWIDDLE_VECTOR_COLS) + 1;
            let x = (index % TWIDDLE_VECTOR_COLS) * 2 + $skip_cols;

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
        const TWIDDLE_VECTOR_COLS : usize = TWIDDLE_COLS / 2;
        const TWIDDLE_VECTOR_COUNT : usize = TWIDDLE_VECTOR_COLS*TWIDDLE_ROWS;
        let mut twiddles = [AvxVector::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index % TWIDDLE_ROWS) + 1;
            let x = (index / TWIDDLE_ROWS) * 2 + $skip_cols;

            twiddles[index] = AvxVector::make_mixedradix_twiddle_chunk(x, y, FFT_LEN, $inverse);
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
        let input12 = input.load_complex(1);
        let input34 = input.load_complex(3);
        
        // swap elements for inputs 3 and 4
        let input43 = AvxVector::reverse_complex_elements(input34);

        // do some prep work before we can start applying twiddle factors
        let [sum12, diff43] = AvxVector::column_butterfly2([input12, input43]);

        let rotation = AvxVector::make_rotation90(true);
        let rotated43 = AvxVector::rotate90(diff43, rotation);

        let [mid14, mid23] = avx64_utils::transpose_2x2_f64([sum12, rotated43]);

        // to compute the first output, compute the sum of all elements. mid14[0] and mid23[0] already have the sum of 1+4 and 2+3 respectively, so if we add them, we'll get the sum of all 4
        let sum1234 = AvxVector::add(_mm256_castpd256_pd128(mid14), _mm256_castpd256_pd128(mid23));
        let output0 = AvxVector::add(_mm256_castpd256_pd128(input0), sum1234);
        
        // apply twiddle factors
        let twiddled_outer14 = AvxVector::mul(mid14, self.twiddles[0]);
        let twiddled_inner14 = AvxVector::mul(mid14, self.twiddles[1]);
        let twiddled14 = AvxVector::fmadd(mid23, self.twiddles[1], twiddled_outer14);
        let twiddled23 = AvxVector::fmadd(mid23, self.twiddles[2], twiddled_inner14);

        // unpack the data for the last butterfly 2
        let [twiddled12, twiddled43] = avx64_utils::transpose_2x2_f64([twiddled14, twiddled23]);
        let [output12, output43] = AvxVector::column_butterfly2([twiddled12, twiddled43]);

        // swap the elements in output43 before writing them out, and add the first input to everything
        let final12  = AvxVector::add(input0, output12);
        let output34 = AvxVector::reverse_complex_elements(output43);
        let final34  = AvxVector::add(input0, output34);

        output.store_partial1_complex(output0, 0);
        output.store_complex(final12, 1);
        output.store_complex(final34, 3);
    }
}


pub struct Butterfly7Avx64<T> {
    twiddles: [__m256d; 5],
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly7Avx64, 7);
impl Butterfly7Avx64<f64> {
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
        let twiddle1 = f64::generate_twiddle_factor(1, 7, inverse);
        let twiddle2 = f64::generate_twiddle_factor(2, 7, inverse);
        let twiddle3 = f64::generate_twiddle_factor(3, 7, inverse);
        Self {
            twiddles: [
                _mm256_set_pd(twiddle1.im, twiddle1.im, twiddle1.re, twiddle1.re),
                _mm256_set_pd(twiddle2.im, twiddle2.im, twiddle2.re, twiddle2.re),
                _mm256_set_pd(twiddle3.im, twiddle3.im, twiddle3.re, twiddle3.re),
                _mm256_set_pd(-twiddle3.im, -twiddle3.im, twiddle3.re, twiddle3.re),
                _mm256_set_pd(-twiddle1.im, -twiddle1.im, twiddle1.re, twiddle1.re),
            ],
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let input0 = _mm256_loadu2_m128d(input.as_ptr() as *const f64, input.as_ptr() as *const f64);
        let input12 = input.load_complex(1);
        let input3 = input.load_partial1_complex(3);
        let input4 = input.load_partial1_complex(4);
        let input56 = input.load_complex(5);

        // reverse the order of input56
        let input65 = AvxVector::reverse_complex_elements(input56);

        // do some prep work before we can start applying twiddle factors
        let [sum12, diff65] = AvxVector::column_butterfly2([input12, input65]);
        let [sum3, diff4]   = AvxVector::column_butterfly2([input3, input4]);

        let rotation = AvxVector::make_rotation90(true);
        let rotated65 = AvxVector::rotate90(diff65, rotation);
        let rotated4  = AvxVector::rotate90(diff4, rotation.lo());

        let [mid16, mid25] = avx64_utils::transpose_2x2_f64([sum12, rotated65]);
        let mid34 = AvxVector128::merge(sum3, rotated4);

        // to compute the first output, compute the sum of all elements. mid16[0], mid25[0], and mid34[0] already have the sum of 1+6, 2+5 and 3+4 respectively, so if we add them, we'll get 1+2+3+4+5+6
        let output0_left  = AvxVector::add(mid16.lo(),  mid25.lo());
        let output0_right = AvxVector::add(input0.lo(), mid34.lo());
        let output0 = AvxVector::add(output0_left, output0_right);
        output.store_partial1_complex(output0, 0);
        
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
        let [twiddled12, twiddled65] = avx64_utils::transpose_2x2_f64([twiddled16, twiddled25]);

        // we can save one add if we add input0 to twiddled3 now. normally we'd add input0 to the final output, but the arrangement of data makes that a little awkward
        let twiddled03 = AvxVector::add(twiddled34.lo(), input0.lo());

        let [output12, output65] = AvxVector::column_butterfly2([twiddled12, twiddled65]);
        let final12  = AvxVector::add(output12, input0);
        let output56 = AvxVector::reverse_complex_elements(output65);
        let final56  = AvxVector::add(output56, input0);

        let [final3, final4] = AvxVector::column_butterfly2([twiddled03, twiddled34.hi()]);

        
        output.store_complex(final12, 1);
        output.store_partial1_complex(final3, 3);
        output.store_partial1_complex(final4, 4);
        output.store_complex(final56, 5);
    }
}


pub struct Butterfly8Avx64<T> {
    twiddles: [__m256d; 2],
    twiddles_butterfly4: Rotation90<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly8Avx64, 8);
impl Butterfly8Avx64<f64> {
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
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let row0 = input.load_complex(0);
        let row1 = input.load_complex(2);
        let row2 = input.load_complex(4);
        let row3 = input.load_complex(6);

        // Do our butterfly 2's down the columns of a 4x2 array
        let [mid0, mid2] = AvxVector::column_butterfly2([row0, row2]);
        let [mid1, mid3] = AvxVector::column_butterfly2([row1, row3]);

        let mid2_twiddled = AvxVector::mul_complex(mid2, self.twiddles[0]);
        let mid3_twiddled = AvxVector::mul_complex(mid3, self.twiddles[1]);

        // transpose to a 2x4 array
        let transposed = avx64_utils::transpose_4x2_to_2x4_f64([mid0, mid2_twiddled], [mid1, mid3_twiddled]);

        // butterfly 4's down the transposed array
        let output_rows = AvxVector::column_butterfly4(transposed, self.twiddles_butterfly4);

        output.store_complex(output_rows[0], 0);
        output.store_complex(output_rows[1], 2);
        output.store_complex(output_rows[2], 4);
        output.store_complex(output_rows[3], 6);
    }
}

pub struct Butterfly9Avx64<T> {
    twiddles: [__m256d; 2],
    twiddles_butterfly3: __m256d,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly9Avx64, 9);
impl Butterfly9Avx64<f64> {
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
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 3x4 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second column as full.
        let mut rows0 = [AvxVector::zero(); 3];
        let mut rows1 = [AvxVector::zero(); 3];

        for r in 0..3 {
            rows0[r] = input.load_partial1_complex(3*r);
            rows1[r] = input.load_complex(3*r+1);
        }

        // do butterfly 4's down the columns
        let mid0 = AvxVector::column_butterfly3(rows0, self.twiddles_butterfly3.lo());
        let mut mid1 = AvxVector::column_butterfly3(rows1, self.twiddles_butterfly3);

        // apply twiddle factors
        for n in 1..3 {
            mid1[n] = AvxVector::mul_complex(mid1[n], self.twiddles[n - 1]);
        }

        // transpose our 3x3 array
        let (transposed0, transposed1) = avx64_utils::transpose_3x3_f64(mid0, mid1);

        // apply butterfly 3's down the columns
        let output0 = AvxVector::column_butterfly3(transposed0, self.twiddles_butterfly3.lo());
        let output1 = AvxVector::column_butterfly3(transposed1, self.twiddles_butterfly3);

        for r in 0..3 {
            output.store_partial1_complex(output0[r], 3*r);
            output.store_complex(output1[r], 3*r+1);
        }
    }
}



pub struct Butterfly12Avx64<T> {
    twiddles: [__m256d; 3],
    twiddles_butterfly3: __m256d,
    twiddles_butterfly4: Rotation90<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly12Avx64, 12);
impl Butterfly12Avx64<f64> {
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
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 3x4 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second column as full.
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];

        for n in 0..4 {
            rows0[n] = input.load_partial1_complex(n * 3);
            rows1[n] = input.load_complex(n * 3 + 1);
        }

        // do butterfly 4's down the columns
        let mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4.lo());
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);

        // apply twiddle factors
        for n in 1..4 {
            mid1[n] = AvxVector::mul_complex(mid1[n], self.twiddles[n - 1]);
        }

        // transpose our 3x4 array to a 4x3 array
        let (transposed0, transposed1) = avx64_utils::transpose_3x4_to_4x3_f64(mid0, mid1);

        // apply butterfly 3's down the columns
        let output0 = AvxVector::column_butterfly3(transposed0, self.twiddles_butterfly3);
        let output1 = AvxVector::column_butterfly3(transposed1, self.twiddles_butterfly3);

        for r in 0..3 {
            output.store_complex(output0[r], 4*r);
            output.store_complex(output1[r], 4*r+2);
        }
    }
}


pub struct Butterfly16Avx64<T> {
    twiddles: [__m256d; 6],
    twiddles_butterfly4: Rotation90<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly16Avx64, 16);
impl Butterfly16Avx64<f64> {
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
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex(4*r);
            rows1[r] = input.load_complex(4*r + 2);
        }

        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        let mut mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4);
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[2*r - 2]);
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[2*r - 1]);
        }

        // Transpose our 4x4 array
        let (transposed0, transposed1) = avx64_utils::transpose_4x4_f64(mid0, mid1);

        // Butterfly 4's down columns of the transposed array
        let output0 = AvxVector::column_butterfly4(transposed0, self.twiddles_butterfly4);
        let output1 = AvxVector::column_butterfly4(transposed1, self.twiddles_butterfly4);

        for r in 0..4 {
            output.store_complex(output0[r], 4*r);
            output.store_complex(output1[r], 4*r+2);
        }
    }
}

pub struct Butterfly18Avx64<T> {
    twiddles: [__m256d; 5],
    twiddles_butterfly3: __m256d,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly18Avx64, 18);
impl Butterfly18Avx64<f64> {
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
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 3x6 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second column as full.
        let mut rows0 = [AvxVector::zero(); 6];
        let mut rows1 = [AvxVector::zero(); 6];
        for n in 0..6 {
            rows0[n] = input.load_partial1_complex(n * 3);
            rows1[n] = input.load_complex(n * 3 + 1);
        }

        // do butterfly 6's down the columns
        let mid0 = AvxVector128::column_butterfly6(rows0, self.twiddles_butterfly3);
        let mut mid1 = AvxVector256::column_butterfly6(rows1, self.twiddles_butterfly3);

        // apply twiddle factors
        for n in 1..6 {
            mid1[n] = AvxVector::mul_complex(mid1[n], self.twiddles[n - 1]);
        }

        // transpose our 3x4 array to a 4x3 array
        let (transposed0, transposed1, transposed2) = avx64_utils::transpose_3x6_to_6x3_f64(mid0, mid1);

        // apply butterfly 3's down the columns
        let output0 = AvxVector::column_butterfly3(transposed0, self.twiddles_butterfly3);
        let output1 = AvxVector::column_butterfly3(transposed1, self.twiddles_butterfly3);
        let output2 = AvxVector::column_butterfly3(transposed2, self.twiddles_butterfly3);

        for r in 0..3 {
            output.store_complex(output0[r], 6*r);
            output.store_complex(output1[r], 6*r+2);
            output.store_complex(output2[r], 6*r+4);
        }
    }
}

pub struct Butterfly24Avx64<T> {
    twiddles: [__m256d; 9],
    twiddles_butterfly3: __m256d,
    twiddles_butterfly4: Rotation90<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly24Avx64, 24);
impl Butterfly24Avx64<f64> {
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
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            twiddles_butterfly4: AvxVector::make_rotation90(inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        let mut rows2 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex(6*r);
            rows1[r] = input.load_complex(6*r + 2);
            rows2[r] = input.load_complex(6*r + 4);
        }

        // We're going to treat our input as a 6x4 2d array. First, do 6 butterfly 4's down the columns of that array.
        let mut mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4);
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);
        let mut mid2 = AvxVector::column_butterfly4(rows2, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[3*r - 3]);
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[3*r - 2]);
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[3*r - 1]);
        }
        
        // Transpose our 6x4 array
        let (transposed0, transposed1) = avx64_utils::transpose_6x4_to_4x6_f64(mid0, mid1, mid2);

        // Butterfly 6's down columns of the transposed array
        let output0 = AvxVector256::column_butterfly6(transposed0, self.twiddles_butterfly3);
        let output1 = AvxVector256::column_butterfly6(transposed1, self.twiddles_butterfly3);

        for r in 0..6 {
            output.store_complex(output0[r], 4*r);
            output.store_complex(output1[r], 4*r+2);
        }
    }
}

pub struct Butterfly27Avx64<T> {
    twiddles: [__m256d; 8],
    twiddles_butterfly9: [__m256d; 3],
    twiddles_butterfly9_lo: [__m256d; 2],
    twiddles_butterfly3: __m256d,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly27Avx64, 27);
impl Butterfly27Avx64<f64> {
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
        let twiddle1 = __m128d::broadcast_twiddle(1, 9, inverse);
        let twiddle2 = __m128d::broadcast_twiddle(2, 9, inverse);
        let twiddle4 = __m128d::broadcast_twiddle(4, 9, inverse);

        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(3, 9, 1, inverse),
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
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 3x6 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second 2 columns as full.
        let mut rows0 = [AvxVector::zero(); 3];
        for n in 0..3 {
            rows0[n] = input.load_partial1_complex(n*9);
        }
        let mid0 = AvxVector::column_butterfly3(rows0, self.twiddles_butterfly3.lo());

        // we're going to load our input as a 3x6 array. We have to load 3 columns, which is a little awkward
        // We can reduce the number of multiplies we do if we load the first column as half-width and the second 2 columns as full.
        let mut rows1 = [AvxVector::zero(); 3];
        let mut rows2 = [AvxVector::zero(); 3];
        for n in 0..3 {
            rows1[n] = input.load_complex(n*9 + 1);
            rows2[n] = input.load_complex(n*9 + 3);
        }
        let mut mid1 = AvxVector::column_butterfly3(rows1, self.twiddles_butterfly3);
        let mut mid2 = AvxVector::column_butterfly3(rows2, self.twiddles_butterfly3);
        for r in 1..3 {
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[4*r - 4]);
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[4*r - 3]);
        }

        // we're going to load our input as a 3x6 array. We have to load 3 columns, which is a little awkward
        let mut rows3 = [AvxVector::zero(); 3];
        let mut rows4 = [AvxVector::zero(); 3];
        for n in 0..3 {
            rows3[n] = input.load_complex(n*9 + 5);
            rows4[n] = input.load_complex(n*9 + 7);
        }
        let mut mid3 = AvxVector::column_butterfly3(rows3, self.twiddles_butterfly3);
        let mut mid4 = AvxVector::column_butterfly3(rows4, self.twiddles_butterfly3);
        for r in 1..3 {
            mid3[r] = AvxVector::mul_complex(mid3[r], self.twiddles[4*r - 2]);
            mid4[r] = AvxVector::mul_complex(mid4[r], self.twiddles[4*r - 1]);
        }

        // transpose our 9x3 array to a 3x9 array
        let (transposed0, transposed1) = avx64_utils::transpose_9x3_to_3x9_f64(mid0, mid1, mid2, mid3, mid4);

        // apply butterfly 9's down the columns
        let output0 = AvxVector128::column_butterfly9(transposed0, self.twiddles_butterfly9_lo, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_partial1_complex(output0[r*3], 9*r);
            output.store_partial1_complex(output0[r*3+1], 9*r+3);
            output.store_partial1_complex(output0[r*3+2], 9*r+6);
        }

        let output1 = AvxVector256::column_butterfly9(transposed1, self.twiddles_butterfly9, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex(output1[r*3], 9*r+1);
            output.store_complex(output1[r*3+1], 9*r+4);
            output.store_complex(output1[r*3+2], 9*r+7);
        }
    }
}



pub struct Butterfly32Avx64<T> {
    twiddles: [__m256d; 12],
    twiddles_butterfly4: Rotation90<__m256d>,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly32Avx64, 32);
impl Butterfly32Avx64<f64> {
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
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        // We can't fit the whole problem into AVX registers at once, so we'll have to spill some things.
        // By computing half of the problem and then not referencing any of it for a while, we're making it easy for the compiler to decide what to spill
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = input.load_complex(8*r);
            rows1[r] = input.load_complex(8*r + 2);
        }
        let mut mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4);
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);
        for r in 1..4 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[4 * r - 4]);
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[4 * r - 3]);
        }

        // One half is done, so the compiler can spill everything above this. Now do the other set of columns
        let mut rows2 = [AvxVector::zero(); 4];
        let mut rows3 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows2[r] = input.load_complex(8*r + 4);
            rows3[r] = input.load_complex(8*r + 6);
        }
        let mut mid2 = AvxVector::column_butterfly4(rows2, self.twiddles_butterfly4);
        let mut mid3 = AvxVector::column_butterfly4(rows3, self.twiddles_butterfly4);
        for r in 1..4 {
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[4 * r - 2]);
            mid3[r] = AvxVector::mul_complex(mid3[r], self.twiddles[4 * r - 1]);
        }

        // Transpose our 8x4 array to a 4x8 array
        let (transposed0, transposed1) = avx64_utils::transpose_8x4_to_4x8_f64(mid0, mid1, mid2, mid3);

        // Do 4 butterfly 8's down columns of the transposed array
        // Same thing as above - Do the half of the butterfly 8's separately to give the compiler a better hint about what to spill
        let output0 = AvxVector::column_butterfly8(transposed0, self.twiddles_butterfly4);
        for r in 0..8 {
            output.store_complex(output0[r], 4*r);
        }
        let output1 = AvxVector::column_butterfly8(transposed1, self.twiddles_butterfly4);
        for r in 0..8 {
            output.store_complex(output1[r], 4*r + 2);
        }
    }
}


pub struct Butterfly36Avx64<T> {
    twiddles: [__m256d; 15],
    twiddles_butterfly3: __m256d,
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly36Avx64, 36);
impl Butterfly36Avx64<f64> {
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
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, inverse),
            inverse: inverse,
            _phantom: PhantomData,
        }
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f64(&self, input: RawSlice<Complex<f64>>, mut output: RawSliceMut<Complex<f64>>) {
        // we're going to load our input as a 6x6 array
        let mut rows0 = [AvxVector::zero(); 6];
        for n in 0..6 {
            rows0[n] = input.load_complex(n*6);
        }
        let mut mid0 = AvxVector256::column_butterfly6(rows0, self.twiddles_butterfly3);
        for r in 1..6 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[r - 1]);
        }

        // we're going to load our input as a 6x6 array
        let mut rows1 = [AvxVector::zero(); 6];
        for n in 0..6 {
            rows1[n] = input.load_complex(n*6+2);
        }
        let mut mid1 = AvxVector256::column_butterfly6(rows1, self.twiddles_butterfly3);
        for r in 1..6 {
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[r + 4]);
        }

        // we're going to load our input as a 6x6 array
        let mut rows2 = [AvxVector::zero(); 6];
        for n in 0..6 {
            rows2[n] = input.load_complex(n*6+4);
        }
        let mut mid2 = AvxVector256::column_butterfly6(rows2, self.twiddles_butterfly3);
        for r in 1..6 {
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[r + 9]);
        }


        // transpose our 6x6 array
        let (transposed0, transposed1, transposed2) = avx64_utils::transpose_6x6_f64(mid0, mid1, mid2);


        // apply butterfly 6's down the columns
        let output0 = AvxVector256::column_butterfly6(transposed0, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex(output0[r*2], 12*r);
            output.store_complex(output0[r*2+1], 12*r+6);
        }

        let output1 = AvxVector256::column_butterfly6(transposed1, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex(output1[r*2], 12*r+2);
            output.store_complex(output1[r*2+1], 12*r+8);
        }

        let output2 = AvxVector256::column_butterfly6(transposed2, self.twiddles_butterfly3);
        for r in 0..3 {
            output.store_complex(output2[r*2], 12*r+4);
            output.store_complex(output2[r*2+1], 12*r+10);
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

    test_avx_butterfly!(test_avx_butterfly5_f64, Butterfly5Avx64, 5);
    test_avx_butterfly!(test_avx_butterfly7_f64, Butterfly7Avx64, 7);
    test_avx_butterfly!(test_avx_mixedradix4x2_f64, Butterfly8Avx64, 8);
    test_avx_butterfly!(test_avx_mixedradix3x3_f64, Butterfly9Avx64, 9);
    test_avx_butterfly!(test_avx_mixedradix4x3_f64, Butterfly12Avx64, 12);
    test_avx_butterfly!(test_avx_mixedradix4x4_f64, Butterfly16Avx64, 16);
    test_avx_butterfly!(test_avx_mixedradix3x6_f64, Butterfly18Avx64, 18);
    test_avx_butterfly!(test_avx_mixedradix4x6_f64, Butterfly24Avx64, 24);
    test_avx_butterfly!(test_avx_mixedradix3x9_f64, Butterfly27Avx64, 27);
    test_avx_butterfly!(test_avx_mixedradix4x8_f64, Butterfly32Avx64, 32);
    test_avx_butterfly!(test_avx_mixedradix6x6_f64, Butterfly36Avx64, 36);
}
