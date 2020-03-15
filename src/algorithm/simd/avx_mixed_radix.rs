use std::sync::Arc;
use std::arch::x86_64::*;

use num_complex::Complex;

use common::{FFTnum, verify_length_inline, verify_length_minimum};

use ::{Length, IsInverse, FftInline};
use twiddles;

use super::avx_utils::AvxComplexArrayf32;
use super::avx_utils;

pub struct MixedRadix2xnAvx<T> {
    twiddles: Box<[__m256]>,
    inner_fft: Arc<FftInline<T>>,
    inverse: bool,
}
impl MixedRadix2xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<FftInline<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let half_len = inner_fft.len();
        let len = half_len * 2;

        assert_eq!(len % 8, 0, "MixedRadix2xnAvx requires its FFT length to be a multiple of 8. Got {}", len);

        let eigth_len = half_len / 4;
        let twiddles : Vec<_> = (0..eigth_len).map(|i| {
            let twiddle_chunk = [
                twiddles::single_twiddle(i*4, len, inverse),
                twiddles::single_twiddle(i*4+1, len, inverse),
                twiddles::single_twiddle(i*4+2, len, inverse),
                twiddles::single_twiddle(i*4+3, len, inverse),
            ];
            twiddle_chunk.load_complex_f32(0)
        }).collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
            inner_fft,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let len = self.len();
        let half_len = len / 2;
        let eigth_len = len / 8;

        // process the column FFTs
        for i in 0..eigth_len {
        	let input0 = buffer.load_complex_f32(i * 4); 
        	let input1 = buffer.load_complex_f32(i * 4 + half_len);

        	let (output0, output1_pretwiddle) = avx_utils::column_butterfly2_f32(input0, input1);

        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i), output1_pretwiddle);

        	buffer.store_complex_f32(i * 4, output0);
        	buffer.store_complex_f32(i * 4 + half_len, output1);
        }

        // process the row FFTs
        for chunk in buffer.chunks_exact_mut(half_len) {
        	self.inner_fft.process_inline(chunk, scratch);
        }

        // transpose for the output. make sure to slice off any extra scratch first
        let scratch = &mut scratch[..len];
        transpose::transpose(buffer, scratch, half_len, 2);
        buffer.copy_from_slice(scratch);
    }
}
default impl<T: FFTnum> FftInline<T> for MixedRadix2xnAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for MixedRadix2xnAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    #[inline]
    fn get_required_scratch_len(&self) -> usize {
        self.len()
    }
}

impl<T> Length for MixedRadix2xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len() * 8
    }
}
impl<T> IsInverse for MixedRadix2xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct MixedRadix4xnAvx<T> {
    twiddle_config: avx_utils::Rotate90Config,
    twiddles: Box<[__m256]>,
    inner_fft: Arc<FftInline<T>>,
    inverse: bool,
}
impl MixedRadix4xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<FftInline<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let quarter_len = inner_fft.len();
        let len = quarter_len * 4;

        assert_eq!(len % 16, 0, "MixedRadix4xnAvx requires its FFT length to be a multiple of 16. Got {}", len);

        let sixteenth_len = quarter_len / 4;
        let mut twiddles = Vec::with_capacity(sixteenth_len * 3);
        for x in 0..sixteenth_len {
        	for y in 1..4 {
        		let twiddle_chunk = [
	                twiddles::single_twiddle(y*(x*4), len, inverse),
	                twiddles::single_twiddle(y*(x*4+1), len, inverse),
	                twiddles::single_twiddle(y*(x*4+2), len, inverse),
	                twiddles::single_twiddle(y*(x*4+3), len, inverse),
	            ];
	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        Self {
        	twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
            twiddles: twiddles.into_boxed_slice(),
            inner_fft,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let len = self.len();
        let quarter_len = len / 4;
        let half_len = len / 2;
        let three_quarter_len = len * 3 / 4;
        let sixteenth_len = len / 16;

        // process the column FFTs
        for i in 0..sixteenth_len {
        	let input0 = buffer.load_complex_f32(i * 4); 
        	let input1 = buffer.load_complex_f32(i * 4 + quarter_len);
        	let input2 = buffer.load_complex_f32(i * 4 + half_len);
        	let input3 = buffer.load_complex_f32(i * 4 + three_quarter_len);

        	let (output0, output1_pretwiddle, output2_pretwiddle, output3_pretwiddle) = avx_utils::column_butterfly4_f32(input0, input1, input2, input3, self.twiddle_config);

        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*3), output1_pretwiddle);
        	let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*3+1), output2_pretwiddle);
        	let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*3+2), output3_pretwiddle);

        	buffer.store_complex_f32(i * 4, output0);
        	buffer.store_complex_f32(i * 4 + quarter_len, output1);
        	buffer.store_complex_f32(i * 4 + half_len, output2);
        	buffer.store_complex_f32(i * 4 + three_quarter_len, output3);
        }

        // process the row FFTs
        for chunk in buffer.chunks_exact_mut(quarter_len) {
        	self.inner_fft.process_inline(chunk, scratch);
        }

        // transpose for the output. make sure to slice off any extra scratch first
        let scratch = &mut scratch[..len];
        transpose::transpose(buffer, scratch, quarter_len, 4);
        buffer.copy_from_slice(scratch);
    }
}
default impl<T: FFTnum> FftInline<T> for MixedRadix4xnAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for MixedRadix4xnAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    #[inline]
    fn get_required_scratch_len(&self) -> usize {
        self.len()
    }
}

impl<T> Length for MixedRadix4xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len() * 16 / 3
    }
}
impl<T> IsInverse for MixedRadix4xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct MixedRadix8xnAvx<T> {
    twiddle_config: avx_utils::Rotate90Config,
    twiddles_butterfly8: __m256,
    twiddles: Box<[__m256]>,
    inner_fft: Arc<FftInline<T>>,
    inverse: bool,
}
impl MixedRadix8xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<FftInline<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let eigth_len = inner_fft.len();
        let len = eigth_len * 8;

        assert_eq!(len % 32, 0, "MixedRadix4xnAvx requires its FFT length to be a multiple of 32. Got {}", len);

        let thirtysecond_len = eigth_len / 4;
        let mut twiddles = Vec::with_capacity(thirtysecond_len * 7);
        for x in 0..thirtysecond_len {
        	for y in 1..8 {
        		let twiddle_chunk = [
	                twiddles::single_twiddle(y*(x*4), len, inverse),
	                twiddles::single_twiddle(y*(x*4+1), len, inverse),
	                twiddles::single_twiddle(y*(x*4+2), len, inverse),
	                twiddles::single_twiddle(y*(x*4+3), len, inverse),
	            ];
	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        Self {
        	twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
        	twiddles_butterfly8: avx_utils::broadcast_complex_f32(twiddles::single_twiddle(1, 8, inverse)),
            twiddles: twiddles.into_boxed_slice(),
            inner_fft,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let len = self.len();
        let eigth_len = len / 8;
        let thirtysecond_len = len / 32;

        // process the column FFTs
        for i in 0..thirtysecond_len {
        	let input0 = buffer.load_complex_f32(i * 4); 
        	let input1 = buffer.load_complex_f32(i * 4 + eigth_len);
        	let input2 = buffer.load_complex_f32(i * 4 + eigth_len*2);
        	let input3 = buffer.load_complex_f32(i * 4 + eigth_len*3);
        	let input4 = buffer.load_complex_f32(i * 4 + eigth_len*4);
        	let input5 = buffer.load_complex_f32(i * 4 + eigth_len*5);
        	let input6 = buffer.load_complex_f32(i * 4 + eigth_len*6);
        	let input7 = buffer.load_complex_f32(i * 4 + eigth_len*7);

        	let (output0, mid1, mid2, mid3, mid4, mid5, mid6, mid7) = avx_utils::column_butterfly8_fma_f32(input0, input1, input2, input3, input4, input5, input6, input7, self.twiddles_butterfly8, self.twiddle_config);

        	debug_assert!(self.twiddles.len() >= (i+1) * 7);
        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7), mid1);
        	let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+1), mid2);
        	let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+2), mid3);
        	let output4 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+3), mid4);
        	let output5 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+4), mid5);
        	let output6 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+5), mid6);
        	let output7 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+6), mid7);

        	buffer.store_complex_f32(i * 4, output0);
        	buffer.store_complex_f32(i * 4 + eigth_len, output1);
        	buffer.store_complex_f32(i * 4 + eigth_len*2, output2);
        	buffer.store_complex_f32(i * 4 + eigth_len*3, output3);
        	buffer.store_complex_f32(i * 4 + eigth_len*4, output4);
        	buffer.store_complex_f32(i * 4 + eigth_len*5, output5);
        	buffer.store_complex_f32(i * 4 + eigth_len*6, output6);
        	buffer.store_complex_f32(i * 4 + eigth_len*7, output7);
        }

        // process the row FFTs
        for chunk in buffer.chunks_exact_mut(eigth_len) {
        	self.inner_fft.process_inline(chunk, scratch);
        }

        // transpose for the output. make sure to slice off any extra scratch first
        let scratch = &mut scratch[..len];
        transpose::transpose(buffer, scratch, eigth_len, 8);
        buffer.copy_from_slice(scratch);
    }
}
default impl<T: FFTnum> FftInline<T> for MixedRadix8xnAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for MixedRadix8xnAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    #[inline]
    fn get_required_scratch_len(&self) -> usize {
        self.len()
    }
}

impl<T> Length for MixedRadix8xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len() * 32 / 7
    }
}
impl<T> IsInverse for MixedRadix8xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct MixedRadix16xnAvx<T> {
    twiddle_config: avx_utils::Rotate90Config,
    twiddles_butterfly16: [__m256; 6],
    twiddles: Box<[__m256]>,
    inner_fft: Arc<FftInline<T>>,
    inverse: bool,
}
impl MixedRadix16xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<FftInline<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let sixteenth_len = inner_fft.len();
        let len = sixteenth_len * 16;

        assert_eq!(len % 64, 0, "MixedRadix4xnAvx requires its FFT length to be a multiple of 32. Got {}", len);

        let sixtyfourth_len = sixteenth_len / 4;
        let mut twiddles = Vec::with_capacity(sixtyfourth_len * 15);
        for x in 0..sixtyfourth_len {
        	for y in 1..16 {
        		let twiddle_chunk = [
	                twiddles::single_twiddle(y*(x*4), len, inverse),
	                twiddles::single_twiddle(y*(x*4+1), len, inverse),
	                twiddles::single_twiddle(y*(x*4+2), len, inverse),
	                twiddles::single_twiddle(y*(x*4+3), len, inverse),
	            ];
	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        Self {
        	twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
        	twiddles_butterfly16: [
                avx_utils::broadcast_complex_f32(twiddles::single_twiddle(1, 16, inverse)),
                avx_utils::broadcast_complex_f32(twiddles::single_twiddle(2, 16, inverse)),
                avx_utils::broadcast_complex_f32(twiddles::single_twiddle(3, 16, inverse)),
                avx_utils::broadcast_complex_f32(twiddles::single_twiddle(4, 16, inverse)),
                avx_utils::broadcast_complex_f32(twiddles::single_twiddle(6, 16, inverse)),
                avx_utils::broadcast_complex_f32(twiddles::single_twiddle(9, 16, inverse)),
            ],
            twiddles: twiddles.into_boxed_slice(),
            inner_fft,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let len = self.len();
        let sixteenth_len = len / 16;
        let sixtyfourth_len = len / 64;

        // process the column FFTs
        for i in 0..sixtyfourth_len {
        	let input0  = buffer.load_complex_f32(i * 4); 
        	let input1  = buffer.load_complex_f32(i * 4 + sixteenth_len);
        	let input2  = buffer.load_complex_f32(i * 4 + sixteenth_len*2);
        	let input3  = buffer.load_complex_f32(i * 4 + sixteenth_len*3);
        	let input4  = buffer.load_complex_f32(i * 4 + sixteenth_len*4);
        	let input5  = buffer.load_complex_f32(i * 4 + sixteenth_len*5);
        	let input6  = buffer.load_complex_f32(i * 4 + sixteenth_len*6);
        	let input7  = buffer.load_complex_f32(i * 4 + sixteenth_len*7);
        	let input8  = buffer.load_complex_f32(i * 4 + sixteenth_len*8); 
        	let input9  = buffer.load_complex_f32(i * 4 + sixteenth_len*9);
        	let input10 = buffer.load_complex_f32(i * 4 + sixteenth_len*10);
        	let input11 = buffer.load_complex_f32(i * 4 + sixteenth_len*11);
        	let input12 = buffer.load_complex_f32(i * 4 + sixteenth_len*12);
        	let input13 = buffer.load_complex_f32(i * 4 + sixteenth_len*13);
        	let input14 = buffer.load_complex_f32(i * 4 + sixteenth_len*14);
        	let input15 = buffer.load_complex_f32(i * 4 + sixteenth_len*15);

        	let (output0, mid1, mid2, mid3, mid4, mid5, mid6, mid7, mid8, mid9, mid10, mid11, mid12, mid13, mid14, mid15)
        		= avx_utils::column_butterfly16_fma_f32(
        			input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, self.twiddles_butterfly16, self.twiddle_config
    			);

        	buffer.store_complex_f32(i * 4, output0);

        	debug_assert!(self.twiddles.len() >= (i+1) * 15);
        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15),  mid1);
        	buffer.store_complex_f32(i * 4 + sixteenth_len, output1);
        	let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+1), mid2);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*2, output2);
        	let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+2), mid3);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*3, output3);
        	let output4 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+3), mid4);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*4, output4);
        	let output5 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+4), mid5);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*5, output5);
        	let output6 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+5), mid6);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*6, output6);
        	let output7 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+6), mid7);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*7, output7);
        	let output8 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+7), mid8);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*8, output8);
        	let output9 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+8), mid9);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*9, output9);
        	let output10 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+9), mid10);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*10, output10);
        	let output11 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+10), mid11);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*11, output11);
        	let output12 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+11), mid12);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*12, output12);
        	let output13 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+12), mid13);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*13, output13);
        	let output14 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+13), mid14);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*14, output14);
        	let output15 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+14), mid15);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*15, output15);
        }

        // process the row FFTs
        for chunk in buffer.chunks_exact_mut(sixteenth_len) {
        	self.inner_fft.process_inline(chunk, scratch);
        }

        // transpose for the output. make sure to slice off any extra scratch first
        let scratch = &mut scratch[..len];
        transpose::transpose(buffer, scratch, sixteenth_len, 16);
        buffer.copy_from_slice(scratch);
    }
}
default impl<T: FFTnum> FftInline<T> for MixedRadix16xnAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for MixedRadix16xnAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    #[inline]
    fn get_required_scratch_len(&self) -> usize {
        self.len()
    }
}

impl<T> Length for MixedRadix16xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len() * 64 / 15
    }
}
impl<T> IsInverse for MixedRadix16xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_inline_fft_algorithm;
    use std::sync::Arc;
    use algorithm::*;

    #[test]
    fn test_mixedradix_2xn_avx() {
        for pow in 4..8 {
            let len = 1 << pow;
            test_mixedradix_2xn_avx_with_length(len, false);
            test_mixedradix_2xn_avx_with_length(len, true);
        }
    }

    fn test_mixedradix_2xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 2, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = MixedRadix2xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_mixedradix_4xn_avx() {
        for pow in 4..8 {
            let len = 1 << pow;
            test_mixedradix_4xn_avx_with_length(len, false);
            test_mixedradix_4xn_avx_with_length(len, true);
        }
    }

    fn test_mixedradix_4xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 4, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = MixedRadix4xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_mixedradix_8xn_avx() {
        for pow in 5..9 {
            let len = 1 << pow;
            test_mixedradix_8xn_avx_with_length(len, false);
            test_mixedradix_8xn_avx_with_length(len, true);
        }
    }

    fn test_mixedradix_8xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 8, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = MixedRadix8xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }
    #[test]
    fn test_mixedradix_16xn_avx() {
        for pow in 6..10 {
            let len = 1 << pow;
            test_mixedradix_16xn_avx_with_length(len, false);
            test_mixedradix_16xn_avx_with_length(len, true);
        }
    }

    fn test_mixedradix_16xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 16, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = MixedRadix16xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }
}