use num_traits::{FromPrimitive, Signed};
use std::fmt::Debug;

use std::sync::Arc;
use algorithm::butterflies::{FFTButterfly, Butterfly8};
use FFT;
use algorithm::mixed_radix_cxn::*;

/// Generic floating point number, implemented for f32 and f64
pub trait FFTnum: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static {
	fn make_butterfly16_as_fft(inverse: bool) -> Arc<dyn FFT<Self>>;
	fn make_butterfly16_as_butterfly(inverse: bool) -> Arc<dyn FFTButterfly<Self>>;
	fn make_4xn(inner_fft: Arc<dyn FFT<Self>>) -> Arc<dyn FFT<Self>>;
}

impl FFTnum for f32 {
	fn make_butterfly16_as_fft(inverse: bool) -> Arc<dyn FFT<f32>> {
		let has_avx = is_x86_feature_detected!("avx");
    	let has_fma = is_x86_feature_detected!("fma");
    	if has_avx && has_fma {
    		Arc::new(MixedRadix4x4Avx::new(inverse))
    	} else {
    		Arc::new(Butterfly8::new(inverse))
    	}
	}
	fn make_butterfly16_as_butterfly(inverse: bool) -> Arc<dyn FFTButterfly<f32>> {
		let has_avx = is_x86_feature_detected!("avx");
    	let has_fma = is_x86_feature_detected!("fma");
    	if has_avx && has_fma {
    		Arc::new(MixedRadix4x4Avx::new(inverse))
    	} else {
    		Arc::new(Butterfly8::new(inverse))
    	}
	}
	fn make_4xn(inner_fft: Arc<dyn FFT<f32>>) -> Arc<dyn FFT<f32>> {
		let has_avx = is_x86_feature_detected!("avx");
    	let has_fma = is_x86_feature_detected!("fma");
    	if has_avx && has_fma {
    		Arc::new(MixedRadix4xnAvx::new(inner_fft))
    	} else {
    		Arc::new(MixedRadix4xN::new(inner_fft))
    	}
	}
}
impl FFTnum for f64 {
	fn make_butterfly16_as_fft(inverse: bool) -> Arc<dyn FFT<f64>> {
		Arc::new(Butterfly8::new(inverse))
	}
	fn make_butterfly16_as_butterfly(inverse: bool) -> Arc<dyn FFTButterfly<f64>> {
		Arc::new(Butterfly8::new(inverse))
	}
	fn make_4xn(inner_fft: Arc<dyn FFT<f64>>) -> Arc<dyn FFT<f64>> {
		Arc::new(MixedRadix4xN::new(inner_fft))
	}
}


#[inline(always)]
pub fn verify_length<T>(input: &[T], output: &[T], expected: usize) {
	assert_eq!(input.len(), expected, "Input is the wrong length. Expected {}, got {}", expected, input.len());
	assert_eq!(output.len(), expected, "Output is the wrong length. Expected {}, got {}", expected, output.len());
}


#[inline(always)]
pub fn verify_length_divisible<T>(input: &[T], output: &[T], expected: usize) {
	assert_eq!(input.len() % expected, 0, "Input is the wrong length. Expected multiple of {}, got {}", expected, input.len());
	assert_eq!(input.len(), output.len(), "Input and output must have the same length. Expected {}, got {}", input.len(), output.len());
}
