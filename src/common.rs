use num_traits::{FromPrimitive, Signed};
use std::fmt::Debug;

use num_complex::Complex;

/// Generic floating point number, implemented for f32 and f64
pub trait FFTnum: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static {
	// two related methodsfor generating twiddle factors. The first is a convenience wrapper around the second.
	fn generate_twiddle_factor(index: usize, fft_len: usize, inverse: bool) -> Complex<Self> {
		Self::generate_twiddle_factor_floatindex(index as f64, fft_len, inverse)
	}
	fn generate_twiddle_factor_floatindex(index: f64, fft_len: usize, inverse: bool) -> Complex<Self>;
}

impl FFTnum for f32 {
	fn generate_twiddle_factor_floatindex(index: f64, fft_len: usize, inverse: bool) -> Complex<Self> {
		let constant = -2f64 * std::f64::consts::PI / fft_len as f64;
		let angle = constant * index;

	    let result = Complex {
	    	re: angle.cos() as f32,
	    	im: angle.sin() as f32,
	    };

	    if inverse {
	    	result.conj()
	    } else {
	    	result
	    }
	}
}
impl FFTnum for f64 {
	fn generate_twiddle_factor_floatindex(index: f64, fft_len: usize, inverse: bool) -> Complex<Self> {
		let constant = -2f64 * std::f64::consts::PI / fft_len as f64;
		let angle = constant * index;

	    let result = Complex {
	    	re: angle.cos(),
	    	im: angle.sin(),
	    };

	    if inverse {
	    	result.conj()
	    } else {
	    	result
	    }
	}
}


#[inline(always)]
pub fn verify_length<T>(input: &[T], output: &[T], expected: usize) {
	assert_eq!(input.len(), expected, "Input is the wrong length. Expected {}, got {}", expected, input.len());
	assert_eq!(output.len(), expected, "Output is the wrong length. Expected {}, got {}", expected, output.len());
}

#[inline(always)]
pub fn verify_length_inline<T>(buffer: &[T], expected: usize) {
	assert_eq!(buffer.len(), expected, "Buffer is the wrong length. Expected {}, got {}", expected, buffer.len());
}

#[inline(always)]
pub fn verify_length_minimum<T>(buffer: &[T], minimum: usize) {
	assert!(buffer.len() >= minimum, "Buffer is the wrong length. Expected {} or greater, got {}", minimum, buffer.len());
}


#[inline(always)]
pub fn verify_length_divisible<T>(input: &[T], output: &[T], expected: usize) {
	assert_eq!(input.len() % expected, 0, "Input is the wrong length. Expected multiple of {}, got {}", expected, input.len());
	assert_eq!(input.len(), output.len(), "Input and output must have the same length. Expected {}, got {}", input.len(), output.len());
}
