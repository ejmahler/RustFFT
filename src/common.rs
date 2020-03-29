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


macro_rules! boilerplate_fft_oop {
    ($struct_name:ident, $len_fn:expr) => (
		impl<T: FFTnum> FFT<T> for $struct_name<T> {
			fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
				assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
				assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
		
				self.perform_fft_out_of_place(input, output);
			}
			fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
				assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
				assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					self.perform_fft_out_of_place(in_chunk, out_chunk);
				}
			}
		}
        impl<T: FFTnum> FftInline<T> for $struct_name<T> {
            fn process_inline(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_required_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                self.perform_fft_out_of_place(buffer, scratch);
                buffer.copy_from_slice(scratch);
            }
            fn process_inline_multi(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_required_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    self.perform_fft_out_of_place(chunk, scratch);
                    chunk.copy_from_slice(scratch);
                }
            }
            #[inline(always)]
            fn get_required_scratch_len(&self) -> usize {
                self.len()
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len_fn(self)
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

// Safety: This macro will call `self::perform_fft_inplace_f32()` which probably has a #[target_feature(enable = "...")] annotation on it.
// Calling functions with that annotation is unsafe, because it doesn't actually check if the CPU has the required features.
// Callers of this macro must guarantee that users can't even obtain an instance of $struct_name if their CPU doesn't have the required CPU features.
#[allow(unused)]
macro_rules! boilerplate_fft_simd_unsafe {
    ($struct_name:ident, $len_fn:expr, $scratch_len_fn:expr) => (
		default impl<T: FFTnum> FftInline<T> for $struct_name<T> {
            fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
				unimplemented!();
            }
            fn process_inline_multi(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                unimplemented!();
            }
            fn get_required_scratch_len(&self) -> usize {
                unimplemented!();
            }
        }
        impl FftInline<f32> for $struct_name<f32> {
            fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_required_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                unsafe { self.perform_fft_inplace_f32(buffer, scratch) };
            }
            fn process_inline_multi(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_required_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    unsafe { self.perform_fft_inplace_f32(chunk, scratch) };
                }
            }
            #[inline(always)]
            fn get_required_scratch_len(&self) -> usize {
                $scratch_len_fn(self)
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len_fn(self)
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