use num_traits::{FromPrimitive, Signed};
use std::fmt::Debug;

use std::arch::x86_64::{__m256, __m256d};

use num_complex::Complex;
use crate::algorithm::avx::AvxVector256;

/// Generic floating point number, implemented for f32 and f64
pub trait FFTnum: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static {
	// two related methodsfor generating twiddle factors. The first is a convenience wrapper around the second.
	fn generate_twiddle_factor(index: usize, fft_len: usize, inverse: bool) -> Complex<Self> {
		Self::generate_twiddle_factor_floatindex(index as f64, fft_len, inverse)
	}
    fn generate_twiddle_factor_floatindex(index: f64, fft_len: usize, inverse: bool) -> Complex<Self>;
    
    // todo: cfg this on x86_64, or on a feature flag
    type AvxType : AvxVector256<ScalarType=Self>;
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
    
    type AvxType = __m256;
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
    
    type AvxType = __m256d;
}

macro_rules! boilerplate_fft_oop {
    ($struct_name:ident, $len_fn:expr) => (
		impl<T: FFTnum> Fft<T> for $struct_name<T> {
			fn process_with_scratch(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
				assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());

                let scratch = &mut scratch[..required_scratch];
		
				self.perform_fft_out_of_place(input, output, scratch);
			}
			fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
				assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());

                let scratch = &mut scratch[..required_scratch];
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					self.perform_fft_out_of_place(in_chunk, out_chunk, scratch);
				}
			}
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                self.perform_fft_out_of_place(buffer, scratch, &mut []);
                buffer.copy_from_slice(scratch);
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    self.perform_fft_out_of_place(chunk, scratch, &mut []);
                    chunk.copy_from_slice(scratch);
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                self.len()
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
                0
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

macro_rules! boilerplate_fft {
    ($struct_name:ident, $len_fn:expr, $inplace_scratch_len_fn:expr, $out_of_place_scratch_len_fn:expr) => (
		impl<T: FFTnum> Fft<T> for $struct_name<T> {
			fn process_with_scratch(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
				assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				self.perform_fft_out_of_place(input, output, scratch);
			}
			fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
				assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					self.perform_fft_out_of_place(in_chunk, out_chunk, scratch);
				}
			}
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                self.perform_fft_inplace(buffer, scratch);
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    self.perform_fft_inplace(chunk, scratch);
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                $inplace_scratch_len_fn(self)
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
                $out_of_place_scratch_len_fn(self)
            }
        }
        impl<T: FFTnum> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len_fn(self)
            }
        }
        impl<T: FFTnum> IsInverse for $struct_name<T> {
            #[inline(always)]
            fn is_inverse(&self) -> bool {
                self.inverse
            }
        }
    )
}


#[allow(unused)]
macro_rules! boilerplate_fft_simd_f32 {
    ($struct_name:ident, $len_fn:expr, $inplace_scratch_len_fn:expr, $out_of_place_scratch_len_fn:expr) => (
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
            fn process_with_scratch(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				self.perform_fft_out_of_place_f32(input, output, scratch);
            }
            fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					self.perform_fft_out_of_place_f32(in_chunk, out_chunk, scratch);
				}
            }
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                self.perform_fft_inplace_f32(buffer, scratch);
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    self.perform_fft_inplace_f32(chunk, scratch);
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                $inplace_scratch_len_fn(self)
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
                $out_of_place_scratch_len_fn(self)
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


#[allow(unused)]
macro_rules! boilerplate_fft_simd {
    ($struct_name:ident, $len_fn:expr) => (
		default impl<T: FFTnum, V> Fft<T> for $struct_name<T, V> {
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
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                self.inplace_scratch_len
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
               self.outofplace_scratch_len
            }
        }
        impl Fft<f32> for $struct_name<f32, __m256> {
            fn process_with_scratch(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				self.perform_fft_out_of_place_f32(input, output, scratch);
            }
            fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					self.perform_fft_out_of_place_f32(in_chunk, out_chunk, scratch);
				}
            }
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                self.perform_fft_inplace_f32(buffer, scratch);
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    self.perform_fft_inplace_f32(chunk, scratch);
                }
            }
        }
        impl Fft<f64> for $struct_name<f64, __m256d> {
            fn process_with_scratch(&self, input: &mut [Complex<f64>], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
                assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				self.perform_fft_out_of_place_f64(input, output, scratch);
            }
            fn process_multi(&self, input: &mut [Complex<f64>], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
                assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					self.perform_fft_out_of_place_f64(in_chunk, out_chunk, scratch);
				}
            }
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                self.perform_fft_inplace_f64(buffer, scratch);
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    self.perform_fft_inplace_f64(chunk, scratch);
                }
            }
        }
        impl<T, V> Length for $struct_name<T, V> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len_fn(self)
            }
        }
        impl<T, V> IsInverse for $struct_name<T, V> {
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
    ($struct_name:ident, $len_fn:expr, $inplace_scratch_len_fn:expr, $out_of_place_scratch_len_fn:expr) => (
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
            fn process_with_scratch(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				unsafe { self.perform_fft_out_of_place_f32(input, output, scratch) };
            }
            fn process_multi(&self, input: &mut [Complex<f32>], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
                assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
                
                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
		
				for (in_chunk, out_chunk) in input.chunks_exact_mut(self.len()).zip(output.chunks_exact_mut(self.len())) {
					unsafe { self.perform_fft_out_of_place_f32(in_chunk, out_chunk, scratch) };
				}
            }
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                unsafe { self.perform_fft_inplace_f32(buffer, scratch) };
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());

                let required_scratch = self.get_inplace_scratch_len();
                assert!(scratch.len() >= required_scratch, "Scratch is the wrong length. Expected {} or greater, got {}", required_scratch, scratch.len());
        
                let scratch = &mut scratch[..required_scratch];
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    unsafe { self.perform_fft_inplace_f32(chunk, scratch) };
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                $inplace_scratch_len_fn(self)
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
                $out_of_place_scratch_len_fn(self)
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