use std::sync::Arc;
use ::Fft;

// Data that most (non-butterfly) SIMD FFT algorithms share
// Algorithms aren't required to use this struct, but it allows for a lot of reduction in code duplication
struct CommonSimdData<T, V> {
    inner_fft: Arc<Fft<T>>,
    twiddles: Box<[V]>,

    len: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,

    inverse: bool,
}

#[allow(unused)]
macro_rules! boilerplate_fft_commondata {
    ($struct_name:ident) => (
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
                self.common_data.inplace_scratch_len
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
                self.common_data.outofplace_scratch_len
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
                self.common_data.len
            }
        }
        impl<T, V> IsInverse for $struct_name<T, V> {
            #[inline(always)]
            fn is_inverse(&self) -> bool {
                self.common_data.inverse
            }
        }
    )
}

mod avx32_utils;
mod avx32_split_radix;
mod avx32_butterflies;

mod avx64_utils;
mod avx64_butterflies;

mod avx_mixed_radix;
mod avx_bluesteins;

pub mod avx_planner;

pub use self::avx32_split_radix::SplitRadixAvx;
pub use self::avx32_butterflies::{MixedRadixAvx4x2, MixedRadixAvx3x3, MixedRadixAvx4x3, MixedRadixAvx4x4, MixedRadixAvx4x6, MixedRadixAvx3x9, MixedRadixAvx4x8, MixedRadixAvx4x9, MixedRadixAvx4x12, MixedRadixAvx6x9, MixedRadixAvx8x8};

pub use self::avx64_butterflies::{MixedRadix64Avx4x2, MixedRadix64Avx3x3, MixedRadix64Avx4x3, MixedRadix64Avx4x4, MixedRadix64Avx3x6, MixedRadix64Avx4x6, MixedRadix64Avx3x9, MixedRadix64Avx4x8};

pub use self::avx_bluesteins::BluesteinsAvx;
pub use self::avx_mixed_radix::{MixedRadix2xnAvx, MixedRadix3xnAvx, MixedRadix4xnAvx, MixedRadix6xnAvx, MixedRadix8xnAvx, MixedRadix9xnAvx, MixedRadix12xnAvx, MixedRadix16xnAvx};