use std::sync::Arc;
use crate::Fft;
pub(crate) use avx_vector::AvxVector256;

// Data that most (non-butterfly) SIMD FFT algorithms share
// Algorithms aren't required to use this struct, but it allows for a lot of reduction in code duplication
struct CommonSimdData<T, V> {
    inner_fft: Arc<dyn Fft<T>>,
    twiddles: Box<[V]>,

    len: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,

    inverse: bool,
}

macro_rules! boilerplate_fft_commondata {
    ($struct_name:ident) => (
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
                self.common_data.inplace_scratch_len
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
                self.common_data.outofplace_scratch_len
            }
        }
        impl<T: FFTnum> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                self.common_data.len
            }
        }
        impl<T: FFTnum> IsInverse for $struct_name<T> {
            #[inline(always)]
            fn is_inverse(&self) -> bool {
                self.common_data.inverse
            }
        }
    )
}

mod avx_vector;

mod avx32_utils;
mod avx32_butterflies;

mod avx64_utils;
mod avx64_butterflies;

mod avx_mixed_radix;
mod avx_bluesteins;

pub mod avx_planner;

pub use self::avx32_butterflies::{
    Butterfly5Avx,
    Butterfly7Avx,
    Butterfly8Avx,
    Butterfly9Avx,
    Butterfly12Avx,
    Butterfly16Avx,
    Butterfly24Avx,
    Butterfly27Avx,
    Butterfly32Avx,
    Butterfly36Avx,
    Butterfly48Avx,
    Butterfly54Avx,
    Butterfly64Avx,
    Butterfly72Avx,
};

pub use self::avx64_butterflies::{
    Butterfly5Avx64,
    Butterfly7Avx64,
    Butterfly8Avx64,
    Butterfly9Avx64,
    Butterfly12Avx64,
    Butterfly16Avx64,
    Butterfly18Avx64,
    Butterfly24Avx64,
    Butterfly27Avx64,
    Butterfly32Avx64,
    Butterfly36Avx64
};

pub use self::avx_bluesteins::BluesteinsAvx;
pub use self::avx_mixed_radix::{MixedRadix2xnAvx, MixedRadix3xnAvx, MixedRadix4xnAvx, MixedRadix5xnAvx, MixedRadix6xnAvx, MixedRadix8xnAvx, MixedRadix9xnAvx, MixedRadix12xnAvx, MixedRadix16xnAvx};