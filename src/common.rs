use num_traits::{FromPrimitive, Signed};
use std::fmt::Debug;

use num_complex::Complex;

use crate::FftDirection;

/// Generic floating point number, implemented for f32 and f64
pub trait FFTnum: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static {
    // two related methodsfor generating twiddle factors. The first is a convenience wrapper around the second.
    fn generate_twiddle_factor(
        index: usize,
        fft_len: usize,
        direction: FftDirection,
    ) -> Complex<Self> {
        Self::generate_twiddle_factor_floatindex(index as f64, fft_len, direction)
    }
    fn generate_twiddle_factor_floatindex(
        index: f64,
        fft_len: usize,
        direction: FftDirection,
    ) -> Complex<Self>;
}

impl<T> FFTnum for T
where
    T: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static,
{
    fn generate_twiddle_factor_floatindex(
        index: f64,
        fft_len: usize,
        direction: FftDirection,
    ) -> Complex<Self> {
        let constant = -2f64 * std::f64::consts::PI / fft_len as f64;
        let angle = constant * index;

        let result = Complex {
            re: Self::from_f64(angle.cos()).unwrap(),
            im: Self::from_f64(angle.sin()).unwrap(),
        };

        match direction {
            FftDirection::Forward => result,
            FftDirection::Inverse => result.conj(),
        }
    }
}

// impl FFTnum for f32 {
// 	fn generate_twiddle_factor_floatindex(index: f64, fft_len: usize, direction: FftDirection) -> Complex<Self> {
// 		let constant = -2f64 * std::f64::consts::PI / fft_len as f64;
// 		let angle = constant * index;

// 	    let result = Complex {
// 	    	re: angle.cos() as f32,
// 	    	im: angle.sin() as f32,
// 	    };

// 	    if inverse {
// 	    	result.conj()
// 	    } else {
// 	    	result
// 	    }
//     }
// }
// impl FFTnum for f64 {
// 	fn generate_twiddle_factor_floatindex(index: f64, fft_len: usize, direction: FftDirection) -> Complex<Self> {
// 		let constant = -2f64 * std::f64::consts::PI / fft_len as f64;
// 		let angle = constant * index;

// 	    let result = Complex {
// 	    	re: angle.cos(),
// 	    	im: angle.sin(),
// 	    };

// 	    if inverse {
// 	    	result.conj()
// 	    } else {
// 	    	result
// 	    }
//     }
// }

macro_rules! boilerplate_fft_oop {
    ($struct_name:ident, $len_fn:expr) => {
        impl<T: FFTnum> Fft<T> for $struct_name<T> {
            fn process_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                assert_eq!(
                    input.len(),
                    self.len(),
                    "Input is the wrong length. Expected {}, got {}",
                    self.len(),
                    input.len()
                );
                assert_eq!(
                    output.len(),
                    self.len(),
                    "Output is the wrong length. Expected {}, got {}",
                    self.len(),
                    output.len()
                );

                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(
                    scratch.len() >= required_scratch,
                    "Scratch is the wrong length. Expected {} or greater, got {}",
                    required_scratch,
                    scratch.len()
                );

                let scratch = &mut scratch[..required_scratch];

                self.perform_fft_out_of_place(input, output, scratch);
            }
            fn process_multi(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                assert!(
                    input.len() % self.len() == 0,
                    "Output is the wrong length. Expected multiple of {}, got {}",
                    self.len(),
                    input.len()
                );
                assert_eq!(
                    input.len(),
                    output.len(),
                    "Output is the wrong length. input = {} output = {}",
                    input.len(),
                    output.len()
                );

                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(
                    scratch.len() >= required_scratch,
                    "Scratch is the wrong length. Expected {} or greater, got {}",
                    required_scratch,
                    scratch.len()
                );

                let scratch = &mut scratch[..required_scratch];

                for (in_chunk, out_chunk) in input
                    .chunks_exact_mut(self.len())
                    .zip(output.chunks_exact_mut(self.len()))
                {
                    self.perform_fft_out_of_place(in_chunk, out_chunk, scratch);
                }
            }
            fn process_inplace_with_scratch(
                &self,
                buffer: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                assert_eq!(
                    buffer.len(),
                    self.len(),
                    "Buffer is the wrong length. Expected {}, got {}",
                    self.len(),
                    buffer.len()
                );

                let required_scratch = self.get_inplace_scratch_len();
                assert!(
                    scratch.len() >= required_scratch,
                    "Scratch is the wrong length. Expected {} or greater, got {}",
                    required_scratch,
                    scratch.len()
                );

                let scratch = &mut scratch[..required_scratch];

                self.perform_fft_out_of_place(buffer, scratch, &mut []);
                buffer.copy_from_slice(scratch);
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                assert_eq!(
                    buffer.len() % self.len(),
                    0,
                    "Buffer is the wrong length. Expected multiple of {}, got {}",
                    self.len(),
                    buffer.len()
                );

                let required_scratch = self.get_inplace_scratch_len();
                assert!(
                    scratch.len() >= required_scratch,
                    "Scratch is the wrong length. Expected {} or greater, got {}",
                    required_scratch,
                    scratch.len()
                );

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
        impl<T> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                self.direction
            }
        }
    };
}

macro_rules! boilerplate_fft {
    ($struct_name:ident, $len_fn:expr, $inplace_scratch_len_fn:expr, $out_of_place_scratch_len_fn:expr) => {
        impl<T: FFTnum> Fft<T> for $struct_name<T> {
            fn process_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                assert_eq!(
                    input.len(),
                    self.len(),
                    "Input is the wrong length. Expected {}, got {}",
                    self.len(),
                    input.len()
                );
                assert_eq!(
                    output.len(),
                    self.len(),
                    "Output is the wrong length. Expected {}, got {}",
                    self.len(),
                    output.len()
                );

                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(
                    scratch.len() >= required_scratch,
                    "Scratch is the wrong length. Expected {} or greater, got {}",
                    required_scratch,
                    scratch.len()
                );

                let scratch = &mut scratch[..required_scratch];

                self.perform_fft_out_of_place(input, output, scratch);
            }
            fn process_multi(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                assert!(
                    input.len() % self.len() == 0,
                    "Output is the wrong length. Expected multiple of {}, got {}",
                    self.len(),
                    input.len()
                );
                assert_eq!(
                    input.len(),
                    output.len(),
                    "Output is the wrong length. input = {} output = {}",
                    input.len(),
                    output.len()
                );

                let required_scratch = self.get_out_of_place_scratch_len();
                assert!(
                    scratch.len() >= required_scratch,
                    "Scratch is the wrong length. Expected {} or greater, got {}",
                    required_scratch,
                    scratch.len()
                );

                let scratch = &mut scratch[..required_scratch];

                for (in_chunk, out_chunk) in input
                    .chunks_exact_mut(self.len())
                    .zip(output.chunks_exact_mut(self.len()))
                {
                    self.perform_fft_out_of_place(in_chunk, out_chunk, scratch);
                }
            }
            fn process_inplace_with_scratch(
                &self,
                buffer: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                assert_eq!(
                    buffer.len(),
                    self.len(),
                    "Buffer is the wrong length. Expected {}, got {}",
                    self.len(),
                    buffer.len()
                );

                let required_scratch = self.get_inplace_scratch_len();
                assert!(
                    scratch.len() >= required_scratch,
                    "Scratch is the wrong length. Expected {} or greater, got {}",
                    required_scratch,
                    scratch.len()
                );

                let scratch = &mut scratch[..required_scratch];

                self.perform_fft_inplace(buffer, scratch);
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                assert_eq!(
                    buffer.len() % self.len(),
                    0,
                    "Buffer is the wrong length. Expected multiple of {}, got {}",
                    self.len(),
                    buffer.len()
                );

                let required_scratch = self.get_inplace_scratch_len();
                assert!(
                    scratch.len() >= required_scratch,
                    "Scratch is the wrong length. Expected {} or greater, got {}",
                    required_scratch,
                    scratch.len()
                );

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
        impl<T: FFTnum> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                self.direction
            }
        }
    };
}
