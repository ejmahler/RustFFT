use std::any::TypeId;
use std::arch::x86_64::*;
use std::sync::Arc;
use num_complex::Complex;

use crate::{common::FftNum, FftDirection};

use crate::array_utils;
use crate::array_utils::workaround_transmute_mut;
use crate::array_utils::DoubleBuf;
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::twiddles;
use crate::{Direction, Fft, Length};

use super::sse_common::{assert_f32, assert_f64};
use super::sse_utils::*;
use super::sse_vector::{SseVector, SseArrayMut};

/* 
This file contains autogenerated butterfly algorithms for small prime-sized FFTs using the SSE instruction set.

NOTE: All of the code in this file was **autogenerated** using the following command in the project root:
    ${command_str}
    
Do not make changes directly to this file. Instead, update the autogeneration script 
(Located at tools/gen_simd_butterflies/src/main.rs) and then re-run the script to generate the new code.

For these sizes, we use a variant of the naive DFT algorithm. Even though this algorithm is O(n^2),
we're able to factor out some redundant computations, and they up being faster than the fancier algorithms.

To generate more or fewer butterfly sizes, simply add or remove numbers from the command above and re-run.
The code generation script will even handle adding or removing sizes from the planner, all you need to do is run the script.
*/

pub const fn prime_butterfly_lens() -> &'static [usize] {
    &[${{ for entry in lengths }}${entry.len}, ${{ endfor }}]
}

/// Safety: The current machine must support the sse4.1 target feature
#[target_feature(enable = "sse4.1")]
pub unsafe fn construct_prime_butterfly<T: FftNum>(len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();
    if id_t == id_f32 {
        match len {
            ${{ for entry in lengths -}}
            ${entry.len} => Arc::new(SseF32Butterfly${entry.len}::new(direction)) as Arc<dyn Fft<T>>,
            ${{endfor -}}
            _ => unimplemented!("Invalid SSE prime butterfly length: {len}"),
        }
    } else if id_t == id_f64 {
        match len {
            ${{ for entry in lengths -}}
            ${entry.len} => Arc::new(SseF64Butterfly${entry.len}::new(direction)) as Arc<dyn Fft<T>>,
            ${{ endfor -}}
            _ => unimplemented!("Invalid SSE prime butterfly length: {len}"),
        }
    } else {
        unimplemented!("Invalid type T for construct_prime_butterfly(...)");
    }
}

#[inline(always)]
fn make_twiddles<const TW: usize, T: FftNum>(len: usize, direction: FftDirection) -> [Complex<T>; TW] {
    let mut i = 1;
    [(); TW].map(|_| {
        let twiddle = twiddles::compute_twiddle(i, len, direction);
        i += 1;
        twiddle
    })
}

${{ for entry in lengths -}}
struct SseF32Butterfly${entry.len}<T> {
    direction: FftDirection,
    twiddles_re: [__m128; ${entry.twiddle_len}],
    twiddles_im: [__m128; ${entry.twiddle_len}],
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_sse_f32_butterfly!(SseF32Butterfly${entry.len}, ${entry.len}, |this: &SseF32Butterfly${entry.len}<_>| this.direction);
boilerplate_fft_sse_common_butterfly!(SseF32Butterfly${entry.len}, ${entry.len}, |this: &SseF32Butterfly${entry.len}<_>| this.direction);
impl<T: FftNum> SseF32Butterfly${entry.len}<T> {
    /// Safety: The current machine must support the SSE4.1 instruction set
    #[target_feature(enable = "sse4.1")]
    unsafe fn new(direction: FftDirection) -> Self {
        assert_f32::<T>();
        let twiddles = make_twiddles(${entry.len}, direction);
        Self {
            direction,
            twiddles_re: twiddles.map(|t| SseVector::broadcast_scalar(t.re)),
            twiddles_im: twiddles.map(|t| SseVector::broadcast_scalar(t.im)),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl SseArrayMut<f32>) {
        let values = read_partial1_complex_to_array!(buffer, {${ entry.loadstore_indexes }});

        let out = self.perform_parallel_fft_direct(values);
        
        write_partial_lo_complex_to_array!(out, buffer, {${ entry.loadstore_indexes }}); 
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_contiguous(&self, mut buffer: impl SseArrayMut<f32>) {
        let input_packed = read_complex_to_array!(buffer, {${ entry.loadstore_indexes_2x }});

        let values = [
${ entry.shuffle_in_str }
        ];

        let out = self.perform_parallel_fft_direct(values);

        let out_packed = [
${ entry.shuffle_out_str }
        ];

        write_complex_to_array_strided!(out_packed, buffer, 2, {${ entry.loadstore_indexes }});
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_parallel_fft_direct(&self, values: [__m128; ${entry.len}]) -> [__m128; ${entry.len}] {
${entry.impl_str}
    }
}

struct SseF64Butterfly${entry.len}<T> {
    direction: FftDirection,
    twiddles_re: [__m128d; ${entry.twiddle_len}],
    twiddles_im: [__m128d; ${entry.twiddle_len}],
    _phantom: std::marker::PhantomData<T>,
}

boilerplate_fft_sse_f64_butterfly!(SseF64Butterfly${entry.len}, ${entry.len}, |this: &SseF64Butterfly${entry.len}<_>| this.direction);
boilerplate_fft_sse_common_butterfly!(SseF64Butterfly${entry.len}, ${entry.len}, |this: &SseF64Butterfly${entry.len}<_>| this.direction);
impl<T: FftNum> SseF64Butterfly${entry.len}<T> {
    /// Safety: The current machine must support the SSE4.1 instruction set
    #[target_feature(enable = "sse4.1")]
    unsafe fn new(direction: FftDirection) -> Self {
        assert_f64::<T>();
        let twiddles = make_twiddles(${entry.len}, direction);
        unsafe {Self {
            direction,
            twiddles_re: twiddles.map(|t| SseVector::broadcast_scalar(t.re)),
            twiddles_im: twiddles.map(|t| SseVector::broadcast_scalar(t.im)),
            _phantom: std::marker::PhantomData,
        }}
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(&self, mut buffer: impl SseArrayMut<f64>) {
        let values = read_complex_to_array!(buffer, {${ entry.loadstore_indexes }});

        let out = self.perform_fft_direct(values);

        write_complex_to_array!(out, buffer, {${ entry.loadstore_indexes }});   
    }

    #[inline(always)]
    pub(crate) unsafe fn perform_fft_direct(&self, values: [__m128d; ${entry.len}]) -> [__m128d; ${entry.len}] {
${entry.impl_str}
    }
}

${{ endfor -}}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    macro_rules! test_butterfly_32_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                assert!(is_x86_feature_detected!("sse4.1"));

                let fwd = unsafe { $struct_name::new(FftDirection::Forward) };
                check_fft_algorithm::<f32>(&fwd, $size, FftDirection::Forward);

                let inv = unsafe { $struct_name::new(FftDirection::Inverse) }; 
                check_fft_algorithm::<f32>(&inv, $size, FftDirection::Inverse);
            }
        };
    }
    macro_rules! test_butterfly_64_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                assert!(is_x86_feature_detected!("sse4.1"));

                let fwd = unsafe { $struct_name::new(FftDirection::Forward) };
                check_fft_algorithm::<f64>(&fwd, $size, FftDirection::Forward);

                let inv = unsafe { $struct_name::new(FftDirection::Inverse) };
                check_fft_algorithm::<f64>(&inv, $size, FftDirection::Inverse);
            }
        };
    }
    ${{ for entry in lengths }}
    test_butterfly_32_func!(test_ssef32_butterfly${entry.len}, SseF32Butterfly${entry.len}, ${entry.len});
    ${{- endfor }}
    ${{ for entry in lengths }}
    test_butterfly_64_func!(test_ssef64_butterfly${entry.len}, SseF64Butterfly${entry.len}, ${entry.len});
    ${{- endfor }}
}
