use std::any::TypeId;

use crate::fft_helper::{fft_helper_immut, fft_helper_inplace, fft_helper_outofplace};

// Helper function to assert we have the right float type
pub fn assert_f32<T: 'static>() {
    let id_f32 = TypeId::of::<f32>();
    let id_t = TypeId::of::<T>();
    assert!(id_t == id_f32, "Wrong float type, must be f32");
}

// Helper function to assert we have the right float type
pub fn assert_f64<T: 'static>() {
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();
    assert!(id_t == id_f64, "Wrong float type, must be f64");
}

// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
macro_rules! interleave_complex_f32 {
    ($input:ident, $offset:literal, { $($idx:literal),* }) => {
        [
        $(
            extract_lo_lo_f32($input[$idx], $input[$idx+$offset]),
            extract_hi_hi_f32($input[$idx], $input[$idx+$offset]),
        )*
        ]
    }
}

// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
// This statement:
// ```
// let values = separate_interleaved_complex_f32!(input, {0, 2, 4});
// ```
// is equivalent to:
// ```
// let values = [
//    extract_lo_lo_f32(input[0], input[1]),
//    extract_lo_lo_f32(input[2], input[3]),
//    extract_lo_lo_f32(input[4], input[5]),
//    extract_hi_hi_f32(input[0], input[1]),
//    extract_hi_hi_f32(input[2], input[3]),
//    extract_hi_hi_f32(input[4], input[5]),
// ];
macro_rules! separate_interleaved_complex_f32 {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            extract_lo_lo_f32($input[$idx], $input[$idx+1]),
        )*
        $(
            extract_hi_hi_f32($input[$idx], $input[$idx+1]),
        )*
        ]
    }
}

// A wrapper for the FFT helper functions that make sure the entire thing happens with the benefit of the SSE target feature,
// so that things like loading twiddle factor registers etc can be lifted out of the loop
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_fft_helper_immut<T>(
    input: &[T],
    output: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    chunk_fn: impl FnMut(&[T], &mut [T], &mut [T]),
) {
    fft_helper_immut(
        input,
        output,
        scratch,
        chunk_size,
        required_scratch,
        chunk_fn,
    )
}

// A wrapper for the FFT helper functions that make sure the entire thing happens with the benefit of the SSE target feature,
// so that things like loading twiddle factor registers etc can be lifted out of the loop
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_fft_helper_outofplace<T>(
    input: &mut [T],
    output: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    chunk_fn: impl FnMut(&mut [T], &mut [T], &mut [T]),
) {
    fft_helper_outofplace(
        input,
        output,
        scratch,
        chunk_size,
        required_scratch,
        chunk_fn,
    )
}

// A wrapper for the FFT helper functions that make sure the entire thing happens with the benefit of the SSE target feature,
// so that things like loading twiddle factor registers etc can be lifted out of the loop
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_fft_helper_inplace<T>(
    buffer: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    chunk_fn: impl FnMut(&mut [T], &mut [T]),
) {
    fft_helper_inplace(buffer, scratch, chunk_size, required_scratch, chunk_fn)
}
