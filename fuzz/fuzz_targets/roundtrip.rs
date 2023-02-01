//! FFT as used by Symphonia, exposed to untrusted input.
//! This test only checks for crashes, but doesn't perform correctness checks.

#![no_main]

use std::f32;
use std::sync::Arc;

use float_eq::assert_float_eq;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use core::num::FpCategory::*;

use libfuzzer_sys::fuzz_target;

const TOLERANCE: f32 = 1.0;
// more info on floating-point tolerance:
// https://jtempest.github.io/float_eq-rs/book/background/float_comparison_algorithms.html#relative-tolerance-comparison

fuzz_target!(|data: Vec<f32>| {
    // convert raw input to Complex<f32>
    let input: Vec<Complex<f32>> = data.chunks_exact(2).filter_map(|pair|
            // remove NaNs and inf from the input, otherwise the result will not roundtrip
            match (pair[0].classify(), pair[1].classify()) {
                (Nan | Infinite, _) => None,
                (_, Nan | Infinite) => None,
                _ => Some(Complex { re: pair[0], im: pair[1] }),
            }
        ).collect();

    // prepare
    let mut planner = rustfft::FftPlanner::new();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(input.len());
    let ifft: Arc<dyn Fft<f32>> = planner.plan_fft_inverse(input.len());

    let mut result = input.clone();
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    // run FFT and invert it
    fft.process_with_scratch(&mut result, &mut scratch); 
    ifft.process_with_scratch(&mut result, &mut scratch);

    // compensate for the constant factor that the library itself doesn't:
    // https://github.com/ejmahler/RustFFT/issues/110#issuecomment-1409354255
    let factor: f32 = 1.0 / (input.len() as f32);
    result.iter_mut().for_each(|c| *c *= factor);

    // compare results 
    for (orig, recovered) in input.iter().zip(result.iter()) {
        assert_float_eq!(orig.re, recovered.re, rmax <= TOLERANCE);
        assert_float_eq!(orig.im, recovered.im, rmax <= TOLERANCE);
    }
    
});
