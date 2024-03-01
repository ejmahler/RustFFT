#![no_main]

use std::sync::Arc;

use libfuzzer_sys::fuzz_target;

use rustfft::num_complex::Complex;
use rustfft::Fft;

fuzz_target!(|data: Vec<f32>| {
    let mut input = floats_to_complex(data);
    let mut planner = rustfft::FftPlannerSse::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(input.len());
    fft.process(&mut input);
});

fn floats_to_complex(floats: Vec<f32>) -> Vec<Complex<f32>> {
    floats.chunks_exact(2).map(|pair| Complex { re: pair[0], im: pair[1] } ).collect()
}