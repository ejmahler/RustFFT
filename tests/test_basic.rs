use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

// Just a very simple test to help debugging
#[test]
fn test_100() {
    let mut p = FftPlanner::<f32>::new();
    let planner: Arc<dyn Fft<f32>> = p.plan_fft_forward(100);

    let mut input: Vec<_> = (0..100).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut output = input.clone();
    let mut scratch = input.clone();
    planner.process_outofplace_with_scratch(&mut input, &mut output, &mut scratch);
}
