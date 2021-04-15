//! Compile something that has a scalar butterfly 4

//use rustfft::num_complex::Complex32;
use rustfft::num_complex::Complex64;
use rustfft::FftPlannerScalar;

fn main() {
    let mut planner = FftPlannerScalar::new();
    let fft = planner.plan_fft_forward(4);

    let mut buffer = vec![Complex64::new(0.0, 0.0); 100];
    fft.process(&mut buffer);
}
