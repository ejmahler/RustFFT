//! Show how to use an `FFT` object from multiple threads

use std::sync::Arc;
use std::thread;

use rustfft::FftPlanner;
use rustfft::FftRealToComplex;
use rustfft::{algorithm::real_to_complex::RealToComplexEven, num_complex::Complex32};

fn main() {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(100);
    let fft2 = Arc::new(RealToComplexEven::new(Arc::clone(&fft)));
    let threads: Vec<thread::JoinHandle<_>> = (0..2)
        .map(|_| {
            let fft_copy = Arc::clone(&fft);
            let fft2_copy = Arc::clone(&fft2);
            thread::spawn(move || {
                let mut buffer = vec![Complex32::new(0.0, 0.0); 100];
                fft_copy.process(&mut buffer);

                let mut input = vec![0.0; 100];
                let mut output = vec![Complex32::new(0.0, 0.0); 51];
                let mut scratch = vec![0.0; fft2_copy.get_scratch_len()];

                fft2_copy.process(&mut input, &mut output, &mut scratch);
            })
        })
        .collect();

    for thread in threads {
        thread.join().unwrap();
    }
}
