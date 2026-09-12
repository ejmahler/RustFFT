use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftDirection};
use std::sync::Arc;
mod config;

use criterion::{criterion_group, criterion_main, Bencher, Criterion};

// Make fft using planner
fn bench_planned_32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f32>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make fft using planner
fn bench_planned_64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

fn bench_radix3_32(b: &mut Bencher, len: usize) {
    let fft: Arc<dyn Fft<f32>> =
        Arc::new(rustfft::algorithm::Radix3::new(len, FftDirection::Forward));

    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f32>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_radix3_64(b: &mut Bencher, len: usize) {
    let fft: Arc<dyn Fft<f32>> =
        Arc::new(rustfft::algorithm::Radix3::new(len, FftDirection::Forward));

    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f32>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    for len in 2..200 {
        c.bench_function(&format!("bench_from2to1024_f32_{len}"), move |b| {
            bench_planned_32(b, len)
        });
        c.bench_function(&format!("bench_from2to1024_f64_{len}"), move |b| {
            bench_planned_64(b, len)
        });
    }

    const POWERS_OF_THREE: &[usize] = &[
        3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969,
    ];
    for &len in POWERS_OF_THREE {
        c.bench_function(
            &format!("bench_power3_planned_scalar_f32_{len:07}"),
            move |b| bench_planned_32(b, len),
        );
        c.bench_function(
            &format!("bench_power3_planned_scalar_f64_{len:07}"),
            move |b| bench_planned_64(b, len),
        );
        c.bench_function(
            &format!("bench_power3_radix3_scalar_f32_{len:07}"),
            move |b| bench_radix3_32(b, len),
        );
        c.bench_function(
            &format!("bench_power3_radix3_scalar_f64_{len:07}"),
            move |b| bench_radix3_64(b, len),
        );
    }
}

criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark
}
criterion_main!(benches);
