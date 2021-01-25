#![allow(non_snake_case)]
extern crate rustfft;

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::sync::Arc;

use iai::black_box;

fn bench_planned_multi_f64(len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>>  = vec![Complex::zero(); 1000 * len];
    let mut output: Vec<Complex<f64>>  = vec![Complex::zero(); 1000 * len];
    let mut scratch: Vec<Complex<f64>>  = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
}

fn bench_planned_multi_f64_setup(len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let buffer: Vec<Complex<f64>> = vec![Complex::zero(); 1000 * len];
    let output: Vec<Complex<f64>>  = vec![Complex::zero(); 1000 * len];
    let scratch: Vec<Complex<f64>>  = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    // make sure we don't optimize away anything..
    black_box(buffer);
    black_box(output);
    black_box(scratch);
}

// all butterflies
fn estimates_10_butterfly_________00000002_0000() { bench_planned_multi_f64(black_box(2)); }
fn estimates_10_butterfly_________00000003_0000() { bench_planned_multi_f64(black_box(3)); }
fn estimates_10_butterfly_________00000004_0000() { bench_planned_multi_f64(black_box(4)); }
fn estimates_10_butterfly_________00000005_0000() { bench_planned_multi_f64(black_box(5)); }
fn estimates_10_butterfly_________00000006_0000() { bench_planned_multi_f64(black_box(6)); }
fn estimates_10_butterfly_________00000007_0000() { bench_planned_multi_f64(black_box(7)); }
fn estimates_10_butterfly_________00000008_0000() { bench_planned_multi_f64(black_box(8)); }
fn estimates_10_butterfly_________00000011_0000() { bench_planned_multi_f64(black_box(11)); }
fn estimates_10_butterfly_________00000013_0000() { bench_planned_multi_f64(black_box(13)); }
fn estimates_10_butterfly_________00000016_0000() { bench_planned_multi_f64(black_box(16)); }
fn estimates_10_butterfly_________00000017_0000() { bench_planned_multi_f64(black_box(17)); }
fn estimates_10_butterfly_________00000019_0000() { bench_planned_multi_f64(black_box(19)); }
fn estimates_10_butterfly_________00000023_0000() { bench_planned_multi_f64(black_box(23)); }
fn estimates_10_butterfly_________00000029_0000() { bench_planned_multi_f64(black_box(29)); }
fn estimates_10_butterfly_________00000031_0000() { bench_planned_multi_f64(black_box(31)); }
fn estimates_10_butterfly_________00000032_0000() { bench_planned_multi_f64(black_box(32)); }

fn estimates_10_butterfly_noop____00000002_0000() { bench_planned_multi_f64_setup(black_box(2)); }
fn estimates_10_butterfly_noop____00000003_0000() { bench_planned_multi_f64_setup(black_box(3)); }
fn estimates_10_butterfly_noop____00000004_0000() { bench_planned_multi_f64_setup(black_box(4)); }
fn estimates_10_butterfly_noop____00000005_0000() { bench_planned_multi_f64_setup(black_box(5)); }
fn estimates_10_butterfly_noop____00000006_0000() { bench_planned_multi_f64_setup(black_box(6)); }
fn estimates_10_butterfly_noop____00000007_0000() { bench_planned_multi_f64_setup(black_box(7)); }
fn estimates_10_butterfly_noop____00000008_0000() { bench_planned_multi_f64_setup(black_box(8)); }
fn estimates_10_butterfly_noop____00000011_0000() { bench_planned_multi_f64_setup(black_box(11)); }
fn estimates_10_butterfly_noop____00000013_0000() { bench_planned_multi_f64_setup(black_box(13)); }
fn estimates_10_butterfly_noop____00000016_0000() { bench_planned_multi_f64_setup(black_box(16)); }
fn estimates_10_butterfly_noop____00000017_0000() { bench_planned_multi_f64_setup(black_box(17)); }
fn estimates_10_butterfly_noop____00000019_0000() { bench_planned_multi_f64_setup(black_box(19)); }
fn estimates_10_butterfly_noop____00000023_0000() { bench_planned_multi_f64_setup(black_box(23)); }
fn estimates_10_butterfly_noop____00000029_0000() { bench_planned_multi_f64_setup(black_box(29)); }
fn estimates_10_butterfly_noop____00000031_0000() { bench_planned_multi_f64_setup(black_box(31)); }
fn estimates_10_butterfly_noop____00000032_0000() { bench_planned_multi_f64_setup(black_box(32)); }

iai::main!(estimates_10_butterfly_________00000002_0000,
    estimates_10_butterfly_noop____00000002_0000,
    estimates_10_butterfly_________00000003_0000,
    estimates_10_butterfly_noop____00000003_0000,
    estimates_10_butterfly_________00000004_0000,
    estimates_10_butterfly_noop____00000004_0000,
    estimates_10_butterfly_________00000005_0000,
    estimates_10_butterfly_noop____00000005_0000,
    estimates_10_butterfly_________00000006_0000,
    estimates_10_butterfly_noop____00000006_0000,
    estimates_10_butterfly_________00000007_0000,
    estimates_10_butterfly_noop____00000007_0000,
    estimates_10_butterfly_________00000008_0000,
    estimates_10_butterfly_noop____00000008_0000,
    estimates_10_butterfly_________00000011_0000,
    estimates_10_butterfly_noop____00000011_0000,
    estimates_10_butterfly_________00000013_0000,
    estimates_10_butterfly_noop____00000013_0000,
    estimates_10_butterfly_________00000016_0000,
    estimates_10_butterfly_noop____00000016_0000,
    estimates_10_butterfly_________00000017_0000,
    estimates_10_butterfly_noop____00000017_0000,
    estimates_10_butterfly_________00000019_0000,
    estimates_10_butterfly_noop____00000019_0000,
    estimates_10_butterfly_________00000023_0000,
    estimates_10_butterfly_noop____00000023_0000,
    estimates_10_butterfly_________00000029_0000,
    estimates_10_butterfly_noop____00000029_0000,
    estimates_10_butterfly_________00000031_0000,
    estimates_10_butterfly_noop____00000031_0000,
    estimates_10_butterfly_________00000032_0000,
    estimates_10_butterfly_noop____00000032_0000
);
