#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate rustfft;
extern crate test;

use rustfft::algorithm::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftDirection};
use std::sync::Arc;
use test::Bencher;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_raders(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let inner_fft = planner.plan_fft_forward(len - 1);

    let fft: Arc<Fft<_>> = Arc::new(RadersAlgorithm::new(inner_fft));

    let mut signal = vec![
        Complex {
            re: 0_f64,
            im: 0_f64
        };
        len
    ];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut signal, &mut scratch);
    });
}

#[bench]
fn iaicheck_a_raders_59(b: &mut Bencher) {
    bench_raders(b, 59);
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Bluestein's Algorithm
fn bench_bluesteins(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let inner_fft = planner.plan_fft_forward((len * 2 - 1).checked_next_power_of_two().unwrap());
    let fft: Arc<Fft<f64>> = Arc::new(BluesteinsAlgorithm::new(len, inner_fft));

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn iaicheck_a_bluesteins_59(b: &mut Bencher) {
    bench_bluesteins(b, 59);
}

fn bench_planned(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn iaicheck_ai_planned_58(b: &mut Bencher) {
    bench_planned(b, 58);
}
#[bench]
fn iaicheck_ai_planned_128(b: &mut Bencher) {
    bench_planned(b, 128);
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_radix4(b: &mut Bencher, len: usize) {
    assert!(len % 4 == 0);

    let fft = Radix4::new(len, FftDirection::Forward);

    let mut signal = vec![
        Complex {
            re: 0_f64,
            im: 0_f64
        };
        len
    ];
    let mut scratch = vec![
        Complex {
            re: 0_f64,
            im: 0_f64
        };
        fft.get_inplace_scratch_len()
    ];
    b.iter(|| {
        fft.process_with_scratch(&mut signal, &mut scratch);
    });
}

#[bench]
fn inicheck_b_radix4_4(b: &mut Bencher) {
    bench_radix4(b, 4);
}
#[bench]
fn inicheck_b_radix4_8(b: &mut Bencher) {
    bench_radix4(b, 8);
}
#[bench]
fn inicheck_b_radix4_16(b: &mut Bencher) {
    bench_radix4(b, 16);
}
#[bench]
fn inicheck_b_radix4_32(b: &mut Bencher) {
    bench_radix4(b, 32);
}
#[bench]
fn inicheck_b_planned_4(b: &mut Bencher) {
    bench_planned(b, 4);
}
#[bench]
fn inicheck_b_planned_8(b: &mut Bencher) {
    bench_planned(b, 8);
}
#[bench]
fn inicheck_b_planned_16(b: &mut Bencher) {
    bench_planned(b, 16);
}
#[bench]
fn inicheck_b_planned_32(b: &mut Bencher) {
    bench_planned(b, 32);
}
