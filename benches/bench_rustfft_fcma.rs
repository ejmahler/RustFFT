use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::sync::Arc;
mod config;

use criterion::{criterion_group, criterion_main, Bencher, Criterion};


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_f32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerFcma::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);
    assert_eq!(fft.len(), len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerFcma::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);
    assert_eq!(fft.len(), len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length.
/// Run the fft on a 10*len vector, similar to how the butterflies are often used.
fn bench_planned_multi_f32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerFcma::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_multi_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerFcma::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// All butterflies
fn fcma_butterfly32_02(b: &mut Bencher) { bench_planned_multi_f32(b, 2);}
fn fcma_butterfly32_03(b: &mut Bencher) { bench_planned_multi_f32(b, 3);}
fn fcma_butterfly32_04(b: &mut Bencher) { bench_planned_multi_f32(b, 4);}
fn fcma_butterfly32_05(b: &mut Bencher) { bench_planned_multi_f32(b, 5);}
fn fcma_butterfly32_06(b: &mut Bencher) { bench_planned_multi_f32(b, 6);}
fn fcma_butterfly32_07(b: &mut Bencher) { bench_planned_multi_f32(b, 7);}
fn fcma_butterfly32_08(b: &mut Bencher) { bench_planned_multi_f32(b, 8);}
fn fcma_butterfly32_09(b: &mut Bencher) { bench_planned_multi_f32(b, 9);}
fn fcma_butterfly32_10(b: &mut Bencher) { bench_planned_multi_f32(b, 10);}
fn fcma_butterfly32_11(b: &mut Bencher) { bench_planned_multi_f32(b, 11);}
fn fcma_butterfly32_12(b: &mut Bencher) { bench_planned_multi_f32(b, 12);}
fn fcma_butterfly32_13(b: &mut Bencher) { bench_planned_multi_f32(b, 13);}
fn fcma_butterfly32_15(b: &mut Bencher) { bench_planned_multi_f32(b, 15);}
fn fcma_butterfly32_16(b: &mut Bencher) { bench_planned_multi_f32(b, 16);}
fn fcma_butterfly32_17(b: &mut Bencher) { bench_planned_multi_f32(b, 17);}
fn fcma_butterfly32_19(b: &mut Bencher) { bench_planned_multi_f32(b, 19);}
fn fcma_butterfly32_23(b: &mut Bencher) { bench_planned_multi_f32(b, 23);}
fn fcma_butterfly32_29(b: &mut Bencher) { bench_planned_multi_f32(b, 29);}
fn fcma_butterfly32_31(b: &mut Bencher) { bench_planned_multi_f32(b, 31);}
fn fcma_butterfly32_32(b: &mut Bencher) { bench_planned_multi_f32(b, 32);}


fn fcma_prime_butterfly32_07(b: &mut Bencher) { bench_planned_multi_f32(b, 7);}
fn fcma_prime_butterfly32_11(b: &mut Bencher) { bench_planned_multi_f32(b, 11);}
fn fcma_prime_butterfly32_13(b: &mut Bencher) { bench_planned_multi_f32(b, 13);}
fn fcma_prime_butterfly32_17(b: &mut Bencher) { bench_planned_multi_f32(b, 17);}
fn fcma_prime_butterfly32_19(b: &mut Bencher) { bench_planned_multi_f32(b, 19);}
fn fcma_prime_butterfly32_23(b: &mut Bencher) { bench_planned_multi_f32(b, 23);}
fn fcma_prime_butterfly32_29(b: &mut Bencher) { bench_planned_multi_f32(b, 29);}
fn fcma_prime_butterfly32_31(b: &mut Bencher) { bench_planned_multi_f32(b, 31);}

fn fcma_prime_butterfly64_07(b: &mut Bencher) { bench_planned_multi_f64(b, 7);}
fn fcma_prime_butterfly64_11(b: &mut Bencher) { bench_planned_multi_f64(b, 11);}
fn fcma_prime_butterfly64_13(b: &mut Bencher) { bench_planned_multi_f64(b, 13);}
fn fcma_prime_butterfly64_17(b: &mut Bencher) { bench_planned_multi_f64(b, 17);}
fn fcma_prime_butterfly64_19(b: &mut Bencher) { bench_planned_multi_f64(b, 19);}
fn fcma_prime_butterfly64_23(b: &mut Bencher) { bench_planned_multi_f64(b, 23);}
fn fcma_prime_butterfly64_29(b: &mut Bencher) { bench_planned_multi_f64(b, 29);}
fn fcma_prime_butterfly64_31(b: &mut Bencher) { bench_planned_multi_f64(b, 31);}

fn fcma_butterfly64_02(b: &mut Bencher) { bench_planned_multi_f64(b, 2);}
fn fcma_butterfly64_03(b: &mut Bencher) { bench_planned_multi_f64(b, 3);}
fn fcma_butterfly64_04(b: &mut Bencher) { bench_planned_multi_f64(b, 4);}
fn fcma_butterfly64_05(b: &mut Bencher) { bench_planned_multi_f64(b, 5);}
fn fcma_butterfly64_06(b: &mut Bencher) { bench_planned_multi_f64(b, 6);}
fn fcma_butterfly64_07(b: &mut Bencher) { bench_planned_multi_f64(b, 7);}
fn fcma_butterfly64_08(b: &mut Bencher) { bench_planned_multi_f64(b, 8);}
fn fcma_butterfly64_09(b: &mut Bencher) { bench_planned_multi_f64(b, 9);}
fn fcma_butterfly64_10(b: &mut Bencher) { bench_planned_multi_f64(b, 10);}
fn fcma_butterfly64_11(b: &mut Bencher) { bench_planned_multi_f64(b, 11);}
fn fcma_butterfly64_12(b: &mut Bencher) { bench_planned_multi_f64(b, 12);}
fn fcma_butterfly64_13(b: &mut Bencher) { bench_planned_multi_f64(b, 13);}
fn fcma_butterfly64_15(b: &mut Bencher) { bench_planned_multi_f64(b, 15);}
fn fcma_butterfly64_16(b: &mut Bencher) { bench_planned_multi_f64(b, 16);}
fn fcma_butterfly64_17(b: &mut Bencher) { bench_planned_multi_f64(b, 17);}
fn fcma_butterfly64_19(b: &mut Bencher) { bench_planned_multi_f64(b, 19);}
fn fcma_butterfly64_23(b: &mut Bencher) { bench_planned_multi_f64(b, 23);}
fn fcma_butterfly64_29(b: &mut Bencher) { bench_planned_multi_f64(b, 29);}
fn fcma_butterfly64_31(b: &mut Bencher) { bench_planned_multi_f64(b, 31);}
fn fcma_butterfly64_32(b: &mut Bencher) { bench_planned_multi_f64(b, 32);}

// Powers of 2
fn fcma_planned32_p2_00000064(b: &mut Bencher) { bench_planned_f32(b, 64); }
fn fcma_planned32_p2_00000128(b: &mut Bencher) { bench_planned_f32(b, 128); }
fn fcma_planned32_p2_00000256(b: &mut Bencher) { bench_planned_f32(b, 256); }
fn fcma_planned32_p2_00000512(b: &mut Bencher) { bench_planned_f32(b, 512); }
fn fcma_planned32_p2_00001024(b: &mut Bencher) { bench_planned_f32(b, 1024); }
fn fcma_planned32_p2_00002048(b: &mut Bencher) { bench_planned_f32(b, 2048); }
fn fcma_planned32_p2_00004096(b: &mut Bencher) { bench_planned_f32(b, 4096); }
fn fcma_planned32_p2_00016384(b: &mut Bencher) { bench_planned_f32(b, 16384); }
fn fcma_planned32_p2_00065536(b: &mut Bencher) { bench_planned_f32(b, 65536); }
fn fcma_planned32_p2_01048576(b: &mut Bencher) { bench_planned_f32(b, 1048576); }

fn fcma_planned64_p2_00000064(b: &mut Bencher) { bench_planned_f64(b, 64); }
fn fcma_planned64_p2_00000128(b: &mut Bencher) { bench_planned_f64(b, 128); }
fn fcma_planned64_p2_00000256(b: &mut Bencher) { bench_planned_f64(b, 256); }
fn fcma_planned64_p2_00000512(b: &mut Bencher) { bench_planned_f64(b, 512); }
fn fcma_planned64_p2_00001024(b: &mut Bencher) { bench_planned_f64(b, 1024); }
fn fcma_planned64_p2_00002048(b: &mut Bencher) { bench_planned_f64(b, 2048); }
fn fcma_planned64_p2_00004096(b: &mut Bencher) { bench_planned_f64(b, 4096); }
fn fcma_planned64_p2_00016384(b: &mut Bencher) { bench_planned_f64(b, 16384); }
fn fcma_planned64_p2_00065536(b: &mut Bencher) { bench_planned_f64(b, 65536); }
fn fcma_planned64_p2_01048576(b: &mut Bencher) { bench_planned_f64(b, 1048576); }


// Powers of 7
fn fcma_planned32_p7_00343(b: &mut Bencher) { bench_planned_f32(b,   343); }
fn fcma_planned32_p7_02401(b: &mut Bencher) { bench_planned_f32(b,  2401); }
fn fcma_planned32_p7_16807(b: &mut Bencher) { bench_planned_f32(b, 16807); }

fn fcma_planned64_p7_00343(b: &mut Bencher) { bench_planned_f64(b,   343); }
fn fcma_planned64_p7_02401(b: &mut Bencher) { bench_planned_f64(b,  2401); }
fn fcma_planned64_p7_16807(b: &mut Bencher) { bench_planned_f64(b, 16807); }

// Prime lengths
fn fcma_planned32_prime_0149(b: &mut Bencher)     { bench_planned_f32(b,  149); }
fn fcma_planned32_prime_0151(b: &mut Bencher)     { bench_planned_f32(b,  151); }
fn fcma_planned32_prime_0251(b: &mut Bencher)     { bench_planned_f32(b,  251); }
fn fcma_planned32_prime_0257(b: &mut Bencher)     { bench_planned_f32(b,  257); }
fn fcma_planned32_prime_2017(b: &mut Bencher)     { bench_planned_f32(b,  2017); }
fn fcma_planned32_prime_2879(b: &mut Bencher)     { bench_planned_f32(b,  2879); }
fn fcma_planned32_prime_65521(b: &mut Bencher)    { bench_planned_f32(b, 65521); }
fn fcma_planned32_prime_746497(b: &mut Bencher)   { bench_planned_f32(b,746497); }

fn fcma_planned64_prime_0149(b: &mut Bencher)     { bench_planned_f64(b,  149); }
fn fcma_planned64_prime_0151(b: &mut Bencher)     { bench_planned_f64(b,  151); }
fn fcma_planned64_prime_0251(b: &mut Bencher)     { bench_planned_f64(b,  251); }
fn fcma_planned64_prime_0257(b: &mut Bencher)     { bench_planned_f64(b,  257); }
fn fcma_planned64_prime_2017(b: &mut Bencher)     { bench_planned_f64(b,  2017); }
fn fcma_planned64_prime_2879(b: &mut Bencher)     { bench_planned_f64(b,  2879); }
fn fcma_planned64_prime_65521(b: &mut Bencher)    { bench_planned_f64(b, 65521); }
fn fcma_planned64_prime_746497(b: &mut Bencher)   { bench_planned_f64(b,746497); }

// small mixed composites
fn fcma_planned32_composite_000018(b: &mut Bencher) { bench_planned_f32(b,  00018); }
fn fcma_planned32_composite_000360(b: &mut Bencher) { bench_planned_f32(b,  00360); }
fn fcma_planned32_composite_001200(b: &mut Bencher) { bench_planned_f32(b,  01200); }
fn fcma_planned32_composite_044100(b: &mut Bencher) { bench_planned_f32(b,  44100); }
fn fcma_planned32_composite_048000(b: &mut Bencher) { bench_planned_f32(b,  48000); }
fn fcma_planned32_composite_046656(b: &mut Bencher) { bench_planned_f32(b,  46656); }

fn fcma_planned64_composite_000018(b: &mut Bencher) { bench_planned_f64(b,  00018); }
fn fcma_planned64_composite_000360(b: &mut Bencher) { bench_planned_f64(b,  00360); }
fn fcma_planned64_composite_001200(b: &mut Bencher) { bench_planned_f64(b,  01200); }
fn fcma_planned64_composite_044100(b: &mut Bencher) { bench_planned_f64(b,  44100); }
fn fcma_planned64_composite_048000(b: &mut Bencher) { bench_planned_f64(b,  48000); }
fn fcma_planned64_composite_046656(b: &mut Bencher) { bench_planned_f64(b,  46656); }




fn criterion_benchmark(c: &mut Criterion) {
    config::register_benchmarks!(
        c,
        fcma_butterfly32_02,
        fcma_butterfly32_03,
        fcma_butterfly32_04,
        fcma_butterfly32_05,
        fcma_butterfly32_06,
        fcma_butterfly32_07,
        fcma_butterfly32_08,
        fcma_butterfly32_09,
        fcma_butterfly32_10,
        fcma_butterfly32_11,
        fcma_butterfly32_12,
        fcma_butterfly32_13,
        fcma_butterfly32_15,
        fcma_butterfly32_16,
        fcma_butterfly32_17,
        fcma_butterfly32_19,
        fcma_butterfly32_23,
        fcma_butterfly32_29,
        fcma_butterfly32_31,
        fcma_butterfly32_32,
        fcma_prime_butterfly32_07,
        fcma_prime_butterfly32_11,
        fcma_prime_butterfly32_13,
        fcma_prime_butterfly32_17,
        fcma_prime_butterfly32_19,
        fcma_prime_butterfly32_23,
        fcma_prime_butterfly32_29,
        fcma_prime_butterfly32_31,
        fcma_prime_butterfly64_07,
        fcma_prime_butterfly64_11,
        fcma_prime_butterfly64_13,
        fcma_prime_butterfly64_17,
        fcma_prime_butterfly64_19,
        fcma_prime_butterfly64_23,
        fcma_prime_butterfly64_29,
        fcma_prime_butterfly64_31,
        fcma_butterfly64_02,
        fcma_butterfly64_03,
        fcma_butterfly64_04,
        fcma_butterfly64_05,
        fcma_butterfly64_06,
        fcma_butterfly64_07,
        fcma_butterfly64_08,
        fcma_butterfly64_09,
        fcma_butterfly64_10,
        fcma_butterfly64_11,
        fcma_butterfly64_12,
        fcma_butterfly64_13,
        fcma_butterfly64_15,
        fcma_butterfly64_16,
        fcma_butterfly64_17,
        fcma_butterfly64_19,
        fcma_butterfly64_23,
        fcma_butterfly64_29,
        fcma_butterfly64_31,
        fcma_butterfly64_32,
        fcma_planned32_p2_00000064,
        fcma_planned32_p2_00000128,
        fcma_planned32_p2_00000256,
        fcma_planned32_p2_00000512,
        fcma_planned32_p2_00001024,
        fcma_planned32_p2_00002048,
        fcma_planned32_p2_00004096,
        fcma_planned32_p2_00016384,
        fcma_planned32_p2_00065536,
        fcma_planned32_p2_01048576,
        fcma_planned64_p2_00000064,
        fcma_planned64_p2_00000128,
        fcma_planned64_p2_00000256,
        fcma_planned64_p2_00000512,
        fcma_planned64_p2_00001024,
        fcma_planned64_p2_00002048,
        fcma_planned64_p2_00004096,
        fcma_planned64_p2_00016384,
        fcma_planned64_p2_00065536,
        fcma_planned64_p2_01048576,
        fcma_planned32_p7_00343,
        fcma_planned32_p7_02401,
        fcma_planned32_p7_16807,
        fcma_planned64_p7_00343,
        fcma_planned64_p7_02401,
        fcma_planned64_p7_16807,
        fcma_planned32_prime_0149,
        fcma_planned32_prime_0151,
        fcma_planned32_prime_0251,
        fcma_planned32_prime_0257,
        fcma_planned32_prime_2017,
        fcma_planned32_prime_2879,
        fcma_planned32_prime_65521,
        fcma_planned32_prime_746497,
        fcma_planned64_prime_0149,
        fcma_planned64_prime_0151,
        fcma_planned64_prime_0251,
        fcma_planned64_prime_0257,
        fcma_planned64_prime_2017,
        fcma_planned64_prime_2879,
        fcma_planned64_prime_65521,
        fcma_planned64_prime_746497,
        fcma_planned32_composite_000018,
        fcma_planned32_composite_000360,
        fcma_planned32_composite_001200,
        fcma_planned32_composite_044100,
        fcma_planned32_composite_048000,
        fcma_planned32_composite_046656,
        fcma_planned64_composite_000018,
        fcma_planned64_composite_000360,
        fcma_planned64_composite_001200,
        fcma_planned64_composite_044100,
        fcma_planned64_composite_048000,
        fcma_planned64_composite_046656,
    );
}

criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark
}
criterion_main!(benches);
