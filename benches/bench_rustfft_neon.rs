use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::sync::Arc;
mod config;

use criterion::{criterion_group, criterion_main, Bencher, Criterion};


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_f32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerNeon::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);
    assert_eq!(fft.len(), len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerNeon::new().unwrap();
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
    let mut planner = rustfft::FftPlannerNeon::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_multi_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerNeon::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// All butterflies
fn neon_butterfly32_02(b: &mut Bencher) { bench_planned_multi_f32(b, 2);}
fn neon_butterfly32_03(b: &mut Bencher) { bench_planned_multi_f32(b, 3);}
fn neon_butterfly32_04(b: &mut Bencher) { bench_planned_multi_f32(b, 4);}
fn neon_butterfly32_05(b: &mut Bencher) { bench_planned_multi_f32(b, 5);}
fn neon_butterfly32_06(b: &mut Bencher) { bench_planned_multi_f32(b, 6);}
fn neon_butterfly32_07(b: &mut Bencher) { bench_planned_multi_f32(b, 7);}
fn neon_butterfly32_08(b: &mut Bencher) { bench_planned_multi_f32(b, 8);}
fn neon_butterfly32_09(b: &mut Bencher) { bench_planned_multi_f32(b, 9);}
fn neon_butterfly32_10(b: &mut Bencher) { bench_planned_multi_f32(b, 10);}
fn neon_butterfly32_11(b: &mut Bencher) { bench_planned_multi_f32(b, 11);}
fn neon_butterfly32_12(b: &mut Bencher) { bench_planned_multi_f32(b, 12);}
fn neon_butterfly32_13(b: &mut Bencher) { bench_planned_multi_f32(b, 13);}
fn neon_butterfly32_15(b: &mut Bencher) { bench_planned_multi_f32(b, 15);}
fn neon_butterfly32_16(b: &mut Bencher) { bench_planned_multi_f32(b, 16);}
fn neon_butterfly32_17(b: &mut Bencher) { bench_planned_multi_f32(b, 17);}
fn neon_butterfly32_19(b: &mut Bencher) { bench_planned_multi_f32(b, 19);}
fn neon_butterfly32_23(b: &mut Bencher) { bench_planned_multi_f32(b, 23);}
fn neon_butterfly32_29(b: &mut Bencher) { bench_planned_multi_f32(b, 29);}
fn neon_butterfly32_31(b: &mut Bencher) { bench_planned_multi_f32(b, 31);}
fn neon_butterfly32_32(b: &mut Bencher) { bench_planned_multi_f32(b, 32);}


fn neon_prime_butterfly32_07(b: &mut Bencher) { bench_planned_multi_f32(b, 7);}
fn neon_prime_butterfly32_11(b: &mut Bencher) { bench_planned_multi_f32(b, 11);}
fn neon_prime_butterfly32_13(b: &mut Bencher) { bench_planned_multi_f32(b, 13);}
fn neon_prime_butterfly32_17(b: &mut Bencher) { bench_planned_multi_f32(b, 17);}
fn neon_prime_butterfly32_19(b: &mut Bencher) { bench_planned_multi_f32(b, 19);}
fn neon_prime_butterfly32_23(b: &mut Bencher) { bench_planned_multi_f32(b, 23);}
fn neon_prime_butterfly32_29(b: &mut Bencher) { bench_planned_multi_f32(b, 29);}
fn neon_prime_butterfly32_31(b: &mut Bencher) { bench_planned_multi_f32(b, 31);}

fn neon_prime_butterfly64_07(b: &mut Bencher) { bench_planned_multi_f64(b, 7);}
fn neon_prime_butterfly64_11(b: &mut Bencher) { bench_planned_multi_f64(b, 11);}
fn neon_prime_butterfly64_13(b: &mut Bencher) { bench_planned_multi_f64(b, 13);}
fn neon_prime_butterfly64_17(b: &mut Bencher) { bench_planned_multi_f64(b, 17);}
fn neon_prime_butterfly64_19(b: &mut Bencher) { bench_planned_multi_f64(b, 19);}
fn neon_prime_butterfly64_23(b: &mut Bencher) { bench_planned_multi_f64(b, 23);}
fn neon_prime_butterfly64_29(b: &mut Bencher) { bench_planned_multi_f64(b, 29);}
fn neon_prime_butterfly64_31(b: &mut Bencher) { bench_planned_multi_f64(b, 31);}

fn neon_butterfly64_02(b: &mut Bencher) { bench_planned_multi_f64(b, 2);}
fn neon_butterfly64_03(b: &mut Bencher) { bench_planned_multi_f64(b, 3);}
fn neon_butterfly64_04(b: &mut Bencher) { bench_planned_multi_f64(b, 4);}
fn neon_butterfly64_05(b: &mut Bencher) { bench_planned_multi_f64(b, 5);}
fn neon_butterfly64_06(b: &mut Bencher) { bench_planned_multi_f64(b, 6);}
fn neon_butterfly64_07(b: &mut Bencher) { bench_planned_multi_f64(b, 7);}
fn neon_butterfly64_08(b: &mut Bencher) { bench_planned_multi_f64(b, 8);}
fn neon_butterfly64_09(b: &mut Bencher) { bench_planned_multi_f64(b, 9);}
fn neon_butterfly64_10(b: &mut Bencher) { bench_planned_multi_f64(b, 10);}
fn neon_butterfly64_11(b: &mut Bencher) { bench_planned_multi_f64(b, 11);}
fn neon_butterfly64_12(b: &mut Bencher) { bench_planned_multi_f64(b, 12);}
fn neon_butterfly64_13(b: &mut Bencher) { bench_planned_multi_f64(b, 13);}
fn neon_butterfly64_15(b: &mut Bencher) { bench_planned_multi_f64(b, 15);}
fn neon_butterfly64_16(b: &mut Bencher) { bench_planned_multi_f64(b, 16);}
fn neon_butterfly64_17(b: &mut Bencher) { bench_planned_multi_f64(b, 17);}
fn neon_butterfly64_19(b: &mut Bencher) { bench_planned_multi_f64(b, 19);}
fn neon_butterfly64_23(b: &mut Bencher) { bench_planned_multi_f64(b, 23);}
fn neon_butterfly64_29(b: &mut Bencher) { bench_planned_multi_f64(b, 29);}
fn neon_butterfly64_31(b: &mut Bencher) { bench_planned_multi_f64(b, 31);}
fn neon_butterfly64_32(b: &mut Bencher) { bench_planned_multi_f64(b, 32);}

// Powers of 2
fn neon_planned32_p2_00000064(b: &mut Bencher) { bench_planned_f32(b, 64); }
fn neon_planned32_p2_00000128(b: &mut Bencher) { bench_planned_f32(b, 128); }
fn neon_planned32_p2_00000256(b: &mut Bencher) { bench_planned_f32(b, 256); }
fn neon_planned32_p2_00000512(b: &mut Bencher) { bench_planned_f32(b, 512); }
fn neon_planned32_p2_00001024(b: &mut Bencher) { bench_planned_f32(b, 1024); }
fn neon_planned32_p2_00002048(b: &mut Bencher) { bench_planned_f32(b, 2048); }
fn neon_planned32_p2_00004096(b: &mut Bencher) { bench_planned_f32(b, 4096); }
fn neon_planned32_p2_00016384(b: &mut Bencher) { bench_planned_f32(b, 16384); }
fn neon_planned32_p2_00065536(b: &mut Bencher) { bench_planned_f32(b, 65536); }
fn neon_planned32_p2_01048576(b: &mut Bencher) { bench_planned_f32(b, 1048576); }

fn neon_planned64_p2_00000064(b: &mut Bencher) { bench_planned_f64(b, 64); }
fn neon_planned64_p2_00000128(b: &mut Bencher) { bench_planned_f64(b, 128); }
fn neon_planned64_p2_00000256(b: &mut Bencher) { bench_planned_f64(b, 256); }
fn neon_planned64_p2_00000512(b: &mut Bencher) { bench_planned_f64(b, 512); }
fn neon_planned64_p2_00001024(b: &mut Bencher) { bench_planned_f64(b, 1024); }
fn neon_planned64_p2_00002048(b: &mut Bencher) { bench_planned_f64(b, 2048); }
fn neon_planned64_p2_00004096(b: &mut Bencher) { bench_planned_f64(b, 4096); }
fn neon_planned64_p2_00016384(b: &mut Bencher) { bench_planned_f64(b, 16384); }
fn neon_planned64_p2_00065536(b: &mut Bencher) { bench_planned_f64(b, 65536); }
fn neon_planned64_p2_01048576(b: &mut Bencher) { bench_planned_f64(b, 1048576); }


// Powers of 7
fn neon_planned32_p7_00343(b: &mut Bencher) { bench_planned_f32(b,   343); }
fn neon_planned32_p7_02401(b: &mut Bencher) { bench_planned_f32(b,  2401); }
fn neon_planned32_p7_16807(b: &mut Bencher) { bench_planned_f32(b, 16807); }

fn neon_planned64_p7_00343(b: &mut Bencher) { bench_planned_f64(b,   343); }
fn neon_planned64_p7_02401(b: &mut Bencher) { bench_planned_f64(b,  2401); }
fn neon_planned64_p7_16807(b: &mut Bencher) { bench_planned_f64(b, 16807); }

// Prime lengths
fn neon_planned32_prime_0149(b: &mut Bencher)     { bench_planned_f32(b,  149); }
fn neon_planned32_prime_0151(b: &mut Bencher)     { bench_planned_f32(b,  151); }
fn neon_planned32_prime_0251(b: &mut Bencher)     { bench_planned_f32(b,  251); }
fn neon_planned32_prime_0257(b: &mut Bencher)     { bench_planned_f32(b,  257); }
fn neon_planned32_prime_2017(b: &mut Bencher)     { bench_planned_f32(b,  2017); }
fn neon_planned32_prime_2879(b: &mut Bencher)     { bench_planned_f32(b,  2879); }
fn neon_planned32_prime_65521(b: &mut Bencher)    { bench_planned_f32(b, 65521); }
fn neon_planned32_prime_746497(b: &mut Bencher)   { bench_planned_f32(b,746497); }

fn neon_planned64_prime_0149(b: &mut Bencher)     { bench_planned_f64(b,  149); }
fn neon_planned64_prime_0151(b: &mut Bencher)     { bench_planned_f64(b,  151); }
fn neon_planned64_prime_0251(b: &mut Bencher)     { bench_planned_f64(b,  251); }
fn neon_planned64_prime_0257(b: &mut Bencher)     { bench_planned_f64(b,  257); }
fn neon_planned64_prime_2017(b: &mut Bencher)     { bench_planned_f64(b,  2017); }
fn neon_planned64_prime_2879(b: &mut Bencher)     { bench_planned_f64(b,  2879); }
fn neon_planned64_prime_65521(b: &mut Bencher)    { bench_planned_f64(b, 65521); }
fn neon_planned64_prime_746497(b: &mut Bencher)   { bench_planned_f64(b,746497); }

// small mixed composites
fn neon_planned32_composite_000018(b: &mut Bencher) { bench_planned_f32(b,  00018); }
fn neon_planned32_composite_000360(b: &mut Bencher) { bench_planned_f32(b,  00360); }
fn neon_planned32_composite_001200(b: &mut Bencher) { bench_planned_f32(b,  01200); }
fn neon_planned32_composite_044100(b: &mut Bencher) { bench_planned_f32(b,  44100); }
fn neon_planned32_composite_048000(b: &mut Bencher) { bench_planned_f32(b,  48000); }
fn neon_planned32_composite_046656(b: &mut Bencher) { bench_planned_f32(b,  46656); }

fn neon_planned64_composite_000018(b: &mut Bencher) { bench_planned_f64(b,  00018); }
fn neon_planned64_composite_000360(b: &mut Bencher) { bench_planned_f64(b,  00360); }
fn neon_planned64_composite_001200(b: &mut Bencher) { bench_planned_f64(b,  01200); }
fn neon_planned64_composite_044100(b: &mut Bencher) { bench_planned_f64(b,  44100); }
fn neon_planned64_composite_048000(b: &mut Bencher) { bench_planned_f64(b,  48000); }
fn neon_planned64_composite_046656(b: &mut Bencher) { bench_planned_f64(b,  46656); }




fn criterion_benchmark(c: &mut Criterion) {
    config::register_benchmarks!(
        c,
        neon_butterfly32_02,
        neon_butterfly32_03,
        neon_butterfly32_04,
        neon_butterfly32_05,
        neon_butterfly32_06,
        neon_butterfly32_07,
        neon_butterfly32_08,
        neon_butterfly32_09,
        neon_butterfly32_10,
        neon_butterfly32_11,
        neon_butterfly32_12,
        neon_butterfly32_13,
        neon_butterfly32_15,
        neon_butterfly32_16,
        neon_butterfly32_17,
        neon_butterfly32_19,
        neon_butterfly32_23,
        neon_butterfly32_29,
        neon_butterfly32_31,
        neon_butterfly32_32,
        neon_prime_butterfly32_07,
        neon_prime_butterfly32_11,
        neon_prime_butterfly32_13,
        neon_prime_butterfly32_17,
        neon_prime_butterfly32_19,
        neon_prime_butterfly32_23,
        neon_prime_butterfly32_29,
        neon_prime_butterfly32_31,
        neon_prime_butterfly64_07,
        neon_prime_butterfly64_11,
        neon_prime_butterfly64_13,
        neon_prime_butterfly64_17,
        neon_prime_butterfly64_19,
        neon_prime_butterfly64_23,
        neon_prime_butterfly64_29,
        neon_prime_butterfly64_31,
        neon_butterfly64_02,
        neon_butterfly64_03,
        neon_butterfly64_04,
        neon_butterfly64_05,
        neon_butterfly64_06,
        neon_butterfly64_07,
        neon_butterfly64_08,
        neon_butterfly64_09,
        neon_butterfly64_10,
        neon_butterfly64_11,
        neon_butterfly64_12,
        neon_butterfly64_13,
        neon_butterfly64_15,
        neon_butterfly64_16,
        neon_butterfly64_17,
        neon_butterfly64_19,
        neon_butterfly64_23,
        neon_butterfly64_29,
        neon_butterfly64_31,
        neon_butterfly64_32,
        neon_planned32_p2_00000064,
        neon_planned32_p2_00000128,
        neon_planned32_p2_00000256,
        neon_planned32_p2_00000512,
        neon_planned32_p2_00001024,
        neon_planned32_p2_00002048,
        neon_planned32_p2_00004096,
        neon_planned32_p2_00016384,
        neon_planned32_p2_00065536,
        neon_planned32_p2_01048576,
        neon_planned64_p2_00000064,
        neon_planned64_p2_00000128,
        neon_planned64_p2_00000256,
        neon_planned64_p2_00000512,
        neon_planned64_p2_00001024,
        neon_planned64_p2_00002048,
        neon_planned64_p2_00004096,
        neon_planned64_p2_00016384,
        neon_planned64_p2_00065536,
        neon_planned64_p2_01048576,
        neon_planned32_p7_00343,
        neon_planned32_p7_02401,
        neon_planned32_p7_16807,
        neon_planned64_p7_00343,
        neon_planned64_p7_02401,
        neon_planned64_p7_16807,
        neon_planned32_prime_0149,
        neon_planned32_prime_0151,
        neon_planned32_prime_0251,
        neon_planned32_prime_0257,
        neon_planned32_prime_2017,
        neon_planned32_prime_2879,
        neon_planned32_prime_65521,
        neon_planned32_prime_746497,
        neon_planned64_prime_0149,
        neon_planned64_prime_0151,
        neon_planned64_prime_0251,
        neon_planned64_prime_0257,
        neon_planned64_prime_2017,
        neon_planned64_prime_2879,
        neon_planned64_prime_65521,
        neon_planned64_prime_746497,
        neon_planned32_composite_000018,
        neon_planned32_composite_000360,
        neon_planned32_composite_001200,
        neon_planned32_composite_044100,
        neon_planned32_composite_048000,
        neon_planned32_composite_046656,
        neon_planned64_composite_000018,
        neon_planned64_composite_000360,
        neon_planned64_composite_001200,
        neon_planned64_composite_044100,
        neon_planned64_composite_048000,
        neon_planned64_composite_046656,
    );
}

criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark
}
criterion_main!(benches);
