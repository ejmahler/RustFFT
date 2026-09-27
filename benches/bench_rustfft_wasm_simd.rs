/// Unfortunately, `cargo bench` does not permit running these benchmarks out-of-the-box
/// on a WebAssembly virtual machine.
///
/// Follow these steps to run these benchmarks:
/// 0. Prerequisites: Install the `wasm32-wasi` target and `wasmer`
///
///
/// 1. Build these benchmarks
/// ```bash
/// cargo build --bench=bench_rustfft_wasm_simd --release  --target wasm32-wasi --features "wasm_simd"
/// ```
///
/// After cargo built the bench binary, cargo stores it inside the
/// `<PROJECT_ROOT>/target/wasm32-wasi/release/deps` directory.
/// The file name of this binary follows this format: `bench_rustfft_wasm_simd-<CHECKSUM>.wasm`.
/// For instance, it could be named
/// `target/wasm32-wasi/release/deps/bench_rustfft_scalar-6d2b3d5a567416f5.wasm`
///
/// 2. Copy the most recently built WASM binary to hex.wasm
/// ```bash
/// cp `ls -t target/wasm32-wasi/release/deps/*.wasm | head -n 1` hex.wasm
/// ```
///
/// 3. Run these benchmark e. g. with [wasmer](https://github.com/wasmerio/wasmer)
/// ```bash
/// wasmer run --dir=. hex.wasm -- --bench
/// ```
///
/// For more information, refer to [Criterion's user guide](https://github.com/bheisler/criterion.rs/blob/dc2b06cd31f7aa34cff6a83a00598e0523186dad/book/src/user_guide/wasi.md)
/// which should be mostly applicable to our use case.
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::sync::Arc;
mod config;

use criterion::{criterion_group, criterion_main, Bencher, Criterion};

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_f32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerWasmSimd::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);
    assert_eq!(fft.len(), len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerWasmSimd::new().unwrap();
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
    let mut planner = rustfft::FftPlannerWasmSimd::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_multi_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerWasmSimd::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// All butterflies
fn wasm_simd_butterfly32_02(b: &mut Bencher) {
    bench_planned_multi_f32(b, 2);
}
fn wasm_simd_butterfly32_03(b: &mut Bencher) {
    bench_planned_multi_f32(b, 3);
}
fn wasm_simd_butterfly32_04(b: &mut Bencher) {
    bench_planned_multi_f32(b, 4);
}
fn wasm_simd_butterfly32_05(b: &mut Bencher) {
    bench_planned_multi_f32(b, 5);
}
fn wasm_simd_butterfly32_06(b: &mut Bencher) {
    bench_planned_multi_f32(b, 6);
}
fn wasm_simd_butterfly32_07(b: &mut Bencher) {
    bench_planned_multi_f32(b, 7);
}
fn wasm_simd_butterfly32_08(b: &mut Bencher) {
    bench_planned_multi_f32(b, 8);
}
fn wasm_simd_butterfly32_09(b: &mut Bencher) {
    bench_planned_multi_f32(b, 9);
}
fn wasm_simd_butterfly32_10(b: &mut Bencher) {
    bench_planned_multi_f32(b, 10);
}
fn wasm_simd_butterfly32_11(b: &mut Bencher) {
    bench_planned_multi_f32(b, 11);
}
fn wasm_simd_butterfly32_12(b: &mut Bencher) {
    bench_planned_multi_f32(b, 12);
}
fn wasm_simd_butterfly32_13(b: &mut Bencher) {
    bench_planned_multi_f32(b, 13);
}
fn wasm_simd_butterfly32_15(b: &mut Bencher) {
    bench_planned_multi_f32(b, 15);
}
fn wasm_simd_butterfly32_16(b: &mut Bencher) {
    bench_planned_multi_f32(b, 16);
}
fn wasm_simd_butterfly32_17(b: &mut Bencher) {
    bench_planned_multi_f32(b, 17);
}
fn wasm_simd_butterfly32_19(b: &mut Bencher) {
    bench_planned_multi_f32(b, 19);
}
fn wasm_simd_butterfly32_23(b: &mut Bencher) {
    bench_planned_multi_f32(b, 23);
}
fn wasm_simd_butterfly32_24(b: &mut Bencher) {
    bench_planned_multi_f32(b, 24);
}
fn wasm_simd_butterfly32_29(b: &mut Bencher) {
    bench_planned_multi_f32(b, 29);
}
fn wasm_simd_butterfly32_31(b: &mut Bencher) {
    bench_planned_multi_f32(b, 31);
}
fn wasm_simd_butterfly32_32(b: &mut Bencher) {
    bench_planned_multi_f32(b, 32);
}

fn wasm_simd_butterfly64_02(b: &mut Bencher) {
    bench_planned_multi_f64(b, 2);
}
fn wasm_simd_butterfly64_03(b: &mut Bencher) {
    bench_planned_multi_f64(b, 3);
}
fn wasm_simd_butterfly64_04(b: &mut Bencher) {
    bench_planned_multi_f64(b, 4);
}
fn wasm_simd_butterfly64_05(b: &mut Bencher) {
    bench_planned_multi_f64(b, 5);
}
fn wasm_simd_butterfly64_06(b: &mut Bencher) {
    bench_planned_multi_f64(b, 6);
}
fn wasm_simd_butterfly64_07(b: &mut Bencher) {
    bench_planned_multi_f64(b, 7);
}
fn wasm_simd_butterfly64_08(b: &mut Bencher) {
    bench_planned_multi_f64(b, 8);
}
fn wasm_simd_butterfly64_09(b: &mut Bencher) {
    bench_planned_multi_f64(b, 9);
}
fn wasm_simd_butterfly64_10(b: &mut Bencher) {
    bench_planned_multi_f64(b, 10);
}
fn wasm_simd_butterfly64_11(b: &mut Bencher) {
    bench_planned_multi_f64(b, 11);
}
fn wasm_simd_butterfly64_12(b: &mut Bencher) {
    bench_planned_multi_f64(b, 12);
}
fn wasm_simd_butterfly64_13(b: &mut Bencher) {
    bench_planned_multi_f64(b, 13);
}
fn wasm_simd_butterfly64_15(b: &mut Bencher) {
    bench_planned_multi_f64(b, 15);
}
fn wasm_simd_butterfly64_16(b: &mut Bencher) {
    bench_planned_multi_f64(b, 16);
}
fn wasm_simd_butterfly64_17(b: &mut Bencher) {
    bench_planned_multi_f64(b, 17);
}
fn wasm_simd_butterfly64_19(b: &mut Bencher) {
    bench_planned_multi_f64(b, 19);
}
fn wasm_simd_butterfly64_23(b: &mut Bencher) {
    bench_planned_multi_f64(b, 23);
}
fn wasm_simd_butterfly64_24(b: &mut Bencher) {
    bench_planned_multi_f64(b, 24);
}
fn wasm_simd_butterfly64_29(b: &mut Bencher) {
    bench_planned_multi_f64(b, 29);
}
fn wasm_simd_butterfly64_31(b: &mut Bencher) {
    bench_planned_multi_f64(b, 31);
}
fn wasm_simd_butterfly64_32(b: &mut Bencher) {
    bench_planned_multi_f64(b, 32);
}

// prime butterflies
fn wasm_simd_prime_butterfly32_07(b: &mut Bencher) { bench_planned_multi_f32(b, 7);}
fn wasm_simd_prime_butterfly32_11(b: &mut Bencher) { bench_planned_multi_f32(b, 11);}
fn wasm_simd_prime_butterfly32_13(b: &mut Bencher) { bench_planned_multi_f32(b, 13);}
fn wasm_simd_prime_butterfly32_17(b: &mut Bencher) { bench_planned_multi_f32(b, 17);}
fn wasm_simd_prime_butterfly32_19(b: &mut Bencher) { bench_planned_multi_f32(b, 19);}
fn wasm_simd_prime_butterfly32_23(b: &mut Bencher) { bench_planned_multi_f32(b, 23);}
fn wasm_simd_prime_butterfly32_29(b: &mut Bencher) { bench_planned_multi_f32(b, 29);}
fn wasm_simd_prime_butterfly32_31(b: &mut Bencher) { bench_planned_multi_f32(b, 31);}
fn wasm_simd_prime_butterfly64_07(b: &mut Bencher) { bench_planned_multi_f64(b, 7);}
fn wasm_simd_prime_butterfly64_11(b: &mut Bencher) { bench_planned_multi_f64(b, 11);}
fn wasm_simd_prime_butterfly64_13(b: &mut Bencher) { bench_planned_multi_f64(b, 13);}
fn wasm_simd_prime_butterfly64_17(b: &mut Bencher) { bench_planned_multi_f64(b, 17);}
fn wasm_simd_prime_butterfly64_19(b: &mut Bencher) { bench_planned_multi_f64(b, 19);}
fn wasm_simd_prime_butterfly64_23(b: &mut Bencher) { bench_planned_multi_f64(b, 23);}
fn wasm_simd_prime_butterfly64_29(b: &mut Bencher) { bench_planned_multi_f64(b, 29);}
fn wasm_simd_prime_butterfly64_31(b: &mut Bencher) { bench_planned_multi_f64(b, 31);}

// Powers of 2
fn wasm_simd_planned32_p2_00000064(b: &mut Bencher) {
    bench_planned_f32(b, 64);
}
fn wasm_simd_planned32_p2_00000128(b: &mut Bencher) {
    bench_planned_f32(b, 128);
}
fn wasm_simd_planned32_p2_00000256(b: &mut Bencher) {
    bench_planned_f32(b, 256);
}
fn wasm_simd_planned32_p2_00000512(b: &mut Bencher) {
    bench_planned_f32(b, 512);
}
fn wasm_simd_planned32_p2_00001024(b: &mut Bencher) {
    bench_planned_f32(b, 1024);
}
fn wasm_simd_planned32_p2_00002048(b: &mut Bencher) {
    bench_planned_f32(b, 2048);
}
fn wasm_simd_planned32_p2_00004096(b: &mut Bencher) {
    bench_planned_f32(b, 4096);
}
fn wasm_simd_planned32_p2_00016384(b: &mut Bencher) {
    bench_planned_f32(b, 16384);
}
fn wasm_simd_planned32_p2_00065536(b: &mut Bencher) {
    bench_planned_f32(b, 65536);
}
fn wasm_simd_planned32_p2_01048576(b: &mut Bencher) {
    bench_planned_f32(b, 1048576);
}

fn wasm_simd_planned64_p2_00000064(b: &mut Bencher) {
    bench_planned_f64(b, 64);
}
fn wasm_simd_planned64_p2_00000128(b: &mut Bencher) {
    bench_planned_f64(b, 128);
}
fn wasm_simd_planned64_p2_00000256(b: &mut Bencher) {
    bench_planned_f64(b, 256);
}
fn wasm_simd_planned64_p2_00000512(b: &mut Bencher) {
    bench_planned_f64(b, 512);
}
fn wasm_simd_planned64_p2_00001024(b: &mut Bencher) {
    bench_planned_f64(b, 1024);
}
fn wasm_simd_planned64_p2_00002048(b: &mut Bencher) {
    bench_planned_f64(b, 2048);
}
fn wasm_simd_planned64_p2_00004096(b: &mut Bencher) {
    bench_planned_f64(b, 4096);
}
fn wasm_simd_planned64_p2_00016384(b: &mut Bencher) {
    bench_planned_f64(b, 16384);
}
fn wasm_simd_planned64_p2_00065536(b: &mut Bencher) {
    bench_planned_f64(b, 65536);
}
fn wasm_simd_planned64_p2_01048576(b: &mut Bencher) {
    bench_planned_f64(b, 1048576);
}

// Powers of 7
fn wasm_simd_planned32_p7_00343(b: &mut Bencher) {
    bench_planned_f32(b, 343);
}
fn wasm_simd_planned32_p7_02401(b: &mut Bencher) {
    bench_planned_f32(b, 2401);
}
fn wasm_simd_planned32_p7_16807(b: &mut Bencher) {
    bench_planned_f32(b, 16807);
}

fn wasm_simd_planned64_p7_00343(b: &mut Bencher) {
    bench_planned_f64(b, 343);
}
fn wasm_simd_planned64_p7_02401(b: &mut Bencher) {
    bench_planned_f64(b, 2401);
}
fn wasm_simd_planned64_p7_16807(b: &mut Bencher) {
    bench_planned_f64(b, 16807);
}

// Prime lengths
fn wasm_simd_planned32_prime_0149(b: &mut Bencher) {
    bench_planned_f32(b, 149);
}
fn wasm_simd_planned32_prime_0151(b: &mut Bencher) {
    bench_planned_f32(b, 151);
}
fn wasm_simd_planned32_prime_0251(b: &mut Bencher) {
    bench_planned_f32(b, 251);
}
fn wasm_simd_planned32_prime_0257(b: &mut Bencher) {
    bench_planned_f32(b, 257);
}
fn wasm_simd_planned32_prime_2017(b: &mut Bencher) {
    bench_planned_f32(b, 2017);
}
fn wasm_simd_planned32_prime_2879(b: &mut Bencher) {
    bench_planned_f32(b, 2879);
}
fn wasm_simd_planned32_prime_65521(b: &mut Bencher) {
    bench_planned_f32(b, 65521);
}
fn wasm_simd_planned32_prime_746497(b: &mut Bencher) {
    bench_planned_f32(b, 746497);
}

fn wasm_simd_planned64_prime_0149(b: &mut Bencher) {
    bench_planned_f64(b, 149);
}
fn wasm_simd_planned64_prime_0151(b: &mut Bencher) {
    bench_planned_f64(b, 151);
}
fn wasm_simd_planned64_prime_0251(b: &mut Bencher) {
    bench_planned_f64(b, 251);
}
fn wasm_simd_planned64_prime_0257(b: &mut Bencher) {
    bench_planned_f64(b, 257);
}
fn wasm_simd_planned64_prime_2017(b: &mut Bencher) {
    bench_planned_f64(b, 2017);
}
fn wasm_simd_planned64_prime_2879(b: &mut Bencher) {
    bench_planned_f64(b, 2879);
}
fn wasm_simd_planned64_prime_65521(b: &mut Bencher) {
    bench_planned_f64(b, 65521);
}
fn wasm_simd_planned64_prime_746497(b: &mut Bencher) {
    bench_planned_f64(b, 746497);
}

// small mixed composites
fn wasm_simd_planned32_composite_000018(b: &mut Bencher) {
    bench_planned_f32(b, 00018);
}
fn wasm_simd_planned32_composite_000360(b: &mut Bencher) {
    bench_planned_f32(b, 00360);
}
fn wasm_simd_planned32_composite_001200(b: &mut Bencher) {
    bench_planned_f32(b, 01200);
}
fn wasm_simd_planned32_composite_044100(b: &mut Bencher) {
    bench_planned_f32(b, 44100);
}
fn wasm_simd_planned32_composite_048000(b: &mut Bencher) {
    bench_planned_f32(b, 48000);
}
fn wasm_simd_planned32_composite_046656(b: &mut Bencher) {
    bench_planned_f32(b, 46656);
}

fn wasm_simd_planned64_composite_000018(b: &mut Bencher) {
    bench_planned_f64(b, 00018);
}
fn wasm_simd_planned64_composite_000360(b: &mut Bencher) {
    bench_planned_f64(b, 00360);
}
fn wasm_simd_planned64_composite_001200(b: &mut Bencher) {
    bench_planned_f64(b, 01200);
}
fn wasm_simd_planned64_composite_044100(b: &mut Bencher) {
    bench_planned_f64(b, 44100);
}
fn wasm_simd_planned64_composite_048000(b: &mut Bencher) {
    bench_planned_f64(b, 48000);
}
fn wasm_simd_planned64_composite_046656(b: &mut Bencher) {
    bench_planned_f64(b, 46656);
}

fn criterion_benchmark(c: &mut Criterion) {
    config::register_benchmarks!(
        c,
        wasm_simd_butterfly32_02,
        wasm_simd_butterfly32_03,
        wasm_simd_butterfly32_04,
        wasm_simd_butterfly32_05,
        wasm_simd_butterfly32_06,
        wasm_simd_butterfly32_07,
        wasm_simd_butterfly32_08,
        wasm_simd_butterfly32_09,
        wasm_simd_butterfly32_10,
        wasm_simd_butterfly32_11,
        wasm_simd_butterfly32_12,
        wasm_simd_butterfly32_13,
        wasm_simd_butterfly32_15,
        wasm_simd_butterfly32_16,
        wasm_simd_butterfly32_17,
        wasm_simd_butterfly32_19,
        wasm_simd_butterfly32_23,
        wasm_simd_butterfly32_24,
        wasm_simd_butterfly32_29,
        wasm_simd_butterfly32_31,
        wasm_simd_butterfly32_32,
        wasm_simd_butterfly64_02,
        wasm_simd_butterfly64_03,
        wasm_simd_butterfly64_04,
        wasm_simd_butterfly64_05,
        wasm_simd_butterfly64_06,
        wasm_simd_butterfly64_07,
        wasm_simd_butterfly64_08,
        wasm_simd_butterfly64_09,
        wasm_simd_butterfly64_10,
        wasm_simd_butterfly64_11,
        wasm_simd_butterfly64_12,
        wasm_simd_butterfly64_13,
        wasm_simd_butterfly64_15,
        wasm_simd_butterfly64_16,
        wasm_simd_butterfly64_17,
        wasm_simd_butterfly64_19,
        wasm_simd_butterfly64_23,
        wasm_simd_butterfly64_24,
        wasm_simd_butterfly64_29,
        wasm_simd_butterfly64_31,
        wasm_simd_butterfly64_32,
        wasm_simd_prime_butterfly32_07,
        wasm_simd_prime_butterfly32_11,
        wasm_simd_prime_butterfly32_13,
        wasm_simd_prime_butterfly32_17,
        wasm_simd_prime_butterfly32_19,
        wasm_simd_prime_butterfly32_23,
        wasm_simd_prime_butterfly32_29,
        wasm_simd_prime_butterfly32_31,
        wasm_simd_prime_butterfly64_07,
        wasm_simd_prime_butterfly64_11,
        wasm_simd_prime_butterfly64_13,
        wasm_simd_prime_butterfly64_17,
        wasm_simd_prime_butterfly64_19,
        wasm_simd_prime_butterfly64_23,
        wasm_simd_prime_butterfly64_29,
        wasm_simd_prime_butterfly64_31,
        wasm_simd_planned32_p2_00000064,
        wasm_simd_planned32_p2_00000128,
        wasm_simd_planned32_p2_00000256,
        wasm_simd_planned32_p2_00000512,
        wasm_simd_planned32_p2_00001024,
        wasm_simd_planned32_p2_00002048,
        wasm_simd_planned32_p2_00004096,
        wasm_simd_planned32_p2_00016384,
        wasm_simd_planned32_p2_00065536,
        wasm_simd_planned32_p2_01048576,
        wasm_simd_planned64_p2_00000064,
        wasm_simd_planned64_p2_00000128,
        wasm_simd_planned64_p2_00000256,
        wasm_simd_planned64_p2_00000512,
        wasm_simd_planned64_p2_00001024,
        wasm_simd_planned64_p2_00002048,
        wasm_simd_planned64_p2_00004096,
        wasm_simd_planned64_p2_00016384,
        wasm_simd_planned64_p2_00065536,
        wasm_simd_planned64_p2_01048576,
        wasm_simd_planned32_p7_00343,
        wasm_simd_planned32_p7_02401,
        wasm_simd_planned32_p7_16807,
        wasm_simd_planned64_p7_00343,
        wasm_simd_planned64_p7_02401,
        wasm_simd_planned64_p7_16807,
        wasm_simd_planned32_prime_0149,
        wasm_simd_planned32_prime_0151,
        wasm_simd_planned32_prime_0251,
        wasm_simd_planned32_prime_0257,
        wasm_simd_planned32_prime_2017,
        wasm_simd_planned32_prime_2879,
        wasm_simd_planned32_prime_65521,
        wasm_simd_planned32_prime_746497,
        wasm_simd_planned64_prime_0149,
        wasm_simd_planned64_prime_0151,
        wasm_simd_planned64_prime_0251,
        wasm_simd_planned64_prime_0257,
        wasm_simd_planned64_prime_2017,
        wasm_simd_planned64_prime_2879,
        wasm_simd_planned64_prime_65521,
        wasm_simd_planned64_prime_746497,
        wasm_simd_planned32_composite_000018,
        wasm_simd_planned32_composite_000360,
        wasm_simd_planned32_composite_001200,
        wasm_simd_planned32_composite_044100,
        wasm_simd_planned32_composite_048000,
        wasm_simd_planned32_composite_046656,
        wasm_simd_planned64_composite_000018,
        wasm_simd_planned64_composite_000360,
        wasm_simd_planned64_composite_001200,
        wasm_simd_planned64_composite_044100,
        wasm_simd_planned64_composite_048000,
        wasm_simd_planned64_composite_046656,
    );
}

criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark
}
criterion_main!(benches);
