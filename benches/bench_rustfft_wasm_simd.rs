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
use rustfft::{num_complex::Complex, simd::simd_array::{SimdComplexArrayMut, SimdComplexArray}};
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::sync::Arc;
mod config;
use std::hint::black_box;
use rustfft::simd::simd_vector::SimdVector;

use criterion::{criterion_group, criterion_main, Bencher, Criterion, BenchmarkId};

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


use core::arch::wasm32::*;
use rustfft::wasm_simd::wasm_simd_vector::{WasmVector32, WasmVector64};


#[target_feature(enable = "simd128")]
unsafe fn mul_complex_f32_deinterleave(
    src: &[num_complex::Complex<f32>],
    mut dest: &mut [num_complex::Complex<f32>],
    multiplier: &[num_complex::Complex<f32>],
) {
    let count = dest.len();
    assert!(src.len() >= count && multiplier.len() >= count);

    let unroll_size = 4;
    let (unroll_count, unroll_remainder) = (count / unroll_size, count % unroll_size);

    for i in 0..unroll_count {
        let vcol = i * unroll_size;

        let src0: WasmVector32 = src.load(vcol);
        let src1: WasmVector32 = src.load(vcol + 2);

        let multiplier_deinterleaved_re: WasmVector32 = multiplier.load(vcol);
        let multiplier_deinterleaved_im: WasmVector32 = multiplier.load(vcol + 2);

        let src_deinterleaved_re = u32x4_shuffle::<0, 2, 4, 6>(src0.0, src1.0);
        let src_deinterleaved_im = u32x4_shuffle::<1, 3, 4, 7>(src0.0, src1.0);

        let dest_deinterleaved_re = f32x4_sub(f32x4_mul(src_deinterleaved_re, multiplier_deinterleaved_re.0), f32x4_mul(src_deinterleaved_im, multiplier_deinterleaved_im.0));
        let dest_deinterleaved_im = f32x4_add(f32x4_mul(src_deinterleaved_re, multiplier_deinterleaved_im.0), f32x4_mul(src_deinterleaved_im, multiplier_deinterleaved_re.0));

        let dst0 = WasmVector32(u32x4_shuffle::<0, 4, 1, 5>(dest_deinterleaved_re, dest_deinterleaved_im));
        let dst1 = WasmVector32(u32x4_shuffle::<2, 6, 3, 7>(dest_deinterleaved_re, dest_deinterleaved_im));

        dest.store(dst0, vcol);
        dest.store(dst1, vcol + 2);
    }

    if unroll_remainder != 0 {
        let vcol = unroll_count * unroll_size;

        for i in vcol..count {
            let a = *src.get_unchecked(i);
            let b = *multiplier.get_unchecked(i);
            *dest.get_unchecked_mut(i) = a * b;
        }
    }
}


/// One complex multiply, conjugating `left` first when `CONJUGATE` is set.
///
/// Written out rather than passed in as a closure, because a closure would not inherit the target
/// feature of the function this ends up inlined into. See `SimdVector::cross_layer_radixn`.
#[inline(always)]
unsafe fn mul_one<V: SimdVector, const CONJUGATE: bool>(left: V, right: V) -> V {
    if CONJUGATE {
        V::mul_complex_conjugated(left, right)
    } else {
        V::mul_complex(left, right)
    }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_complex_f32_interleave(
    src: &[num_complex::Complex<f32>],
    mut dest: &mut [num_complex::Complex<f32>],
    multiplier: &[num_complex::Complex<f32>],
) {
    let count = dest.len();
    assert!(src.len() >= count && multiplier.len() >= count);

    let per_vector = 2;
    let full_vectors = count / per_vector;

    for i in 0..full_vectors / 2 {
        let index = i * 2 * per_vector;
        let next = index + per_vector;

        let left_a = src.load(index);
        let left_b = src.load(next);

        let product_a = mul_one::<WasmVector32, false>(left_a, multiplier.load(index));
        let product_b = mul_one::<WasmVector32, false>(left_b, multiplier.load(next));

        dest.store(product_a, index);
        dest.store(product_b, next);
    }

    // an odd vector count leaves one behind
    if full_vectors % 2 == 1 {
        let index = (full_vectors - 1) * per_vector;
        let left = src.load(index);
        let product = mul_one::<WasmVector32, false>(left, multiplier.load(index));
        dest.store(product, index);
    }

    // The leftover element, if there is one. `COMPLEX_PER_VECTOR` is a constant, so this whole
    // branch vanishes on the single-complex vectors, whose partial load and store are
    // unimplemented.
    if per_vector > 1 && count % per_vector != 0 {
        let index = full_vectors * per_vector;
        let left = src.load1_lo(index);
        let product = mul_one::<WasmVector32, false>(left, multiplier.load1_lo(index));
        dest.store1_lo(product, index);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_complex_f32_scalar(
    src: &[num_complex::Complex<f32>],
    dest: &mut [num_complex::Complex<f32>],
    multiplier: &[num_complex::Complex<f32>],
) {
    let count = dest.len();
    assert!(src.len() >= count && multiplier.len() >= count);

    for i in 0..count {
        let a = *src.get_unchecked(i);
        let b = *multiplier.get_unchecked(i);
        *dest.get_unchecked_mut(i) = a * b;
    }
}

fn bench_mul_complex_f32_scalar(b: &mut Bencher, len: usize) {
    let src = vec![Complex::zero(); len];
    let multiplier = vec![Complex::zero(); len];
    let mut dest = vec![Complex::zero(); len];

    b.iter(|| {
        unsafe { mul_complex_f32_scalar(&src, &mut dest, &multiplier) };
        black_box(&dest);
    });
}
fn bench_mul_complex_f32_interleave(b: &mut Bencher, len: usize) {
    let src = vec![Complex::zero(); len];
    let multiplier = vec![Complex::zero(); len];
    let mut dest = vec![Complex::zero(); len];

    b.iter(|| {
        unsafe { mul_complex_f32_interleave(&src, &mut dest, &multiplier) };
        black_box(&dest);
    });
}
fn bench_mul_complex_f32_deinterleave(b: &mut Bencher, len: usize) {
    let src = vec![Complex::zero(); len];
    let multiplier = vec![Complex::zero(); len];
    let mut dest = vec![Complex::zero(); len];

    b.iter(|| {
        unsafe { mul_complex_f32_deinterleave(&src, &mut dest, &multiplier) };
        black_box(&dest);
    });
}

#[target_feature(enable = "simd128")]
unsafe fn mul_complex_f64_deinterleave(
    src: &[num_complex::Complex<f64>],
    mut dest: &mut [num_complex::Complex<f64>],
    multiplier: &[num_complex::Complex<f64>],
) {
    let count = dest.len();
    assert!(src.len() >= count && multiplier.len() >= count);

    let unroll_size = 2;
    let (unroll_count, unroll_remainder) = (count / unroll_size, count % unroll_size);

    for i in 0..unroll_count {
        let vcol = i * unroll_size;

        let src0: WasmVector64 = src.load(vcol);
        let src1: WasmVector64 = src.load(vcol + 2);

        let multiplier_deinterleaved_re: WasmVector64 = multiplier.load(vcol);
        let multiplier_deinterleaved_im: WasmVector64 = multiplier.load(vcol + 2);

        let src_deinterleaved_re = u64x2_shuffle::<0, 2>(src0.0, src1.0);
        let src_deinterleaved_im = u64x2_shuffle::<1, 3>(src0.0, src1.0);

        let dest_deinterleaved_re = f64x2_sub(f64x2_mul(src_deinterleaved_re, multiplier_deinterleaved_re.0), f64x2_mul(src_deinterleaved_im, multiplier_deinterleaved_im.0));
        let dest_deinterleaved_im = f64x2_add(f64x2_mul(src_deinterleaved_re, multiplier_deinterleaved_im.0), f64x2_mul(src_deinterleaved_im, multiplier_deinterleaved_re.0));

        let dst0 = WasmVector64(u64x2_shuffle::<0, 2>(dest_deinterleaved_re, dest_deinterleaved_im));
        let dst1 = WasmVector64(u64x2_shuffle::<1, 3>(dest_deinterleaved_re, dest_deinterleaved_im));

        dest.store(dst0, vcol);
        dest.store(dst1, vcol + 2);

    }

    if unroll_remainder != 0 {
        let vcol = unroll_count * unroll_size;

        for i in vcol..count {
            let a = *src.get_unchecked(i);
            let b = *multiplier.get_unchecked(i);
            *dest.get_unchecked_mut(i) = a * b;
        }
    }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_complex_f64_interleave(
    src: &[num_complex::Complex<f64>],
    mut dest: &mut [num_complex::Complex<f64>],
    multiplier: &[num_complex::Complex<f64>],
) {
    let count = dest.len();
    assert!(src.len() >= count && multiplier.len() >= count);
    
    let per_vector = 1;
    let full_vectors = count / per_vector;

    for i in 0..full_vectors / 2 {
        let index = i * 2 * per_vector;
        let next = index + per_vector;

        let left_a = src.load(index);
        let left_b = src.load(next);

        let product_a = mul_one::<WasmVector64, false>(left_a, multiplier.load(index));
        let product_b = mul_one::<WasmVector64, false>(left_b, multiplier.load(next));

        dest.store(product_a, index);
        dest.store(product_b, next);
    }

    // an odd vector count leaves one behind
    if full_vectors % 2 == 1 {
        let index = (full_vectors - 1) * per_vector;
        let left = src.load(index);
        let product = mul_one::<WasmVector64, false>(left, multiplier.load(index));
        dest.store(product, index);
    }

    // The leftover element, if there is one. `COMPLEX_PER_VECTOR` is a constant, so this whole
    // branch vanishes on the single-complex vectors, whose partial load and store are
    // unimplemented.
    if per_vector > 1 && count % per_vector != 0 {
        let index = full_vectors * per_vector;
        let left = src.load1_lo(index);
        let product = mul_one::<WasmVector64, false>(left, multiplier.load1_lo(index));
        dest.store1_lo(product, index);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_complex_f64_scalar(
    src: &[num_complex::Complex<f64>],
    dest: &mut [num_complex::Complex<f64>],
    multiplier: &[num_complex::Complex<f64>],
) {
    let count = dest.len();
    assert!(src.len() >= count && multiplier.len() >= count);

    for i in 0..count {
        let a = *src.get_unchecked(i);
        let b = *multiplier.get_unchecked(i);
        *dest.get_unchecked_mut(i) = a * b;
    }
}

fn bench_mul_complex_f64_scalar(b: &mut Bencher, len: usize) {
    let src = vec![Complex::zero(); len];
    let multiplier = vec![Complex::zero(); len];
    let mut dest = vec![Complex::zero(); len];

    b.iter(|| {
        unsafe { mul_complex_f64_scalar(&src, &mut dest, &multiplier) };
        black_box(&dest);
    });
}
fn bench_mul_complex_f64_interleave(b: &mut Bencher, len: usize) {
    let src = vec![Complex::zero(); len];
    let multiplier = vec![Complex::zero(); len];
    let mut dest = vec![Complex::zero(); len];

    b.iter(|| {
        unsafe { mul_complex_f64_interleave(&src, &mut dest, &multiplier) };
        black_box(&dest);
    });
}
fn bench_mul_complex_f64_deinterleave(b: &mut Bencher, len: usize) {
    let src = vec![Complex::zero(); len];
    let multiplier = vec![Complex::zero(); len];
    let mut dest = vec![Complex::zero(); len];

    b.iter(|| {
        unsafe { mul_complex_f64_deinterleave(&src, &mut dest, &multiplier) };
        black_box(&dest);
    });
}


fn compare_mul_complex(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("mul_complex_f32");
        for len in [101, 1001, 10001, 100001] {
            group.bench_with_input(BenchmarkId::new("scalar", len), &len, |b, _| {
                bench_mul_complex_f32_scalar(b, len);
            });
            group.bench_with_input(BenchmarkId::new("interleave", len), &len, |b, _| {
                bench_mul_complex_f32_interleave(b, len);
            });
            group.bench_with_input(BenchmarkId::new("deinterleave", len), &len, |b, _| {
                bench_mul_complex_f32_deinterleave(b, len);
            });
        }
    }
    {
        let mut group = c.benchmark_group("mul_complex_f64");
        for len in [101, 1001, 10001, 100001] {
            group.bench_with_input(BenchmarkId::new("scalar", len), &len, |b, _| {
                bench_mul_complex_f64_scalar(b, len);
            });
            group.bench_with_input(BenchmarkId::new("interleave", len), &len, |b, _| {
                bench_mul_complex_f64_interleave(b, len);
            });
            group.bench_with_input(BenchmarkId::new("deinterleave", len), &len, |b, _| {
                bench_mul_complex_f64_deinterleave(b, len);
            });
        }
    }
}


criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark, compare_mul_complex
}
criterion_main!(benches);
