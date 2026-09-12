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
extern crate rustfft;

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::sync::Arc;
mod config;

use criterion::{criterion_group, criterion_main, Bencher, Criterion};

// Make fft using scalar planner
fn bench_scalar_32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f32>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make fft using scalar planner
fn bench_scalar_64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make fft using WASM SIMD planner
fn bench_wasmsimd_32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerWasmSimd::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f32>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make fft using WASM SIMD planner
fn bench_wasmsimd_64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerWasmSimd::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    const LENGTHS: &[usize] = &[
        4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576, 2097152, 4194304,
    ];

    for &len in LENGTHS {
        c.bench_function(&format!("wasmsimdcomparison_{len}_f32_scalar"), move |b| {
            bench_scalar_32(b, len)
        });
        c.bench_function(&format!("wasmsimdcomparison_{len}_f64_scalar"), move |b| {
            bench_scalar_64(b, len)
        });
        c.bench_function(
            &format!("wasmsimdcomparison_{len}_f32_wasmsimd"),
            move |b| bench_wasmsimd_32(b, len),
        );
        c.bench_function(
            &format!("wasmsimdcomparison_{len}_f64_wasmsimd"),
            move |b| bench_wasmsimd_64(b, len),
        );
    }
}

criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark
}
criterion_main!(benches);
