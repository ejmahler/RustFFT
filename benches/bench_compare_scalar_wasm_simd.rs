#![feature(test)]
extern crate rustfft;
extern crate test;

use paste::paste;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::sync::Arc;
use test::Bencher;

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

// Create benches using functions taking one argument
macro_rules! make_benches {
    ($name:ident, { $($len:literal),* }) => {
        paste! {
            $(
                #[bench]
                fn [<$name _ $len _f32_scalar>](b: &mut Bencher)  {
                    [<bench_scalar_32>](b, $len);
                }

                #[bench]
                fn [<$name _ $len _f64_scalar>](b: &mut Bencher)  {
                    [<bench_scalar_64>](b, $len);
                }

                #[bench]
                fn [<$name _ $len _f32_wasmsimd>](b: &mut Bencher)  {
                    [<bench_wasmsimd_32>](b, $len);
                }

                #[bench]
                fn [<$name _ $len _f64_wasmsimd>](b: &mut Bencher)  {
                    [<bench_wasmsimd_64>](b, $len);
                }
            )*
        }
    }
}

make_benches!(wasmsimdcomparison, {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072});
make_benches!(wasmsimdcomparison, { 262144, 524288, 1048576, 2097152, 4194304 });
