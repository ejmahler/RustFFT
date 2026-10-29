use rustfft::{FftNum, num_complex::Complex};
use rustfft::num_traits::Zero;
use rustfft::Fft;
use rustfft::avx::avx_radix4::AvxRadix4;
use std::any::TypeId;
use std::arch::x86_64::__m256d;
use std::{arch::x86_64::__m256, sync::Arc};
mod config;

use criterion::{Bencher, BenchmarkId, Criterion, criterion_group, criterion_main};


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_f32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerAvx::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);
    assert_eq!(fft.len(), len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerAvx::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);
    assert_eq!(fft.len(), len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Powers of 2
fn avx_planned32_p2_00000064(b: &mut Bencher) { bench_planned_f32(b, 64); }
fn avx_planned32_p2_00000128(b: &mut Bencher) { bench_planned_f32(b, 128); }
fn avx_planned32_p2_00000256(b: &mut Bencher) { bench_planned_f32(b, 256); }
fn avx_planned32_p2_00000512(b: &mut Bencher) { bench_planned_f32(b, 512); }
fn avx_planned32_p2_00001024(b: &mut Bencher) { bench_planned_f32(b, 1024); }
fn avx_planned32_p2_00002048(b: &mut Bencher) { bench_planned_f32(b, 2048); }
fn avx_planned32_p2_00004096(b: &mut Bencher) { bench_planned_f32(b, 4096); }
fn avx_planned32_p2_00016384(b: &mut Bencher) { bench_planned_f32(b, 16384); }
fn avx_planned32_p2_00065536(b: &mut Bencher) { bench_planned_f32(b, 65536); }
fn avx_planned32_p2_01048576(b: &mut Bencher) { bench_planned_f32(b, 1048576); }

fn avx_planned64_p2_00000064(b: &mut Bencher) { bench_planned_f64(b, 64); }
fn avx_planned64_p2_00000128(b: &mut Bencher) { bench_planned_f64(b, 128); }
fn avx_planned64_p2_00000256(b: &mut Bencher) { bench_planned_f64(b, 256); }
fn avx_planned64_p2_00000512(b: &mut Bencher) { bench_planned_f64(b, 512); }
fn avx_planned64_p2_00001024(b: &mut Bencher) { bench_planned_f64(b, 1024); }
fn avx_planned64_p2_00002048(b: &mut Bencher) { bench_planned_f64(b, 2048); }
fn avx_planned64_p2_00004096(b: &mut Bencher) { bench_planned_f64(b, 4096); }
fn avx_planned64_p2_00016384(b: &mut Bencher) { bench_planned_f64(b, 16384); }
fn avx_planned64_p2_00065536(b: &mut Bencher) { bench_planned_f64(b, 65536); }
fn avx_planned64_p2_01048576(b: &mut Bencher) { bench_planned_f64(b, 1048576); }


// Powers of 7
fn avx_planned32_p7_00343(b: &mut Bencher) { bench_planned_f32(b,   343); }
fn avx_planned32_p7_02401(b: &mut Bencher) { bench_planned_f32(b,  2401); }
fn avx_planned32_p7_16807(b: &mut Bencher) { bench_planned_f32(b, 16807); }

fn avx_planned64_p7_00343(b: &mut Bencher) { bench_planned_f64(b,   343); }
fn avx_planned64_p7_02401(b: &mut Bencher) { bench_planned_f64(b,  2401); }
fn avx_planned64_p7_16807(b: &mut Bencher) { bench_planned_f64(b, 16807); }

// small mixed composites
fn avx_planned32_composite_000018(b: &mut Bencher) { bench_planned_f32(b,  00018); }
fn avx_planned32_composite_000360(b: &mut Bencher) { bench_planned_f32(b,  00360); }
fn avx_planned32_composite_001200(b: &mut Bencher) { bench_planned_f32(b,  01200); }
fn avx_planned32_composite_044100(b: &mut Bencher) { bench_planned_f32(b,  44100); }
fn avx_planned32_composite_048000(b: &mut Bencher) { bench_planned_f32(b,  48000); }
fn avx_planned32_composite_046656(b: &mut Bencher) { bench_planned_f32(b,  46656); }

fn avx_planned64_composite_000018(b: &mut Bencher) { bench_planned_f64(b,  00018); }
fn avx_planned64_composite_000360(b: &mut Bencher) { bench_planned_f64(b,  00360); }
fn avx_planned64_composite_001200(b: &mut Bencher) { bench_planned_f64(b,  01200); }
fn avx_planned64_composite_044100(b: &mut Bencher) { bench_planned_f64(b,  44100); }
fn avx_planned64_composite_048000(b: &mut Bencher) { bench_planned_f64(b,  48000); }
fn avx_planned64_composite_046656(b: &mut Bencher) { bench_planned_f64(b,  46656); }

fn construst_radix4<T: FftNum>(base_fft: Arc<dyn Fft<T>>, k: u32) -> Arc<dyn Fft<T>> {
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    if id_t == id_f32 {
        Arc::new(AvxRadix4::<__m256, T>::new(k, base_fft))
    } else if id_t == id_f64 {
        Arc::new(AvxRadix4::<__m256d, T>::new(k, base_fft))
    } else {
        panic!("Invalid type for constructing simd fft")
    }
}

fn bench_radix4<T: FftNum>(b: &mut Bencher, k: u32, base_len: usize) {
    let base_fft = rustfft::FftPlannerAvx::<T>::new().unwrap().plan_fft_forward(base_len);
    let fft = construst_radix4(base_fft, k);

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

fn bench_power2(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("avx64_base256_power4");

        let base = 256;
        for k in 1..15 {
            let len = base * 4usize.pow(k);
            group.bench_with_input(BenchmarkId::new("radix4", len), &len, |b, _| {
                bench_radix4::<f64>(b, k, base);
            });
            group.bench_with_input(BenchmarkId::new("planned", len), &len, |b, _| {
                bench_planned_f64(b, len);
            });
        }
    }
    {
        let mut group = c.benchmark_group("avx32_base256_power4");

        let base = 256;
        for k in 1..15 {
            let len = base * 4usize.pow(k);
            group.bench_with_input(BenchmarkId::new("radix4", len), &len, |b, _| {
                bench_radix4::<f32>(b, k, base);
            });
            group.bench_with_input(BenchmarkId::new("planned", len), &len, |b, _| {
                bench_planned_f32(b, len);
            });
        }
    }
}




fn criterion_benchmark(c: &mut Criterion) {
    config::register_benchmarks!(
        c,
        avx_planned32_p2_00000064,
        avx_planned32_p2_00000128,
        avx_planned32_p2_00000256,
        avx_planned32_p2_00000512,
        avx_planned32_p2_00001024,
        avx_planned32_p2_00002048,
        avx_planned32_p2_00004096,
        avx_planned32_p2_00016384,
        avx_planned32_p2_00065536,
        avx_planned32_p2_01048576,
        avx_planned64_p2_00000064,
        avx_planned64_p2_00000128,
        avx_planned64_p2_00000256,
        avx_planned64_p2_00000512,
        avx_planned64_p2_00001024,
        avx_planned64_p2_00002048,
        avx_planned64_p2_00004096,
        avx_planned64_p2_00016384,
        avx_planned64_p2_00065536,
        avx_planned64_p2_01048576,
        avx_planned32_p7_00343,
        avx_planned32_p7_02401,
        avx_planned32_p7_16807,
        avx_planned64_p7_00343,
        avx_planned64_p7_02401,
        avx_planned64_p7_16807,
        avx_planned32_composite_000018,
        avx_planned32_composite_000360,
        avx_planned32_composite_001200,
        avx_planned32_composite_044100,
        avx_planned32_composite_048000,
        avx_planned32_composite_046656,
        avx_planned64_composite_000018,
        avx_planned64_composite_000360,
        avx_planned64_composite_001200,
        avx_planned64_composite_044100,
        avx_planned64_composite_048000,
        avx_planned64_composite_046656,
    );
}

criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark, bench_power2
}
criterion_main!(benches);
