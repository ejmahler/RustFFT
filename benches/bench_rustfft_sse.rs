use rustfft::sse::sse_butterflies::*;
use rustfft::sse::sse_prime_butterflies::{SseF32Butterfly7, SseF32Butterfly31, SseF64Butterfly7, SseF64Butterfly31};
use rustfft::sse::sse_radix4::SseRadix4;
use rustfft::sse::sse_radix4_otf::SseRadix4OnTheFly;
use rustfft::sse::sse_radix4_table::SseRadix4Table;
use rustfft::{FftNum, num_complex::Complex};
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftDirection};
use std::any::TypeId;
use std::sync::Arc;
mod config;

use criterion::{Bencher, BenchmarkId, Criterion, criterion_group, criterion_main};


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_f32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerSse::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);
    assert_eq!(fft.len(), len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerSse::new().unwrap();
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
    let mut planner = rustfft::FftPlannerSse::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_planned_multi_f64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerSse::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// All butterflies
fn sse_butterfly32_02(b: &mut Bencher) { bench_planned_multi_f32(b, 2);}
fn sse_butterfly32_03(b: &mut Bencher) { bench_planned_multi_f32(b, 3);}
fn sse_butterfly32_04(b: &mut Bencher) { bench_planned_multi_f32(b, 4);}
fn sse_butterfly32_05(b: &mut Bencher) { bench_planned_multi_f32(b, 5);}
fn sse_butterfly32_06(b: &mut Bencher) { bench_planned_multi_f32(b, 6);}
fn sse_butterfly32_07(b: &mut Bencher) { bench_planned_multi_f32(b, 7);}
fn sse_butterfly32_08(b: &mut Bencher) { bench_planned_multi_f32(b, 8);}
fn sse_butterfly32_09(b: &mut Bencher) { bench_planned_multi_f32(b, 9);}
fn sse_butterfly32_10(b: &mut Bencher) { bench_planned_multi_f32(b, 10);}
fn sse_butterfly32_11(b: &mut Bencher) { bench_planned_multi_f32(b, 11);}
fn sse_butterfly32_12(b: &mut Bencher) { bench_planned_multi_f32(b, 12);}
fn sse_butterfly32_13(b: &mut Bencher) { bench_planned_multi_f32(b, 13);}
fn sse_butterfly32_15(b: &mut Bencher) { bench_planned_multi_f32(b, 15);}
fn sse_butterfly32_16(b: &mut Bencher) { bench_planned_multi_f32(b, 16);}
fn sse_butterfly32_17(b: &mut Bencher) { bench_planned_multi_f32(b, 17);}
fn sse_butterfly32_19(b: &mut Bencher) { bench_planned_multi_f32(b, 19);}
fn sse_butterfly32_23(b: &mut Bencher) { bench_planned_multi_f32(b, 23);}
fn sse_butterfly32_24(b: &mut Bencher) { bench_planned_multi_f32(b, 24);}
fn sse_butterfly32_29(b: &mut Bencher) { bench_planned_multi_f32(b, 29);}
fn sse_butterfly32_31(b: &mut Bencher) { bench_planned_multi_f32(b, 31);}
fn sse_butterfly32_32(b: &mut Bencher) { bench_planned_multi_f32(b, 32);}

fn sse_butterfly64_02(b: &mut Bencher) { bench_planned_multi_f64(b, 2);}
fn sse_butterfly64_03(b: &mut Bencher) { bench_planned_multi_f64(b, 3);}
fn sse_butterfly64_04(b: &mut Bencher) { bench_planned_multi_f64(b, 4);}
fn sse_butterfly64_05(b: &mut Bencher) { bench_planned_multi_f64(b, 5);}
fn sse_butterfly64_06(b: &mut Bencher) { bench_planned_multi_f64(b, 6);}
fn sse_butterfly64_07(b: &mut Bencher) { bench_planned_multi_f64(b, 7);}
fn sse_butterfly64_08(b: &mut Bencher) { bench_planned_multi_f64(b, 8);}
fn sse_butterfly64_09(b: &mut Bencher) { bench_planned_multi_f64(b, 9);}
fn sse_butterfly64_10(b: &mut Bencher) { bench_planned_multi_f64(b, 10);}
fn sse_butterfly64_11(b: &mut Bencher) { bench_planned_multi_f64(b, 11);}
fn sse_butterfly64_12(b: &mut Bencher) { bench_planned_multi_f64(b, 12);}
fn sse_butterfly64_13(b: &mut Bencher) { bench_planned_multi_f64(b, 13);}
fn sse_butterfly64_15(b: &mut Bencher) { bench_planned_multi_f64(b, 15);}
fn sse_butterfly64_16(b: &mut Bencher) { bench_planned_multi_f64(b, 16);}
fn sse_butterfly64_17(b: &mut Bencher) { bench_planned_multi_f64(b, 17);}
fn sse_butterfly64_19(b: &mut Bencher) { bench_planned_multi_f64(b, 19);}
fn sse_butterfly64_23(b: &mut Bencher) { bench_planned_multi_f64(b, 23);}
fn sse_butterfly64_24(b: &mut Bencher) { bench_planned_multi_f64(b, 24);}
fn sse_butterfly64_29(b: &mut Bencher) { bench_planned_multi_f64(b, 29);}
fn sse_butterfly64_31(b: &mut Bencher) { bench_planned_multi_f64(b, 31);}
fn sse_butterfly64_32(b: &mut Bencher) { bench_planned_multi_f64(b, 32);}

// prime butterflies
fn sse_prime_butterfly32_07(b: &mut Bencher) { bench_planned_multi_f32(b, 7);}
fn sse_prime_butterfly32_11(b: &mut Bencher) { bench_planned_multi_f32(b, 11);}
fn sse_prime_butterfly32_13(b: &mut Bencher) { bench_planned_multi_f32(b, 13);}
fn sse_prime_butterfly32_17(b: &mut Bencher) { bench_planned_multi_f32(b, 17);}
fn sse_prime_butterfly32_19(b: &mut Bencher) { bench_planned_multi_f32(b, 19);}
fn sse_prime_butterfly32_23(b: &mut Bencher) { bench_planned_multi_f32(b, 23);}
fn sse_prime_butterfly32_29(b: &mut Bencher) { bench_planned_multi_f32(b, 29);}
fn sse_prime_butterfly32_31(b: &mut Bencher) { bench_planned_multi_f32(b, 31);}
fn sse_prime_butterfly64_07(b: &mut Bencher) { bench_planned_multi_f64(b, 7);}
fn sse_prime_butterfly64_11(b: &mut Bencher) { bench_planned_multi_f64(b, 11);}
fn sse_prime_butterfly64_13(b: &mut Bencher) { bench_planned_multi_f64(b, 13);}
fn sse_prime_butterfly64_17(b: &mut Bencher) { bench_planned_multi_f64(b, 17);}
fn sse_prime_butterfly64_19(b: &mut Bencher) { bench_planned_multi_f64(b, 19);}
fn sse_prime_butterfly64_23(b: &mut Bencher) { bench_planned_multi_f64(b, 23);}
fn sse_prime_butterfly64_29(b: &mut Bencher) { bench_planned_multi_f64(b, 29);}
fn sse_prime_butterfly64_31(b: &mut Bencher) { bench_planned_multi_f64(b, 31);}

// Powers of 2
fn sse_planned32_p2_00000064(b: &mut Bencher) { bench_planned_f32(b, 64); }
fn sse_planned32_p2_00000128(b: &mut Bencher) { bench_planned_f32(b, 128); }
fn sse_planned32_p2_00000256(b: &mut Bencher) { bench_planned_f32(b, 256); }
fn sse_planned32_p2_00000512(b: &mut Bencher) { bench_planned_f32(b, 512); }
fn sse_planned32_p2_00001024(b: &mut Bencher) { bench_planned_f32(b, 1024); }
fn sse_planned32_p2_00002048(b: &mut Bencher) { bench_planned_f32(b, 2048); }
fn sse_planned32_p2_00004096(b: &mut Bencher) { bench_planned_f32(b, 4096); }
fn sse_planned32_p2_00016384(b: &mut Bencher) { bench_planned_f32(b, 16384); }
fn sse_planned32_p2_00065536(b: &mut Bencher) { bench_planned_f32(b, 65536); }
fn sse_planned32_p2_01048576(b: &mut Bencher) { bench_planned_f32(b, 1048576); }

fn sse_planned64_p2_00000064(b: &mut Bencher) { bench_planned_f64(b, 64); }
fn sse_planned64_p2_00000128(b: &mut Bencher) { bench_planned_f64(b, 128); }
fn sse_planned64_p2_00000256(b: &mut Bencher) { bench_planned_f64(b, 256); }
fn sse_planned64_p2_00000512(b: &mut Bencher) { bench_planned_f64(b, 512); }
fn sse_planned64_p2_00001024(b: &mut Bencher) { bench_planned_f64(b, 1024); }
fn sse_planned64_p2_00002048(b: &mut Bencher) { bench_planned_f64(b, 2048); }
fn sse_planned64_p2_00004096(b: &mut Bencher) { bench_planned_f64(b, 4096); }
fn sse_planned64_p2_00016384(b: &mut Bencher) { bench_planned_f64(b, 16384); }
fn sse_planned64_p2_00065536(b: &mut Bencher) { bench_planned_f64(b, 65536); }
fn sse_planned64_p2_01048576(b: &mut Bencher) { bench_planned_f64(b, 1048576); }


// Powers of 7
fn sse_planned32_p7_00343(b: &mut Bencher) { bench_planned_f32(b,   343); }
fn sse_planned32_p7_02401(b: &mut Bencher) { bench_planned_f32(b,  2401); }
fn sse_planned32_p7_16807(b: &mut Bencher) { bench_planned_f32(b, 16807); }

fn sse_planned64_p7_00343(b: &mut Bencher) { bench_planned_f64(b,   343); }
fn sse_planned64_p7_02401(b: &mut Bencher) { bench_planned_f64(b,  2401); }
fn sse_planned64_p7_16807(b: &mut Bencher) { bench_planned_f64(b, 16807); }

// Prime lengths
fn sse_planned32_prime_0149(b: &mut Bencher)     { bench_planned_f32(b,  149); }
fn sse_planned32_prime_0151(b: &mut Bencher)     { bench_planned_f32(b,  151); }
fn sse_planned32_prime_0251(b: &mut Bencher)     { bench_planned_f32(b,  251); }
fn sse_planned32_prime_0257(b: &mut Bencher)     { bench_planned_f32(b,  257); }
fn sse_planned32_prime_2017(b: &mut Bencher)     { bench_planned_f32(b,  2017); }
fn sse_planned32_prime_2879(b: &mut Bencher)     { bench_planned_f32(b,  2879); }
fn sse_planned32_prime_65521(b: &mut Bencher)    { bench_planned_f32(b, 65521); }
fn sse_planned32_prime_746497(b: &mut Bencher)   { bench_planned_f32(b,746497); }

fn sse_planned64_prime_0149(b: &mut Bencher)     { bench_planned_f64(b,  149); }
fn sse_planned64_prime_0151(b: &mut Bencher)     { bench_planned_f64(b,  151); }
fn sse_planned64_prime_0251(b: &mut Bencher)     { bench_planned_f64(b,  251); }
fn sse_planned64_prime_0257(b: &mut Bencher)     { bench_planned_f64(b,  257); }
fn sse_planned64_prime_2017(b: &mut Bencher)     { bench_planned_f64(b,  2017); }
fn sse_planned64_prime_2879(b: &mut Bencher)     { bench_planned_f64(b,  2879); }
fn sse_planned64_prime_65521(b: &mut Bencher)    { bench_planned_f64(b, 65521); }
fn sse_planned64_prime_746497(b: &mut Bencher)   { bench_planned_f64(b,746497); }

// small mixed composites
fn sse_planned32_composite_000018(b: &mut Bencher) { bench_planned_f32(b,  00018); }
fn sse_planned32_composite_000360(b: &mut Bencher) { bench_planned_f32(b,  00360); }
fn sse_planned32_composite_001200(b: &mut Bencher) { bench_planned_f32(b,  01200); }
fn sse_planned32_composite_044100(b: &mut Bencher) { bench_planned_f32(b,  44100); }
fn sse_planned32_composite_048000(b: &mut Bencher) { bench_planned_f32(b,  48000); }
fn sse_planned32_composite_046656(b: &mut Bencher) { bench_planned_f32(b,  46656); }

fn sse_planned64_composite_000018(b: &mut Bencher) { bench_planned_f64(b,  00018); }
fn sse_planned64_composite_000360(b: &mut Bencher) { bench_planned_f64(b,  00360); }
fn sse_planned64_composite_001200(b: &mut Bencher) { bench_planned_f64(b,  01200); }
fn sse_planned64_composite_044100(b: &mut Bencher) { bench_planned_f64(b,  44100); }
fn sse_planned64_composite_048000(b: &mut Bencher) { bench_planned_f64(b,  48000); }
fn sse_planned64_composite_046656(b: &mut Bencher) { bench_planned_f64(b,  46656); }

fn plan_butterfly_fft<T: FftNum>(len: usize) -> Arc<dyn Fft<T>> {
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    if id_t == id_f32 {
        unsafe { 
            match len {
                2 => Arc::new(SseF32Butterfly2::new(FftDirection::Forward)),
                3 => Arc::new(SseF32Butterfly3::new(FftDirection::Forward)),
                4 => Arc::new(SseF32Butterfly4::new(FftDirection::Forward)),
                5 => Arc::new(SseF32Butterfly5::new(FftDirection::Forward)),
                6 => Arc::new(SseF32Butterfly6::new(FftDirection::Forward)),
                7 => Arc::new(SseF32Butterfly7::new(FftDirection::Forward)),
                8 => Arc::new(SseF32Butterfly8::new(FftDirection::Forward)),
                16 => Arc::new(SseF32Butterfly16::new(FftDirection::Forward)),
                31 => Arc::new(SseF32Butterfly31::new(FftDirection::Forward)),
                32 => Arc::new(SseF32Butterfly32::new(FftDirection::Forward)),
                _ => panic!("Invalid butterfly size: {}", len),
            }
        }
    } else if id_t == id_f64 {
        unsafe {
            match len {
                2 => Arc::new(SseF64Butterfly2::new(FftDirection::Forward)),
                3 => Arc::new(SseF64Butterfly3::new(FftDirection::Forward)),
                4 => Arc::new(SseF64Butterfly4::new(FftDirection::Forward)),
                5 => Arc::new(SseF64Butterfly5::new(FftDirection::Forward)),
                6 => Arc::new(SseF64Butterfly6::new(FftDirection::Forward)),
                7 => Arc::new(SseF64Butterfly7::new(FftDirection::Forward)),
                8 => Arc::new(SseF64Butterfly8::new(FftDirection::Forward)),
                16 => Arc::new(SseF64Butterfly16::new(FftDirection::Forward)),
                31 => Arc::new(SseF64Butterfly31::new(FftDirection::Forward)),
                32 => Arc::new(SseF64Butterfly32::new(FftDirection::Forward)),
                _ => panic!("Invalid butterfly size: {}", len),
            }
        }
    } else {
        panic!("Invalid T for constructing SSE FFT");
    }
}

fn bench_radix4_old<T: FftNum>(b: &mut Bencher, power2: u32) {
    let len = 2usize.pow(power2);
    let base_len = if power2 % 2 == 0 {
        16
    } else {
        32
    };
    assert!(len > base_len);
    let base_fft = plan_butterfly_fft(base_len);
    let k = (power2 - base_len.trailing_zeros()) / 2;
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    let fft = if id_t == id_f32 {
        Arc::new(SseRadix4::<f32, T>::new(k, base_fft).unwrap()) as Arc<dyn Fft<T>>
    } else if id_t == id_f64 {
        Arc::new(SseRadix4::<f64, T>::new(k, base_fft).unwrap()) as Arc<dyn Fft<T>>
    } else {
        panic!("Invalid T for constructing SSE FFT");
    };

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_radix4_table<T: FftNum>(b: &mut Bencher, power2: u32) {
    let len = 2usize.pow(power2);
    let base_len = if power2 % 2 == 0 {
        16
    } else {
        32
    };
    assert!(len > base_len);
    let base_fft = plan_butterfly_fft(base_len);
    let k = (power2 - base_len.trailing_zeros()) / 2;
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    let fft = if id_t == id_f32 {
        Arc::new(SseRadix4Table::<f32, T>::new(k, base_fft)) as Arc<dyn Fft<T>>
    } else if id_t == id_f64 {
        Arc::new(SseRadix4Table::<f64, T>::new(k, base_fft)) as Arc<dyn Fft<T>>
    } else {
        panic!("Invalid T for constructing SSE FFT");
    };

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}
fn bench_radix4_otf<T: FftNum>(b: &mut Bencher, power2: u32) {
    let len = 2usize.pow(power2);
    let base_len = if power2 % 2 == 0 {
        16
    } else {
        32
    };
    assert!(len > base_len);
    let base_fft = plan_butterfly_fft(base_len);
    let k = (power2 - base_len.trailing_zeros()) / 2;
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    let fft = if id_t == id_f32 {
        Arc::new(SseRadix4OnTheFly::<f32, T>::new(k, base_fft)) as Arc<dyn Fft<T>>
    } else if id_t == id_f64 {
        Arc::new(SseRadix4OnTheFly::<f64, T>::new(k, base_fft)) as Arc<dyn Fft<T>>
    } else {
        panic!("Invalid T for constructing SSE FFT");
    };

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}


fn criterion_benchmark_radix4(c: &mut Criterion) {
    {
        let mut group32 = c.benchmark_group("compare_radix4_sse_f32");
        
        for power2 in 6..30 {
            let len = 2usize.pow(power2);
            group32.bench_with_input(BenchmarkId::new("old", len), &len,  |b, _| {
                bench_radix4_old::<f32>(b, power2)
            });
            group32.bench_with_input(BenchmarkId::new("tab", len), &len,  |b, _| {
                bench_radix4_table::<f32>(b, power2)
            });
            group32.bench_with_input(BenchmarkId::new("fly", len), &len,  |b, _| {
                bench_radix4_otf::<f32>(b, power2)
            });
        }
    }

    {
        let mut group64 = c.benchmark_group("compare_radix4_sse_f64");

        for power2 in 6..30 {
            let len = 2usize.pow(power2);
            group64.bench_with_input(BenchmarkId::new("old", len), &len, |b, _| {
                bench_radix4_old::<f64>(b, power2)
            });
            group64.bench_with_input(BenchmarkId::new("tab", len), &len, |b, _| {
                bench_radix4_table::<f64>(b, power2)
            });
            group64.bench_with_input(BenchmarkId::new("fly", len), &len, |b, _| {
                bench_radix4_otf::<f64>(b, power2)
            });
        }
    }
}


fn criterion_benchmark(c: &mut Criterion) {
    config::register_benchmarks!(
        c,
        sse_butterfly32_02,
        sse_butterfly32_03,
        sse_butterfly32_04,
        sse_butterfly32_05,
        sse_butterfly32_06,
        sse_butterfly32_07,
        sse_butterfly32_08,
        sse_butterfly32_09,
        sse_butterfly32_10,
        sse_butterfly32_11,
        sse_butterfly32_12,
        sse_butterfly32_13,
        sse_butterfly32_15,
        sse_butterfly32_16,
        sse_butterfly32_17,
        sse_butterfly32_19,
        sse_butterfly32_23,
        sse_butterfly32_24,
        sse_butterfly32_29,
        sse_butterfly32_31,
        sse_butterfly32_32,
        sse_butterfly64_02,
        sse_butterfly64_03,
        sse_butterfly64_04,
        sse_butterfly64_05,
        sse_butterfly64_06,
        sse_butterfly64_07,
        sse_butterfly64_08,
        sse_butterfly64_09,
        sse_butterfly64_10,
        sse_butterfly64_11,
        sse_butterfly64_12,
        sse_butterfly64_13,
        sse_butterfly64_15,
        sse_butterfly64_16,
        sse_butterfly64_17,
        sse_butterfly64_19,
        sse_butterfly64_23,
        sse_butterfly64_24,
        sse_butterfly64_29,
        sse_butterfly64_31,
        sse_butterfly64_32,
        sse_prime_butterfly32_07,
        sse_prime_butterfly32_11,
        sse_prime_butterfly32_13,
        sse_prime_butterfly32_17,
        sse_prime_butterfly32_19,
        sse_prime_butterfly32_23,
        sse_prime_butterfly32_29,
        sse_prime_butterfly32_31,
        sse_prime_butterfly64_07,
        sse_prime_butterfly64_11,
        sse_prime_butterfly64_13,
        sse_prime_butterfly64_17,
        sse_prime_butterfly64_19,
        sse_prime_butterfly64_23,
        sse_prime_butterfly64_29,
        sse_prime_butterfly64_31,
        sse_planned32_p2_00000064,
        sse_planned32_p2_00000128,
        sse_planned32_p2_00000256,
        sse_planned32_p2_00000512,
        sse_planned32_p2_00001024,
        sse_planned32_p2_00002048,
        sse_planned32_p2_00004096,
        sse_planned32_p2_00016384,
        sse_planned32_p2_00065536,
        sse_planned32_p2_01048576,
        sse_planned64_p2_00000064,
        sse_planned64_p2_00000128,
        sse_planned64_p2_00000256,
        sse_planned64_p2_00000512,
        sse_planned64_p2_00001024,
        sse_planned64_p2_00002048,
        sse_planned64_p2_00004096,
        sse_planned64_p2_00016384,
        sse_planned64_p2_00065536,
        sse_planned64_p2_01048576,
        sse_planned32_p7_00343,
        sse_planned32_p7_02401,
        sse_planned32_p7_16807,
        sse_planned64_p7_00343,
        sse_planned64_p7_02401,
        sse_planned64_p7_16807,
        sse_planned32_prime_0149,
        sse_planned32_prime_0151,
        sse_planned32_prime_0251,
        sse_planned32_prime_0257,
        sse_planned32_prime_2017,
        sse_planned32_prime_2879,
        sse_planned32_prime_65521,
        sse_planned32_prime_746497,
        sse_planned64_prime_0149,
        sse_planned64_prime_0151,
        sse_planned64_prime_0251,
        sse_planned64_prime_0257,
        sse_planned64_prime_2017,
        sse_planned64_prime_2879,
        sse_planned64_prime_65521,
        sse_planned64_prime_746497,
        sse_planned32_composite_000018,
        sse_planned32_composite_000360,
        sse_planned32_composite_001200,
        sse_planned32_composite_044100,
        sse_planned32_composite_048000,
        sse_planned32_composite_046656,
        sse_planned64_composite_000018,
        sse_planned64_composite_000360,
        sse_planned64_composite_001200,
        sse_planned64_composite_044100,
        sse_planned64_composite_048000,
        sse_planned64_composite_046656,
    );
}

criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark, criterion_benchmark_radix4
}
criterion_main!(benches);
