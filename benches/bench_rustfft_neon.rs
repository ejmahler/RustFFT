use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftNum, FftDirection, RadixFactor};
use rustfft::neon::neon_butterflies::*;
use rustfft::neon::neon_prime_butterflies::*;
use rustfft::neon::neon_radixn::NeonRadixN;
use rustfft::simd::simd_vector::SimdVector;
use rustfft::simd::simd_transpose::table_transpose;
use rustfft::array_utils::{reverse_remainders, TransposeFactor, validate_and_zip};
use std::sync::Arc;
use std::any::TypeId;
mod config;
use core::arch::aarch64::*;
use criterion::BenchmarkId;

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


fn plan_butterfly_fft<T: FftNum>(len: usize) -> Option<Arc<dyn Fft<T>>> {
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    if id_t == id_f32 {
        unsafe { 
            match len {
                2 => Some(Arc::new(NeonF32Butterfly2::new(FftDirection::Forward))),
                3 => Some(Arc::new(NeonF32Butterfly3::new(FftDirection::Forward))),
                4 => Some(Arc::new(NeonF32Butterfly4::new(FftDirection::Forward))),
                5 => Some(Arc::new(NeonF32Butterfly5::new(FftDirection::Forward))),
                6 => Some(Arc::new(NeonF32Butterfly6::new(FftDirection::Forward))),
                7 => Some(Arc::new(NeonF32Butterfly7::new(FftDirection::Forward))),
                8 => Some(Arc::new(NeonF32Butterfly8::new(FftDirection::Forward))),
                11 => Some(Arc::new(NeonF32Butterfly11::new(FftDirection::Forward))),
                16 => Some(Arc::new(NeonF32Butterfly16::new(FftDirection::Forward))),
                31 => Some(Arc::new(NeonF32Butterfly31::new(FftDirection::Forward))),
                32 => Some(Arc::new(NeonF32Butterfly32::new(FftDirection::Forward))),
                _ => None,
            }
        }
    } else if id_t == id_f64 {
        unsafe {
            match len {
                2 => Some(Arc::new(NeonF64Butterfly2::new(FftDirection::Forward))),
                3 => Some(Arc::new(NeonF64Butterfly3::new(FftDirection::Forward))),
                4 => Some(Arc::new(NeonF64Butterfly4::new(FftDirection::Forward))),
                5 => Some(Arc::new(NeonF64Butterfly5::new(FftDirection::Forward))),
                6 => Some(Arc::new(NeonF64Butterfly6::new(FftDirection::Forward))),
                7 => Some(Arc::new(NeonF64Butterfly7::new(FftDirection::Forward))),
                8 => Some(Arc::new(NeonF64Butterfly8::new(FftDirection::Forward))),
                11 => Some(Arc::new(NeonF64Butterfly11::new(FftDirection::Forward))),
                16 => Some(Arc::new(NeonF64Butterfly16::new(FftDirection::Forward))),
                31 => Some(Arc::new(NeonF64Butterfly31::new(FftDirection::Forward))),
                32 => Some(Arc::new(NeonF64Butterfly32::new(FftDirection::Forward))),
                _ => None,
            }
        }
    } else {
        panic!("Invalid T for constructing NEON FFT");
    }
}

fn construct_radixn<T: FftNum>(factors: &[RadixFactor], base_fft: Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
    
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    if id_t == id_f32 {
        Arc::new(NeonRadixN::<float32x4_t, T>::new(factors, base_fft))
    } else if id_t == id_f64 {
        Arc::new(NeonRadixN::<float64x2_t, T>::new(factors, base_fft))
    } else {
        panic!("Invalid T for constructing NEON FFT")
    }
}

fn find_butterfly<T: FftNum>(base: usize, mut power: u32) -> (u32, Arc<dyn Fft<T>>) {
    loop {
        if let Some(butterfly) = plan_butterfly_fft(base.pow(power)) {
            return (power, butterfly);
        }

        power -= 1;
        if power == 0 {
            panic!("Couldn't find butterfly for base={base}");
        }
    }
}


// Tests incorporating a factor of 31 by using it as a cross-FFT for RadixN
fn bench_radixn<V: SimdVector>(b: &mut Bencher, factor: RadixFactor, power: u32) {
    let len : usize = factor.radix().pow(power);
    let (base_power, base_fft) = find_butterfly::<V::ScalarType>(factor.radix(), power);

    let cross_power = power - base_power;
    let factors = std::iter::repeat_n(factor, cross_power as usize).collect::<Vec<_>>();

    let height : usize = base_fft.len();
    let width : usize = len / height;

    // set up our list of transpose factors - it's the same list but reversed, and we want to
    // collapse duplicates. Note that we are only de-duplicating adjacent factors: if we're
    // passed 7 * 2 * 7, we can't collapse the sevens because the exact order matters.
    let mut transpose_factors: Vec<TransposeFactor> = Vec::with_capacity(factors.len());
    for f in factors.iter().rev() {
        let mut push_new = true;
        if let Some(last) = transpose_factors.last_mut() {
            if last.factor == *f {
                last.count += 1;
                push_new = false;
            }
        }
        if push_new {
            transpose_factors.push(TransposeFactor {
                factor: *f,
                count: 1,
            });
        }
    }

    // Precompute where each column lands. Working this out per call costs two hardware divides
    // and an out-of-line `reverse_remainders` call per column, which is a large share of a
    // short FFT, and much more so on x86 where a 64-bit divide takes tens of cycles.
    let reversed_columns: Box<[usize]> = (0..width)
        .map(|x| reverse_remainders(x, &transpose_factors))
        .collect();
    // `table_transpose` indexes unchecked, and relies on this.
    assert!(reversed_columns.iter().all(|&r| r < width));

    let buffer_in : Vec<Complex<V::ScalarType>> = vec![Complex::zero(); len * 10];
    let mut buffer_out : Vec<Complex<V::ScalarType>> = vec![Complex::zero(); len * 10];
    b.iter(|| {
        validate_and_zip(&buffer_in, &mut buffer_out, &mut [], len, 0, |chunk_in, chunk_out, _| {
            unsafe { table_transpose::<V>(height, &reversed_columns, chunk_in, chunk_out) };
        })
        
    });
}
fn criterion_benchmark_radixn(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("neon64_radixn_power7");

        for power7 in 2..8 {
            let len = 7usize.pow(power7);
            group.bench_with_input(BenchmarkId::new("radixn", len), &len, |b, _| {
                bench_radixn::<float64x2_t>(b, RadixFactor::Factor7, power7)
            });
        }
    }
}



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
    targets = criterion_benchmark, criterion_benchmark_radixn
}
criterion_main!(benches);
