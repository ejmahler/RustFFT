use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::RadixFactor;
use rustfft::simd::simd_vector::SimdVector;
use rustfft::simd::simd_transpose::{unrolled_transpose, table_transpose_original, table_transpose};
use rustfft::array_utils::{reverse_remainders, TransposeFactor, validate_and_zip};
mod config;
use core::arch::aarch64::*;
use criterion::BenchmarkId;
use std::hint::unreachable_unchecked;

use criterion::{criterion_group, criterion_main, Bencher, Criterion};
/*

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



fn plan_butterfly_fft<T: FftNum>(len: usize) -> Option<Arc<dyn Fft<T>>> {
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    if id_t == id_f32 {
        unsafe { 
            match len {
                2 => Some(Arc::new(SseF32Butterfly2::new(FftDirection::Forward))),
                3 => Some(Arc::new(SseF32Butterfly3::new(FftDirection::Forward))),
                4 => Some(Arc::new(SseF32Butterfly4::new(FftDirection::Forward))),
                5 => Some(Arc::new(SseF32Butterfly5::new(FftDirection::Forward))),
                6 => Some(Arc::new(SseF32Butterfly6::new(FftDirection::Forward))),
                7 => Some(Arc::new(SseF32Butterfly7::new(FftDirection::Forward))),
                8 => Some(Arc::new(SseF32Butterfly8::new(FftDirection::Forward))),
                11 => Some(Arc::new(SseF32Butterfly11::new(FftDirection::Forward))),
                16 => Some(Arc::new(SseF32Butterfly16::new(FftDirection::Forward))),
                31 => Some(Arc::new(SseF32Butterfly31::new(FftDirection::Forward))),
                32 => Some(Arc::new(SseF32Butterfly32::new(FftDirection::Forward))),
                _ => None,
            }
        }
    } else if id_t == id_f64 {
        unsafe {
            match len {
                2 => Some(Arc::new(SseF64Butterfly2::new(FftDirection::Forward))),
                3 => Some(Arc::new(SseF64Butterfly3::new(FftDirection::Forward))),
                4 => Some(Arc::new(SseF64Butterfly4::new(FftDirection::Forward))),
                5 => Some(Arc::new(SseF64Butterfly5::new(FftDirection::Forward))),
                6 => Some(Arc::new(SseF64Butterfly6::new(FftDirection::Forward))),
                7 => Some(Arc::new(SseF64Butterfly7::new(FftDirection::Forward))),
                8 => Some(Arc::new(SseF64Butterfly8::new(FftDirection::Forward))),
                11 => Some(Arc::new(SseF64Butterfly11::new(FftDirection::Forward))),
                16 => Some(Arc::new(SseF64Butterfly16::new(FftDirection::Forward))),
                31 => Some(Arc::new(SseF64Butterfly31::new(FftDirection::Forward))),
                32 => Some(Arc::new(SseF64Butterfly32::new(FftDirection::Forward))),
                _ => None,
            }
        }
    } else {
        panic!("Invalid T for constructing sse FFT");
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
}*/





// Tests incorporating a factor of 31 by using it as a cross-FFT for RadixN
fn bench_original_transpose<V: SimdVector>(b: &mut Bencher, factor: RadixFactor, power: u32, base_len: usize) {
    let cross_len = factor.radix().pow(power);
    let len : usize = base_len * cross_len;
    let factors = std::iter::repeat_n(factor, power as usize).collect::<Vec<_>>();

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
    let width = len / base_len;
    let reversed_columns: Box<[usize]> = (0..width)
        .map(|x| reverse_remainders(x, &transpose_factors))
        .collect();
    // `table_transpose` indexes unchecked, and relies on this.
    assert!(reversed_columns.iter().all(|&r| r < width));
    let unroll_factor = transpose_factors.first().map(|f| f.factor).unwrap();


    let buffer_in : Vec<Complex<V::ScalarType>> = vec![Complex::zero(); len * 10];
    let mut buffer_out : Vec<Complex<V::ScalarType>> = vec![Complex::zero(); len * 10];
    b.iter(|| {
        validate_and_zip(&buffer_in, &mut buffer_out, &mut [], len, 0, |chunk_in, chunk_out, _| {
            let (height, columns) = (base_len, &*reversed_columns);
            match unroll_factor {
                RadixFactor::Factor2 => unsafe { table_transpose_original::<_, 2>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor3 => unsafe { table_transpose_original::<_, 3>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor4 => unsafe { table_transpose_original::<_, 4>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor5 => unsafe { table_transpose_original::<_, 5>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor6 => unsafe { table_transpose_original::<_, 6>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor7 => unsafe { table_transpose_original::<_, 7>(height, columns, chunk_in, chunk_out) },
                _ => unsafe { unreachable_unchecked() },
            }
        })
        
    });
}

// Tests incorporating a factor of 31 by using it as a cross-FFT for RadixN
fn bench_simd_transpose<V: SimdVector>(b: &mut Bencher, factor: RadixFactor, power: u32, base_len: usize) {
    let cross_len = factor.radix().pow(power);
    let len : usize = base_len * cross_len;
    let factors = std::iter::repeat_n(factor, power as usize).collect::<Vec<_>>();

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
    let width = len / base_len;
    let reversed_columns: Box<[usize]> = (0..width)
        .map(|x| reverse_remainders(x, &transpose_factors))
        .collect();
    // `table_transpose` indexes unchecked, and relies on this.
    assert!(reversed_columns.iter().all(|&r| r < width));
    let unroll_factor = *factors.first().unwrap();

    let buffer_in : Vec<Complex<V::ScalarType>> = vec![Complex::zero(); len * 10];
    let mut buffer_out : Vec<Complex<V::ScalarType>> = vec![Complex::zero(); len * 10];
    b.iter(|| {
        validate_and_zip(&buffer_in, &mut buffer_out, &mut [], len, 0, |chunk_in, chunk_out, _| {
            let (height, columns) = (base_len, &*reversed_columns);
            match unroll_factor {
                RadixFactor::Factor2 => unsafe { table_transpose::<V, 2>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor3 => unsafe { table_transpose::<V, 3>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor4 => unsafe { table_transpose::<V, 4>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor5 => unsafe { table_transpose::<V, 5>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor6 => unsafe { table_transpose::<V, 6>(height, columns, chunk_in, chunk_out) },
                RadixFactor::Factor7 => unsafe { table_transpose::<V, 7>(height, columns, chunk_in, chunk_out) },
                _ => unreachable!(),
            }
        })
        
    });
}


// Tests incorporating a factor of 31 by using it as a cross-FFT for RadixN
fn bench_unrolled_transpose<V: SimdVector>(b: &mut Bencher, factor: RadixFactor, power: u32, base_len: usize) {
    assert!(power == 1);

    let cross_len = factor.radix();
    let len : usize = base_len * cross_len;

    let buffer_in : Vec<Complex<V::ScalarType>> = vec![Complex::zero(); len * 10];
    let mut buffer_out : Vec<Complex<V::ScalarType>> = vec![Complex::zero(); len * 10];
    b.iter(|| {
        validate_and_zip(&buffer_in, &mut buffer_out, &mut [], len, 0, |chunk_in, chunk_out, _| {
            match factor {
                RadixFactor::Factor2 => unsafe { unrolled_transpose::<V, 2>(base_len, chunk_in, chunk_out) },
                RadixFactor::Factor3 => unsafe { unrolled_transpose::<V, 3>(base_len, chunk_in, chunk_out) },
                RadixFactor::Factor4 => unsafe { unrolled_transpose::<V, 4>(base_len, chunk_in, chunk_out) },
                RadixFactor::Factor5 => unsafe { unrolled_transpose::<V, 5>(base_len, chunk_in, chunk_out) },
                RadixFactor::Factor6 => unsafe { unrolled_transpose::<V, 6>(base_len, chunk_in, chunk_out) },
                RadixFactor::Factor7 => unsafe { unrolled_transpose::<V, 7>(base_len, chunk_in, chunk_out) },
                _ => unreachable!(),
            }
        })
        
    });
}



fn criterion_benchmark_transpose(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("neon64_transpose_power7_base7");

        for power7 in 1..9 {
            let len = 7usize.pow(power7) * 7;
            group.bench_with_input(BenchmarkId::new("original", len), &len, |b, _| {
                bench_original_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 7)
            });
            group.bench_with_input(BenchmarkId::new("simd_4x", len), &len, |b, _| {
                bench_simd_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 7)
            });
            if power7 == 1 {
                group.bench_with_input(BenchmarkId::new("unrolled", len), &len, |b, _| {
                    bench_unrolled_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 7)
                });
            }
        }
    }
    {
        let mut group = c.benchmark_group("neon64_transpose_power7_base10k");

        for power7 in 1..5 {
            let len = 7usize.pow(power7) * 10000;
            group.bench_with_input(BenchmarkId::new("original", len), &len, |b, _| {
                bench_original_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 10000)
            });
            group.bench_with_input(BenchmarkId::new("simd_4x", len), &len, |b, _| {
                bench_simd_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 10000)
            });
            if power7 == 1 {
                group.bench_with_input(BenchmarkId::new("unrolled", len), &len, |b, _| {
                    bench_unrolled_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 10000)
                });
            }
        }
    }
    {
        let mut group = c.benchmark_group("neon64_transpose_power7_base32");

        for power7 in 1..8 {
            let len = 7usize.pow(power7) * 32;
            group.bench_with_input(BenchmarkId::new("original", len), &len, |b, _| {
                bench_original_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 32)
            });
            group.bench_with_input(BenchmarkId::new("simd_4x", len), &len, |b, _| {
                bench_simd_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 32)
            });

            if power7 == 1 {
                group.bench_with_input(BenchmarkId::new("unrolled", len), &len, |b, _| {
                    bench_unrolled_transpose::<float64x2_t>(b, RadixFactor::Factor7, power7, 32)
                });
            }

        }
    }
    {
        let mut group = c.benchmark_group("neon64_transpose_power4_base32");

        for power4 in 1..12 {
            let len = 4usize.pow(power4) * 32;
            group.bench_with_input(BenchmarkId::new("original", len), &len, |b, _| {
                bench_original_transpose::<float64x2_t>(b, RadixFactor::Factor4, power4, 32)
            });
            group.bench_with_input(BenchmarkId::new("simd_4x", len), &len, |b, _| {
                bench_simd_transpose::<float64x2_t>(b, RadixFactor::Factor4, power4, 32)
            });
            if power4 == 1 {
                group.bench_with_input(BenchmarkId::new("unrolled", len), &len, |b, _| {
                    bench_unrolled_transpose::<float64x2_t>(b, RadixFactor::Factor4, power4, 32)
                });
            }
        }
    }
    {
        let mut group = c.benchmark_group("neon64_transpose_power4_base10k");

        for power4 in 1..12 {
            let len = 4usize.pow(power4) * 10000;
            group.bench_with_input(BenchmarkId::new("original", len), &len, |b, _| {
                bench_original_transpose::<float64x2_t>(b, RadixFactor::Factor4, power4, 10000)
            });
            group.bench_with_input(BenchmarkId::new("simd_4x", len), &len, |b, _| {
                bench_simd_transpose::<float64x2_t>(b, RadixFactor::Factor4, power4, 10000)
            });
            if power4 == 1 {
                group.bench_with_input(BenchmarkId::new("unrolled", len), &len, |b, _| {
                    bench_unrolled_transpose::<float64x2_t>(b, RadixFactor::Factor4, power4, 10000)
                });
            }
        }
    }
    /*
    {
        let mut group = c.benchmark_group("neon32_transpose_power7_base7");

        for power7 in 1..9 {
            let len = 7usize.pow(power7) * 7;
            group.bench_with_input(BenchmarkId::new("original", len), &len, |b, _| {
                bench_original_transpose::<float32x4_t>(b, RadixFactor::Factor7, power7, 7)
            });
            group.bench_with_input(BenchmarkId::new("simd_4x", len), &len, |b, _| {
                bench_simd_transpose::<float32x4_t>(b, RadixFactor::Factor7, power7, 7)
            });

            if power7 == 1 {
                group.bench_with_input(BenchmarkId::new("unrolled", len), &len, |b, _| {
                    bench_unrolled_transpose::<float32x4_t>(b, RadixFactor::Factor7, power7, 7)
                });
            }
        }
    }*/
    {
        let mut group = c.benchmark_group("neon32_transpose_power3_base9");

        for power3 in 1..12 {
            let len = 3usize.pow(power3) * 9;
            group.bench_with_input(BenchmarkId::new("original", len), &len, |b, _| {
                bench_original_transpose::<float32x4_t>(b, RadixFactor::Factor3, power3, 9)
            });
            group.bench_with_input(BenchmarkId::new("simd_4x", len), &len, |b, _| {
                bench_simd_transpose::<float32x4_t>(b, RadixFactor::Factor3, power3, 9)
            });

            if power3 == 1 {
                group.bench_with_input(BenchmarkId::new("unrolled", len), &len, |b, _| {
                    bench_unrolled_transpose::<float32x4_t>(b, RadixFactor::Factor3, power3, 9)
                });
            }
        }
    }
    {
        let mut group = c.benchmark_group("neon32_transpose_power4_base32");

        for power4 in 1..12 {
            let len = 4usize.pow(power4) * 32;
            group.bench_with_input(BenchmarkId::new("original", len), &len, |b, _| {
                bench_original_transpose::<float32x4_t>(b, RadixFactor::Factor4, power4, 32)
            });
            group.bench_with_input(BenchmarkId::new("simd_4x", len), &len, |b, _| {
                bench_simd_transpose::<float32x4_t>(b, RadixFactor::Factor4, power4, 32)
            });
            if power4 == 1 {
                group.bench_with_input(BenchmarkId::new("unrolled", len), &len, |b, _| {
                    bench_unrolled_transpose::<float32x4_t>(b, RadixFactor::Factor4, power4, 32)
                });
            }
        }
    }
}



/*
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
}*/

criterion_group! {
    name = benches;
    config = config::fast();
    targets = criterion_benchmark_transpose
}
criterion_main!(benches);
