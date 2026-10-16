use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use rustfft::FftNum;
use rustfft::RadixFactor;
use rustfft::FftDirection;
use rustfft::neon::neon_butterflies::*;
use rustfft::neon::neon_prime_butterflies::*;
use rustfft::neon::neon_radixn::NeonRadixN;
use rustfft::algorithm::{MixedRadix, MixedRadixSmall};
use std::sync::Arc;
use std::any::TypeId;
use core::arch::aarch64::*;
mod config;

use criterion::{BenchmarkId, criterion_group, criterion_main, Bencher, Criterion};


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



fn plan_butterfly_fft<T: FftNum>(len: usize) -> Arc<dyn Fft<T>> {
    let id_f32 = TypeId::of::<f32>();
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();

    if id_t == id_f32 {
        unsafe { 
            match len {
                2 => Arc::new(NeonF32Butterfly2::new(FftDirection::Forward)),
                3 => Arc::new(NeonF32Butterfly3::new(FftDirection::Forward)),
                4 => Arc::new(NeonF32Butterfly4::new(FftDirection::Forward)),
                5 => Arc::new(NeonF32Butterfly5::new(FftDirection::Forward)),
                6 => Arc::new(NeonF32Butterfly6::new(FftDirection::Forward)),
                7 => Arc::new(NeonF32Butterfly7::new(FftDirection::Forward)),
                8 => Arc::new(NeonF32Butterfly8::new(FftDirection::Forward)),
                11 => Arc::new(NeonF32Butterfly11::new(FftDirection::Forward)),
                16 => Arc::new(NeonF32Butterfly16::new(FftDirection::Forward)),
                31 => Arc::new(NeonF32Butterfly31::new(FftDirection::Forward)),
                32 => Arc::new(NeonF32Butterfly32::new(FftDirection::Forward)),
                _ => panic!("Invalid butterfly size: {}", len),
            }
        }
    } else if id_t == id_f64 {
        unsafe {
            match len {
                2 => Arc::new(NeonF64Butterfly2::new(FftDirection::Forward)),
                3 => Arc::new(NeonF64Butterfly3::new(FftDirection::Forward)),
                4 => Arc::new(NeonF64Butterfly4::new(FftDirection::Forward)),
                5 => Arc::new(NeonF64Butterfly5::new(FftDirection::Forward)),
                6 => Arc::new(NeonF64Butterfly6::new(FftDirection::Forward)),
                7 => Arc::new(NeonF64Butterfly7::new(FftDirection::Forward)),
                8 => Arc::new(NeonF64Butterfly8::new(FftDirection::Forward)),
                11 => Arc::new(NeonF64Butterfly11::new(FftDirection::Forward)),
                16 => Arc::new(NeonF64Butterfly16::new(FftDirection::Forward)),
                31 => Arc::new(NeonF64Butterfly31::new(FftDirection::Forward)),
                32 => Arc::new(NeonF64Butterfly32::new(FftDirection::Forward)),
                _ => panic!("Invalid butterfly size: {}", len),
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

fn plan_mixedradix_power31<T: FftNum>(power31: u32) -> Arc<dyn Fft<T>> {
    fn construct_branch<T: FftNum>(power: u32, butterfly31: &Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
        if power == 1 {
            Arc::clone(butterfly31)
        } else {
            let left = power / 2;
            let right = power - left;
            let fft_left = construct_branch(left, butterfly31);
            let fft_right = construct_branch(right, butterfly31);

            if fft_left.len() == 31 && fft_right.len() == 31 {
                Arc::new(MixedRadixSmall::new(fft_left, fft_right))
            } else {
                Arc::new(MixedRadix::new(fft_left, fft_right))
            }
        }
    }
    
    let butterfly31 = plan_butterfly_fft(31);
    construct_branch(power31, &butterfly31)
}

// Tests incorporating a factor of 31 by using it as a cross-FFT for RadixN
fn bench_factor31_radixncross<T: FftNum>(b: &mut Bencher, power31: u32, power2: u32) {
    let base_power2 = match power2 {
        0 => unimplemented!(),
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4, 
        5 => 5,
        other =>  if other % 2 == 0 { 4 } else { 5 }
    };

    let power2_cross = power2 - base_power2;
    assert!(power2_cross % 2 == 0);
    let power4_cross = power2_cross / 2;

    let base_fft = plan_butterfly_fft::<T>(1 << base_power2);
    
    let mut factors : Vec<RadixFactor> = Vec::new();
    for _ in 0..power31 {
        factors.push(RadixFactor::Factor31);
    }
    for _ in 0..power4_cross {
        factors.push(RadixFactor::Factor4);
    }

    let fft = construct_radixn(&factors, base_fft);
    let len = fft.len();

    let mut buffer_in = vec![Complex::zero(); len * 10];
    let mut buffer_out = vec![Complex::zero(); len * 10];
    let mut buffer_scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer_in, &mut buffer_out, &mut buffer_scratch);
    });
}
// Tests incorporating a factor of 31 by using it as the base of RadixN
fn bench_factor31_radixnbase<T: FftNum>(b: &mut Bencher, power31: u32, power2: u32) {
    let base_fft = plan_mixedradix_power31::<T>(power31);
    
    let mut factors : Vec<RadixFactor> = Vec::new();
    if power2 % 2 == 1 {
        factors.push(RadixFactor::Factor2);
    }
    let power4_cross = power2 / 2;
    for _ in 0..power4_cross {
        factors.push(RadixFactor::Factor4);
    }

    let fft = construct_radixn(&factors, base_fft);
    let len = fft.len();

    let mut buffer_in = vec![Complex::zero(); len * 10];
    let mut buffer_out = vec![Complex::zero(); len * 10];
    let mut buffer_scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer_in, &mut buffer_out, &mut buffer_scratch);
    });
}

fn criterion_benchmark_factor31(c: &mut Criterion) {
    {
        let mut group32 = c.benchmark_group("factor31_neon_f32");
        
        for power31 in 1..3 {
            for power2 in 1..10 {
                let len = 31usize.pow(power31)*2usize.pow(power2);
                group32.bench_with_input(BenchmarkId::new("radixnbase", len), &len,  |b, _| {
                    bench_factor31_radixnbase::<f32>(b, power31, power2)
                });
                group32.bench_with_input(BenchmarkId::new("radixncross", len), &len,  |b, _| {
                    bench_factor31_radixncross::<f32>(b, power31, power2)
                });
            }
        }
    }

    {
        let mut group64 = c.benchmark_group("factor31_neon_f64");

        for power31 in 1..3 {
            for power2 in 1..10 {
                let len = 31usize.pow(power31)*2usize.pow(power2);
                group64.bench_with_input(BenchmarkId::new("radixnbase", len), &len, |b, _| {
                    bench_factor31_radixnbase::<f64>(b, power31, power2)
                });
                group64.bench_with_input(BenchmarkId::new("radixncross", len), &len, |b, _| {
                    bench_factor31_radixncross::<f64>(b, power31, power2)
                });
            }
        }
    }
}


fn plan_mixedradix_power11<T: FftNum>(power11: u32) -> Arc<dyn Fft<T>> {
    fn construct_branch<T: FftNum>(power: u32, butterfly: &Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
        if power == 1 {
            Arc::clone(butterfly)
        } else {
            let left = power / 2;
            let right = power - left;
            let fft_left = construct_branch(left, butterfly);
            let fft_right = construct_branch(right, butterfly);

            if left == 1 && right == 1 {
                Arc::new(MixedRadixSmall::new(fft_left, fft_right))
            } else {
                Arc::new(MixedRadix::new(fft_left, fft_right))
            }
        }
    }
    
    let butterfly = plan_butterfly_fft(11);
    construct_branch(power11, &butterfly)
}

// Tests incorporating a factor of 11 by using it as a cross-FFT for RadixN
fn bench_factor11_radixncross<T: FftNum>(b: &mut Bencher, power11: u32, power2: u32) {
    let base_power2 = match power2 {
        0 => unimplemented!(),
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4, 
        5 => 5,
        other =>  if other % 2 == 0 { 4 } else { 5 }
    };

    let power2_cross = power2 - base_power2;
    assert!(power2_cross % 2 == 0);
    let power4_cross = power2_cross / 2;

    let base_fft = plan_butterfly_fft::<T>(1 << base_power2);
    
    let mut factors : Vec<RadixFactor> = Vec::new();
    for _ in 0..power11 {
        factors.push(RadixFactor::Factor11);
    }
    for _ in 0..power4_cross {
        factors.push(RadixFactor::Factor4);
    }

    let fft = construct_radixn(&factors, base_fft);
    let len = fft.len();

    let mut buffer_in = vec![Complex::zero(); len * 10];
    let mut buffer_out = vec![Complex::zero(); len * 10];
    let mut buffer_scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer_in, &mut buffer_out, &mut buffer_scratch);
    });
}
// Tests incorporating a factor of 31 by using it as the base of RadixN
fn bench_factor11_radixnbase<T: FftNum>(b: &mut Bencher, power11: u32, power2: u32) {
    let base_fft = plan_mixedradix_power11::<T>(power11);
    
    let mut factors : Vec<RadixFactor> = Vec::new();
    if power2 % 2 == 1 {
        factors.push(RadixFactor::Factor2);
    }
    let power4_cross = power2 / 2;
    for _ in 0..power4_cross {
        factors.push(RadixFactor::Factor4);
    }

    let fft = construct_radixn(&factors, base_fft);
    let len = fft.len();

    let mut buffer_in = vec![Complex::zero(); len * 10];
    let mut buffer_out = vec![Complex::zero(); len * 10];
    let mut buffer_scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer_in, &mut buffer_out, &mut buffer_scratch);
    });
}

fn criterion_benchmark_factor11(c: &mut Criterion) {
    {
        let mut group32 = c.benchmark_group("factor11_neon_f32");
        
        for power11 in 1..3 {
            for power2 in 1..15 {
                let len = 11usize.pow(power11)*2usize.pow(power2);
                group32.bench_with_input(BenchmarkId::new("radixnbase", len), &len,  |b, _| {
                    bench_factor11_radixnbase::<f32>(b, power11, power2)
                });
                group32.bench_with_input(BenchmarkId::new("radixncross", len), &len,  |b, _| {
                    bench_factor11_radixncross::<f32>(b, power11, power2)
                });
            }
        }
    }

    {
        let mut group64 = c.benchmark_group("factor11_neon_f64");

        for power11 in 1..3 {
            for power2 in 1..15 {
                let len = 11usize.pow(power11)*2usize.pow(power2);
                group64.bench_with_input(BenchmarkId::new("radixnbase", len), &len, |b, _| {
                    bench_factor11_radixnbase::<f64>(b, power11, power2)
                });
                group64.bench_with_input(BenchmarkId::new("radixncross", len), &len, |b, _| {
                    bench_factor11_radixncross::<f64>(b, power11, power2)
                });
            }
        }
    }
}

fn plan_mixedradix_power7<T: FftNum>(power7: u32) -> Arc<dyn Fft<T>> {
    fn construct_branch<T: FftNum>(power: u32, butterfly: &Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
        if power == 1 {
            Arc::clone(butterfly)
        } else {
            let left = power / 2;
            let right = power - left;
            let fft_left = construct_branch(left, butterfly);
            let fft_right = construct_branch(right, butterfly);

            if left == 1 && right == 1 {
                Arc::new(MixedRadixSmall::new(fft_left, fft_right))
            } else {
                Arc::new(MixedRadix::new(fft_left, fft_right))
            }
        }
    }
    
    let butterfly = plan_butterfly_fft(7);
    construct_branch(power7, &butterfly)
}

// Tests incorporating a factor of 7 by using it as a cross-FFT for RadixN
fn bench_factor7_radixncross<T: FftNum>(b: &mut Bencher, power7: u32, power2: u32) {
    let base_power2 = match power2 {
        0 => unimplemented!(),
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4, 
        5 => 5,
        other =>  if other % 2 == 0 { 4 } else { 5 }
    };

    let power2_cross = power2 - base_power2;
    assert!(power2_cross % 2 == 0);
    let power4_cross = power2_cross / 2;

    let base_fft = plan_butterfly_fft::<T>(1 << base_power2);
    
    let mut factors : Vec<RadixFactor> = Vec::new();
    for _ in 0..power7 {
        factors.push(RadixFactor::Factor7);
    }
    for _ in 0..power4_cross {
        factors.push(RadixFactor::Factor4);
    }

    let fft = construct_radixn(&factors, base_fft);
    let len = fft.len();

    let mut buffer_in = vec![Complex::zero(); len * 10];
    let mut buffer_out = vec![Complex::zero(); len * 10];
    let mut buffer_scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer_in, &mut buffer_out, &mut buffer_scratch);
    });
}
// Tests incorporating a factor of 31 by using it as the base of RadixN
fn bench_factor7_radixnbase<T: FftNum>(b: &mut Bencher, power7: u32, power2: u32) {
    let base_fft = plan_mixedradix_power7::<T>(power7);
    
    let mut factors : Vec<RadixFactor> = Vec::new();
    if power2 % 2 == 1 {
        factors.push(RadixFactor::Factor2);
    }
    let power4_cross = power2 / 2;
    for _ in 0..power4_cross {
        factors.push(RadixFactor::Factor4);
    }

    let fft = construct_radixn(&factors, base_fft);
    let len = fft.len();

    let mut buffer_in = vec![Complex::zero(); len * 10];
    let mut buffer_out = vec![Complex::zero(); len * 10];
    let mut buffer_scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer_in, &mut buffer_out, &mut buffer_scratch);
    });
}

fn criterion_benchmark_factor7(c: &mut Criterion) {
    {
        let mut group32 = c.benchmark_group("factor7_neon_f32");
        
        for power7 in 1..3 {
            for power2 in 1..16 {
                let len = 7usize.pow(power7)*2usize.pow(power2);
                group32.bench_with_input(BenchmarkId::new("radixnbase", len), &len,  |b, _| {
                    bench_factor7_radixnbase::<f32>(b, power7, power2)
                });
                group32.bench_with_input(BenchmarkId::new("radixncross", len), &len,  |b, _| {
                    bench_factor7_radixncross::<f32>(b, power7, power2)
                });
            }
        }
    }

    {
        let mut group64 = c.benchmark_group("factor7_neon_f64");

        for power7 in 1..3 {
            for power2 in 1..16 {
                let len = 7usize.pow(power7)*2usize.pow(power2);
                group64.bench_with_input(BenchmarkId::new("radixnbase", len), &len, |b, _| {
                    bench_factor7_radixnbase::<f64>(b, power7, power2)
                });
                group64.bench_with_input(BenchmarkId::new("radixncross", len), &len, |b, _| {
                    bench_factor7_radixncross::<f64>(b, power7, power2)
                });
            }
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
    targets = criterion_benchmark, criterion_benchmark_factor31, criterion_benchmark_factor11, criterion_benchmark_factor7
}
criterion_main!(benches);
