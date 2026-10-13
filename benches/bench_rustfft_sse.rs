use rustfft::simd::simd_array::{SimdComplexArray, SimdComplexArrayMut};
use rustfft::num_complex::Complex;
use rustfft::simd::simd_vector::SimdVector;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::hint::black_box;
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
fn sse_planned32_prime_65539(b: &mut Bencher)    { bench_planned_f32(b, 65539); }
fn sse_planned32_prime_746497(b: &mut Bencher)   { bench_planned_f32(b,746497); }

fn sse_planned64_prime_0149(b: &mut Bencher)     { bench_planned_f64(b,  149); }
fn sse_planned64_prime_0151(b: &mut Bencher)     { bench_planned_f64(b,  151); }
fn sse_planned64_prime_0251(b: &mut Bencher)     { bench_planned_f64(b,  251); }
fn sse_planned64_prime_0257(b: &mut Bencher)     { bench_planned_f64(b,  257); }
fn sse_planned64_prime_2017(b: &mut Bencher)     { bench_planned_f64(b,  2017); }
fn sse_planned64_prime_2879(b: &mut Bencher)     { bench_planned_f64(b,  2879); }
fn sse_planned64_prime_65521(b: &mut Bencher)    { bench_planned_f64(b, 65521); }
fn sse_planned64_prime_65539(b: &mut Bencher)    { bench_planned_f64(b, 65539); }
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
        sse_planned32_prime_65539,
        sse_planned32_prime_746497,
        sse_planned64_prime_0149,
        sse_planned64_prime_0151,
        sse_planned64_prime_0251,
        sse_planned64_prime_0257,
        sse_planned64_prime_2017,
        sse_planned64_prime_2879,
        sse_planned64_prime_65521,
        sse_planned64_prime_65539,
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

use std::arch::x86_64::*;


#[target_feature(enable = "sse4.1")]
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

        let src0: __m128 = src.load(vcol);
        let src1: __m128 = src.load(vcol + 2);

        let multiplier_deinterleaved_re: __m128 = multiplier.load(vcol);
        let multiplier_deinterleaved_im: __m128 = multiplier.load(vcol + 2);

        let src_mix_0 = _mm_unpacklo_ps(src0, src1);
        let src_mix_1 = _mm_unpackhi_ps(src0, src1);

        let src_deinterleaved_re = _mm_unpacklo_ps(src_mix_0, src_mix_1);
        let src_deinterleaved_im = _mm_unpackhi_ps(src_mix_0, src_mix_1);
        let dest_deinterleaved_re = _mm_sub_ps(_mm_mul_ps(src_deinterleaved_re, multiplier_deinterleaved_re), _mm_mul_ps(src_deinterleaved_im, multiplier_deinterleaved_im));
        let dest_deinterleaved_im = _mm_add_ps(_mm_mul_ps(src_deinterleaved_re, multiplier_deinterleaved_im), _mm_mul_ps(src_deinterleaved_im, multiplier_deinterleaved_re));

        let dst0 = _mm_unpacklo_ps(dest_deinterleaved_re, dest_deinterleaved_im);
        let dst1 = _mm_unpackhi_ps(dest_deinterleaved_re, dest_deinterleaved_im);

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

#[target_feature(enable = "sse4.1")]
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

        let product_a = mul_one::<__m128, false>(left_a, multiplier.load(index));
        let product_b = mul_one::<__m128, false>(left_b, multiplier.load(next));

        dest.store(product_a, index);
        dest.store(product_b, next);
    }

    // an odd vector count leaves one behind
    if full_vectors % 2 == 1 {
        let index = (full_vectors - 1) * per_vector;
        let left = src.load(index);
        let product = mul_one::<__m128, false>(left, multiplier.load(index));
        dest.store(product, index);
    }

    // The leftover element, if there is one. `COMPLEX_PER_VECTOR` is a constant, so this whole
    // branch vanishes on the single-complex vectors, whose partial load and store are
    // unimplemented.
    if per_vector > 1 && count % per_vector != 0 {
        let index = full_vectors * per_vector;
        let left = src.load1_lo(index);
        let product = mul_one::<__m128, false>(left, multiplier.load1_lo(index));
        dest.store1_lo(product, index);
    }
}


#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

        let src0: __m128d = src.load(vcol);
        let src1: __m128d = src.load(vcol + 2);

        let multiplier_deinterleaved_re: __m128d = multiplier.load(vcol);
        let multiplier_deinterleaved_im: __m128d = multiplier.load(vcol + 2);

        let src_deinterleaved_re = _mm_unpacklo_pd(src0, src1);
        let src_deinterleaved_im = _mm_unpackhi_pd(src0, src1);
        let dest_deinterleaved_re = _mm_sub_pd(_mm_mul_pd(src_deinterleaved_re, multiplier_deinterleaved_re), _mm_mul_pd(src_deinterleaved_im, multiplier_deinterleaved_im));
        let dest_deinterleaved_im = _mm_add_pd(_mm_mul_pd(src_deinterleaved_re, multiplier_deinterleaved_im), _mm_mul_pd(src_deinterleaved_im, multiplier_deinterleaved_re));

        let dst0 = _mm_unpacklo_pd(dest_deinterleaved_re, dest_deinterleaved_im);
        let dst1 = _mm_unpackhi_pd(dest_deinterleaved_re, dest_deinterleaved_im);

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

#[target_feature(enable = "sse4.1")]
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

        let product_a = mul_one::<__m128d, false>(left_a, multiplier.load(index));
        let product_b = mul_one::<__m128d, false>(left_b, multiplier.load(next));

        dest.store(product_a, index);
        dest.store(product_b, next);
    }

    // an odd vector count leaves one behind
    if full_vectors % 2 == 1 {
        let index = (full_vectors - 1) * per_vector;
        let left = src.load(index);
        let product = mul_one::<__m128d, false>(left, multiplier.load(index));
        dest.store(product, index);
    }

    // The leftover element, if there is one. `COMPLEX_PER_VECTOR` is a constant, so this whole
    // branch vanishes on the single-complex vectors, whose partial load and store are
    // unimplemented.
    if per_vector > 1 && count % per_vector != 0 {
        let index = full_vectors * per_vector;
        let left = src.load1_lo(index);
        let product = mul_one::<__m128d, false>(left, multiplier.load1_lo(index));
        dest.store1_lo(product, index);
    }
}

#[target_feature(enable = "sse4.1")]
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
