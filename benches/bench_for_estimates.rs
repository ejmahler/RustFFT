#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate rustfft;
extern crate test;

use rustfft::algorithm::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftDirection};
use std::sync::Arc;
use test::Bencher;
//use rustfft::algorithm::butterflies::*;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
//fn bench_planned_f64(b: &mut Bencher, len: usize) {
//
//    let mut planner = rustfft::FftPlannerScalar::new();
//    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);
//
//    let mut buffer = vec![Complex::zero(); len];
//    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
//    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
//}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_multi_f64(b: &mut Bencher, len: usize, repeats: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer = vec![Complex::zero(); 1000 * repeats];
    let mut output = vec![Complex::zero(); 1000 * repeats];
    let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
    });
}

// all butterflies
#[bench]
fn estimates_10_butterfly_________00000002_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 2); }
#[bench]
fn estimates_10_butterfly_________00000003_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 3); }
#[bench]
fn estimates_10_butterfly_________00000004_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 4); }
#[bench]
fn estimates_10_butterfly_________00000005_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 5); }
#[bench]
fn estimates_10_butterfly_________00000006_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 6); }
#[bench]
fn estimates_10_butterfly_________00000007_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 7); }
#[bench]
fn estimates_10_butterfly_________00000008_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 8); }
#[bench]
fn estimates_10_butterfly_________00000011_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 11); }
#[bench]
fn estimates_10_butterfly_________00000013_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 13); }
#[bench]
fn estimates_10_butterfly_________00000016_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 16); }
#[bench]
fn estimates_10_butterfly_________00000017_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 17); }
#[bench]
fn estimates_10_butterfly_________00000019_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 19); }
#[bench]
fn estimates_10_butterfly_________00000023_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 23); }
#[bench]
fn estimates_10_butterfly_________00000029_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 29); }
#[bench]
fn estimates_10_butterfly_________00000031_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 31); }
#[bench]
fn estimates_10_butterfly_________00000032_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 32); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Good-Thomas algorithm
fn bench_good_thomas(b: &mut Bencher, width: usize, height: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let width_fft = planner.plan_fft_forward(width);
    let height_fft = planner.plan_fft_forward(height);

    let fft: Arc<Fft<f64>> = Arc::new(GoodThomasAlgorithm::new(width_fft, height_fft));
    let mut signal = vec![Complex { re: 0_f64, im: 0_f64 }; width * height ];
    let mut spectrum = signal.clone();
    let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch);
    });
}
#[bench]
fn estimates_21_good_thomas_______00000002_0003(b: &mut Bencher) { bench_good_thomas(b, 2, 3); }
#[bench]
fn estimates_21_good_thomas_______00000003_0004(b: &mut Bencher) { bench_good_thomas(b, 3, 4); }
#[bench]
fn estimates_21_good_thomas_______00000004_0005(b: &mut Bencher) { bench_good_thomas(b, 4, 5); }
#[bench]
fn estimates_21_good_thomas_______00000007_0032(b: &mut Bencher) { bench_good_thomas(b, 7, 32); }
#[bench]
fn estimates_21_good_thomas_______00000011_0017(b: &mut Bencher) { bench_good_thomas(b, 11, 17); }
#[bench]
fn estimates_21_good_thomas_______00000017_0031(b: &mut Bencher) { bench_good_thomas(b, 17, 31); }
#[bench]
fn estimates_21_good_thomas_______00000029_0031(b: &mut Bencher) { bench_good_thomas(b, 29, 31); }
#[bench]
fn estimates_21_good_thomas_______00000128_0256(b: &mut Bencher) { bench_good_thomas(b, 128, 256); }
#[bench]
fn estimates_21_good_thomas_______00000128_0181(b: &mut Bencher) { bench_good_thomas(b, 128, 181); }
#[bench]
fn estimates_21_good_thomas_______00000181_0191(b: &mut Bencher) { bench_good_thomas(b, 181, 191); }
#[bench]
fn estimates_21_good_thomas_______00000256_0256(b: &mut Bencher) { bench_good_thomas(b, 256, 256); }
#[bench]
fn estimates_21_good_thomas_______00000512_0512(b: &mut Bencher) { bench_good_thomas(b, 512, 512); }
#[bench]
fn estimates_21_good_thomas_______00001024_1024(b: &mut Bencher) { bench_good_thomas(b, 1024, 1024); }
#[bench]
fn estimates_21_good_thomas_______00002048_2048(b: &mut Bencher) { bench_good_thomas(b, 2048, 2048); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix algorithm
fn bench_mixed_radix(b: &mut Bencher, width: usize, height: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let width_fft = planner.plan_fft_forward(width);
    let height_fft = planner.plan_fft_forward(height);

    let fft: Arc<Fft<_>> = Arc::new(MixedRadix::new(width_fft, height_fft));

    let mut signal = vec![Complex { re: 0_f64, im: 0_f64 }; width * height ];
    let mut spectrum = signal.clone();
    let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch);
    });
}
#[bench]
fn estimates_22_mixed_radix_______00000002_0003(b: &mut Bencher) { bench_mixed_radix(b, 2, 3); }
#[bench]
fn estimates_22_mixed_radix_______00000003_0004(b: &mut Bencher) { bench_mixed_radix(b, 3, 4); }
#[bench]
fn estimates_22_mixed_radix_______00000004_0005(b: &mut Bencher) { bench_mixed_radix(b, 4, 5); }
#[bench]
fn estimates_22_mixed_radix_______00000007_0032(b: &mut Bencher) { bench_mixed_radix(b, 7, 32); }
#[bench]
fn estimates_22_mixed_radix_______00000011_0017(b: &mut Bencher) { bench_mixed_radix(b, 11, 17); }
#[bench]
fn estimates_22_mixed_radix_______00000017_0031(b: &mut Bencher) { bench_mixed_radix(b, 17, 31); }
#[bench]
fn estimates_22_mixed_radix_______00000029_0031(b: &mut Bencher) { bench_mixed_radix(b, 29, 31); }
#[bench]
fn estimates_22_mixed_radix_______00000128_0256(b: &mut Bencher) { bench_mixed_radix(b, 128, 256); }
#[bench]
fn estimates_22_mixed_radix_______00000128_0181(b: &mut Bencher) { bench_mixed_radix(b, 128, 181); }
#[bench]
fn estimates_22_mixed_radix_______00000181_0191(b: &mut Bencher) { bench_mixed_radix(b, 181, 191); }
#[bench]
fn estimates_22_mixed_radix_______00000256_0256(b: &mut Bencher) { bench_mixed_radix(b, 256, 256); }
#[bench]
fn estimates_22_mixed_radix_______00000512_0512(b: &mut Bencher) { bench_mixed_radix(b, 512, 512); }
#[bench]
fn estimates_22_mixed_radix_______00001024_1024(b: &mut Bencher) { bench_mixed_radix(b, 1024, 1024); }
#[bench]
fn estimates_22_mixed_radix_______00002048_2048(b: &mut Bencher) { bench_mixed_radix(b, 2048, 2048); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the MixedRadixSmall algorithm
fn bench_mixed_radix_small(b: &mut Bencher, width: usize, height: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let width_fft = planner.plan_fft_forward(width);
    let height_fft = planner.plan_fft_forward(height);

    let fft: Arc<Fft<_>> = Arc::new(MixedRadixSmall::new(width_fft, height_fft));

    let mut signal = vec![Complex { re: 0_f64, im: 0_f64 }; width * height ];
    let mut spectrum = signal.clone();
    let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch);
    });
}
#[bench]
fn estimates_23_mixed_radix_small_00000002_0003(b: &mut Bencher) { bench_mixed_radix_small(b, 2, 3); }
#[bench]
fn estimates_23_mixed_radix_small_00000003_0004(b: &mut Bencher) { bench_mixed_radix_small(b, 3, 4); }
#[bench]
fn estimates_23_mixed_radix_small_00000004_0005(b: &mut Bencher) { bench_mixed_radix_small(b, 4, 5); }
#[bench]
fn estimates_23_mixed_radix_small_00000007_0032(b: &mut Bencher) { bench_mixed_radix_small(b, 7, 32); }
#[bench]
fn estimates_23_mixed_radix_small_00000011_0017(b: &mut Bencher) { bench_mixed_radix_small(b, 11, 17); }
#[bench]
fn estimates_23_mixed_radix_small_00000017_0031(b: &mut Bencher) { bench_mixed_radix_small(b, 17, 31); }
#[bench]
fn estimates_23_mixed_radix_small_00000029_0031(b: &mut Bencher) { bench_mixed_radix_small(b, 29, 31); }
#[bench]
fn estimates_23_mixed_radix_small_00000128_0256(b: &mut Bencher) { bench_mixed_radix_small(b, 128, 256); }
#[bench]
fn estimates_23_mixed_radix_small_00000128_0181(b: &mut Bencher) { bench_mixed_radix_small(b, 128, 181); }
#[bench]
fn estimates_23_mixed_radix_small_00000181_0191(b: &mut Bencher) { bench_mixed_radix_small(b, 181, 191); }
#[bench]
fn estimates_23_mixed_radix_small_00000256_0256(b: &mut Bencher) { bench_mixed_radix_small(b, 256, 256); }
#[bench]
fn estimates_23_mixed_radix_small_00000512_0512(b: &mut Bencher) { bench_mixed_radix_small(b, 512, 512); }
#[bench]
fn estimates_23_mixed_radix_small_00001024_1024(b: &mut Bencher) { bench_mixed_radix_small(b, 1024, 1024); }
#[bench]
fn estimates_23_mixed_radix_small_00002048_2048(b: &mut Bencher) { bench_mixed_radix_small(b, 2048, 2048); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix Double Butterfly algorithm
fn bench_good_thomas_small(b: &mut Bencher, width: usize, height: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let width_fft = planner.plan_fft_forward(width);
    let height_fft = planner.plan_fft_forward(height);

    let fft: Arc<Fft<_>> = Arc::new(GoodThomasAlgorithmSmall::new(width_fft, height_fft));

    let mut signal = vec![Complex { re: 0_f64, im: 0_f64 }; width * height ];
    let mut spectrum = signal.clone();
    let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch);
    });
}
#[bench]
fn estimates_24_good_thomas_small_00000002_0003(b: &mut Bencher) { bench_good_thomas_small(b, 2, 3); }
#[bench]
fn estimates_24_good_thomas_small_00000003_0004(b: &mut Bencher) { bench_good_thomas_small(b, 3, 4); }
#[bench]
fn estimates_24_good_thomas_small_00000004_0005(b: &mut Bencher) { bench_good_thomas_small(b, 4, 5); }
#[bench]
fn estimates_24_good_thomas_small_00000007_0032(b: &mut Bencher) { bench_good_thomas_small(b, 7, 32); }
#[bench]
fn estimates_24_good_thomas_small_00000011_0017(b: &mut Bencher) { bench_good_thomas_small(b, 11, 17); }
#[bench]
fn estimates_24_good_thomas_small_00000017_0031(b: &mut Bencher) { bench_good_thomas_small(b, 17, 31); }
#[bench]
fn estimates_24_good_thomas_small_00000029_0031(b: &mut Bencher) { bench_good_thomas_small(b, 29, 31); }
#[bench]
fn estimates_24_good_thomas_small_00000128_0256(b: &mut Bencher) { bench_good_thomas_small(b, 128, 256); }
#[bench]
fn estimates_24_good_thomas_small_00000128_0181(b: &mut Bencher) { bench_good_thomas_small(b, 128, 181); }
#[bench]
fn estimates_24_good_thomas_small_00000181_0191(b: &mut Bencher) { bench_good_thomas_small(b, 181, 191); }
#[bench]
fn estimates_24_good_thomas_small_00000256_0256(b: &mut Bencher) { bench_good_thomas_small(b, 256, 256); }
#[bench]
fn estimates_24_good_thomas_small_00000512_0512(b: &mut Bencher) { bench_good_thomas_small(b, 512, 512); }
#[bench]
fn estimates_24_good_thomas_small_00001024_1024(b: &mut Bencher) { bench_good_thomas_small(b, 1024, 1024); }
#[bench]
fn estimates_24_good_thomas_small_00002048_2048(b: &mut Bencher) { bench_good_thomas_small(b, 2048, 2048); }

#[bench]
fn estimates_20_mixed_radix_inner_00000002_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 2); }
#[bench]
fn estimates_20_mixed_radix_inner_00000003_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 3); }
#[bench]
fn estimates_20_mixed_radix_inner_00000004_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 4); }
#[bench]
fn estimates_20_mixed_radix_inner_00000005_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 5); }
#[bench]
fn estimates_20_mixed_radix_inner_00000007_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 7); }
#[bench]
fn estimates_20_mixed_radix_inner_00000011_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 11); }
#[bench]
fn estimates_20_mixed_radix_inner_00000017_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 17); }
#[bench]
fn estimates_20_mixed_radix_inner_00000029_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 29); }
#[bench]
fn estimates_20_mixed_radix_inner_00000031_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 31); }
#[bench]
fn estimates_20_mixed_radix_inner_00000032_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 32); }
#[bench]
fn estimates_20_mixed_radix_inner_00000128_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 128); }
#[bench]
fn estimates_20_mixed_radix_inner_00000181_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 181); }
#[bench]
fn estimates_20_mixed_radix_inner_00000191_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 191); }
#[bench]
fn estimates_20_mixed_radix_inner_00000256_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 256); }
#[bench]
fn estimates_20_mixed_radix_inner_00000512_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 512); }
#[bench]
fn estimates_20_mixed_radix_inner_00001024_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 1024); }
#[bench]
fn estimates_20_mixed_radix_inner_00002048_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 2048); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_raders_scalar(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let inner_fft = planner.plan_fft_forward(len - 1);

    let fft: Arc<Fft<_>> = Arc::new(RadersAlgorithm::new(inner_fft));

    let mut signal = vec![Complex { re: 0_f64, im: 0_f64 }; len ];
    let mut spectrum = signal.clone();
    let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch);
    });
}

fn bench_raders_inner(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft = planner.plan_fft_forward(len - 1);

    let mut signal = vec![Complex { re: 0_f64, im: 0_f64 }; len-1 ];
    let mut spectrum = signal.clone();
    let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch);
    });
}

#[bench]
fn estimates_30_raders____________00000005_0000(b: &mut Bencher) { bench_raders_scalar(b, 5); }
#[bench]
fn estimates_30_raders_inner______00000005_0000(b: &mut Bencher) { bench_raders_inner(b, 5); }
#[bench]
fn estimates_30_raders____________00000017_0000(b: &mut Bencher) { bench_raders_scalar(b, 17); }
#[bench]
fn estimates_30_raders_inner______00000017_0000(b: &mut Bencher) { bench_raders_inner(b, 17); }
#[bench]
fn estimates_30_raders____________00000149_0000(b: &mut Bencher) { bench_raders_scalar(b, 149); }
#[bench]
fn estimates_30_raders_inner______00000149_0000(b: &mut Bencher) { bench_raders_inner(b, 149); }
#[bench]
fn estimates_30_raders____________00000151_0000(b: &mut Bencher) { bench_raders_scalar(b, 151); }
#[bench]
fn estimates_30_raders_inner______00000151_0000(b: &mut Bencher) { bench_raders_inner(b, 151); }
#[bench]
fn estimates_30_raders____________00000251_0000(b: &mut Bencher) { bench_raders_scalar(b, 251); }
#[bench]
fn estimates_30_raders_inner______00000251_0000(b: &mut Bencher) { bench_raders_inner(b, 251); }
#[bench]
fn estimates_30_raders____________00000257_0000(b: &mut Bencher) { bench_raders_scalar(b, 257); }
#[bench]
fn estimates_30_raders_inner______00000257_0000(b: &mut Bencher) { bench_raders_inner(b, 257); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Bluestein's Algorithm
fn bench_bluesteins_scalar_prime(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let inner_fft = planner.plan_fft_forward((len * 2 - 1).checked_next_power_of_two().unwrap());
    let fft: Arc<Fft<f64>> = Arc::new(BluesteinsAlgorithm::new(len, inner_fft));

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

fn bench_bluesteins_scalar_inner(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let inner_len = (len * 2 - 1).checked_next_power_of_two().unwrap();
    let fft = planner.plan_fft_forward(inner_len);

    let mut buffer = vec![Complex { re: 0_f64, im: 0_f64 }; inner_len ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn estimates_31_bluesteins_prime__00000005_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 5); }
#[bench]
fn estimates_31_bluesteins_inner__00000005_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b, 5); }
#[bench]
fn estimates_31_bluesteins_prime__00000017_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 17); }
#[bench]
fn estimates_31_bluesteins_inner__00000017_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b, 17); }
#[bench]
fn estimates_31_bluesteins_prime__00000149_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 149); }
#[bench]
fn estimates_31_bluesteins_inner__00000149_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b, 149); }
#[bench]
fn estimates_31_bluesteins_prime__00000151_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 151); }
#[bench]
fn estimates_31_bluesteins_inner__00000151_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b, 151); }
#[bench]
fn estimates_31_bluesteins_prime__00000251_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 251); }
#[bench]
fn estimates_31_bluesteins_inner__00000251_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b, 251); }
#[bench]
fn estimates_31_bluesteins_prime__00000257_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 257); }
#[bench]
fn estimates_31_bluesteins_inner__00000257_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b, 257); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_radix4(b: &mut Bencher, len: usize) {
    assert!(len % 4 == 0);

    let fft = Radix4::new(len, FftDirection::Forward);

    let mut signal = vec![ Complex { re: 0_f64, im: 0_f64 }; len ];
    let mut spectrum = signal.clone();
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_outofplace_scratch_len() ];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch);
    });
}

#[bench]
fn estimates_40_radix4____________00000064_0000(b: &mut Bencher) { bench_radix4(b, 64); }
#[bench]
fn estimates_40_radix4____________00000256_0000(b: &mut Bencher) { bench_radix4(b, 256); }
#[bench]
fn estimates_40_radix4____________00001024_0000(b: &mut Bencher) { bench_radix4(b, 1024); }
#[bench]
fn estimates_40_radix4____________00004096_0000(b: &mut Bencher) { bench_radix4(b, 4096); }
#[bench]
fn estimates_40_radix4____________00016384_0000(b: &mut Bencher) { bench_radix4(b, 16384); }
#[bench]
fn estimates_40_radix4____________00065536_0000(b: &mut Bencher) { bench_radix4(b, 65536); }
#[bench]
fn estimates_40_radix4____________00262144_0000(b: &mut Bencher) { bench_radix4(b, 262144); }
#[bench]
fn estimates_40_radix4____________01048576_0000(b: &mut Bencher) { bench_radix4(b, 1048576); }
#[bench]
fn estimates_40_radix4____________04194304_0000(b: &mut Bencher) { bench_radix4(b, 4194304); }

fn bench_dft_multi_f64(b: &mut Bencher, len: usize) {
    let fft = Dft::new(len, FftDirection::Forward);

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; 1000 * len ];
    let mut output = vec![ Complex { re: 0_f64, im: 0_f64 }; 1000 * len ];
    let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
    });
}

// all butterflies
#[bench]
fn estimates_50_dft_______________00000002_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 2); }
#[bench]
fn estimates_50_dft_______________00000003_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 3); }
#[bench]
fn estimates_50_dft_______________00000004_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 4); }
#[bench]
fn estimates_50_dft_______________00000005_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 5); }
#[bench]
fn estimates_50_dft_______________00000006_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 6); }
#[bench]
fn estimates_50_dft_______________00000007_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 7); }
#[bench]
fn estimates_50_dft_______________00000008_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 8); }
#[bench]
fn estimates_50_dft_______________00000011_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 11); }
#[bench]
fn estimates_50_dft_______________00000013_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 13); }
#[bench]
fn estimates_50_dft_______________00000016_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 16); }
#[bench]
fn estimates_50_dft_______________00000017_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 17); }
#[bench]
fn estimates_50_dft_______________00000019_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 19); }
#[bench]
fn estimates_50_dft_______________00000023_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 23); }
#[bench]
fn estimates_50_dft_______________00000027_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 27); }
#[bench]
fn estimates_50_dft_______________00000029_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 29); }
#[bench]
fn estimates_50_dft_______________00000031_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 31); }
#[bench]
fn estimates_50_dft_______________00000032_0000(b: &mut Bencher) { bench_dft_multi_f64(b, 32); }
