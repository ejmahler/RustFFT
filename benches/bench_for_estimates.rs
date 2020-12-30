#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate test;
extern crate rustfft;

use std::sync::Arc;
use test::Bencher;
use rustfft::{Fft, FFTnum, Length, IsInverse};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::algorithm::*;
use rustfft::algorithm::butterflies::*;

struct Noop {
    len: usize,
    inverse: bool,
}
impl<T: FFTnum> Fft<T> for Noop {
    fn process_with_scratch(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {}
    fn process_multi(&self, _input: &mut [Complex<T>], _output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {}
    fn process_inplace_with_scratch(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {}
    fn process_inplace_multi(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {}
    fn get_inplace_scratch_len(&self) -> usize { self.len }
    fn get_out_of_place_scratch_len(&self) -> usize { 0 }
}
impl Length for Noop {
    fn len(&self) -> usize { self.len }
}
impl IsInverse for Noop {
    fn is_inverse(&self) -> bool { self.inverse }
}




/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_f64(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FftPlannerScalar::new(false);
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_multi_f64(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FftPlannerScalar::new(false);
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); 1000*len];
    let mut output = vec![Complex::zero(); 1000*len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_multi(&mut buffer, &mut output, &mut scratch); });
}

// all butterflies
#[bench] fn estimates_10_butterfly_________0002_0000(b: &mut Bencher) { bench_planned_multi_f64(b,        2); }
#[bench] fn estimates_10_butterfly_________0003_0000(b: &mut Bencher) { bench_planned_multi_f64(b,        3); }
#[bench] fn estimates_10_butterfly_________0004_0000(b: &mut Bencher) { bench_planned_multi_f64(b,        4); }
#[bench] fn estimates_10_butterfly_________0005_0000(b: &mut Bencher) { bench_planned_multi_f64(b,        5); }
#[bench] fn estimates_10_butterfly_________0006_0000(b: &mut Bencher) { bench_planned_multi_f64(b,        6); }
#[bench] fn estimates_10_butterfly_________0007_0000(b: &mut Bencher) { bench_planned_multi_f64(b,        7); }
#[bench] fn estimates_10_butterfly_________0008_0000(b: &mut Bencher) { bench_planned_multi_f64(b,        8); }
#[bench] fn estimates_10_butterfly_________0011_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       11); }
#[bench] fn estimates_10_butterfly_________0013_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       13); }
#[bench] fn estimates_10_butterfly_________0016_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       16); }
#[bench] fn estimates_10_butterfly_________0017_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       17); }
#[bench] fn estimates_10_butterfly_________0019_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       19); }
#[bench] fn estimates_10_butterfly_________0023_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       23); }
#[bench] fn estimates_10_butterfly_________0027_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       27); }
#[bench] fn estimates_10_butterfly_________0029_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       29); }
#[bench] fn estimates_10_butterfly_________0031_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       31); }
#[bench] fn estimates_10_butterfly_________0032_0000(b: &mut Bencher) { bench_planned_multi_f64(b,       32); }


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Good-Thomas algorithm
fn bench_good_thomas(b: &mut Bencher, width: usize, height: usize) {

    let mut planner = rustfft::FftPlannerScalar::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<Fft<f64>> = Arc::new(GoodThomasAlgorithm::new(width_fft, height_fft));

    let mut buffer = vec![Complex::zero(); width * height];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}
#[bench] fn estimates_21_good_thomas_______0002_0003(b: &mut Bencher) { bench_good_thomas(b,  2,  3); }
#[bench] fn estimates_21_good_thomas_______0003_0004(b: &mut Bencher) { bench_good_thomas(b,  3,  4); }
#[bench] fn estimates_21_good_thomas_______0004_0005(b: &mut Bencher) { bench_good_thomas(b,  4,  5); }
#[bench] fn estimates_21_good_thomas_______0007_0032(b: &mut Bencher) { bench_good_thomas(b,  7, 32); }
#[bench] fn estimates_21_good_thomas_______0011_0017(b: &mut Bencher) { bench_good_thomas(b, 11, 17); }
#[bench] fn estimates_21_good_thomas_______0017_0031(b: &mut Bencher) { bench_good_thomas(b, 17, 31); }
#[bench] fn estimates_21_good_thomas_______0029_0031(b: &mut Bencher) { bench_good_thomas(b, 29, 31); }
#[bench] fn estimates_21_good_thomas_______0128_0256(b: &mut Bencher) { bench_good_thomas(b,128, 256); }
#[bench] fn estimates_21_good_thomas_______0128_0181(b: &mut Bencher) { bench_good_thomas(b,128, 181); }
#[bench] fn estimates_21_good_thomas_______0181_0191(b: &mut Bencher) { bench_good_thomas(b,181, 191); }




/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix algorithm
fn bench_mixed_radix(b: &mut Bencher, width: usize, height: usize) {

    let mut planner = rustfft::FftPlannerScalar::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<Fft<_>> = Arc::new(MixedRadix::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f64, im: 0_f64}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}
#[bench] fn estimates_22_mixed_radix_______0002_0003(b: &mut Bencher) { bench_mixed_radix(b,  2,  3); }
#[bench] fn estimates_22_mixed_radix_______0003_0004(b: &mut Bencher) { bench_mixed_radix(b,  3,  4); }
#[bench] fn estimates_22_mixed_radix_______0004_0005(b: &mut Bencher) { bench_mixed_radix(b,  4,  5); }
#[bench] fn estimates_22_mixed_radix_______0007_0032(b: &mut Bencher) { bench_mixed_radix(b,  7, 32); }
#[bench] fn estimates_22_mixed_radix_______0011_0017(b: &mut Bencher) { bench_mixed_radix(b, 11, 17); }
#[bench] fn estimates_22_mixed_radix_______0017_0031(b: &mut Bencher) { bench_mixed_radix(b, 17, 31); }
#[bench] fn estimates_22_mixed_radix_______0029_0031(b: &mut Bencher) { bench_mixed_radix(b, 29, 31); }
#[bench] fn estimates_22_mixed_radix_______0128_0256(b: &mut Bencher) { bench_mixed_radix(b,128, 256); }
#[bench] fn estimates_22_mixed_radix_______0128_0181(b: &mut Bencher) { bench_mixed_radix(b,128, 181); }
#[bench] fn estimates_22_mixed_radix_______0181_0191(b: &mut Bencher) { bench_mixed_radix(b,181, 191); }


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the MixedRadixSmall algorithm
fn bench_mixed_radix_small(b: &mut Bencher, width: usize, height: usize) {

    let mut planner = rustfft::FftPlannerScalar::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<Fft<_>> = Arc::new(MixedRadixSmall::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f64, im: 0_f64}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}
#[bench] fn estimates_23_mixed_radix_small_0002_0003(b: &mut Bencher) { bench_mixed_radix_small(b,  2,  3); }
#[bench] fn estimates_23_mixed_radix_small_0003_0004(b: &mut Bencher) { bench_mixed_radix_small(b,  3,  4); }
#[bench] fn estimates_23_mixed_radix_small_0004_0005(b: &mut Bencher) { bench_mixed_radix_small(b,  4,  5); }
#[bench] fn estimates_23_mixed_radix_small_0007_0032(b: &mut Bencher) { bench_mixed_radix_small(b,  7, 32); }
#[bench] fn estimates_23_mixed_radix_small_0011_0017(b: &mut Bencher) { bench_mixed_radix_small(b, 11, 17); }
#[bench] fn estimates_23_mixed_radix_small_0017_0031(b: &mut Bencher) { bench_mixed_radix_small(b, 17, 31); }
#[bench] fn estimates_23_mixed_radix_small_0029_0031(b: &mut Bencher) { bench_mixed_radix_small(b, 29, 31); }
#[bench] fn estimates_23_mixed_radix_small_0128_0256(b: &mut Bencher) { bench_mixed_radix_small(b,128, 256); }
#[bench] fn estimates_23_mixed_radix_small_0128_0281(b: &mut Bencher) { bench_mixed_radix_small(b,128, 181); }
#[bench] fn estimates_23_mixed_radix_small_0181_0191(b: &mut Bencher) { bench_mixed_radix_small(b,181, 191); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix Double Butterfly algorithm
fn bench_good_thomas_small(b: &mut Bencher, width: usize, height: usize) {

    let mut planner = rustfft::FftPlannerScalar::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<Fft<_>> = Arc::new(GoodThomasAlgorithmSmall::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f64, im: 0_f64}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}
#[bench] fn estimates_24_good_thomas_small_0002_0003(b: &mut Bencher) { bench_good_thomas_small(b,  2,  3); }
#[bench] fn estimates_24_good_thomas_small_0003_0004(b: &mut Bencher) { bench_good_thomas_small(b,  3,  4); }
#[bench] fn estimates_24_good_thomas_small_0004_0005(b: &mut Bencher) { bench_good_thomas_small(b,  4,  5); }
#[bench] fn estimates_24_good_thomas_small_0007_0032(b: &mut Bencher) { bench_good_thomas_small(b,  7, 32); }
#[bench] fn estimates_24_good_thomas_small_0011_0017(b: &mut Bencher) { bench_good_thomas_small(b, 11, 17); }
#[bench] fn estimates_24_good_thomas_small_0017_0031(b: &mut Bencher) { bench_good_thomas_small(b, 17, 31); }
#[bench] fn estimates_24_good_thomas_small_0029_0031(b: &mut Bencher) { bench_good_thomas_small(b, 29, 31); }
#[bench] fn estimates_24_good_thomas_small_0128_0256(b: &mut Bencher) { bench_good_thomas_small(b,128, 256); }
#[bench] fn estimates_24_good_thomas_small_0128_0181(b: &mut Bencher) { bench_good_thomas_small(b,128, 181); }
#[bench] fn estimates_24_good_thomas_small_0181_0191(b: &mut Bencher) { bench_good_thomas_small(b,181, 191); }

#[bench] fn estimates_20_mixed_radix_inner_0002_0000(b: &mut Bencher) { bench_planned_multi_f64(b,  2); }
#[bench] fn estimates_20_mixed_radix_inner_0003_0000(b: &mut Bencher) { bench_planned_multi_f64(b,  3); }
#[bench] fn estimates_20_mixed_radix_inner_0004_0000(b: &mut Bencher) { bench_planned_multi_f64(b,  4); }
#[bench] fn estimates_20_mixed_radix_inner_0005_0000(b: &mut Bencher) { bench_planned_multi_f64(b,  5); }
#[bench] fn estimates_20_mixed_radix_inner_0007_0000(b: &mut Bencher) { bench_planned_multi_f64(b,  7); }
#[bench] fn estimates_20_mixed_radix_inner_0011_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 11); }
#[bench] fn estimates_20_mixed_radix_inner_0017_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 17); }
#[bench] fn estimates_20_mixed_radix_inner_0029_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 29); }
#[bench] fn estimates_20_mixed_radix_inner_0031_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 31); }
#[bench] fn estimates_20_mixed_radix_inner_0032_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 32); }
#[bench] fn estimates_20_mixed_radix_inner_0128_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 128); }
#[bench] fn estimates_20_mixed_radix_inner_0181_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 181); }
#[bench] fn estimates_20_mixed_radix_inner_0191_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 191); }
#[bench] fn estimates_20_mixed_radix_inner_0256_0000(b: &mut Bencher) { bench_planned_multi_f64(b, 256); }



/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_raders_scalar(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FftPlannerScalar::new(false);
    let inner_fft = planner.plan_fft(len - 1);

    let fft : Arc<Fft<_>> = Arc::new(RadersAlgorithm::new(inner_fft));

    let mut buffer = vec![Complex{re: 0_f64, im: 0_f64}; len];
    let mut scratch = vec![Complex{re: 0_f64, im: 0_f64}; fft.get_inplace_scratch_len()];
    b.iter(|| {fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}

fn bench_raders_inner(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FftPlannerScalar::new(false);
    let fft = planner.plan_fft(len - 1);

    let mut buffer = vec![Complex{re: 0_f64, im: 0_f64}; len-1];
    let mut scratch = vec![Complex{re: 0_f64, im: 0_f64}; fft.get_inplace_scratch_len()];
    b.iter(|| {fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn estimates_30_raders____________0005_0000(b: &mut Bencher) { bench_raders_scalar(b,  5); }
#[bench] fn estimates_30_raders_inner______0005_0000(b: &mut Bencher) { bench_raders_inner(b,  5); }
#[bench] fn estimates_30_raders____________0017_0000(b: &mut Bencher) { bench_raders_scalar(b,  17); }
#[bench] fn estimates_30_raders_inner______0017_0000(b: &mut Bencher) { bench_raders_inner(b,  17); }
#[bench] fn estimates_30_raders____________0149_0000(b: &mut Bencher) { bench_raders_scalar(b,  149); }
#[bench] fn estimates_30_raders_inner______0149_0000(b: &mut Bencher) { bench_raders_inner(b,  149); }
#[bench] fn estimates_30_raders____________0151_0000(b: &mut Bencher) { bench_raders_scalar(b,  151); }
#[bench] fn estimates_30_raders_inner______0151_0000(b: &mut Bencher) { bench_raders_inner(b,  151); }
#[bench] fn estimates_30_raders____________0251_0000(b: &mut Bencher) { bench_raders_scalar(b,  251); }
#[bench] fn estimates_30_raders_inner______0251_0000(b: &mut Bencher) { bench_raders_inner(b,  251); }
#[bench] fn estimates_30_raders____________0257_0000(b: &mut Bencher) { bench_raders_scalar(b,  257); }
#[bench] fn estimates_30_raders_inner______0257_0000(b: &mut Bencher) { bench_raders_inner(b,  257); }



/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Bluestein's Algorithm
fn bench_bluesteins_scalar_prime(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new(false);
    let inner_fft = planner.plan_fft((len * 2 - 1).checked_next_power_of_two().unwrap());
    let fft : Arc<Fft<f64>> = Arc::new(BluesteinsAlgorithm::new(len, inner_fft));

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}

fn bench_bluesteins_scalar_inner(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new(false);
    let inner_len = (len * 2 - 1).checked_next_power_of_two().unwrap();
    let fft = planner.plan_fft(inner_len);

    let mut buffer = vec![Complex{re: 0_f64, im: 0_f64}; inner_len];
    let mut scratch = vec![Complex{re: 0_f64, im: 0_f64}; fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn estimates_31_bluesteins_prime__0005_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  5); }
#[bench] fn estimates_31_bluesteins_inner__0005_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b,  5); }
#[bench] fn estimates_31_bluesteins_prime__0017_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  17); }
#[bench] fn estimates_31_bluesteins_inner__0017_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b,  17); }
#[bench] fn estimates_31_bluesteins_prime__0149_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  149); }
#[bench] fn estimates_31_bluesteins_inner__0149_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b,  149); }
#[bench] fn estimates_31_bluesteins_prime__0151_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  151); }
#[bench] fn estimates_31_bluesteins_inner__0151_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b,  151); }
#[bench] fn estimates_31_bluesteins_prime__0251_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  251); }
#[bench] fn estimates_31_bluesteins_inner__0251_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b,  251); }
#[bench] fn estimates_31_bluesteins_prime__0257_0000(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  257); }
#[bench] fn estimates_31_bluesteins_inner__0257_0000(b: &mut Bencher) { bench_bluesteins_scalar_inner(b,  257); }



/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_radix4(b: &mut Bencher, len: usize) {
    assert!(len % 4 == 0);

    let fft = Radix4::new(len, false);

    let mut signal = vec![Complex{re: 0_f64, im: 0_f64}; len];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}

#[bench] fn estimates_40_radix4____________0064_0000(b: &mut Bencher) { bench_radix4(b, 64); }
#[bench] fn estimates_40_radix4____________0256_0000(b: &mut Bencher) { bench_radix4(b, 256); }
#[bench] fn estimates_40_radix4____________1024_0000(b: &mut Bencher) { bench_radix4(b, 1024); }
#[bench] fn estimates_40_radix4____________4096_0000(b: &mut Bencher) { bench_radix4(b, 4096); }
