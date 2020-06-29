#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate test;
extern crate rustfft;

use std::sync::Arc;
use test::Bencher;
use rustfft::{Fft};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::algorithm::*;
use rustfft::algorithm::butterflies::*;
use rustfft::algorithm::avx::*;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_f32(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch); 
    });
}


// Powers of 4
#[bench] fn planned32_p2_00000064(b: &mut Bencher) { bench_planned_f32(b,       64); }
#[bench] fn planned32_p2_00000128(b: &mut Bencher) { bench_planned_f32(b,      128); }
#[bench] fn planned32_p2_00000256(b: &mut Bencher) { bench_planned_f32(b,      256); }
#[bench] fn planned32_p2_00000512(b: &mut Bencher) { bench_planned_f32(b,      512); }
#[bench] fn planned32_p2_00001024(b: &mut Bencher) { bench_planned_f32(b,     1024); }
#[bench] fn planned32_p2_00002048(b: &mut Bencher) { bench_planned_f32(b,     2048); }
#[bench] fn planned32_p2_00004096(b: &mut Bencher) { bench_planned_f32(b,     4096); }
#[bench] fn planned32_p2_00016384(b: &mut Bencher) { bench_planned_f32(b,    16384); }
#[bench] fn planned32_p2_00065536(b: &mut Bencher) { bench_planned_f32(b,    65536); }
#[bench] fn planned32_p2_01048576(b: &mut Bencher) { bench_planned_f32(b,  1048576); }
#[bench] fn planned32_p2_16777216(b: &mut Bencher) { bench_planned_f32(b, 16777216); }


// Powers of 5
#[bench] fn planned32_p5_00125(b: &mut Bencher) { bench_planned_f32(b, 125); }
#[bench] fn planned32_p5_00625(b: &mut Bencher) { bench_planned_f32(b, 625); }
#[bench] fn planned32_p5_03125(b: &mut Bencher) { bench_planned_f32(b, 3125); }
#[bench] fn planned32_p5_15625(b: &mut Bencher) { bench_planned_f32(b, 15625); }

// Powers of 7
#[bench] fn planned32_p7_00343(b: &mut Bencher) { bench_planned_f32(b,   343); }
#[bench] fn planned32_p7_02401(b: &mut Bencher) { bench_planned_f32(b,  2401); }
#[bench] fn planned32_p7_16807(b: &mut Bencher) { bench_planned_f32(b, 16807); }

// Prime lengths
// Prime lengths
#[bench] fn planned32_prime_0005(b: &mut Bencher)     { bench_planned_f32(b,  5); }
#[bench] fn planned32_prime_0017(b: &mut Bencher)     { bench_planned_f32(b,  17); }
#[bench] fn planned32_prime_0149(b: &mut Bencher)     { bench_planned_f32(b,  149); }
#[bench] fn planned32_prime_0151(b: &mut Bencher)     { bench_planned_f32(b,  151); }
#[bench] fn planned32_prime_0251(b: &mut Bencher)     { bench_planned_f32(b,  251); }
#[bench] fn planned32_prime_0257(b: &mut Bencher)     { bench_planned_f32(b,  257); }
#[bench] fn planned32_prime_1009(b: &mut Bencher)     { bench_planned_f32(b,  1009); }
#[bench] fn planned32_prime_2017(b: &mut Bencher)     { bench_planned_f32(b,  2017); }
#[bench] fn planned32_prime_2879(b: &mut Bencher)     { bench_planned_f32(b,  2879); }
#[bench] fn planned32_prime_32767(b: &mut Bencher)    { bench_planned_f32(b, 32767); }
#[bench] fn planned32_prime_65521(b: &mut Bencher)    { bench_planned_f32(b, 65521); }
#[bench] fn planned32_prime_65537(b: &mut Bencher)    { bench_planned_f32(b, 65537); }
#[bench] fn planned32_prime_746483(b: &mut Bencher)   { bench_planned_f32(b,746483); }
#[bench] fn planned32_prime_746497(b: &mut Bencher)   { bench_planned_f32(b,746497); }

//primes raised to a power
#[bench] fn planned32_primepower_044521(b: &mut Bencher) { bench_planned_f32(b, 44521); } // 211^2
#[bench] fn planned32_primepower_160801(b: &mut Bencher) { bench_planned_f32(b, 160801); } // 401^2

// numbers times powers of two
#[bench] fn planned32_composite_024576(b: &mut Bencher) { bench_planned_f32(b,  24576); }
#[bench] fn planned32_composite_020736(b: &mut Bencher) { bench_planned_f32(b,  20736); }

// power of 2 times large prime
#[bench] fn planned32_composite_032192(b: &mut Bencher) { bench_planned_f32(b,  32192); }
#[bench] fn planned32_composite_024028(b: &mut Bencher) { bench_planned_f32(b,  24028); }

// small mixed composites times a large prime
#[bench] fn planned32_composite_030270(b: &mut Bencher) { bench_planned_f32(b,  30270); }

// small mixed composites
#[bench] fn planned32_composite_000018(b: &mut Bencher) { bench_planned_f32(b,  00018); }
#[bench] fn planned32_composite_000360(b: &mut Bencher) { bench_planned_f32(b,  00360); }
#[bench] fn planned32_composite_044100(b: &mut Bencher) { bench_planned_f32(b,  44100); }
#[bench] fn planned32_composite_048000(b: &mut Bencher) { bench_planned_f32(b,  48000); }
#[bench] fn planned32_composite_046656(b: &mut Bencher) { bench_planned_f32(b,  46656); }
#[bench] fn planned32_composite_100000(b: &mut Bencher) { bench_planned_f32(b,  100000); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_planned_f64(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

#[bench] fn planned64_p2_00000064(b: &mut Bencher) { bench_planned_f64(b,       64); }
#[bench] fn planned64_p2_00000128(b: &mut Bencher) { bench_planned_f64(b,      128); }
#[bench] fn planned64_p2_00000256(b: &mut Bencher) { bench_planned_f64(b,      256); }
#[bench] fn planned64_p2_00000512(b: &mut Bencher) { bench_planned_f64(b,      512); }
#[bench] fn planned64_p2_00001024(b: &mut Bencher) { bench_planned_f64(b,     1024); }
#[bench] fn planned64_p2_00002048(b: &mut Bencher) { bench_planned_f64(b,     2048); }
#[bench] fn planned64_p2_00004096(b: &mut Bencher) { bench_planned_f64(b,     4096); }
#[bench] fn planned64_p2_00016384(b: &mut Bencher) { bench_planned_f64(b,    16384); }
#[bench] fn planned64_p2_00065536(b: &mut Bencher) { bench_planned_f64(b,    65536); }
#[bench] fn planned64_p2_01048576(b: &mut Bencher) { bench_planned_f64(b,  1048576); }
//#[bench] fn planned64_p2_16777216(b: &mut Bencher) { bench_planned_f64(b, 16777216); }

#[bench] fn planned64_p7_00343(b: &mut Bencher) { bench_planned_f64(b,   343); }
#[bench] fn planned64_p7_02401(b: &mut Bencher) { bench_planned_f64(b,  2401); }
#[bench] fn planned64_p7_16807(b: &mut Bencher) { bench_planned_f64(b, 16807); }

// Prime lengths
#[bench] fn planned64_prime_0005(b: &mut Bencher)     { bench_planned_f64(b,  5); }
#[bench] fn planned64_prime_0017(b: &mut Bencher)     { bench_planned_f64(b,  17); }
#[bench] fn planned64_prime_0149(b: &mut Bencher)     { bench_planned_f64(b,  149); }
#[bench] fn planned64_prime_0151(b: &mut Bencher)     { bench_planned_f64(b,  151); }
#[bench] fn planned64_prime_0251(b: &mut Bencher)     { bench_planned_f64(b,  251); }
#[bench] fn planned64_prime_0257(b: &mut Bencher)     { bench_planned_f64(b,  257); }
#[bench] fn planned64_prime_1009(b: &mut Bencher)     { bench_planned_f64(b,  1009); }
#[bench] fn planned64_prime_2017(b: &mut Bencher)     { bench_planned_f64(b,  2017); }
#[bench] fn planned64_prime_2879(b: &mut Bencher)     { bench_planned_f64(b,  2879); }
#[bench] fn planned64_prime_32767(b: &mut Bencher)    { bench_planned_f64(b, 32767); }
#[bench] fn planned64_prime_65521(b: &mut Bencher)    { bench_planned_f64(b, 65521); }
#[bench] fn planned64_prime_65537(b: &mut Bencher)    { bench_planned_f64(b, 65537); }
#[bench] fn planned64_prime_746483(b: &mut Bencher)   { bench_planned_f64(b,746483); }
#[bench] fn planned64_prime_746497(b: &mut Bencher)   { bench_planned_f64(b,746497); }

//primes raised to a power
#[bench] fn planned64_primepower_044521(b: &mut Bencher) { bench_planned_f64(b, 44521); } // 211^2
#[bench] fn planned64_primepower_160801(b: &mut Bencher) { bench_planned_f64(b, 160801); } // 401^2

// numbers times powers of two
#[bench] fn planned64_composite_024576(b: &mut Bencher) { bench_planned_f64(b,  24576); }
#[bench] fn planned64_composite_020736(b: &mut Bencher) { bench_planned_f64(b,  20736); }

// power of 2 times large prime
#[bench] fn planned64_composite_032192(b: &mut Bencher) { bench_planned_f64(b,  32192); }
#[bench] fn planned64_composite_024028(b: &mut Bencher) { bench_planned_f64(b,  24028); }

// small mixed composites times a large prime
#[bench] fn planned64_composite_030270(b: &mut Bencher) { bench_planned_f64(b,  30270); }

// small mixed composites
#[bench] fn planned64_composite_000018(b: &mut Bencher) { bench_planned_f64(b,  00018); }
#[bench] fn planned64_composite_000360(b: &mut Bencher) { bench_planned_f64(b,  00360); }
#[bench] fn planned64_composite_044100(b: &mut Bencher) { bench_planned_f64(b,  44100); }
#[bench] fn planned64_composite_048000(b: &mut Bencher) { bench_planned_f64(b,  48000); }
#[bench] fn planned64_composite_046656(b: &mut Bencher) { bench_planned_f64(b,  46656); }
#[bench] fn planned64_composite_100000(b: &mut Bencher) { bench_planned_f64(b,  100000); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Good-Thomas algorithm
fn bench_good_thomas(b: &mut Bencher, width: usize, height: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<Fft<f32>> = Arc::new(GoodThomasAlgorithm::new(width_fft, height_fft));

    let mut buffer = vec![Complex::zero(); width * height];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn good_thomas_0002_3(b: &mut Bencher) { bench_good_thomas(b,  2, 3); }
#[bench] fn good_thomas_0003_4(b: &mut Bencher) { bench_good_thomas(b,  3, 4); }
#[bench] fn good_thomas_0004_5(b: &mut Bencher) { bench_good_thomas(b,  4, 5); }
#[bench] fn good_thomas_0007_32(b: &mut Bencher) { bench_good_thomas(b, 7, 32); }
#[bench] fn good_thomas_0032_27(b: &mut Bencher) { bench_good_thomas(b,  32, 27); }
#[bench] fn good_thomas_0256_243(b: &mut Bencher) { bench_good_thomas(b,  256, 243); }
#[bench] fn good_thomas_2048_3(b: &mut Bencher) { bench_good_thomas(b,  2048, 3); }
#[bench] fn good_thomas_2048_2187(b: &mut Bencher) { bench_good_thomas(b,  2048, 2187); }

/// Times just the FFT setup (not execution)
/// for a given length, specific to the Good-Thomas algorithm
fn bench_good_thomas_setup(b: &mut Bencher, width: usize, height: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    b.iter(|| { 
        let fft : Arc<Fft<f32>> = Arc::new(GoodThomasAlgorithm::new(Arc::clone(&width_fft), Arc::clone(&height_fft)));
        test::black_box(fft);
    });
}

#[bench] fn good_thomas_setup_0002_3(b: &mut Bencher) { bench_good_thomas_setup(b,  2, 3); }
#[bench] fn good_thomas_setup_0003_4(b: &mut Bencher) { bench_good_thomas_setup(b,  3, 4); }
#[bench] fn good_thomas_setup_0004_5(b: &mut Bencher) { bench_good_thomas_setup(b,  4, 5); }
#[bench] fn good_thomas_setup_0007_32(b: &mut Bencher) { bench_good_thomas_setup(b, 7, 32); }
#[bench] fn good_thomas_setup_0032_27(b: &mut Bencher) { bench_good_thomas_setup(b,  32, 27); }
#[bench] fn good_thomas_setup_0256_243(b: &mut Bencher) { bench_good_thomas_setup(b,  256, 243); }
#[bench] fn good_thomas_setup_2048_3(b: &mut Bencher) { bench_good_thomas_setup(b,  2048, 3); }
#[bench] fn good_thomas_setup_2048_2187(b: &mut Bencher) { bench_good_thomas_setup(b,  2048, 2187); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix algorithm
fn bench_mixed_radix(b: &mut Bencher, width: usize, height: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<Fft<_>> = Arc::new(MixedRadix::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}

#[bench] fn mixed_radix_0002_3(b: &mut Bencher) { bench_mixed_radix(b,  2, 3); }
#[bench] fn mixed_radix_0003_4(b: &mut Bencher) { bench_mixed_radix(b,  3, 4); }
#[bench] fn mixed_radix_0004_5(b: &mut Bencher) { bench_mixed_radix(b,  4, 5); }
#[bench] fn mixed_radix_0007_32(b: &mut Bencher) { bench_mixed_radix(b, 7, 32); }
#[bench] fn mixed_radix_0032_27(b: &mut Bencher) { bench_mixed_radix(b,  32, 27); }
#[bench] fn mixed_radix_0256_243(b: &mut Bencher) { bench_mixed_radix(b,  256, 243); }
#[bench] fn mixed_radix_2048_3(b: &mut Bencher) { bench_mixed_radix(b,  2048, 3); }
#[bench] fn mixed_radix_2048_2187(b: &mut Bencher) { bench_mixed_radix(b,  2048, 2187); }

fn plan_butterfly_fft(len: usize) -> Arc<Fft<f32>> {
    match len {
        2 => Arc::new(Butterfly2::new(false)),
        3 => Arc::new(Butterfly3::new(false)),
        4 => Arc::new(Butterfly4::new(false)),
        5 => Arc::new(Butterfly5::new(false)),
        6 => Arc::new(Butterfly6::new(false)),
        7 => Arc::new(Butterfly7::new(false)),
        8 => Arc::new(Butterfly8::new(false)),
        16 => Arc::new(Butterfly16::new(false)),
        32 => Arc::new(Butterfly32::new(false)),
        _ => panic!("Invalid butterfly size: {}", len),
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the MixedRadixSmall algorithm
fn bench_mixed_radix_small(b: &mut Bencher, width: usize, height: usize) {

    let width_fft = plan_butterfly_fft(width);
    let height_fft = plan_butterfly_fft(height);

    let fft : Arc<Fft<_>> = Arc::new(MixedRadixSmall::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process_inplace_with_scratch(&mut signal, &mut spectrum);} );
}

#[bench] fn mixed_radix_small_0002_3(b: &mut Bencher) { bench_mixed_radix_small(b,  2, 3); }
#[bench] fn mixed_radix_small_0003_4(b: &mut Bencher) { bench_mixed_radix_small(b,  3, 4); }
#[bench] fn mixed_radix_small_0004_5(b: &mut Bencher) { bench_mixed_radix_small(b,  4, 5); }
#[bench] fn mixed_radix_small_0007_32(b: &mut Bencher) { bench_mixed_radix_small(b, 7, 32); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix Double Butterfly algorithm
fn bench_good_thomas_small(b: &mut Bencher, width: usize, height: usize) {

    let width_fft = plan_butterfly_fft(width);
    let height_fft = plan_butterfly_fft(height);

    let fft : Arc<Fft<_>> = Arc::new(GoodThomasAlgorithmSmall::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process_inplace_with_scratch(&mut signal, &mut spectrum);} );
}

#[bench] fn good_thomas_small_0002_3(b: &mut Bencher) { bench_good_thomas_small(b,  2, 3); }
#[bench] fn good_thomas_small_0003_4(b: &mut Bencher) { bench_good_thomas_small(b,  3, 4); }
#[bench] fn good_thomas_small_0004_5(b: &mut Bencher) { bench_good_thomas_small(b,  4, 5); }
#[bench] fn good_thomas_small_0007_32(b: &mut Bencher) { bench_good_thomas_small(b, 7, 32); }


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_raders(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let inner_fft = planner.plan_fft(len - 1);

    let fft : Arc<Fft<_>> = Arc::new(RadersAlgorithm::new(inner_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}

#[bench] fn raders_prime_0005(b: &mut Bencher) { bench_raders(b,  5); }
#[bench] fn raders_prime_0017(b: &mut Bencher) { bench_raders(b,  17); }
#[bench] fn raders_prime_0149(b: &mut Bencher) { bench_raders(b,  149); }
#[bench] fn raders_prime_0151(b: &mut Bencher) { bench_raders(b,  151); }
#[bench] fn raders_prime_0251(b: &mut Bencher) { bench_raders(b,  251); }
#[bench] fn raders_prime_0257(b: &mut Bencher) { bench_raders(b,  257); }
#[bench] fn raders_prime_1009(b: &mut Bencher) { bench_raders(b,  1009); }
#[bench] fn raders_prime_2017(b: &mut Bencher) { bench_raders(b,  2017); }
#[bench] fn raders_prime_12289(b: &mut Bencher) { bench_raders(b, 12289); }
#[bench] fn raders_prime_18433(b: &mut Bencher) { bench_raders(b, 18433); }
#[bench] fn raders_prime_65521(b: &mut Bencher) { bench_raders(b, 65521); }
#[bench] fn raders_prime_65537(b: &mut Bencher) { bench_raders(b, 65537); }
#[bench] fn raders_prime_746483(b: &mut Bencher) { bench_raders(b,746483); }
#[bench] fn raders_prime_746497(b: &mut Bencher) { bench_raders(b,746497); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_raders_power2(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let inner_fft = planner.plan_fft(len - 1);

    let fft : Arc<Fft<_>> = Arc::new(RadersAlgorithm::new(inner_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}


#[bench] fn raders_power2_0017(b: &mut Bencher) { bench_raders_power2(b,  17); }
#[bench] fn raders_power2_0257(b: &mut Bencher) { bench_raders_power2(b,  257); }
#[bench] fn raders_power2_65537(b: &mut Bencher) { bench_raders_power2(b, 65537); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Bluestein's Algorithm
fn bench_bluesteins_scalar_prime(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FFTplanner::new(false);
    let inner_fft = planner.plan_fft((len * 2 - 1).checked_next_power_of_two().unwrap());
    let fft : Arc<Fft<f32>> = Arc::new(BluesteinsAlgorithm::new(len, inner_fft));

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn bench_bluesteins_scalar_prime_0005(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  5); }
#[bench] fn bench_bluesteins_scalar_prime_0017(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  17); }
#[bench] fn bench_bluesteins_scalar_prime_0149(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  149); }
#[bench] fn bench_bluesteins_scalar_prime_0151(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  151); }
#[bench] fn bench_bluesteins_scalar_prime_0251(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  251); }
#[bench] fn bench_bluesteins_scalar_prime_0257(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  257); }
#[bench] fn bench_bluesteins_scalar_prime_1009(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  1009); }
#[bench] fn bench_bluesteins_scalar_prime_2017(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,  2017); }
#[bench] fn bench_bluesteins_scalar_prime_32767(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 32767); }
#[bench] fn bench_bluesteins_scalar_prime_65521(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 65521); }
#[bench] fn bench_bluesteins_scalar_prime_65537(b: &mut Bencher) { bench_bluesteins_scalar_prime(b, 65537); }
#[bench] fn bench_bluesteins_scalar_prime_746483(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,746483); }
#[bench] fn bench_bluesteins_scalar_prime_746497(b: &mut Bencher) { bench_bluesteins_scalar_prime(b,746497); }

#[allow(unused)]
fn plan_new_bluesteins_f32(len: usize) -> Arc<Fft<f32>> {
    assert!(len > 1); // Internal consistency check: The logic in this method doesn't work for a length of 1

    // Plan a step of Bluestein's Algorithm
    // Bluestein's computes a FFT of size `len` by reorganizing it as a FFT of ANY size greter than or equal to len * 2 - 1
    // an obvious choice is the next power of two larger than  len * 2 - 1, but if we can find a smaller FFT that will go faster, we can save a lot of time!
    // We can very efficiently compute any 3 * 2^n, so we can take the next power of 2, divide it by 4, then multiply it by 3. If the result >= len*2 - 1, use it!

    // TODO: if we get the ability to compute arbitrary powers of 3 on the fast path, we can also try max / 16 * 9, max / 32 * 27, max / 128 * 81, to give alternative sizes

    // One caveat is that the size-12 blutterfly is slower than size-16, so we only want to do this if the next power of two is greater than 16
    let min_size = len*2 - 1;
    let max_size = min_size.checked_next_power_of_two().unwrap();

    let potential_3x = max_size / 4 * 3;
    let inner_fft_len = if max_size > 16 && potential_3x >= min_size {
        potential_3x
    } else {
        max_size
    };

    let mut planner = rustfft::FFTplanner::new(false);
    let inner_power2 = planner.plan_fft(inner_fft_len);
    Arc::new(BluesteinsAvx::new(len, inner_power2).unwrap())
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Bluestein's Algorithm
fn bench_bluesteins_avx_prime(b: &mut Bencher, len: usize) {
    let fft : Arc<Fft<f32>> = plan_new_bluesteins_f32(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn bench_bluesteins_avx_prime_0005(b: &mut Bencher) { bench_bluesteins_avx_prime(b,  5); }
#[bench] fn bench_bluesteins_avx_prime_0017(b: &mut Bencher) { bench_bluesteins_avx_prime(b,  17); }
#[bench] fn bench_bluesteins_avx_prime_0149(b: &mut Bencher) { bench_bluesteins_avx_prime(b,  149); }
#[bench] fn bench_bluesteins_avx_prime_0151(b: &mut Bencher) { bench_bluesteins_avx_prime(b,  151); }
#[bench] fn bench_bluesteins_avx_prime_0251(b: &mut Bencher) { bench_bluesteins_avx_prime(b,  251); }
#[bench] fn bench_bluesteins_avx_prime_0257(b: &mut Bencher) { bench_bluesteins_avx_prime(b,  257); }
#[bench] fn bench_bluesteins_avx_prime_1009(b: &mut Bencher) { bench_bluesteins_avx_prime(b,  1009); }
#[bench] fn bench_bluesteins_avx_prime_2017(b: &mut Bencher) { bench_bluesteins_avx_prime(b,  2017); }
#[bench] fn bench_bluesteins_avx_prime_12289(b: &mut Bencher) { bench_bluesteins_avx_prime(b, 12289); }
#[bench] fn bench_bluesteins_avx_prime_18433(b: &mut Bencher) { bench_bluesteins_avx_prime(b, 18433); }
#[bench] fn bench_bluesteins_avx_prime_32767(b: &mut Bencher) { bench_bluesteins_avx_prime(b, 32767); }
#[bench] fn bench_bluesteins_avx_prime_65521(b: &mut Bencher) { bench_bluesteins_avx_prime(b, 65521); }
#[bench] fn bench_bluesteins_avx_prime_65537(b: &mut Bencher) { bench_bluesteins_avx_prime(b, 65537); }
#[bench] fn bench_bluesteins_avx_prime_746483(b: &mut Bencher) { bench_bluesteins_avx_prime(b,746483); }
#[bench] fn bench_bluesteins_avx_prime_746497(b: &mut Bencher) { bench_bluesteins_avx_prime(b,746497); }

/// Times just the FFT setup (not execution)
/// for a given length, specific to Rader's algorithm
fn bench_raders_setup(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let inner_fft = planner.plan_fft(len - 1);

    b.iter(|| { 
        let fft : Arc<Fft<f32>> = Arc::new(RadersAlgorithm::new(Arc::clone(&inner_fft)));
        test::black_box(fft);
    });
}

#[bench] fn raders_setup_0005(b: &mut Bencher) { bench_raders_setup(b,  5); }
#[bench] fn raders_setup_0017(b: &mut Bencher) { bench_raders_setup(b,  17); }
#[bench] fn raders_setup_0151(b: &mut Bencher) { bench_raders_setup(b,  151); }
#[bench] fn raders_setup_0257(b: &mut Bencher) { bench_raders_setup(b,  257); }
#[bench] fn raders_setup_1009(b: &mut Bencher) { bench_raders_setup(b,  1009); }
#[bench] fn raders_setup_2017(b: &mut Bencher) { bench_raders_setup(b,  2017); }
#[bench] fn raders_setup_65537(b: &mut Bencher) { bench_raders_setup(b, 65537); }
#[bench] fn raders_setup_746497(b: &mut Bencher) { bench_raders_setup(b,746497); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_radix4(b: &mut Bencher, len: usize) {
    assert!(len % 4 == 0);

    let fft = Radix4::new(len, false);

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}

#[bench] fn radix4_______64(b: &mut Bencher) { bench_radix4(b, 64); }
#[bench] fn radix4______256(b: &mut Bencher) { bench_radix4(b, 256); }
#[bench] fn radix4_____1024(b: &mut Bencher) { bench_radix4(b, 1024); }
#[bench] fn radix4____65536(b: &mut Bencher) { bench_radix4(b, 65536); }
#[bench] fn radix4__1048576(b: &mut Bencher) { bench_radix4(b, 1048576); }
//#[bench] fn radix4_16777216(b: &mut Bencher) { bench_radix4(b, 16777216); }

fn get_2xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(Butterfly8Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(Butterfly16Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(Butterfly32Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(Butterfly64Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_2xn_avx(len / 2);
            Arc::new(MixedRadix2xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_2xn_avx(b: &mut Bencher, len: usize) {
    assert!(len % 2 == 0);
    let fft = get_2xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_2xn_avx__0000128(b: &mut Bencher) { bench_mixed_2xn_avx(b, 128); }
#[bench] fn mixed_2xn_avx__0000256(b: &mut Bencher) { bench_mixed_2xn_avx(b, 256); }
#[bench] fn mixed_2xn_avx__0000512(b: &mut Bencher) { bench_mixed_2xn_avx(b, 512); }
#[bench] fn mixed_2xn_avx__0001024(b: &mut Bencher) { bench_mixed_2xn_avx(b, 1024); }
#[bench] fn mixed_2xn_avx__0002048(b: &mut Bencher) { bench_mixed_2xn_avx(b, 2048); }
#[bench] fn mixed_2xn_avx__0004096(b: &mut Bencher) { bench_mixed_2xn_avx(b, 4096); }
#[bench] fn mixed_2xn_avx__0008192(b: &mut Bencher) { bench_mixed_2xn_avx(b, 8192); }
#[bench] fn mixed_2xn_avx__0016384(b: &mut Bencher) { bench_mixed_2xn_avx(b, 16384); }
#[bench] fn mixed_2xn_avx__0032768(b: &mut Bencher) { bench_mixed_2xn_avx(b, 32768); }
#[bench] fn mixed_2xn_avx__0065536(b: &mut Bencher) { bench_mixed_2xn_avx(b, 65536); }
#[bench] fn mixed_2xn_avx__1048576(b: &mut Bencher) { bench_mixed_2xn_avx(b, 1048576); }

fn get64_2xn_avx(len: usize) -> Arc<dyn Fft<f64>> {
    match len {
        8 => Arc::new(Butterfly8Avx64::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(Butterfly16Avx64::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(Butterfly32Avx64::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get64_2xn_avx(len / 2);
            Arc::new(MixedRadix2xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed64_2xn_avx(b: &mut Bencher, len: usize) {
    assert!(len % 2 == 0);
    let fft = get64_2xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed64_2xn_avx__0000128(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 128); }
#[bench] fn mixed64_2xn_avx__0000256(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 256); }
#[bench] fn mixed64_2xn_avx__0000512(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 512); }
#[bench] fn mixed64_2xn_avx__0001024(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 1024); }
#[bench] fn mixed64_2xn_avx__0002048(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 2048); }
#[bench] fn mixed64_2xn_avx__0004096(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 4096); }
#[bench] fn mixed64_2xn_avx__0008192(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 8192); }
#[bench] fn mixed64_2xn_avx__0016384(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 16384); }
#[bench] fn mixed64_2xn_avx__0032768(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 32768); }
#[bench] fn mixed64_2xn_avx__0065536(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 65536); }
#[bench] fn mixed64_2xn_avx__1048576(b: &mut Bencher) { bench_mixed64_2xn_avx(b, 1048576); }

fn get_3xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        27 => Arc::new(Butterfly27Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_3xn_avx(len / 3);
            Arc::new(MixedRadix3xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_3xn_avx(b: &mut Bencher, len: usize) {
    let fft = get_3xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_3xn_avx__000081(b: &mut Bencher) { bench_mixed_3xn_avx(b, 81); }
#[bench] fn mixed_3xn_avx__000243(b: &mut Bencher) { bench_mixed_3xn_avx(b, 243); }
#[bench] fn mixed_3xn_avx__000729(b: &mut Bencher) { bench_mixed_3xn_avx(b, 729); }
#[bench] fn mixed_3xn_avx__002187(b: &mut Bencher) { bench_mixed_3xn_avx(b, 2187); }
#[bench] fn mixed_3xn_avx__006561(b: &mut Bencher) { bench_mixed_3xn_avx(b, 6561); }
#[bench] fn mixed_3xn_avx__019683(b: &mut Bencher) { bench_mixed_3xn_avx(b, 19683); }
#[bench] fn mixed_3xn_avx__059049(b: &mut Bencher) { bench_mixed_3xn_avx(b, 59049); }
#[bench] fn mixed_3xn_avx__177147(b: &mut Bencher) { bench_mixed_3xn_avx(b, 177147); }


fn get_9xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        9 => Arc::new(Butterfly9Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        27 => Arc::new(Butterfly27Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_9xn_avx(len / 9);
            Arc::new(MixedRadix9xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_9xn_avx(b: &mut Bencher, len: usize) {
    let fft = get_9xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_9xn_avx__000081(b: &mut Bencher) { bench_mixed_9xn_avx(b, 81); }
#[bench] fn mixed_9xn_avx__000243(b: &mut Bencher) { bench_mixed_9xn_avx(b, 243); }
#[bench] fn mixed_9xn_avx__000729(b: &mut Bencher) { bench_mixed_9xn_avx(b, 729); }
#[bench] fn mixed_9xn_avx__002187(b: &mut Bencher) { bench_mixed_9xn_avx(b, 2187); }
#[bench] fn mixed_9xn_avx__006561(b: &mut Bencher) { bench_mixed_9xn_avx(b, 6561); }
#[bench] fn mixed_9xn_avx__019683(b: &mut Bencher) { bench_mixed_9xn_avx(b, 19683); }
#[bench] fn mixed_9xn_avx__059049(b: &mut Bencher) { bench_mixed_9xn_avx(b, 59049); }
#[bench] fn mixed_9xn_avx__177147(b: &mut Bencher) { bench_mixed_9xn_avx(b, 177147); }

fn get_4xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(Butterfly8Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(Butterfly16Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(Butterfly32Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(Butterfly64Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_4xn_avx(len / 4);
            Arc::new(MixedRadix4xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_4xn_avx(b: &mut Bencher, len: usize) {
    assert!(len % 4 == 0);
    let fft = get_4xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_4xn_avx__0000128(b: &mut Bencher) { bench_mixed_4xn_avx(b, 128); }
#[bench] fn mixed_4xn_avx__0000256(b: &mut Bencher) { bench_mixed_4xn_avx(b, 256); }
#[bench] fn mixed_4xn_avx__0000512(b: &mut Bencher) { bench_mixed_4xn_avx(b, 512); }
#[bench] fn mixed_4xn_avx__0001024(b: &mut Bencher) { bench_mixed_4xn_avx(b, 1024); }
#[bench] fn mixed_4xn_avx__0002048(b: &mut Bencher) { bench_mixed_4xn_avx(b, 2048); }
#[bench] fn mixed_4xn_avx__0004096(b: &mut Bencher) { bench_mixed_4xn_avx(b, 4096); }
#[bench] fn mixed_4xn_avx__0008192(b: &mut Bencher) { bench_mixed_4xn_avx(b, 8192); }
#[bench] fn mixed_4xn_avx__0016384(b: &mut Bencher) { bench_mixed_4xn_avx(b, 16384); }
#[bench] fn mixed_4xn_avx__0032768(b: &mut Bencher) { bench_mixed_4xn_avx(b, 32768); }
#[bench] fn mixed_4xn_avx__0065536(b: &mut Bencher) { bench_mixed_4xn_avx(b, 65536); }
#[bench] fn mixed_4xn_avx__1048576(b: &mut Bencher) { bench_mixed_4xn_avx(b, 1048576); }


fn get_5xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        5 => Arc::new(Butterfly5Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_5xn_avx(len / 5);
            Arc::new(MixedRadix5xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_5xn_avx(b: &mut Bencher, len: usize) {
    assert!(len % 5 == 0);
    let fft = get_5xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_5xn_avx__0000025(b: &mut Bencher) { bench_mixed_5xn_avx(b, 25); }
#[bench] fn mixed_5xn_avx__0000125(b: &mut Bencher) { bench_mixed_5xn_avx(b, 125); }
#[bench] fn mixed_5xn_avx__0000625(b: &mut Bencher) { bench_mixed_5xn_avx(b, 625); }
#[bench] fn mixed_5xn_avx__0003125(b: &mut Bencher) { bench_mixed_5xn_avx(b, 3125); }
#[bench] fn mixed_5xn_avx__0015625(b: &mut Bencher) { bench_mixed_5xn_avx(b, 15625); }
#[bench] fn mixed_5xn_avx__0078125(b: &mut Bencher) { bench_mixed_5xn_avx(b, 78125); }
#[bench] fn mixed_5xn_avx__0390625(b: &mut Bencher) { bench_mixed_5xn_avx(b, 390625); }

fn get_8xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(Butterfly8Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(Butterfly16Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(Butterfly32Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(Butterfly64Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_8xn_avx(len / 8);
            Arc::new(MixedRadix8xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

fn get_6xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        32 => Arc::new(Butterfly32Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(Butterfly64Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_6xn_avx(len / 6);
            assert_eq!(inner.len(), len / 6);
            Arc::new(MixedRadix6xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_6xn_avx(b: &mut Bencher, len: usize) {
    let fft = get_6xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_6xn_avx__0000192(b: &mut Bencher) { bench_mixed_6xn_avx(b, 192); }
#[bench] fn mixed_6xn_avx__0000384(b: &mut Bencher) { bench_mixed_6xn_avx(b, 384); }
#[bench] fn mixed_6xn_avx__0001152(b: &mut Bencher) { bench_mixed_6xn_avx(b, 1152); }
#[bench] fn mixed_6xn_avx__0002304(b: &mut Bencher) { bench_mixed_6xn_avx(b, 2304); }
#[bench] fn mixed_6xn_avx__0006912(b: &mut Bencher) { bench_mixed_6xn_avx(b, 6912); }
#[bench] fn mixed_6xn_avx__0013824(b: &mut Bencher) { bench_mixed_6xn_avx(b, 13824); }
#[bench] fn mixed_6xn_avx__0041472(b: &mut Bencher) { bench_mixed_6xn_avx(b, 41472); }
#[bench] fn mixed_6xn_avx__0082944(b: &mut Bencher) { bench_mixed_6xn_avx(b, 82944); }
#[bench] fn mixed_6xn_avx__0497664(b: &mut Bencher) { bench_mixed_6xn_avx(b, 497664); }
#[bench] fn mixed_6xn_avx__1492992(b: &mut Bencher) { bench_mixed_6xn_avx(b, 1492992); }

fn get_7xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        7 => Arc::new(Butterfly7Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_7xn_avx(len / 7);
            Arc::new(MixedRadix7xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_7xn_avx(b: &mut Bencher, len: usize) {
    assert!(len % 7 == 0);
    let fft = get_7xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_7xn_avx__0000049(b: &mut Bencher) { bench_mixed_7xn_avx(b, 49); }
#[bench] fn mixed_7xn_avx__0000343(b: &mut Bencher) { bench_mixed_7xn_avx(b, 343); }
#[bench] fn mixed_7xn_avx__0002401(b: &mut Bencher) { bench_mixed_7xn_avx(b, 2401); }
#[bench] fn mixed_7xn_avx__0016807(b: &mut Bencher) { bench_mixed_7xn_avx(b, 16807); }
#[bench] fn mixed_7xn_avx__0390625(b: &mut Bencher) { bench_mixed_7xn_avx(b, 117649); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_bluesteins_7xn_avx(b: &mut Bencher, len: usize) {
    assert!(len % 7 == 0);
    let fft = plan_new_bluesteins_f32(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn bluesteins_7xn_avx__0000049(b: &mut Bencher) { bench_bluesteins_7xn_avx(b, 49); }
#[bench] fn bluesteins_7xn_avx__0000343(b: &mut Bencher) { bench_bluesteins_7xn_avx(b, 343); }
#[bench] fn bluesteins_7xn_avx__0002401(b: &mut Bencher) { bench_bluesteins_7xn_avx(b, 2401); }
#[bench] fn bluesteins_7xn_avx__0016807(b: &mut Bencher) { bench_bluesteins_7xn_avx(b, 16807); }
#[bench] fn bluesteins_7xn_avx__0390625(b: &mut Bencher) { bench_bluesteins_7xn_avx(b, 117649); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_8xn_avx(b: &mut Bencher, len: usize) {
    assert!(len % 8 == 0);
    let fft = get_8xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_8xn_avx__0000128(b: &mut Bencher) { bench_mixed_8xn_avx(b, 128); }
#[bench] fn mixed_8xn_avx__0000256(b: &mut Bencher) { bench_mixed_8xn_avx(b, 256); }
#[bench] fn mixed_8xn_avx__0000512(b: &mut Bencher) { bench_mixed_8xn_avx(b, 512); }
#[bench] fn mixed_8xn_avx__0001024(b: &mut Bencher) { bench_mixed_8xn_avx(b, 1024); }
#[bench] fn mixed_8xn_avx__0002048(b: &mut Bencher) { bench_mixed_8xn_avx(b, 2048); }
#[bench] fn mixed_8xn_avx__0004096(b: &mut Bencher) { bench_mixed_8xn_avx(b, 4096); }
#[bench] fn mixed_8xn_avx__0008192(b: &mut Bencher) { bench_mixed_8xn_avx(b, 8192); }
#[bench] fn mixed_8xn_avx__0016384(b: &mut Bencher) { bench_mixed_8xn_avx(b, 16384); }
#[bench] fn mixed_8xn_avx__0032768(b: &mut Bencher) { bench_mixed_8xn_avx(b, 32768); }
#[bench] fn mixed_8xn_avx__0065536(b: &mut Bencher) { bench_mixed_8xn_avx(b, 65536); }
#[bench] fn mixed_8xn_avx__0262144(b: &mut Bencher) { bench_mixed_8xn_avx(b, 262144); }
#[bench] fn mixed_8xn_avx__1048576(b: &mut Bencher) { bench_mixed_8xn_avx(b, 1048576); }
#[bench] fn mixed_8xn_avx__16777216(b: &mut Bencher) { bench_mixed_8xn_avx(b, 16777216); }


fn get_12xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        32 => Arc::new(Butterfly32Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(Butterfly64Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_12xn_avx(len / 12);
            assert_eq!(inner.len(), len / 12);
            Arc::new(MixedRadix12xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_12xn_avx(b: &mut Bencher, len: usize) {
    let fft = get_12xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_12xn_avx__0000384(b: &mut Bencher) { bench_mixed_12xn_avx(b, 384); }
#[bench] fn mixed_12xn_avx__0000768(b: &mut Bencher) { bench_mixed_12xn_avx(b, 768); }
#[bench] fn mixed_12xn_avx__0004608(b: &mut Bencher) { bench_mixed_12xn_avx(b, 4608); }
#[bench] fn mixed_12xn_avx__0009216(b: &mut Bencher) { bench_mixed_12xn_avx(b, 9216); }
#[bench] fn mixed_12xn_avx__0055296(b: &mut Bencher) { bench_mixed_12xn_avx(b, 55296); }
#[bench] fn mixed_12xn_avx__0110592(b: &mut Bencher) { bench_mixed_12xn_avx(b, 110592); }
#[bench] fn mixed_12xn_avx__0663552(b: &mut Bencher) { bench_mixed_12xn_avx(b, 663552); }
#[bench] fn mixed_12xn_avx__1327104(b: &mut Bencher) { bench_mixed_12xn_avx(b, 1327104); }

fn get_16xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(Butterfly8Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(Butterfly16Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(Butterfly32Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(Butterfly64Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_16xn_avx(len / 16);
            Arc::new(MixedRadix16xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_16xn_avx(b: &mut Bencher, len: usize) {
    assert!(len % 16 == 0);
    let fft = get_16xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_16xn_avx__0000128(b: &mut Bencher) { bench_mixed_16xn_avx(b, 128); }
#[bench] fn mixed_16xn_avx__0000256(b: &mut Bencher) { bench_mixed_16xn_avx(b, 256); }
#[bench] fn mixed_16xn_avx__0000512(b: &mut Bencher) { bench_mixed_16xn_avx(b, 512); }
#[bench] fn mixed_16xn_avx__0001024(b: &mut Bencher) { bench_mixed_16xn_avx(b, 1024); }
#[bench] fn mixed_16xn_avx__0002048(b: &mut Bencher) { bench_mixed_16xn_avx(b, 2048); }
#[bench] fn mixed_16xn_avx__0004096(b: &mut Bencher) { bench_mixed_16xn_avx(b, 4096); }
#[bench] fn mixed_16xn_avx__0008192(b: &mut Bencher) { bench_mixed_16xn_avx(b, 8192); }
#[bench] fn mixed_16xn_avx__0016384(b: &mut Bencher) { bench_mixed_16xn_avx(b, 16384); }
#[bench] fn mixed_16xn_avx__0032768(b: &mut Bencher) { bench_mixed_16xn_avx(b, 32768); }
#[bench] fn mixed_16xn_avx__0065536(b: &mut Bencher) { bench_mixed_16xn_avx(b, 65536); }
#[bench] fn mixed_16xn_avx__0262144(b: &mut Bencher) { bench_mixed_16xn_avx(b, 262144); }
#[bench] fn mixed_16xn_avx__1048576(b: &mut Bencher) { bench_mixed_16xn_avx(b, 1048576); }

fn get_mixed_3x2n_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        6 => Arc::new(Butterfly6::new(false)),
        12 => Arc::new(Butterfly12Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        24 => Arc::new(Butterfly24Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        48 => Arc::new(Butterfly48Avx::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        96 => Arc::new(MixedRadix2xnAvx::new(get_mixed_3x2n_avx(len/2)).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        192 => Arc::new(MixedRadix4xnAvx::new(get_mixed_3x2n_avx(len/4)).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        384 => Arc::new(MixedRadix8xnAvx::new(get_mixed_3x2n_avx(len/8)).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        768 => Arc::new(MixedRadix16xnAvx::new(get_mixed_3x2n_avx(len/16)).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_mixed_3x2n_avx(len / 8);
            Arc::new(MixedRadix8xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_3x2n_avx(b: &mut Bencher, len: usize) {
    let fft = get_mixed_3x2n_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_3x2n_avx_0096(b: &mut Bencher) { bench_mixed_3x2n_avx(b, 96); }
#[bench] fn mixed_3x2n_avx_0192(b: &mut Bencher) { bench_mixed_3x2n_avx(b, 192); }
#[bench] fn mixed_3x2n_avx_0384(b: &mut Bencher) { bench_mixed_3x2n_avx(b, 384); }
#[bench] fn mixed_3x2n_avx_0768(b: &mut Bencher) { bench_mixed_3x2n_avx(b, 768); }
#[bench] fn mixed_3x2n_avx_1536(b: &mut Bencher) { bench_mixed_3x2n_avx(b, 1536); }

fn get_mixed_radix_power2(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(Butterfly8::new( false)),
        16 => Arc::new(Butterfly16::new(false)),
        32 => Arc::new(Butterfly32::new(false)),
        _ => {
            let zeroes = len.trailing_zeros();
            assert!(zeroes % 2 == 0);
            let half_zeroes = zeroes / 2;
            let inner = get_mixed_radix_power2(1 << half_zeroes);
            Arc::new(MixedRadix::new(Arc::clone(&inner), inner))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_radix_power2(b: &mut Bencher, len: usize) {
    let fft = get_mixed_radix_power2(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_radix_power2__00000256(b: &mut Bencher) { bench_mixed_radix_power2(b, 256); }
#[bench] fn mixed_radix_power2__00001024(b: &mut Bencher) { bench_mixed_radix_power2(b, 1024); }
#[bench] fn mixed_radix_power2__00004096(b: &mut Bencher) { bench_mixed_radix_power2(b, 4096); }
#[bench] fn mixed_radix_power2__00065536(b: &mut Bencher) { bench_mixed_radix_power2(b, 65536); }
#[bench] fn mixed_radix_power2__01048576(b: &mut Bencher) { bench_mixed_radix_power2(b, 1048576); }
#[bench] fn mixed_radix_power2__16777216(b: &mut Bencher) { bench_mixed_radix_power2(b, 16777216); }


fn get_mixed_radix_inline_power2(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(Butterfly8::new( false)),
        16 => Arc::new(Butterfly16::new(false)),
        32 => Arc::new(Butterfly32::new(false)),
        _ => {
            let zeroes = len.trailing_zeros();
            assert!(zeroes % 2 == 0);
            let half_zeroes = zeroes / 2;
            let inner = get_mixed_radix_inline_power2(1 << half_zeroes);
            Arc::new(MixedRadix::new(Arc::clone(&inner), inner))
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_mixed_radix_inline_power2(b: &mut Bencher, len: usize) {
    let fft = get_mixed_radix_inline_power2(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_radix_power2_inline__00000256(b: &mut Bencher) { bench_mixed_radix_inline_power2(b, 256); }
#[bench] fn mixed_radix_power2_inline__00001024(b: &mut Bencher) { bench_mixed_radix_inline_power2(b, 1024); }
#[bench] fn mixed_radix_power2_inline__00004096(b: &mut Bencher) { bench_mixed_radix_inline_power2(b, 4096); }
#[bench] fn mixed_radix_power2_inline__00065536(b: &mut Bencher) { bench_mixed_radix_inline_power2(b, 65536); }
#[bench] fn mixed_radix_power2_inline__01048576(b: &mut Bencher) { bench_mixed_radix_inline_power2(b, 1048576); }
#[bench] fn mixed_radix_power2_inline__16777216(b: &mut Bencher) { bench_mixed_radix_inline_power2(b, 16777216); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_butterfly32(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_multi(&mut buffer, &mut scratch); });
}

#[bench] fn butterfly32_02(b: &mut Bencher) { bench_butterfly32(b, 2); }
#[bench] fn butterfly32_03(b: &mut Bencher) { bench_butterfly32(b, 3); }
#[bench] fn butterfly32_04(b: &mut Bencher) { bench_butterfly32(b, 4); }
#[bench] fn butterfly32_05(b: &mut Bencher) { bench_butterfly32(b, 5); }
#[bench] fn butterfly32_06(b: &mut Bencher) { bench_butterfly32(b, 6); }
#[bench] fn butterfly32_07(b: &mut Bencher) { bench_butterfly32(b, 7); }
#[bench] fn butterfly32_08(b: &mut Bencher) { bench_butterfly32(b, 8); }
#[bench] fn butterfly32_09(b: &mut Bencher) { bench_butterfly32(b, 9); }
#[bench] fn butterfly32_12(b: &mut Bencher) { bench_butterfly32(b, 12); }
#[bench] fn butterfly32_16(b: &mut Bencher) { bench_butterfly32(b, 16); }
#[bench] fn butterfly32_24(b: &mut Bencher) { bench_butterfly32(b, 24); }
#[bench] fn butterfly32_27(b: &mut Bencher) { bench_butterfly32(b, 27); }
#[bench] fn butterfly32_32(b: &mut Bencher) { bench_butterfly32(b, 32); }
#[bench] fn butterfly32_36(b: &mut Bencher) { bench_butterfly32(b, 36); }
#[bench] fn butterfly32_48(b: &mut Bencher) { bench_butterfly32(b, 48); }
#[bench] fn butterfly32_54(b: &mut Bencher) { bench_butterfly32(b, 54); }
#[bench] fn butterfly32_64(b: &mut Bencher) { bench_butterfly32(b, 64); }
#[bench] fn butterfly32_72(b: &mut Bencher) { bench_butterfly32(b, 72); }
#[bench] fn butterfly32_128(b: &mut Bencher) { bench_butterfly32(b, 128); }
#[bench] fn butterfly32_256(b: &mut Bencher) { bench_butterfly32(b, 256); }
#[bench] fn butterfly32_512(b: &mut Bencher) { bench_butterfly32(b, 512); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_butterfly64(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_multi(&mut buffer, &mut scratch); });
}

#[bench] fn butterfly64_02(b: &mut Bencher) { bench_butterfly64(b, 2); }
#[bench] fn butterfly64_03(b: &mut Bencher) { bench_butterfly64(b, 3); }
#[bench] fn butterfly64_04(b: &mut Bencher) { bench_butterfly64(b, 4); }
#[bench] fn butterfly64_05(b: &mut Bencher) { bench_butterfly64(b, 5); }
#[bench] fn butterfly64_06(b: &mut Bencher) { bench_butterfly64(b, 6); }
#[bench] fn butterfly64_07(b: &mut Bencher) { bench_butterfly64(b, 7); }
#[bench] fn butterfly64_08(b: &mut Bencher) { bench_butterfly64(b, 8); }
#[bench] fn butterfly64_09(b: &mut Bencher) { bench_butterfly64(b, 9); }
#[bench] fn butterfly64_12(b: &mut Bencher) { bench_butterfly64(b, 12); }
#[bench] fn butterfly64_16(b: &mut Bencher) { bench_butterfly64(b, 16); }
#[bench] fn butterfly64_18(b: &mut Bencher) { bench_butterfly64(b, 18); }
#[bench] fn butterfly64_24(b: &mut Bencher) { bench_butterfly64(b, 24); }
#[bench] fn butterfly64_27(b: &mut Bencher) { bench_butterfly64(b, 27); }
#[bench] fn butterfly64_32(b: &mut Bencher) { bench_butterfly64(b, 32); }
#[bench] fn butterfly64_36(b: &mut Bencher) { bench_butterfly64(b, 36); }
#[bench] fn butterfly64_64(b: &mut Bencher) { bench_butterfly64(b, 64); }
#[bench] fn butterfly64_128(b: &mut Bencher) { bench_butterfly64(b, 128); }
#[bench] fn butterfly64_256(b: &mut Bencher) { bench_butterfly64(b, 256); }
#[bench] fn butterfly64_512(b: &mut Bencher) { bench_butterfly64(b, 512); }


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_avx_remainder32(b: &mut Bencher, radix: usize, remainder: usize) {

    let inner_fft : Arc<dyn Fft<f32>> = match remainder {
        1 => Arc::new(DFT::new(1, false)),
        2 => Arc::new( Butterfly2::new(false)),
        3 => Arc::new(Butterfly3::new(false)),
        _ => unimplemented!(),
    };

    let fft : Arc<dyn Fft<f32>> = match radix {
        2  => Arc::new(MixedRadix2xnAvx::new(inner_fft).unwrap()),
        3  => Arc::new(MixedRadix3xnAvx::new(inner_fft).unwrap()),
        4  => Arc::new(MixedRadix4xnAvx::new(inner_fft).unwrap()),
        6  => Arc::new(MixedRadix6xnAvx::new(inner_fft).unwrap()),
        8  => Arc::new(MixedRadix8xnAvx::new(inner_fft).unwrap()),
        9  => Arc::new(MixedRadix9xnAvx::new(inner_fft).unwrap()),
        12 => Arc::new(MixedRadix12xnAvx::new(inner_fft).unwrap()),
        16 => Arc::new(MixedRadix16xnAvx::new(inner_fft).unwrap()),
        _ => unreachable!(),
    };

    let mut buffer = vec![Complex::zero(); fft.len() * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_multi(&mut buffer, &mut scratch); });
}

#[bench] fn remainder32_radix02_1(b: &mut Bencher) { bench_avx_remainder32(b, 2, 1); }
#[bench] fn remainder32_radix03_1(b: &mut Bencher) { bench_avx_remainder32(b, 3, 1); }
#[bench] fn remainder32_radix04_1(b: &mut Bencher) { bench_avx_remainder32(b, 4, 1); }
#[bench] fn remainder32_radix06_1(b: &mut Bencher) { bench_avx_remainder32(b, 6, 1); }
#[bench] fn remainder32_radix08_1(b: &mut Bencher) { bench_avx_remainder32(b, 8, 1); }
#[bench] fn remainder32_radix09_1(b: &mut Bencher) { bench_avx_remainder32(b, 9, 1); }
#[bench] fn remainder32_radix12_1(b: &mut Bencher) { bench_avx_remainder32(b, 12, 1); }
#[bench] fn remainder32_radix16_1(b: &mut Bencher) { bench_avx_remainder32(b, 16, 1); }

#[bench] fn remainder32_radix02_2(b: &mut Bencher) { bench_avx_remainder32(b, 2, 2); }
#[bench] fn remainder32_radix03_2(b: &mut Bencher) { bench_avx_remainder32(b, 3, 2); }
#[bench] fn remainder32_radix04_2(b: &mut Bencher) { bench_avx_remainder32(b, 4, 2); }
#[bench] fn remainder32_radix06_2(b: &mut Bencher) { bench_avx_remainder32(b, 6, 2); }
#[bench] fn remainder32_radix08_2(b: &mut Bencher) { bench_avx_remainder32(b, 8, 2); }
#[bench] fn remainder32_radix09_2(b: &mut Bencher) { bench_avx_remainder32(b, 9, 2); }
#[bench] fn remainder32_radix12_2(b: &mut Bencher) { bench_avx_remainder32(b, 12, 2); }
#[bench] fn remainder32_radix16_2(b: &mut Bencher) { bench_avx_remainder32(b, 16, 2); }

#[bench] fn remainder32_radix02_3(b: &mut Bencher) { bench_avx_remainder32(b, 2, 3); }
#[bench] fn remainder32_radix03_3(b: &mut Bencher) { bench_avx_remainder32(b, 3, 3); }
#[bench] fn remainder32_radix04_3(b: &mut Bencher) { bench_avx_remainder32(b, 4, 3); }
#[bench] fn remainder32_radix06_3(b: &mut Bencher) { bench_avx_remainder32(b, 6, 3); }
#[bench] fn remainder32_radix08_3(b: &mut Bencher) { bench_avx_remainder32(b, 8, 3); }
#[bench] fn remainder32_radix09_3(b: &mut Bencher) { bench_avx_remainder32(b, 9, 3); }
#[bench] fn remainder32_radix12_3(b: &mut Bencher) { bench_avx_remainder32(b, 12, 3); }
#[bench] fn remainder32_radix16_3(b: &mut Bencher) { bench_avx_remainder32(b, 16, 3); }