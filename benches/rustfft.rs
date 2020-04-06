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
fn bench_fft(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); len];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}


// Powers of 4
#[bench] fn planned_p2_00000064(b: &mut Bencher) { bench_fft(b,       64); }
#[bench] fn planned_p2_00000256(b: &mut Bencher) { bench_fft(b,      256); }
#[bench] fn planned_p2_00001024(b: &mut Bencher) { bench_fft(b,     1024); }
#[bench] fn planned_p2_00004096(b: &mut Bencher) { bench_fft(b,     4096); }
#[bench] fn planned_p2_00016384(b: &mut Bencher) { bench_fft(b,    16384); }
#[bench] fn planned_p2_00065536(b: &mut Bencher) { bench_fft(b,    65536); }
#[bench] fn planned_p2_01048576(b: &mut Bencher) { bench_fft(b,  1048576); }
#[bench] fn planned_p2_16777216(b: &mut Bencher) { bench_fft(b, 16777216); }


// Powers of 7
#[bench] fn planned_p7_00343(b: &mut Bencher) { bench_fft(b,   343); }
#[bench] fn planned_p7_02401(b: &mut Bencher) { bench_fft(b,  2401); }
#[bench] fn planned_p7_16807(b: &mut Bencher) { bench_fft(b, 16807); }

// Prime lengths
// Prime lengths
#[bench] fn planned_prime_0005(b: &mut Bencher)     { bench_fft(b,  5); }
#[bench] fn planned_prime_0017(b: &mut Bencher)     { bench_fft(b,  17); }
#[bench] fn planned_prime_0149(b: &mut Bencher)     { bench_fft(b,  149); }
#[bench] fn planned_prime_0151(b: &mut Bencher)     { bench_fft(b,  151); }
#[bench] fn planned_prime_0251(b: &mut Bencher)     { bench_fft(b,  251); }
#[bench] fn planned_prime_0257(b: &mut Bencher)     { bench_fft(b,  257); }
#[bench] fn planned_prime_1009(b: &mut Bencher)     { bench_fft(b,  1009); }
#[bench] fn planned_prime_2017(b: &mut Bencher)     { bench_fft(b,  2017); }
#[bench] fn planned_prime_2879(b: &mut Bencher)     { bench_fft(b,  2879); }
#[bench] fn planned_prime_32767(b: &mut Bencher)    { bench_fft(b, 32767); }
#[bench] fn planned_prime_65521(b: &mut Bencher)    { bench_fft(b, 65521); }
#[bench] fn planned_prime_65537(b: &mut Bencher)    { bench_fft(b, 65537); }
#[bench] fn planned_prime_746483(b: &mut Bencher)   { bench_fft(b,746483); }
#[bench] fn planned_prime_746497(b: &mut Bencher)   { bench_fft(b,746497); }

//primes raised to a power
#[bench] fn planned_primepower_44521(b: &mut Bencher) { bench_fft(b, 44521); } // 211^2
#[bench] fn planned_primepower_160801(b: &mut Bencher) { bench_fft(b, 160801); } // 401^2

// numbers times powers of two
#[bench] fn planned_composite_024576(b: &mut Bencher) { bench_fft(b,  24576); }
#[bench] fn planned_composite_020736(b: &mut Bencher) { bench_fft(b,  20736); }

// power of 2 times large prime
#[bench] fn planned_composite_032192(b: &mut Bencher) { bench_fft(b,  32192); }
#[bench] fn planned_composite_024028(b: &mut Bencher) { bench_fft(b,  24028); }

// small mixed composites times a large prime
#[bench] fn planned_composite_030270(b: &mut Bencher) { bench_fft(b,  30270); }

// small mixed composites
#[bench] fn planned_composite_000018(b: &mut Bencher) { bench_fft(b,  00018); }
#[bench] fn planned_composite_000360(b: &mut Bencher) { bench_fft(b,  00360); }
#[bench] fn planned_composite_044100(b: &mut Bencher) { bench_fft(b,  44100); }
#[bench] fn planned_composite_048000(b: &mut Bencher) { bench_fft(b,  48000); }
#[bench] fn planned_composite_046656(b: &mut Bencher) { bench_fft(b,  46656); }
#[bench] fn planned_composite_100000(b: &mut Bencher) { bench_fft(b,  100000); }

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

    let fft : Arc<Fft<_>> = Arc::new(RadersAlgorithm::new(len, inner_fft));

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

    let fft : Arc<Fft<_>> = Arc::new(RadersAlgorithm::new(len, inner_fft));

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
        let fft : Arc<Fft<f32>> = Arc::new(RadersAlgorithm::new(len, Arc::clone(&inner_fft)));
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

fn get_splitradix_scalar(len: usize) -> Arc<Fft<f32>> {
    match len {
        8 => Arc::new(Butterfly8::new(false)),
        16 => Arc::new(Butterfly16::new(false)),
        32 => Arc::new(Butterfly32::new(false)),
        _ => {
             let mut radishes = Vec::new();
            radishes.push(Arc::new(Butterfly16::new(false)) as Arc<Fft<f32>>);
            radishes.push(Arc::new(Butterfly32::new(false)) as Arc<Fft<f32>>);

            while radishes.last().unwrap().len() < len {
                let quarter = Arc::clone(&radishes[radishes.len() - 2]);
                let half = Arc::clone(&radishes[radishes.len() - 1]);
                radishes.push(Arc::new(SplitRadix::new(half, quarter)) as Arc<Fft<f32>>);
            }
            Arc::clone(&radishes.last().unwrap())
        }
    }
}


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_splitradix_scalar(b: &mut Bencher, len: usize) {
    assert!(len % 4 == 0);

    let fft = get_splitradix_scalar(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch);} );
}

#[bench] fn splitradix_scalar__0000128(b: &mut Bencher) { bench_splitradix_scalar(b, 128); }
#[bench] fn splitradix_scalar__0000256(b: &mut Bencher) { bench_splitradix_scalar(b, 256); }
#[bench] fn splitradix_scalar__0000512(b: &mut Bencher) { bench_splitradix_scalar(b, 512); }
#[bench] fn splitradix_scalar__0001024(b: &mut Bencher) { bench_splitradix_scalar(b, 1024); }
#[bench] fn splitradix_scalar__0002048(b: &mut Bencher) { bench_splitradix_scalar(b, 2048); }
#[bench] fn splitradix_scalar__0004096(b: &mut Bencher) { bench_splitradix_scalar(b, 4096); }
#[bench] fn splitradix_scalar__0008192(b: &mut Bencher) { bench_splitradix_scalar(b, 8192); }
#[bench] fn splitradix_scalar__0016384(b: &mut Bencher) { bench_splitradix_scalar(b, 16384); }
#[bench] fn splitradix_scalar__0032768(b: &mut Bencher) { bench_splitradix_scalar(b, 32768); }
#[bench] fn splitradix_scalar__0065536(b: &mut Bencher) { bench_splitradix_scalar(b, 65536); }
#[bench] fn splitradix_scalar__1048576(b: &mut Bencher) { bench_splitradix_scalar(b, 1048576); }

use std::collections::HashMap;

fn get_splitradix_avx(len: usize) -> Arc<Fft<f32>> {
    let mut fft_map = HashMap::new();

    fft_map.insert(8, Arc::new(MixedRadixAvx4x2::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")) as Arc<Fft<f32>>);
    fft_map.insert(16, Arc::new(MixedRadixAvx4x4::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")) as Arc<Fft<f32>>);
    fft_map.insert(32, Arc::new(MixedRadixAvx4x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")) as Arc<Fft<f32>>);
    fft_map.insert(64, Arc::new(MixedRadixAvx8x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")) as Arc<Fft<f32>>);

    let len_zeros = len.trailing_zeros();
    for i in 7..(len_zeros+1) {
        let len = 1 << i;
        let half_fft = Arc::clone(fft_map.get(&(len / 2)).unwrap());
        let quarter_fft = Arc::clone(fft_map.get(&(len / 4)).unwrap());

        fft_map.insert(len, Arc::new(SplitRadixAvx::new(half_fft, quarter_fft).expect("Can't run benchmark because this machine doesn't have the required instruction sets")));
    }

    Arc::clone(fft_map.get(&len).unwrap())
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_splitradix_avx(b: &mut Bencher, len: usize) {
    assert!(len % 16 == 0);

    let fft = get_splitradix_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_inplace_with_scratch(&mut buffer, &mut scratch);
    });
}


#[bench] fn splitradix_avx_00000128(b: &mut Bencher) { bench_splitradix_avx(b, 128); }
#[bench] fn splitradix_avx_00000256(b: &mut Bencher) { bench_splitradix_avx(b, 256); }
#[bench] fn splitradix_avx_00000512(b: &mut Bencher) { bench_splitradix_avx(b, 512); }
#[bench] fn splitradix_avx_00001024(b: &mut Bencher) { bench_splitradix_avx(b, 1024); }
#[bench] fn splitradix_avx_00002048(b: &mut Bencher) { bench_splitradix_avx(b, 2048); }
#[bench] fn splitradix_avx_00004096(b: &mut Bencher) { bench_splitradix_avx(b, 4096); }
#[bench] fn splitradix_avx_00008192(b: &mut Bencher) { bench_splitradix_avx(b, 8192); }
#[bench] fn splitradix_avx_00016384(b: &mut Bencher) { bench_splitradix_avx(b, 16384); }
#[bench] fn splitradix_avx_00032768(b: &mut Bencher) { bench_splitradix_avx(b, 32768); }
#[bench] fn splitradix_avx_00065536(b: &mut Bencher) { bench_splitradix_avx(b, 65536); }
#[bench] fn splitradix_avx_01048576(b: &mut Bencher) { bench_splitradix_avx(b, 1048576); }
#[bench] fn splitradix_avx_16777216(b: &mut Bencher) { bench_splitradix_avx(b, 16777216); }

fn get_2xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(MixedRadixAvx4x2::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(MixedRadixAvx4x4::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(MixedRadixAvx4x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(MixedRadixAvx8x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
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

fn get_4xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(MixedRadixAvx4x2::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(MixedRadixAvx4x4::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(MixedRadixAvx4x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(MixedRadixAvx8x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
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

fn get_8xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(MixedRadixAvx4x2::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(MixedRadixAvx4x4::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(MixedRadixAvx4x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(MixedRadixAvx8x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
            let inner = get_8xn_avx(len / 8);
            Arc::new(MixedRadix8xnAvx::new(inner).expect("Can't run benchmark because this machine doesn't have the required instruction sets"))
        }
    }
}

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
#[bench] fn mixed_8xn_avx__1048576(b: &mut Bencher) { bench_mixed_8xn_avx(b, 1048576); }

fn get_16xn_avx(len: usize) -> Arc<dyn Fft<f32>> {
    match len {
        8 => Arc::new(MixedRadixAvx4x2::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(MixedRadixAvx4x4::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(MixedRadixAvx4x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(MixedRadixAvx8x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
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
        12 => Arc::new(MixedRadixAvx4x3::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        24 => Arc::new(MixedRadixAvx4x6::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        48 => Arc::new(MixedRadixAvx4x12::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
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
fn bench_butterfly(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); len * 10];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_multi(&mut buffer, &mut scratch); });
}


#[bench] fn butterfly_02(b: &mut Bencher) { bench_butterfly(b, 2); }
#[bench] fn butterfly_03(b: &mut Bencher) { bench_butterfly(b, 3); }
#[bench] fn butterfly_04(b: &mut Bencher) { bench_butterfly(b, 4); }
#[bench] fn butterfly_05(b: &mut Bencher) { bench_butterfly(b, 5); }
#[bench] fn butterfly_06(b: &mut Bencher) { bench_butterfly(b, 6); }
#[bench] fn butterfly_07(b: &mut Bencher) { bench_butterfly(b, 7); }
#[bench] fn butterfly_08(b: &mut Bencher) { bench_butterfly(b, 8); }
#[bench] fn butterfly_12(b: &mut Bencher) { bench_butterfly(b, 12); }
#[bench] fn butterfly_16(b: &mut Bencher) { bench_butterfly(b, 16); }
#[bench] fn butterfly_24(b: &mut Bencher) { bench_butterfly(b, 24); }
#[bench] fn butterfly_32(b: &mut Bencher) { bench_butterfly(b, 32); }
#[bench] fn butterfly_48(b: &mut Bencher) { bench_butterfly(b, 48); }
#[bench] fn butterfly_64(b: &mut Bencher) { bench_butterfly(b, 64); }