#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate test;
extern crate rustfft;


use std::sync::Arc;
use test::Bencher;
use rustfft::{FFT, FftInline};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::algorithm::*;
use rustfft::algorithm::butterflies::*;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_fft(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let fft = planner.plan_fft(len);

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}


// Powers of 4
#[bench] fn complex_p2_00000064(b: &mut Bencher) { bench_fft(b,       64); }
#[bench] fn complex_p2_00000256(b: &mut Bencher) { bench_fft(b,      256); }
#[bench] fn complex_p2_00001024(b: &mut Bencher) { bench_fft(b,     1024); }
#[bench] fn complex_p2_00004096(b: &mut Bencher) { bench_fft(b,     4096); }
#[bench] fn complex_p2_00016384(b: &mut Bencher) { bench_fft(b,    16384); }
#[bench] fn complex_p2_00065536(b: &mut Bencher) { bench_fft(b,    65536); }
#[bench] fn complex_p2_01048576(b: &mut Bencher) { bench_fft(b,  1048576); }
#[bench] fn complex_p2_16777216(b: &mut Bencher) { bench_fft(b, 16777216); }


// Powers of 7
#[bench] fn complex_p7_00343(b: &mut Bencher) { bench_fft(b,   343); }
#[bench] fn complex_p7_02401(b: &mut Bencher) { bench_fft(b,  2401); }
#[bench] fn complex_p7_16807(b: &mut Bencher) { bench_fft(b, 16807); }

// Prime lengths
#[bench] fn complex_prime_00005(b: &mut Bencher) { bench_fft(b, 5); }
#[bench] fn complex_prime_00017(b: &mut Bencher) { bench_fft(b, 17); }
#[bench] fn complex_prime_00151(b: &mut Bencher) { bench_fft(b, 151); }
#[bench] fn complex_prime_00257(b: &mut Bencher) { bench_fft(b, 257); }
#[bench] fn complex_prime_01009(b: &mut Bencher) { bench_fft(b, 1009); }
#[bench] fn complex_prime_02017(b: &mut Bencher) { bench_fft(b, 2017); }
#[bench] fn complex_prime_65537(b: &mut Bencher) { bench_fft(b, 65537); }
#[bench] fn complex_prime_746497(b: &mut Bencher) { bench_fft(b, 746497); }

//primes raised to a power
#[bench] fn complex_primepower_44521(b: &mut Bencher) { bench_fft(b, 44521); } // 211^2
#[bench] fn complex_primepower_160801(b: &mut Bencher) { bench_fft(b, 160801); } // 401^2

// numbers times powers of two
#[bench] fn complex_composite_24576(b: &mut Bencher) { bench_fft(b,  24576); }
#[bench] fn complex_composite_20736(b: &mut Bencher) { bench_fft(b,  20736); }

// power of 2 times large prime
#[bench] fn complex_composite_32192(b: &mut Bencher) { bench_fft(b,  32192); }
#[bench] fn complex_composite_24028(b: &mut Bencher) { bench_fft(b,  24028); }

// small mixed composites times a large prime
#[bench] fn complex_composite_30270(b: &mut Bencher) { bench_fft(b,  30270); }


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Good-Thomas algorithm
fn bench_good_thomas(b: &mut Bencher, width: usize, height: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<FFT<_>> = Arc::new(GoodThomasAlgorithm::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
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
        let fft : Arc<FFT<f32>> = Arc::new(GoodThomasAlgorithm::new(Arc::clone(&width_fft), Arc::clone(&height_fft)));
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

    let fft : Arc<FFT<_>> = Arc::new(MixedRadix::new(width_fft, height_fft));

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



fn plan_butterfly(len: usize) -> Arc<FFTButterfly<f32>> {
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
/// for a given length, specific to the Mixed-Radix Double Butterfly algorithm
fn bench_mixed_radix_butterfly(b: &mut Bencher, width: usize, height: usize) {

    let width_fft = plan_butterfly(width);
    let height_fft = plan_butterfly(height);

    let fft : Arc<FFT<_>> = Arc::new(MixedRadixDoubleButterfly::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}

#[bench] fn mixed_radix_butterfly_0002_3(b: &mut Bencher) { bench_mixed_radix_butterfly(b,  2, 3); }
#[bench] fn mixed_radix_butterfly_0003_4(b: &mut Bencher) { bench_mixed_radix_butterfly(b,  3, 4); }
#[bench] fn mixed_radix_butterfly_0004_5(b: &mut Bencher) { bench_mixed_radix_butterfly(b,  4, 5); }
#[bench] fn mixed_radix_butterfly_0007_32(b: &mut Bencher) { bench_mixed_radix_butterfly(b, 7, 32); }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix Double Butterfly algorithm
fn bench_good_thomas_butterfly(b: &mut Bencher, width: usize, height: usize) {

    let width_fft = plan_butterfly(width);
    let height_fft = plan_butterfly(height);

    let fft : Arc<FFT<_>> = Arc::new(GoodThomasAlgorithmDoubleButterfly::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}

#[bench] fn good_thomas_butterfly_0002_3(b: &mut Bencher) { bench_good_thomas_butterfly(b,  2, 3); }
#[bench] fn good_thomas_butterfly_0003_4(b: &mut Bencher) { bench_good_thomas_butterfly(b,  3, 4); }
#[bench] fn good_thomas_butterfly_0004_5(b: &mut Bencher) { bench_good_thomas_butterfly(b,  4, 5); }
#[bench] fn good_thomas_butterfly_0007_32(b: &mut Bencher) { bench_good_thomas_butterfly(b, 7, 32); }


/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_raders(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let inner_fft = planner.plan_fft(len - 1);

    let fft : Arc<FFT<_>> = Arc::new(RadersAlgorithm::new(len, inner_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&mut signal, &mut spectrum);} );
}

#[bench] fn raders_0005(b: &mut Bencher) { bench_raders(b,  5); }
#[bench] fn raders_0017(b: &mut Bencher) { bench_raders(b,  17); }
#[bench] fn raders_0151(b: &mut Bencher) { bench_raders(b,  151); }
#[bench] fn raders_0257(b: &mut Bencher) { bench_raders(b,  257); }
#[bench] fn raders_1009(b: &mut Bencher) { bench_raders(b,  1009); }
#[bench] fn raders_2017(b: &mut Bencher) { bench_raders(b,  2017); }
#[bench] fn raders_65537(b: &mut Bencher) { bench_raders(b, 65537); }
#[bench] fn raders_746497(b: &mut Bencher) { bench_raders(b,746497); }

/// Times just the FFT setup (not execution)
/// for a given length, specific to Rader's algorithm
fn bench_raders_setup(b: &mut Bencher, len: usize) {

    let mut planner = rustfft::FFTplanner::new(false);
    let inner_fft = planner.plan_fft(len - 1);

    b.iter(|| { 
        let fft : Arc<FFT<f32>> = Arc::new(RadersAlgorithm::new(len, Arc::clone(&inner_fft)));
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

fn get_splitradix_scalar(len: usize) -> Arc<FftInline<f32>> {
    match len {
        8 => Arc::new(Butterfly8::new(false)),
        16 => Arc::new(Butterfly16::new(false)),
        32 => Arc::new(Butterfly32::new(false)),
        _ => {
             let mut radishes = Vec::new();
            radishes.push(Arc::new(Butterfly16::new(false)) as Arc<FftInline<f32>>);
            radishes.push(Arc::new(Butterfly32::new(false)) as Arc<FftInline<f32>>);

            while radishes.last().unwrap().len() < len {
                let quarter = Arc::clone(&radishes[radishes.len() - 2]);
                let half = Arc::clone(&radishes[radishes.len() - 1]);
                radishes.push(Arc::new(SplitRadix::new(half, quarter)) as Arc<FftInline<f32>>);
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
    let mut scratch = vec![Zero::zero(); fft.get_required_scratch_len()];
    b.iter(|| { fft.process_inline(&mut buffer, &mut scratch);} );
}

#[bench] fn splitradix_scalar______128(b: &mut Bencher) { bench_splitradix_scalar(b, 128); }
#[bench] fn splitradix_scalar______256(b: &mut Bencher) { bench_splitradix_scalar(b, 256); }
#[bench] fn splitradix_scalar_____1024(b: &mut Bencher) { bench_splitradix_scalar(b, 1024); }
#[bench] fn splitradix_scalar____65536(b: &mut Bencher) { bench_splitradix_scalar(b, 65536); }
#[bench] fn splitradix_scalar__1048576(b: &mut Bencher) { bench_splitradix_scalar(b, 1048576); }
#[bench] fn splitradix_scalar_16777216(b: &mut Bencher) { bench_splitradix_scalar(b, 16777216); }


fn get_splitradix_avx(len: usize) -> Arc<FftInline<f32>> {
    match len {
        8 => Arc::new(MixedRadixAvx4x2::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        16 => Arc::new(MixedRadixAvx4x4::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        32 => Arc::new(MixedRadixAvx4x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        64 => Arc::new(MixedRadixAvx8x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")),
        _ => {
             let mut radishes = Vec::new();
            radishes.push(Arc::new(MixedRadixAvx4x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")) as Arc<FftInline<f32>>);
            radishes.push(Arc::new(MixedRadixAvx8x8::new(false).expect("Can't run benchmark because this machine doesn't have the required instruction sets")) as Arc<FftInline<f32>>);

            while radishes.last().unwrap().len() < len {
                let quarter = Arc::clone(&radishes[radishes.len() - 2]);
                let half = Arc::clone(&radishes[radishes.len() - 1]);
                radishes.push(Arc::new(SplitRadixAvx::new(half, quarter).expect("Can't run benchmark because this machine doesn't have the required instruction sets")) as Arc<FftInline<f32>>);
            }
            Arc::clone(&radishes.last().unwrap())
        }
    }
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_splitradix_avx(b: &mut Bencher, len: usize) {
    assert!(len % 16 == 0);

    let fft = get_splitradix_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_required_scratch_len()];
    b.iter(|| {
        fft.process_inline(&mut buffer, &mut scratch);
    });
}


#[bench] fn splitradix_avx______128(b: &mut Bencher) { bench_splitradix_avx(b, 128); }
#[bench] fn splitradix_avx______256(b: &mut Bencher) { bench_splitradix_avx(b, 256); }
#[bench] fn splitradix_avx______512(b: &mut Bencher) { bench_splitradix_avx(b, 512); }
#[bench] fn splitradix_avx_____1024(b: &mut Bencher) { bench_splitradix_avx(b, 1024); }
#[bench] fn splitradix_avx_____2048(b: &mut Bencher) { bench_splitradix_avx(b, 2048); }
#[bench] fn splitradix_avx_____4096(b: &mut Bencher) { bench_splitradix_avx(b, 4096); }
#[bench] fn splitradix_avx_____8192(b: &mut Bencher) { bench_splitradix_avx(b, 8192); }
#[bench] fn splitradix_avx____16384(b: &mut Bencher) { bench_splitradix_avx(b, 16384); }
#[bench] fn splitradix_avx____32768(b: &mut Bencher) { bench_splitradix_avx(b, 32768); }
#[bench] fn splitradix_avx___262144(b: &mut Bencher) { bench_splitradix_avx(b, 262144); }
#[bench] fn splitradix_avx__1048576(b: &mut Bencher) { bench_splitradix_avx(b, 1048576); }
//#[bench] fn splitradix_avx_16777216(b: &mut Bencher) { bench_splitradix_avx(b, 16777216); }

fn get_8xn_avx(len: usize) -> Arc<dyn FftInline<f32>> {
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
    assert!(len % 32 == 0);
    let fft = get_8xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_required_scratch_len()];
    b.iter(|| {
        fft.process_inline(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_8xn_avx______128(b: &mut Bencher) { bench_mixed_8xn_avx(b, 128); }
#[bench] fn mixed_8xn_avx______256(b: &mut Bencher) { bench_mixed_8xn_avx(b, 256); }
#[bench] fn mixed_8xn_avx______512(b: &mut Bencher) { bench_mixed_8xn_avx(b, 512); }
#[bench] fn mixed_8xn_avx_____1024(b: &mut Bencher) { bench_mixed_8xn_avx(b, 1024); }
#[bench] fn mixed_8xn_avx_____2048(b: &mut Bencher) { bench_mixed_8xn_avx(b, 2048); }
#[bench] fn mixed_8xn_avx_____4096(b: &mut Bencher) { bench_mixed_8xn_avx(b, 4096); }
#[bench] fn mixed_8xn_avx_____8192(b: &mut Bencher) { bench_mixed_8xn_avx(b, 8192); }
#[bench] fn mixed_8xn_avx____16384(b: &mut Bencher) { bench_mixed_8xn_avx(b, 16384); }
#[bench] fn mixed_8xn_avx____32768(b: &mut Bencher) { bench_mixed_8xn_avx(b, 32768); }
#[bench] fn mixed_8xn_avx____65536(b: &mut Bencher) { bench_mixed_8xn_avx(b, 65536); }
#[bench] fn mixed_8xn_avx__1048576(b: &mut Bencher) { bench_mixed_8xn_avx(b, 1048576); }

fn get_16xn_avx(len: usize) -> Arc<dyn FftInline<f32>> {
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
    assert!(len % 64 == 0);
    let fft = get_16xn_avx(len);

    let mut buffer = vec![Zero::zero(); len];
    let mut scratch = vec![Zero::zero(); fft.get_required_scratch_len()];
    b.iter(|| {
        fft.process_inline(&mut buffer, &mut scratch);
    });
}

#[bench] fn mixed_16xn_avx______128(b: &mut Bencher) { bench_mixed_16xn_avx(b, 128); }
#[bench] fn mixed_16xn_avx______256(b: &mut Bencher) { bench_mixed_16xn_avx(b, 256); }
#[bench] fn mixed_16xn_avx______512(b: &mut Bencher) { bench_mixed_16xn_avx(b, 512); }
#[bench] fn mixed_16xn_avx_____1024(b: &mut Bencher) { bench_mixed_16xn_avx(b, 1024); }
#[bench] fn mixed_16xn_avx_____2048(b: &mut Bencher) { bench_mixed_16xn_avx(b, 2048); }
#[bench] fn mixed_16xn_avx_____4096(b: &mut Bencher) { bench_mixed_16xn_avx(b, 4096); }
#[bench] fn mixed_16xn_avx_____8192(b: &mut Bencher) { bench_mixed_16xn_avx(b, 8192); }
#[bench] fn mixed_16xn_avx____16384(b: &mut Bencher) { bench_mixed_16xn_avx(b, 16384); }
#[bench] fn mixed_16xn_avx____32768(b: &mut Bencher) { bench_mixed_16xn_avx(b, 32768); }
#[bench] fn mixed_16xn_avx____65536(b: &mut Bencher) { bench_mixed_16xn_avx(b, 65536); }
#[bench] fn mixed_16xn_avx___262144(b: &mut Bencher) { bench_mixed_16xn_avx(b, 262144); }
#[bench] fn mixed_16xn_avx__1048576(b: &mut Bencher) { bench_mixed_16xn_avx(b, 1048576); }
//#[bench] fn mixed_4xn_avx_16777216(b: &mut Bencher) { bench_mixed_4xn_avx(b, 16777216); }