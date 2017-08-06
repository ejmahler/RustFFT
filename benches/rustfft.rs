#![feature(test)]
extern crate test;
extern crate rustfft;

use test::Bencher;
use rustfft::num_complex::Complex;

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
#[bench] fn complex_p2_00000032(b: &mut Bencher) { bench_fft(b,       32); }
#[bench] fn complex_p2_00000064(b: &mut Bencher) { bench_fft(b,       64); }
#[bench] fn complex_p2_00000256(b: &mut Bencher) { bench_fft(b,      256); }
#[bench] fn complex_p2_00001024(b: &mut Bencher) { bench_fft(b,     1024); }
#[bench] fn complex_p2_00004096(b: &mut Bencher) { bench_fft(b,     4096); }
#[bench] fn complex_p2_00016384(b: &mut Bencher) { bench_fft(b,    16384); }
#[bench] fn complex_p2_00065536(b: &mut Bencher) { bench_fft(b,    65536); }
#[bench] fn complex_p2_01048576(b: &mut Bencher) { bench_fft(b,  1048576); }
#[bench] fn complex_p2_08388608(b: &mut Bencher) { bench_fft(b,  8388608); }
#[bench] fn complex_p2_16777216(b: &mut Bencher) { bench_fft(b, 16777216); }


// Powers of 7
#[bench] fn complex_p7_00343(b: &mut Bencher) { bench_fft(b,   343); }
#[bench] fn complex_p7_02401(b: &mut Bencher) { bench_fft(b,  2401); }
#[bench] fn complex_p7_16807(b: &mut Bencher) { bench_fft(b, 16807); }

// Prime lengths
#[bench] fn complex_prime_0019(b: &mut Bencher) { bench_fft(b, 19); }
#[bench] fn complex_prime_0151(b: &mut Bencher) { bench_fft(b, 151); }
#[bench] fn complex_prime_1009(b: &mut Bencher) { bench_fft(b, 1009); }
#[bench] fn complex_prime_2017(b: &mut Bencher) { bench_fft(b, 2017); }

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