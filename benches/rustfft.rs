#![feature(test)]
extern crate test;
extern crate num;
extern crate rustfft;

use test::Bencher;
use num::Complex;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_fft(b: &mut Bencher, len: usize) {
    let mut fft = rustfft::FFT::new(len, false);
    let signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&signal[..], &mut spectrum[..]);} );
}

// Powers of 2
#[bench] fn complex_p2_00128(b: &mut Bencher) { bench_fft(b,    128); }
#[bench] fn complex_p2_00512(b: &mut Bencher) { bench_fft(b,   512); }
#[bench] fn complex_p2_02048(b: &mut Bencher) { bench_fft(b,  2048); }
#[bench] fn complex_p2_08192(b: &mut Bencher) { bench_fft(b,  8192); }
#[bench] fn complex_p2_32768(b: &mut Bencher) { bench_fft(b, 32768); }

// Powers of 4
#[bench] fn complex_p4_00064(b: &mut Bencher) { bench_fft(b,    64); }
#[bench] fn complex_p4_00256(b: &mut Bencher) { bench_fft(b,   256); }
#[bench] fn complex_p4_01024(b: &mut Bencher) { bench_fft(b,  1024); }
#[bench] fn complex_p4_04096(b: &mut Bencher) { bench_fft(b,  4096); }
#[bench] fn complex_p4_16384(b: &mut Bencher) { bench_fft(b, 16384); }

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