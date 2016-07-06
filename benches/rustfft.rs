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
    let signal = vec![Complex{re: 0.0, im: 0.0}; len];
    let mut spectrum = signal.clone();
    b.iter(|| {fft.process(&signal[..], &mut spectrum[..]);} );
}

fn bench_fft_with_setup(b: &mut Bencher, len: usize) {
    b.iter(|| {
    	let mut fft = rustfft::FFT::new(len, false);
	    let signal = vec![Complex{re: 0.0, im: 0.0}; len];
	    let mut spectrum = signal.clone();
    	fft.process(&signal[..], &mut spectrum[..]);
    } );
}
/*
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

// some random composite lengths
#[bench] fn complex_composite_0100(b: &mut Bencher) { bench_fft(b,  100); }
#[bench] fn complex_composite_0900(b: &mut Bencher) { bench_fft(b,  900); }
#[bench] fn complex_composite_1000(b: &mut Bencher) { bench_fft(b, 1000); }
#[bench] fn complex_composite_1260(b: &mut Bencher) { bench_fft(b, 1260); }*/

#[bench] fn complex_prime_0211(b: &mut Bencher) { bench_fft(b, 211); }
#[bench] fn complex_prime_1009(b: &mut Bencher) { bench_fft(b, 1009); }
#[bench] fn complex_prime_2017(b: &mut Bencher) { bench_fft(b, 2017); }

#[bench] fn complex_prime_with_setup_0211(b: &mut Bencher) { bench_fft_with_setup(b, 211); }
#[bench] fn complex_prime_with_setup_1009(b: &mut Bencher) { bench_fft_with_setup(b, 1009); }
#[bench] fn complex_prime_with_setup_2017(b: &mut Bencher) { bench_fft_with_setup(b, 2017); }
