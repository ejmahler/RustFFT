extern crate test;
extern crate kissfft;

use test::Bencher;
use std::iter::repeat;
use kissfft::Complex;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_fft(b: &mut Bencher, len: usize) {
    let mut kiss_fft = kissfft::KissFFT::new(len, false);
    let signal: Vec<Complex> = repeat(Complex{r:0.,i:0.}).take(len).collect();
    let mut spectrum: Vec<Complex> = repeat(Complex{r:0.,i:0.}).take(len).collect();
    b.iter(|&mut:| {kiss_fft.transform(signal.as_slice(), spectrum.as_mut_slice());} );
}

// Powers of 7
#[bench]
fn kiss_fft_7pow3_fft(b: &mut Bencher) { bench_fft(b, 343); }
#[bench]
fn kiss_fft_7pow4_fft(b: &mut Bencher) { bench_fft(b, 2401); }
#[bench]
fn kiss_fft_7pow5_fft(b: &mut Bencher) { bench_fft(b, 16807); }

// Powers of 2
#[bench]
fn kiss_fft_2pow8_fft(b: &mut Bencher) { bench_fft(b, 256); }
#[bench]
fn kiss_fft_2pow10_fft(b: &mut Bencher) { bench_fft(b, 1024); }
#[bench]
fn kiss_fft_2pow12_fft(b: &mut Bencher) { bench_fft(b, 4096); }
#[bench]
fn kiss_fft_2pow14_fft(b: &mut Bencher) { bench_fft(b, 16384); }

// Mixed powers of 2, 3, and 5
#[bench]
fn kiss_fft_goodmix222_fft(b: &mut Bencher) { bench_fft(b, 900); }
#[bench]
fn kiss_fft_goodmix332_fft(b: &mut Bencher) { bench_fft(b, 5400); }
