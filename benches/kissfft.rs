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

#[bench]
fn kiss_fft_16807_fft(b: &mut Bencher) {
    bench_fft(b, 16807);
}

#[bench]
fn kiss_fft_2401_fft(b: &mut Bencher) {
    bench_fft(b, 2401);
}

#[bench]
fn kiss_fft_343_fft(b: &mut Bencher) {
    bench_fft(b, 343);
}
