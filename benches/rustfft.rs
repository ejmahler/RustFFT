extern crate test;
extern crate num;
extern crate rustfft;

use test::Bencher;
use num::Complex;
use std::iter::repeat;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_fft(b: &mut Bencher, len: usize) {
    let mut fft = rustfft::FFT::new(len);
    let signal: Vec<Complex<f32>> = repeat(Complex{re:0.,im:0.}).take(len).collect();
    let mut spectrum: Vec<Complex<f32>> = repeat(Complex{re:0.,im:0.}).take(len).collect();
    b.iter(|&mut:| {fft.process(signal.as_slice(), spectrum.as_mut_slice());} );
}

#[bench]
fn rust_fft_2401_fft(b: &mut Bencher) {
    bench_fft(b, 2401);
}

#[bench]
fn rust_fft_343_fft(b: &mut Bencher) {
    bench_fft(b, 343);
}
