extern crate test;
extern crate num;
extern crate rustfft;

use test::Bencher;
use num::Complex;

#[bench]
fn rust_fft_343_fft(b: &mut Bencher) {
    const len: usize = 343;
    let mut fft = rustfft::FFT::new(len);
    let fin = [Complex { re: 0.0, im: 0.0 }; len];
    let mut fout = [Complex { re: 0.0, im: 0.0 }; len];
    b.iter(|&mut:| {fft.process(&fin, &mut fout);} );
}
