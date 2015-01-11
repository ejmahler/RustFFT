extern crate test;
extern crate kissfft;

use test::Bencher;

#[bench]
fn kiss_fft_343_fft(b: &mut Bencher) {
    const len: usize = 343;
    let mut kiss_fft = kissfft::KissFFT::new(len, false);
    let fin = [kissfft::Complex { r: 0.0, i: 0.0 }; len];
    let mut fout = [kissfft::Complex { r: 0.0, i: 0.0 }; len];
    b.iter(|&mut:| {kiss_fft.transform(&fin, &mut fout);} );
}
