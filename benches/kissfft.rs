extern crate test;
extern crate kissfft;
extern crate libc;

use test::Bencher;
use std::iter::repeat;
use kissfft::Complex;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_fft(b: &mut Bencher, len: usize) {
    let mut kiss_fft = kissfft::KissFFT::new(len, false);
    let signal: Vec<Complex> = repeat(Complex{r:0.,i:0.}).take(len).collect();
    let mut spectrum: Vec<Complex> = repeat(Complex{r:0.,i:0.}).take(len).collect();
    b.iter(|| {kiss_fft.transform(signal.as_slice(), spectrum.as_mut_slice());} );
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

#[bench]
fn kiss_fft_butterfly_2(b: &mut Bencher) {
    let stride = 4us;
    let num_ffts = 1000us;
    let len = 2 * stride * num_ffts;

    let cfg = unsafe {
        kissfft::binding::kiss_fft_alloc(len as libc::c_int, 0 as libc::c_int, std::ptr::null_mut(), std::ptr::null_mut())
    };

    let mut data: Vec<Complex> = repeat(Complex{r:0.,i:0.}).take(len).collect();

    b.iter(||
    unsafe {
        kissfft::binding::kf_bfly2(data.as_mut_ptr(), stride as libc::size_t, cfg, num_ffts as libc::c_int)
    }
    );
}
