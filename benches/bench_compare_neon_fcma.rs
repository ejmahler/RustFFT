#![feature(test)]
extern crate rustfft;
extern crate test;

use pastey::paste;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::Fft;
use std::sync::Arc;
use test::Bencher;

// Make fft using the Neon planner
fn bench_neon_32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerNeon::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f32>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make fft using the Neon planner
fn bench_neon_64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerNeon::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make fft using the FCMA planner
fn bench_fcma_32(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerFcma::new().unwrap();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f32>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make fft using the FCMA planner
fn bench_fcma_64(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerFcma::new().unwrap();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Create benches using functions taking one argument
macro_rules! make_benches {
    ($name:ident, { $($len:literal),* }) => {
        paste! {
            $(
                #[bench]
                fn [<$name _ $len _f32_neon>](b: &mut Bencher)  {
                    [<bench_neon_32>](b, $len);
                }

                #[bench]
                fn [<$name _ $len _f32_fcma>](b: &mut Bencher)  {
                    [<bench_fcma_32>](b, $len);
                }

                #[bench]
                fn [<$name _ $len _f64_neon>](b: &mut Bencher)  {
                    [<bench_neon_64>](b, $len);
                }

                #[bench]
                fn [<$name _ $len _f64_fcma>](b: &mut Bencher)  {
                    [<bench_fcma_64>](b, $len);
                }
            )*
        }
    }
}

// Powers of two, which go through radix4 and the hand written butterflies
make_benches!(fcmacomparison, {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072});

// The prime butterfly lengths, some composites built from them, and larger primes that end up in
// Rader's or Bluestein's with a prime butterfly somewhere inside
make_benches!(fcmacomparison, {7, 11, 13, 17, 19, 23, 29, 31, 49, 121, 289, 961, 1000, 1009, 5000, 5003});
