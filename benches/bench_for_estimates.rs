#![feature(test)]
extern crate rustfft;
extern crate test;

use paste::paste;
use rustfft::algorithm::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftDirection};
use std::sync::Arc;
use test::Bencher;

// Make fft using planner
fn bench_planned(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make fft using planner
fn bench_planned_multi(b: &mut Bencher, len: usize, reps: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);
    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len * reps];
    let mut output = buffer.clone();
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    b.iter(|| {
        fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
    });
}

// Make a radix4
fn bench_radix4(b: &mut Bencher, len: usize) {
    let fft = Radix4::new(len, FftDirection::Forward);
    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make a mixed radix that uses a planner for the inners
fn bench_mixedradix(b: &mut Bencher, len_a: usize, len_b: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a * len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(MixedRadix::new(fft_a, fft_b));

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); totlen];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make a mixed radix that uses a planner for the inners
fn bench_mixedradixsmall(b: &mut Bencher, len_a: usize, len_b: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a * len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(MixedRadixSmall::new(fft_a, fft_b));

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); totlen];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make a good thomas that uses a planner for the inners
fn bench_goodthomas(b: &mut Bencher, len_a: usize, len_b: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a * len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(GoodThomasAlgorithm::new(fft_a, fft_b));

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); totlen];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make a good thomas that uses a planner for the inners
fn bench_goodthomassmall(b: &mut Bencher, len_a: usize, len_b: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a * len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(GoodThomasAlgorithmSmall::new(fft_a, fft_b));

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); totlen];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make a Raders that uses a planner for the inner
fn bench_raders(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft_inner: Arc<dyn Fft<_>> = planner.plan_fft_forward(len - 1);

    let fft: Arc<dyn Fft<_>> = Arc::new(RadersAlgorithm::new(fft_inner));

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Make a Raders that uses a planner for the inner
fn bench_bluesteins(b: &mut Bencher, len: usize, inner_len: usize) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft_inner: Arc<dyn Fft<_>> = planner.plan_fft_forward(inner_len);

    let fft: Arc<dyn Fft<_>> = Arc::new(BluesteinsAlgorithm::new(len, fft_inner));

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    b.iter(|| {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    });
}

// Create benches using functions taking one argument
macro_rules! make_benches {
    ($name:ident, $fname:ident, { $($len:literal),* }) => {
        paste! {
            $(
                #[bench]
                fn [<bench_ $name _ $len>](b: &mut Bencher)  {
                    [<bench_ $fname>](b, $len);
                }
            )*
        }
    }
}

// Create benches using functions taking two arguments
macro_rules! make_benches_two_args {
    ($name:ident, $fname:ident, { $(($leftlen:literal, $rightlen:literal)),* }) => {
        paste! {
            $(
                #[bench]
                fn [<bench_ $name _ $leftlen _ $rightlen>](b: &mut Bencher)  {
                    [<bench_ $fname>](b, $leftlen, $rightlen);
                }
            )*
        }
    }
}

// A series of power-of-two for fitting Radix4.
make_benches!(radix4, radix4, {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304});

// Mixed radixes. Runs the same combinations for MixedRadix and GoodThomas for easy comparison.
make_benches_two_args!(mixedradix, mixedradix, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233), (127, 1031), (1031, 2003)});
make_benches_two_args!(goodthomas, goodthomas, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233), (127, 1031), (1031, 2003)});
make_benches_two_args!(mixedradixsmall, mixedradixsmall, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)});
make_benches_two_args!(goodthomassmall, goodthomassmall, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)});

// Make a series of different number of repeats for all butterflies. This will give both the fixed overhead of calling a butterfly, and the additional cost of adding one more repeat.
make_benches_two_args!(planned_multi, planned_multi, {(2,1),(3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (11,1), (13,1), (16,1), (17,1), (19,1), (23,1), (29,1), (31,1), (32,1)});
make_benches_two_args!(planned_multi, planned_multi, {(2,2),(3,2), (4,2), (5,2), (6,2), (7,2), (8,2), (11,2), (13,2), (16,2), (17,2), (19,2), (23,2), (29,2), (31,2), (32,2)});
make_benches_two_args!(planned_multi, planned_multi, {(2,3),(3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (11,3), (13,3), (16,3), (17,3), (19,3), (23,3), (29,3), (31,3), (32,3)});
make_benches_two_args!(planned_multi, planned_multi, {(2,5),(3,5), (4,5), (5,5), (6,5), (7,5), (8,5), (11,5), (13,5), (16,5), (17,5), (19,5), (23,5), (29,5), (31,5), (32,5)});
make_benches_two_args!(planned_multi, planned_multi, {(2,8),(3,8), (4,8), (5,8), (6,8), (7,8), (8,8), (11,8), (13,8), (16,8), (17,8), (19,8), (23,8), (29,8), (31,8), (32,8)});
make_benches_two_args!(planned_multi, planned_multi, {(2,13),(3,13), (4,13), (5,13), (6,13), (7,13), (8,13), (11,13), (13,13), (16,13), (17,13), (19,13), (23,13), (29,13), (31,13), (32,13)});
make_benches_two_args!(planned_multi, planned_multi, {(2,21),(3,21), (4,21), (5,21), (6,21), (7,21), (8,21), (11,21), (13,21), (16,21), (17,21), (19,21), (23,21), (29,21), (31,21), (32,21)});
make_benches_two_args!(planned_multi, planned_multi, {(2,34),(3,34), (4,34), (5,34), (6,34), (7,34), (8,34), (11,34), (13,34), (16,34), (17,34), (19,34), (23,34), (29,34), (31,34), (32,34)});
make_benches_two_args!(planned_multi, planned_multi, {(2,55),(3,55), (4,55), (5,55), (6,55), (7,55), (8,55), (11,55), (13,55), (16,55), (17,55), (19,55), (23,55), (29,55), (31,55), (32,55)});
make_benches_two_args!(planned_multi, planned_multi, {(2,89),(3,89), (4,89), (5,89), (6,89), (7,89), (8,89), (11,89), (13,89), (16,89), (17,89), (19,89), (23,89), (29,89), (31,89), (32,89)});

// Measure the inners used in the mixed radix benches above. Measuring these separately instead of using estimated values gives less noise in the data.
make_benches_two_args!(mixinners, planned_multi, {(3, 4), (4, 3), (3, 5), (5, 3), (3, 7), (7, 3),  (3,13), (13,3), (3,31), (31,3)});
make_benches_two_args!(mixinners, planned_multi, {(7, 31), (31, 7), (23, 31), (31, 23), (29,31), (31,29)});
make_benches_two_args!(mixinners, planned_multi, {(31, 127), (127, 31), (31, 233), (233, 31), (127, 233), (233, 127)});
make_benches_two_args!(mixinners, planned_multi, {(127, 1031), (1031, 127), (1031, 2003), (2003, 1031)});

// Raders and the corresponding inners.
make_benches!(raders, raders, {73, 179, 283, 419, 547, 661, 811, 947, 1087, 1229});
make_benches!(planned, planned, {72, 178, 282, 418, 546, 660, 810, 946, 1086, 1228});

// Bluesteins
// Series with fixed fft length and varying inner length.
make_benches_two_args!(bluesteins, bluesteins, {(50,128),(50,256), (50,512), (50,1024), (50,2048)});
// Series with varying fft length and fixed inner length.
make_benches_two_args!(bluesteins, bluesteins, {(10,512),(30,512), (70,512), (90,512)});
// Inners.
make_benches!(planned, planned, {128, 256, 512, 1024, 2048});
