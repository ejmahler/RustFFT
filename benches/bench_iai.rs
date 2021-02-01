#![feature(custom_test_frameworks)]
#![test_runner(iai::runner)]
extern crate rustfft;

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::algorithm::*;
use rustfft::{Fft, FftDirection};
use std::sync::Arc;
use paste::paste;

use iai::black_box;
use iai::iai;

// Make fft using planner
fn bench_planned(len: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>>  = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Make fft using planner
fn bench_planned_multi(len: usize, reps: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);
    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); len*reps];
    let mut scratch: Vec<Complex<f64>>  = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    let mut warmup_buffer: Vec<Complex<f64>> = vec![Complex::zero(); len];
    fft.process_with_scratch(&mut warmup_buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
    black_box(warmup_buffer);
}

// Make a radix4
fn bench_radix4(len: usize, process: bool) {
    assert!(len % 4 == 0);

    let fft = Radix4::new(len, FftDirection::Forward);

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; len ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Make a mixed radix that uses two radix4 as inners
fn bench_mixedradix_rx4(len: usize, process: bool) {

    let totlen = len*len;
    let fft_a: Arc<dyn Fft<_>> = Arc::new(Radix4::new(len, FftDirection::Forward));
    let fft_b: Arc<dyn Fft<_>> = Arc::new(Radix4::new(len, FftDirection::Forward));

    let fft: Arc<dyn Fft<_>> = Arc::new(MixedRadix::new(fft_a, fft_b));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; totlen ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Make a mixed radix that uses a planner for the inners
fn bench_mixedradix(len_a: usize, len_b: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a*len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(MixedRadix::new(fft_a, fft_b));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; totlen ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Make a mixed radix that uses a planner for the inners
fn bench_mixedradixsmall(len_a: usize, len_b: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a*len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(MixedRadixSmall::new(fft_a, fft_b));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; totlen ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Make a good thomas that uses a planner for the inners
fn bench_goodthomas(len_a: usize, len_b: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a*len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(GoodThomasAlgorithm::new(fft_a, fft_b));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; totlen ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Make a good thomas that uses a planner for the inners
fn bench_goodthomassmall(len_a: usize, len_b: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a*len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(GoodThomasAlgorithmSmall::new(fft_a, fft_b));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; totlen ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Make a Raders that uses a planner for the inner
fn bench_raders(len: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft_inner: Arc<dyn Fft<_>> = planner.plan_fft_forward(len-1);

    let fft: Arc<dyn Fft<_>> = Arc::new(RadersAlgorithm::new(fft_inner));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; len ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Make a Raders that uses a planner for the inner
fn bench_bluesteins(len: usize, inner_len: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft_inner: Arc<dyn Fft<_>> = planner.plan_fft_forward(inner_len);

    let fft: Arc<dyn Fft<_>> = Arc::new(BluesteinsAlgorithm::new(len, fft_inner));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; len ];
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_inplace_scratch_len() ];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    if process {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    black_box(buffer);
    black_box(scratch);
}

// Create benches using functions taking one argument
macro_rules! make_benches {
    ($name:ident, $fname:ident, { $($len:literal),* }) => {
        paste! {
            $(
                #[iai]
                fn [<bench_ $name _ $len>]() {
                    [<bench_ $fname>](black_box($len), true);
                }
                #[iai]
                fn [<bench_ $name _setup_ $len>]() {
                    [<bench_ $fname>](black_box($len), false);
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
                #[iai]
                fn [<bench_ $name _ $leftlen _ $rightlen>]() {
                    [<bench_ $fname>](black_box($leftlen), black_box($rightlen), true);
                }
                #[iai]
                fn [<bench_ $name _setup_ $leftlen _ $rightlen>]() {
                    [<bench_ $fname>](black_box($leftlen), black_box($rightlen), false);
                }
            )*
        }
    }
}

//make_benches!(planned, planned, {2,3,4,5,6,7,8,11,13,16,17,19,23,29,31,32,127,233});
//make_benches!(radix4, radix4, {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304});
//make_benches!(mixedradix_rx4, mixedradix_rx4, {32, 64, 128, 256, 512, 1024, 2048});
make_benches_two_args!(mixedradix, mixedradix, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233)});
make_benches_two_args!(goodthomas, goodthomas, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233)});
make_benches_two_args!(mixedradixsmall, mixedradixsmall, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)});
make_benches_two_args!(goodthomassmall, goodthomassmall, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)});
make_benches_two_args!(planned_multi, planned_multi, {(2,1),(3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (11,1), (13,1), (16,1), (17,1), (19,1), (23,1), (29,1), (31,1), (32,1), (127,1), (233,1)});
make_benches_two_args!(planned_multi, planned_multi, {(2,2),(3,2), (4,2), (5,2), (6,2), (7,2), (8,2), (11,2), (13,2), (16,2), (17,2), (19,2), (23,2), (29,2), (31,2), (32,2), (127,2), (233,2)});
make_benches_two_args!(planned_multi, planned_multi, {(2,3),(3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (11,3), (13,3), (16,3), (17,3), (19,3), (23,3), (29,3), (31,3), (32,3), (127,3), (233,3)});
make_benches_two_args!(planned_multi, planned_multi, {(2,4),(3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (11,4), (13,4), (16,4), (17,4), (19,4), (23,4), (29,4), (31,4), (32,4), (127,4), (233,4)});
make_benches_two_args!(planned_multi, planned_multi, {(2,5),(3,5), (4,5), (5,5), (6,5), (7,5), (8,5), (11,5), (13,5), (16,5), (17,5), (19,5), (23,5), (29,5), (31,5), (32,5), (127,5), (233,5)});
make_benches_two_args!(planned_multi, planned_multi, {(2,6),(3,6), (4,6), (5,6), (6,6), (7,6), (8,6), (11,6), (13,6), (16,6), (17,6), (19,6), (23,6), (29,6), (31,6), (32,6), (127,6), (233,6)});
make_benches_two_args!(planned_multi, planned_multi, {(2,7),(3,7), (4,7), (5,7), (6,7), (7,7), (8,7), (11,7), (13,7), (16,7), (17,7), (19,7), (23,7), (29,7), (31,7), (32,7), (127,7), (233,7)});
make_benches_two_args!(planned_multi, planned_multi, {(2,8),(3,8), (4,8), (5,8), (6,8), (7,8), (8,8), (11,8), (13,8), (16,8), (17,8), (19,8), (23,8), (29,8), (31,8), (32,8), (127,8), (233,8)});
make_benches_two_args!(planned_multi, planned_multi, {(2,9),(3,9), (4,9), (5,9), (6,9), (7,9), (8,9), (11,9), (13,9), (16,9), (17,9), (19,9), (23,9), (29,9), (31,9), (32,9), (127,9), (233,9)});
make_benches_two_args!(planned_multi, planned_multi, {(2,10),(3,10), (4,10), (5,10), (6,10), (7,10), (8,10), (11,10), (13,10), (16,10), (17,10), (19,10), (23,10), (29,10), (31,10), (32,10), (127,10), (233,10)});

make_benches!(raders, raders, {73, 179, 283, 419, 547, 661, 811, 947, 1087, 1229});
make_benches!(planned, planned, {72, 178, 282, 418, 546, 660, 810, 946, 1086, 1228});

make_benches_two_args!(bluesteins, bluesteins, {(50,128),(50,256), (50,512), (50,1024), (50,2048)});
make_benches_two_args!(bluesteins, bluesteins, {(10,512),(30,512), (70,512), (90,512)});
make_benches!(planned, planned, {128, 256, 512, 1024, 2048});