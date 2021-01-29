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
    let mut output: Vec<Complex<f64>>  = vec![Complex::zero(); len];
    let mut scratch: Vec<Complex<f64>>  = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
    if process {
        fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
    }
    else {
        black_box(buffer);
        black_box(output);
        black_box(scratch);
    }
}

// Make a radix4
fn bench_radix4(len: usize, process: bool) {
    assert!(len % 4 == 0);

    let fft = Radix4::new(len, FftDirection::Forward);

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; len ];
    let mut output = buffer.clone();
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_outofplace_scratch_len() ];
    if process {
        fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
    }
    else {
        black_box(buffer);
        black_box(output);
        black_box(scratch);
    }
}

// Make a mixed radix that uses two radix4 as inners
fn bench_mixedradix_rx4(len: usize, process: bool) {

    let totlen = len*len;
    let fft_a: Arc<dyn Fft<_>> = Arc::new(Radix4::new(len, FftDirection::Forward));
    let fft_b: Arc<dyn Fft<_>> = Arc::new(Radix4::new(len, FftDirection::Forward));

    let fft: Arc<dyn Fft<_>> = Arc::new(MixedRadix::new(fft_a, fft_b));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; totlen ];
    let mut output = buffer.clone();
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_outofplace_scratch_len() ];
    if process {
        fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
    }
    else {
        black_box(buffer);
        black_box(output);
        black_box(scratch);
    }
}

// Make a mixed radix that uses a planner for the inners
fn bench_mixedradix(len_a: usize, len_b: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a*len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(MixedRadix::new(fft_a, fft_b));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; totlen ];
    let mut output = buffer.clone();
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_outofplace_scratch_len() ];
    if process {
        fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
    }
    else {
        black_box(buffer);
        black_box(output);
        black_box(scratch);
    }
}

// Make a good thomas that uses a planner for the inners
fn bench_goodthomas(len_a: usize, len_b: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let totlen = len_a*len_b;
    let fft_a: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_a);
    let fft_b: Arc<dyn Fft<_>> = planner.plan_fft_forward(len_b);

    let fft: Arc<dyn Fft<_>> = Arc::new(GoodThomasAlgorithm::new(fft_a, fft_b));

    let mut buffer = vec![ Complex { re: 0_f64, im: 0_f64 }; totlen ];
    let mut output = buffer.clone();
    let mut scratch = vec![ Complex { re: 0_f64, im: 0_f64 }; fft.get_outofplace_scratch_len() ];
    if process {
        fft.process_outofplace_with_scratch(&mut buffer, &mut output, &mut scratch);
    }
    else {
        black_box(buffer);
        black_box(output);
        black_box(scratch);
    }
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

make_benches!(butterfly, planned, {2,3,4,5,6,7,8,11,13,16,17,19,23,29,31,32});
make_benches!(radix4, radix4, {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304});
make_benches!(mixedradix_rx4, mixedradix_rx4, {32, 64, 128, 256, 512, 1024, 2048});
make_benches_two_args!(mixedradix, mixedradix, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)});
make_benches_two_args!(goodthomas, goodthomas, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)});


