//#![feature(custom_test_frameworks)]
//#![test_runner(iai::runner)]
//#![allow(non_snake_case)]
extern crate rustfft;

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::algorithm::*;
use rustfft::{Fft, FftDirection};
use std::sync::Arc;
use paste::paste;

use iai::black_box;
use iai::iai;

fn bench_butterfly(len: usize, process: bool) {
    let mut planner = rustfft::FftPlannerScalar::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); 1000 * len];
    let mut output: Vec<Complex<f64>>  = vec![Complex::zero(); 1000 * len];
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

macro_rules! make_benches {
    ($name:ident { $($len:literal),* }) => {
        paste! {
            $(
                //#[iai]
                fn [<bench_ $name _ $len>]() {
                    [<bench_ $name>](black_box($len), true);
                }
                //#[iai]
                fn [<bench_ $name _setup_ $len>]() {
                    [<bench_ $name>](black_box($len), false);
                }
            )*
        }
    }
}

make_benches!(butterfly {2,3,4,5,6,7,8,11,13,16,17,19,23,29,31,32});
make_benches!(radix4 {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304});

macro_rules! run_benches {
    ($({$name:ident { $($len:literal),* }}), *) => {
        paste! {
            iai::main!(
                $(
                    $(
                        [<bench_ $name _ $len>],
                        [<bench_ $name _setup_ $len>],
                    )*
                )*
            );
        }
    }
}

run_benches!(
    { butterfly {2,3,4,5,6,7,8,11,13,16,17,19,23,29,31,32}},
    { radix4 {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304} }
);



