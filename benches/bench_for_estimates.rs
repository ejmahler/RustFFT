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
make_benches!(radix4, radix4, {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216});

// Mixed radixes. Runs the same combinations for MixedRadix and GoodThomas for easy comparison.
make_benches_two_args!(mixedradix, mixedradix, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233), (127, 1031), (1031, 2003), (2003, 2048), (4093, 4096)});
make_benches_two_args!(goodthomas, goodthomas, {(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233), (127, 1031), (1031, 2003), (2003, 2048), (4093, 4096)});
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
make_benches_two_args!(planned_multi, planned_multi, {(2,256),(3,256), (4,256), (5,256), (6,256), (7,256), (8,256), (11,256), (13,256), (16,256), (17,256), (19,256), (23,256), (29,256), (31,256), (32,256)});
make_benches_two_args!(planned_multi, planned_multi, {(2,512),(3,512), (4,512), (5,512), (6,512), (7,512), (8,512), (11,512), (13,512), (16,512), (17,512), (19,512), (23,512), (29,512), (31,512), (32,512)});
make_benches_two_args!(planned_multi, planned_multi, {(2,1024),(3,1024), (4,1024), (5,1024), (6,1024), (7,1024), (8,1024), (11,1024), (13,1024), (16,1024), (17,1024), (19,1024), (23,1024), (29,1024), (31,1024), (32,1024)});
make_benches_two_args!(planned_multi, planned_multi, {(2,2048),(3,2048), (4,2048), (5,2048), (6,2048), (7,2048), (8,2048), (11,2048), (13,2048), (16,2048), (17,2048), (19,2048), (23,2048), (29,2048), (31,2048), (32,2048)});
make_benches_two_args!(planned_multi, planned_multi, {(2,4096),(3,4096), (4,4096), (5,4096), (6,4096), (7,4096), (8,4096), (11,4096), (13,4096), (16,4096), (17,4096), (19,4096), (23,4096), (29,4096), (31,4096), (32,4096)});
make_benches_two_args!(planned_multi, planned_multi, {(2,8192),(3,8192), (4,8192), (5,8192), (6,8192), (7,8192), (8,8192), (11,8192), (13,8192), (16,8192), (17,8192), (19,8192), (23,8192), (29,8192), (31,8192), (32,8192)});
make_benches_two_args!(planned_multi, planned_multi, {(2,16384),(3,16384), (4,16384), (5,16384), (6,16384), (7,16384), (8,16384), (11,16384), (13,16384), (16,16384), (17,16384), (19,16384), (23,16384), (29,16384), (31,16384), (32,16384)});
make_benches_two_args!(planned_multi, planned_multi, {(2,32768),(3,32768), (4,32768), (5,32768), (6,32768), (7,32768), (8,32768), (11,32768), (13,32768), (16,32768), (17,32768), (19,32768), (23,32768), (29,32768), (31,32768), (32,32768)});
make_benches_two_args!(planned_multi, planned_multi, {(2,65536),(3,65536), (4,65536), (5,65536), (6,65536), (7,65536), (8,65536), (11,65536), (13,65536), (16,65536), (17,65536), (19,65536), (23,65536), (29,65536), (31,65536), (32,65536)});


// Measure the inners used in the mixed radix benches above. Measuring these separately instead of using estimated values gives less noise in the data.
make_benches_two_args!(mixinners, planned_multi, {(3, 4), (4, 3), (3, 5), (5, 3), (3, 7), (7, 3),  (3,13), (13,3), (3,31), (31,3)});
make_benches_two_args!(mixinners, planned_multi, {(7, 31), (31, 7), (23, 31), (31, 23), (29,31), (31,29)});
make_benches_two_args!(mixinners, planned_multi, {(31, 127), (127, 31), (31, 233), (233, 31), (127, 233), (233, 127)});
make_benches_two_args!(mixinners, planned_multi, {(127, 1031), (1031, 127), (1031, 2003), (2003, 1031), (2003, 2048), (2048, 2003), (4093, 4096), (4096, 4093)});

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



//fn bench_power2_direct(b: &mut Bencher, len: usize) {
//    let fft = Arc::new(Radix4::new(len, FftDirection::Forward)) as Arc<dyn Fft<f64>>;
//
//    let mut buffer = vec![Complex::zero(); len];
//    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len() ];
//    b.iter(|| {
//        fft.process_with_scratch(&mut buffer, &mut scratch);
//    });
//}
//
//fn bench_power2_split(b: &mut Bencher, len: usize) {
//    let zeros = len.trailing_zeros();
//    let left_zeros = zeros/ 2;
//    let right_zeros = zeros - left_zeros;
//    let left_size = 1 << left_zeros;
//    let right_size = 1 << right_zeros;
//    let left_fft = Arc::new(Radix4::new(left_size, FftDirection::Forward)) as Arc<dyn Fft<f64>>;
//    let right_fft = Arc::new(Radix4::new(right_size, FftDirection::Forward)) as Arc<dyn Fft<f64>>;
//    let fft = Arc::new(MixedRadix::new(left_fft, right_fft));
//
//    let mut buffer = vec![Complex::zero(); len];
//    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len() ];
//    b.iter(|| {
//        fft.process_with_scratch(&mut buffer, &mut scratch);
//    });
//}
//make_benches!(power2_direct, power2_direct, {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304});
//make_benches!(power2_split, power2_split, {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304});