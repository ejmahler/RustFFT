//! The FCMA backend is a copy of the Neon one with the complex arithmetic swapped out, so the two
//! must agree. The FCMA instructions fuse operations that the Neon code does separately, which
//! rounds slightly differently, so this compares within a tolerance rather than exactly.
//!
//! This test only runs on AArch64 with the "fcma" feature, and it needs a CPU implementing
//! ARMv8.3-A to run at all.

#![cfg(all(target_arch = "aarch64", feature = "fcma"))]

use num_traits::Float;
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rand::{rngs::StdRng, SeedableRng};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FftDirection, FftNum, FftPlannerFcma, FftPlannerNeon};

const RNG_SEED: [u8; 32] = [
    1, 9, 1, 0, 1, 1, 4, 3, 1, 4, 9, 8, 4, 1, 4, 8, 2, 8, 1, 2, 2, 2, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

fn random_signal<T: FftNum + SampleUniform>(length: usize) -> Vec<Complex<T>> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist: Uniform<T> = Uniform::new(T::zero(), T::one());
    let mut rng: StdRng = SeedableRng::from_seed(RNG_SEED);
    for _ in 0..length {
        sig.push(Complex {
            re: normal_dist.sample(&mut rng),
            im: normal_dist.sample(&mut rng),
        });
    }
    sig
}

/// The mean difference between the two, relative to the mean magnitude of the reference.
fn relative_error<T: FftNum + Float>(fcma: &[Complex<T>], neon: &[Complex<T>]) -> T {
    assert_eq!(fcma.len(), neon.len());
    let mut sum_error = T::zero();
    let mut sum_magnitude = T::zero();
    for (a, b) in fcma.iter().zip(neon.iter()) {
        sum_error = sum_error + (a - b).norm();
        sum_magnitude = sum_magnitude + b.norm();
    }
    if sum_magnitude == T::zero() {
        return sum_error;
    }
    sum_error / sum_magnitude
}

fn compare_planners<T: FftNum + Float + SampleUniform>(
    len: usize,
    direction: FftDirection,
    tolerance: T,
) {
    let mut fcma_planner = FftPlannerFcma::new().unwrap();
    let mut neon_planner = FftPlannerNeon::new().unwrap();

    let fcma_fft = fcma_planner.plan_fft(len, direction);
    let neon_fft = neon_planner.plan_fft(len, direction);

    let signal = random_signal::<T>(len);

    let mut fcma_buffer = signal.clone();
    let mut fcma_scratch = vec![Complex::zero(); fcma_fft.get_inplace_scratch_len()];
    fcma_fft.process_with_scratch(&mut fcma_buffer, &mut fcma_scratch);

    let mut neon_buffer = signal;
    let mut neon_scratch = vec![Complex::zero(); neon_fft.get_inplace_scratch_len()];
    neon_fft.process_with_scratch(&mut neon_buffer, &mut neon_scratch);

    let error = relative_error(&fcma_buffer, &neon_buffer);
    assert!(
        error < tolerance,
        "len {len}, {direction:?}: relative error {error:?} is above the tolerance {tolerance:?}"
    );
}

#[test]
fn fcma_matches_neon_f32() {
    for len in 1..=256 {
        compare_planners::<f32>(len, FftDirection::Forward, 1e-5);
        compare_planners::<f32>(len, FftDirection::Inverse, 1e-5);
    }
    for len in [1000, 1024, 1031, 2048, 4096, 5000, 65536] {
        compare_planners::<f32>(len, FftDirection::Forward, 1e-5);
        compare_planners::<f32>(len, FftDirection::Inverse, 1e-5);
    }
}

#[test]
fn fcma_matches_neon_f64() {
    for len in 1..=256 {
        compare_planners::<f64>(len, FftDirection::Forward, 1e-13);
        compare_planners::<f64>(len, FftDirection::Inverse, 1e-13);
    }
    for len in [1000, 1024, 1031, 2048, 4096, 5000, 65536] {
        compare_planners::<f64>(len, FftDirection::Forward, 1e-13);
        compare_planners::<f64>(len, FftDirection::Inverse, 1e-13);
    }
}
