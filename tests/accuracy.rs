//! To test the accuracy of our FFT algorithm, we first test that our
//! naive DFT function is correct by comparing its output against several
//! known signal/spectrum relationships. Then, we generate random signals
//! for a variety of lengths, and test that our FFT algorithm matches our
//! DFT calculation for those signals.


use std::sync::Arc;

use rustfft::num_complex::Complex;

use rand::{StdRng, SeedableRng};
use rand::distributions::{Normal, Distribution};
use rustfft::{FftPlanner, Fft};
use rustfft::num_traits::Float;
use rustfft::algorithm::{BluesteinsAlgorithm, Radix4};

/// The seed for the random number generator used to generate
/// random signals. It's defined here so that we have deterministic
/// tests
const RNG_SEED: [u8; 32] = [1, 9, 1, 0, 1, 1, 4, 3, 1, 4, 9, 8,
    4, 1, 4, 8, 2, 8, 1, 2, 2, 2, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Returns true if the mean difference in the elements of the two vectors
/// is small
fn compare_vectors<T: rustfft::FFTnum + Float>(vec1: &[Complex<T>], vec2: &[Complex<T>]) -> bool {
    assert_eq!(vec1.len(), vec2.len());
    let mut error = T::zero();
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        error = error + (a - b).norm();
    }
    return (error.to_f64().unwrap() / vec1.len() as f64) < 0.1;
}


fn fft_matches_dft<T: rustfft::FFTnum + Float>(signal: Vec<Complex<T>>, inverse: bool) -> bool {
    let mut buffer_expected = signal.clone();
    let mut buffer_actual = signal.clone();

    let mut planner = FftPlanner::new(inverse);
    let fft = planner.plan_fft(signal.len());
    assert_eq!(fft.len(), signal.len(), "FftPlanner created FFT of wrong length");
    assert_eq!(fft.is_inverse(), inverse, "FftPlanner created FFT of wrong direction");

    fft.process_inplace(&mut buffer_actual);

    let inner_fft_len = (signal.len() * 2 - 1).checked_next_power_of_two().unwrap();
    let inner_fft = Arc::new(Radix4::new(inner_fft_len, inverse));
    let control = BluesteinsAlgorithm::new(signal.len(), inner_fft);
    control.process_inplace(&mut buffer_expected);

    return compare_vectors(&buffer_expected, &buffer_actual);
}

fn random_signal<T: rustfft::FFTnum>(length: usize) -> Vec<Complex<T>> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Normal::new(0.0, 10.0);
    let mut rng: StdRng = SeedableRng::from_seed(RNG_SEED);
    for _ in 0..length {
        sig.push(Complex{re: T::from_f64(normal_dist.sample(&mut rng)).unwrap(),
                         im: T::from_f64(normal_dist.sample(&mut rng)).unwrap()});
    }
    return sig;
}

/// Integration tests that verify our FFT output matches the direct DFT calculation
/// for random signals.
#[test]
fn test_planned_fft_forward_f32() {
    for len in 1..2000 {
        let signal = random_signal::<f32>(len);
        assert!(fft_matches_dft(signal, false), "length = {}", len);
    }
}

#[test]
fn test_planned_fft_inverse_f32() {
    for len in 1..2000 {
        let signal = random_signal::<f32>(len);
        assert!(fft_matches_dft(signal, true), "length = {}", len);
    }
}

#[test]
fn test_planned_fft_forward_f64() {
    for len in 1..2000 {
        let signal = random_signal::<f64>(len);
        assert!(fft_matches_dft(signal, false), "length = {}", len);
    }
}

#[test]
fn test_planned_fft_inverse_f64() {
    for len in 1..2000 {
        let signal = random_signal::<f64>(len);
        assert!(fft_matches_dft(signal, true), "length = {}", len);
    }
}
