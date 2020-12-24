//! To test the accuracy of our FFT algorithm, we first test that our
//! naive DFT function is correct by comparing its output against several
//! known signal/spectrum relationships. Then, we generate random signals
//! for a variety of lengths, and test that our FFT algorithm matches our
//! DFT calculation for those signals.

use std::sync::Arc;

use num_traits::Float;
use rustfft::num_traits::Zero;
use rustfft::{
    algorithm::{Bluesteins, Radix4},
    num_complex::Complex,
    FFTnum, FftPlanner, Fft,
};

use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rand::{rngs::StdRng, SeedableRng};

/// The seed for the random number generator used to generate
/// random signals. It's defined here so that we have deterministic
/// tests
const RNG_SEED: [u8; 32] = [
    1, 9, 1, 0, 1, 1, 4, 3, 1, 4, 9, 8, 4, 1, 4, 8, 2, 8, 1, 2, 2, 2, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

/// Returns true if the mean difference in the elements of the two vectors
/// is small
fn compare_vectors<T: FFTnum + Float>(vec1: &[Complex<T>], vec2: &[Complex<T>]) -> bool {
    assert_eq!(vec1.len(), vec2.len());
    let mut sse = T::zero();
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        sse = sse + (a - b).norm();
    }
    return (sse / T::from_usize(vec1.len()).unwrap()) < T::from_f32(0.1).unwrap();
}

fn fft_matches_control<T: FFTnum + Float>(control: Arc<dyn Fft<T>>, input: &[Complex<T>]) -> bool {
    let mut control_input = input.to_vec();
    let mut test_input = input.to_vec();

    let mut control_output = vec![Zero::zero(); control.len()];
    let mut test_output = vec![Zero::zero(); control.len()];

    let mut planner = FftPlanner::new(control.is_inverse());
    let fft = planner.plan_fft(control.len());
    assert_eq!(
        fft.len(),
        control.len(),
        "FFTplanner created FFT of wrong length"
    );
    assert_eq!(
        fft.is_inverse(),
        control.is_inverse(),
        "FFTplanner created FFT of wrong direction"
    );

    control.process(&mut control_input, &mut control_output);
    fft.process(&mut test_input, &mut test_output);

    return compare_vectors(&test_output, &control_output);
}

fn random_signal<T: FFTnum + SampleUniform>(length: usize) -> Vec<Complex<T>> {
    let mut sig = Vec::with_capacity(length);
    let dist: Uniform<T> = Uniform::new(T::zero(), T::from_f64(10.0).unwrap());
    let mut rng: StdRng = SeedableRng::from_seed(RNG_SEED);
    for _ in 0..length {
        sig.push(Complex {
            re: (dist.sample(&mut rng)),
            im: (dist.sample(&mut rng)),
        });
    }
    return sig;
}

// A cache that makes setup for integration tests faster
struct ControlCache<T: FFTnum> {
    fft_cache: Vec<Arc<dyn Fft<T>>>,
}
impl<T: FFTnum> ControlCache<T> {
    pub fn new(max_outer_len: usize, inverse: bool) -> Self {
        let max_inner_len = (max_outer_len * 2 - 1).checked_next_power_of_two().unwrap();
        let max_power = max_inner_len.trailing_zeros() as usize;

        Self {
            fft_cache: (0..=max_power)
                .map(|i| {
                    let len = 1 << i;
                    Arc::new(Radix4::new(len, inverse)) as Arc<dyn Fft<_>>
                })
                .collect(),
        }
    }

    pub fn plan_fft(&self, len: usize) -> Arc<dyn Fft<T>> {
        let inner_fft_len = (len * 2 - 1).checked_next_power_of_two().unwrap();
        let inner_fft_index = inner_fft_len.trailing_zeros() as usize;
        let inner_fft = Arc::clone(&self.fft_cache[inner_fft_index]);
        Arc::new(Bluesteins::new(len, inner_fft))
    }
}

const TEST_MAX: usize = 1001;

/// Integration tests that verify our FFT output matches the direct DFT calculation
/// for random signals.
#[test]
fn test_planned_fft_forward_f32() {
    let is_inverse = false;
    let cache: ControlCache<f32> = ControlCache::new(TEST_MAX, is_inverse);

    for len in 1..TEST_MAX {
        let control = cache.plan_fft(len);
        assert_eq!(control.len(), len);
        assert_eq!(control.is_inverse(), is_inverse);

        let signal = random_signal(len);
        assert!(fft_matches_control(control, &signal), "length = {}", len);
    }
}

#[test]
fn test_planned_fft_inverse_f32() {
    let is_inverse = true;
    let cache: ControlCache<f32> = ControlCache::new(TEST_MAX, is_inverse);

    for len in 1..TEST_MAX {
        let control = cache.plan_fft(len);
        assert_eq!(control.len(), len);
        assert_eq!(control.is_inverse(), is_inverse);

        let signal = random_signal(len);
        assert!(fft_matches_control(control, &signal), "length = {}", len);
    }
}

#[test]
fn test_planned_fft_forward_f64() {
    let is_inverse = false;
    let cache: ControlCache<f64> = ControlCache::new(TEST_MAX, is_inverse);

    for len in 1..TEST_MAX {
        let control = cache.plan_fft(len);
        assert_eq!(control.len(), len);
        assert_eq!(control.is_inverse(), is_inverse);

        let signal = random_signal(len);
        assert!(fft_matches_control(control, &signal), "length = {}", len);
    }
}

#[test]
fn test_planned_fft_inverse_f64() {
    let is_inverse = true;
    let cache: ControlCache<f64> = ControlCache::new(TEST_MAX, is_inverse);

    for len in 1..TEST_MAX {
        let control = cache.plan_fft(len);
        assert_eq!(control.len(), len);
        assert_eq!(control.is_inverse(), is_inverse);

        let signal = random_signal(len);
        assert!(fft_matches_control(control, &signal), "length = {}", len);
    }
}
