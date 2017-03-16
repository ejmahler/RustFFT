use num::Complex;

use rand::{StdRng, SeedableRng};
use rand::distributions::{Normal, IndependentSample};

/// The seed for the random number generator used to generate
/// random signals. It's defined here so that we have deterministic
/// tests
const RNG_SEED: [usize; 5] = [1910, 11431, 4984, 14828, 12226];

pub fn random_signal(length: usize) -> Vec<Complex<f32>> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Normal::new(0.0, 10.0);
    let mut rng: StdRng = SeedableRng::from_seed(&RNG_SEED[..]);
    for _ in 0..length {
        sig.push(Complex{re: (normal_dist.ind_sample(&mut rng) as f32),
                         im: (normal_dist.ind_sample(&mut rng) as f32)});
    }
    return sig;
}

pub fn compare_vectors(vec1: &[Complex<f32>], vec2: &[Complex<f32>]) -> bool {
    assert_eq!(vec1.len(), vec2.len());
    let mut sse = 0f32;
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        sse = sse + (a - b).norm();
    }
    return (sse / vec1.len() as f32) < 0.1f32;
}