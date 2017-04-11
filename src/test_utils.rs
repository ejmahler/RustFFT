use num_complex::Complex;
use num_traits::Zero;

use rand::{StdRng, SeedableRng};
use rand::distributions::{Normal, IndependentSample};

use algorithm::DFT;
use FFT;


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

pub fn check_fft_algorithm(fft: &FFT<f32>, size: usize, inverse: bool) {
    assert_eq!(fft.len(), size, "Algorithm reported incorrect size");
    assert_eq!(fft.is_inverse(), inverse, "Algorithm reported incorrect inverse value");

    let n = 5;

    //test the forward direction
    let dft = DFT::new(size, inverse);

    // set up buffers
    let mut expected_input = random_signal(size * n);
    let mut actual_input = expected_input.clone();
    let mut multi_input = expected_input.clone();

    let mut expected_output = vec![Zero::zero(); size * n];
    let mut actual_output = expected_output.clone();
    let mut multi_output = expected_output.clone();

    // perform the test
    dft.process_multi(&mut expected_input, &mut expected_output);
    fft.process_multi(&mut multi_input, &mut multi_output);

    for (input_chunk, output_chunk) in actual_input.chunks_mut(size).zip(actual_output.chunks_mut(size)) {
        fft.process(input_chunk, output_chunk);
    }

    assert!(compare_vectors(&expected_output, &actual_output), "process() failed, length = {}, inverse = {}", size, inverse);
    assert!(compare_vectors(&expected_output, &multi_output), "process_multi() failed, length = {}, inverse = {}", size, inverse);
}