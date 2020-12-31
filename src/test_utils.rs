use num_complex::Complex;
use num_traits::{Float, One, Zero};

use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rand::{rngs::StdRng, SeedableRng};

use crate::{Fft, FftDirection};
use crate::{algorithm::DFT, FFTnum};

/// The seed for the random number generator used to generate
/// random signals. It's defined here so that we have deterministic
/// tests
const RNG_SEED: [u8; 32] = [
    1, 9, 1, 0, 1, 1, 4, 3, 1, 4, 9, 8, 4, 1, 4, 8, 2, 8, 1, 2, 2, 2, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

pub fn random_signal<T: FFTnum + SampleUniform>(length: usize) -> Vec<Complex<T>> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist: Uniform<T> = Uniform::new(T::zero(), T::from_f32(10.0).unwrap());
    let mut rng: StdRng = SeedableRng::from_seed(RNG_SEED);
    for _ in 0..length {
        sig.push(Complex {
            re: normal_dist.sample(&mut rng),
            im: normal_dist.sample(&mut rng),
        });
    }
    return sig;
}

pub fn compare_vectors<T: FFTnum + Float>(vec1: &[Complex<T>], vec2: &[Complex<T>]) -> bool {
    assert_eq!(vec1.len(), vec2.len());
    let mut error = T::zero();
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        error = error + (a - b).norm();
    }
    return (error.to_f64().unwrap() / vec1.len() as f64) < 0.1f64;
}

#[allow(unused)]
fn transppose_diagnostic<T: FFTnum + Float>(expected: &[Complex<T>], actual: &[Complex<T>]) {
    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        if (e - a).norm().to_f32().unwrap() > 0.01 {
            if let Some(found_index) = expected
                .iter()
                .position(|&ev| (ev - a).norm().to_f32().unwrap() < 0.01)
            {
                println!("{} incorrectly contained {}", i, found_index);
            } else {
                println!("{} X", i);
            }
        }
    }
}

pub fn check_fft_algorithm<T: FFTnum + Float + SampleUniform>(
    fft: &dyn Fft<T>,
    len: usize,
    direction: FftDirection,
) {
    assert_eq!(
        fft.len(),
        len,
        "Algorithm reported incorrect size. Expected {}, got {}",
        len,
        fft.len()
    );
    assert_eq!(
        fft.fft_direction(),
        direction,
        "Algorithm reported incorrect FFT direction"
    );

    let n = 1;

    //test the forward direction
    let dft = DFT::new(len, direction);

    let dirty_scratch_value = Complex::one() * T::from_i32(100).unwrap();

    // set up buffers
    let reference_input = random_signal(len * n);
    let mut expected_input = reference_input.clone();
    let mut expected_output = vec![Zero::zero(); len * n];
    dft.process_multi(&mut expected_input, &mut expected_output, &mut []);

    // test process()
    {
        let mut input = reference_input.clone();
        let mut output = expected_output.clone();

        for (input_chunk, output_chunk) in input.chunks_mut(len).zip(output.chunks_mut(len)) {
            fft.process(input_chunk, output_chunk);
        }
        dbg!(&expected_output);
        dbg!(&output);
        transppose_diagnostic(&expected_output, &output);
        assert!(
            compare_vectors(&expected_output, &output),
            "process() failed, length = {}, direction = {}",
            len,
            direction
        );
    }

    // test process_with_scratch()
    {
        let mut input = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_out_of_place_scratch_len()];
        let mut output = expected_output.clone();

        for (input_chunk, output_chunk) in input.chunks_mut(len).zip(output.chunks_mut(len)) {
            fft.process_with_scratch(input_chunk, output_chunk, &mut scratch);
        }
        assert!(
            compare_vectors(&expected_output, &output),
            "process_with_scratch() failed, length = {}, direction = {}",
            len,
            direction
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            input.copy_from_slice(&reference_input);
            for (input_chunk, output_chunk) in input.chunks_mut(len).zip(output.chunks_mut(len)) {
                fft.process_with_scratch(input_chunk, output_chunk, &mut scratch);
            }
            assert!(
                compare_vectors(&expected_output, &output),
                "process_with_scratch() failed the 'dirty scratch' test, length = {}, direction = {}",
                len,
                direction
            );
        }
    }

    // test process_multi()
    {
        let mut input = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_out_of_place_scratch_len()];
        let mut output = expected_output.clone();

        fft.process_multi(&mut input, &mut output, &mut scratch);
        assert!(
            compare_vectors(&expected_output, &output),
            "process_multi() failed, length = {}, direction = {}",
            len,
            direction
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            input.copy_from_slice(&reference_input);
            fft.process_multi(&mut input, &mut output, &mut scratch);

            assert!(
                compare_vectors(&expected_output, &output),
                "process_multi() failed the 'dirty scratch' test, length = {}, direction = {}",
                len,
                direction
            );
        }
    }

    // test process_inplace()
    {
        let mut buffer = reference_input.clone();

        for chunk in buffer.chunks_mut(len) {
            fft.process_inplace(chunk);
        }
        assert!(
            compare_vectors(&expected_output, &buffer),
            "process_inplace() failed, length = {}, direction = {}",
            len,
            direction
        );
    }

    // test process_inplace_with_scratch()
    {
        let mut buffer = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];

        for chunk in buffer.chunks_mut(len) {
            fft.process_inplace_with_scratch(chunk, &mut scratch);
        }
        assert!(
            compare_vectors(&expected_output, &buffer),
            "process_inplace_with_scratch() failed, length = {}, direction = {}",
            len,
            direction
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            buffer.copy_from_slice(&reference_input);
            for chunk in buffer.chunks_mut(len) {
                fft.process_inplace_with_scratch(chunk, &mut scratch);
            }
            assert!(compare_vectors(&expected_output, &buffer), "process_inplace_with_scratch() failed the 'dirty scratch' test, length = {}, direction = {}", len, direction);
        }
    }

    // test process_inplace_multi()
    {
        let mut buffer = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];

        fft.process_inplace_multi(&mut buffer, &mut scratch);
        assert!(
            compare_vectors(&expected_output, &buffer),
            "process_inplace_multi() failed, length = {}, direction = {}",
            len,
            direction
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            buffer.copy_from_slice(&reference_input);
            fft.process_inplace_multi(&mut buffer, &mut scratch);

            assert!(compare_vectors(&expected_output, &buffer), "process_inplace_multi() failed the 'dirty scratch' test, length = {}, direction = {}", len, direction);
        }
    }
}
