use num_complex::Complex;
use num_traits::{Float, One, Zero};
use rand::distributions::uniform::SampleUniform;

use crate::{
    test_utils::{compare_vectors, first_diff, random_signal},
    FftDirection, FftNd, FftNum, FftPlanner,
};

// Multidimensional analogue of test_utils::check_fft_algoirithm
pub fn check_multidimensional_fft_algorithm<const D: usize, T: FftNum + Float + SampleUniform>(
    fft: &dyn FftNd<T, D>,
    shape: [usize; D],
    direction: FftDirection,
) {
    let len: usize = if D == 0 { 0 } else { shape.iter().product() };

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

    let n = 3;
    let dirty_scratch_value = Complex::one() * T::from_i32(100).unwrap();

    // set up buffers
    let reference_input = random_signal(len * n);
    let mut expected_output = reference_input.clone();
    control_fft_nd(&mut expected_output, shape, direction);

    // test process()
    {
        let mut buffer = reference_input.clone();

        fft.process(&mut buffer);

        if !compare_vectors(&expected_output, &buffer) {
            panic!(
                "process() failed, length = {}, direction = {}, first diff = {:?}",
                len,
                direction,
                first_diff(&expected_output, &buffer)
            );
        }
    }

    // test process_with_scratch()
    {
        let mut buffer = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_inplace_scratch_len()];

        fft.process_with_scratch(&mut buffer, &mut scratch);

        assert!(
            compare_vectors(&expected_output, &buffer),
            "process_with_scratch() failed, length = {}, direction = {}",
            len,
            direction
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            buffer.copy_from_slice(&reference_input);

            fft.process_with_scratch(&mut buffer, &mut scratch);

            assert!(compare_vectors(&expected_output, &buffer), "process_with_scratch() failed the 'dirty scratch' test, length = {}, direction = {}", len, direction);
        }
    }

    // test process_outofplace_with_scratch()
    {
        let mut input = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_outofplace_scratch_len()];
        let mut output = vec![Zero::zero(); n * len];

        fft.process_outofplace_with_scratch(&mut input, &mut output, &mut scratch);

        assert!(
            compare_vectors(&expected_output, &output),
            "process_outofplace_with_scratch() failed, length = {}, direction = {}",
            len,
            direction
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            input.copy_from_slice(&reference_input);

            fft.process_outofplace_with_scratch(&mut input, &mut output, &mut scratch);

            assert!(
                compare_vectors(&expected_output, &output),
                "process_outofplace_with_scratch() failed the 'dirty scratch' test, length = {}, direction = {}",
                len,
                direction
            );
        }
    }

    // test process_immutable_with_scratch()
    {
        let mut input = reference_input.clone();
        let mut scratch = vec![Zero::zero(); fft.get_immutable_scratch_len()];
        let mut output = vec![Zero::zero(); n * len];

        fft.process_immutable_with_scratch(&input, &mut output, &mut scratch);

        assert!(
            compare_vectors(&expected_output, &output),
            "process_immutable_with_scratch() failed, length = {}, direction = {}",
            len,
            direction
        );

        // make sure this algorithm works correctly with dirty scratch
        if scratch.len() > 0 {
            for item in scratch.iter_mut() {
                *item = dirty_scratch_value;
            }
            input.copy_from_slice(&reference_input);

            fft.process_immutable_with_scratch(&input, &mut output, &mut scratch);

            assert!(
                compare_vectors(&expected_output, &output),
                "process_immutable_with_scratch() failed the 'dirty scratch' test, length = {}, direction = {}",
            len,
            direction
        );
        }
    }
}

// Computes a multidimensional FFT uses the simplest possible strided access. Used to test faster/more sophisticated algorithms.
pub fn control_fft_nd<const D: usize, T: FftNum>(
    buffer: &mut [Complex<T>],
    shape: [usize; D],
    direction: FftDirection,
) {
    let len = if D == 0 { 0 } else { shape.iter().product() };
    if len == 0 {
        return;
    }

    let mut planner = FftPlanner::new();
    let mut local_buffer = Vec::with_capacity(len);

    for chunk in buffer.chunks_exact_mut(len) {
        let mut stride = 1;
        for dimension in shape.iter().rev() {
            let fft = planner.plan_fft(*dimension, direction);

            // Copy strided data into a locally contiguous buffer
            local_buffer.clear();
            for base in 0..stride {
                for i in (base..chunk.len()).step_by(stride) {
                    local_buffer.push(chunk[i]);
                }
            }

            // execute FFTs for this dimension
            fft.process(&mut local_buffer);

            // copy data back out to strided positions
            let mut j = 0;
            for base in 0..stride {
                for i in (base..chunk.len()).step_by(stride) {
                    chunk[i] = local_buffer[j];
                    j += 1;
                }
            }
            stride = stride * dimension;
        }
    }
}
