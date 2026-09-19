use std::sync::Arc;

use num_complex::Complex;

use crate::{
    fft_helper::{fft_helper_immut, fft_helper_inplace, fft_helper_outofplace},
    Direction, Fft, FftDirection, FftNd, FftNum, Length,
};

struct FftDimension<T> {
    fft: Arc<dyn Fft<T>>,
    len: usize,
    transpose_height: usize,
}

/// Computes multidimensional FFTs by doing one transpose per dimension to make that dimension contiguous, then computing contiguous FFTs for that dimension
pub struct FftNdTranspose<T: FftNum, const DIMENSIONS: usize> {
    ffts: [FftDimension<T>; DIMENSIONS],
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,
    direction: FftDirection,
}

impl<T: FftNum, const DIMENSIONS: usize> FftNdTranspose<T, DIMENSIONS> {
    /// Constructs a new FftMultiDimension instance which will compute multidimensional FFTs. Dimensions are defined by the profidved `ffts` array.
    pub fn new(len: usize, direction: FftDirection, ffts: [Arc<dyn Fft<T>>; DIMENSIONS]) -> Self {
        // Internal sanity check: Verify that the length and direction are internally consistent
        if DIMENSIONS > 0 {
            let mut computed_len: usize = 1;
            for fft in ffts.iter() {
                assert_eq!(fft.fft_direction(), direction);
                computed_len = computed_len
                    .checked_mul(fft.len())
                    .expect("NdFftTranspose length overflow");
            }
            assert_eq!(computed_len, len);
        } else {
            assert_eq!(len, 0);
        };

        // Compute how much scratch we need for each of our code paths.
        let mut inplace_scratch_len = 0;
        let mut outofplace_scratch_len = 0;
        let mut immut_scratch_len = 0;

        if DIMENSIONS == 1 {
            // When dimension is 1, we just forward directly to our one child FFT
            inplace_scratch_len = ffts[0].get_inplace_scratch_len();
            outofplace_scratch_len = ffts[0].get_outofplace_scratch_len();
            immut_scratch_len = ffts[0].get_immutable_scratch_len();
        } else if DIMENSIONS > 1 {
            for (i, dimension) in ffts.iter().enumerate() {
                // todo: all of these scratch requirements can be reduced
                let dimension_inplace = dimension.get_inplace_scratch_len();
                let dimension_outofplace = dimension.get_outofplace_scratch_len();
                let dimension_immut = dimension.get_immutable_scratch_len();

                // In place scratch len: the final FFT is computed differently based on how many dimensions we have
                if i == ffts.len() - 1 {
                    if DIMENSIONS % 2 == 0 {
                        inplace_scratch_len = inplace_scratch_len.max(len + dimension_inplace);
                    } else {
                        inplace_scratch_len = inplace_scratch_len.max(len + dimension_outofplace);
                    }
                } else {
                    inplace_scratch_len = inplace_scratch_len.max(len + dimension_inplace);
                }

                // Out of place scratch len: the final FFT is computed differently based on how many dimensions we have
                if i == ffts.len() - 1 {
                    if DIMENSIONS % 2 == 0 {
                        outofplace_scratch_len = outofplace_scratch_len.max(dimension_outofplace);
                    } else {
                        outofplace_scratch_len =
                            outofplace_scratch_len.max(len + dimension_inplace);
                    }
                } else {
                    outofplace_scratch_len = outofplace_scratch_len.max(len + dimension_inplace);
                }

                // Immutable scratch is different for the final FFT
                if i == ffts.len() - 1 {
                    immut_scratch_len = immut_scratch_len.max(len + dimension_immut);
                } else {
                    immut_scratch_len = immut_scratch_len.max(len + dimension_inplace);
                }
            }
        }
        Self {
            ffts: ffts.map(|fft| {
                let d = fft.len();
                FftDimension {
                    fft,
                    len: d,
                    transpose_height: if d > 0 { len / d } else { 0 },
                }
            }),
            len,
            inplace_scratch_len,
            outofplace_scratch_len,
            immut_scratch_len,
            direction,
        }
    }
}
impl<T: FftNum, const DIMENSIONS: usize> Length for FftNdTranspose<T, DIMENSIONS> {
    fn len(&self) -> usize {
        self.len
    }
}
impl<T: FftNum, const DIMENSIONS: usize> Direction for FftNdTranspose<T, DIMENSIONS> {
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

impl<T: FftNum, const DIMENSIONS: usize> FftNd<T, DIMENSIONS> for FftNdTranspose<T, DIMENSIONS> {
    /// Returns FFT length of each dimension of this multidimensional FFT
    fn shape(&self) -> [usize; DIMENSIONS] {
        self.ffts.each_ref().map(|fft| fft.len)
    }
    fn get_inplace_scratch_len(&self) -> usize {
        self.inplace_scratch_len
    }

    fn get_outofplace_scratch_len(&self) -> usize {
        self.outofplace_scratch_len
    }

    fn get_immutable_scratch_len(&self) -> usize {
        self.immut_scratch_len
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        if DIMENSIONS == 0 || self.len == 0 {
            return;
        } else if DIMENSIONS == 1 {
            return self.ffts[0].fft.process_with_scratch(buffer, scratch);
        }

        fft_helper_inplace(
            buffer,
            scratch,
            self.len(),
            self.get_inplace_scratch_len(),
            |chunk, scratch| {
                let (scratch, inner_scratch) = scratch.split_at_mut(self.len());
                let (last_fft, other_ffts) = self.ffts.split_last().unwrap();

                // Our main loop bounces data back and forth between chunk and scratch. We want it to end up in chunk,
                // So our first FFT needs to initialize the bouncing into the correct place, which differs based on how many total FFTs there are to do
                let (mut step_input, mut step_output) = if DIMENSIONS % 2 == 0 {
                    last_fft.fft.process_with_scratch(chunk, inner_scratch);
                    (chunk, scratch)
                } else {
                    last_fft
                        .fft
                        .process_outofplace_with_scratch(chunk, scratch, inner_scratch);
                    (scratch, chunk)
                };

                transpose::transpose(
                    step_input,
                    step_output,
                    last_fft.len,
                    last_fft.transpose_height,
                );

                // Execute the remaining FFTs, bouncing the data back and forth between output and scratch
                for inner_fft in other_ffts.iter().rev() {
                    (step_input, step_output) = (step_output, step_input);
                    inner_fft
                        .fft
                        .process_with_scratch(step_input, inner_scratch);
                    transpose::transpose(
                        step_input,
                        step_output,
                        inner_fft.len,
                        inner_fft.transpose_height,
                    );
                }
            },
        );
    }

    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        if DIMENSIONS == 0 || self.len == 0 {
            return;
        } else if DIMENSIONS == 1 {
            return self.ffts[0]
                .fft
                .process_outofplace_with_scratch(input, output, scratch);
        }

        fft_helper_outofplace(
            input,
            output,
            scratch,
            self.len(),
            self.get_outofplace_scratch_len(),
            |chunk_in, chunk_out, scratch| {
                let (last_fft, other_ffts) = self.ffts.split_last().unwrap();

                // Our main loop bounces data back and forth between chunk_in and chunk_out. We want it to end up in chunk_out,
                // So our first FFT needs to initialize the bouncing into the correct place, which differs based on how many total FFTs there are to do
                let (mut step_input, mut step_output) = if DIMENSIONS % 2 == 0 {
                    last_fft
                        .fft
                        .process_outofplace_with_scratch(chunk_in, chunk_out, scratch);
                    (chunk_out, chunk_in)
                } else {
                    last_fft.fft.process_with_scratch(chunk_in, scratch);
                    (chunk_in, chunk_out)
                };
                transpose::transpose(
                    step_input,
                    step_output,
                    last_fft.len,
                    last_fft.transpose_height,
                );

                // Execute the remaining FFTs, bouncing the data back and forth between output and scratch
                for inner_fft in other_ffts.iter().rev() {
                    (step_input, step_output) = (step_output, step_input);
                    inner_fft.fft.process_with_scratch(step_input, scratch);
                    transpose::transpose(
                        step_input,
                        step_output,
                        inner_fft.len,
                        inner_fft.transpose_height,
                    );
                }
            },
        );
    }

    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        if DIMENSIONS == 0 || self.len == 0 {
            return;
        } else if DIMENSIONS == 1 {
            return self.ffts[0]
                .fft
                .process_immutable_with_scratch(input, output, scratch);
        }

        fft_helper_immut(
            input,
            output,
            scratch,
            self.len(),
            self.get_immutable_scratch_len(),
            |chunk_in, chunk_out, scratch| {
                let (scratch, inner_scratch) = scratch.split_at_mut(self.len());
                let (last_fft, other_ffts) = self.ffts.split_last().unwrap();

                // Our main loop bounces data back and forth between scratch and chunk_out. We want it to end up in chunk_out,
                // So we want to initialize the bouncing to the correct spot so that the data is in the right spot when the final bounce finishes.
                let (mut step_input, mut step_output) = if DIMENSIONS % 2 == 0 {
                    (chunk_out, scratch)
                } else {
                    (scratch, chunk_out)
                };

                // Execute the first FFT and the first transpose.
                last_fft
                    .fft
                    .process_immutable_with_scratch(chunk_in, step_input, inner_scratch);
                transpose::transpose(
                    step_input,
                    step_output,
                    last_fft.len,
                    last_fft.transpose_height,
                );

                // Execute the remaining FFTs, bouncing the data back and forth between output and scratch
                for inner_fft in other_ffts.iter().rev() {
                    (step_input, step_output) = (step_output, step_input);
                    inner_fft
                        .fft
                        .process_with_scratch(step_input, inner_scratch);
                    transpose::transpose(
                        step_input,
                        step_output,
                        inner_fft.len,
                        inner_fft.transpose_height,
                    );
                }
            },
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use std::sync::Arc;

    use num_complex::Complex;
    use num_traits::Zero;

    use crate::{
        multidimensional::{
            fft_nd_transpose::FftNdTranspose,
            known_test_data::{self, KnownTestData},
            multidimensional_test_utils::{check_multidimensional_fft_algorithm, control_fft_nd},
        },
        test_utils::{compare_vectors, first_diff, BigScratchAlgorithm, InPlaceOnlyAlgorithm},
        Fft, FftDirection, FftNd, FftPlanner, Length,
    };

    #[test]
    fn test_multidimensional_known_values_2d() {
        for t in known_test_data::known_test_data_2d() {
            test_known_values(t);
        }
    }

    #[test]
    fn test_multidimensional_known_values_3d() {
        for t in known_test_data::known_test_data_3d() {
            test_known_values(t);
        }
    }

    #[test]
    fn test_multidimensional_known_values_4d() {
        for t in known_test_data::known_test_data_4d() {
            test_known_values(t);
        }
    }

    fn test_known_values<const D: usize>(t: KnownTestData<D>) {
        // The known values tests accomplish 2 goals:
        // - They establish a baseline trust that our algorithms work, without having to worry about comparison against another algorithm we devloped ourselves
        // - They verify that our "control algorithm" also works, building trust that it will also give correct answers at other sizes.

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_multidimensional(t.shape, FftDirection::Forward);
        let inverse_fft = planner.plan_fft_multidimensional(t.shape, FftDirection::Inverse);

        // Test control_fft_nd()
        {
            let mut buffer = t.expected_in.clone();
            control_fft_nd(&mut buffer, t.shape, FftDirection::Forward);

            if !compare_vectors(&t.expected_out, &buffer) {
                panic!(
                    "control_fft_nd() failed, shape = {:?}, first diff = {:?}",
                    t.shape,
                    first_diff(&t.expected_out, &buffer)
                );
            }

            // verify that the inverse gets us back to where we started
            control_fft_nd(&mut buffer, t.shape, FftDirection::Inverse);
            for e in &mut buffer {
                *e = *e / fft.len() as f32;
            }

            assert!(
                compare_vectors(&t.expected_in, &buffer),
                "Expected out: {:?}, actual out: {:?}",
                t.expected_in,
                buffer
            );
        }

        // test process()
        {
            let mut buffer = t.expected_in.clone();
            fft.process(&mut buffer);

            if !compare_vectors(&t.expected_out, &buffer) {
                panic!(
                    "process() failed, shape = {:?}, first diff = {:?}",
                    t.shape,
                    first_diff(&t.expected_out, &buffer)
                );
            }

            // verify that the inverse gets us back to where we started
            inverse_fft.process(&mut buffer);
            for e in &mut buffer {
                *e = *e / fft.len() as f32;
            }

            assert!(
                compare_vectors(&t.expected_in, &buffer),
                "Expected out: {:?}, actual out: {:?}",
                t.expected_in,
                buffer
            );
        }

        // test process_with_scratch()
        {
            let mut buffer = t.expected_in.clone();
            let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
            fft.process_with_scratch(&mut buffer, &mut scratch);

            if !compare_vectors(&t.expected_out, &buffer) {
                panic!(
                    "process() failed, shape = {:?}, first diff = {:?}",
                    t.shape,
                    first_diff(&t.expected_out, &buffer)
                );
            }

            // verify that the inverse gets us back to where we started
            inverse_fft.process_with_scratch(&mut buffer, &mut scratch);
            for e in &mut buffer {
                *e = *e / fft.len() as f32;
            }

            assert!(
                compare_vectors(&t.expected_in, &buffer),
                "Expected out: {:?}, actual out: {:?}",
                t.expected_in,
                buffer
            );
        }

        // test process_outofplace_with_scratch()
        {
            let mut buffer_in = t.expected_in.clone();
            let mut buffer_out = t.expected_in.clone();
            let mut scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
            fft.process_outofplace_with_scratch(&mut buffer_in, &mut buffer_out, &mut scratch);

            if !compare_vectors(&t.expected_out, &buffer_out) {
                panic!(
                    "process() failed, shape = {:?}, first diff = {:?}",
                    t.shape,
                    first_diff(&t.expected_out, &buffer_out)
                );
            }

            // verify that the inverse gets us back to where we started
            inverse_fft.process_outofplace_with_scratch(
                &mut buffer_out,
                &mut buffer_in,
                &mut scratch,
            );
            for e in &mut buffer_in {
                *e = *e / fft.len() as f32;
            }

            assert!(
                compare_vectors(&t.expected_in, &buffer_in),
                "Expected out: {:?}, actual out: {:?}",
                t.expected_in,
                buffer_in
            );
        }

        // test process_immutable_with_scratch()
        {
            let mut buffer_in = t.expected_in.clone();
            let mut buffer_out = t.expected_in.clone();
            let mut scratch = vec![Complex::zero(); fft.get_immutable_scratch_len()];
            fft.process_immutable_with_scratch(&buffer_in, &mut buffer_out, &mut scratch);

            if !compare_vectors(&t.expected_out, &buffer_out) {
                panic!(
                    "process() failed, shape = {:?}, first diff = {:?}",
                    t.shape,
                    first_diff(&t.expected_out, &buffer_out)
                );
            }

            // verify that the inverse gets us back to where we started
            inverse_fft.process_immutable_with_scratch(&buffer_out, &mut buffer_in, &mut scratch);
            for e in &mut buffer_in {
                *e = *e / fft.len() as f32;
            }

            assert!(
                compare_vectors(&t.expected_in, &buffer_in),
                "Expected out: {:?}, actual out: {:?}",
                t.expected_in,
                buffer_in
            );
        }
    }

    fn test_multidimensional_procedural_nd<const D: usize>(
        planner: &mut FftPlanner<f32>,
        shape: [usize; D],
    ) {
        let len = if D == 0 { 0 } else { shape.iter().product() };

        // Forward FFT
        let fft_forward = FftNdTranspose::new(
            len,
            FftDirection::Forward,
            shape.map(|s| planner.plan_fft(s, FftDirection::Forward)),
        );
        check_multidimensional_fft_algorithm(&fft_forward, shape, FftDirection::Forward);

        // Inverse FFT
        let fft_inverse = FftNdTranspose::new(
            len,
            FftDirection::Inverse,
            shape.map(|s| planner.plan_fft(s, FftDirection::Inverse)),
        );
        check_multidimensional_fft_algorithm(&fft_inverse, shape, FftDirection::Inverse);
    }

    #[test]
    fn test_multidimensional_procedural_0d() {
        let mut planner = FftPlanner::new();

        // There's no reason to create a 0d multimensional FFT, but that doesn't mean it shouldn't work if someone does
        test_multidimensional_procedural_nd(&mut planner, []);
    }

    #[test]
    fn test_multidimensional_procedural_1d() {
        let mut planner = FftPlanner::<f32>::new();

        // There's no reason to create a 1d multimensional FFT, but that doesn't mean it shouldn't work if someone does
        for a in 0..10 {
            test_multidimensional_procedural_nd(&mut planner, [a]);
        }
    }

    #[test]
    fn test_multidimensional_procedural_2d() {
        let mut planner = FftPlanner::<f32>::new();

        for a in 0..10 {
            for b in 0..10 {
                test_multidimensional_procedural_nd(&mut planner, [a, b]);
            }
        }
    }

    #[test]
    fn test_multidimensional_procedural_3d() {
        let mut planner = FftPlanner::<f32>::new();

        for a in 0..8 {
            for b in 0..8 {
                for c in 0..8 {
                    test_multidimensional_procedural_nd(&mut planner, [a, b, c]);
                }
            }
        }
    }

    #[test]
    fn test_multidimensional_procedural_4d() {
        let mut planner = FftPlanner::<f32>::new();

        for a in 0..5 {
            for b in 0..5 {
                for c in 0..5 {
                    for d in 0..5 {
                        test_multidimensional_procedural_nd(&mut planner, [a, b, c, d]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_multidimensional_procedural_5d() {
        let mut planner = FftPlanner::<f32>::new();

        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        for e in 0..4 {
                            test_multidimensional_procedural_nd(&mut planner, [a, b, c, d, e]);
                        }
                    }
                }
            }
        }
    }

    fn test_multidimensional_inner_scratch_nd<const D: usize>(ffts: [Arc<dyn Fft<f32>>; D]) {
        const { assert!(D > 0) };
        let fft = FftNdTranspose::new(
            ffts.iter().map(|f| f.len()).product(),
            ffts[0].fft_direction(),
            ffts,
        );

        let mut inplace_buffer = vec![Complex::zero(); fft.len()];
        let mut inplace_scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
        fft.process_with_scratch(&mut inplace_buffer, &mut inplace_scratch);

        let mut outofplace_input = vec![Complex::zero(); fft.len()];
        let mut outofplace_output = vec![Complex::zero(); fft.len()];
        let mut outofplace_scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
        fft.process_outofplace_with_scratch(
            &mut outofplace_input,
            &mut outofplace_output,
            &mut outofplace_scratch,
        );

        let immut_input = vec![Complex::zero(); fft.len()];
        let mut immut_output = vec![Complex::zero(); fft.len()];
        let mut immut_scratch = vec![Complex::zero(); fft.get_immutable_scratch_len()];
        fft.process_immutable_with_scratch(&immut_input, &mut immut_output, &mut immut_scratch);
    }

    // Verify that FftNdTranspose algorithm correctly provides scratch space to inner FFTs
    #[test]
    fn test_multidimensional_inner_scratch() {
        // This test can go through a very bad combinatoric explosion if we aren't careful. So we're going to choose our test data carefully to avoid that explosion
        // Specifically, we're going to bake in knoweldge that the first d - 1 inner ffts of each n-d transpose fft DO NOT TOUCH the out of place or immutable code paths
        // And we'll use InPlaceOnlyAlgorithm for those FFTs, so that if that changes we'll know
        let fft_lengths = [1, 3, 9];
        let direction = FftDirection::Forward;

        let mut inner_ffts_inplace = Vec::new();
        let mut inner_ffts_any = Vec::new();

        for &len in &fft_lengths {
            let scratch_lengths = [0, len, 25];
            for &inplace_scratch in &scratch_lengths {
                inner_ffts_inplace.push(Arc::new(InPlaceOnlyAlgorithm {
                    len,
                    inplace_scratch,
                    direction,
                }) as Arc<dyn Fft<f32>>);
                for &outofplace_scratch in &scratch_lengths {
                    for &immut_scratch in &scratch_lengths {
                        inner_ffts_any.push(Arc::new(BigScratchAlgorithm {
                            len,
                            inplace_scratch,
                            outofplace_scratch,
                            immut_scratch,
                            direction,
                        }) as Arc<dyn Fft<f32>>);
                    }
                }
            }
        }

        // 1d
        for a in inner_ffts_any.iter() {
            test_multidimensional_inner_scratch_nd([a].map(Arc::clone));
        }

        // 2d
        for a in inner_ffts_inplace.iter() {
            for b in inner_ffts_any.iter() {
                test_multidimensional_inner_scratch_nd([a, b].map(Arc::clone));
            }
        }

        // 3d
        for a in inner_ffts_inplace.iter() {
            for b in inner_ffts_inplace.iter() {
                for c in inner_ffts_any.iter() {
                    test_multidimensional_inner_scratch_nd([a, b, c].map(Arc::clone));
                }
            }
        }
    }
}
