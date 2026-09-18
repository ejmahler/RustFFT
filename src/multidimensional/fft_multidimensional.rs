use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use crate::{
    fft_helper::{fft_helper_immut, fft_helper_inplace, fft_helper_outofplace},
    Fft, FftDirection, FftNum, FftPlanner,
};

struct FftDimension<T> {
    fft: Arc<dyn Fft<T>>,
    len: usize,
    transpose_height: usize,
}

/// Implementation of multi-dimensional FFTs
///
/// `FftMultiDimensional` wraps all the logic needed to compute a multi-dimensional FFT. It is generic over the number of dimensions, so FFTs can computed with arbitrary dimensionality.
///
/// To construct a `FftMultiDimensional`, call [`FftMultiDimensional::plan`](crate::FftMultiDimensional::plan). This method takes an array of dimensions, which can be any length, allowing for 2d FFTs, 3d FFTs, or even more.
///
/// Once constructed, call [`process()`](crate::FftMultiDimensional::process) to compute the FFT.
///
/// ### Examples
///
/// #### 2D FFT
/// ~~~
/// // Perform a 2d Fft of size 3x5
/// use std::sync::Arc;
/// use rustfft::{FftPlanner, FftMultiDimensional, num_complex::Complex};
///
/// let mut planner = FftPlanner::new();
/// let shape = [3, 5];
/// let fft = FftMultiDimensional::plan(&mut planner, shape);
///
/// // Since shape[1] is 5 and shape[0] is 3,
/// // we'll lay out our data so that each row has 5 elements, and there are 3 rows.
/// let mut buffer = vec![
///      11.0, 12.0, 13.0, 14.0, 15.0,
///      21.0, 22.0, 23.0, 24.0, 25.0,
///      31.0, 32.0, 33.0, 34.0, 35.0,
/// ];
///
/// fft.process(&mut buffer);
/// ~~~
///
/// #### 3D FFT
///
/// ~~~
/// // Perform a 3d Fft of size 2x3x5
/// use std::sync::Arc;
/// use rustfft::{FftPlanner, FftMultiDimensional, num_complex::Complex};
///
/// let mut planner = FftPlanner::new();
/// let shape = [2, 3, 5];
/// let fft = FftMultiDimensional::plan(&mut planner, shape);
///
/// // Since shape[2] is 5, shape[1] is 3, and shape[0] is 2,
/// // We'll lay out our data so that each row has 5 elements, each block has 3 rows, and there are 2 blocks.
/// let mut buffer = vec![
///      111.0, 112.0, 113.0, 114.0, 115.0,
///      121.0, 122.0, 123.0, 124.0, 125.0,
///      131.0, 132.0, 133.0, 134.0, 135.0,
///
///      211.0, 212.0, 213.0, 214.0, 215.0,
///      221.0, 222.0, 223.0, 224.0, 225.0,
///      231.0, 232.0, 233.0, 234.0, 235.0,
/// ];
///
/// fft.process(&mut buffer);
/// ~~~
/// ### Data Layout
///
/// Data should be provided as one contiguous buffer, stored in row-major order. The last dimension is stored contiguously, and each preceding dimension has a larger and larger stride.
/// Generally speaking, dimension `i` will have a stride of `dimensions.iter().skip(i + 1).product()`.
///
/// #### 2D FFTs
///
/// For 2D FFTs, the data is interpreted as row-major order, where `shape[1]` specifies the row width, and `shape[0]` specifies the number of rows.
///
/// #### 3D FFTs
///
/// For 3D FFTs, the data layout is a logical extension of row-major order.
/// - `shape[2]` specifies the row width. In memory, rows are stored contiguously.
/// - `shape[1]` specifies the number of rows. In memory, each row starts where the previous ends, IE a stride of `shape[2]`.
/// - `shape[0]` specifies the number of "groups of rows". In memory, each group of rows starts where the previous group ends, IE a stride of `shape[2] * shape[1]`.
pub struct FftMultiDimensional<T: FftNum, const DIMENSIONS: usize> {
    ffts: [FftDimension<T>; DIMENSIONS],
    len: usize,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,
    direction: FftDirection,
}

impl<T: FftNum, const DIMENSIONS: usize> FftMultiDimensional<T, DIMENSIONS> {
    /// Constructs a new FftMultiDimension instance which will compute multidimensional FFTs. Dimensions are defined by the profidved `ffts` array.
    pub fn new(ffts: [Arc<dyn Fft<T>>; DIMENSIONS]) -> Self {
        assert!(DIMENSIONS > 0, "Can't compute a FFT with 0 dimensions");

        let direction = ffts[0].fft_direction();
        for i in 1..DIMENSIONS {
            assert_eq!(
                direction,
                ffts[i].fft_direction(),
                "All provided FFTs must have the same direction"
            );
        }
        let mut len: usize = 1;
        for fft in ffts.iter() {
            len = len
                .checked_mul(fft.len())
                .expect("FftMultiDimension length overflow");
        }

        Self::new_with_dimensions(
            len,
            direction,
            ffts.map(|fft| {
                let d = fft.len();
                FftDimension {
                    fft,
                    len: d,
                    transpose_height: len / d,
                }
            }),
        )
    }

    /// Constructs a new FftMultiDimension instance which will compute multidimensional FFTs. Dimensions are defined by the provided `dimensions` array.
    ///
    /// Uses the provided `planner` to construct `Fft` instances corresponding to each dimension.
    pub fn plan(
        planner: &mut FftPlanner<T>,
        direction: FftDirection,
        shape: [usize; DIMENSIONS],
    ) -> Self {
        assert!(DIMENSIONS > 0, "Can't compute a FFT with 0 dimensions");

        let mut len: usize = 1;
        for d in shape {
            len = len
                .checked_mul(d)
                .expect("FftMultiDimension length overflow");
        }
        Self::new_with_dimensions(
            len,
            direction,
            shape.map(|d| FftDimension {
                fft: planner.plan_fft(d, direction),
                len: d,
                transpose_height: len / d,
            }),
        )
    }

    fn new_with_dimensions(
        len: usize,
        direction: FftDirection,
        ffts: [FftDimension<T>; DIMENSIONS],
    ) -> Self {
        // Compute how much scratch we need for each of our code paths.
        let mut inplace_scratch_len = 0;
        let mut outofplace_scratch_len = 0;
        let mut immut_scratch_len = 0;

        for (i, dimension) in ffts.iter().enumerate() {
            // todo: all of these scratch requirements can be reduced
            let dimension_inplace = dimension.fft.get_inplace_scratch_len();
            let dimension_outofplace = dimension.fft.get_outofplace_scratch_len();
            let dimension_immut = dimension.fft.get_immutable_scratch_len();

            // In place scratch len: the final FFT is computed differently based on how many dimensions we have
            if i == ffts.len() - 1 {
                if DIMENSIONS % 2 == 0 {
                    inplace_scratch_len = inplace_scratch_len.max(len + dimension_inplace);
                } else {
                    inplace_scratch_len = inplace_scratch_len.max(dimension_outofplace);
                }
            } else {
                inplace_scratch_len = inplace_scratch_len.max(len + dimension_inplace);
            }

            // Out of place scratch len: the final FFT is computed differently based on how many dimensions we have
            if i == ffts.len() - 1 {
                if DIMENSIONS % 2 == 0 {
                    outofplace_scratch_len = outofplace_scratch_len.max(dimension_outofplace);
                } else {
                    outofplace_scratch_len = outofplace_scratch_len.max(len + dimension_inplace);
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
        Self {
            ffts,
            len,
            inplace_scratch_len,
            outofplace_scratch_len,
            immut_scratch_len,
            direction,
        }
    }

    /// Returns FFT length of each dimension of this multidimensional FFT
    pub fn dimensions(&self) -> [usize; DIMENSIONS] {
        self.ffts.each_ref().map(|fft| fft.len)
    }

    /// Returns the total number of elements processed at a time by this multimensional FFT instance. Defined as the product of each of our dimensions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns FftDirection::Forward if this instance computes forward FFTs, or FftDirection::Inverse for inverse FFTs
    pub fn fft_direction(&self) -> FftDirection {
        self.direction
    }

    /// Returns the size of the scratch buffer required by `process_with_scratch`
    ///
    /// The returned value may change from one version of RustFFT to the next.
    pub fn get_inplace_scratch_len(&self) -> usize {
        self.inplace_scratch_len
    }

    /// Returns the size of the scratch buffer required by `process_outofplace_with_scratch`
    ///
    /// The returned value may change from one version of RustFFT to the next.
    pub fn get_outofplace_scratch_len(&self) -> usize {
        self.outofplace_scratch_len
    }

    /// Returns the size of the scratch buffer required by `process_immutable_with_scratch`
    ///
    /// The returned value may change from one version of RustFFT to the next.
    pub fn get_immutable_scratch_len(&self) -> usize {
        self.immut_scratch_len
    }

    /// Computes a multi-dimensional FFT in-place.
    ///
    /// Convenience method that allocates a `Vec` with the required scratch space and calls `self.process_with_scratch`.
    /// If you want to re-use that allocation across multiple FFT computations, consider calling `process_with_scratch` instead.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() % self.len_product() > 0`
    /// - `buffer.len() < self.len_product()`
    pub fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_inplace_scratch_len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    /// Divides `buffer` into chunks of size `self.len()`, and computes a multi-dimensional FFT on each chunk.
    ///
    /// Uses the `scratch` buffer as scratch space, so the contents of `scratch` should be considered garbage
    /// after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `buffer.len() % self.len() > 0`
    /// - `buffer.len() < self.len()`
    /// - `scratch.len() < self.get_inplace_scratch_len()`
    pub fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
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

    /// Divides `input` and `output` into chunks of size `self.len()`, and computes a multi-dimensional FFT on each chunk.
    ///
    /// This method uses both the `input` buffer and `scratch` buffer as scratch space, so the contents of both should be
    /// considered garbage after calling.
    ///
    /// This is a more niche way of computing a FFT. It's useful to avoid a `copy_from_slice()` if you need the output
    /// in a different buffer than the input for some reason. This happens frequently in RustFFT internals, but is probably
    /// less common among RustFFT users.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `output.len() != input.len()`
    /// - `input.len() % self.len() > 0`
    /// - `input.len() < self.len()`
    /// - `scratch.len() < self.get_outofplace_scratch_len()`
    pub fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
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

    /// Divides `input` and `output` into chunks of `self.len()`, and computes a multi-dimensional FFT on each chunk while
    /// keeping `input` untouched.
    ///
    /// This method uses the `scratch` buffer as scratch space, so the contents should be considered garbage after calling.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `output.len() ! input.len()`
    /// - `input.len() % self.len() > 0`
    /// - `input.len() < self.len()`
    /// - `scratch.len() < get_immutable_scratch_len()`
    pub fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
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
    use num_complex::Complex;
    use num_traits::Zero;

    use crate::{
        multidimensional::known_test_data::{self, KnownTestData},
        test_utils::{compare_vectors, first_diff},
        FftDirection, FftMultiDimensional, FftPlanner,
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

    // Computes a multidimensional FFT uses the simplest possible strided access. Used to test faster/more sophisticated algorithms.
    fn control_fft_nd<const D: usize>(
        buffer: &mut [Complex<f32>],
        direction: FftDirection,
        shape: [usize; D],
    ) {
        let len: usize = shape.iter().product();
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

    fn test_known_values<const D: usize>(t: KnownTestData<D>) {
        // The known values tests accomplish 2 goals:
        // - They establish a baseline trust that our algorithms work, without having to worry about comparison against another algorithm we devloped ourselves
        // - They verify that our "control algorithm" also works, building trust that it will also give correct answers at other sizes.

        let mut planner = FftPlanner::new();
        let fft = FftMultiDimensional::plan(&mut planner, FftDirection::Forward, t.shape);
        let inverse_fft = FftMultiDimensional::plan(&mut planner, FftDirection::Inverse, t.shape);

        // Test control_fft_nd()
        {
            let mut buffer = t.expected_in.clone();
            control_fft_nd(&mut buffer, FftDirection::Forward, t.shape);

            if !compare_vectors(&t.expected_out, &buffer) {
                panic!(
                    "control_fft_nd() failed, shape = {:?}, first diff = {:?}",
                    t.shape,
                    first_diff(&t.expected_out, &buffer)
                );
            }

            // verify that the inverse gets us back to where we started
            control_fft_nd(&mut buffer, FftDirection::Inverse, t.shape);
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
}
