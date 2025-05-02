use num_traits::{FromPrimitive, Signed};
use std::fmt::Debug;

/// Generic floating point number, implemented for f32 and f64
pub trait FftNum: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static {}

impl<T> FftNum for T where T: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static {}

// Prints an error raised by an in-place FFT algorithm's `process_inplace` method
// Marked cold and inline never to keep all formatting code out of the many monomorphized process_inplace methods
#[cold]
#[inline(never)]
pub fn fft_error_inplace(
    expected_len: usize,
    actual_len: usize,
    expected_scratch: usize,
    actual_scratch: usize,
) {
    assert!(
        actual_len >= expected_len,
        "Provided FFT buffer was too small. Expected len = {}, got len = {}",
        expected_len,
        actual_len
    );
    assert_eq!(
        actual_len % expected_len,
        0,
        "Input FFT buffer must be a multiple of FFT length. Expected multiple of {}, got len = {}",
        expected_len,
        actual_len
    );
    assert!(
        actual_scratch >= expected_scratch,
        "Not enough scratch space was provided. Expected scratch len >= {}, got scratch len = {}",
        expected_scratch,
        actual_scratch
    );
}

// Prints an error raised by an in-place FFT algorithm's `process_inplace` method
// Marked cold and inline never to keep all formatting code out of the many monomorphized process_inplace methods
#[cold]
#[inline(never)]
pub fn fft_error_outofplace(
    expected_len: usize,
    actual_input: usize,
    actual_output: usize,
    expected_scratch: usize,
    actual_scratch: usize,
) {
    assert_eq!(actual_input, actual_output, "Provided FFT input buffer and output buffer must have the same length. Got input.len() = {}, output.len() = {}", actual_input, actual_output);
    assert!(
        actual_input >= expected_len,
        "Provided FFT buffer was too small. Expected len = {}, got len = {}",
        expected_len,
        actual_input
    );
    assert_eq!(
        actual_input % expected_len,
        0,
        "Input FFT buffer must be a multiple of FFT length. Expected multiple of {}, got len = {}",
        expected_len,
        actual_input
    );
    assert!(
        actual_scratch >= expected_scratch,
        "Not enough scratch space was provided. Expected scratch len >= {}, got scratch len = {}",
        expected_scratch,
        actual_scratch
    );
}

macro_rules! boilerplate_fft_oop {
    ($struct_name:ident, $len_fn:expr) => {
        impl<T: FftNum> Fft<T> for $struct_name<T> {
            fn process_immutable_with_scratch(
                &self,
                input: &[Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                if self.len() == 0 {
                    return;
                }

                let required_scratch = self.get_immutable_scratch_len();
                if input.len() < self.len()
                    || output.len() != input.len()
                    || scratch.len() < required_scratch
                {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(
                        self.len(),
                        input.len(),
                        output.len(),
                        required_scratch,
                        scratch.len(),
                    );
                    return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
                }

                let result = array_utils::iter_chunks_zipped(
                    input,
                    output,
                    self.len(),
                    |in_chunk, out_chunk| self.perform_fft_immut(in_chunk, out_chunk, scratch),
                );

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                }
            }
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                if self.len() == 0 {
                    return;
                }

                let required_scratch = self.get_outofplace_scratch_len();
                if input.len() < self.len()
                    || output.len() != input.len()
                    || scratch.len() < required_scratch
                {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(
                        self.len(),
                        input.len(),
                        output.len(),
                        required_scratch,
                        scratch.len(),
                    );
                    return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
                }

                let result = array_utils::iter_chunks_zipped_mut(
                    input,
                    output,
                    self.len(),
                    |in_chunk, out_chunk| {
                        self.perform_fft_out_of_place(in_chunk, out_chunk, scratch)
                    },
                );

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                }
            }
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                if self.len() == 0 {
                    return;
                }

                let required_scratch = self.get_inplace_scratch_len();
                if scratch.len() < required_scratch || buffer.len() < self.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_inplace(
                        self.len(),
                        buffer.len(),
                        self.get_inplace_scratch_len(),
                        scratch.len(),
                    );
                    return; // Unreachable, because fft_error_inplace asserts, but it helps codegen to put it here
                }

                let (scratch, extra_scratch) = scratch.split_at_mut(self.len());
                let result = array_utils::iter_chunks_mut(buffer, self.len(), |chunk| {
                    self.perform_fft_out_of_place(chunk, scratch, extra_scratch);
                    chunk.copy_from_slice(scratch);
                });

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_inplace(
                        self.len(),
                        buffer.len(),
                        self.get_inplace_scratch_len(),
                        scratch.len(),
                    );
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                self.inplace_scratch_len()
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                self.outofplace_scratch_len()
            }
            #[inline(always)]
            fn get_immutable_scratch_len(&self) -> usize {
                self.inplace_scratch_len()
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len_fn(self)
            }
        }
        impl<T> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                self.direction
            }
        }
    };
}

macro_rules! boilerplate_fft {
    ($struct_name:ident, $len_fn:expr, $inplace_scratch_len_fn:expr, $out_of_place_scratch_len_fn:expr, $immut_scratch_len:expr) => {
        impl<T: FftNum> Fft<T> for $struct_name<T> {
            fn process_immutable_with_scratch(
                &self,
                input: &[Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                if self.len() == 0 {
                    return;
                }
                let required_scratch = self.get_immutable_scratch_len();
                if scratch.len() < required_scratch
                    || input.len() < self.len()
                    || output.len() != input.len()
                {
                    fft_error_outofplace(
                        self.len(),
                        input.len(),
                        output.len(),
                        required_scratch,
                        scratch.len(),
                    );
                }

                let scratch = &mut scratch[..required_scratch];
                let result = array_utils::iter_chunks_zipped(
                    input,
                    output,
                    self.len(),
                    |in_chunk, out_chunk| self.perform_fft_immut(in_chunk, out_chunk, scratch),
                );
                if result.is_err() {
                    fft_error_outofplace(
                        self.len(),
                        input.len(),
                        output.len(),
                        required_scratch,
                        scratch.len(),
                    );
                }
            }

            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                if self.len() == 0 {
                    return;
                }

                let required_scratch = self.get_outofplace_scratch_len();
                if scratch.len() < required_scratch
                    || input.len() < self.len()
                    || output.len() != input.len()
                {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(
                        self.len(),
                        input.len(),
                        output.len(),
                        self.get_outofplace_scratch_len(),
                        scratch.len(),
                    );
                    return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
                }

                let scratch = &mut scratch[..required_scratch];
                let result = array_utils::iter_chunks_zipped_mut(
                    input,
                    output,
                    self.len(),
                    |in_chunk, out_chunk| {
                        self.perform_fft_out_of_place(in_chunk, out_chunk, scratch)
                    },
                );

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(
                        self.len(),
                        input.len(),
                        output.len(),
                        self.get_outofplace_scratch_len(),
                        scratch.len(),
                    );
                }
            }
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                if self.len() == 0 {
                    return;
                }

                let required_scratch = self.get_inplace_scratch_len();
                if scratch.len() < required_scratch || buffer.len() < self.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_inplace(
                        self.len(),
                        buffer.len(),
                        self.get_inplace_scratch_len(),
                        scratch.len(),
                    );
                    return; // Unreachable, because fft_error_inplace asserts, but it helps codegen to put it here
                }

                let scratch = &mut scratch[..required_scratch];
                let result = array_utils::iter_chunks_mut(buffer, self.len(), |chunk| {
                    self.perform_fft_inplace(chunk, scratch)
                });

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_inplace(
                        self.len(),
                        buffer.len(),
                        self.get_inplace_scratch_len(),
                        scratch.len(),
                    );
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                $inplace_scratch_len_fn(self)
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                $out_of_place_scratch_len_fn(self)
            }
            #[inline(always)]
            fn get_immutable_scratch_len(&self) -> usize {
                $immut_scratch_len(self)
            }
        }
        impl<T: FftNum> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len_fn(self)
            }
        }
        impl<T: FftNum> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                self.direction
            }
        }
    };
}

#[non_exhaustive]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum RadixFactor {
    Factor2,
    Factor3,
    Factor4,
    Factor5,
    Factor6,
    Factor7,
}
impl RadixFactor {
    pub const fn radix(&self) -> usize {
        // note: if we had rustc 1.66, we could just turn these values explicit discriminators on the enum
        match self {
            RadixFactor::Factor2 => 2,
            RadixFactor::Factor3 => 3,
            RadixFactor::Factor4 => 4,
            RadixFactor::Factor5 => 5,
            RadixFactor::Factor6 => 6,
            RadixFactor::Factor7 => 7,
        }
    }
}
