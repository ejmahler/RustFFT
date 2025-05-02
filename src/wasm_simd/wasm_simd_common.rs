use std::any::TypeId;

/// Helper function to assert we have the right float type
pub fn assert_f32<T: 'static>() {
    let id_f32 = TypeId::of::<f32>();
    let id_t = TypeId::of::<T>();
    assert!(id_t == id_f32, "Wrong float type, must be f32");
}

/// Helper function to assert we have the right float type
pub fn assert_f64<T: 'static>() {
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();
    assert!(id_t == id_f64, "Wrong float type, must be f64");
}

/// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
macro_rules! interleave_complex_f32 {
    ($input:ident, $offset:literal, { $($idx:literal),* }) => {
        [
        $(
            extract_lo_lo_f32_v128($input[$idx], $input[$idx+$offset]),
            extract_hi_hi_f32_v128($input[$idx], $input[$idx+$offset]),
        )*
        ]
    }
}

/// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
/// This statement:
/// ```
/// let values = separate_interleaved_complex_f32!(input, {0, 2, 4});
/// ```
/// is equivalent to:
/// ```
/// let values = [
///    extract_lo_lo_f32(input[0], input[1]),
///    extract_lo_lo_f32(input[2], input[3]),
///    extract_lo_lo_f32(input[4], input[5]),
///    extract_hi_hi_f32(input[0], input[1]),
///    extract_hi_hi_f32(input[2], input[3]),
///    extract_hi_hi_f32(input[4], input[5]),
/// ];
macro_rules! separate_interleaved_complex_f32 {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            extract_lo_lo_f32_v128($input[$idx], $input[$idx+1]),
        )*
        $(
            extract_hi_hi_f32_v128($input[$idx], $input[$idx+1]),
        )*
        ]
    }
}

macro_rules! boilerplate_fft_wasm_simd_oop {
    ($struct_name:ident, $len_fn:expr) => {
        impl<S: WasmNum, T: FftNum> Fft<T> for $struct_name<S, T> {
            fn process_immutable_with_scratch(
                &self,
                input: &[Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                if self.len() == 0 {
                    return;
                }

                if input.len() < self.len() || output.len() != input.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                    return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
                }

                let result = unsafe {
                    array_utils::iter_chunks_zipped(
                        input,
                        output,
                        self.len(),
                        |in_chunk, out_chunk| self.perform_fft_immut(in_chunk, out_chunk, &mut []),
                    )
                };

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
                _scratch: &mut [Complex<T>],
            ) {
                if self.len() == 0 {
                    return;
                }

                if input.len() < self.len() || output.len() != input.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                    return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
                }

                let result = unsafe {
                    array_utils::iter_chunks_zipped_mut(
                        input,
                        output,
                        self.len(),
                        |in_chunk, out_chunk| {
                            self.perform_fft_out_of_place(in_chunk, out_chunk, &mut [])
                        },
                    )
                };

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

                let scratch = &mut scratch[..required_scratch];
                let result = unsafe {
                    array_utils::iter_chunks_mut(buffer, self.len(), |chunk| {
                        self.perform_fft_out_of_place(chunk, scratch, &mut []);
                        chunk.copy_from_slice(scratch);
                    })
                };
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
                self.len()
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_immutable_scratch_len(&self) -> usize {
                self.len()
            }
        }
        impl<S: WasmNum, T> Length for $struct_name<S, T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len_fn(self)
            }
        }
        impl<S: WasmNum, T> Direction for $struct_name<S, T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                self.direction
            }
        }
    };
}
