use std::any::TypeId;

/// Calculate the sum of an expression consisting of just plus and minus, like `value = a + b - c + d`.
/// The expression is rewritten to `value = a + (b - (c - d))` (note the flipped sign on d).
/// After this the `$add` and `$sub` functions are used to make the calculation.
/// For f32 using `f32x4_add` and `f32x4_sub`, the expression `value = a + b - c + d` becomes:
/// ```
/// let value = f32x4_add(a, f32x4_sub(b, f32x4_sub(c, d)));
/// ```
/// Only plus and minus are supported, and all the terms must be plain scalar variables.
/// Using array indices, like `value = temp[0] + temp[1]` is not supported.
macro_rules! calc_sum {
    ($add:ident, $sub:ident, + $acc:tt + $($rest:tt)*)=> {
        $add($acc, calc_sum!($add, $sub, + $($rest)*))
    };
    ($add:ident, $sub:ident, + $acc:tt - $($rest:tt)*)=> {
        $sub($acc, calc_sum!($add, $sub, - $($rest)*))
    };
    ($add:ident, $sub:ident, - $acc:tt + $($rest:tt)*)=> {
        $sub($acc, calc_sum!($add, $sub, + $($rest)*))
    };
    ($add:ident, $sub:ident, - $acc:tt - $($rest:tt)*)=> {
        $add($acc, calc_sum!($add, $sub, - $($rest)*))
    };
    ($add:ident, $sub:ident, $acc:tt + $($rest:tt)*)=> {
        $add($acc, calc_sum!($add, $sub, + $($rest)*))
    };
    ($add:ident, $sub:ident, $acc:tt - $($rest:tt)*)=> {
        $sub($acc, calc_sum!($add, $sub, - $($rest)*))
    };
    ($add:ident, $sub:ident, + $val:tt) => {$val};
    ($add:ident, $sub:ident, - $val:tt) => {$val};
}

/// Calculate the sum of an expression consisting of just plus and minus, like a + b - c + d
macro_rules! calc_f32 {
    ($($tokens:tt)*) => { calc_sum!(f32x4_add, f32x4_sub, $($tokens)*)};
}

/// Calculate the sum of an expression consisting of just plus and minus, like a + b - c + d
macro_rules! calc_f64 {
    ($($tokens:tt)*) => { calc_sum!(f64x2_add, f64x2_sub, $($tokens)*)};
}

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
            extract_lo_lo_f32($input[$idx], $input[$idx+$offset]),
            extract_hi_hi_f32($input[$idx], $input[$idx+$offset]),
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
            extract_lo_lo_f32($input[$idx], $input[$idx+1]),
        )*
        $(
            extract_hi_hi_f32($input[$idx], $input[$idx+1]),
        )*
        ]
    }
}

macro_rules! boilerplate_fft_wasm_simd_oop {
    ($struct_name:ident, $len_fn:expr) => {
        impl<S: WasmNum, T: FftNum> Fft<T> for $struct_name<S, T> {
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
                    array_utils::iter_chunks_zipped(
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
                    array_utils::iter_chunks(buffer, self.len(), |chunk| {
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

#[cfg(test)]
mod unit_tests {
    use core::arch::wasm32::*;

    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_calc_f32() {
        unsafe {
            let a = f32x4(1.0, 1.0, 1.0, 1.0);
            let b = f32x4(2.0, 2.0, 2.0, 2.0);
            let c = f32x4(3.0, 3.0, 3.0, 3.0);
            let d = f32x4(4.0, 4.0, 4.0, 4.0);
            let e = f32x4(5.0, 5.0, 5.0, 5.0);
            let f = f32x4(6.0, 6.0, 6.0, 6.0);
            let g = f32x4(7.0, 7.0, 7.0, 7.0);
            let h = f32x4(8.0, 8.0, 8.0, 8.0);
            let i = f32x4(9.0, 9.0, 9.0, 9.0);
            let expected: f32 = 1.0 + 2.0 - 3.0 + 4.0 - 5.0 + 6.0 - 7.0 - 8.0 + 9.0;
            let res = calc_f32!(a + b - c + d - e + f - g - h + i);
            let sum = std::mem::transmute::<v128, [f32; 4]>(res);
            assert_eq!(sum[0], expected);
            assert_eq!(sum[1], expected);
            assert_eq!(sum[2], expected);
            assert_eq!(sum[3], expected);
        }
    }
    #[wasm_bindgen_test]
    fn test_calc_f64() {
        unsafe {
            let a = f64x2(1.0, 1.0);
            let b = f64x2(2.0, 2.0);
            let c = f64x2(3.0, 3.0);
            let d = f64x2(4.0, 4.0);
            let e = f64x2(5.0, 5.0);
            let f = f64x2(6.0, 6.0);
            let g = f64x2(7.0, 7.0);
            let h = f64x2(8.0, 8.0);
            let i = f64x2(9.0, 9.0);
            let expected: f64 = 1.0 + 2.0 - 3.0 + 4.0 - 5.0 + 6.0 - 7.0 - 8.0 + 9.0;
            let res = calc_f64!(a + b - c + d - e + f - g - h + i);
            let sum = std::mem::transmute::<v128, [f64; 2]>(res);
            assert_eq!(sum[0], expected);
            assert_eq!(sum[1], expected);
        }
    }
}
