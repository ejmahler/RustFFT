use std::any::TypeId;
use core::arch::x86_64::*;

// Calculate the sum of an expression consisting of just plus and minus, like a + b - c + d 
macro_rules! calc_f32 {
    (+ $acc:tt + $($rest:tt)*)=> {
        _mm_add_ps($acc, calc_f32!(+ $($rest)*))
    };
    (+ $acc:tt - $($rest:tt)*)=> {
        _mm_sub_ps($acc, calc_f32!(- $($rest)*))
    };
    (- $acc:tt + $($rest:tt)*)=> {
        _mm_sub_ps($acc, calc_f32!(+ $($rest)*))
    };
    (- $acc:tt - $($rest:tt)*)=> {
        _mm_add_ps($acc, calc_f32!(- $($rest)*))
    };
    ($acc:tt + $($rest:tt)*)=> {
        _mm_add_ps($acc, calc_f32!(+ $($rest)*))
    };
    ($acc:tt - $($rest:tt)*)=> {
        _mm_sub_ps($acc, calc_f32!(- $($rest)*))
    };
    (+ $val:tt) => {$val};
    (- $val:tt) => {$val};
}

// macro_rules! math_op_f32 {
//     ($acc:ident + $val:ident ) => { $acc = _mm_add_ps($acc, $val); };
//     ($acc:ident - $val:ident ) => { $acc = _mm_sub_ps($acc, $val); };
// }

// // Calculate the sum of an expression consisting of just plus and minus, like a + b - c + d 
// macro_rules! calc_f32 {
//     ($acc:ident, $first:ident $($op:tt $val:ident)* ) => { 
//         $acc = $first;
//         $(
//             math_op_f32!($acc $op $val);
//         )*
//     }
// }


macro_rules! math_op_f64 {
    ($acc:ident + $val:ident ) => { $acc = _mm_add_pd($acc, $val); };
    ($acc:ident - $val:ident ) => { $acc = _mm_sub_pd($acc, $val); };
}

// Calculate the sum of an expression consisting of just plus and minus, like a + b - c + d 
macro_rules! calc_f64 {
    ($acc:ident, $first:ident $($op:tt $val:ident)* ) => { 
        $acc = $first;
        $(
            math_op_f64!($acc $op $val);
        )*
    }
}

// Helper function to assert we have the right float type
pub fn assert_f32<T: 'static>() {
    let id_f32 = TypeId::of::<f32>();
    let id_t = TypeId::of::<T>();
    assert!(id_t == id_f32, "Wrong float type, must be f32");
}

// Helper function to assert we have the right float type
pub fn assert_f64<T: 'static>() {
    let id_f64 = TypeId::of::<f64>();
    let id_t = TypeId::of::<T>();
    assert!(id_t == id_f64, "Wrong float type, must be f64");
}

// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
macro_rules! interleave_complex_f32 {
    ($input:ident, $offset:literal, { $($idx:literal),* }) => {
        [
        $(
            pack_1st_f32($input[$idx], $input[$idx+$offset]),
            pack_2nd_f32($input[$idx], $input[$idx+$offset]),
        )*
        ]
    }
}

// Shuffle elements to interleave two contiguous sets of f32, from an array of simd vectors to a new array of simd vectors
// This statement:
// ```
// let values = separate_interleaved_complex_f32!(input, {0, 2, 4});
// ```
// is equivalent to:
// ```
// let values = [
//    pack_1st_f32(input[0], input[1]),
//    pack_1st_f32(input[2], input[3]),
//    pack_1st_f32(input[4], input[5]),
//    pack_2nd_f32(input[0], input[1]),
//    pack_2nd_f32(input[2], input[3]),
//    pack_2nd_f32(input[4], input[5]),
// ];
macro_rules! separate_interleaved_complex_f32 {
    ($input:ident, { $($idx:literal),* }) => {
        [
        $(
            pack_1st_f32($input[$idx], $input[$idx+1]),
        )*
        $(
            pack_2nd_f32($input[$idx], $input[$idx+1]),
        )*
        ]
    }
}


macro_rules! boilerplate_fft_sse_oop {
    ($struct_name:ident, $len_fn:expr) => {
        impl<T: FftNum> Fft<T> for $struct_name<T> {
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

/* Not used now, but maybe later for the mixed radixes etc
macro_rules! boilerplate_sse_fft {
    ($struct_name:ident, $len_fn:expr, $inplace_scratch_len_fn:expr, $out_of_place_scratch_len_fn:expr) => {
        impl<T: FftNum> Fft<T> for $struct_name<T> {
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
                let result = array_utils::iter_chunks_zipped(
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
                let result = array_utils::iter_chunks(buffer, self.len(), |chunk| {
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
*/

#[cfg(test)]
mod unit_tests {
    use super::*;
    use core::arch::x86_64::*;

    #[test]
    fn test_calc_f32() {
        unsafe {
            let a = _mm_set_ps(1.0, 1.0, 1.0, 1.0);
            let b = _mm_set_ps(2.0, 2.0, 2.0, 2.0);
            let c = _mm_set_ps(3.0, 3.0, 3.0, 3.0);
            let d = _mm_set_ps(4.0, 4.0, 4.0, 4.0);
            let e = _mm_set_ps(5.0, 5.0, 5.0, 5.0);
            let f = _mm_set_ps(6.0, 6.0, 6.0, 6.0);
            let g = _mm_set_ps(7.0, 7.0, 7.0, 7.0);
            let h = _mm_set_ps(8.0, 8.0, 8.0, 8.0);
            let i = _mm_set_ps(9.0, 9.0, 9.0, 9.0);
            let expected: f32 = 1.0 + 2.0 - 3.0 + 4.0 - 5.0 + 6.0 -7.0 - 8.0 + 9.0;
            let res = calc_f32!(a + b - c + d - e + f - g - h + i);
            let sum = std::mem::transmute::<__m128, [f32; 4]>(res);
            assert_eq!(sum[0], expected);
            assert_eq!(sum[1], expected);
            assert_eq!(sum[2], expected);
            assert_eq!(sum[3], expected);
        }
    }
}

