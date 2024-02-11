use num_complex::Complex;
use num_integer::Integer;

use std::any::TypeId;
use std::arch::x86_64::{__m128, __m128d};
use std::sync::Arc;

use crate::array_utils::{self, reverse_bits, workaround_transmute, workaround_transmute_mut};
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::sse::sse_utils::transpose_complex_2x2_f32;
use crate::{common::FftNum, FftDirection};
use crate::{Direction, Fft, Length};

use super::SseNum;

use super::sse_vector::{Deinterleaved, SseArray, SseArrayMut, SseVector};

/// FFT algorithm optimized for power-of-two sizes, SSE accelerated version.
/// This is designed to be used via a Planner, and not created directly.

pub struct SseRadix4<S: SseNum, T> {
    twiddles: Box<[Deinterleaved<S::VectorType>]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
}

impl<S: SseNum, T: FftNum> SseRadix4<S, T> {
    /// Constructs a new SseRadix4 which computes FFTs of size 4^k * base_fft.len()
    #[inline]
    pub fn new(k: u32, base_fft: Arc<dyn Fft<T>>) -> Result<Self, ()> {
        // Internal sanity check: Make sure that S == T.
        // This struct has two generic parameters S and T, but they must always be the same, and are only kept separate to help work around the lack of specialization.
        // It would be cool if we could do this as a static_assert instead
        let id_a = TypeId::of::<S>();
        let id_t = TypeId::of::<T>();
        assert_eq!(id_a, id_t);

        let has_sse = is_x86_feature_detected!("sse4.1");
        if has_sse {
            // Safety: new_with_sse requires the "sse4.1" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_sse(k, base_fft) })
        } else {
            Err(())
        }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn new_with_sse(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        let direction = base_fft.fft_direction();
        let base_len = base_fft.len();

        // note that we can eventually release this restriction - we just need to update the rest of the code in here to handle remainders
        assert!(base_len % (S::VectorType::SCALAR_PER_VECTOR) == 0 && base_len > 0);

        let len = base_len * (1 << (k * 2));

        // precompute the twiddle factors this algorithm will use.
        // we're doing the same precomputation of twiddle factors as the mixed radix algorithm where width=4 and height=len/4
        // but mixed radix only does one step and then calls itself recusrively, and this algorithm does every layer all the way down
        // so we're going to pack all the "layers" of twiddle factors into a single array, starting with the bottom layer and going up
        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = base_len;
        let mut twiddle_factors = Vec::with_capacity(len * 2);
        while cross_fft_len < len {
            let num_scalar_columns = cross_fft_len;
            cross_fft_len *= ROW_COUNT;

            let (quotient, remainder) =
                num_scalar_columns.div_rem(&S::VectorType::SCALAR_PER_VECTOR);
            let num_vector_columns = quotient + if remainder > 0 { 1 } else { 0 };

            for i in 0..num_vector_columns {
                for k in 1..ROW_COUNT {
                    let twiddle0 = SseVector::make_mixedradix_twiddle_chunk(
                        i * S::VectorType::SCALAR_PER_VECTOR,
                        k,
                        cross_fft_len,
                        direction,
                    );
                    let twiddle1 = SseVector::make_mixedradix_twiddle_chunk(
                        i * S::VectorType::SCALAR_PER_VECTOR + S::VectorType::COMPLEX_PER_VECTOR,
                        k,
                        cross_fft_len,
                        direction,
                    );
                    let deinterleaved_twiddles = SseVector::deinterleave(twiddle0, twiddle1);
                    twiddle_factors.push(deinterleaved_twiddles);
                }
            }
        }

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

            base_fft,
            base_len,

            len,
            direction,
        }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn perform_fft_out_of_place(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // copy the data into the output vector
        if self.len() == self.base_len {
            output.copy_from_slice(input);
        } else {
            // Hack: Making a f32 vs f64 agaonstic version of this seems hard. Avoid it for now, and hopefully we can make one later
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let input = workaround_transmute(input);
                let output = workaround_transmute_mut(output);
                sse_bitreversed_transpose_f32(self.base_len, input, output);
            } else {
                let input = workaround_transmute(input);
                let output = workaround_transmute_mut(output);
                sse_bitreversed_transpose_f64(self.base_len, input, output);
            }
        }

        // Base-level FFTs
        self.base_fft.process_with_scratch(output, &mut []);

        // cross-FFTs
        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = self.base_len;
        let mut layer_twiddles: &[Deinterleaved<S::VectorType>] = &self.twiddles;

        while cross_fft_len < input.len() {
            let columns = cross_fft_len;
            let first = cross_fft_len == self.base_len;
            cross_fft_len *= ROW_COUNT;
            let last = cross_fft_len == self.len();

            if first && last {
                for data in output.chunks_exact_mut(cross_fft_len) {
                    butterfly_4::<S, T, true, true>(data, layer_twiddles, columns, self.direction)
                }
            } else if first {
                for data in output.chunks_exact_mut(cross_fft_len) {
                    butterfly_4::<S, T, true, false>(data, layer_twiddles, columns, self.direction)
                }
            } else if last {
                for data in output.chunks_exact_mut(cross_fft_len) {
                    butterfly_4::<S, T, false, true>(data, layer_twiddles, columns, self.direction)
                }
            } else {
                for data in output.chunks_exact_mut(cross_fft_len) {
                    butterfly_4::<S, T, false, false>(data, layer_twiddles, columns, self.direction)
                }
            }

            // skip past all the twiddle factors used in this layer
            let (quotient, remainder) = columns.div_rem(&S::VectorType::SCALAR_PER_VECTOR);
            let num_vector_columns = quotient + if remainder > 0 { 1 } else { 0 };

            let twiddle_offset = num_vector_columns * (ROW_COUNT - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];
        }
    }
}
boilerplate_fft_sse_oop!(SseRadix4, |this: &SseRadix4<_, _>| this.len);

#[inline(always)]
fn load_debug_checked<T: Copy>(buffer: &[T], idx: usize) -> T {
    debug_assert!(idx < buffer.len());
    unsafe { *buffer.get_unchecked(idx) }
}

#[inline(always)]
unsafe fn load<S: SseNum, const DEINTERLEAVE: bool>(
    buffer: &[Complex<S>],
    idx: usize,
) -> Deinterleaved<S::VectorType> {
    if DEINTERLEAVE {
        let a = buffer.load_complex(idx);
        let b = buffer.load_complex(idx + S::VectorType::COMPLEX_PER_VECTOR);
        SseVector::deinterleave(a, b)
    } else {
        let a = buffer.load_complex(idx);
        let b = buffer.load_complex(idx + S::VectorType::COMPLEX_PER_VECTOR);
        Deinterleaved { re: a, im: b }
    }
}

#[inline(always)]
unsafe fn store<S: SseNum, const INTERLEAVE: bool>(
    mut buffer: &mut [Complex<S>],
    vector: Deinterleaved<S::VectorType>,
    idx: usize,
) {
    if INTERLEAVE {
        let (a, b) = SseVector::interleave(vector);
        buffer.store_complex(a, idx);
        buffer.store_complex(b, idx + S::VectorType::COMPLEX_PER_VECTOR);
    } else {
        buffer.store_complex(vector.re, idx);
        buffer.store_complex(vector.im, idx + S::VectorType::COMPLEX_PER_VECTOR);
    }
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
unsafe fn butterfly_4<S: SseNum, T: FftNum, const D: bool, const I: bool>(
    data: &mut [Complex<T>],
    twiddles: &[Deinterleaved<S::VectorType>],
    num_scalar_columns: usize,
    direction: FftDirection,
) {
    let num_vector_columns = num_scalar_columns / S::VectorType::SCALAR_PER_VECTOR;
    let buffer: &mut [Complex<S>] = workaround_transmute_mut(data);

    for i in 0..num_vector_columns {
        let idx = i * S::VectorType::SCALAR_PER_VECTOR;
        let tw_idx = i * 3;
        let mut scratch = [
            load::<S, D>(buffer, idx + 0 * num_scalar_columns),
            load::<S, D>(buffer, idx + 1 * num_scalar_columns),
            load::<S, D>(buffer, idx + 2 * num_scalar_columns),
            load::<S, D>(buffer, idx + 3 * num_scalar_columns),
        ];

        let tw1 = load_debug_checked(twiddles, tw_idx + 0);
        let tw2 = load_debug_checked(twiddles, tw_idx + 1);
        let tw3 = load_debug_checked(twiddles, tw_idx + 2);

        scratch[1] = Deinterleaved::mul_complex(scratch[1], tw1);
        scratch[2] = Deinterleaved::mul_complex(scratch[2], tw2);
        scratch[3] = Deinterleaved::mul_complex(scratch[3], tw3);

        let scratch = Deinterleaved::butterfly4(scratch, direction);

        store::<S, I>(buffer, scratch[0], idx + 0 * num_scalar_columns);
        store::<S, I>(buffer, scratch[1], idx + 1 * num_scalar_columns);
        store::<S, I>(buffer, scratch[2], idx + 2 * num_scalar_columns);
        store::<S, I>(buffer, scratch[3], idx + 3 * num_scalar_columns);
    }
}

#[inline(always)]
unsafe fn load4_complex_f32(buffer: &[Complex<f32>], idx: usize) -> [__m128; 2] {
    [
        buffer.load_complex(idx),
        buffer.load_complex(idx + <f32 as SseNum>::VectorType::COMPLEX_PER_VECTOR),
    ]
}
#[inline(always)]
unsafe fn transpose_complex_4x4_f32(rows: [[__m128; 2]; 4]) -> [[__m128; 2]; 4] {
    let transposed0 = transpose_complex_2x2_f32(rows[0][0], rows[1][0]);
    let transposed1 = transpose_complex_2x2_f32(rows[0][1], rows[1][1]);
    let transposed2 = transpose_complex_2x2_f32(rows[2][0], rows[3][0]);
    let transposed3 = transpose_complex_2x2_f32(rows[2][1], rows[3][1]);

    [
        [transposed0[0], transposed2[0]],
        [transposed0[1], transposed2[1]],
        [transposed1[0], transposed3[0]],
        [transposed1[1], transposed3[1]],
    ]
}
#[inline(always)]
unsafe fn store4_complex_f32(mut buffer: &mut [Complex<f32>], data: [__m128; 2], idx: usize) {
    buffer.store_complex(data[0], idx);
    buffer.store_complex(
        data[1],
        idx + <f32 as SseNum>::VectorType::COMPLEX_PER_VECTOR,
    );
}

// Utility to help reorder data as a part of computing RadixD FFTs. Conceputally, it works like a transpose, but with the column indexes bit-reversed.
// Use a lookup table to avoid repeating the slow bit reverse operations.
// Unrolling the outer loop by a factor D helps speed things up.
// const parameter D (for Divisor) determines the divisor to use for the "bit reverse", and how much to unroll. `input.len() / height` must be a power of D.
#[inline(never)]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_bitreversed_transpose_f32(
    height: usize,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    let width = input.len() / height;
    const WIDTH_UNROLL: usize = 4;
    const HEIGHT_UNROLL: usize = <f32 as SseNum>::VectorType::SCALAR_PER_VECTOR;

    // Let's make sure the arguments are ok
    assert!(
        height % <f32 as SseNum>::VectorType::SCALAR_PER_VECTOR == 0
            && width % WIDTH_UNROLL == 0
            && input.len() % height == 0
            && input.len() == output.len()
    );

    let width_bits = width.trailing_zeros();
    let d_bits = WIDTH_UNROLL.trailing_zeros();

    // verify that width is a power of d
    assert!(width_bits % d_bits == 0);
    let rev_digits = width_bits / d_bits;
    let strided_width = width / WIDTH_UNROLL;
    let strided_height = height / HEIGHT_UNROLL;
    for x in 0..strided_width {
        let x_rev = [
            reverse_bits::<WIDTH_UNROLL>(WIDTH_UNROLL * x + 0, rev_digits) * height,
            reverse_bits::<WIDTH_UNROLL>(WIDTH_UNROLL * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH_UNROLL>(WIDTH_UNROLL * x + 2, rev_digits) * height,
            reverse_bits::<WIDTH_UNROLL>(WIDTH_UNROLL * x + 3, rev_digits) * height,
        ];

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // we add HEIGHT_UNROLL * y to each x_rev, which goes up to height exclusive
        // so verify that r + height isn't more than our length
        for r in x_rev {
            assert!(r <= input.len() - height);
        }
        for y in 0..strided_height {
            unsafe {
                // Load data in HEIGHT_UNROLL rows, with each row containing WIDTH_UNROLL=4 complex elements
                // for f32, HEIGHT_UNROLL=4, this translates to 4 rows of 2 SSE vectors each,
                // overall storing 4x4=16 complex elements
                let base_input_idx = WIDTH_UNROLL * x + 0 + y * HEIGHT_UNROLL * width;
                let rows = [
                    load4_complex_f32(input, base_input_idx + width * 0),
                    load4_complex_f32(input, base_input_idx + width * 1),
                    load4_complex_f32(input, base_input_idx + width * 2),
                    load4_complex_f32(input, base_input_idx + width * 3),
                ];
                let transposed = transpose_complex_4x4_f32(rows);

                store4_complex_f32(output, transposed[0], HEIGHT_UNROLL * y + x_rev[0]);
                store4_complex_f32(output, transposed[1], HEIGHT_UNROLL * y + x_rev[1]);
                store4_complex_f32(output, transposed[2], HEIGHT_UNROLL * y + x_rev[2]);
                store4_complex_f32(output, transposed[3], HEIGHT_UNROLL * y + x_rev[3]);
            }
        }
    }
}

#[inline(always)]
unsafe fn load4_complex_f64(buffer: &[Complex<f64>], idx: usize) -> [__m128d; 4] {
    [
        buffer.load_complex(idx + 0),
        buffer.load_complex(idx + 1),
        buffer.load_complex(idx + 2),
        buffer.load_complex(idx + 3),
    ]
}
#[inline(always)]
unsafe fn transpose_complex_4x2_f64(rows: [[__m128d; 4]; 2]) -> [[__m128d; 2]; 4] {
    [
        [rows[0][0], rows[1][0]],
        [rows[0][1], rows[1][1]],
        [rows[0][2], rows[1][2]],
        [rows[0][3], rows[1][3]],
    ]
}
#[inline(always)]
unsafe fn store2_complex_f64(mut buffer: &mut [Complex<f64>], data: [__m128d; 2], idx: usize) {
    buffer.store_complex(data[0], idx);
    buffer.store_complex(
        data[1],
        idx + <f64 as SseNum>::VectorType::COMPLEX_PER_VECTOR,
    );
}

// Utility to help reorder data as a part of computing RadixD FFTs. Conceputally, it works like a transpose, but with the column indexes bit-reversed.
// Use a lookup table to avoid repeating the slow bit reverse operations.
// Unrolling the outer loop by a factor D helps speed things up.
// const parameter D (for Divisor) determines the divisor to use for the "bit reverse", and how much to unroll. `input.len() / height` must be a power of D.
#[inline(never)]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_bitreversed_transpose_f64(
    height: usize,
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
) {
    let width = input.len() / height;
    const WIDTH_UNROLL: usize = 4;
    const HEIGHT_UNROLL: usize = <f64 as SseNum>::VectorType::SCALAR_PER_VECTOR;

    // Let's make sure the arguments are ok
    assert!(
        height % <f64 as SseNum>::VectorType::SCALAR_PER_VECTOR == 0
            && width % WIDTH_UNROLL == 0
            && input.len() % height == 0
            && input.len() == output.len()
    );

    let width_bits = width.trailing_zeros();
    let d_bits = WIDTH_UNROLL.trailing_zeros();

    // verify that width is a power of d
    assert!(width_bits % d_bits == 0);
    let rev_digits = width_bits / d_bits;
    let strided_width = width / WIDTH_UNROLL;
    let strided_height = height / HEIGHT_UNROLL;
    for x in 0..strided_width {
        let x_rev = [
            reverse_bits::<WIDTH_UNROLL>(WIDTH_UNROLL * x + 0, rev_digits) * height,
            reverse_bits::<WIDTH_UNROLL>(WIDTH_UNROLL * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH_UNROLL>(WIDTH_UNROLL * x + 2, rev_digits) * height,
            reverse_bits::<WIDTH_UNROLL>(WIDTH_UNROLL * x + 3, rev_digits) * height,
        ];

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // we add HEIGHT_UNROLL * y to each x_rev, which goes up to height exclusive
        // so verify that r + height isn't more than our length
        for r in x_rev {
            assert!(r <= input.len() - height);
        }
        for y in 0..strided_height {
            unsafe {
                // Load data in HEIGHT_UNROLL rows, with each row containing WIDTH_UNROLL=4 complex elements
                // for f64, HEIGHT_UNROLL=2, this translates to 2 rows of 4 SSE vectors each,
                // overall storing 2x4=8 complex elements
                let base_input_idx = WIDTH_UNROLL * x + 0 + y * HEIGHT_UNROLL * width;
                let rows = [
                    load4_complex_f64(input, base_input_idx + width * 0),
                    load4_complex_f64(input, base_input_idx + width * 1),
                ];
                let transposed = transpose_complex_4x2_f64(rows);

                store2_complex_f64(output, transposed[0], HEIGHT_UNROLL * y + x_rev[0]);
                store2_complex_f64(output, transposed[1], HEIGHT_UNROLL * y + x_rev[1]);
                store2_complex_f64(output, transposed[2], HEIGHT_UNROLL * y + x_rev[2]);
                store2_complex_f64(output, transposed[3], HEIGHT_UNROLL * y + x_rev[3]);
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::{check_fft_algorithm, construct_base};

    #[test]
    fn test_sse_radix4_64() {
        for base in [2, 4, 6, 8, 12, 16] {
            let base_forward = construct_base(base, FftDirection::Forward);
            let base_inverse = construct_base(base, FftDirection::Inverse);
            for k in 0..4 {
                test_sse_radix4_64_with_base(k, Arc::clone(&base_forward));
                test_sse_radix4_64_with_base(k, Arc::clone(&base_inverse));
            }
        }
    }

    fn test_sse_radix4_64_with_base(k: u32, base_fft: Arc<dyn Fft<f64>>) {
        let len = base_fft.len() * 4usize.pow(k);
        let direction = base_fft.fft_direction();
        let fft = SseRadix4::<f64, f64>::new(k, base_fft).unwrap();
        check_fft_algorithm::<f64>(&fft, len, direction);
    }

    #[test]
    fn test_sse_radix4_32() {
        for base in [4, 8, 12, 16] {
            let base_forward = construct_base(base, FftDirection::Forward);
            let base_inverse = construct_base(base, FftDirection::Inverse);
            for k in 0..4 {
                test_sse_radix4_32_with_base(k, Arc::clone(&base_forward));
                test_sse_radix4_32_with_base(k, Arc::clone(&base_inverse));
            }
        }
    }

    fn test_sse_radix4_32_with_base(k: u32, base_fft: Arc<dyn Fft<f32>>) {
        let len = base_fft.len() * 4usize.pow(k);
        let direction = base_fft.fft_direction();
        let fft = SseRadix4::<f32, f32>::new(k, base_fft).unwrap();
        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
