use num_complex::Complex;
use num_integer::Integer;

use std::any::TypeId;
use std::sync::Arc;

use crate::array_utils::{self, bitreversed_transpose, workaround_transmute_mut};
use crate::common::{fft_error_inplace, fft_error_outofplace};
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
            bitreversed_transpose::<Complex<T>, 4>(self.base_len, input, output);
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
unsafe fn load<S: SseNum, const DEINTERLEAVE: bool>(buffer: &[Complex<S>], idx: usize) -> Deinterleaved<S::VectorType> {
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
