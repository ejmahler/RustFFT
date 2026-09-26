//! The body of the SIMD `RadixN` implementations, shared by every SIMD backend.
//!
//! This mirrors `src/algorithm/radixn.rs`: one flat transpose down to a base FFT, then a stack of
//! in-place cross-FFT layers over a single packed twiddle array. The only difference is that the
//! cross-FFT layers use SIMD column butterflies instead of the scalar ones, so a whole vector of
//! columns is processed per butterfly call.
//!
//! Everything here is generic over `SimdVector`, from `simd_vector.rs`. A backend implements that
//! trait once per vector type and gets the algorithm, so `SimdRadixN` is the only copy of it.
//!
//! Because a column butterfly consumes `COMPLEX_PER_VECTOR` columns at a time, the column count at
//! every layer has to be a whole number of vectors. The column count starts at `base_len` and only
//! ever grows by whole factors, so requiring `base_len % COMPLEX_PER_VECTOR == 0` is enough. That
//! is 1 for f64 (no restriction) and 2 for f32.

use std::any::TypeId;
use std::sync::Arc;

use num_complex::Complex;

use crate::array_utils::{reverse_bits, workaround_transmute_mut};
use crate::common::FftNum;
use crate::simd::simd_radixn::{cross_layer_chunks, table_transpose};
use crate::{Direction, Fft, FftDirection, Length};

use super::simd_vector::SimdVector;

const RADIX : usize = 4;

/// FFT algorithm for lengths that factor into small radixes, SIMD accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub struct SimdRadix4<V: SimdVector, T> {
    twiddles: Box<[V]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    reversed_columns: Box<[usize]>,
    rotation: V::Rotation,

    len: usize,
    direction: FftDirection,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,
}

impl<V: SimdVector, T: FftNum> SimdRadix4<V, T> {
    /// Constructs a SimdRadixN which computes FFTs of length `factor_product * base_fft.len()`.
    pub fn new(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        // Internal sanity check: Make sure that the vector's scalar type is T.
        // This struct has two generic parameters V and T, but T must always be V's scalar type,
        // and they are only kept separate to help work around the lack of specialization.
        assert_eq!(TypeId::of::<V::ScalarType>(), TypeId::of::<T>());

        let base_len = base_fft.len();
        let direction = base_fft.fft_direction();
        let complex_per_vector = V::COMPLEX_PER_VECTOR;

        // Every cross-FFT layer processes a whole vector of columns at a time. The column count
        // starts at base_len and is only ever multiplied by a factor, so this one check covers
        // every layer.
        assert!(
            k == 0 || base_len % complex_per_vector == 0,
            "SimdRadixN requires a base length divisible by {}, got {}",
            complex_per_vector,
            base_len
        );

        // set up our cross FFT butterfly instances. simultaneously, compute the number of twiddles
        let rotation = unsafe { V::make_rotate90(direction) };
        let mut cross_fft_len = base_len;
        let mut twiddle_count = 0;

        for _ in 0..k {
            // twiddles are stored a vector at a time, so a layer needs one chunk per vector column
            twiddle_count += (cross_fft_len / complex_per_vector) * (RADIX - 1);
            cross_fft_len *= RADIX;
        }
        let len = cross_fft_len;

        // Precompute where each column lands. Working this out per call costs two hardware divides
        // and an out-of-line `reverse_remainders` call per column, which is a large share of a
        // short FFT, and much more so on x86 where a 64-bit divide takes tens of cycles.
        let width = len / base_len;
        let rev_digits = width.trailing_zeros() / RADIX.trailing_zeros();
        let reversed_columns: Box<[usize]> = (0..width)
            .map(|x| reverse_bits::<RADIX>(x, rev_digits))
            .collect();
        // `table_transpose` indexes unchecked, and relies on this.
        assert!(reversed_columns.iter().all(|&r| r < width));

        // Same packing as RadixN: all layers in one array, bottom layer first, and
        // within a layer, (radix - 1) = 3 twiddles per column.
        let mut twiddle_factors: Vec<V> = Vec::with_capacity(twiddle_count);
        let mut cross_fft_len = base_len;
        for _ in 0..k {
            let num_vector_columns = cross_fft_len / complex_per_vector;
            cross_fft_len *= RADIX;

            for i in 0..num_vector_columns {
                for k in 1..RADIX {
                    unsafe {
                        twiddle_factors.push(V::make_mixedradix_twiddle_chunk(
                            i * complex_per_vector,
                            k,
                            cross_fft_len,
                            direction,
                        ));
                    }
                }
            }
        }

        // figure out how much scratch space we need to request from callers
        let base_inplace_scratch = base_fft.get_inplace_scratch_len();
        // the in-place path transposes into its own scratch and runs the base out of place from
        // there back into the caller's buffer, so it needs the base's out-of-place scratch on top
        let inplace_scratch_len = len + base_fft.get_outofplace_scratch_len();
        let outofplace_scratch_len = if base_inplace_scratch > len {
            base_inplace_scratch
        } else {
            0
        };

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

            base_fft,
            base_len,

            reversed_columns,
            rotation,

            len,
            direction,

            inplace_scratch_len,
            outofplace_scratch_len,
            immut_scratch_len: base_inplace_scratch,
        }
    }

    /// The flat transpose that reorders the input down to base-sized chunks.
    #[inline(always)]
    fn transpose(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        if self.len > self.base_len {
            let (height, columns) = (self.base_len, &*self.reversed_columns);
            table_transpose::<_, RADIX>(height, columns, input, output);
        } else {
            // no factors, so just pass data straight to our base
            output.copy_from_slice(input);
        }
    }

    /// The stack of in-place cross-FFT layers, run after the base FFTs.
    #[inline(always)]
    unsafe fn cross_ffts(&self, output: &mut [Complex<T>]) {
        let out: &mut [Complex<V::ScalarType>] = workaround_transmute_mut(output);

        let mut cross_fft_len = self.base_len;
        let mut layer_twiddles: &[V] = &self.twiddles;

        while cross_fft_len < out.len() {
            let num_columns = cross_fft_len;
            cross_fft_len *= RADIX;

            cross_layer_chunks::<V, RADIX, _>(out, layer_twiddles, num_columns, |v| {
                V::column_butterfly4(v, self.rotation)
            });

            // skip past all the twiddle factors used in this layer
            let twiddle_offset = (num_columns / V::COMPLEX_PER_VECTOR) * (RADIX - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];
        }
    }
}

impl<V: SimdVector, T: FftNum> Fft<T> for SimdRadix4<V, T> {
    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        unsafe {
            V::fft_helper_immut(
                input,
                output,
                scratch,
                self.len(),
                self.get_immutable_scratch_len(),
                |input, output, scratch| {
                    self.transpose(input, output);
                    self.base_fft.process_with_scratch(output, scratch);
                    self.cross_ffts(output);
                },
            );
        }
    }
    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        unsafe {
            V::fft_helper_outofplace(
                input,
                output,
                scratch,
                self.len(),
                self.get_outofplace_scratch_len(),
                |input, output, scratch| {
                    self.transpose(input, output);
                    // the input is free once the transpose is done, so use it as base scratch
                    // when we weren't handed any of our own
                    let base_scratch = if !scratch.is_empty() { scratch } else { input };
                    self.base_fft.process_with_scratch(output, base_scratch);
                    self.cross_ffts(output);
                },
            );
        }
    }
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        unsafe {
            V::fft_helper_inplace(
                buffer,
                scratch,
                self.len(),
                self.get_inplace_scratch_len(),
                |chunk, scratch| {
                    let (transposed, inner_scratch) = scratch.split_at_mut(self.len());
                    self.transpose(chunk, transposed);
                    // Run the base out of place, from the scratch back into the caller's chunk.
                    // The cross layers are then in place on the chunk, and the whole thing ends
                    // where the caller wants it without a final copy.
                    self.base_fft
                        .process_outofplace_with_scratch(transposed, chunk, inner_scratch);
                    self.cross_ffts(chunk);
                },
            )
        }
    }
    #[inline(always)]
    fn get_inplace_scratch_len(&self) -> usize {
        self.inplace_scratch_len
    }
    #[inline(always)]
    fn get_outofplace_scratch_len(&self) -> usize {
        self.outofplace_scratch_len
    }
    #[inline(always)]
    fn get_immutable_scratch_len(&self) -> usize {
        self.immut_scratch_len
    }
}
impl<V: SimdVector, T> Length for SimdRadix4<V, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<V: SimdVector, T> Direction for SimdRadix4<V, T> {
    #[inline(always)]
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

/// The test bodies, shared the same way the algorithm is. Every backend runs all of them, from a
/// thin test function per body that names its own vector types. The bodies that exercise both
/// element types take both vector types, so `V32` is always the f32 vector and `V64` the f64 one.
#[cfg(test)]
pub mod test_bodies {
    use super::*;
    use crate::test_utils::{check_fft_algorithm, construct_base};
    use num_traits::Float;
    use rand::distr::uniform::SampleUniform;

    /// Every empty, one-factor and two-factor recipe over each of `bases`, both directions.
    pub fn factor_pairs<V>(bases: &[usize])
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        for base in bases {
            let base_forward = construct_base(*base, FftDirection::Forward);
            let base_inverse = construct_base(*base, FftDirection::Inverse);

            for k in 0..3 {
                check::<V>(k, Arc::clone(&base_forward));
                check::<V>(k, Arc::clone(&base_inverse));
            }
        }
    }

    /// The base doesn't have to be a scratch-free butterfly. A composite base is a recursive
    /// recipe that needs its own scratch, which is the case `design_radixn` hits whenever a length
    /// has factors above 7 (for example 11 * 13 = 143).
    pub fn composite_base<V32, V64>()
    where
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
    {
        let mut planner64 = crate::FftPlannerScalar::<f64>::new();
        let mut planner32 = crate::FftPlannerScalar::<f32>::new();

        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            // odd base, f64 only
            for base_len in [143, 55, 65] {
                let base = planner64.plan_fft(base_len, direction);
                assert!(
                    base.get_inplace_scratch_len() > 0,
                    "base {} was expected to need scratch",
                    base_len
                );
                for k in 0..3 {
                    check::<V64>(k, Arc::clone(&base));
                }
            }

            // even base, usable by both element types
            for base_len in [22, 26, 110] {
                let base32 = planner32.plan_fft(base_len, direction);
                let base64 = planner64.plan_fft(base_len, direction);
                assert!(
                    base32.get_inplace_scratch_len() > 0,
                    "base {} was expected to need scratch",
                    base_len
                );
                assert!(
                    base64.get_inplace_scratch_len() > 0,
                    "base {} was expected to need scratch",
                    base_len
                );

                for k in 0..3 {
                    check::<V32>(k, Arc::clone(&base32));
                    check::<V64>(k, Arc::clone(&base64));
                }
            }
        }
    }

    fn check<V>(k: u32, base_fft: Arc<dyn Fft<V::ScalarType>>)
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        let len = base_fft.len() * RADIX.pow(k);
        let direction = base_fft.fft_direction();
        let fft: SimdRadix4<V, V::ScalarType> = SimdRadix4::new(k, base_fft);

        check_fft_algorithm::<V::ScalarType>(&fft, len, direction);
    }
}
