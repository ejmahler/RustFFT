//! The body of the SIMD `Radix4` implementations, shared by every SIMD backend.
//!
//! This is essentially `SimdRadixN`, monomorphized for the case where all factors are 4. The
//! reduced branching and reduced memory usage squeezes out a little more performance for the very
//! common case of power-of-two FFTs.
//!
//! Everything here is generic over `SimdVector`, from `simd_vector.rs`. A backend implements that
//! trait once per vector type and gets the algorithm, so `SimdRadix4` is the only copy of it.
//!
//! A column butterfly consumes `COMPLEX_PER_VECTOR` columns at a time, 2 for f32 and 1 for f64 and
//! the scalar adapter, so a column count that isn't a whole number of vectors leaves a column over.
//! The count starts at `base_len` and only ever grows by whole factors, so for f32 an odd base
//! means an odd count at every layer. Each layer therefore ends with a partial column, handled the
//! same way as in `SimdRadixN`: one complex loaded into the low half of a vector and only the low
//! half stored back.

use std::any::TypeId;
use std::sync::Arc;

use num_complex::Complex;

use crate::array_utils::{reverse_bits, workaround_transmute, workaround_transmute_mut};
use crate::common::{FftNum, RadixFactor};
use crate::simd::simd_transpose::{TransposeAlgorithm, table_transpose, unrolled_transpose};
use crate::{Fft, FftDirection};

use super::simd_vector::SimdVector;

use crate::simd::simd_radixn::cross_layer_chunks;

const RADIX: usize = 4;

/// FFT algorithm for lengths that factor into small radixes, SIMD accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub struct SimdRadix4<V: SimdVector, T> {
    twiddles: Box<[V]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    transpose_algorithm: TransposeAlgorithm,
    butterfly4: V::Butterfly4,

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

        // set up our cross FFT butterfly instances. simultaneously, compute the number of twiddles
        let mut cross_fft_len = base_len;
        let mut twiddle_count = 0;

        for _ in 0..k {
            // twiddles are stored a vector at a time, so a layer needs one chunk per vector column
            twiddle_count += cross_fft_len.div_ceil(complex_per_vector) * (RADIX - 1);
            cross_fft_len *= RADIX;
        }
        let len = cross_fft_len;

        let transpose_algorithm = match k {
            0 => TransposeAlgorithm::Copy,
            1 => TransposeAlgorithm::UnrolledTranspose(RadixFactor::Factor4),
            _ => {
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

                TransposeAlgorithm::TableTranspose(RadixFactor::Factor4, reversed_columns)
            }
        };

        // Same packing as RadixN: all layers in one array, bottom layer first, and
        // within a layer, (radix - 1) = 3 twiddles per column.
        let mut twiddle_factors: Vec<V> = Vec::with_capacity(twiddle_count);
        let mut cross_fft_len = base_len;
        for _ in 0..k {
            let num_vector_columns = cross_fft_len.div_ceil(complex_per_vector);
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

            transpose_algorithm,
            butterfly4: unsafe { V::make_butterfly4(direction) },

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
        let input = unsafe { workaround_transmute(input) };
        let output = unsafe { workaround_transmute_mut(output) };

        match &self.transpose_algorithm {
            TransposeAlgorithm::Copy => {
                output.copy_from_slice(input);
            }
            TransposeAlgorithm::UnrolledTranspose(_) => unsafe { unrolled_transpose::<V, 4>(self.base_len, input, output) },
            TransposeAlgorithm::TableTranspose(_, reversed_columns) => unsafe { table_transpose::<V, 4>(self.base_len, reversed_columns, input, output) },
        }
    }

    /// The stack of in-place cross-FFT layers, run after the base FFTs.
    #[inline(always)]
    unsafe fn cross_ffts(&self, output: &mut [Complex<T>]) {
        unsafe {
            let out: &mut [Complex<V::ScalarType>] = workaround_transmute_mut(output);

            let mut cross_fft_len = self.base_len;
            let mut layer_twiddles: &[V] = &self.twiddles;

            while cross_fft_len < out.len() {
                let num_columns = cross_fft_len;
                cross_fft_len *= RADIX;

                // Dispatch once per layer rather than once per chunk, so each layer runs a single
                // monomorphized loop over its chunks. Mirrors the scalar `RadixN`.
                cross_layer_chunks::<V, 4>(out, layer_twiddles, num_columns, &self.butterfly4);

                // skip past all the twiddle factors used in this layer
                let twiddle_offset = num_columns.div_ceil(V::COMPLEX_PER_VECTOR) * (RADIX - 1);
                layer_twiddles = &layer_twiddles[twiddle_offset..];
            }
        }
    }

    #[inline(always)]
    pub fn process_single_immutable(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.transpose(input, output);
        self.base_fft.process_with_scratch(output, scratch);
        unsafe { self.cross_ffts(output) };
    }

    #[inline(always)]
    pub fn process_single_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.transpose(input, output);
        // the input is free once the transpose is done, so use it as base scratch
        // when we weren't handed any of our own
        let base_scratch = if !scratch.is_empty() { scratch } else { input };
        self.base_fft.process_with_scratch(output, base_scratch);
        unsafe { self.cross_ffts(output) };
    }

    #[inline(always)]
    pub fn process_single(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (transposed, inner_scratch) = scratch.split_at_mut(self.len());
        self.transpose(buffer, transposed);
        // Run the base out of place, from the scratch back into the caller's chunk.
        // The cross layers are then in place on the chunk, and the whole thing ends
        // where the caller wants it without a final copy.
        self.base_fft
            .process_outofplace_with_scratch(transposed, buffer, inner_scratch);
        unsafe { self.cross_ffts(buffer) };
    }
    #[inline(always)]
    pub fn get_inplace_scratch_len(&self) -> usize {
        self.inplace_scratch_len
    }
    #[inline(always)]
    pub fn get_outofplace_scratch_len(&self) -> usize {
        self.outofplace_scratch_len
    }
    #[inline(always)]
    pub fn get_immutable_scratch_len(&self) -> usize {
        self.immut_scratch_len
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    pub fn fft_direction(&self) -> FftDirection {
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

    type NewFn<V> = dyn Fn(
        u32,
        Arc<dyn Fft<<V as SimdVector>::ScalarType>>,
    ) -> Arc<dyn Fft<<V as SimdVector>::ScalarType>>;

    /// Every empty, one-factor and two-factor recipe over each of `bases`, both directions.
    pub fn factor_pairs<V>(bases: &[usize], new_fn: &NewFn<V>)
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        for base in bases {
            let base_forward = construct_base(*base, FftDirection::Forward);
            let base_inverse = construct_base(*base, FftDirection::Inverse);

            for k in 0..3 {
                check::<V>(k, Arc::clone(&base_forward), new_fn);
                check::<V>(k, Arc::clone(&base_inverse), new_fn);
            }
        }
    }

    /// The base doesn't have to be a scratch-free butterfly. A composite base is a recursive
    /// recipe that needs its own scratch, which is the case `design_radixn` hits whenever a length
    /// has factors above 7 (for example 11 * 13 = 143).
    pub fn composite_base<V32, V64>(new_fn32: &NewFn<V32>, new_fn64: &NewFn<V64>)
    where
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
    {
        let mut planner64 = crate::FftPlannerScalar::<f64>::new();
        let mut planner32 = crate::FftPlannerScalar::<f32>::new();

        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            for base_len in [22, 26, 110, 143, 55, 65] {
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
                    check::<V32>(k, Arc::clone(&base32), new_fn32);
                    check::<V64>(k, Arc::clone(&base64), new_fn64);
                }
            }
        }
    }

    fn check<V>(k: u32, base_fft: Arc<dyn Fft<V::ScalarType>>, new_fn: &NewFn<V>)
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        let len = base_fft.len() * RADIX.pow(k);
        let direction = base_fft.fft_direction();
        let fft = new_fn(k, base_fft);

        check_fft_algorithm::<V::ScalarType>(fft.as_ref(), len, direction);
    }
}
