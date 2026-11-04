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

use crate::array_utils::{
    ArcSlice, DoubleBuf, reverse_bits, workaround_transmute, workaround_transmute_mut,
};
use crate::common::{FftNum, RadixFactor};
use crate::simd::parallel_fft::ParallelFft;
use crate::simd::simd_radixn::cross_layer;
use crate::simd::simd_transpose::{TransposeAlgorithm, table_transpose, unrolled_transpose};
use crate::simd::simd_vector::SimdVector;
use crate::{Fft, FftDirection};

const RADIX: usize = 4;

/// FFT algorithm for lengths that factor into base * 4**k
/// This is generic over SIMD instruction set and is designed to be embedded into a
/// feature-specific struct (such as SseRadix4 or NeonRadix4), and not created directly.
pub struct SimdRadix4<V: SimdVector, T: FftNum, L: Layer<V, T>> {
    transpose_algorithm: TransposeAlgorithm,
    cross_ffts: SimdRadix4Child<V, T, L>,

    base_len: usize,
    len: usize,
    direction: FftDirection,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,

    _phantom: std::marker::PhantomData<(V, T)>,
}

impl<V: SimdVector, T: FftNum, L: Layer<V, T>> SimdRadix4<V, T, L> {
    /// Constructs a SimdRadix4 which computes FFTs of length `k**4 * base_fft.len()`.
    #[inline(always)]
    pub fn new(
        k: u32,
        base_fft: Arc<dyn Fft<T>>,
        mut construct_layer_fn: impl FnMut(
            ArcSlice<V>,
            SimdRadix4Child<V, T, L>,
            FftDirection,
        ) -> Box<L>,
    ) -> Self {
        // Internal sanity check: Make sure that the vector's scalar type is T.
        // This struct has two generic parameters V and T, but T must always be V's scalar type,
        // and they are only kept separate to help work around the lack of specialization.
        assert_eq!(TypeId::of::<V::ScalarType>(), TypeId::of::<T>());

        let base_len = base_fft.len();
        let len = base_len << (2 * k);
        let direction = base_fft.fft_direction();

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

        // Set up our twiddle factors.
        // Same packing as RadixN: all layers in one array, bottom layer first, and
        // within a layer, (radix - 1) = 3 twiddles per column.
        let required_twiddle_count = |layer: u32| {
            let cross_len = base_len << (2 * layer);
            cross_len.div_ceil(V::COMPLEX_PER_VECTOR) * (RADIX - 1)
        };

        let twiddle_count: usize = (0..k).map(required_twiddle_count).sum();
        let mut twiddle_factors = Vec::with_capacity(twiddle_count);
        let mut cross_fft_len = base_len;
        for _ in 0..k {
            let num_vector_columns = cross_fft_len.div_ceil(V::COMPLEX_PER_VECTOR);
            cross_fft_len *= RADIX;

            for i in 0..num_vector_columns {
                for k in 1..RADIX {
                    unsafe {
                        twiddle_factors.push(V::make_mixedradix_twiddle_chunk(
                            i * V::COMPLEX_PER_VECTOR,
                            k,
                            cross_fft_len,
                            direction,
                        ));
                    }
                }
            }
        }
        let twiddle_factors = Arc::from(twiddle_factors);

        // figure out how much scratch space we need to request from callers
        let base_inplace_scratch = base_fft.get_inplace_scratch_len();
        // the in-place path transposes into its own scratch and runs the base out of place from
        // there back into the caller's buffer, so it needs the base's out-of-place scratch on top
        let inplace_scratch_len = if base_inplace_scratch > len {
            len
        } else {
            len + base_inplace_scratch
        };
        let outofplace_scratch_len = if base_inplace_scratch > len {
            base_inplace_scratch
        } else {
            0
        };

        // Build our tree of child FFTs
        let mut cross_ffts = SimdRadix4Child::Base(base_fft);
        let mut twiddle_start = 0;
        for i in 0..k {
            let twiddle_len = required_twiddle_count(i);
            let twiddle_end = twiddle_start + twiddle_len;

            let layer_twiddles =
                ArcSlice::new(Arc::clone(&twiddle_factors), twiddle_start..twiddle_end);
            cross_ffts = SimdRadix4Child::ChildLayer(
                construct_layer_fn(layer_twiddles, cross_ffts, direction),
                std::marker::PhantomData,
            );

            twiddle_start = twiddle_end;
        }

        Self {
            transpose_algorithm,
            cross_ffts,

            base_len,
            len,
            direction,

            inplace_scratch_len,
            outofplace_scratch_len,
            immut_scratch_len: base_inplace_scratch,

            _phantom: std::marker::PhantomData,
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
            TransposeAlgorithm::UnrolledTranspose(_) => unsafe {
                unrolled_transpose::<V, 4>(self.base_len, input, output)
            },
            TransposeAlgorithm::TableTranspose(_, reversed_columns) => unsafe {
                table_transpose::<V, 4>(self.base_len, reversed_columns, input, output)
            },
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

        match &self.cross_ffts {
            SimdRadix4Child::Base(base_fft) => {
                // We have no cross layers, so just directly call the child FFT
                base_fft.process_with_scratch(output, scratch);
            }
            SimdRadix4Child::ChildLayer(layer, _) => {
                // Call the top cross layer, which will recursively call the child layers
                unsafe {
                    layer.process_layer(output, scratch);
                };
            }
        }
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

        match &self.cross_ffts {
            SimdRadix4Child::Base(base_fft) => {
                // We have no cross layers, so just directly call the child FFT
                base_fft.process_with_scratch(output, base_scratch);
            }
            SimdRadix4Child::ChildLayer(layer, _) => {
                // Call the top cross layer, which will recursively call the child layers
                unsafe {
                    layer.process_layer(output, base_scratch);
                };
            }
        }
    }

    #[inline(always)]
    pub fn process_single(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (transposed, inner_scratch) = scratch.split_at_mut(self.len());
        self.transpose(buffer, transposed);

        match &self.cross_ffts {
            SimdRadix4Child::Base(base_fft) => {
                // We have no cross layers, so just directly call the child FFT
                base_fft.process_outofplace_with_scratch(transposed, buffer, inner_scratch);
            }
            SimdRadix4Child::ChildLayer(layer, _) => {
                // Call the top cross layer, which will recursively call the child layers
                unsafe {
                    layer.process_layer_outofplace(transposed, buffer, inner_scratch);
                };
            }
        }
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

pub trait Layer<V: SimdVector, T: FftNum> {
    // SAFETY:
    // - The required CPU features for `V` must be present
    unsafe fn process_layer(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    // SAFETY:
    // - The required CPU features for `V` must be present
    unsafe fn process_layer_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );
}

pub enum SimdRadix4Child<V: SimdVector, T: FftNum, L: Layer<V, T>> {
    Base(Arc<dyn Fft<T>>),
    ChildLayer(Box<L>, std::marker::PhantomData<V>),
}

pub struct SimdRadix4Layer<V: SimdVector, T: FftNum, L: Layer<V, T>, B: ParallelFft<V, 4>> {
    twiddles: ArcSlice<V>,

    child_fft: SimdRadix4Child<V, T, L>,
    butterfly: B,
}

impl<V: SimdVector, T: FftNum, L: Layer<V, T>, B: ParallelFft<V, 4>> SimdRadix4Layer<V, T, L, B> {
    pub fn new(twiddles: ArcSlice<V>, child_fft: SimdRadix4Child<V, T, L>, butterfly: B) -> Self {
        Self {
            twiddles,
            child_fft,
            butterfly,
        }
    }

    #[inline(always)]
    pub unsafe fn process_layer(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // first, process the child FFTs, IE we're doing a preorder recursive traversal
        match &self.child_fft {
            SimdRadix4Child::Base(base_fft) => base_fft.process_with_scratch(buffer, scratch),
            SimdRadix4Child::ChildLayer(child, _) => {
                let inner_fft_len = buffer.len() / RADIX;
                let mut chunks = buffer.chunks_exact_mut(inner_fft_len);
                for _ in 0..RADIX {
                    let chunk = unsafe { chunks.next().unwrap_unchecked() };

                    unsafe { child.process_layer(chunk, scratch) };
                }
                debug_assert!(chunks.next().is_none() && chunks.into_remainder().is_empty());
            }
        }

        // Now that the children are done, handle this layer's cross ffts
        let buffer = unsafe { workaround_transmute_mut(buffer) };
        let num_columns = buffer.len() / 4;
        unsafe { cross_layer::<V, RADIX>(buffer, &self.twiddles, num_columns, &self.butterfly) }
    }

    #[inline(always)]
    pub unsafe fn process_layer_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        // first, process the child FFTs, IE we're doing a preorder recursive traversal
        match &self.child_fft {
            SimdRadix4Child::Base(base_fft) => {
                let base_scratch = if scratch.is_empty() {
                    &mut output[..]
                } else {
                    scratch
                };
                base_fft.process_with_scratch(input, base_scratch)
            }
            SimdRadix4Child::ChildLayer(child, _) => {
                let inner_fft_len = input.len() / RADIX;
                let mut child_buffer = &mut input[..];
                while let Some((chunk, remainder)) =
                    child_buffer.split_at_mut_checked(inner_fft_len)
                {
                    child_buffer = remainder;

                    unsafe { child.process_layer(chunk, scratch) };
                }
            }
        }

        // Now that the children are done, handle this layer's cross ffts
        let buffer = DoubleBuf {
            input: unsafe { workaround_transmute(input) },
            output: unsafe { workaround_transmute_mut(output) },
        };
        unsafe { cross_layer::<V, RADIX>(buffer, &self.twiddles, input.len() / 4, &self.butterfly) }
    }
}

// Boilerplate macro for Radix4 that implements the layer struct, implements the SimdRadix4Layer trait for it,
// and declares wrapper functions with the appropriate target feature attributes
macro_rules! boilerplate_simd_radix4 {
    ($struct_name: ident, $layer_struct_name:ident, $(#[$attr:meta])*) => {
        struct $layer_struct_name<V: SimdVector, T: FftNum>(crate::simd::simd_radix4::SimdRadix4Layer<V, T, $layer_struct_name<V, T>, V::Butterfly4>);
        impl<V: SimdVector, T: FftNum> $layer_struct_name<V, T> {
            pub fn new(
                twiddles: crate::array_utils::ArcSlice<V>,
                child: crate::simd::simd_radix4::SimdRadix4Child<V, T, $layer_struct_name<V, T>>,
                direction: crate::FftDirection,
            ) -> Box<Self> {
                Box::new(Self(crate::simd::simd_radix4::SimdRadix4Layer::new(twiddles, child, unsafe {
                    V::make_butterfly4(direction)
                })))
            }
        }
        impl<V: SimdVector, T: FftNum> crate::simd::simd_radix4::Layer<V, T> for $layer_struct_name<V, T> {
            $(#[$attr])*
            unsafe fn process_layer(
                &self,
                buffer: &mut [num_complex::Complex<T>],
                base_scratch: &mut [num_complex::Complex<T>],
            ) {
                unsafe { self.0.process_layer(buffer, base_scratch) }
            }

            $(#[$attr])*
            unsafe fn process_layer_outofplace(
                &self,
                input: &mut [num_complex::Complex<T>],
                output: &mut [num_complex::Complex<T>],
                base_scratch: &mut [num_complex::Complex<T>],
            ) {
                unsafe {
                    self.0
                        .process_layer_outofplace(input, output, base_scratch)
                }
            }
        }
    };
}
pub(crate) use boilerplate_simd_radix4;

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
