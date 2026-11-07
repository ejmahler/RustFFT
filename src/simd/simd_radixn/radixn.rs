//! The body of the SIMD `RadixN` implementations, shared by every SIMD backend.
//!
//! We execute one flat transpose akin to a bit reversal, then execute a base FFT, then execute a
//! stack of in-place cross-FFT layers over a single packed twiddle array. Each cross FFT can be
//! a different size, hence the name RadixN.
//!
//! Everything here is generic over `SimdVector`, from `simd_vector.rs`. A backend implements that
//! trait once per vector type and gets the algorithm, so `SimdRadixN` is the only copy of it.
//!
//! A column butterfly consumes `COMPLEX_PER_VECTOR` columns at a time, 2 for f32 and 1 for f64 and
//! the scalar adapter, so a column count that isn't a whole number of vectors leaves a column over.
//! The count starts at `base_len` and only ever grows by whole factors, so for f32 an odd base
//! means an odd count at every layer, and no factor reordering helps: the first layer is the one
//! with the fewest columns. Each layer therefore ends with a partial column, one complex loaded
//! into the low half of a vector and only the low half stored back. The butterflies are lane
//! independent, so the same vector butterfly computes it, with the high half left as zeros.

use std::any::TypeId;
use std::sync::Arc;

use num_complex::Complex;

use crate::array_utils::{
    ArcSlice, TransposeFactor, choose_scratch, reverse_remainders, workaround_transmute,
    workaround_transmute_mut,
};
use crate::common::FftNum;
use crate::simd::simd_vector::SimdVector;
use crate::{Fft, FftDirection};

use super::{
    cross_layer::{BaseLayer, CrossLayer, LayerFactory},
    radix_factor::RadixFactor,
    transpose::{TransposeAlgorithm, table_transpose, unrolled_transpose},
};

/// FFT algorithm for lengths that factor into small radixes.
/// This is generic over SIMD instruction set and is designed to be embedded into a
/// feature-specific struct (such as SseRadixN or NeonRadixN), and not created directly.
pub struct SimdRadixN<V: SimdVector, T: FftNum> {
    transpose_algorithm: TransposeAlgorithm,
    cross_ffts: Arc<dyn CrossLayer<V, T>>,

    base_len: usize,
    len: usize,
    direction: FftDirection,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,
}

impl<V: SimdVector, T: FftNum> SimdRadixN<V, T> {
    /// Constructs a SimdRadixN which computes FFTs of length `factor_product * base_fft.len()`.
    #[inline(always)]
    pub fn new(
        factors: &[RadixFactor],
        base_fft: Arc<dyn Fft<T>>,
        layer_factory: impl LayerFactory<V, T>,
    ) -> Self {
        // Internal sanity check: Make sure that the vector's scalar type is T.
        // This struct has two generic parameters V and T, but T must always be V's scalar type,
        // and they are only kept separate to help work around the lack of specialization.
        assert_eq!(TypeId::of::<V::ScalarType>(), TypeId::of::<T>());

        let base_len = base_fft.len();
        let direction = base_fft.fft_direction();

        // set up our cross FFT butterfly instances. simultaneously, compute the number of twiddles
        let mut cross_fft_len = base_len;
        let mut twiddle_count = 0;

        for factor in factors {
            // Twiddles are stored a vector at a time, so a layer needs one chunk per vector column.
            // A column count that isn't a whole number of vectors gets one more chunk for the
            // partial column, of which only the first twiddle is used.
            twiddle_count += cross_fft_len.div_ceil(V::COMPLEX_PER_VECTOR) * (factor.radix() - 1);
            cross_fft_len *= factor.radix();
        }
        let len = cross_fft_len;

        let transpose_algorithm = if let Some(first_factor) = factors.first() {
            if factors.len() > 1 {
                // We have multiple factors, so we have to do a table transpose.

                // set up our list of transpose factors - it's the same list but reversed, and we want to
                // collapse duplicates. Note that we are only de-duplicating adjacent factors: if we're
                // passed 7 * 2 * 7, we can't collapse the sevens because the exact order matters.
                let mut transpose_factors: Vec<TransposeFactor> = Vec::with_capacity(factors.len());
                for f in factors.iter().rev() {
                    if let Some(last) = transpose_factors.last_mut()
                        && last.factor == *f
                    {
                        last.count += 1;
                    } else {
                        transpose_factors.push(TransposeFactor {
                            factor: *f,
                            count: 1,
                        });
                    }
                }

                // Precompute where each column lands. Working this out per call costs two hardware divides
                // and an out-of-line `reverse_remainders` call per column, which is a large share of a
                // short FFT, and much more so on x86 where a 64-bit divide takes tens of cycles.
                let width = len / base_len;
                let reversed_columns: Box<[usize]> = (0..width)
                    .map(|x| reverse_remainders(x, &transpose_factors))
                    .collect();
                // `table_transpose` indexes unchecked, and relies on this.
                assert!(reversed_columns.iter().all(|&r| r < width));

                TransposeAlgorithm::TableTranspose(*first_factor, reversed_columns)
            } else {
                // We only have one factor, so we can do an unrolled transpose
                TransposeAlgorithm::UnrolledTranspose(*first_factor)
            }
        } else {
            TransposeAlgorithm::Copy
        };

        // Same packing as the scalar RadixN: all layers in one array, bottom layer first, and
        // within a layer, (radix - 1) twiddles per column. The difference is that a "column" here
        // is a whole vector of columns, so each entry is a twiddle chunk rather than one twiddle.
        let mut twiddle_factors: Vec<V> = Vec::with_capacity(twiddle_count);
        let mut cross_fft_len = base_len;
        for factor in factors.iter() {
            let num_vector_columns = cross_fft_len.div_ceil(V::COMPLEX_PER_VECTOR);
            cross_fft_len *= factor.radix();

            for i in 0..num_vector_columns {
                for k in 1..factor.radix() {
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
        let mut cross_ffts = Arc::new(BaseLayer(base_fft)) as Arc<dyn CrossLayer<V, T>>;
        let mut cross_fft_len = base_len;
        let mut twiddle_start = 0;
        for factor in factors {
            let twiddle_len = cross_fft_len.div_ceil(V::COMPLEX_PER_VECTOR) * (factor.radix() - 1);
            let twiddle_end = twiddle_start + twiddle_len;
            cross_fft_len *= factor.radix();

            let layer_twiddles =
                ArcSlice::new(Arc::clone(&twiddle_factors), twiddle_start..twiddle_end);
            cross_ffts = layer_factory.construct_layer(
                *factor,
                cross_ffts,
                cross_fft_len,
                layer_twiddles,
                direction,
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
            TransposeAlgorithm::UnrolledTranspose(unroll_factor) => match *unroll_factor {
                RadixFactor::Factor2 => unsafe {
                    unrolled_transpose::<V, 2>(self.base_len, input, output)
                },
                RadixFactor::Factor3 => unsafe {
                    unrolled_transpose::<V, 3>(self.base_len, input, output)
                },
                RadixFactor::Factor4 => unsafe {
                    unrolled_transpose::<V, 4>(self.base_len, input, output)
                },
                RadixFactor::Factor5 => unsafe {
                    unrolled_transpose::<V, 5>(self.base_len, input, output)
                },
                RadixFactor::Factor6 => unsafe {
                    unrolled_transpose::<V, 6>(self.base_len, input, output)
                },
                RadixFactor::Factor7 => unsafe {
                    unrolled_transpose::<V, 7>(self.base_len, input, output)
                },
            },
            TransposeAlgorithm::TableTranspose(unroll_factor, reversed_columns) => {
                match *unroll_factor {
                    RadixFactor::Factor2 => unsafe {
                        table_transpose::<V, 2>(self.base_len, reversed_columns, input, output)
                    },
                    RadixFactor::Factor3 => unsafe {
                        table_transpose::<V, 3>(self.base_len, reversed_columns, input, output)
                    },
                    RadixFactor::Factor4 => unsafe {
                        table_transpose::<V, 4>(self.base_len, reversed_columns, input, output)
                    },
                    RadixFactor::Factor5 => unsafe {
                        table_transpose::<V, 5>(self.base_len, reversed_columns, input, output)
                    },
                    RadixFactor::Factor6 => unsafe {
                        table_transpose::<V, 6>(self.base_len, reversed_columns, input, output)
                    },
                    RadixFactor::Factor7 => unsafe {
                        table_transpose::<V, 7>(self.base_len, reversed_columns, input, output)
                    },
                }
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
        unsafe { self.cross_ffts.process_layer(output, scratch) };
    }

    #[inline(always)]
    pub fn process_single_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.transpose(input, output);
        unsafe {
            self.cross_ffts
                .process_layer(output, choose_scratch(input, scratch))
        };
    }

    #[inline(always)]
    pub fn process_single(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (transposed, inner_scratch) = scratch.split_at_mut(self.len());
        self.transpose(buffer, transposed);
        unsafe {
            self.cross_ffts
                .process_layer_outofplace(transposed, buffer, inner_scratch)
        };
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

    type NewFn<T> = dyn Fn(&[RadixFactor], Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>>;

    const FACTOR_LIST: &[RadixFactor] = &[
        RadixFactor::Factor2,
        RadixFactor::Factor3,
        RadixFactor::Factor4,
        RadixFactor::Factor5,
        RadixFactor::Factor6,
        RadixFactor::Factor7,
    ];

    /// Every empty, one-factor and two-factor recipe over each of `bases`, both directions.
    pub fn factor_pairs<V>(bases: &[usize], new_fn: &NewFn<V::ScalarType>)
    where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        for base in bases {
            let base_forward = construct_base(*base, FftDirection::Forward);
            let base_inverse = construct_base(*base, FftDirection::Inverse);

            check::<V>(&[], Arc::clone(&base_forward), new_fn);
            check::<V>(&[], Arc::clone(&base_inverse), new_fn);

            for factor_a in FACTOR_LIST {
                check::<V>(&[*factor_a], Arc::clone(&base_forward), new_fn);
                check::<V>(&[*factor_a], Arc::clone(&base_inverse), new_fn);

                for factor_b in FACTOR_LIST {
                    let factors = &[*factor_a, *factor_b];
                    check::<V>(factors, Arc::clone(&base_forward), new_fn);
                    check::<V>(factors, Arc::clone(&base_inverse), new_fn);
                }
            }
        }
    }

    /// The base doesn't have to be a scratch-free butterfly. A composite base is a recursive
    /// recipe that needs its own scratch, which is the case `design_radixn` hits whenever a length
    /// has factors above 7 (for example 11 * 13 = 143).
    pub fn composite_base<V32, V64>(
        new_fn32: &NewFn<V32::ScalarType>,
        new_fn64: &NewFn<V64::ScalarType>,
    ) where
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
    {
        let mut planner64 = crate::FftPlannerScalar::<f64>::new();
        let mut planner32 = crate::FftPlannerScalar::<f32>::new();

        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            // odd base, so f32 runs its partial column path with a composite base too
            for base_len in [143, 55, 65] {
                let base = planner64.plan_fft(base_len, direction);
                assert!(
                    base.get_inplace_scratch_len() > 0,
                    "base {} was expected to need scratch",
                    base_len
                );
                check::<V64>(
                    &[RadixFactor::Factor6, RadixFactor::Factor4],
                    base,
                    new_fn64,
                );

                let base = planner32.plan_fft(base_len, direction);
                check::<V32>(
                    &[RadixFactor::Factor6, RadixFactor::Factor4],
                    base,
                    new_fn32,
                );
            }

            // even base
            for base_len in [22, 26, 110] {
                let base = planner32.plan_fft(base_len, direction);
                assert!(
                    base.get_inplace_scratch_len() > 0,
                    "base {} was expected to need scratch",
                    base_len
                );
                check::<V32>(
                    &[RadixFactor::Factor3, RadixFactor::Factor4],
                    base,
                    new_fn32,
                );

                let base = planner64.plan_fft(base_len, direction);
                check::<V64>(
                    &[RadixFactor::Factor3, RadixFactor::Factor4],
                    base,
                    new_fn64,
                );
            }
        }
    }

    /// The recipes the spike was benchmarked on, so the layer shapes that actually matter stay
    /// covered. The benchmarked lengths used much bigger bases, but the base is just a butterfly
    /// that the other tests already cover, so a small one is used here. That keeps the naive `Dft`
    /// the result is checked against affordable.
    pub fn large_recipes<V32, V64>(
        new_fn32: &NewFn<V32::ScalarType>,
        new_fn64: &NewFn<V64::ScalarType>,
    ) where
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
    {
        use RadixFactor::*;
        // (factors, f64 base, f32 base). The f32 bases alternate between even and odd, so both the
        // whole-vector and the partial column path are covered at every one of these depths.
        let cases: [(&[RadixFactor], usize, usize); 5] = [
            (&[Factor6, Factor6, Factor6], 1, 3),          // 216, 648
            (&[Factor6, Factor5, Factor5], 1, 2),          // 150, 300
            (&[Factor6, Factor6, Factor4], 1, 3),          // 144, 432
            (&[Factor6, Factor6, Factor3], 1, 2),          // 108, 216
            (&[Factor6, Factor6, Factor6, Factor4], 1, 1), // 864, 864
        ];
        recipes::<V32, V64>(&cases, new_fn32, new_fn64);
    }

    /// The deepest benchmarked recipe, six layers. A base of 1 still leaves 14400 points, and the
    /// naive `Dft` they are checked against takes minutes on a debug build, so this one is kept out
    /// of the normal run. Run it with
    /// `cargo test --release -- --ignored radixn_six_layers`.
    pub fn six_layers<V32, V64>(
        new_fn32: &NewFn<V32::ScalarType>,
        new_fn64: &NewFn<V64::ScalarType>,
    ) where
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
    {
        use RadixFactor::*;
        let cases: [(&[RadixFactor], usize, usize); 1] = [(
            &[Factor6, Factor6, Factor5, Factor5, Factor4, Factor4],
            1,
            1,
        )]; // 14400, 14400
        recipes::<V32, V64>(&cases, new_fn32, new_fn64);
    }

    /// Runs each (factors, f64 base, f32 base) case in both directions.
    fn recipes<V32, V64>(
        cases: &[(&[RadixFactor], usize, usize)],
        new_fn32: &NewFn<V32::ScalarType>,
        new_fn64: &NewFn<V64::ScalarType>,
    ) where
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
    {
        for (factors, base64, base32) in cases {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                check::<V64>(factors, construct_base(*base64, direction), new_fn64);
                check::<V32>(factors, construct_base(*base32, direction), new_fn32);
            }
        }
    }

    fn check<V>(
        factors: &[RadixFactor],
        base_fft: Arc<dyn Fft<V::ScalarType>>,
        new_fn: &NewFn<V::ScalarType>,
    ) where
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        let len = base_fft.len() * factors.iter().map(|f| f.radix()).product::<usize>();
        let direction = base_fft.fft_direction();
        let fft = new_fn(factors, base_fft);

        check_fft_algorithm::<V::ScalarType>(fft.as_ref(), len, direction);
    }
}
