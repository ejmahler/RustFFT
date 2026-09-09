//! The body of the SIMD `RadixN` implementations, shared by every SIMD backend.
//!
//! This mirrors `src/algorithm/radixn.rs`: one flat transpose down to a base FFT, then a stack of
//! in-place cross-FFT layers over a single packed twiddle array. The only difference is that the
//! cross-FFT layers use SIMD column butterflies instead of the scalar ones, so a whole vector of
//! columns is processed per butterfly call.
//!
//! Everything here is generic over `RadixNVector`, which is the handful of vector operations the
//! algorithm needs. A backend implements that trait once per vector type and gets the algorithm,
//! so `SimdRadixN` is the only copy of it.
//!
//! Because a column butterfly consumes `COMPLEX_PER_VECTOR` columns at a time, the column count at
//! every layer has to be a whole number of vectors. The column count starts at `base_len` and only
//! ever grows by whole factors, so requiring `base_len % COMPLEX_PER_VECTOR == 0` is enough. That
//! is 1 for f64 (no restriction) and 2 for f32.

use std::any::TypeId;
use std::sync::Arc;

use num_complex::Complex;

use crate::array_utils::{factor_transpose, workaround_transmute_mut, TransposeFactor};
use crate::common::{FftNum, RadixFactor};
use crate::{Direction, Fft, FftDirection, Length};

/// Everything `SimdRadixN` needs from a backend's vector type.
///
/// Radix 2 and 4 are vector-generic in every backend, so they map straight onto the backend's
/// vector trait. Radix 3, 5, 6 and 7 only exist as element-type-specific structs holding
/// precomputed twiddles, so this trait pairs each vector type with its own set of them. For f64 a
/// vector is one complex number and the plain `perform_fft_direct` is already a single column; for
/// f32 a vector is two complex numbers and `perform_parallel_fft_direct` does two columns at once.
///
/// Safety: every method here requires the current machine to support the backend's SIMD
/// instruction set.
pub trait RadixNVector: Copy + Send + Sync + Sized {
    const COMPLEX_PER_VECTOR: usize;

    /// The scalar this vector holds. Always the same type as the `T` of the `SimdRadixN` using it.
    type ScalarType: FftNum;

    /// The backend's precomputed 90 degree rotation, which the radix 4 butterfly needs.
    type Rotation: Copy + Send + Sync;

    type Butterfly3: Send + Sync;
    type Butterfly5: Send + Sync;
    type Butterfly6: Send + Sync;
    type Butterfly7: Send + Sync;

    unsafe fn load(data: &[Complex<Self::ScalarType>], index: usize) -> Self;
    unsafe fn store(data: &mut [Complex<Self::ScalarType>], value: Self, index: usize);

    /// Pairwise multiply the complex numbers in `left` with the complex numbers in `right`.
    unsafe fn mul_complex(left: Self, right: Self) -> Self;

    /// Generates a chunk of twiddle factors starting at (X,Y) and incrementing X
    /// `COMPLEX_PER_VECTOR` times.
    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: FftDirection,
    ) -> Self;

    unsafe fn make_rotate90(direction: FftDirection) -> Self::Rotation;
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3;
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5;
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6;
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7;

    /// Each of these interprets the input as rows of a `COMPLEX_PER_VECTOR`-by-N 2D array, and
    /// computes parallel butterflies down the columns of the 2D array.
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2];
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3];
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Self::Rotation) -> [Self; 4];
    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5];
    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6];
    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7];

    /// The three `fft_helper_*` wrappers from the backend's `*_common.rs`, which run the whole
    /// chunk loop with the backend's target feature enabled so that things like loading twiddle
    /// factor registers can be lifted out of the loop.
    unsafe fn fft_helper_immut<E>(
        input: &[E],
        output: &mut [E],
        scratch: &mut [E],
        chunk_size: usize,
        required_scratch: usize,
        chunk_fn: impl FnMut(&[E], &mut [E], &mut [E]),
    );
    unsafe fn fft_helper_outofplace<E>(
        input: &mut [E],
        output: &mut [E],
        scratch: &mut [E],
        chunk_size: usize,
        required_scratch: usize,
        chunk_fn: impl FnMut(&mut [E], &mut [E], &mut [E]),
    );
    unsafe fn fft_helper_inplace<E>(
        buffer: &mut [E],
        scratch: &mut [E],
        chunk_size: usize,
        required_scratch: usize,
        chunk_fn: impl FnMut(&mut [E], &mut [E]),
    );
}

/// The per-layer cross-FFT kernels, holding whatever precomputed state each radix needs.
enum InternalRadixFactor<V: RadixNVector> {
    Factor2,
    Factor3(V::Butterfly3),
    Factor4(V::Rotation),
    Factor5(V::Butterfly5),
    Factor6(V::Butterfly6),
    Factor7(V::Butterfly7),
}

impl<V: RadixNVector> InternalRadixFactor<V> {
    fn radix(&self) -> usize {
        match self {
            InternalRadixFactor::Factor2 => 2,
            InternalRadixFactor::Factor3(_) => 3,
            InternalRadixFactor::Factor4(_) => 4,
            InternalRadixFactor::Factor5(_) => 5,
            InternalRadixFactor::Factor6(_) => 6,
            InternalRadixFactor::Factor7(_) => 7,
        }
    }
}

/// FFT algorithm for lengths that factor into small radixes, SIMD accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub struct SimdRadixN<V: RadixNVector, T> {
    twiddles: Box<[V]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    factors: Box<[TransposeFactor]>,
    butterflies: Box<[InternalRadixFactor<V>]>,

    len: usize,
    direction: FftDirection,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,
}

impl<V: RadixNVector, T: FftNum> SimdRadixN<V, T> {
    /// Constructs a SimdRadixN which computes FFTs of length `factor_product * base_fft.len()`.
    pub fn new(factors: &[RadixFactor], base_fft: Arc<dyn Fft<T>>) -> Self {
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
            factors.is_empty() || base_len % complex_per_vector == 0,
            "SimdRadixN requires a base length divisible by {}, got {}",
            complex_per_vector,
            base_len
        );

        // set up our cross FFT butterfly instances. simultaneously, compute the number of twiddles
        let mut butterflies = Vec::with_capacity(factors.len());
        let mut cross_fft_len = base_len;
        let mut twiddle_count = 0;

        for factor in factors {
            // twiddles are stored a vector at a time, so a layer needs one chunk per vector column
            twiddle_count += (cross_fft_len / complex_per_vector) * (factor.radix() - 1);

            butterflies.push(unsafe {
                match factor {
                    RadixFactor::Factor2 => InternalRadixFactor::Factor2,
                    RadixFactor::Factor3 => {
                        InternalRadixFactor::Factor3(V::make_butterfly3(direction))
                    }
                    RadixFactor::Factor4 => {
                        InternalRadixFactor::Factor4(V::make_rotate90(direction))
                    }
                    RadixFactor::Factor5 => {
                        InternalRadixFactor::Factor5(V::make_butterfly5(direction))
                    }
                    RadixFactor::Factor6 => {
                        InternalRadixFactor::Factor6(V::make_butterfly6(direction))
                    }
                    RadixFactor::Factor7 => {
                        InternalRadixFactor::Factor7(V::make_butterfly7(direction))
                    }
                }
            });

            cross_fft_len *= factor.radix();
        }
        let len = cross_fft_len;

        // set up our list of transpose factors - it's the same list but reversed, and we want to
        // collapse duplicates. Note that we are only de-duplicating adjacent factors: if we're
        // passed 7 * 2 * 7, we can't collapse the sevens because the exact order matters.
        let mut transpose_factors: Vec<TransposeFactor> = Vec::with_capacity(factors.len());
        for f in factors.iter().rev() {
            let mut push_new = true;
            if let Some(last) = transpose_factors.last_mut() {
                if last.factor == *f {
                    last.count += 1;
                    push_new = false;
                }
            }
            if push_new {
                transpose_factors.push(TransposeFactor {
                    factor: *f,
                    count: 1,
                });
            }
        }

        // Same packing as the scalar RadixN: all layers in one array, bottom layer first, and
        // within a layer, (radix - 1) twiddles per column. The difference is that a "column" here
        // is a whole vector of columns, so each entry is a twiddle chunk rather than one twiddle.
        let mut twiddle_factors: Vec<V> = Vec::with_capacity(twiddle_count);
        let mut cross_fft_len = base_len;
        for factor in factors {
            let num_vector_columns = cross_fft_len / complex_per_vector;
            cross_fft_len *= factor.radix();

            for i in 0..num_vector_columns {
                for k in 1..factor.radix() {
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
        let inplace_scratch_len = if base_inplace_scratch > len {
            len + base_inplace_scratch
        } else {
            len
        };
        let outofplace_scratch_len = if base_inplace_scratch > len {
            base_inplace_scratch
        } else {
            0
        };

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

            base_fft,
            base_len,

            factors: transpose_factors.into_boxed_slice(),
            butterflies: butterflies.into_boxed_slice(),

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
        if let Some(unroll_factor) = self.factors.first() {
            // for performance, we really, really want to unroll the transpose, but we need to make
            // sure the output length is divisible by the unroll amount. choosing the first factor
            // seems to reliably perform well
            match unroll_factor.factor {
                RadixFactor::Factor2 => {
                    factor_transpose::<Complex<T>, 2>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor3 => {
                    factor_transpose::<Complex<T>, 3>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor4 => {
                    factor_transpose::<Complex<T>, 4>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor5 => {
                    factor_transpose::<Complex<T>, 5>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor6 => {
                    factor_transpose::<Complex<T>, 6>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor7 => {
                    factor_transpose::<Complex<T>, 7>(self.base_len, input, output, &self.factors)
                }
            }
        } else {
            // no factors, so just pass data straight to our base
            output.copy_from_slice(input);
        }
    }

    /// The stack of in-place cross-FFT layers, run after the base FFTs.
    unsafe fn cross_ffts(&self, output: &mut [Complex<T>]) {
        let out: &mut [Complex<V::ScalarType>] = workaround_transmute_mut(output);

        let mut cross_fft_len = self.base_len;
        let mut layer_twiddles: &[V] = &self.twiddles;

        for factor in self.butterflies.iter() {
            let num_columns = cross_fft_len;
            cross_fft_len *= factor.radix();

            for data in out.chunks_exact_mut(cross_fft_len) {
                match factor {
                    InternalRadixFactor::Factor2 => {
                        cross_layer::<V, 2, _>(data, layer_twiddles, num_columns, |v| {
                            V::column_butterfly2(v)
                        })
                    }
                    InternalRadixFactor::Factor3(bf) => {
                        cross_layer::<V, 3, _>(data, layer_twiddles, num_columns, |v| {
                            V::column_butterfly3(bf, v)
                        })
                    }
                    InternalRadixFactor::Factor4(rotation) => {
                        cross_layer::<V, 4, _>(data, layer_twiddles, num_columns, |v| {
                            V::column_butterfly4(v, *rotation)
                        })
                    }
                    InternalRadixFactor::Factor5(bf) => {
                        cross_layer::<V, 5, _>(data, layer_twiddles, num_columns, |v| {
                            V::column_butterfly5(bf, v)
                        })
                    }
                    InternalRadixFactor::Factor6(bf) => {
                        cross_layer::<V, 6, _>(data, layer_twiddles, num_columns, |v| {
                            V::column_butterfly6(bf, v)
                        })
                    }
                    InternalRadixFactor::Factor7(bf) => {
                        cross_layer::<V, 7, _>(data, layer_twiddles, num_columns, |v| {
                            V::column_butterfly7(bf, v)
                        })
                    }
                }
            }

            // skip past all the twiddle factors used in this layer
            let twiddle_offset = (num_columns / V::COMPLEX_PER_VECTOR) * (factor.radix() - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];
        }
    }

    unsafe fn perform_fft_immut(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.transpose(input, output);
        self.base_fft.process_with_scratch(output, scratch);
        self.cross_ffts(output);
    }

    unsafe fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.transpose(input, output);

        // the input is free once the transpose is done, so use it as base scratch when we weren't
        // handed any of our own
        let base_scratch = if !scratch.is_empty() { scratch } else { input };
        self.base_fft.process_with_scratch(output, base_scratch);

        self.cross_ffts(output);
    }
}

impl<V: RadixNVector, T: FftNum> Fft<T> for SimdRadixN<V, T> {
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
                |in_chunk, out_chunk, scratch| self.perform_fft_immut(in_chunk, out_chunk, scratch),
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
                |in_chunk, out_chunk, scratch| {
                    self.perform_fft_out_of_place(in_chunk, out_chunk, scratch)
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
                    let (self_scratch, inner_scratch) = scratch.split_at_mut(self.len());
                    self.perform_fft_out_of_place(chunk, self_scratch, inner_scratch);
                    chunk.copy_from_slice(self_scratch);
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
impl<V: RadixNVector, T> Length for SimdRadixN<V, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<V: RadixNVector, T> Direction for SimdRadixN<V, T> {
    #[inline(always)]
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

/// One cross-FFT layer: for each vector of columns, gather RADIX rows strided by `num_columns`,
/// apply the twiddles, run the column butterfly, scatter back.
///
/// Unrolled two vectors at a time, which is what the SIMD Radix4's `butterfly_4` does and is what
/// gets the two independent dependency chains needed to keep the FMA pipeline busy.
#[inline(always)]
unsafe fn cross_layer<V: RadixNVector, const RADIX: usize, F>(
    data: &mut [Complex<V::ScalarType>],
    twiddles: &[V],
    num_columns: usize,
    butterfly: F,
) where
    F: Fn([V; RADIX]) -> [V; RADIX],
{
    let complex_per_vector = V::COMPLEX_PER_VECTOR;
    let num_vector_columns = num_columns / complex_per_vector;
    let tw_stride = RADIX - 1;

    debug_assert!(twiddles.len() >= num_vector_columns * tw_stride);

    // The row-0 twiddle is always 1, so it's neither stored nor applied.
    let gather = |data: &[Complex<V::ScalarType>], idx: usize, tw_base: usize| -> [V; RADIX] {
        // row 0 first, so the array is fully initialized without `array::from_fn`, which is
        // newer than the crate MSRV
        let mut rows = [V::load(data, idx); RADIX];
        for (r, row) in rows.iter_mut().enumerate().skip(1) {
            let v = V::load(data, idx + r * num_columns);
            *row = V::mul_complex(v, *twiddles.get_unchecked(tw_base + r - 1));
        }
        rows
    };

    let mut vcol = 0;
    while vcol + 2 <= num_vector_columns {
        let idx = vcol * complex_per_vector;

        let a = gather(data, idx, vcol * tw_stride);
        let b = gather(data, idx + complex_per_vector, (vcol + 1) * tw_stride);

        let a = butterfly(a);
        let b = butterfly(b);

        for (r, (a_row, b_row)) in a.iter().zip(b.iter()).enumerate() {
            V::store(data, *a_row, idx + r * num_columns);
            V::store(data, *b_row, idx + complex_per_vector + r * num_columns);
        }

        vcol += 2;
    }

    // an odd vector column count leaves one behind
    if vcol < num_vector_columns {
        let idx = vcol * complex_per_vector;
        let a = butterfly(gather(data, idx, vcol * tw_stride));
        for (r, a_row) in a.iter().enumerate() {
            V::store(data, *a_row, idx + r * num_columns);
        }
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
    use rand::distributions::uniform::SampleUniform;

    const FACTOR_LIST: &[RadixFactor] = &[
        RadixFactor::Factor2,
        RadixFactor::Factor3,
        RadixFactor::Factor4,
        RadixFactor::Factor5,
        RadixFactor::Factor6,
        RadixFactor::Factor7,
    ];

    /// Every empty, one-factor and two-factor recipe over each of `bases`, both directions.
    pub fn factor_pairs<V>(bases: &[usize])
    where
        V: RadixNVector,
        V::ScalarType: Float + SampleUniform,
    {
        for base in bases {
            let base_forward = construct_base(*base, FftDirection::Forward);
            let base_inverse = construct_base(*base, FftDirection::Inverse);

            check::<V>(&[], Arc::clone(&base_forward));
            check::<V>(&[], Arc::clone(&base_inverse));

            for factor_a in FACTOR_LIST {
                check::<V>(&[*factor_a], Arc::clone(&base_forward));
                check::<V>(&[*factor_a], Arc::clone(&base_inverse));

                for factor_b in FACTOR_LIST {
                    let factors = &[*factor_a, *factor_b];
                    check::<V>(factors, Arc::clone(&base_forward));
                    check::<V>(factors, Arc::clone(&base_inverse));
                }
            }
        }
    }

    /// The base doesn't have to be a scratch-free butterfly. A composite base is a recursive
    /// recipe that needs its own scratch, which is the case `design_radixn` hits whenever a length
    /// has factors above 7 (for example 11 * 13 = 143).
    pub fn composite_base<V32, V64>()
    where
        V32: RadixNVector<ScalarType = f32>,
        V64: RadixNVector<ScalarType = f64>,
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
                check::<V64>(&[RadixFactor::Factor6, RadixFactor::Factor4], base);
            }

            // even base, usable by both element types
            for base_len in [22, 26, 110] {
                let base = planner32.plan_fft(base_len, direction);
                assert!(
                    base.get_inplace_scratch_len() > 0,
                    "base {} was expected to need scratch",
                    base_len
                );
                check::<V32>(&[RadixFactor::Factor3, RadixFactor::Factor4], base);

                let base = planner64.plan_fft(base_len, direction);
                check::<V64>(&[RadixFactor::Factor3, RadixFactor::Factor4], base);
            }
        }
    }

    /// The recipes the spike was benchmarked on, so the layer shapes that actually matter stay
    /// covered. The benchmarked lengths used much bigger bases, but the base is just a butterfly
    /// that the other tests already cover, so the smallest legal one is used here. That keeps the
    /// naive `Dft` the result is checked against affordable.
    pub fn large_recipes<V32, V64>()
    where
        V32: RadixNVector<ScalarType = f32>,
        V64: RadixNVector<ScalarType = f64>,
    {
        use RadixFactor::*;
        // (factors, f64 base, f32 base). f32 needs an even base, so it gets its own.
        let cases: [(&[RadixFactor], usize, usize); 5] = [
            (&[Factor6, Factor6, Factor6], 1, 2),          // 216, 432
            (&[Factor6, Factor5, Factor5], 1, 2),          // 150, 300
            (&[Factor6, Factor6, Factor4], 1, 2),          // 144, 288
            (&[Factor6, Factor6, Factor3], 1, 2),          // 108, 216
            (&[Factor6, Factor6, Factor6, Factor4], 1, 2), // 864, 1728
        ];
        recipes::<V32, V64>(&cases);
    }

    /// The deepest benchmarked recipe, six layers. The smallest legal base still leaves 14400
    /// (f64) and 28800 (f32) points, and the naive `Dft` they are checked against takes minutes
    /// on a debug build, so this one is kept out of the normal run. Run it with
    /// `cargo test --release -- --ignored radixn_six_layers`.
    pub fn six_layers<V32, V64>()
    where
        V32: RadixNVector<ScalarType = f32>,
        V64: RadixNVector<ScalarType = f64>,
    {
        use RadixFactor::*;
        let cases: [(&[RadixFactor], usize, usize); 1] = [(
            &[Factor6, Factor6, Factor5, Factor5, Factor4, Factor4],
            1,
            2,
        )]; // 14400, 28800
        recipes::<V32, V64>(&cases);
    }

    /// Runs each (factors, f64 base, f32 base) case in both directions.
    fn recipes<V32, V64>(cases: &[(&[RadixFactor], usize, usize)])
    where
        V32: RadixNVector<ScalarType = f32>,
        V64: RadixNVector<ScalarType = f64>,
    {
        for (factors, base64, base32) in cases {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                check::<V64>(factors, construct_base(*base64, direction));
                check::<V32>(factors, construct_base(*base32, direction));
            }
        }
    }

    fn check<V>(factors: &[RadixFactor], base_fft: Arc<dyn Fft<V::ScalarType>>)
    where
        V: RadixNVector,
        V::ScalarType: Float + SampleUniform,
    {
        let len = base_fft.len() * factors.iter().map(|f| f.radix()).product::<usize>();
        let direction = base_fft.fft_direction();
        let fft: SimdRadixN<V, V::ScalarType> = SimdRadixN::new(factors, base_fft);

        check_fft_algorithm::<V::ScalarType>(&fft, len, direction);
    }
}
