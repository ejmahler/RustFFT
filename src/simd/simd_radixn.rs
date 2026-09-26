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

use crate::array_utils::{reverse_remainders, workaround_transmute_mut, TransposeFactor};
use crate::common::{FftNum, RadixFactor};
use crate::{Direction, Fft, FftDirection, Length};

use super::simd_vector::SimdVector;

/// The per-layer cross-FFT kernels, holding whatever precomputed state each radix needs.
enum InternalRadixFactor<V: SimdVector> {
    Factor2,
    Factor3(V::Butterfly3),
    Factor4(V::Rotation),
    Factor5(V::Butterfly5),
    Factor6(V::Butterfly6),
    Factor7(V::Butterfly7),
}

impl<V: SimdVector> InternalRadixFactor<V> {
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
pub struct SimdRadixN<V: SimdVector, T> {
    twiddles: Box<[V]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    // The factor the transpose is unrolled by, and the output column of each input column. None
    // when there are no factors and the transpose is a plain copy.
    unroll_factor: Option<RadixFactor>,
    reversed_columns: Box<[usize]>,
    butterflies: Box<[InternalRadixFactor<V>]>,

    len: usize,
    direction: FftDirection,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,
}

impl<V: SimdVector, T: FftNum> SimdRadixN<V, T> {
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

        // Precompute where each column lands. Working this out per call costs two hardware divides
        // and an out-of-line `reverse_remainders` call per column, which is a large share of a
        // short FFT, and much more so on x86 where a 64-bit divide takes tens of cycles.
        let width = len / base_len;
        let reversed_columns: Box<[usize]> = (0..width)
            .map(|x| reverse_remainders(x, &transpose_factors))
            .collect();
        // `table_transpose` indexes unchecked, and relies on this.
        assert!(reversed_columns.iter().all(|&r| r < width));
        let unroll_factor = transpose_factors.first().map(|f| f.factor);

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

            unroll_factor,
            reversed_columns,
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
        if let Some(unroll_factor) = self.unroll_factor {
            // for performance, we really, really want to unroll the transpose, but we need to make
            // sure the output length is divisible by the unroll amount. choosing the first factor
            // seems to reliably perform well
            let (height, columns) = (self.base_len, &*self.reversed_columns);
            match unroll_factor {
                RadixFactor::Factor2 => table_transpose::<_, 2>(height, columns, input, output),
                RadixFactor::Factor3 => table_transpose::<_, 3>(height, columns, input, output),
                RadixFactor::Factor4 => table_transpose::<_, 4>(height, columns, input, output),
                RadixFactor::Factor5 => table_transpose::<_, 5>(height, columns, input, output),
                RadixFactor::Factor6 => table_transpose::<_, 6>(height, columns, input, output),
                RadixFactor::Factor7 => table_transpose::<_, 7>(height, columns, input, output),
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

            // Dispatch once per layer rather than once per chunk, so each layer runs a single
            // monomorphized loop over its chunks. Mirrors the scalar `RadixN`.
            match factor {
                InternalRadixFactor::Factor2 => V::cross_layer2(out, layer_twiddles, num_columns),
                InternalRadixFactor::Factor3(bf) => {
                    V::cross_layer3(out, layer_twiddles, num_columns, bf)
                }
                InternalRadixFactor::Factor4(rotation) => {
                    V::cross_layer4(out, layer_twiddles, num_columns, *rotation)
                }
                InternalRadixFactor::Factor5(bf) => {
                    V::cross_layer5(out, layer_twiddles, num_columns, bf)
                }
                InternalRadixFactor::Factor6(bf) => {
                    V::cross_layer6(out, layer_twiddles, num_columns, bf)
                }
                InternalRadixFactor::Factor7(bf) => {
                    V::cross_layer7(out, layer_twiddles, num_columns, bf)
                }
            }

            // skip past all the twiddle factors used in this layer
            let twiddle_offset = (num_columns / V::COMPLEX_PER_VECTOR) * (factor.radix() - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];
        }
    }
}

impl<V: SimdVector, T: FftNum> Fft<T> for SimdRadixN<V, T> {
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
impl<V: SimdVector, T> Length for SimdRadixN<V, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<V: SimdVector, T> Direction for SimdRadixN<V, T> {
    #[inline(always)]
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

/// Run `cross_layer` over every chunk of `data`, each `num_columns * RADIX` long.
///
/// This is `chunks_exact_mut` without the divide it does to find the chunk count. At short lengths
/// that one divide per layer is a measurable share of the whole FFT.
#[inline(always)]
pub(crate) unsafe fn cross_layer_chunks<V: SimdVector, const RADIX: usize, F>(
    data: &mut [Complex<V::ScalarType>],
    twiddles: &[V],
    num_columns: usize,
    butterfly: F,
) where
    F: Fn([V; RADIX]) -> [V; RADIX],
{
    let chunk_len = num_columns * RADIX;
    let mut rest = data;
    while rest.len() >= chunk_len {
        let (chunk, tail) = rest.split_at_mut(chunk_len);
        cross_layer::<V, RADIX, _>(chunk, twiddles, num_columns, &butterfly);
        rest = tail;
    }
    debug_assert!(rest.is_empty());
}

/// `factor_transpose` with the reversed column indices looked up instead of recomputed.
///
/// `reversed_columns[x]` is the output column of input column `x`, so its length is the width and
/// nothing here needs to divide. Every entry must be below the width, which `SimdRadixN::new`
/// asserts, and `D` must divide the width.
#[inline(always)]
fn table_transpose<T: Copy, const D: usize>(
    height: usize,
    reversed_columns: &[usize],
    input: &[T],
    output: &mut [T],
) {
    let width = reversed_columns.len();
    assert!(width % D == 0 && input.len() == width * height && output.len() == input.len());

    for (group, rev) in reversed_columns.chunks_exact(D).enumerate() {
        let x = group * D;
        let rev: &[usize; D] = rev.try_into().unwrap();
        for y in 0..height {
            let row = x + y * width;
            for (i, &r) in rev.iter().enumerate() {
                // Both indexes below are unchecked, so here is why neither can leave the slices.
                // Both slices are `width * height` long, asserted above, so an index is in range
                // as long as it stays below `width * height`.
                //
                // The read is from `row + i`, which is `group * D + y * width + i`. The loops cap
                // each term: `group` reaches `width / D - 1` because `chunks_exact(D)` over a
                // slice of length `width` yields `width / D` chunks, `y` reaches `height - 1`,
                // and `i` reaches `D - 1`. Substituting all three gives
                // `(width / D - 1) * D + (height - 1) * width + D - 1`, and since D divides
                // `width` that simplifies to `width * height - 1`.
                let value = unsafe { *input.get_unchecked(row + i) };

                // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
                // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
                // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
                // largest index is `height - 1 + (width - 1) * height`, which is
                // `width * height - 1`.
                unsafe {
                    *output.get_unchecked_mut(y + r * height) = value;
                }
            }
        }
    }
}

/// One cross-FFT layer: for each vector of columns, gather RADIX rows strided by `num_columns`,
/// apply the twiddles, run the column butterfly, scatter back.
///
/// Unrolled two vectors at a time, which is what the SIMD Radix4's `butterfly_4` does and is what
/// gets the two independent dependency chains needed to keep the FMA pipeline busy.
#[inline(always)]
unsafe fn cross_layer<V: SimdVector, const RADIX: usize, F>(
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
        std::array::from_fn(|r| {
            let v = V::load(data, idx + r * num_columns);
            if r == 0 {
                v
            } else {
                V::mul_complex(v, *twiddles.get_unchecked(tw_base + r - 1))
            }
        })
    };

    let (unroll_count, unroll_remainder) = (num_vector_columns / 2, num_vector_columns % 2);
    for i in 0..unroll_count {
        let vcol = i * 2;
        let idx = vcol * complex_per_vector;

        let a = gather(data, idx, vcol * tw_stride);
        let b = gather(data, idx + complex_per_vector, (vcol + 1) * tw_stride);

        let a = butterfly(a);
        let b = butterfly(b);

        for (r, (a_row, b_row)) in a.iter().zip(b.iter()).enumerate() {
            V::store(data, *a_row, idx + r * num_columns);
            V::store(data, *b_row, idx + complex_per_vector + r * num_columns);
        }
    }

    // an odd vector column count leaves one behind
    if unroll_remainder > 0 {
        let vcol = unroll_count * 2;
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
    use rand::distr::uniform::SampleUniform;

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
        V: SimdVector,
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
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
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
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
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
        V32: SimdVector<ScalarType = f32>,
        V64: SimdVector<ScalarType = f64>,
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
        V: SimdVector,
        V::ScalarType: Float + SampleUniform,
    {
        let len = base_fft.len() * factors.iter().map(|f| f.radix()).product::<usize>();
        let direction = base_fft.fft_direction();
        let fft: SimdRadixN<V, V::ScalarType> = SimdRadixN::new(factors, base_fft);

        check_fft_algorithm::<V::ScalarType>(&fft, len, direction);
    }
}
