//! Estimating planning for the SIMD planners: enumerate the recipes that could compute a length,
//! price each one with a cost model, and keep the cheapest.
//!
//! The cost model is built by reading the source rather than by measuring. Every arithmetic count
//! comes from the butterflies and vector traits of the backend (derived in
//! `tools/planner_tuning/OP-COUNTS.md`), and on top of that sits a coarse memory term: each pass
//! over the buffer is charged per element touched, scaled by how it walks memory. The handful of
//! weights that cannot be read off the source, which set the price of a memory access relative to
//! one arithmetic instruction, were fitted by sweeping lengths 1 to 1000 on several machines with
//! the harness in `tools/planner_tuning`. `COST-MODEL.md` there explains the whole model.
//!
//! Costs are priced on a [`Shape`], which names a recipe's top-level algorithm and the lengths of
//! its inner FFTs, but not the inner recipes themselves. The caller supplies the cost of each
//! inner length. That lets a planner memoise the best cost per length next to its recipe cache,
//! so the search recurses over the divisors of a length rather than over whole recipe trees.
//! Memoising by length alone is sound because a node's cost depends only on its own subtree,
//! never on the transform it is nested in.

use crate::common::RadixFactor;
use crate::math_utils::PrimeFactors;
use crate::FftNum;

use super::simd_planner::complex_per_vector;

/// The most candidates priced at one length. A highly composite length has hundreds of two-way
/// splits; see `cap_candidates` for which ones are dropped.
const MAX_CANDIDATES: usize = 48;

/// Bases a SIMD `Radix4` can be built on, before the per element type vector pair filter.
const RADIX4_BASES: [usize; 10] = [1, 2, 4, 8, 16, 32, 3, 6, 12, 24];

/// Bases a SIMD `RadixN` can be built on, before the per element type vector filter. Wider than
/// the `Radix4` set because `SimdRadixN` needs only a whole number of vectors in the base.
const RADIXN_BASES: [usize; 12] = [4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 24, 32];

/// Bluestein's inner lengths are each of these, doubled until at least `2 * len - 1`.
const BLUESTEIN_MULTIPLIERS: [usize; 6] = [1, 3, 5, 7, 9, 15];

/// The top level of a recipe, with its inner FFTs given only by length.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Shape {
    Butterfly(usize),
    Radix4 {
        k: u32,
        base_len: usize,
    },
    RadixN {
        factors: Box<[RadixFactor]>,
        base_len: usize,
    },
    MixedRadix {
        left_len: usize,
        right_len: usize,
        small: bool,
    },
    GoodThomas {
        left_len: usize,
        right_len: usize,
        small: bool,
    },
    Raders {
        len: usize,
    },
    Bluesteins {
        len: usize,
        inner_len: usize,
    },
}

impl Shape {
    /// The lengths of the inner FFTs, which have to be planned before this shape can be priced.
    pub fn child_lens(&self) -> impl Iterator<Item = usize> {
        let (first, second) = match self {
            Shape::Butterfly(_) => (None, None),
            Shape::Radix4 { base_len, .. } | Shape::RadixN { base_len, .. } => {
                (Some(*base_len), None)
            }
            Shape::MixedRadix {
                left_len,
                right_len,
                ..
            }
            | Shape::GoodThomas {
                left_len,
                right_len,
                ..
            } => (Some(*left_len), Some(*right_len)),
            Shape::Raders { len } => (Some(len - 1), None),
            Shape::Bluesteins { inner_len, .. } => (Some(*inner_len), None),
        };
        first.into_iter().chain(second)
    }
}

/// Which backend's instruction counts to use.
///
/// The decomposition each butterfly uses is the same on NEON and SSE, but the cost of the
/// primitives is not: SSE4.1 has no FMA, so `fmadd` is a separate multiply and add, and its
/// complex multiply needs six instructions rather than four.
// Only one backend is compiled for a given target, so the other variant always looks unused.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InstructionSet {
    Neon,
    Sse,
}

impl InstructionSet {
    /// `mul_complex` in the backend's vector trait.
    fn mul_complex(self, complex_per_vector: usize) -> f64 {
        match (self, complex_per_vector) {
            // vcombine + vneg + vmulq_laneq + vfmaq_laneq
            (InstructionSet::Neon, 1) => 4.0,
            // vtrn1q + vtrn2q + vnegq + vmulq + vrev64q + vfmaq
            (InstructionSet::Neon, _) => 6.0,
            // unpacklo + unpackhi + 2 mul + shuffle + addsub
            (InstructionSet::Sse, _) => 6.0,
        }
    }

    /// Instructions for one `perform_fft_direct`, excluding the load and store of each element.
    /// Hand-counted from the backend's butterflies; the derivation is in `OP-COUNTS.md`.
    fn butterfly_compute(self, len: usize, complex_per_vector: usize) -> Option<f64> {
        // f32 on NEON is counted from `perform_parallel_fft_direct`, which computes two FFTs at
        // once, and stored here as the per-FFT figure.
        if let (InstructionSet::Neon, 2) = (self, complex_per_vector) {
            return Some(match len {
                1 => 0.0,
                2 => 2.0,
                3 => 4.0,
                4 => 5.0,
                5 => 16.0,
                6 => 11.0,
                7 => 19.5,
                8 => 19.0,
                9 => 36.0,
                10 => 37.0,
                11 => 44.5,
                12 => 31.0,
                13 => 61.5,
                15 => 68.0,
                16 => 66.0,
                17 => 100.0,
                19 => 125.5,
                23 => 180.0,
                24 => 113.5,
                29 => 279.5,
                31 => 321.0,
                32 => 178.0,
                _ => return None,
            });
        }
        // The generated prime butterflies come out as a closed form, because the generator is a
        // pair of loops. Only the weight of the fmadd chain differs between the backends.
        if matches!(len, 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31) {
            let h = ((len + 1) / 2) as f64;
            return Some(match self {
                InstructionSet::Neon => (h - 1.0) * (2.0 * h + 5.0),
                InstructionSet::Sse => (h - 1.0) * (4.0 * h + 2.0),
            });
        }
        let count = match self {
            InstructionSet::Neon => match len {
                1 => 0,
                2 => 2,
                3 => 8,
                4 => 10,
                5 => 28,
                6 => 22,
                8 => 38,
                9 => 64,
                10 => 66,
                12 => 62,
                15 => 124,
                16 => 114,
                24 => 201,
                32 => 314,
                _ => return None,
            },
            // Same decompositions, but bf3 is written without FMA and bf8 reaches for
            // rotate_45/rotate_135 where NEON uses explicit multiplies.
            InstructionSet::Sse => match len {
                1 => 0,
                2 => 2,
                3 => 10,
                4 => 10,
                5 => 28,
                6 => 26,
                8 => 38,
                9 => 84,
                10 => 66,
                12 => 70,
                15 => 134,
                16 => 122,
                24 => 233,
                32 => 346,
                _ => return None,
            },
        };
        Some(count as f64)
    }
}

/// How a pass walks memory.
#[derive(Copy, Clone)]
enum Pattern {
    /// Contiguous run, one cache line feeding many elements.
    Sequential,
    /// Fixed stride greater than a line, as in a transpose or a cross-FFT layer.
    Strided,
    /// Data-dependent scatter or gather: digit reversal, CRT reindexing, Rader's permutation.
    Permuted,
}

/// Prices recipes for one backend and element type, in units of one arithmetic instruction.
///
/// The public fields are the weights that are not read off the source. They are public so the
/// tuning harness can override them.
#[derive(Copy, Clone, Debug)]
pub struct CostModel {
    pub instruction_set: InstructionSet,
    /// Complex numbers in one 128-bit vector: 1 for f64, 2 for f32.
    pub complex_per_vector: usize,
    /// A fixed-stride pass against a sequential one.
    pub strided: f64,
    /// A gather or scatter against a sequential pass.
    pub permuted: f64,
    /// One element of Rader's permutation passes, on top of the gather or scatter itself: the
    /// load of its index from the precomputed `u32` table.
    ///
    /// Before ejmahler#178 the index came from a loop-carried modular-multiply chain, latency
    /// bound, and this was 30 (f64) and 45 (f32). With the table, a survey of 300 random lengths
    /// up to a million on an M1 scores 2 and 8 about the same, and both far better than the old
    /// values, which made Rader's look expensive enough to trade for a large Bluestein's.
    pub rader_index: f64,
    /// Extra cost per element per cross-FFT layer of the generic `SimdRadixN` driver, over the
    /// hand-written `Radix4` kernel doing the same work. Expected near zero on NEON's 32 vector
    /// registers and positive on SSE's 16, since a layer holds two rows per radix live at once.
    pub radixn_extra: f64,
    /// Fixed cost per row charged to the general MixedRadix and GoodThomas, for the blocked
    /// transpose and on-the-fly CRT mapping that the `Small` variants do not have.
    pub general_row: f64,
    /// One outer iteration of `transpose_small`, charged to the `Small` variants for the width
    /// by which the pair is wider than it is tall. A tie-break between the two orderings.
    pub small_row: f64,
    /// Complex numbers that still fit in cache. A transpose of more than this many is charged
    /// `dram` per access on top of its pattern. Infinite prices every transpose the same.
    ///
    /// A node is charged at the size of the buffer it walks itself, never at the size of the
    /// transform it sits inside, so a recipe's cost stays a function of its own subtree and can
    /// be cached by length.
    ///
    /// The default is 256 KiB worth of complex numbers, which is the smallest last-level cache
    /// worth planning for rather than any particular machine's. A step is what the hardware
    /// does, and unlike a cost that grows smoothly with length it leaves every shorter length
    /// priced exactly as before, so the weights fitted by sweeping 1 to 1000 stay valid: no pick
    /// below length 16385 changes at this threshold. Lowering it to 128 KiB starts moving picks
    /// from 8194, raising it to 2 MiB moves none below 20000 and is worse in the f32 tail.
    pub cache_elems: f64,
    /// What one access of a transpose costs once the buffer no longer fits in cache, relative to
    /// the same access in cache. Anything from 2 to 5 scores the same, so this is an order of
    /// magnitude rather than a fitted value.
    pub dram: f64,
}

impl CostModel {
    /// The fitted weights for this instruction set and element type.
    pub fn new(instruction_set: InstructionSet, complex_per_vector: usize) -> Self {
        let f32 = complex_per_vector == 2;
        let (strided, radixn_extra) = match (instruction_set, f32) {
            (InstructionSet::Neon, _) => (1.5, 0.0),
            (InstructionSet::Sse, false) => (1.5, 6.0),
            (InstructionSet::Sse, true) => (2.5, 1.0),
        };
        Self {
            instruction_set,
            complex_per_vector,
            strided,
            permuted: 2.5,
            rader_index: 2.0,
            radixn_extra,
            general_row: 30.0,
            small_row: 10.0,
            cache_elems: 256.0 * 1024.0 / 16.0 * complex_per_vector as f64,
            dram: 2.0,
        }
    }

    /// The fitted weights for this instruction set and the element type `T`.
    pub fn for_type<T: FftNum>(instruction_set: InstructionSet) -> Self {
        Self::new(instruction_set, complex_per_vector::<T>())
    }

    /// Cost of touching `accesses` elements, counting each load and each store once, in a pass
    /// over a buffer of `ws` complex numbers.
    fn mem(&self, accesses: f64, pattern: Pattern, ws: f64) -> f64 {
        let cost = match pattern {
            // A gather or scatter computes an address per complex number, so it cannot fill a
            // vector.
            Pattern::Permuted => accesses * self.permuted,
            Pattern::Strided => accesses / self.complex_per_vector as f64 * self.strided,
            Pattern::Sequential => accesses / self.complex_per_vector as f64,
        };
        let _ = ws;
        cost
    }

    /// Cost of a pass that transposes the whole buffer, which is the one access pattern that
    /// really falls out of cache.
    ///
    /// A RadixN or Radix4 cross layer looks strided but gathers its rows from inside the chunk it
    /// is already working on, so it keeps its locality at any size. A MixedRadix or GoodThomas
    /// transpose walks the whole rectangle, so once that no longer fits in cache every element
    /// costs a fresh line.
    fn transpose_mem(&self, accesses: f64, pattern: Pattern, ws: f64) -> f64 {
        let cost = self.mem(accesses, pattern, ws);
        if ws > self.cache_elems {
            cost * self.dram
        } else {
            cost
        }
    }

    fn mul_complex(&self) -> f64 {
        self.instruction_set.mul_complex(self.complex_per_vector)
    }

    fn butterfly(&self, len: usize) -> Option<f64> {
        self.instruction_set
            .butterfly_compute(len, self.complex_per_vector)
    }

    /// Estimated cost of one FFT of this shape. `child_cost` gives the cost of an inner FFT by
    /// length, for every length `shape.child_lens()` returns.
    ///
    /// `None` if a butterfly has no counted entry, so a gap in the tables fails loudly rather than
    /// pricing a recipe as free.
    pub(crate) fn cost(&self, shape: &Shape, child_cost: impl Fn(usize) -> f64) -> Option<f64> {
        let cpv = self.complex_per_vector as f64;
        Some(match shape {
            Shape::Butterfly(len) => {
                self.butterfly(*len)?
                    + self.mem(2.0 * *len as f64, Pattern::Sequential, *len as f64)
            }
            Shape::Radix4 { k, base_len } => {
                let len = (*base_len << (2 * k)) as f64;
                let reps = len / *base_len as f64;
                // One digit-reversal transpose, then the base FFTs, then k cross layers of
                // len/4 column_butterfly4, each four butterfly2 plus a rotate90 (10 instructions)
                // and three twiddle multiplies.
                let mut c = self.mem(2.0 * len, Pattern::Permuted, len);
                c += reps * child_cost(*base_len);
                c += *k as f64
                    * ((len / (4.0 * cpv)) * (10.0 + 3.0 * self.mul_complex())
                        + self.mem(2.0 * len, Pattern::Strided, len));
                c
            }
            Shape::RadixN { factors, base_len } => {
                let radix_product: usize = factors.iter().map(|f| f.radix()).product();
                let len = (*base_len * radix_product) as f64;
                let reps = len / *base_len as f64;
                let mut c = self.mem(2.0 * len, Pattern::Permuted, len);
                c += reps * child_cost(*base_len);
                for factor in factors.iter() {
                    let radix = factor.radix();
                    // The cross-FFT layers call the very same butterfly kernels, so the counted
                    // table applies directly. Row 0 needs no twiddle, hence radix - 1.
                    c += (len / radix as f64)
                        * (self.butterfly(radix)? + (radix as f64 - 1.0) * self.mul_complex());
                    c += self.mem(2.0 * len, Pattern::Strided, len);
                    c += len * self.radixn_extra;
                }
                c
            }
            Shape::MixedRadix {
                left_len,
                right_len,
                small,
            } => {
                let len = (left_len * right_len) as f64;
                // Three transposes, one full twiddle pass, and the two inner dimensions.
                //
                // `MixedRadixSmall` calls `transpose_small`, whose read index strides by `width`
                // and so touches a fresh cache line per element: that is what `Permuted` prices.
                // `MixedRadix` hands the job to the `transpose` crate, which tiles the rectangle
                // to get cache reuse back and pays `general_row` per row for it.
                let mut c = if *small {
                    3.0 * self.transpose_mem(2.0 * len, Pattern::Permuted, len)
                        + self.small_row * left_len.saturating_sub(*right_len) as f64
                } else {
                    3.0 * self.transpose_mem(2.0 * len, Pattern::Strided, len)
                        + self.general_row * 1.5 * (left_len + right_len) as f64
                };
                c += (len / cpv) * self.mul_complex()
                    + self.mem(2.0 * len, Pattern::Sequential, len);
                c += *right_len as f64 * child_cost(*left_len);
                c += *left_len as f64 * child_cost(*right_len);
                c
            }
            Shape::GoodThomas {
                left_len,
                right_len,
                small,
            } => {
                let len = (left_len * right_len) as f64;
                // Two CRT reindexing passes and one transpose, but no twiddle multiplies. Both
                // reindexing passes are permuted in either variant; the transpose splits the two
                // exactly as in MixedRadix.
                let mut c = 2.0 * self.transpose_mem(2.0 * len, Pattern::Permuted, len);
                c += if *small {
                    self.transpose_mem(2.0 * len, Pattern::Permuted, len)
                        + self.small_row * left_len.saturating_sub(*right_len) as f64
                } else {
                    self.transpose_mem(2.0 * len, Pattern::Strided, len)
                        + self.general_row * 1.5 * (left_len + right_len) as f64
                };
                c += *right_len as f64 * child_cost(*left_len);
                c += *left_len as f64 * child_cost(*right_len);
                c
            }
            Shape::Raders { len } => {
                let len_f = *len as f64;
                // The inner FFT runs twice, around a permuting gather and scatter.
                let mut c = 2.0 * child_cost(len - 1);
                c += 2.0
                    * (self.mem(2.0 * len_f, Pattern::Permuted, len_f) + len_f * self.rader_index);
                c += len_f * self.mul_complex() + self.mem(2.0 * len_f, Pattern::Sequential, len_f);
                c
            }
            Shape::Bluesteins { len, inner_len } => {
                let outer = *len as f64;
                let inner = *inner_len as f64;
                // Inner FFT twice, pointwise multiply over the padded inner length, and a
                // twiddle-and-pad pass in and out over the outer length.
                let mut c = 2.0 * child_cost(*inner_len);
                c += inner * self.mul_complex() + self.mem(2.0 * inner, Pattern::Sequential, inner);
                c += 2.0
                    * (outer * self.mul_complex()
                        + self.mem(2.0 * outer, Pattern::Sequential, outer));
                c
            }
        })
    }
}

/// Whether a length has anything to decide.
///
/// Two classes do not, and both were checked against measurement rather than assumed. A length
/// with its own butterfly: over lengths 8 to 128 on NEON the bare butterfly is fastest at every
/// one. And a power of two: at every power of two from 64 up, across four datasets, the fixed
/// planner's Radix4 is exactly the fastest measured candidate. Skipping enumeration there matters,
/// because plan time is the largest fraction of plan-plus-build at exactly these short lengths.
pub fn has_choice(len: usize, all_butterflies: &[usize]) -> bool {
    !len.is_power_of_two() && all_butterflies.binary_search(&len).is_err()
}

/// The shapes worth pricing at `len`, with `fixed` (the fixed planner's pick) first.
///
/// The order is significant: the planner keeps the first of equally cheap shapes, so the fixed
/// planner's pick wins every tie.
///
/// Each two-way split is offered only with the smaller side as the width. The reverse ordering
/// roughly doubles the candidate count at a highly composite length for almost no information:
/// the smaller-width ordering is the better one in 90 to 97% of measured pairs for the `Small`
/// variants, and the general variants are usually indistinguishable.
pub(crate) fn candidates(
    len: usize,
    factors: &PrimeFactors,
    fixed: Shape,
    all_butterflies: &[usize],
    complex_per_vector: usize,
) -> Vec<Shape> {
    let mut out = vec![fixed];
    let push = |shape: Shape, out: &mut Vec<Shape>| {
        if !out.contains(&shape) {
            out.push(shape);
        }
    };

    for left_len in 2..=(len / 2) {
        if len % left_len != 0 {
            continue;
        }
        let right_len = len / left_len;
        if left_len > right_len {
            break;
        }
        let coprime = num_integer::gcd(left_len, right_len) == 1;
        let small_allowed = left_len < 33 && right_len < 33;
        for small in [false, true] {
            if small && !small_allowed {
                continue;
            }
            push(
                Shape::MixedRadix {
                    left_len,
                    right_len,
                    small,
                },
                &mut out,
            );
            if coprime {
                push(
                    Shape::GoodThomas {
                        left_len,
                        right_len,
                        small,
                    },
                    &mut out,
                );
            }
        }
    }

    // Radix4 on every base that leaves a power of four. The kernels need a whole number of
    // vector pairs in the base.
    for base_len in RADIX4_BASES {
        if base_len % (2 * complex_per_vector) != 0 || len % base_len != 0 {
            continue;
        }
        let cross = len / base_len;
        if cross.is_power_of_two() && cross.trailing_zeros() % 2 == 0 {
            let k = cross.trailing_zeros() / 2;
            push(Shape::Radix4 { k, base_len }, &mut out);
        }
    }

    // RadixN on every base that leaves only radixes a cross layer can take. `SimdRadixN` needs
    // only a whole number of vectors in the base.
    for base_len in RADIXN_BASES {
        if base_len % complex_per_vector != 0 || len % base_len != 0 || len == base_len {
            continue;
        }
        let mut cross = len / base_len;
        let mut radixes = Vec::new();
        for (radix, factor) in [
            (7, RadixFactor::Factor7),
            (6, RadixFactor::Factor6),
            (5, RadixFactor::Factor5),
            (4, RadixFactor::Factor4),
            (3, RadixFactor::Factor3),
            (2, RadixFactor::Factor2),
        ] {
            while cross % radix == 0 {
                cross /= radix;
                radixes.push(factor);
            }
        }
        if cross == 1 {
            // Benchmarking upstream suggests the 4s want to go last.
            radixes.sort_by_key(|factor| *factor == RadixFactor::Factor4);
            push(
                Shape::RadixN {
                    factors: radixes.into_boxed_slice(),
                    base_len,
                },
                &mut out,
            );
        }
    }

    if len > 3 && factors.is_prime() {
        push(Shape::Raders { len }, &mut out);
    }

    // Bluestein's needs no prime, only an inner FFT of at least 2 * len - 1, so it is offered at
    // composite lengths too: 671 = 11 x 61 measured 1.46x faster as Bluestein's than as a split
    // around a Rader's for 61. But only where some prime factor has no butterfly of its own. If
    // every one has, the whole length decomposes into butterflies, and across four 1..1000
    // sweeps not one such length was won by Bluestein's.
    let uncovered_factor = factors
        .get_other_factors()
        .iter()
        .any(|factor| all_butterflies.binary_search(&factor.value).is_err());
    if len > 3 && uncovered_factor {
        let min_inner_len = 2 * len - 1;
        let mut inner_lens: Vec<usize> = BLUESTEIN_MULTIPLIERS
            .iter()
            .map(|&multiplier| {
                let mut inner_len = multiplier;
                while inner_len < min_inner_len {
                    inner_len *= 2;
                }
                inner_len
            })
            .collect();
        inner_lens.sort_unstable();
        inner_lens.dedup();
        for inner_len in inner_lens {
            push(Shape::Bluesteins { len, inner_len }, &mut out);
        }
    }

    cap_candidates(out)
}

/// Trim a candidate list to `MAX_CANDIDATES`.
///
/// Everything structural is kept (the fixed planner's pick, Radix4 and RadixN, Rader's and
/// Bluestein's), and only splits are dropped, most lopsided first, on the grounds that a split
/// with a tiny side is mostly its large side plus a transpose. The kept splits move behind the
/// structural shapes, ordered from most to least balanced.
fn cap_candidates(all: Vec<Shape>) -> Vec<Shape> {
    if all.len() <= MAX_CANDIDATES {
        return all;
    }
    let imbalance = |shape: &Shape| match shape {
        Shape::MixedRadix {
            left_len,
            right_len,
            ..
        }
        | Shape::GoodThomas {
            left_len,
            right_len,
            ..
        } => Some(((*right_len as f64).ln() - (*left_len as f64).ln()).abs()),
        _ => None,
    };

    let mut kept = Vec::with_capacity(MAX_CANDIDATES);
    let mut splits = Vec::new();
    for (index, shape) in all.into_iter().enumerate() {
        if index == 0 || imbalance(&shape).is_none() {
            kept.push(shape);
        } else {
            splits.push(shape);
        }
    }
    splits.sort_by(|a, b| imbalance(a).unwrap().total_cmp(&imbalance(b).unwrap()));
    let room = MAX_CANDIDATES.saturating_sub(kept.len());
    kept.extend(splits.into_iter().take(room));
    kept
}
