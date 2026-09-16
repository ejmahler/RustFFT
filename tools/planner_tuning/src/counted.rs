//! A cost model built by counting instructions in the source, not by measuring.
//!
//! Every leaf cost here comes from reading `src/neon/*.rs`; the derivation is written up in
//! `OP-COUNTS.md`. On top of the arithmetic count sits a coarse memory term: each pass over the
//! buffer is charged per element touched, scaled by how it walks memory (sequential, strided, or
//! permuted) and by which level of an *assumed* cache hierarchy the working set lands in.
//!
//! The point of the exercise is that nothing in here needs a machine. The only quantities that
//! are not read off the source are the handful of weights in `Params`, which set the price of a
//! memory access relative to one arithmetic instruction.

use rustfft::tuning::Spec;

/// Which backend's kernels to price. The decomposition each butterfly uses is the same on both,
/// but the instruction cost of the primitives is not: SSE4.1 has no FMA, so `fmadd` is a separate
/// multiply and add, and its complex multiply needs six instructions rather than four.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Backend {
    Neon,
    Sse,
}

/// Which element type. One 128-bit vector holds one complex f64 or two complex f32, so this
/// changes both the instruction counts and how many elements each memory access covers.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Elem {
    F32,
    F64,
}

impl Elem {
    /// Complex numbers per 128-bit vector.
    pub fn complex_per_vector(&self) -> f64 {
        match self {
            Elem::F32 => 2.0,
            Elem::F64 => 1.0,
        }
    }
}

impl Backend {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "neon" => Some(Backend::Neon),
            "sse" => Some(Backend::Sse),
            _ => None,
        }
    }

    /// `NeonVector::mul_complex` is 4 instructions; `SseVector::mul_complex` is 6
    /// (unpacklo, unpackhi, two muls, shuffle, addsub).
    pub fn mul_complex(&self, elem: Elem) -> f64 {
        match (self, elem) {
            // vcombine + vneg + vmulq_laneq + vfmaq_laneq
            (Backend::Neon, Elem::F64) => 4.0,
            // vtrn1q + vtrn2q + vnegq + vmulq + vrev64q + vfmaq
            (Backend::Neon, Elem::F32) => 6.0,
            // unpacklo + unpackhi + 2 mul + shuffle + addsub
            (Backend::Sse, _) => 6.0,
        }
    }

    /// `column_butterfly4` is four `column_butterfly2` plus one `apply_rotate90` on both.
    pub fn column_butterfly4(&self) -> f64 {
        10.0
    }

    /// Architectural vector registers: 32 `v` registers on aarch64, 16 `xmm` on x86-64 SSE.
    ///
    /// This matters because `cross_layer` in `src/simd_radixn.rs` gathers **two** vector columns
    /// before transforming either, so a radix-R layer holds 2R rows live at once, plus the
    /// butterfly's own temporaries. At radix 7 that is 14 rows before temporaries, which fits
    /// comfortably in 32 registers and not at all in 16.
    pub fn registers(&self) -> f64 {
        match self {
            Backend::Neon => 32.0,
            Backend::Sse => 16.0,
        }
    }

    /// Instructions for one `perform_fft_direct`, excluding the load and store of each element.
    /// Hand-counted from `src/neon/neon_butterflies.rs` and `src/sse/sse_butterflies.rs`; the
    /// derivation is in `OP-COUNTS.md`.
    pub fn butterfly_compute(&self, len: usize, elem: Elem) -> Option<f64> {
        // The generated prime butterflies come out as a closed form, because the generator is a
        // pair of loops. Only the weight of the fmadd chain differs between the backends.
        // f32 on NEON is counted from `perform_parallel_fft_direct`, which computes two FFTs
        // at once, and stored here as the per-FFT figure. See OP-COUNTS.md.
        if let (Backend::Neon, Elem::F32) = (self, elem) {
            return Some(match len {
                1 => 0.0, 2 => 2.0, 3 => 4.0, 4 => 5.0, 5 => 16.0, 6 => 11.0, 7 => 19.5,
                8 => 19.0, 9 => 36.0, 10 => 37.0, 11 => 44.5, 12 => 31.0, 13 => 61.5,
                15 => 68.0, 16 => 66.0, 17 => 100.0, 19 => 125.5, 23 => 180.0, 24 => 113.5,
                29 => 279.5, 31 => 321.0, 32 => 178.0,
                _ => return None,
            });
        }
        if matches!(len, 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31) {
            let h = ((len + 1) / 2) as f64;
            return Some(match self {
                Backend::Neon => (h - 1.0) * (2.0 * h + 5.0),
                Backend::Sse => (h - 1.0) * (4.0 * h + 2.0),
            });
        }
        let v = match self {
            Backend::Neon => match len {
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
            // Same decompositions, but bf3 is written without FMA (10, not a re-weighted 8) and
            // bf8 reaches for rotate_45/rotate_135 where NEON uses explicit multiplies.
            Backend::Sse => match len {
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
        Some(v as f64)
    }
}

/// How a pass walks memory.
#[derive(Copy, Clone)]
pub enum Pattern {
    /// Contiguous run, one cache line feeding many elements.
    Sequential,
    /// Fixed stride greater than a line, as in a transpose or a cross-FFT layer.
    Strided,
    /// Data-dependent scatter or gather: digit reversal, CRT reindexing, Rader's permutation.
    Permuted,
}

/// The weights that are not read off the source.
///
/// Costs are in units of one arithmetic instruction. `seq` is the price of a single load or
/// store at each level of the hierarchy; the multipliers raise that for less friendly patterns.
#[derive(Copy, Clone, Debug)]
pub struct Params {
    /// Complex numbers that fit in the first level of cache.
    pub l1_elems: f64,
    /// Complex numbers that fit in the last level of cache.
    pub l2_elems: f64,
    /// Cost of one load or store at L1, L2 and memory.
    pub seq: [f64; 3],
    pub strided_mult: f64,
    pub permuted_mult: f64,
    /// Cost of computing one Rader's permutation index, in arithmetic-instruction equivalents.
    ///
    /// `raders_algorithm.rs` recomputes `index = index * root % len` per element, where `len` is
    /// a `StrengthReducedU64`, so that `%` is two 64x64->128 widening multiplies plus a shift and
    /// a subtract: about 7 instructions on aarch64. The instruction count alone understates it,
    /// because `index` feeds the next iteration, making the chain loop-carried and latency-bound
    /// rather than throughput-bound. This weight converts the counted instructions into the
    /// effective cost of that serial chain.
    ///
    /// **Size it from the latency, not the instruction count.** The carried chain is
    /// `mul -> umulh -> mul -> sub`, which is about 10 cycles on both an M1 firestorm core and
    /// Coffee Lake. Ten cycles of a core that retires 4 to 6 instructions per cycle is 40 to 60
    /// instruction slots, not the 7 instructions counted. The first default of 20 assumed a 3x
    /// latency inflation and got 12 of 54 prime lengths wrong, at up to 1.64x.
    ///
    /// The term is linear in `len` while everything around it grows as `len log len`, so it
    /// matters most at small lengths, which is why the original 33 lengths (all 1000 or larger)
    /// could not pin it down. It decides the Rader's-versus-Bluestein's call.
    ///
    /// **It is not one value per machine, so the default is a compromise across machines.** Slots
    /// per cycle depend on the core, and the optimum moves with it: 40 on the M1 and 25 to 30 on a
    /// Cortex-A76. The default is chosen by the 1..1000 sweep on the M1, the Pi 5 and the
    /// ThinkCentre, to be acceptable on all three rather than optimal on one:
    ///
    /// - f64: **30**. Fewer losses beyond 2% than 45 on every machine (33/28/46 against
    ///   54/116/54), for 0.2% of geometric mean on the M1.
    /// - f32: **45**. At or near the best on all three; 30 costs 2 to 4% of geometric mean.
    ///
    /// Why f32 wants a larger value is not understood. `complex_per_vector` already halves the
    /// arithmetic per element while this chain stays per element.
    ///
    /// Negative means "use that per-element default"; see `CountedModel::rader_index`.
    pub rader_index: f64,
    /// Cost of one spilled vector per unrolled group in a RadixN cross layer, as a store plus a
    /// reload. Zero disables the register-pressure term entirely.
    pub spill: f64,
    /// Extra cost per element per cross-FFT layer for the **generic** `SimdRadixN` driver,
    /// over the hand-written `Radix4` kernel doing the same work.
    ///
    /// They are different code. `cross_layer` in `src/simd_radixn.rs` is generic over the radix
    /// and gathers two vector columns before transforming either, so it holds 2R rows live plus
    /// the butterfly's temporaries; `sse_radix4.rs` is a hardcoded 2x unroll over six twiddles.
    /// 2R rows fits aarch64's 32 vector registers at every radix it supports, and does not fit
    /// x86-64's 16 xmm registers, so this is expected to be near zero on NEON and positive on SSE.
    ///
    /// The defaults, chosen by the 1..1000 sweep on the ThinkCentre after the transpose indices
    /// were precomputed (see `NEXT-STEPS.md`):
    ///
    /// - NEON: **0**, both element types.
    /// - SSE f64: **6**. 38 losses beyond 2% against 50 at 5, 84 at 8 and 180 at 12. The
    ///   33-length dump alone points at 12 to 16, but every one of those lengths is 1000 or more.
    /// - SSE f32: **1**. 20 losses against 25 at both 0 and 2, and a worst case of 0.909 against
    ///   0.861 at 2.
    ///
    /// The SSE values were 5 and 2 before that change. Removing a per-call overhead let the f32
    /// value come down, as expected; the f64 value moved up by one, not down.
    ///
    /// Negative means "use that per-backend default"; see `CountedModel::radixn_extra`.
    pub radixn_extra: f64,
    /// Override for the cost of one complex multiply. Negative means "use the backend's counted
    /// value". Exists to test whether SSE's shuffle-heavy `mul_complex` costs more than its
    /// instruction count suggests: three of its six instructions are shuffle-class, and on Intel
    /// those all issue to a single port, whereas NEON spreads them over symmetric pipes.
    pub mul_complex: f64,
    /// Charge permuted passes per complex number rather than per vector. On by default.
    ///
    /// A gather or scatter moves one complex number at a time: digit reversal, CRT reindexing and
    /// Rader's permutation all compute a destination per element, so there is no contiguous run to
    /// fill a vector with. Dividing their access count by `complex_per_vector` therefore
    /// under-charges them by exactly that factor, which is invisible at f64 (where the factor is
    /// 1) and a factor of 2 at f32.
    ///
    /// Setting this false restores the original behaviour, which is what `--permuted-vector` is
    /// for. That costs nothing at f64, where both f64 datasets score byte-identically either way,
    /// and at f32 it forces the fitted `permuted_mult` up from 2.5 to between 4.0 and 6.0 to
    /// absorb the same factor. See the f32-on-SSE section of `RESULTS.md`.
    pub permuted_scalar: bool,
    /// Fixed cost per row of a pass, charged to the **general** MixedRadix and GoodThomas
    /// variants and not to their `Small` counterparts.
    ///
    /// The two pairs move the same data in the same order; they differ in implementation.
    /// `MixedRadixSmall` and `GoodThomasAlgorithmSmall` call `array_utils::transpose_small`,
    /// a naive strided double loop over the whole rectangle, and read their permutation from a
    /// precomputed table. `MixedRadix` and `GoodThomasAlgorithm` call the `transpose` crate's
    /// blocked transpose, which keeps both streams cache resident, and `GoodThomasAlgorithm`
    /// additionally computes the CRT mapping on the fly with one `StrengthReducedUsize::div_rem`
    /// and a branch per row rather than per element.
    ///
    /// The general form is therefore cheaper per element and dearer per row, which is a
    /// crossover, and the measurements are the shape of one. General over small on the M1:
    ///
    /// ```text
    /// len        22     28     45    104    496    992
    /// gt/gts   1.341  1.270  1.182  1.056  1.001  1.027
    /// mr/mrs   1.115  1.059  1.048  1.013  0.931  0.948
    /// ```
    ///
    /// The advantage decays towards 1 and MixedRadix crosses under it near len 200. A per-element
    /// difference alone could produce neither: it would hold roughly constant in ratio, and it
    /// could never change sign. Without this term the model has only per-element costs, so it
    /// prices the pair by pattern alone and picks the general form at every length.
    ///
    /// Every one of the twelve pairs above is called correctly for `general_row` anywhere in
    /// 21 to 42; the binding constraints are Good-Thomas at 992 below and MixedRadix at 496
    /// above. 30 sits in the middle of that window.
    pub general_row: f64,
    /// Cost of one outer-loop iteration of `array_utils::transpose_small`, charged to the
    /// **Small** MixedRadix and GoodThomas variants only.
    ///
    /// This is the term that makes the model prefer one ordering of a factor pair over its
    /// reverse. `transpose_small` is a naive double loop:
    ///
    /// ```text
    /// for x in 0..width { for y in 0..height { out[y + x*height] = in[x + y*width] } }
    /// ```
    ///
    /// The outer loop runs `width` times and the read index strides by `width`, so the cost
    /// depends on which dimension is which. The general variants call the `transpose` crate,
    /// which tiles the rectangle and so does not care: that contrast is the evidence, because
    /// `GoodThomasAlgorithm` and `GoodThomasAlgorithmSmall` perform the *same single transpose
    /// in the same orientation* and differ only in the implementation. Over reversed pairs the
    /// small form measures smaller-width-faster at 90 to 97% on both machines and both element
    /// types, while the general form splits about evenly and its median gap is 0.00 ns.
    ///
    /// Outer-loop iterations, counted from the source:
    ///
    /// - `GoodThomasAlgorithmSmall`: one transpose, `(width, height)`, so `width`.
    /// - `MixedRadixSmall`: three, `(w,h)`, `(h,w)`, `(w,h)`, so `2*width + height`.
    ///
    /// Both change by exactly `width - height` when the pair is reversed, which predicts that
    /// the two should show the same asymmetry per unit of `w - h` despite having different
    /// absolute transpose counts. On SSE they measure 1.48 and 1.47 ns respectively.
    ///
    /// So the charge is `small_row * max(width - height, 0)`, not the raw iteration count. Both
    /// variants differ by exactly `width - height` iterations between the two orderings, so one
    /// weight covers both; charging the worse ordering that difference and the better one
    /// nothing reproduces it. Only the *difference* is evidenced here, because the absolute
    /// level of a Small variant against a general one is what `general_row` already carries,
    /// fitted. Charging the difference rather than the count keeps three properties that matter:
    ///
    /// - a square pair is charged nothing, since there is no ordering to get wrong;
    /// - the better ordering keeps exactly the cost it had before this term existed, so
    ///   `general_row` stays valid and the Small-versus-general balance is untouched;
    /// - the cost never goes negative.
    ///
    /// Charging the raw count instead regressed SSE f64 at length 1215, where the recipe is
    /// `mr(b15, mrs(b9,b9))`: the nested square pair was inflated by its 15 repetitions and the
    /// whole recipe lost to an `rn(3.3.3.3,b15)` that is 1.21x slower.
    ///
    /// The value is one outer iteration in instruction-equivalents: about 1.48 ns on the i3 and
    /// 0.7 to 1.0 ns on the M1, which at each machine's ns-per-cost-unit is 9 to 13 either way.
    /// It barely matters. Scores are byte-identical for anything from 2 to 24 on every dataset,
    /// because the term only ever separates two orderings that are otherwise exactly equal in
    /// cost. It is a tie-break with a derivation, not a fitted weight.
    pub small_row: f64,
    /// Which backend's instruction costs to use.
    pub backend: Backend,
    /// Which element type.
    pub elem: Elem,
}

impl Default for Params {
    fn default() -> Self {
        // Apple M1 performance core: 128 KiB L1d, 12 MiB L2, at 16 bytes per complex f64.
        Self {
            l1_elems: 8192.0,
            l2_elems: 786432.0,
            seq: [1.0, 2.0, 6.0],
            strided_mult: 1.5,
            permuted_mult: 2.5,
            rader_index: -1.0,
            spill: 0.0,
            radixn_extra: -1.0,
            mul_complex: -1.0,
            permuted_scalar: true,
            general_row: 30.0,
            small_row: 10.0,
            backend: Backend::Neon,
            elem: Elem::F64,
        }
    }
}

pub struct CountedModel {
    pub params: Params,
}

impl CountedModel {
    pub fn new(params: Params) -> Self {
        Self { params }
    }

    /// The complex-multiply cost actually in force.
    fn mul_complex(&self) -> f64 {
        if self.params.mul_complex >= 0.0 {
            self.params.mul_complex
        } else {
            self.params.backend.mul_complex(self.params.elem)
        }
    }

    /// The Rader's index cost actually in force. See `Params::rader_index` for the values.
    fn rader_index(&self) -> f64 {
        if self.params.rader_index >= 0.0 {
            self.params.rader_index
        } else {
            match self.params.elem {
                Elem::F64 => 30.0,
                Elem::F32 => 45.0,
            }
        }
    }

    /// The RadixN driver cost actually in force. See `Params::radixn_extra` for the values.
    fn radixn_extra(&self) -> f64 {
        if self.params.radixn_extra >= 0.0 {
            self.params.radixn_extra
        } else {
            match (self.params.backend, self.params.elem) {
                (Backend::Neon, _) => 0.0,
                (Backend::Sse, Elem::F64) => 6.0,
                (Backend::Sse, Elem::F32) => 1.0,
            }
        }
    }

    /// Cost of touching `accesses` elements (counting each load and each store once) with the
    /// given pattern, when the enclosing buffer holds `ws` complex numbers.
    /// Rows walked by the three passes of a width x height decomposition.
    ///
    /// The passes run over `height`, `width` and `height` rows respectively, so the exact total is
    /// `2h + w`. The model deliberately ties `mr(A,B)` with `mr(B,A)`, so use the mean of the two
    /// orderings, `1.5 * (w + h)`, rather than introduce an asymmetry here alone.
    fn rows(&self, left: &Spec, right: &Spec) -> f64 {
        1.5 * (left.len() as f64 + right.len() as f64)
    }

    fn mem(&self, accesses: f64, pattern: Pattern, ws: f64) -> f64 {
        let p = &self.params;
        // One load or store moves a whole vector, which is one complex f64 or two complex f32,
        // except under a permutation, where each element's address is computed separately.
        let per_access = match (pattern, p.permuted_scalar) {
            (Pattern::Permuted, true) => 1.0,
            _ => p.elem.complex_per_vector(),
        };
        let accesses = accesses / per_access;
        let level = if ws <= p.l1_elems {
            0
        } else if ws <= p.l2_elems {
            1
        } else {
            2
        };
        let mult = match pattern {
            Pattern::Sequential => 1.0,
            Pattern::Strided => p.strided_mult,
            Pattern::Permuted => p.permuted_mult,
        };
        accesses * p.seq[level] * mult
    }

    /// Estimated cost of one FFT of this recipe, in arithmetic-instruction equivalents.
    ///
    /// `None` if a butterfly length has no counted entry, so a gap fails loudly.
    pub fn cost(&self, spec: &Spec) -> Option<f64> {
        self.cost_ws(spec, spec.len() as f64)
    }

    /// `ws` is the working set of the whole transform, threaded down unchanged: every pass of
    /// every nested algorithm walks the same top-level buffer, so that is what decides which
    /// cache level the traffic is served from.
    fn cost_ws(&self, spec: &Spec, ws: f64) -> Option<f64> {
        let p = &self.params;
        Some(match spec {
            Spec::Dft(n) => {
                let n = *n as f64;
                100.0 * n * n
            }
            Spec::Butterfly(len) => {
                p.backend.butterfly_compute(*len, p.elem)? + self.mem(2.0 * *len as f64, Pattern::Sequential, ws)
            }
            Spec::Radix4 { k, base } => {
                let len = spec.len() as f64;
                let reps = len / base.len() as f64;
                // One digit-reversal transpose, then the base FFTs, then k cross layers.
                let mut c = self.mem(2.0 * len, Pattern::Permuted, ws);
                c += reps * self.cost_ws(base, ws)?;
                for _ in 0..*k {
                    // len/4 column_butterfly4, each with three twiddle multiplies.
                    c += (len / (4.0 * p.elem.complex_per_vector()))
                        * (p.backend.column_butterfly4() + 3.0 * self.mul_complex());
                    c += self.mem(2.0 * len, Pattern::Strided, ws);
                }
                c
            }
            Spec::RadixN { radixes, base } => {
                let len = spec.len() as f64;
                let reps = len / base.len() as f64;
                let mut c = self.mem(2.0 * len, Pattern::Permuted, ws);
                c += reps * self.cost_ws(base, ws)?;
                for r in radixes.iter() {
                    let rf = *r as f64;
                    // The cross-FFT layers call the very same butterfly kernels, so the counted
                    // table applies directly. Row 0 needs no twiddle, hence r - 1.
                    c += (len / rf) * (p.backend.butterfly_compute(*r, p.elem)? + (rf - 1.0) * self.mul_complex());
                    c += self.mem(2.0 * len, Pattern::Strided, ws);
                    c += len * self.radixn_extra();
                    // Register pressure: the layer keeps 2R rows live across the two-column
                    // unroll. Anything past the architectural register file becomes a spill and a
                    // reload, once per element of the group.
                    if p.spill > 0.0 {
                        let live = 2.0 * rf;
                        let over = (live - p.backend.registers()).max(0.0);
                        c += over * p.spill * (len / rf);
                    }
                }
                c
            }
            Spec::MixedRadix { left, right, small } => {
                let len = spec.len() as f64;
                // Three transposes, one full twiddle pass, two inner dimensions.
                //
                // Both variants transpose the same rectangle three times, but not the same way.
                // `MixedRadixSmall` calls `transpose_small`, whose read index strides by `width`
                // and so touches a fresh cache line per element once `width` exceeds a line:
                // line-wasting, which is what `Permuted` prices. `MixedRadix` hands the job to
                // the `transpose` crate, which tiles the rectangle to get that reuse back, and
                // pays `general_row` per row of setup for it.
                let pat = if *small { Pattern::Permuted } else { Pattern::Strided };
                let mut c = 3.0 * self.mem(2.0 * len, pat, ws);
                if *small {
                    // transpose_small at (w,h), (h,w), (w,h) is 2*width + height outer
                    // iterations; reversing the pair gives 2*height + width, so the two
                    // orderings differ by width - height. See `small_row`.
                    c += p.small_row * (left.len() as f64 - right.len() as f64).max(0.0);
                } else {
                    c += p.general_row * self.rows(left, right);
                }
                c += (len / p.elem.complex_per_vector()) * self.mul_complex()
                    + self.mem(2.0 * len, Pattern::Sequential, ws);
                c += right.len() as f64 * self.cost_ws(left, ws)?;
                c += left.len() as f64 * self.cost_ws(right, ws)?;
                c
            }
            Spec::GoodThomas { left, right, small } => {
                let len = spec.len() as f64;
                // Two CRT reindexing passes and one transpose, but no twiddle multiplies at all:
                // dropping them is the whole point of Good-Thomas, and it pays in index work.
                //
                // Both reindexing passes are `Permuted` in either variant. The small one gathers
                // through a precomputed table; the general one walks `destination_index` forward
                // by `width + 1` and wraps modulo `len`, which cycles over the whole buffer and
                // is no friendlier to a cache than a table would be. The transpose splits the two
                // exactly as in MixedRadix, and the general form pays the same per-row setup.
                let pat = if *small { Pattern::Permuted } else { Pattern::Strided };
                let mut c = 2.0 * self.mem(2.0 * len, Pattern::Permuted, ws);
                c += self.mem(2.0 * len, pat, ws);
                if *small {
                    // One transpose_small at (width, height): `width` outer iterations, against
                    // `height` reversed. The same width - height difference as MixedRadixSmall,
                    // which is why one weight serves both.
                    c += p.small_row * (left.len() as f64 - right.len() as f64).max(0.0);
                } else {
                    c += p.general_row * self.rows(left, right);
                }
                c += right.len() as f64 * self.cost_ws(left, ws)?;
                c += left.len() as f64 * self.cost_ws(right, ws)?;
                c
            }
            Spec::Raders { inner } => {
                let len = spec.len() as f64;
                // The inner FFT runs twice, and the permutation is precomputed into a u32 table,
                // so it is a gather and a scatter rather than a modular multiply per element.
                let mut c = 2.0 * self.cost_ws(inner, ws)?;
                // Two permutation passes, each a scatter or gather whose index comes from a
                // serial modular-multiply chain rather than from a table.
                c += 2.0 * (self.mem(2.0 * len, Pattern::Permuted, ws) + len * self.rader_index());
                c += len * self.mul_complex() + self.mem(2.0 * len, Pattern::Sequential, ws);
                c
            }
            Spec::Bluesteins { len, inner } => {
                let outer = *len as f64;
                let ilen = inner.len() as f64;
                // Inner FFT twice, pointwise multiply over the padded inner length, and a
                // twiddle-and-pad pass in and out over the outer length.
                let mut c = 2.0 * self.cost_ws(inner, ws)?;
                c += ilen * self.mul_complex() + self.mem(2.0 * ilen, Pattern::Sequential, ws);
                c += 2.0 * (outer * self.mul_complex() + self.mem(2.0 * outer, Pattern::Sequential, ws));
                c
            }
        })
    }
}
