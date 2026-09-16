//! Support code for tuning the FFT planners.
//!
//! Compiled only when the non-default `tuning` feature is enabled, and not part of the public
//! API. It exists so that measurement tools can build, name, and enumerate the recipes a planner
//! works with, rather than reimplementing construction and risking measuring something no
//! planner would ever build.
//!
//! Every planner has its own `Recipe` type, and the three SIMD ones are structurally identical
//! while the scalar one additionally has `RadixN`. Rather than duplicate the tooling per
//! planner, everything here is written against [`Spec`], a planner-independent description, and
//! each planner supplies a thin adapter implementing [`TunablePlanner`].

use std::collections::HashMap;
use std::sync::Arc;

use num_integer::gcd;

use crate::{Fft, FftDirection, FftNum};

mod adapters;
pub use adapters::*;

// ---------------------------------------------------------------------------
// Planner-independent recipe description
// ---------------------------------------------------------------------------

/// A recipe, described independently of which planner will build it.
#[derive(Debug, Clone, PartialEq)]
pub enum Spec {
    Dft(usize),
    /// A dedicated kernel for this length. Which concrete butterfly that is, including whether
    /// it is one of the SIMD planners' prime butterflies, is the adapter's business.
    Butterfly(usize),
    Radix4 {
        k: u32,
        base: Arc<Spec>,
    },
    /// Only the scalar planner has this.
    RadixN {
        radixes: Vec<usize>,
        base: Arc<Spec>,
    },
    MixedRadix {
        left: Arc<Spec>,
        right: Arc<Spec>,
        small: bool,
    },
    GoodThomas {
        left: Arc<Spec>,
        right: Arc<Spec>,
        small: bool,
    },
    Raders {
        inner: Arc<Spec>,
    },
    Bluesteins {
        len: usize,
        inner: Arc<Spec>,
    },
}

impl Spec {
    pub fn len(&self) -> usize {
        match self {
            Spec::Dft(len) | Spec::Butterfly(len) => *len,
            Spec::Radix4 { k, base } => base.len() << (2 * k),
            Spec::RadixN { radixes, base } => base.len() * radixes.iter().product::<usize>(),
            Spec::MixedRadix { left, right, .. } | Spec::GoodThomas { left, right, .. } => {
                left.len() * right.len()
            }
            Spec::Raders { inner } => inner.len() + 1,
            Spec::Bluesteins { len, .. } => *len,
        }
    }

    /// A short tag naming the algorithm, used to attribute measured overhead.
    pub fn kind(&self) -> &'static str {
        match self {
            Spec::Dft(_) => "dft",
            Spec::Butterfly(_) => "butterfly",
            Spec::Radix4 { .. } => "r4",
            Spec::RadixN { .. } => "rn",
            Spec::MixedRadix { small: false, .. } => "mr",
            Spec::MixedRadix { small: true, .. } => "mrs",
            Spec::GoodThomas { small: false, .. } => "gt",
            Spec::GoodThomas { small: true, .. } => "gts",
            Spec::Raders { .. } => "rad",
            Spec::Bluesteins { .. } => "bs",
        }
    }

    pub fn children(&self) -> Vec<&Arc<Spec>> {
        match self {
            Spec::Radix4 { base, .. } | Spec::RadixN { base, .. } => vec![base],
            Spec::Raders { inner } | Spec::Bluesteins { inner, .. } => vec![inner],
            Spec::MixedRadix { left, right, .. } | Spec::GoodThomas { left, right, .. } => {
                vec![left, right]
            }
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Naming
// ---------------------------------------------------------------------------

/// Render a spec in the syntax accepted by [`parse`].
pub fn to_spec_string(spec: &Spec) -> String {
    match spec {
        Spec::Dft(len) => format!("dft({})", len),
        Spec::Butterfly(len) => format!("b{}", len),
        Spec::Radix4 { k, base } => format!("r4({},{})", k, to_spec_string(base)),
        Spec::RadixN { radixes, base } => format!(
            "rn({},{})",
            radixes
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join("."),
            to_spec_string(base)
        ),
        Spec::MixedRadix { left, right, small } => format!(
            "{}({},{})",
            if *small { "mrs" } else { "mr" },
            to_spec_string(left),
            to_spec_string(right)
        ),
        Spec::GoodThomas { left, right, small } => format!(
            "{}({},{})",
            if *small { "gts" } else { "gt" },
            to_spec_string(left),
            to_spec_string(right)
        ),
        Spec::Raders { inner } => format!("rad({})", to_spec_string(inner)),
        Spec::Bluesteins { len, inner } => format!("bs({},{})", len, to_spec_string(inner)),
    }
}

struct Parser<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.s[self.pos..].chars().next()
    }

    fn eat(&mut self, c: char) -> Result<(), String> {
        match self.peek() {
            Some(got) if got == c => {
                self.pos += got.len_utf8();
                Ok(())
            }
            other => Err(format!(
                "expected '{}' at offset {}, found {:?}",
                c, self.pos, other
            )),
        }
    }

    fn ident(&mut self) -> String {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric()) {
            self.pos += 1;
        }
        self.s[start..self.pos].to_string()
    }

    fn number(&mut self) -> Result<usize, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        self.s[start..self.pos]
            .parse()
            .map_err(|_| format!("expected a number at offset {}", start))
    }

    fn spec(&mut self) -> Result<Arc<Spec>, String> {
        let name = self.ident();
        if name.is_empty() {
            return Err(format!("expected a recipe at offset {}", self.pos));
        }
        if let Some(size) = name.strip_prefix('b') {
            if let Ok(size) = size.parse::<usize>() {
                return Ok(Arc::new(Spec::Butterfly(size)));
            }
        }

        self.eat('(')?;
        let result = match name.as_str() {
            "dft" => Arc::new(Spec::Dft(self.number()?)),
            "r4" => {
                let k = self.number()? as u32;
                self.eat(',')?;
                let base = self.spec()?;
                Arc::new(Spec::Radix4 { k, base })
            }
            "rn" => {
                let mut radixes = vec![self.number()?];
                while self.peek() == Some('.') {
                    self.eat('.')?;
                    radixes.push(self.number()?);
                }
                self.eat(',')?;
                let base = self.spec()?;
                Arc::new(Spec::RadixN { radixes, base })
            }
            "rad" => Arc::new(Spec::Raders {
                inner: self.spec()?,
            }),
            "bs" => {
                let len = self.number()?;
                self.eat(',')?;
                let inner = self.spec()?;
                Arc::new(Spec::Bluesteins { len, inner })
            }
            "mr" | "mrs" | "gt" | "gts" => {
                let left = self.spec()?;
                self.eat(',')?;
                let right = self.spec()?;
                let small = name.ends_with('s');
                Arc::new(if name.starts_with("mr") {
                    Spec::MixedRadix { left, right, small }
                } else {
                    Spec::GoodThomas { left, right, small }
                })
            }
            other => return Err(format!("unknown recipe '{}'", other)),
        };
        self.eat(')')?;
        Ok(result)
    }
}

/// Parse a spec string, eg `mr(r4(2,b16),b12)`.
pub fn parse(spec: &str) -> Result<Arc<Spec>, String> {
    let cleaned: String = spec.chars().filter(|c| !c.is_whitespace()).collect();
    let mut parser = Parser {
        s: &cleaned,
        pos: 0,
    };
    let parsed = parser.spec()?;
    if parser.pos != cleaned.len() {
        return Err(format!("trailing junk at offset {}", parser.pos));
    }
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// The planner adapter
// ---------------------------------------------------------------------------

/// A planner that measurement tools can drive.
pub trait TunablePlanner<T: FftNum>: Sized {
    /// Name used to select this planner on the command line.
    fn label() -> &'static str;

    fn new() -> Self;

    /// The recipe this planner picks for `len`, as a [`Spec`].
    fn plan(&mut self, len: usize) -> Arc<Spec>;

    /// Build an FFT for an arbitrary spec.
    ///
    /// Each call uses a planner of its own, so that nothing is shared between the recipes being
    /// compared. Panics if the spec is not something this planner can express.
    fn build(&mut self, spec: &Spec, direction: FftDirection) -> Arc<dyn Fft<T>>;

    /// Lengths this planner has a dedicated kernel for.
    fn butterfly_lens() -> Vec<usize>;

    /// Base lengths a `Radix4` can be built on for element type `T`.
    fn radix4_bases() -> Vec<usize>;

    /// Base lengths a `RadixN` can be built on for element type `T`.
    ///
    /// Separate from [`radix4_bases`] because the constraint is looser: a `Radix4` kernel needs a
    /// whole number of vector *pairs* in the base, while `SimdRadixN` needs only a whole number of
    /// vectors. Defaults to the `Radix4` set for planners where the two coincide.
    fn radixn_bases() -> Vec<usize> {
        Self::radix4_bases()
    }

    /// Whether this planner can express `RadixN`.
    fn has_radixn() -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Candidate enumeration
// ---------------------------------------------------------------------------

/// Walks a spec and fails if one length appears with two different recipes.
///
/// Planner algorithm caches are keyed on length alone, so a tree containing two different
/// recipes of the same length would silently build the same FFT twice and quietly invalidate
/// whatever was measured. Every spec goes through this before it is built.
pub fn check_unambiguous(spec: &Spec, seen: &mut HashMap<usize, String>) -> Result<(), String> {
    let rendered = to_spec_string(spec);
    if let Some(previous) = seen.insert(spec.len(), rendered.clone()) {
        if previous != rendered {
            return Err(format!(
                "length {} appears as both '{}' and '{}'; the algorithm cache cannot hold both",
                spec.len(),
                previous,
                rendered
            ));
        }
    }
    for child in spec.children() {
        check_unambiguous(child, seen)?;
    }
    Ok(())
}

/// Plausible alternatives to the planner's choice for `len`, the planner's own pick first.
///
/// Deliberately broader than what any planner would consider: the point is to find what the best
/// available recipe actually is, so a planner's pick can be scored against it.
pub fn candidates<T: FftNum, P: TunablePlanner<T>>(planner: &mut P, len: usize) -> Vec<Arc<Spec>> {
    candidates_inner(planner, len, false)
}

/// `candidates`, optionally skipping the wider-of-the-two-first ordering of each split.
///
/// The prune has to happen here rather than as a filter afterwards. Generating a candidate costs
/// far more than pricing one: it renders the spec to a string, scans the seen-list linearly, and
/// walks the tree to check it is unambiguous. Filtering after the fact at length 1200 cut the
/// candidate count from 48 to 32 but plan time only from 211us to 181us; skipping the work up
/// front is what actually saves it.
fn candidates_inner<T: FftNum, P: TunablePlanner<T>>(
    planner: &mut P,
    len: usize,
    planning: bool,
) -> Vec<Arc<Spec>> {
    let mut out: Vec<Arc<Spec>> = vec![planner.plan(len)];
    let mut seen: Vec<String> = vec![to_spec_string(&out[0])];

    let mut push = |spec: Arc<Spec>, out: &mut Vec<Arc<Spec>>, seen: &mut Vec<String>| {
        if spec.len() != len {
            return;
        }
        let rendered = to_spec_string(&spec);
        if !seen.contains(&rendered) && check_unambiguous(&spec, &mut HashMap::new()).is_ok() {
            seen.push(rendered);
            out.push(spec);
        }
    };

    let butterflies = P::butterfly_lens();

    // Every two-way split, in both orders, as each algorithm that can express it.
    for left_len in 2..=(len / 2) {
        if len % left_len != 0 {
            continue;
        }
        let right_len = len / left_len;
        if planning && left_len > right_len {
            continue;
        }
        let left = planner.plan(left_len);
        let right = planner.plan(right_len);
        let coprime = gcd(left_len, right_len) == 1;
        let small = left_len < 33 && right_len < 33;

        // Both orderings, unless pruning. The loop above already restricts `left_len` to the
        // smaller half when pruning, so the surviving order is the smaller-width-first one.
        let orders: Vec<(Arc<Spec>, Arc<Spec>)> = if planning {
            vec![(Arc::clone(&left), Arc::clone(&right))]
        } else {
            vec![
                (Arc::clone(&left), Arc::clone(&right)),
                (Arc::clone(&right), Arc::clone(&left)),
            ]
        };
        for (left, right) in orders {
            for small_flag in if small {
                vec![false, true]
            } else {
                vec![false]
            } {
                push(
                    Arc::new(Spec::MixedRadix {
                        left: Arc::clone(&left),
                        right: Arc::clone(&right),
                        small: small_flag,
                    }),
                    &mut out,
                    &mut seen,
                );
                if coprime {
                    push(
                        Arc::new(Spec::GoodThomas {
                            left: Arc::clone(&left),
                            right: Arc::clone(&right),
                            small: small_flag,
                        }),
                        &mut out,
                        &mut seen,
                    );
                }
            }
        }
    }

    // Radix4 on every base that divides out to a power of four.
    for base in P::radix4_bases() {
        if base == 0 || len % base != 0 {
            continue;
        }
        let cross = len / base;
        if !cross.is_power_of_two() || cross.trailing_zeros() % 2 != 0 {
            continue;
        }
        push(
            Arc::new(Spec::Radix4 {
                k: cross.trailing_zeros() / 2,
                base: planner.plan(base),
            }),
            &mut out,
            &mut seen,
        );
    }

    // RadixN, where available, over the factorisations the algorithm supports.
    if P::has_radixn() {
        for base in P::radixn_bases() {
            if base == 0 || len % base != 0 {
                continue;
            }
            let mut cross = len / base;
            if cross <= 1 {
                continue;
            }
            let mut radixes = Vec::new();
            for radix in [7usize, 6, 5, 4, 3, 2] {
                while cross % radix == 0 {
                    cross /= radix;
                    radixes.push(radix);
                }
            }
            if cross == 1 && !radixes.is_empty() {
                // Benchmarking upstream suggests the 4s want to go last.
                radixes.sort_by_key(|r| if *r == 4 { 1 } else { 0 });
                push(
                    Arc::new(Spec::RadixN {
                        radixes,
                        base: planner.plan(base),
                    }),
                    &mut out,
                    &mut seen,
                );
            }
        }
    }

    // Prime lengths: Rader's. It needs len - 1 to factor, so it is genuinely prime-only.
    if len > 3 && crate::math_utils::PrimeFactors::compute(len).is_prime() {
        push(
            Arc::new(Spec::Raders {
                inner: planner.plan(len - 1),
            }),
            &mut out,
            &mut seen,
        );
    }

    // Bluestein's over a range of inner lengths, at **every** length and not only at primes.
    //
    // Nothing about Bluestein's needs a prime: the only requirement is an inner FFT of at least
    // 2*len - 1. The shipping planner reaches it solely from `design_prime`, so a composite can
    // never be given it, and that is a real loss rather than a theoretical one. Length 671 is
    // 11 x 61; its only candidates were a MixedRadix or GoodThomas wrapped around a Rader's for
    // 61, and it measured slower than both of its prime neighbours 673 and 677, which do get
    // Bluestein's. Built by hand, bs(671, r4(3,b24)) is 1.46x faster than the planner's pick.
    //
    // Over 1..1000 the lengths whose pick is more than 5% slower than Bluestein's costs nearby
    // number 374 on NEON f32 at a geometric mean of 1.405x, and 357 on SSE f32 at 1.306x. The
    // pattern is a composite whose factorisation forces Rader's onto a large prime factor:
    // Rader's permutation work is scalar, so at f32 it does not shrink while everything around
    // it does.
    // When planning, offer Bluestein's only where the direct route can actually be bad: some
    // prime factor with no butterfly of its own, which is what forces Rader's or an awkward
    // split. If every prime factor has a butterfly the decomposition is all butterflies and
    // Bluestein's, which needs an inner FFT of at least 2*len - 1, cannot compete. Across the
    // four 1..1000 sweeps the model picked Bluestein's at 1315 lengths and **not one** of them
    // had all its prime factors covered, so this costs nothing and skips the enumeration at
    // every smooth length. `candidates` still offers it everywhere, which is how that was
    // checked and how it would be re-checked.
    let bluesteins_worth_it = !planning || {
        let butterflies = P::butterfly_lens();
        let mut n = len;
        let mut uncovered = false;
        let mut d = 2;
        while d * d <= n {
            while n % d == 0 {
                uncovered |= !butterflies.contains(&d);
                n /= d;
            }
            d += 1;
        }
        if n > 1 {
            uncovered |= !butterflies.contains(&n);
        }
        uncovered
    };

    if len > 3 && bluesteins_worth_it {
        let min_inner = 2 * len - 1;
        let mut inner_lens: Vec<usize> = Vec::new();
        for multiplier in [1usize, 3, 5, 7, 9, 15] {
            let mut candidate = multiplier;
            while candidate < min_inner {
                candidate *= 2;
            }
            inner_lens.push(candidate);
        }
        inner_lens.sort_unstable();
        inner_lens.dedup();
        for inner_len in inner_lens {
            push(
                Arc::new(Spec::Bluesteins {
                    len,
                    inner: planner.plan(inner_len),
                }),
                &mut out,
                &mut seen,
            );
        }
    }

    // A bare butterfly, when one exists at this size.
    if butterflies.contains(&len) {
        push(Arc::new(Spec::Butterfly(len)), &mut out, &mut seen);
    }

    out
}

/// [`candidates`], trimmed to at most `cap` entries.
///
/// A length like 100800 has hundreds of two-way splits. Everything structural is kept (the
/// planner's pick, Radix4 and RadixN shapes, Rader's and Bluestein's) and only splits are
/// dropped, most lopsided first, on the grounds that a split with a tiny side is mostly just its
/// large side plus a transpose.
/// What an estimating planner would actually enumerate at `len`.
///
/// Identical to `candidates_capped`, except that it returns the fixed planner's pick immediately
/// at lengths where there is nothing to decide. Enumeration is pure overhead there, and it is the
/// overhead that matters most: plan-time is a large fraction of plan-plus-build at small lengths
/// and a negligible one at large lengths, so the cheapest lengths are exactly where an estimating
/// planner can least afford to enumerate.
///
/// Two classes need no decision, and both were checked against measurement rather than assumed:
///
/// - **A length with its own butterfly.** A single hand-written kernel beats any decomposition.
///   Over lengths 8..128 on NEON the bare butterfly is fastest at all fifteen such lengths, and
///   is never beaten by a split.
/// - **A power of two.** Radix4 wins, and the fixed planner already picks the right base, which
///   is the part that is not obvious: at 1024 `r4(3,b16)` beats `r4(4,b4)` by 1.15x. Across four
///   datasets, at every power of two from 64 up the fixed planner's pick is exactly the fastest
///   measured candidate, regret 1.000.
///
/// It also drops the wider-of-the-two-first ordering of every split. Each two-way split is
/// otherwise enumerated twice, which roughly doubles the candidate count at a highly composite
/// length for almost no information: the two orderings differ only in how `transpose_small` walks
/// the rectangle and in which inner FFT runs first. The smaller-width ordering is the better one
/// in 90 to 97% of measured pairs for the Small variants, and for the general variants the two
/// are usually indistinguishable, which makes dropping one free rather than merely cheap.
///
/// Measured over four datasets, that keeps 58 to 63% of candidates for a geometric mean regret of
/// 1.0016 or better against the full set. The worst single case is 1.129x at length 62 on NEON
/// f32, where `gts(b31,b2)` beats `gts(b2,b31)`; both known exceptions involve b31 or b32, where
/// the parallel-pair f32 butterflies make the chunk count matter in a way none of this models.
/// The planner's own pick is always element zero, so pruning can never leave an estimating
/// planner worse than the fixed one.
///
/// `candidates` and `candidates_capped` stay exhaustive, because scoring a planner's pick needs
/// the alternatives even where a planner would not look at them. That is how every claim above
/// was established, and re-establishing them after a kernel change needs the same breadth.
pub fn plan_candidates<T: FftNum, P: TunablePlanner<T>>(
    planner: &mut P,
    len: usize,
    cap: usize,
) -> Vec<Arc<Spec>> {
    if len.is_power_of_two() || P::butterfly_lens().contains(&len) {
        return vec![planner.plan(len)];
    }
    cap_list(candidates_inner(planner, len, true), cap)
}

pub fn candidates_capped<T: FftNum, P: TunablePlanner<T>>(
    planner: &mut P,
    len: usize,
    cap: usize,
) -> Vec<Arc<Spec>> {
    cap_list(candidates(planner, len), cap)
}

/// Trim a candidate list to `cap`, keeping the planner's pick and the most balanced splits.
fn cap_list(all: Vec<Arc<Spec>>, cap: usize) -> Vec<Arc<Spec>> {
    if all.len() <= cap {
        return all;
    }

    let imbalance = |spec: &Spec| -> Option<f64> {
        match spec {
            Spec::MixedRadix { left, right, .. } | Spec::GoodThomas { left, right, .. } => {
                Some(((left.len() as f64).ln() - (right.len() as f64).ln()).abs())
            }
            _ => None,
        }
    };

    let mut kept: Vec<Arc<Spec>> = Vec::with_capacity(cap);
    let mut splits: Vec<Arc<Spec>> = Vec::new();
    for (index, spec) in all.into_iter().enumerate() {
        if index == 0 || imbalance(&spec).is_none() {
            kept.push(spec);
        } else {
            splits.push(spec);
        }
    }
    splits.sort_by(|a, b| {
        imbalance(a)
            .unwrap()
            .partial_cmp(&imbalance(b).unwrap())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    kept.extend(splits.into_iter().take(cap.saturating_sub(kept.len())));
    kept
}
