//! A cost model for FFT recipes.
//!
//! Table-driven rather than curve-fitted. Each planner has a finite and small set of primitives
//! (a couple of dozen butterflies, a few dozen valid Radix4 shapes), so their costs are simply
//! measured and stored. That removes the extrapolation error a fitted closed form introduces,
//! which is what made the 2021 scalar attempt mis-rank a direct Radix4 against a split one.
//!
//! Only the composing algorithms need fitted numbers, and each needs one: the cost per element
//! of the transposes and twiddle multiplies they add on top of their inner FFTs.

use rustfft::tuning::Spec;
use std::collections::HashMap;

/// What an algorithm's per-element overhead scales with.
///
/// Usually its own length. Two exceptions: Bluestein's pointwise multiply and zero-padding run
/// over the padded inner length, which can be nearly four times the outer length; and RadixN
/// makes one pass per factor, so its overhead scales with length times the number of levels.
pub fn overhead_scale(spec: &Spec) -> f64 {
    match spec {
        Spec::Bluesteins { inner, .. } => inner.len() as f64,
        Spec::RadixN { radixes, .. } => (spec.len() * radixes.len()) as f64,
        other => other.len() as f64,
    }
}

/// Which log2 bucket a spec's overhead belongs in, bucketed on the same quantity the overhead is
/// charged per.
pub fn bucket_of(spec: &Spec) -> u32 {
    overhead_scale(spec).log2() as u32
}

#[derive(Default, Clone)]
pub struct Model {
    /// Measured nanoseconds for one FFT, by butterfly length.
    pub butterfly: HashMap<usize, f64>,
    /// Measured nanoseconds for one FFT, by (base length, k).
    pub radix4: HashMap<(usize, u32), f64>,
    /// Fitted nanoseconds per element of overhead, by algorithm, as a curve over log2 of the
    /// working set. Entries are sorted by bucket. A single entry means a flat constant.
    pub overhead: HashMap<&'static str, Vec<(u32, f64)>>,
}

impl Model {
    /// Overhead per element at a working set of `scale` elements, linearly interpolated between
    /// measured buckets and clamped outside the measured range.
    pub fn overhead_at(&self, kind: &str, scale: f64) -> Option<f64> {
        let table = self.overhead.get(kind)?;
        match table.len() {
            0 => None,
            1 => Some(table[0].1),
            _ => {
                let x = scale.log2();
                if x <= table[0].0 as f64 {
                    return Some(table[0].1);
                }
                if x >= table[table.len() - 1].0 as f64 {
                    return Some(table[table.len() - 1].1);
                }
                for pair in table.windows(2) {
                    let (lo_bucket, lo_value) = pair[0];
                    let (hi_bucket, hi_value) = pair[1];
                    if x <= hi_bucket as f64 {
                        let span = (hi_bucket - lo_bucket) as f64;
                        let t = if span > 0.0 {
                            (x - lo_bucket as f64) / span
                        } else {
                            0.0
                        };
                        return Some(lo_value + t * (hi_value - lo_value));
                    }
                }
                Some(table[table.len() - 1].1)
            }
        }
    }

    /// Estimated nanoseconds for one FFT of this recipe.
    ///
    /// `None` if the recipe needs a primitive that was never measured or an overhead that was
    /// never fitted, so a gap is a loud failure rather than a silently wrong ranking.
    pub fn cost(&self, spec: &Spec) -> Option<f64> {
        Some(match spec {
            Spec::Dft(n) => {
                // Only reached for degenerate sizes; a quadratic keeps it last.
                let n = *n as f64;
                100.0 * n * n
            }
            Spec::Butterfly(len) => *self.butterfly.get(len)?,
            Spec::Radix4 { k, base } => *self.radix4.get(&(base.len(), *k))?,
            Spec::RadixN { radixes, base } => {
                let scale = overhead_scale(spec);
                radixes.iter().product::<usize>() as f64 * self.cost(base)?
                    + self.overhead_at("rn", scale)? * scale
            }
            Spec::MixedRadix { left, right, .. } | Spec::GoodThomas { left, right, .. } => {
                let scale = overhead_scale(spec);
                right.len() as f64 * self.cost(left)?
                    + left.len() as f64 * self.cost(right)?
                    + self.overhead_at(spec.kind(), scale)? * scale
            }
            // Rader's runs its inner FFT twice per transform, once forward and once to invert
            // the convolution, exactly like Bluestein's.
            Spec::Raders { inner } => {
                let scale = overhead_scale(spec);
                2.0 * self.cost(inner)? + self.overhead_at("rad", scale)? * scale
            }
            Spec::Bluesteins { inner, .. } => {
                let scale = overhead_scale(spec);
                2.0 * self.cost(inner)? + self.overhead_at("bs", scale)? * scale
            }
        })
    }

    /// The sum of the inner FFT costs only, with no overhead for `spec` itself. Subtracting this
    /// from a measurement is what isolates one algorithm's overhead.
    pub fn inner_cost(&self, spec: &Spec) -> Option<f64> {
        Some(match spec {
            Spec::MixedRadix { left, right, .. } | Spec::GoodThomas { left, right, .. } => {
                right.len() as f64 * self.cost(left)? + left.len() as f64 * self.cost(right)?
            }
            Spec::RadixN { radixes, base } => {
                radixes.iter().product::<usize>() as f64 * self.cost(base)?
            }
            Spec::Raders { inner } => 2.0 * self.cost(inner)?,
            Spec::Bluesteins { inner, .. } => 2.0 * self.cost(inner)?,
            other => self.cost(other)?,
        })
    }

    /// True if every strict descendant is a primitive or an already-fitted kind.
    ///
    /// Overheads are fitted in dependency order, because the residual for a MixedRadix
    /// containing a MixedRadixSmall only means anything once the Small's overhead is known.
    pub fn descendants_known(&self, spec: &Spec, fitted: &[&'static str]) -> bool {
        spec.children().iter().all(|child| {
            let k = child.kind();
            let ok = matches!(k, "butterfly" | "r4" | "dft") || fitted.contains(&k);
            ok && self.descendants_known(child, fitted)
        })
    }

    pub fn describe(&self) -> String {
        let mut kinds: Vec<(&&str, &Vec<(u32, f64)>)> = self.overhead.iter().collect();
        kinds.sort_by_key(|(k, _)| **k);
        let mut out = format!(
            "{} butterflies, {} radix4 shapes measured\n",
            self.butterfly.len(),
            self.radix4.len()
        );
        out.push_str("overhead, ns per element, by working set:\n");
        for (kind, table) in kinds {
            let rendered: Vec<String> = table
                .iter()
                .map(|(bucket, value)| format!("{}:{:.2}", 1usize << bucket, value))
                .collect();
            out.push_str(&format!("  {:<5} {}\n", kind, rendered.join("  ")));
        }
        out
    }
}

/// The order overheads must be fitted in, so that each kind's inners are already known.
pub const FIT_ORDER: [&str; 7] = ["mrs", "gts", "mr", "gt", "rn", "rad", "bs"];
