//! Pricing whole [`Spec`] trees with the estimating planner's cost model.
//!
//! The planners price a recipe one level at a time, taking the cost of each inner length from
//! their cache. Measurement tools need the cost of an arbitrary recipe instead, such as every
//! candidate in a dump, so this walks the tree and feeds the model the costs of the actual inner
//! recipes.

use std::collections::HashMap;

use super::Spec;
use crate::common::RadixFactor;
use crate::simd::simd_estimate::Shape;

pub use crate::simd::simd_estimate::{CostModel, InstructionSet};

/// Estimated cost of one FFT of `spec`, or `None` if it contains a butterfly the model has no
/// counts for.
pub fn spec_cost(model: &CostModel, spec: &Spec) -> Option<f64> {
    let shape = match spec {
        // Only ever used at length 0 by the planners. Priced as its quadratic work, so that it
        // never looks attractive.
        Spec::Dft(len) => return Some(100.0 * (*len as f64) * (*len as f64)),
        Spec::Butterfly(len) => Shape::Butterfly(*len),
        Spec::Radix4 { k, base } => Shape::Radix4 {
            k: *k,
            base_len: base.len(),
        },
        Spec::RadixN { radixes, base } => Shape::RadixN {
            factors: radixes
                .iter()
                .map(|radix| match radix {
                    2 => Some(RadixFactor::Factor2),
                    3 => Some(RadixFactor::Factor3),
                    4 => Some(RadixFactor::Factor4),
                    5 => Some(RadixFactor::Factor5),
                    6 => Some(RadixFactor::Factor6),
                    7 => Some(RadixFactor::Factor7),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
            base_len: base.len(),
        },
        Spec::MixedRadix { left, right, small } => Shape::MixedRadix {
            left_len: left.len(),
            right_len: right.len(),
            small: *small,
        },
        Spec::GoodThomas { left, right, small } => Shape::GoodThomas {
            left_len: left.len(),
            right_len: right.len(),
            small: *small,
        },
        Spec::Raders { inner } => Shape::Raders {
            len: inner.len() + 1,
        },
        Spec::Bluesteins { len, inner } => Shape::Bluesteins {
            len: *len,
            inner_len: inner.len(),
        },
    };

    // A spec's children have distinct lengths, except a square split, whose two sides are the
    // same recipe anyway.
    let mut child_costs = HashMap::new();
    for child in spec.children() {
        child_costs.insert(child.len(), spec_cost(model, child)?);
    }
    model.cost(&shape, |child_len| child_costs[&child_len])
}
