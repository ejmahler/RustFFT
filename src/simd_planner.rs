//! The FFT design decisions that every SIMD planner makes the same way.
//!
//! `FftPlannerNeon`, `FftPlannerSse` and `FftPlannerWasmSimd` each own a private `Recipe` enum
//! and a private cache, so they can't share the planner itself. What they can share is the
//! arithmetic that picks a plan, which is a pure function of the length's prime factors and of
//! how many complex numbers fit in one of the backend's vectors. These functions do that part
//! and hand back plain numbers; the caller turns them into its own recipes.
//!
//! The scalar planner in `src/plan.rs` deliberately stays out of this. Sharing the choice logic
//! would tie the SIMD backends to the scalar planner's algorithm set, and the two have never been
//! required to match: each backend has its own butterflies, its own Radix4 bases, and its own
//! answer for primes. Its `design_radixn` already differs in ways that change plans, not just
//! style, so folding it in would be a planner change to measure, not a deduplication.

use crate::common::RadixFactor;
use crate::math_utils::PrimeFactors;
use crate::FftNum;

use std::any::TypeId;

const MAX_RADIXN_FACTOR: usize = 7; // The largest butterfly factor that the RadixN algorithm can handle

/// How many complex numbers fit in one SIMD vector, which is what decides the column-count
/// constraints on RadixN and Radix4.
///
/// Every backend here has 128 bit vectors, so this only depends on the float type.
pub fn complex_per_vector<T: FftNum>() -> usize {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        2
    } else {
        1
    }
}

/// What `design_radixn` decided, in terms the caller turns into its own recipe.
pub enum RadixNPlan {
    Radix4 {
        k: u32,
        base_len: usize,
    },
    RadixN {
        factors: Box<[RadixFactor]>,
        base_len: usize,
    },
}

impl RadixNPlan {
    /// The base FFT length, which both variants carry, so the caller can build it once.
    pub fn base_len(&self) -> usize {
        match self {
            RadixNPlan::Radix4 { base_len, .. } => *base_len,
            RadixNPlan::RadixN { base_len, .. } => *base_len,
        }
    }
}

/// Can we do this as a mixed radix with just two butterflies?
///
/// Loops through and finds all combinations. If more than one is found, keeps the one where the
/// factors are closer together. For example length 20, where 10x2 and 5x4 are possible, gives 5x4.
///
/// `all_butterflies` is the sorted list of lengths the backend has a butterfly for.
pub fn design_butterfly_product(len: usize, all_butterflies: &[usize]) -> Option<(usize, usize)> {
    // If the length is below 14, or over 1024 we don't need to try this.
    if len <= 13 || len > 1024 {
        return None;
    }

    let mut bf_left = 0;
    let mut bf_right = 0;
    for (n, bf_l) in all_butterflies.iter().enumerate() {
        if len % bf_l == 0 {
            let bf_r = len / bf_l;
            if all_butterflies.iter().skip(n).any(|&m| m == bf_r) {
                bf_left = *bf_l;
                bf_right = bf_r;
            }
        }
    }
    if bf_left == 0 {
        return None;
    }

    Some((bf_left, bf_right))
}

/// Design a RadixN: fold any factors too big for a cross-FFT layer into the base, pick a base
/// for what's left, and turn the rest into a list of radixes. Mirrors `design_radixn` in
/// `src/plan.rs`, which the scalar planner uses for the same job.
///
/// Returns None when RadixN can't cover this length, which happens for f32 when no legal base
/// is available. The caller falls back to mixed radix in that case.
pub fn design_radixn(factors: &PrimeFactors, complex_per_vector: usize) -> Option<RadixNPlan> {
    // With no factors small enough for a cross-FFT layer, the base would have to be the whole
    // length and there would be nothing left for RadixN to do.
    if !factors.has_factors_leq(MAX_RADIXN_FACTOR) {
        return None;
    }

    let len = factors.get_product();
    let p2 = factors.get_power_of(2);
    let p3 = factors.get_power_of(3);
    let p5 = factors.get_power_of(5);
    let p7 = factors.get_power_of(7);

    let mut base_len: usize = if factors.has_factors_gt(MAX_RADIXN_FACTOR) {
        // Factors larger than a cross-FFT layer can handle *must* go in the base
        factors.product_above(MAX_RADIXN_FACTOR)
    } else if p7 == 0 && p5 == 0 && p3 < 2 {
        // pure powers of two, and 3 * 2^k. Use the same bases design_radix4 does, so that the
        // Radix4 escape below hands these over unchanged.
        if p3 == 0 {
            if p2 % 2 == 1 {
                32
            } else {
                16
            }
        } else if p2 % 2 == 1 {
            24
        } else {
            12
        }
    } else if p2 > 0 && p3 > 0 {
        // a mixed bag of 2s and 3s
        match p2.saturating_sub(p3) {
            0 => 6,
            1 => 12,
            _ => 24,
        }
    } else if p3 > 2 {
        27
    } else if p3 > 1 {
        9
    } else if p7 > 0 {
        7
    } else {
        assert!(p5 > 0);
        5
    };

    // An f32 vector holds two complex numbers, so every cross-FFT layer needs an even column
    // count. The column count starts at base_len, so an odd base is unusable: fold one of the
    // length's factors of two into it instead. If there isn't one to spare, RadixN can't do
    // this length at all.
    if base_len % complex_per_vector != 0 {
        if (len / base_len) % 2 != 0 {
            return None;
        }
        base_len *= 2;
    }

    if base_len >= len || len % base_len != 0 {
        return None;
    }

    let cross_len = len / base_len;

    // Radix4 is faster than the generic driver on pure powers of four, so hand those over. It
    // needs twice the column count RadixN does, hence the extra check on the base.
    let cross_bits = cross_len.trailing_zeros();
    if cross_len.is_power_of_two()
        && cross_bits % 2 == 0
        && base_len % (2 * complex_per_vector) == 0
    {
        return Some(RadixNPlan::Radix4 {
            k: cross_bits / 2,
            base_len,
        });
    }

    // Split what's left into cross-FFT layers. Every factor too big for a layer went into the
    // base above, so the split can't fail, and the same expect guards it in `src/plan.rs`.
    Some(RadixNPlan::RadixN {
        factors: RadixFactor::split_cross_len(cross_len)
            .expect("Every factor RadixN can't handle should have gone into the base"),
        base_len,
    })
}
