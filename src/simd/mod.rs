//! Code shared by every SIMD backend, written once against the `SimdVector` trait.
//!
//! Nothing here is reachable from the planners yet: wiring `SimdRadixN` into them is a separate
//! PR, so that the plan changes can be measured on their own. Until then the whole module is
//! only exercised by its own tests.
#![allow(dead_code)]

pub mod simd_radixn;
pub mod simd_vector;
