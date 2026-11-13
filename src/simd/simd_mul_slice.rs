//! The pairwise complex multiply loop of Bluestein's and Rader's, shared by every SIMD backend.
//!
//! These are plain `#[inline(always)]` functions. The backend wrappers (`NeonBluesteins` and so on)
//! carry the target feature, and everything here is inlined into them.

use std::ops::Deref;

use num_complex::Complex;

use crate::simd::simd_array::{SimdComplexArray, SimdComplexArrayMut};
use crate::simd::simd_vector::{MulSliceAlgorithm, SimdVector};

/// One complex multiply, conjugating `left` first when `CONJUGATE` is set.
#[inline(always)]
unsafe fn mul_one<V: SimdVector, const CONJUGATE: bool>(left: V, right: V) -> V {
    unsafe {
        if CONJUGATE {
            V::mul_complex_conjugated(left, right)
        } else {
            V::mul_complex(left, right)
        }
    }
}

/// `srcdest[i] = srcdest[i] * multiplier[i]` for every element of `srcdest`, conjugating
/// `srcdest[i]` first when `CONJUGATE` is set.
///
/// `srcdest` is either a `&mut` slice, to multiply in place, or a `DoubleBuf`, to read from its
/// input and write to its output. A `DoubleBuf`'s input and output must be the same length, and
/// `multiplier` must be at least as long as both.
///
/// `V::MUL_SLICE_ALGORITHM` picks how the loop is written. The interleaved one is unrolled two
/// vectors at a time, to get two independent dependency chains, the same as `SimdRadixN`'s cross
/// layers. No backend holds more than two complex numbers in a vector, so a length that isn't a
/// whole number of vectors leaves exactly one element over. That one is loaded into and stored from
/// the low half of a vector, the same as the partial columns in `SimdRadixN`.
///
/// Safety: requires the current machine to support `V`'s instruction set.
#[inline(always)]
pub unsafe fn mul_complex_slice<V, A, const CONJUGATE: bool>(
    mut srcdest: A,
    multiplier: &[Complex<V::ScalarType>],
) where
    V: SimdVector,
    A: SimdComplexArrayMut<V> + Deref<Target = [Complex<V::ScalarType>]>,
{
    unsafe {
        // `Deref` reaches a `DoubleBuf`'s input and `DerefMut` its output, and the loads and
        // stores below are unchecked, so check both here.
        let count = srcdest.len();
        assert_eq!(count, srcdest.deref_mut().len());
        assert!(multiplier.len() >= count);

        match V::MUL_SLICE_ALGORITHM {
            MulSliceAlgorithm::Autovectorize => {
                for i in 0..count {
                    let left = *srcdest.get_unchecked(i);
                    let right = *multiplier.get_unchecked(i);
                    *srcdest.get_unchecked_mut(i) = if CONJUGATE {
                        left.conj() * right
                    } else {
                        left * right
                    };
                }
            }
            MulSliceAlgorithm::Interleaved => {
                let per_vector = V::COMPLEX_PER_VECTOR;
                let full_vectors = count / per_vector;

                for i in 0..full_vectors / 2 {
                    let index = i * 2 * per_vector;
                    let next = index + per_vector;

                    let left_a: V = srcdest.load(index);
                    let left_b: V = srcdest.load(next);

                    let product_a = mul_one::<V, CONJUGATE>(left_a, multiplier.load(index));
                    let product_b = mul_one::<V, CONJUGATE>(left_b, multiplier.load(next));

                    srcdest.store(product_a, index);
                    srcdest.store(product_b, next);
                }

                // an odd vector count leaves one behind
                if full_vectors % 2 == 1 {
                    let index = (full_vectors - 1) * per_vector;
                    let left: V = srcdest.load(index);
                    let product = mul_one::<V, CONJUGATE>(left, multiplier.load(index));
                    srcdest.store(product, index);
                }

                // The leftover element, if there is one. `COMPLEX_PER_VECTOR` is a constant, so
                // this whole branch vanishes on the single-complex vectors, whose partial load and
                // store are unimplemented.
                if per_vector > 1 && !count.is_multiple_of(per_vector) {
                    let index = full_vectors * per_vector;
                    let left: V = srcdest.load1_lo(index);
                    let product = mul_one::<V, CONJUGATE>(left, multiplier.load1_lo(index));
                    srcdest.store1_lo(product, index);
                }
            }
        }
    }
}
