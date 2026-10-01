//! The vector operations shared by every SIMD backend.
//!
//! Each backend has its own vector trait (`NeonVector`, `SseVector`, `WasmVector`), and those
//! don't share a supertrait. `SimdVector` is a separate, stripped down trait that each backend
//! implements once per vector type, next to its own vector trait impls. Algorithms written against
//! it, like `SimdRadixN`, then only exist once.

use num_complex::Complex;

use crate::common::FftNum;
use crate::simd::simd_radixn::InternalRadixFactor;
use crate::FftDirection;

/// The vector operations the shared SIMD algorithms need from a backend's vector type.
///
/// Radix 2 and 4 are vector-generic in every backend, so they map straight onto the backend's
/// vector trait. Radix 3, 5, 6 and 7 only exist as element-type-specific structs holding
/// precomputed twiddles, so this trait pairs each vector type with its own set of them. For f64 a
/// vector is one complex number and the plain `perform_fft_direct` is already a single column; for
/// f32 a vector is two complex numbers and `perform_parallel_fft_direct` does two columns at once.
///
/// Safety: every method here requires the current machine to support the backend's SIMD
/// instruction set.
#[allow(dead_code)]
pub trait SimdVector: Copy + Send + Sync + Sized {
    const COMPLEX_PER_VECTOR: usize;
    const RADIXN_CROSS_LAYER_UNROLL: bool; // If true, this platform benefits from doing a 2x unroll of the RadixN cross layers

    /// True if the backend has instructions that multiply two complex numbers directly, rather than
    /// having to build the multiply out of real multiplies and lane shuffles. Only FCMA does.
    ///
    /// The pairwise multiply loops use this to decide whether their vector path is worth taking at
    /// all. With a single complex number per vector, interleaved data and a shuffle-based multiply,
    /// those loops lose to the plain scalar loop LLVM autovectorizes into deinterleaving loads, so
    /// they take the scalar path instead. FCMA's two-instruction multiply wins, and so does every
    /// vector holding two complex numbers.
    const FUSED_COMPLEX_MULTIPLY: bool;

    /// The scalar this vector holds. Always the same type as the `T` of the algorithm using it.
    type ScalarType: FftNum;

    /// The backend's precomputed 90 degree rotation, which the radix 4 butterfly needs.
    type Rotation: Copy + Send + Sync;

    type Butterfly3: Send + Sync;
    type Butterfly5: Send + Sync;
    type Butterfly6: Send + Sync;
    type Butterfly7: Send + Sync;

    unsafe fn zero_vector() -> Self;

    // loads of complex numbers
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self;
    unsafe fn load1_lo_complex(ptr: *const Complex<Self::ScalarType>) -> Self;
    unsafe fn load1_dup_complex(ptr: *const Complex<Self::ScalarType>) -> Self;

    // stores of complex numbers
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);
    unsafe fn store1_lo_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);

    // Keep this around even though it's unused - research went into how to do it, keeping it ensures that research doesn't need to be repeated
    #[allow(dead_code)]
    unsafe fn store1_hi_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);

    // math ops
    unsafe fn neg(a: Self) -> Self;
    unsafe fn add(a: Self, b: Self) -> Self;
    unsafe fn sub(a: Self, b: Self) -> Self;
    unsafe fn mul(a: Self, b: Self) -> Self;
    unsafe fn fmadd(acc: Self, a: Self, b: Self) -> Self;
    unsafe fn nmadd(acc: Self, a: Self, b: Self) -> Self;

    unsafe fn broadcast_scalar(value: Self::ScalarType) -> Self;

    /// Pairwise multiply the complex numbers in `left` with the complex numbers in `right`.
    unsafe fn mul_complex(left: Self, right: Self) -> Self;

    /// The same as `mul_complex`, except that `left` is conjugated first: the result is
    /// `left.conj() * right`.
    ///
    /// Bluestein's and Rader's both need a product conjugated, and `(a * b).conj()` is
    /// `a.conj() * b.conj()`, so storing the precomputed side pre-conjugated turns every one of
    /// those into this operation. Complex multiplication is commutative, so this also covers
    /// `left * right.conj()` by swapping the arguments. Every backend reaches it in the same
    /// instruction count as `mul_complex`, or one more.
    unsafe fn mul_complex_conjugated(left: Self, right: Self) -> Self;

    /// Generates a chunk of twiddle factors starting at (X,Y) and incrementing X
    /// `COMPLEX_PER_VECTOR` times.
    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: FftDirection,
    ) -> Self;

    unsafe fn make_rotate90(direction: FftDirection) -> Self::Rotation;
    unsafe fn apply_rotate90(direction: Self::Rotation, values: Self) -> Self;

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

    /// Run one cross-FFT layer over `data`, with the backend's target feature enabled.
    ///
    /// The layer's butterfly is chosen by matching on `factor` inside this function, rather than by
    /// the caller passing a closure, because a closure inherits target features from the function it
    /// is written in and nothing else propagates them. A closure written outside a
    /// `#[target_feature]` function is compiled without the feature, and on a target where the
    /// backend's instruction set is not in the baseline, wasm's `simd128` and aarch64's `fcma` both
    /// being examples, every intrinsic reached from it then becomes an out-of-line call.
    ///
    /// Relying on `#[inline(always)]` to pull the layer up into the `fft_helper_*` boundary instead
    /// does not work, because the chunk closure that boundary takes is itself written outside it.
    /// That is what made wasm_simd 12x slower than the old WASM Radix4. Every backend implements
    /// this with `simd_vector_cross_layer!`, so the whole layer lands inside the feature whatever
    /// the inliner decides, at the cost of one non-inlinable call per layer.
    unsafe fn cross_layer_radixn(
        data: &mut [Complex<Self::ScalarType>],
        twiddles: &[Self],
        num_columns: usize,
        factor: &InternalRadixFactor<Self>,
    );
    unsafe fn cross_layer_radix4(
        data: &mut [Complex<Self::ScalarType>],
        twiddles: &[Self],
        num_columns: usize,
        rotation: &Self::Rotation,
    );

    /// `dest[i] = src[i] * multiplier[i]`, for every element of `dest`.
    ///
    /// These three are the pairwise multiply loops of Bluestein's and Rader's, and carry the
    /// backend's target feature for the same reason `cross_layer_radixn` does. Every backend
    /// implements them with `simd_vector_multiply_loops!`.
    unsafe fn mul_complex_slice(
        src: &[Complex<Self::ScalarType>],
        dest: &mut [Complex<Self::ScalarType>],
        multiplier: &[Complex<Self::ScalarType>],
    );
    /// `dest[i] = src[i].conj() * multiplier[i]`, for every element of `dest`.
    unsafe fn mul_complex_conjugated_slice(
        src: &[Complex<Self::ScalarType>],
        dest: &mut [Complex<Self::ScalarType>],
        multiplier: &[Complex<Self::ScalarType>],
    );
    /// `buffer[i] = buffer[i].conj() * multiplier[i]`, for every element of `buffer`.
    unsafe fn mul_complex_conjugated_slice_inplace(
        buffer: &mut [Complex<Self::ScalarType>],
        multiplier: &[Complex<Self::ScalarType>],
    );

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

/// The `SimdVector::cross_layer` impl, which is the same for every backend apart from the attribute
/// it carries, so each backend invokes this once per vector type inside its `impl` and passes that
/// attribute. A backend whose instruction set is not in the target baseline has to pass its
/// `#[target_feature]`, so that the butterfly closures below inherit it; see `cross_layer` for why.
/// One whose set is in the baseline passes `#[inline(always)]` instead, which leaves the layer
/// inlined into the caller as it would be without any of this.
macro_rules! simd_vector_cross_layer {
    ($(#[$attr:meta])*) => {
        $(#[$attr])*
        unsafe fn cross_layer_radixn(
            data: &mut [num_complex::Complex<Self::ScalarType>],
            twiddles: &[Self],
            num_columns: usize,
            factor: &crate::simd::simd_radixn::InternalRadixFactor<Self>,
        ) {
            use crate::simd::simd_radixn::{cross_layer_chunks, InternalRadixFactor};
            use crate::simd::simd_vector::SimdVector as Sv;

            match factor {
                InternalRadixFactor::Factor2 => {
                    cross_layer_chunks::<Self, 2, _>(data, twiddles, num_columns, |rows| {
                        <Self as Sv>::column_butterfly2(rows)
                    })
                }
                InternalRadixFactor::Factor3(bf) => {
                    cross_layer_chunks::<Self, 3, _>(data, twiddles, num_columns, |rows| {
                        <Self as Sv>::column_butterfly3(bf, rows)
                    })
                }
                InternalRadixFactor::Factor4(rotation) => {
                    cross_layer_chunks::<Self, 4, _>(data, twiddles, num_columns, |rows| {
                        <Self as Sv>::column_butterfly4(rows, *rotation)
                    })
                }
                InternalRadixFactor::Factor5(bf) => {
                    cross_layer_chunks::<Self, 5, _>(data, twiddles, num_columns, |rows| {
                        <Self as Sv>::column_butterfly5(bf, rows)
                    })
                }
                InternalRadixFactor::Factor6(bf) => {
                    cross_layer_chunks::<Self, 6, _>(data, twiddles, num_columns, |rows| {
                        <Self as Sv>::column_butterfly6(bf, rows)
                    })
                }
                InternalRadixFactor::Factor7(bf) => {
                    cross_layer_chunks::<Self, 7, _>(data, twiddles, num_columns, |rows| {
                        <Self as Sv>::column_butterfly7(bf, rows)
                    })
                }
            }
        }
        $(#[$attr])*
        unsafe fn cross_layer_radix4(
            data: &mut [num_complex::Complex<Self::ScalarType>],
            twiddles: &[Self],
            num_columns: usize,
            rotation: &Self::Rotation,
        ) {
            use crate::simd::simd_radixn::cross_layer_chunks;
            use crate::simd::simd_vector::SimdVector as Sv;
            cross_layer_chunks::<Self, 4, _>(data, twiddles, num_columns, |rows| {
                <Self as Sv>::column_butterfly4(rows, *rotation)
            })
        }
    };
}
pub(crate) use simd_vector_cross_layer;

/// One complex multiply, conjugating `left` first when `CONJUGATE` is set.
///
/// Written out rather than passed in as a closure, because a closure would not inherit the target
/// feature of the function this ends up inlined into. See `SimdVector::cross_layer_radixn`.
#[inline(always)]
unsafe fn mul_one<V: SimdVector, const CONJUGATE: bool>(left: V, right: V) -> V {
    if CONJUGATE {
        V::mul_complex_conjugated(left, right)
    } else {
        V::mul_complex(left, right)
    }
}

/// Whether the vector path of the multiply loops below is faster than a scalar loop.
/// See `SimdVector::FUSED_COMPLEX_MULTIPLY`.
#[inline(always)]
fn vectors_worthwhile<V: SimdVector>() -> bool {
    V::COMPLEX_PER_VECTOR > 1 || V::FUSED_COMPLEX_MULTIPLY
}

/// The vector path of the multiply loops: `dest[i] = src[i] * multiplier[i]` for `count` elements,
/// conjugating `src[i]` first when `CONJUGATE` is set.
///
/// Unrolled two vectors at a time to get two independent dependency chains, the same as
/// `SimdRadixN`'s cross layers. No backend holds more than two complex numbers in a vector, so a
/// `count` that isn't a whole number of vectors leaves exactly one element over; that one is loaded
/// into and stored from the low half of a vector, the same as the partial columns in `SimdRadixN`.
///
/// These are raw pointers so that the in-place case can pass the same one for `src` and `dest`. All
/// three must be valid for `count` elements, which the callers below check.
#[inline(always)]
unsafe fn mul_complex_vectors<V: SimdVector, const CONJUGATE: bool>(
    src: *const Complex<V::ScalarType>,
    dest: *mut Complex<V::ScalarType>,
    multiplier: *const Complex<V::ScalarType>,
    count: usize,
) {
    let per_vector = V::COMPLEX_PER_VECTOR;
    let full_vectors = count / per_vector;

    for i in 0..full_vectors / 2 {
        let index = i * 2 * per_vector;
        let next = index + per_vector;

        let left_a = V::load_complex(src.add(index));
        let left_b = V::load_complex(src.add(next));

        let product_a = mul_one::<V, CONJUGATE>(left_a, V::load_complex(multiplier.add(index)));
        let product_b = mul_one::<V, CONJUGATE>(left_b, V::load_complex(multiplier.add(next)));

        V::store_complex(dest.add(index), product_a);
        V::store_complex(dest.add(next), product_b);
    }

    // an odd vector count leaves one behind
    if full_vectors % 2 == 1 {
        let index = (full_vectors - 1) * per_vector;
        let left = V::load_complex(src.add(index));
        let product = mul_one::<V, CONJUGATE>(left, V::load_complex(multiplier.add(index)));
        V::store_complex(dest.add(index), product);
    }

    // The leftover element, if there is one. `COMPLEX_PER_VECTOR` is a constant, so this whole
    // branch vanishes on the single-complex vectors, whose partial load and store are
    // unimplemented.
    if per_vector > 1 && count % per_vector != 0 {
        let index = full_vectors * per_vector;
        let left = V::load1_lo_complex(src.add(index));
        let product = mul_one::<V, CONJUGATE>(left, V::load1_lo_complex(multiplier.add(index)));
        V::store1_lo_complex(dest.add(index), product);
    }
}

/// `dest[i] = src[i] * multiplier[i]`, for every element of `dest`, conjugating `src[i]` first when
/// `CONJUGATE` is set. `src` and `multiplier` must be at least as long as `dest`.
#[inline(always)]
pub(crate) unsafe fn mul_complex_chunks<V: SimdVector, const CONJUGATE: bool>(
    src: &[Complex<V::ScalarType>],
    dest: &mut [Complex<V::ScalarType>],
    multiplier: &[Complex<V::ScalarType>],
) {
    let count = dest.len();
    assert!(src.len() >= count && multiplier.len() >= count);

    if vectors_worthwhile::<V>() {
        mul_complex_vectors::<V, CONJUGATE>(
            src.as_ptr(),
            dest.as_mut_ptr(),
            multiplier.as_ptr(),
            count,
        );
    } else {
        // Three separate slices, so the compiler knows they don't alias and is free to
        // autovectorize this into deinterleaving loads.
        for ((dest, src), multiplier) in dest.iter_mut().zip(src.iter()).zip(multiplier.iter()) {
            *dest = if CONJUGATE {
                src.conj() * multiplier
            } else {
                src * multiplier
            };
        }
    }
}

/// `buffer[i] = buffer[i].conj() * multiplier[i]`, the in-place form of `mul_complex_chunks`.
#[inline(always)]
pub(crate) unsafe fn mul_complex_conjugated_chunks_inplace<V: SimdVector>(
    buffer: &mut [Complex<V::ScalarType>],
    multiplier: &[Complex<V::ScalarType>],
) {
    let count = buffer.len();
    assert!(multiplier.len() >= count);

    if vectors_worthwhile::<V>() {
        let ptr = buffer.as_mut_ptr();
        mul_complex_vectors::<V, true>(ptr, ptr, multiplier.as_ptr(), count);
    } else {
        for (buffer, multiplier) in buffer.iter_mut().zip(multiplier.iter()) {
            *buffer = buffer.conj() * multiplier;
        }
    }
}

/// The three pairwise multiply loop impls, which like `simd_vector_cross_layer` are the same for
/// every backend apart from the attribute they carry. Each backend invokes this once per vector
/// type inside its `impl`, passing the same attribute it passes there.
macro_rules! simd_vector_multiply_loops {
    ($(#[$attr:meta])*) => {
        $(#[$attr])*
        unsafe fn mul_complex_slice(
            src: &[num_complex::Complex<Self::ScalarType>],
            dest: &mut [num_complex::Complex<Self::ScalarType>],
            multiplier: &[num_complex::Complex<Self::ScalarType>],
        ) {
            crate::simd::simd_vector::mul_complex_chunks::<Self, false>(src, dest, multiplier)
        }
        $(#[$attr])*
        unsafe fn mul_complex_conjugated_slice(
            src: &[num_complex::Complex<Self::ScalarType>],
            dest: &mut [num_complex::Complex<Self::ScalarType>],
            multiplier: &[num_complex::Complex<Self::ScalarType>],
        ) {
            crate::simd::simd_vector::mul_complex_chunks::<Self, true>(src, dest, multiplier)
        }
        $(#[$attr])*
        unsafe fn mul_complex_conjugated_slice_inplace(
            buffer: &mut [num_complex::Complex<Self::ScalarType>],
            multiplier: &[num_complex::Complex<Self::ScalarType>],
        ) {
            crate::simd::simd_vector::mul_complex_conjugated_chunks_inplace::<Self>(
                buffer,
                multiplier,
            )
        }
    };
}
pub(crate) use simd_vector_multiply_loops;
