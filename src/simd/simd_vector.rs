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
pub trait SimdVector: Copy + Send + Sync + Sized {
    const COMPLEX_PER_VECTOR: usize;

    /// The scalar this vector holds. Always the same type as the `T` of the algorithm using it.
    type ScalarType: FftNum;

    /// The backend's precomputed 90 degree rotation, which the radix 4 butterfly needs.
    type Rotation: Copy + Send + Sync;

    type Butterfly3: Send + Sync;
    type Butterfly5: Send + Sync;
    type Butterfly6: Send + Sync;
    type Butterfly7: Send + Sync;

    unsafe fn zero() -> Self;
    unsafe fn load(data: &[Complex<Self::ScalarType>], index: usize) -> Self;
    unsafe fn store(data: &mut [Complex<Self::ScalarType>], value: Self, index: usize);

    /// Pairwise multiply the complex numbers in `left` with the complex numbers in `right`.
    unsafe fn mul_complex(left: Self, right: Self) -> Self;

    /// Generates a chunk of twiddle factors starting at (X,Y) and incrementing X
    /// `COMPLEX_PER_VECTOR` times.
    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: FftDirection,
    ) -> Self;

    unsafe fn make_rotate90(direction: FftDirection) -> Self::Rotation;
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
