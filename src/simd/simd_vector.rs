//! The vector operations shared by every SIMD backend.
//!
//! Each backend has its own vector trait (`NeonVector`, `SseVector`, `WasmVector`), and those
//! don't share a supertrait. `SimdVector` is a separate, stripped down trait that each backend
//! implements once per vector type, next to its own vector trait impls. Algorithms written against
//! it, like `SimdRadixN`, then only exist once.

use num_complex::Complex;

use crate::common::FftNum;
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
