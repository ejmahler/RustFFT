//! The vector operations shared by every SIMD backend.
//!
//! Each backend has its own vector trait (`NeonVector`, `SseVector`, `WasmVector`), and those
//! don't share a supertrait. `SimdVector` is a separate, stripped down trait that each backend
//! implements once per vector type, next to its own vector trait impls. Algorithms written against
//! it, like `SimdRadixN`, then only exist once.

use num_complex::Complex;

use crate::FftDirection;
use crate::common::FftNum;
use crate::simd::parallel_fft::ParallelFft;

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

    /// The scalar this vector holds. Always the same type as the `T` of the algorithm using it.
    type ScalarType: FftNum;

    type Butterfly2: ParallelFft<Self, 2>;
    type Butterfly3: ParallelFft<Self, 3>;
    type Butterfly4: ParallelFft<Self, 4>;
    type Butterfly5: ParallelFft<Self, 5>;
    type Butterfly6: ParallelFft<Self, 6>;
    type Butterfly7: ParallelFft<Self, 7>;

    unsafe fn zero_vector() -> Self;

    // loads of complex numbers
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self;

    /// Load a single complex number into the low element, leaving the rest zeroed.
    ///
    /// Only a vector that holds more than one complex number can do this, so the f32 vectors
    /// implement it while the single-complex f64 and scalar ones panic. Code generic over the
    /// vector type must only reach it when `COMPLEX_PER_VECTOR > 1`.
    unsafe fn load1_lo_complex(ptr: *const Complex<Self::ScalarType>) -> Self;
    unsafe fn load1_dup_complex(ptr: *const Complex<Self::ScalarType>) -> Self;

    // stores of complex numbers
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);

    /// Store the low element of a vector, writing a single complex number and leaving whatever
    /// follows it alone. Panics on the single-complex vectors, for the same reason as
    /// `load1_lo_complex`.
    unsafe fn store1_lo_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);

    // Keep this around even though it's unused - research went into how to do it, keeping it ensures that research doesn't need to be repeated
    #[allow(dead_code)]
    unsafe fn store1_hi_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);

    // Takes a row-major 2d array with a width of D and a height of Self::COMPLEX_PER_VECTOR
    // Transposes it to a 2d array with a width of Self::COMPLEX_PER_VECTOR and height of D
    unsafe fn transpose_contiguous_to_strided<const D: usize>(data: [Self; D]) -> [Self; D];
    unsafe fn transpose_square<const D: usize>(data: [Self; D]) -> [Self; D];

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

    /// Generates a chunk of twiddle factors starting at (X,Y) and incrementing X
    /// `COMPLEX_PER_VECTOR` times.
    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: FftDirection,
    ) -> Self;

    unsafe fn make_butterfly2(direction: FftDirection) -> Self::Butterfly2;
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3;
    unsafe fn make_butterfly4(direction: FftDirection) -> Self::Butterfly4;
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5;
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6;
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7;
}
