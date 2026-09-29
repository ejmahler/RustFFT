use std::ops::Neg;

use num_complex::Complex;
use num_traits::Zero;

use crate::{
    algorithm::butterflies::{
        Butterfly2, Butterfly3, Butterfly4, Butterfly5, Butterfly6, Butterfly7,
    },
    fft_helper::{fft_helper_immut, fft_helper_inplace, fft_helper_outofplace},
    simd::simd_vector::{simd_vector_cross_layer, SimdVector},
    twiddles, FftDirection, FftNum,
};

// Adapter for SimdVector. If we pretend that a Complex<T> is a simd vector, we can plug Complex<T> into the simd-based algorithms and avoid having separate scalar and simd implementations
// We obviously won't be directly using simd intrinsics, but the fact that the simd algorithms are optimized for simd should make the scalar code highly auto-simd-able anyways.
impl<T: FftNum> SimdVector for Complex<T> {
    const COMPLEX_PER_VECTOR: usize = 1;

    // The unroll severely hurts wasm32 scalar performance, but significantly benefits other platforms, so disable it just for wasm
    #[cfg(target_arch = "wasm32")]
    const RADIXN_CROSS_LAYER_UNROLL: bool = false;
    #[cfg(not(target_arch = "wasm32"))]
    const RADIXN_CROSS_LAYER_UNROLL: bool = true;

    type ScalarType = T;

    type Rotation = FftDirection;

    type Butterfly3 = Butterfly3<T>;

    type Butterfly5 = Butterfly5<T>;

    type Butterfly6 = Butterfly6<T>;

    type Butterfly7 = Butterfly7<T>;

    unsafe fn zero_vector() -> Self {
        Zero::zero()
    }

    unsafe fn load_complex(ptr: *const num_complex::Complex<Self::ScalarType>) -> Self {
        ptr.read()
    }

    unsafe fn load1_lo_complex(_ptr: *const num_complex::Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a partial load of complex scalars");
    }

    unsafe fn load1_dup_complex(_ptr: *const num_complex::Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a partial load of complex scalars");
    }

    unsafe fn store_complex(ptr: *mut num_complex::Complex<Self::ScalarType>, data: Self) {
        ptr.write(data);
    }

    unsafe fn store1_lo_complex(_ptr: *mut num_complex::Complex<Self::ScalarType>, _data: Self) {
        unimplemented!("Impossible to do a partial store of complex scalars");
    }

    unsafe fn store1_hi_complex(_ptr: *mut num_complex::Complex<Self::ScalarType>, _data: Self) {
        unimplemented!("Impossible to do a partial store of complex scalars");
    }

    unsafe fn neg(a: Self) -> Self {
        a.neg()
    }

    unsafe fn add(a: Self, b: Self) -> Self {
        a + b
    }

    unsafe fn sub(a: Self, b: Self) -> Self {
        a - b
    }

    unsafe fn mul(a: Self, b: Self) -> Self {
        Complex {
            re: a.re * b.re,
            im: a.im * b.im,
        }
    }

    unsafe fn fmadd(acc: Self, a: Self, b: Self) -> Self {
        Complex {
            re: a.re * b.re + acc.re,
            im: a.im * b.im + acc.im,
        }
    }

    unsafe fn nmadd(acc: Self, a: Self, b: Self) -> Self {
        Complex {
            re: -a.re * b.re + acc.re,
            im: -a.im * b.im + acc.im,
        }
    }

    unsafe fn broadcast_scalar(value: Self::ScalarType) -> Self {
        Complex {
            re: value,
            im: value,
        }
    }

    unsafe fn mul_complex(left: Self, right: Self) -> Self {
        left * right
    }

    unsafe fn make_mixedradix_twiddle_chunk(
        x: usize,
        y: usize,
        len: usize,
        direction: crate::FftDirection,
    ) -> Self {
        twiddles::compute_twiddle(x * y, len, direction)
    }

    unsafe fn make_rotate90(direction: FftDirection) -> Self::Rotation {
        direction
    }

    unsafe fn apply_rotate90(direction: Self::Rotation, values: Self) -> Self {
        twiddles::rotate_90(values, direction)
    }

    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        Butterfly3::new(direction)
    }

    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        Butterfly5::new(direction)
    }

    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        Butterfly6::new(direction)
    }

    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        Butterfly7::new(direction)
    }

    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        let mut result = rows;
        Butterfly2::new(FftDirection::Forward).perform_fft_butterfly(&mut result);
        result
    }

    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        let mut result = rows;
        bf.perform_fft_butterfly(&mut result);
        result
    }

    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Self::Rotation) -> [Self; 4] {
        let mut result = rows;
        Butterfly4::new(rotation).perform_fft_butterfly(&mut result);
        result
    }

    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5] {
        let mut result = rows;
        bf.perform_fft_butterfly(&mut result);
        result
    }

    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6] {
        let mut result = rows;
        bf.perform_fft_butterfly(&mut result);
        result
    }

    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7] {
        let mut result = rows;
        bf.perform_fft_butterfly(&mut result);
        result
    }

    simd_vector_cross_layer!(#[inline(always)]);

    #[inline(always)]
    unsafe fn fft_helper_immut<E>(
        input: &[E],
        output: &mut [E],
        scratch: &mut [E],
        chunk_size: usize,
        required_scratch: usize,
        chunk_fn: impl FnMut(&[E], &mut [E], &mut [E]),
    ) {
        fft_helper_immut(
            input,
            output,
            scratch,
            chunk_size,
            required_scratch,
            chunk_fn,
        );
    }
    #[inline(always)]
    unsafe fn fft_helper_outofplace<E>(
        input: &mut [E],
        output: &mut [E],
        scratch: &mut [E],
        chunk_size: usize,
        required_scratch: usize,
        chunk_fn: impl FnMut(&mut [E], &mut [E], &mut [E]),
    ) {
        fft_helper_outofplace(
            input,
            output,
            scratch,
            chunk_size,
            required_scratch,
            chunk_fn,
        );
    }
    #[inline(always)]
    unsafe fn fft_helper_inplace<E>(
        buffer: &mut [E],
        scratch: &mut [E],
        chunk_size: usize,
        required_scratch: usize,
        chunk_fn: impl FnMut(&mut [E], &mut [E]),
    ) {
        fft_helper_inplace(buffer, scratch, chunk_size, required_scratch, chunk_fn);
    }
}
