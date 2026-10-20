use std::ops::Neg;

use num_complex::Complex;
use num_traits::Zero;

use crate::algorithm::scalar_column_butterflies::{
    ScalarParallelButterfly2, ScalarParallelButterfly3, ScalarParallelButterfly4,
    ScalarParallelButterfly5, ScalarParallelButterfly6, ScalarParallelButterfly7,
};
use crate::{simd::simd_vector::SimdVector, twiddles, FftDirection, FftNum};

// Adapter for SimdVector. If we pretend that a Complex<T> is a simd vector, we can plug Complex<T> into the simd-based algorithms and avoid having separate scalar and simd implementations
// We obviously won't be directly using simd intrinsics, but the fact that the simd algorithms are optimized for simd should make the scalar code highly auto-simd-able anyways.
impl<T: FftNum> SimdVector for Complex<T> {
    const COMPLEX_PER_VECTOR: usize = 1;

    // The unroll severely hurts wasm32 scalar performance, but significantly benefits other platforms, so disable it just for wasm
    const RADIXN_CROSS_LAYER_UNROLL: bool = !cfg!(target_arch = "wasm32");

    type ScalarType = T;

    type Butterfly2 = ScalarParallelButterfly2<T>;
    type Butterfly3 = ScalarParallelButterfly3<T>;
    type Butterfly4 = ScalarParallelButterfly4<T>;
    type Butterfly5 = ScalarParallelButterfly5<T>;
    type Butterfly6 = ScalarParallelButterfly6<T>;
    type Butterfly7 = ScalarParallelButterfly7<T>;

    unsafe fn zero_vector() -> Self {
        Zero::zero()
    }

    unsafe fn load_complex(ptr: *const num_complex::Complex<Self::ScalarType>) -> Self { unsafe {
        ptr.read()
    }}

    unsafe fn load1_lo_complex(_ptr: *const num_complex::Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a partial load of complex scalars");
    }

    unsafe fn load1_dup_complex(_ptr: *const num_complex::Complex<Self::ScalarType>) -> Self {
        unimplemented!("Unused: a dup load of one complex scalar is just load_complex");
    }

    unsafe fn store_complex(ptr: *mut num_complex::Complex<Self::ScalarType>, data: Self) { unsafe {
        ptr.write(data);
    }}

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

    unsafe fn make_butterfly2(direction: FftDirection) -> Self::Butterfly2 {
        Self::Butterfly2::new(direction)
    }
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        Self::Butterfly3::new(direction)
    }
    unsafe fn make_butterfly4(direction: FftDirection) -> Self::Butterfly4 {
        Self::Butterfly4::new(direction)
    }
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        Self::Butterfly5::new(direction)
    }
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        Self::Butterfly6::new(direction)
    }
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        Self::Butterfly7::new(direction)
    }
}
