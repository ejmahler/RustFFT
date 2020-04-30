use std::arch::x86_64::*;
use std::fmt::Debug;

use num_complex::Complex;

use crate::common::FFTnum;

/// A SIMD vector of complex numbers, stored with the real values and imaginary values interleaved. 
/// Implemented for __m128, __m128d, __m256, __m256d, but these all require the AVX instruction set.
///
/// The goal of this trait is to reduce code duplication by letting code be generic over the vector type 
pub trait AvxVector : Copy + Debug {
    type ScalarType : FFTnum;
    const SCALAR_PER_VECTOR : usize;
    const COMPLEX_PER_VECTOR : usize;

    unsafe fn zero() -> Self;
    unsafe fn add(left: Self, right: Self) -> Self;
    unsafe fn sub(left: Self, right: Self) -> Self;
    unsafe fn xor(left: Self, right: Self) -> Self;
    unsafe fn mul(left: Self, right: Self) -> Self;
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self;
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self;
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self;

    // Reverse the order of complex numbers in the vector, so that the last is the first and the first is the last
    unsafe fn reverse_complex_elements(self) -> Self;

    /// Swap each real number with its corresponding imaginary number
    unsafe fn swap_complex_components(self) -> Self;

    /// first return is the reals duplicated into the imaginaries, second return is the imaginaries duplicated into the reals
    unsafe fn duplicate_complex_components(self) -> (Self, Self);

    // Fill a vector by repeating the provided complex number as many times as possible
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self;

    // create a Rotator90 instance to rotate complex numbers either 90 or 270 degrees, based on the value of `inverse`
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self>;


    

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn mul_complex(left: Self, right: Self) -> Self {
        // Extract the real and imaginary components from left into 2 separate registers
        let (left_real, left_imag) = Self::duplicate_complex_components(left);

        // create a shuffled version of right where the imaginary values are swapped with the reals
        let right_shuffled = Self::swap_complex_components(right);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = Self::mul(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then alternatingly add and subtract the left side from the right
        Self::fmaddsub(left_real, right, output_right)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn rotate90(self, rotation: Rotation90<Self>) -> Self {
        // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an inverse or not
        let elements_swapped = Self::swap_complex_components(self);

        // Use the pre-computed vector stored in the Rotation90 instance to negate either the reals or imaginaries
        Self::xor(elements_swapped, rotation.0)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        [
            Self::add(rows[0], rows[1]),
            Self::sub(rows[0], rows[1]),
        ]
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterfly3(rows: [Self; 3], twiddles: Self) -> [Self; 3] {
        // This algorithm is derived directly from the definition of the DFT of size 3
        // We'd theoretically have to do 4 complex multiplications, but all of the twiddles we'd be multiplying by are conjugates of each other
        // By doing some algebra to expand the complex multiplications and factor out the multiplications, we get this

        let [mut mid1, mid2] = Self::column_butterfly2([rows[1], rows[2]]);
        let output0 = Self::add(rows[0], mid1);

        let (twiddle_real, twiddle_imag) = Self::duplicate_complex_components(twiddles);

        mid1 = Self::fmadd(mid1, twiddle_real, rows[0]);
        
        let rotation = Self::make_rotation90(true);
        let mid2_rotated = Self::rotate90(mid2, rotation);

        let output1 = Self::fmadd(mid2_rotated, twiddle_imag, mid1);
        let output2 = Self::fnmadd(mid2_rotated, twiddle_imag, mid1);

        [output0, output1, output2]
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Rotation90<Self>) -> [Self; 4] {
        // Algorithm: 2x2 mixed radix

        // Perform the first set of size-2 FFTs.
        let [mid0, mid2] = Self::column_butterfly2([rows[0], rows[2]]);
        let [mid1, mid3] = Self::column_butterfly2([rows[1], rows[3]]);

        // Apply twiddle factors (in this case just a rotation)
        let mid3_rotated = mid3.rotate90(rotation);

        // Perform the second set of size-2 FFTs
        let [output0, output1] = Self::column_butterfly2([mid0, mid1]);
        let [output2, output3] = Self::column_butterfly2([mid2, mid3_rotated]);

        // Swap outputs 1 and 2 in the output to do a square transpose
        [output0, output2, output1, output3]
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Rotation90<V>(V);
impl<V: AvxVector256> Rotation90<V> {
    #[target_feature(enable = "avx", enable = "fma")]
    pub unsafe fn lo(self) -> Rotation90<V::HalfVector> {
        Rotation90(self.0.lo())
    }
}

/// A 256-bit SIMD vector of complex numbers, stored with the real values and imaginary values interleaved. 
/// Implemented for __m256, __m256d, but these are all oriented around AVX
///
/// This trait implements things specific to 256-types, like splitting a 256 vector into 128 vectors
pub trait AvxVector256 : AvxVector {
    type HalfVector : AvxVector128;

    unsafe fn lo(self) -> Self::HalfVector;
    unsafe fn hi(self) -> Self::HalfVector;
}

/// A 128-bit SIMD vector of complex numbers, stored with the real values and imaginary values interleaved. 
/// Implemented for __m128, __m128d, but these are all oriented around AVX
///
/// This trait implements things specific to 128-types, like merging 2 128 vectors into a 256 vector
pub trait AvxVector128 : AvxVector {
    type FullVector : AvxVector256;

    unsafe fn combine(lo: Self, hi: Self) -> Self::FullVector;
}


impl AvxVector for __m256 {
    type ScalarType = f32;
    const SCALAR_PER_VECTOR : usize = 8;
    const COMPLEX_PER_VECTOR : usize = 4;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn zero() -> Self {
        _mm256_setzero_ps()
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn xor(left: Self, right: Self) -> Self {
        _mm256_xor_ps(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn add(left: Self, right: Self) -> Self {
        _mm256_add_ps(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn sub(left: Self, right: Self) -> Self {
        _mm256_sub_ps(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn mul(left: Self, right: Self) -> Self {
        _mm256_mul_ps(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmadd_ps(left, right, add)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fnmadd_ps(left, right, add)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmaddsub_ps(left, right, add)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn reverse_complex_elements(self) -> Self {
        // swap the elements in-lane
        let permuted = _mm256_permute_ps(self, 0x4E);
        // swap the lanes
        _mm256_permute2f128_ps(permuted, permuted, 0x01)
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn swap_complex_components(self) -> Self {
        _mm256_permute_ps(self, 0xB1)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn duplicate_complex_components(self) -> (Self, Self) {
        (_mm256_moveldup_ps(self), _mm256_movehdup_ps(self))
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm256_set_ps(value.im, value.re, value.im, value.re, value.im, value.re, value.im, value.re)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self> {
        if !inverse {
            Rotation90(Self::broadcast_complex_elements(Complex::new(0.0, -0.0)))
        } else {
            Rotation90(Self::broadcast_complex_elements(Complex::new(-0.0, 0.0)))
        }
    }
}
impl AvxVector256 for __m256 {
    type HalfVector = __m128;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn lo(self) -> Self::HalfVector {
        _mm256_castps256_ps128(self)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn hi(self) -> Self::HalfVector {
        _mm256_extractf128_ps(self, 1)
    }
}



impl AvxVector for __m128 {
    type ScalarType = f32;
    const SCALAR_PER_VECTOR : usize = 4;
    const COMPLEX_PER_VECTOR : usize = 2;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn zero() -> Self {
        _mm_setzero_ps()
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn xor(left: Self, right: Self) -> Self {
        _mm_xor_ps(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn add(left: Self, right: Self) -> Self {
        _mm_add_ps(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn sub(left: Self, right: Self) -> Self {
        _mm_sub_ps(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn mul(left: Self, right: Self) -> Self {
        _mm_mul_ps(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fmadd_ps(left, right, add)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fnmadd_ps(left, right, add)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self {
        _mm_fmaddsub_ps(left, right, add)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn reverse_complex_elements(self) -> Self {
        // swap the elements in-lane
        _mm_permute_ps(self, 0x4E)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn swap_complex_components(self) -> Self {
        _mm_permute_ps(self, 0xB1)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn duplicate_complex_components(self) -> (Self, Self) {
        (_mm_moveldup_ps(self), _mm_movehdup_ps(self))
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm_set_ps(value.im, value.re, value.im, value.re)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self> {
        if !inverse {
            Rotation90(Self::broadcast_complex_elements(Complex::new(0.0, -0.0)))
        } else {
            Rotation90(Self::broadcast_complex_elements(Complex::new(-0.0, 0.0)))
        }
    }
}
impl AvxVector128 for __m128 {
    type FullVector = __m256;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn combine(lo: Self, hi: Self) -> Self::FullVector {
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1)
    }
}


impl AvxVector for __m256d {
    type ScalarType = f64;
    const SCALAR_PER_VECTOR : usize = 4;
    const COMPLEX_PER_VECTOR : usize = 2;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn zero() -> Self {
        _mm256_setzero_pd()
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn xor(left: Self, right: Self) -> Self {
        _mm256_xor_pd(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn add(left: Self, right: Self) -> Self {
        _mm256_add_pd(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn sub(left: Self, right: Self) -> Self {
        _mm256_sub_pd(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn mul(left: Self, right: Self) -> Self {
        _mm256_mul_pd(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmadd_pd(left, right, add)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fnmadd_pd(left, right, add)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmaddsub_pd(left, right, add)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn reverse_complex_elements(self) -> Self {
        _mm256_permute2f128_pd(self, self, 0x01)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn swap_complex_components(self) -> Self {
        _mm256_permute_pd(self, 0x05)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn duplicate_complex_components(self) -> (Self, Self) {
        (_mm256_movedup_pd(self), _mm256_permute_pd(self, 0x0F))
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm256_set_pd(value.im, value.re, value.im, value.re)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self> {
        if !inverse {
            Rotation90(Self::broadcast_complex_elements(Complex::new(0.0, -0.0)))
        } else {
            Rotation90(Self::broadcast_complex_elements(Complex::new(-0.0, 0.0)))
        }
    }
}
impl AvxVector256 for __m256d {
    type HalfVector = __m128d;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn lo(self) -> Self::HalfVector {
        _mm256_castpd256_pd128(self)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn hi(self) -> Self::HalfVector {
        _mm256_extractf128_pd(self, 1)
    }
}



impl AvxVector for __m128d {
    type ScalarType = f64;
    const SCALAR_PER_VECTOR : usize = 2;
    const COMPLEX_PER_VECTOR : usize = 1;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn zero() -> Self {
        _mm_setzero_pd()
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn xor(left: Self, right: Self) -> Self {
        _mm_xor_pd(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn add(left: Self, right: Self) -> Self {
        _mm_add_pd(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn sub(left: Self, right: Self) -> Self {
        _mm_sub_pd(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn mul(left: Self, right: Self) -> Self {
        _mm_mul_pd(left, right)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fmadd_pd(left, right, add)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fnmadd_pd(left, right, add)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self {
        _mm_fmaddsub_pd(left, right, add)
    }
    
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn reverse_complex_elements(self) -> Self {
        // nothing to reverse
        self
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn swap_complex_components(self) -> Self {
        _mm_permute_pd(self, 0x05)
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn duplicate_complex_components(self) -> (Self, Self) {
        (_mm_movedup_pd(self), _mm_permute_pd(self, 0x0F))
    }
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm_set_pd(value.im, value.re)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self> {
        if !inverse {
            Rotation90(Self::broadcast_complex_elements(Complex::new(0.0, -0.0)))
        } else {
            Rotation90(Self::broadcast_complex_elements(Complex::new(-0.0, 0.0)))
        }
    }
}
impl AvxVector128 for __m128d {
    type FullVector = __m256d;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn combine(lo: Self, hi: Self) -> Self::FullVector {
        _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1)
    }
}