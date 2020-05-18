use std::arch::x86_64::*;
use std::fmt::Debug;

use num_complex::Complex;
use num_traits::Zero;

use crate::common::FFTnum;
use super::avx32_utils::AvxComplexArrayf32;
use super::avx64_utils::AvxComplexArray64;
use crate::array_utils::{RawSlice, RawSliceMut};

/// A SIMD vector of complex numbers, stored with the real values and imaginary values interleaved. 
/// Implemented for __m128, __m128d, __m256, __m256d, but these all require the AVX instruction set.
///
/// The goal of this trait is to reduce code duplication by letting code be generic over the vector type 
pub trait AvxVector : Copy + Debug {
    type ScalarType : FFTnum;
    const SCALAR_PER_VECTOR : usize;
    const COMPLEX_PER_VECTOR : usize;

    // useful constants
    unsafe fn zero() -> Self;
    unsafe fn half_root2() -> Self; // an entire vector filled with 0.5.sqrt()

    // Basic operations that map directly to 1-2 AVX intrinsics
    unsafe fn add(left: Self, right: Self) -> Self;
    unsafe fn sub(left: Self, right: Self) -> Self;
    unsafe fn xor(left: Self, right: Self) -> Self;
    unsafe fn mul(left: Self, right: Self) -> Self;
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self;
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self;
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self;

    // loads/stores of complex numbers
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self;
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);

    // More basic operations that end up being implemented in 1-2 intrinsics, but unlike the ones above, these have higher-level meaning than just arithmetic
    /// Swap each real number with its corresponding imaginary number
    unsafe fn swap_complex_components(self) -> Self;

    /// first return is the reals duplicated into the imaginaries, second return is the imaginaries duplicated into the reals
    unsafe fn duplicate_complex_components(self) -> (Self, Self);

    /// Reverse the order of complex numbers in the vector, so that the last is the first and the first is the last
    unsafe fn reverse_complex_elements(self) -> Self;

    /// Fill a vector by repeating the provided complex number as many times as possible
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self;

    /// Fill a vector by computing a twiddle factor and repeating it across the whole vector
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self;

    /// create a Rotator90 instance to rotate complex numbers either 90 or 270 degrees, based on the value of `inverse`
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self>;

    /// Generates a chunk of twiddle factors starting at (X,Y) and incrementing X `COMPLEX_PER_VECTOR` times.
    /// The result will be [twiddle(x*y, len), twiddle((x+1)*y, len), twiddle((x+2)*y, len), ...] for as many complex numbers fit in a vector
    unsafe fn make_mixedradix_twiddle_chunk(x: usize, y: usize, len: usize, inverse: bool) -> Self;


    

    #[inline(always)]
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

    #[inline(always)]
    unsafe fn rotate90(self, rotation: Rotation90<Self>) -> Self {
        // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an inverse or not
        let elements_swapped = Self::swap_complex_components(self);

        // Use the pre-computed vector stored in the Rotation90 instance to negate either the reals or imaginaries
        Self::xor(elements_swapped, rotation.0)
    }



    #[inline(always)]
    unsafe fn column_butterfly2(rows: [Self; 2]) -> [Self; 2] {
        [
            Self::add(rows[0], rows[1]),
            Self::sub(rows[0], rows[1]),
        ]
    }

    #[inline(always)]
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

    #[inline(always)]
    unsafe fn column_butterfly4(rows: [Self; 4], rotation: Rotation90<Self>) -> [Self; 4] {
        // Algorithm: 2x2 mixed radix

        // Perform the first set of size-2 FFTs.
        let [mid0, mid2] = Self::column_butterfly2([rows[0], rows[2]]);
        let [mid1, mid3] = Self::column_butterfly2([rows[1], rows[3]]);

        // Apply twiddle factors (in this case just a rotation)
        let mid3_rotated = mid3.rotate90(rotation);

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = Self::column_butterfly2([mid0, mid1]);
        let [output2, output3] = Self::column_butterfly2([mid2, mid3_rotated]);

        // Swap outputs 1 and 2 in the output to do a square transpose
        [output0, output2, output1, output3]
    }

    // A niche variant of column_butterfly4 that negates row 3 before performing the FFT. It's able to roll it into existing instructions, so the negation is free
    #[inline(always)]
    unsafe fn column_butterfly4_negaterow3(rows: [Self; 4], rotation: Rotation90<Self>) -> [Self; 4] {
        // Algorithm: 2x2 mixed radix

        // Perform the first set of size-2 FFTs.
        let [mid0, mid2] = Self::column_butterfly2([rows[0], rows[2]]);
        let (mid1, mid3) = (Self::sub(rows[1], rows[3]), Self::add(rows[1], rows[3])); // to negate row 3, swap add and sub in the butterfly 2

        // Apply twiddle factors (in this case just a rotation)
        let mid3_rotated = mid3.rotate90(rotation);

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = Self::column_butterfly2([mid0, mid1]);
        let [output2, output3] = Self::column_butterfly2([mid2, mid3_rotated]);

        // Swap outputs 1 and 2 in the output to do a square transpose
        [output0, output2, output1, output3]
    }

    #[inline(always)]
    unsafe fn column_butterfly8(rows: [Self; 8], rotation: Rotation90<Self>) -> [Self; 8] {
        // Algorithm: 4x2 mixed radix

        // Size-4 FFTs down the columns
        let mid0     = Self::column_butterfly4([rows[0], rows[2], rows[4], rows[6]], rotation);
        let mut mid1 = Self::column_butterfly4([rows[1], rows[3], rows[5], rows[7]], rotation);

        // Apply twiddle factors
        mid1[1] = apply_butterfly8_twiddle1(mid1[1], rotation);
        mid1[2] = mid1[2].rotate90(rotation);
        mid1[3] = apply_butterfly8_twiddle3(mid1[3], rotation);

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = Self::column_butterfly2([mid0[0], mid1[0]]);
        let [output2, output3] = Self::column_butterfly2([mid0[1], mid1[1]]);
        let [output4, output5] = Self::column_butterfly2([mid0[2], mid1[2]]);
        let [output6, output7] = Self::column_butterfly2([mid0[3], mid1[3]]);

        [output0, output2, output4, output6, output1, output3, output5, output7]
    }

    #[inline(always)]
    unsafe fn column_butterfly16(rows: [Self; 16], twiddles: [Self; 2], rotation: Rotation90<Self>) -> [Self; 16] {
        // Algorithm: 4x4 mixed radix

        // Size-4 FFTs down the columns
        let mid0     = Self::column_butterfly4([rows[0], rows[4], rows[8],  rows[12]], rotation);
        let mut mid1 = Self::column_butterfly4([rows[1], rows[5], rows[9],  rows[13]], rotation);
        let mut mid2 = Self::column_butterfly4([rows[2], rows[6], rows[10], rows[14]], rotation);
        let mut mid3 = Self::column_butterfly4([rows[3], rows[7], rows[11], rows[15]], rotation);

        // Apply twiddle factors
        mid1[1] = Self::mul_complex(mid1[1], twiddles[0]);

        // for twiddle(2, 16), we can use the butterfly8 twiddle1 instead, which takes fewer instructions and fewer multiplies
        mid2[1] = apply_butterfly8_twiddle1(mid2[1], rotation);
        mid1[2] = apply_butterfly8_twiddle1(mid1[2], rotation);

        // for twiddle(3,16), we can use twiddle(1,16), sort of, but we'd need a branch, and at this point it's easier to just have another vector
        mid3[1] = Self::mul_complex(mid3[1], twiddles[1]);
        mid1[3] = Self::mul_complex(mid1[3], twiddles[1]);

        // twiddle(4,16) is just a rotate
        mid2[2] = mid2[2].rotate90(rotation);

        // for twiddle(6, 16), we can use the butterfly8 twiddle3 instead, which takes fewer instructions and fewer multiplies
        mid3[2] = apply_butterfly8_twiddle3(mid3[2], rotation);
        mid2[3] = apply_butterfly8_twiddle3(mid2[3], rotation);

        // twiddle(9, 16) is twiddle (1,16) negated. we're just going to use the same twiddle for now, and apply the negation as a part of our subsequent butterfly 4's
        mid3[3] = Self::mul_complex(mid3[3], twiddles[0]);

        // Up next is a transpose, but since everything is already in registers, we don't actually have to transpose anything!
        // "transpose" and thne apply butterfly 4's across the columns of our 4x4 array
        let output0 = Self::column_butterfly4([mid0[0], mid1[0], mid2[0], mid3[0]], rotation);
        let output1 = Self::column_butterfly4([mid0[1], mid1[1], mid2[1], mid3[1]], rotation);
        let output2 = Self::column_butterfly4([mid0[2], mid1[2], mid2[2], mid3[2]], rotation);
        let output3 = Self::column_butterfly4_negaterow3([mid0[3], mid1[3], mid2[3], mid3[3]], rotation); // finish the twiddle of the last row by negating it

        // finally, one more transpose
        [output0[0], output1[0], output2[0], output3[0], output0[1], output1[1], output2[1], output3[1], output0[2], output1[2], output2[2], output3[2], output0[3], output1[3], output2[3], output3[3]]
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
    unsafe fn split(self) -> (Self::HalfVector, Self::HalfVector) {
        (self.lo(), self.hi())
    }
    unsafe fn merge(lo: Self::HalfVector, hi: Self::HalfVector) -> Self;

    // loads/stores of partial vectors of complex numbers. When loading, empty elements are zeroed
    // unimplemented!() if Self::COMPLEX_PER_VECTOR is not greater than the partial count
    unsafe fn load_partial1_complex(ptr: *const Complex<Self::ScalarType>) -> Self::HalfVector;
    unsafe fn load_partial2_complex(ptr: *const Complex<Self::ScalarType>) -> Self::HalfVector;
    unsafe fn load_partial3_complex(ptr: *const Complex<Self::ScalarType>) -> Self;
    unsafe fn store_partial1_complex(ptr: *mut Complex<Self::ScalarType>, data: Self::HalfVector);
    unsafe fn store_partial2_complex(ptr: *mut Complex<Self::ScalarType>, data: Self::HalfVector);
    unsafe fn store_partial3_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);

    #[inline(always)]
    unsafe fn column_butterfly6(rows: [Self; 6], twiddles: Self) -> [Self; 6] {
        // Algorithm: 3x2 good-thomas

        // Size-3 FFTs down the columns of our reordered array
        let mid0 = Self::column_butterfly3([rows[0], rows[2], rows[4]], twiddles);
        let mid1 = Self::column_butterfly3([rows[3], rows[5], rows[1]], twiddles);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = Self::column_butterfly2([mid0[0], mid1[0]]);
        let [output2, output3] = Self::column_butterfly2([mid0[1], mid1[1]]);
        let [output4, output5] = Self::column_butterfly2([mid0[2], mid1[2]]);

        // Reorder into output
        [output0, output3, output4, output1, output2, output5]
    }

    #[inline(always)]
    unsafe fn column_butterfly9(rows: [Self; 9], twiddles: [Self;3], butterfly3_twiddles: Self) -> [Self; 9] {
        // Algorithm: 3x3 mixed radix

        // Size-3 FFTs down the columns
        let mid0 = Self::column_butterfly3([rows[0], rows[3], rows[6]], butterfly3_twiddles);
        let mut mid1 = Self::column_butterfly3([rows[1], rows[4], rows[7]], butterfly3_twiddles);
        let mut mid2 = Self::column_butterfly3([rows[2], rows[5], rows[8]], butterfly3_twiddles);

        // Apply twiddle factors. Note that we're re-using twiddles[1]
        mid1[1] = Self::mul_complex(twiddles[0], mid1[1]);
        mid1[2] = Self::mul_complex(twiddles[1], mid1[2]);
        mid2[1] = Self::mul_complex(twiddles[1], mid2[1]);
        mid2[2] = Self::mul_complex(twiddles[2], mid2[2]);

        let [output0, output1, output2] = Self::column_butterfly3([mid0[0], mid1[0], mid2[0]], butterfly3_twiddles);
        let [output3, output4, output5] = Self::column_butterfly3([mid0[1], mid1[1], mid2[1]], butterfly3_twiddles);
        let [output6, output7, output8] = Self::column_butterfly3([mid0[2], mid1[2], mid2[2]], butterfly3_twiddles);

        [output0, output3, output6, output1, output4, output7, output2, output5, output8]
    }

    #[inline(always)]
    unsafe fn column_butterfly12(rows: [Self; 12], butterfly3_twiddles: Self, rotation: Rotation90<Self>) -> [Self; 12] {
        // Algorithm: 4x3 good-thomas

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = Self::column_butterfly4([rows[0], rows[3], rows[6], rows[9]], rotation);
        let mid1 = Self::column_butterfly4([rows[4], rows[7], rows[10],rows[1]], rotation);
        let mid2 = Self::column_butterfly4([rows[8], rows[11],rows[2], rows[5]], rotation);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1, output2] = Self::column_butterfly3([mid0[0], mid1[0], mid2[0]], butterfly3_twiddles);
        let [output3, output4, output5] = Self::column_butterfly3([mid0[1], mid1[1], mid2[1]], butterfly3_twiddles);
        let [output6, output7, output8] = Self::column_butterfly3([mid0[2], mid1[2], mid2[2]], butterfly3_twiddles);
        let [output9, output10,output11]= Self::column_butterfly3([mid0[3], mid1[3], mid2[3]], butterfly3_twiddles);

        [output0, output4, output8, output9, output1, output5, output6, output10, output2, output3, output7, output11]
    }
}

/// A 128-bit SIMD vector of complex numbers, stored with the real values and imaginary values interleaved. 
/// Implemented for __m128, __m128d, but these are all oriented around AVX
///
/// This trait implements things specific to 128-types, like merging 2 128 vectors into a 256 vector
pub trait AvxVector128 : AvxVector {
    type FullVector : AvxVector256;

    unsafe fn merge(lo: Self, hi: Self) -> Self::FullVector;

    unsafe fn lo(input: Self::FullVector) -> Self;
    unsafe fn hi(input: Self::FullVector) -> Self;
    unsafe fn split(input: Self::FullVector) -> (Self, Self) {
        (Self::lo(input), Self::hi(input))
    }
    unsafe fn lo_rotation(input: Rotation90<Self::FullVector>) -> Rotation90<Self>;

    #[inline(always)]
    unsafe fn column_butterfly6(rows: [Self; 6], twiddles: Self::FullVector) -> [Self; 6] {
        // Algorithm: 3x2 good-thomas

        // if we merge some of our 128 registers into 256 registers, we can do 1 inner butterfly3 instead of 2
        let rows03 = Self::merge(rows[0], rows[3]);
        let rows25 = Self::merge(rows[2], rows[5]);
        let rows41 = Self::merge(rows[4], rows[1]);

        // Size-3 FFTs down the columns of our reordered array
        let mid = Self::FullVector::column_butterfly3([rows03, rows25, rows41], twiddles);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // we can't use our merged columns anymore. so split them back into half vectors
        let (mid0_0, mid1_0) = Self::split(mid[0]);
        let (mid0_1, mid1_1) = Self::split(mid[1]);
        let (mid0_2, mid1_2) = Self::split(mid[2]);

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = Self::column_butterfly2([mid0_0, mid1_0]);
        let [output2, output3] = Self::column_butterfly2([mid0_1, mid1_1]);
        let [output4, output5] = Self::column_butterfly2([mid0_2, mid1_2]);

        // Reorder into output
        [output0, output3, output4, output1, output2, output5]
    }

    #[inline(always)]
    unsafe fn column_butterfly9(rows: [Self; 9], twiddles_merged: [Self::FullVector;2], butterfly3_twiddles: Self::FullVector) -> [Self; 9] {
        // Algorithm: 3x3 mixed radix

        // if we merge some of our 128 registers into 256 registers, we can do 2 inner butterfly3's instead of 3
        let rows12 = Self::merge(rows[1], rows[2]);
        let rows45 = Self::merge(rows[4], rows[5]);
        let rows78 = Self::merge(rows[7], rows[8]);

        let mid0 = Self::column_butterfly3([rows[0], rows[3], rows[6]], Self::lo(butterfly3_twiddles));
        let mut mid12 = Self::FullVector::column_butterfly3([rows12, rows45, rows78], butterfly3_twiddles);

        // Apply twiddle factors. we're applying them on the merged set of vectors, so we need slightly different twiddle factors
        mid12[1] = Self::FullVector::mul_complex(twiddles_merged[0], mid12[1]);
        mid12[2] = Self::FullVector::mul_complex(twiddles_merged[1], mid12[2]);

        // we can't use our merged columns anymore. so split them back into half vectors
        let (mid1_0, mid2_0) = Self::split(mid12[0]);
        let (mid1_1, mid2_1) = Self::split(mid12[1]);
        let (mid1_2, mid2_2) = Self::split(mid12[2]);
        
        // Re-merge our half vectors into different, transposed full vectors. Thankfully the compiler is smart enough to combine these inserts and extracts into permutes
        let transposed12 = Self::merge(mid0[1], mid0[2]);
        let transposed45 = Self::merge(mid1_1, mid1_2);
        let transposed78 = Self::merge(mid2_1, mid2_2);

        let [output0, output1, output2] = Self::column_butterfly3([mid0[0], mid1_0, mid2_0], Self::lo(butterfly3_twiddles));
        let [output36, output47, output58] = Self::FullVector::column_butterfly3([transposed12, transposed45, transposed78], butterfly3_twiddles);

        // Finally, extract our second set of merged columns
        let (output3, output6) = Self::split(output36);
        let (output4, output7) = Self::split(output47);
        let (output5, output8) = Self::split(output58);

        [output0, output3, output6, output1, output4, output7, output2, output5, output8]
    }

    #[inline(always)]
    unsafe fn column_butterfly12(rows: [Self; 12], butterfly3_twiddles: Self::FullVector, rotation: Rotation90<Self::FullVector>) -> [Self; 12] {
        // Algorithm: 4x3 good-thomas

        // if we merge some of our 128 registers into 256 registers, we can do 2 inner butterfly4's instead of 3
        let rows48  = Self::merge(rows[4], rows[8]);
        let rows711 = Self::merge(rows[7], rows[11]);
        let rows102 = Self::merge(rows[10], rows[2]);
        let rows15  = Self::merge(rows[1], rows[5]);

        // Size-4 FFTs down the columns of our reordered array
        let mid0 = Self::column_butterfly4([rows[0], rows[3], rows[6], rows[9]], Self::lo_rotation(rotation));
        let mid12 = Self::FullVector::column_butterfly4([rows48, rows711, rows102, rows15], rotation);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // we can't use our merged columns anymore. so split them back into half vectors
        let (mid1_0, mid2_0) = Self::split(mid12[0]);
        let (mid1_1, mid2_1) = Self::split(mid12[1]);
        let (mid1_2, mid2_2) = Self::split(mid12[2]);
        let (mid1_3, mid2_3) = Self::split(mid12[3]);

        // Re-merge our half vectors into different, transposed full vectors. This will let us do 2 inner butterfly 3's instead of 4!
        // Thankfully the compiler is smart enough to combine these inserts and extracts into permutes
        let transposed03 = Self::merge(mid0[0], mid0[1]);
        let transposed14 = Self::merge(mid1_0, mid1_1);
        let transposed25 = Self::merge(mid2_0, mid2_1);

        let transposed69  = Self::merge(mid0[2], mid0[3]);
        let transposed710 = Self::merge(mid1_2, mid1_3);
        let transposed811 = Self::merge(mid2_2, mid2_3);

        // Transpose the data and do size-2 FFTs down the columns
        let [output03, output14, output25] = Self::FullVector::column_butterfly3([transposed03, transposed14, transposed25], butterfly3_twiddles);
        let [output69, output710, output811] = Self::FullVector::column_butterfly3([transposed69, transposed710, transposed811], butterfly3_twiddles);

        // Finally, extract our second set of merged columns
        let (output0, output3) = Self::split(output03);
        let (output1, output4) = Self::split(output14);
        let (output2, output5) = Self::split(output25);
        let (output6, output9) = Self::split(output69);
        let (output7, output10) = Self::split(output710);
        let (output8, output11) = Self::split(output811);

        [output0, output4, output8, output9, output1, output5, output6, output10, output2, output3, output7, output11]
    }
}

#[inline(always)]
unsafe fn apply_butterfly8_twiddle1<V: AvxVector>(input: V, rotation: Rotation90<V>) -> V {
    let rotated = input.rotate90(rotation);
    let combined = V::add(rotated, input);
    V::mul(V::half_root2(), combined)
}
#[inline(always)]
unsafe fn apply_butterfly8_twiddle3<V: AvxVector>(input: V, rotation: Rotation90<V>) -> V {
    let rotated = input.rotate90(rotation);
    let combined = V::sub(rotated, input);
    V::mul(V::half_root2(), combined)
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Rotation90<V>(V);
impl<V: AvxVector256> Rotation90<V> {
    #[inline(always)]
    pub unsafe fn lo(self) -> Rotation90<V::HalfVector> {
        Rotation90(self.0.lo())
    }
}


impl AvxVector for __m256 {
    type ScalarType = f32;
    const SCALAR_PER_VECTOR : usize = 8;
    const COMPLEX_PER_VECTOR : usize = 4;

    #[inline(always)]
    unsafe fn zero() -> Self {
        _mm256_setzero_ps()
    }
    #[inline(always)]
    unsafe fn half_root2() -> Self {
        // note: we're computing a square root here, but checking the assembly says the compiler is smart enough to turn this into a constant
        _mm256_broadcast_ss(&0.5f32.sqrt())
    }

    #[inline(always)]
    unsafe fn xor(left: Self, right: Self) -> Self {
        _mm256_xor_ps(left, right)
    }
    #[inline(always)]
    unsafe fn add(left: Self, right: Self) -> Self {
        _mm256_add_ps(left, right)
    }
    #[inline(always)]
    unsafe fn sub(left: Self, right: Self) -> Self {
        _mm256_sub_ps(left, right)
    }
    #[inline(always)]
    unsafe fn mul(left: Self, right: Self) -> Self {
        _mm256_mul_ps(left, right)
    }
    #[inline(always)]
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmadd_ps(left, right, add)
    }
    #[inline(always)]
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fnmadd_ps(left, right, add)
    }
    #[inline(always)]
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmaddsub_ps(left, right, add)
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm256_loadu_ps(ptr as *const Self::ScalarType)
    }
    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm256_storeu_ps(ptr as *mut Self::ScalarType, data)
    }

    #[inline(always)]
    unsafe fn reverse_complex_elements(self) -> Self {
        // swap the elements in-lane
        let permuted = _mm256_permute_ps(self, 0x4E);
        // swap the lanes
        _mm256_permute2f128_ps(permuted, permuted, 0x01)
    }
    
    #[inline(always)]
    unsafe fn swap_complex_components(self) -> Self {
        _mm256_permute_ps(self, 0xB1)
    }

    #[inline(always)]
    unsafe fn duplicate_complex_components(self) -> (Self, Self) {
        (_mm256_moveldup_ps(self), _mm256_movehdup_ps(self))
    }

    #[inline(always)]
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm256_set_ps(value.im, value.re, value.im, value.re, value.im, value.re, value.im, value.re)
    }

    #[inline(always)]
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self> {
        if !inverse {
            Rotation90(Self::broadcast_complex_elements(Complex::new(0.0, -0.0)))
        } else {
            Rotation90(Self::broadcast_complex_elements(Complex::new(-0.0, 0.0)))
        }
    }

    #[inline(always)]
    unsafe fn make_mixedradix_twiddle_chunk(x: usize, y: usize, len: usize, inverse: bool) -> Self {
        let mut twiddle_chunk = [Complex::zero(); Self::COMPLEX_PER_VECTOR];
        for i in 0..Self::COMPLEX_PER_VECTOR {
            twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x + i), len, inverse);
        }

        twiddle_chunk.load_complex_f32(0)
    }

    #[inline(always)]
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self {
        Self::broadcast_complex_elements(Self::ScalarType::generate_twiddle_factor(index, len, inverse))
    }
}
impl AvxVector256 for __m256 {
    type HalfVector = __m128;

    #[inline(always)]
    unsafe fn lo(self) -> Self::HalfVector {
        _mm256_castps256_ps128(self)
    }
    #[inline(always)]
    unsafe fn hi(self) -> Self::HalfVector {
        _mm256_extractf128_ps(self, 1)
    }
    #[inline(always)]
    unsafe fn merge(lo: Self::HalfVector, hi: Self::HalfVector) -> Self {
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1)
    }
    #[inline(always)]
    unsafe fn load_partial1_complex(ptr: *const Complex<Self::ScalarType>) -> Self::HalfVector {
        let data = _mm_load_sd(ptr as *const f64);
        _mm_castpd_ps(data)
    }
    #[inline(always)]
    unsafe fn load_partial2_complex(ptr: *const Complex<Self::ScalarType>) -> Self::HalfVector {
        _mm_loadu_ps(ptr as *const f32)
    }
    #[inline(always)]
    unsafe fn load_partial3_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        let lo = Self::load_partial2_complex(ptr);
        let hi = Self::load_partial1_complex(ptr.add(2));
        Self::merge(lo, hi)
    }
    #[inline(always)]
    unsafe fn store_partial1_complex(ptr: *mut Complex<Self::ScalarType>, data: Self::HalfVector) {
        _mm_store_sd(ptr as *mut f64, _mm_castps_pd(data));
    }
    #[inline(always)]
    unsafe fn store_partial2_complex(ptr: *mut Complex<Self::ScalarType>, data: Self::HalfVector) {
        _mm_storeu_ps(ptr as *mut f32, data);
    }
    #[inline(always)]
    unsafe fn store_partial3_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        Self::store_partial2_complex(ptr, data.lo());
        Self::store_partial1_complex(ptr.add(2), data.hi());
    }
}



impl AvxVector for __m128 {
    type ScalarType = f32;
    const SCALAR_PER_VECTOR : usize = 4;
    const COMPLEX_PER_VECTOR : usize = 2;

    #[inline(always)]
    unsafe fn zero() -> Self {
        _mm_setzero_ps()
    }
    #[inline(always)]
    unsafe fn half_root2() -> Self {
        // note: we're computing a square root here, but checking the assembly says the compiler is smart enough to turn this into a constant
        _mm_broadcast_ss(&0.5f32.sqrt())
    }

    #[inline(always)]
    unsafe fn xor(left: Self, right: Self) -> Self {
        _mm_xor_ps(left, right)
    }
    #[inline(always)]
    unsafe fn add(left: Self, right: Self) -> Self {
        _mm_add_ps(left, right)
    }
    #[inline(always)]
    unsafe fn sub(left: Self, right: Self) -> Self {
        _mm_sub_ps(left, right)
    }
    #[inline(always)]
    unsafe fn mul(left: Self, right: Self) -> Self {
        _mm_mul_ps(left, right)
    }
    #[inline(always)]
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fmadd_ps(left, right, add)
    }
    #[inline(always)]
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fnmadd_ps(left, right, add)
    }
    #[inline(always)]
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self {
        _mm_fmaddsub_ps(left, right, add)
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm_loadu_ps(ptr as *const Self::ScalarType)
    }
    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm_storeu_ps(ptr as *mut Self::ScalarType, data)
    }

    #[inline(always)]
    unsafe fn reverse_complex_elements(self) -> Self {
        // swap the elements in-lane
        _mm_permute_ps(self, 0x4E)
    }
    #[inline(always)]
    unsafe fn swap_complex_components(self) -> Self {
        _mm_permute_ps(self, 0xB1)
    }
    #[inline(always)]
    unsafe fn duplicate_complex_components(self) -> (Self, Self) {
        (_mm_moveldup_ps(self), _mm_movehdup_ps(self))
    }
    #[inline(always)]
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm_set_ps(value.im, value.re, value.im, value.re)
    }

    #[inline(always)]
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self> {
        if !inverse {
            Rotation90(Self::broadcast_complex_elements(Complex::new(0.0, -0.0)))
        } else {
            Rotation90(Self::broadcast_complex_elements(Complex::new(-0.0, 0.0)))
        }
    }
    #[inline(always)]
    unsafe fn make_mixedradix_twiddle_chunk(x: usize, y: usize, len: usize, inverse: bool) -> Self {
        let mut twiddle_chunk = [Complex::zero(); Self::COMPLEX_PER_VECTOR];
        for i in 0..Self::COMPLEX_PER_VECTOR {
            twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x + i), len, inverse);
        }

        twiddle_chunk.load_complex_f32_lo(0)
    }
    #[inline(always)]
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self {
        Self::broadcast_complex_elements(Self::ScalarType::generate_twiddle_factor(index, len, inverse))
    }
}
impl AvxVector128 for __m128 {
    type FullVector = __m256;

    #[inline(always)]
    unsafe fn lo(input: Self::FullVector) -> Self {
        _mm256_castps256_ps128(input)
    }
    #[inline(always)]
    unsafe fn hi(input: Self::FullVector) -> Self {
        _mm256_extractf128_ps(input, 1)
    }
    #[inline(always)]
    unsafe fn merge(lo: Self, hi: Self) -> Self::FullVector {
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1)
    }
    #[inline(always)]
    unsafe fn lo_rotation(input: Rotation90<Self::FullVector>) -> Rotation90<Self> {
        input.lo()
    }
}


impl AvxVector for __m256d {
    type ScalarType = f64;
    const SCALAR_PER_VECTOR : usize = 4;
    const COMPLEX_PER_VECTOR : usize = 2;

    #[inline(always)]
    unsafe fn zero() -> Self {
        _mm256_setzero_pd()
    }
    #[inline(always)]
    unsafe fn half_root2() -> Self {
        // note: we're computing a square root here, but checking the assembly says the compiler is smart enough to turn this into a constant
        _mm256_broadcast_sd(&0.5f64.sqrt())
    }

    #[inline(always)]
    unsafe fn xor(left: Self, right: Self) -> Self {
        _mm256_xor_pd(left, right)
    }
    #[inline(always)]
    unsafe fn add(left: Self, right: Self) -> Self {
        _mm256_add_pd(left, right)
    }
    #[inline(always)]
    unsafe fn sub(left: Self, right: Self) -> Self {
        _mm256_sub_pd(left, right)
    }
    #[inline(always)]
    unsafe fn mul(left: Self, right: Self) -> Self {
        _mm256_mul_pd(left, right)
    }
    #[inline(always)]
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmadd_pd(left, right, add)
    }
    #[inline(always)]
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fnmadd_pd(left, right, add)
    }
    #[inline(always)]
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmaddsub_pd(left, right, add)
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm256_loadu_pd(ptr as *const Self::ScalarType)
    }
    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm256_storeu_pd(ptr as *mut Self::ScalarType, data)
    }

    #[inline(always)]
    unsafe fn reverse_complex_elements(self) -> Self {
        _mm256_permute2f128_pd(self, self, 0x01)
    }
    #[inline(always)]
    unsafe fn swap_complex_components(self) -> Self {
        _mm256_permute_pd(self, 0x05)
    }
    #[inline(always)]
    unsafe fn duplicate_complex_components(self) -> (Self, Self) {
        (_mm256_movedup_pd(self), _mm256_permute_pd(self, 0x0F))
    }
    #[inline(always)]
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm256_set_pd(value.im, value.re, value.im, value.re)
    }

    #[inline(always)]
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self> {
        if !inverse {
            Rotation90(Self::broadcast_complex_elements(Complex::new(0.0, -0.0)))
        } else {
            Rotation90(Self::broadcast_complex_elements(Complex::new(-0.0, 0.0)))
        }
    }
    #[inline(always)]
    unsafe fn make_mixedradix_twiddle_chunk(x: usize, y: usize, len: usize, inverse: bool) -> Self {
        let mut twiddle_chunk = [Complex::zero(); Self::COMPLEX_PER_VECTOR];
        for i in 0..Self::COMPLEX_PER_VECTOR {
            twiddle_chunk[i] = f64::generate_twiddle_factor(y*(x + i), len, inverse);
        }

        twiddle_chunk.load_complex_f64(0)
    }
    #[inline(always)]
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self {
        Self::broadcast_complex_elements(Self::ScalarType::generate_twiddle_factor(index, len, inverse))
    }
}
impl AvxVector256 for __m256d {
    type HalfVector = __m128d;

    #[inline(always)]
    unsafe fn lo(self) -> Self::HalfVector {
        _mm256_castpd256_pd128(self)
    }
    #[inline(always)]
    unsafe fn hi(self) -> Self::HalfVector {
        _mm256_extractf128_pd(self, 1)
    }
    #[inline(always)]
    unsafe fn merge(lo: Self::HalfVector, hi: Self::HalfVector) -> Self {
        _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1)
    }
    
    #[inline(always)]
    unsafe fn load_partial1_complex(ptr: *const Complex<Self::ScalarType>) -> Self::HalfVector {
        _mm_loadu_pd(ptr as *const f64)
    }
    #[inline(always)]
    unsafe fn load_partial2_complex(_ptr: *const Complex<Self::ScalarType>) -> Self::HalfVector {
        unimplemented!("Impossible to do a partial load of 2 complex f64's")
    }
    #[inline(always)]
    unsafe fn load_partial3_complex(_ptr: *const Complex<Self::ScalarType>) -> Self {
        unimplemented!("Impossible to do a partial load of 3 complex f64's")
    }
    #[inline(always)]
    unsafe fn store_partial1_complex(ptr: *mut Complex<Self::ScalarType>, data: Self::HalfVector) {
        _mm_storeu_pd(ptr as *mut f64, data);
    }
    #[inline(always)]
    unsafe fn store_partial2_complex(_ptr: *mut Complex<Self::ScalarType>, _data: Self::HalfVector) {
        unimplemented!("Impossible to do a partial store of 2 complex f64's")
    }
    #[inline(always)]
    unsafe fn store_partial3_complex(_ptr: *mut Complex<Self::ScalarType>, _data: Self) {
        unimplemented!("Impossible to do a partial store of 3 complex f64's")
    }
}



impl AvxVector for __m128d {
    type ScalarType = f64;
    const SCALAR_PER_VECTOR : usize = 2;
    const COMPLEX_PER_VECTOR : usize = 1;

    #[inline(always)]
    unsafe fn zero() -> Self {
        _mm_setzero_pd()
    }
    #[inline(always)]
    unsafe fn half_root2() -> Self {
        // note: we're computing a square root here, but checking the assembly says the compiler is smart enough to turn this into a constant
        _mm_load1_pd(&0.5f64.sqrt())
    }

    #[inline(always)]
    unsafe fn xor(left: Self, right: Self) -> Self {
        _mm_xor_pd(left, right)
    }
    #[inline(always)]
    unsafe fn add(left: Self, right: Self) -> Self {
        _mm_add_pd(left, right)
    }
    #[inline(always)]
    unsafe fn sub(left: Self, right: Self) -> Self {
        _mm_sub_pd(left, right)
    }
    #[inline(always)]
    unsafe fn mul(left: Self, right: Self) -> Self {
        _mm_mul_pd(left, right)
    }
    #[inline(always)]
    unsafe fn fmadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fmadd_pd(left, right, add)
    }
    #[inline(always)]
    unsafe fn fnmadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fnmadd_pd(left, right, add)
    }
    #[inline(always)]
    unsafe fn fmaddsub(left: Self, right: Self, add: Self) -> Self {
        _mm_fmaddsub_pd(left, right, add)
    }

    #[inline(always)]
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self {
        _mm_loadu_pd(ptr as *const Self::ScalarType)
    }
    #[inline(always)]
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self) {
        _mm_storeu_pd(ptr as *mut Self::ScalarType, data)
    }
    
    #[inline(always)]
    unsafe fn reverse_complex_elements(self) -> Self {
        // nothing to reverse
        self
    }
    #[inline(always)]
    unsafe fn swap_complex_components(self) -> Self {
        _mm_permute_pd(self, 0x05)
    }
    #[inline(always)]
    unsafe fn duplicate_complex_components(self) -> (Self, Self) {
        (_mm_movedup_pd(self), _mm_permute_pd(self, 0x0F))
    }
    #[inline(always)]
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm_set_pd(value.im, value.re)
    }

    #[inline(always)]
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self> {
        if !inverse {
            Rotation90(Self::broadcast_complex_elements(Complex::new(0.0, -0.0)))
        } else {
            Rotation90(Self::broadcast_complex_elements(Complex::new(-0.0, 0.0)))
        }
    }
    #[inline(always)]
    unsafe fn make_mixedradix_twiddle_chunk(x: usize, y: usize, len: usize, inverse: bool) -> Self {
        let mut twiddle_chunk = [Complex::zero(); Self::COMPLEX_PER_VECTOR];
        for i in 0..Self::COMPLEX_PER_VECTOR {
            twiddle_chunk[i] = f64::generate_twiddle_factor(y*(x + i), len, inverse);
        }

        twiddle_chunk.load_complex_f64_lo(0)
    }
    #[inline(always)]
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self {
        Self::broadcast_complex_elements(Self::ScalarType::generate_twiddle_factor(index, len, inverse))
    }
}
impl AvxVector128 for __m128d {
    type FullVector = __m256d;

    #[inline(always)]
    unsafe fn lo(input: Self::FullVector) -> Self {
        _mm256_castpd256_pd128(input)
    }
    #[inline(always)]
    unsafe fn hi(input: Self::FullVector) -> Self {
        _mm256_extractf128_pd(input, 1)
    }
    #[inline(always)]
    unsafe fn merge(lo: Self, hi: Self) -> Self::FullVector {
        _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1)
    }
    #[inline(always)]
    unsafe fn lo_rotation(input: Rotation90<Self::FullVector>) -> Rotation90<Self> {
        input.lo()
    }
}

pub trait AvxArray<V> {
    unsafe fn load_complex(&self, index: usize) -> V;
}
pub trait AvxArrayMut<V> {
    unsafe fn store_complex(&mut self, data: V, index: usize);
}

impl<V: AvxVector> AvxArray<V> for [Complex<V::ScalarType>] {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + V::COMPLEX_PER_VECTOR);
        V::load_complex(self.get_unchecked(index) as *const Complex<_>)
    }
}
impl<V: AvxVector> AvxArray<V> for RawSlice<Complex<V::ScalarType>> {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + V::COMPLEX_PER_VECTOR);
        V::load_complex(self.as_ptr().add(index))
    }
}

impl<V: AvxVector> AvxArrayMut<V> for [Complex<V::ScalarType>] {
    #[inline(always)]
    unsafe fn store_complex(&mut self, data: V, index: usize) {
        debug_assert!(self.len() >= index + V::COMPLEX_PER_VECTOR);
        V::store_complex(self.get_unchecked_mut(index) as *mut Complex<_>, data);
    }
}
impl<V: AvxVector> AvxArrayMut<V> for RawSliceMut<Complex<V::ScalarType>> {
    #[inline(always)]
    unsafe fn store_complex(&mut self, data: V, index: usize) {
        debug_assert!(self.len() >= index + V::COMPLEX_PER_VECTOR);
        V::store_complex(self.as_mut_ptr().add(index), data);
    }
}

pub trait AvxArray256<V: AvxVector256> {
    unsafe fn load_partial1_complex(&self, index: usize) -> V::HalfVector;
    unsafe fn load_partial2_complex(&self, index: usize) -> V::HalfVector;
    unsafe fn load_partial3_complex(&self, index: usize) -> V;
}
pub trait AvxArray256Mut<V: AvxVector256> {
    unsafe fn store_partial1_complex(&mut self, data: V::HalfVector, index: usize);
    unsafe fn store_partial2_complex(&mut self, data: V::HalfVector, index: usize);
    unsafe fn store_partial3_complex(&mut self, data: V, index: usize);
}

impl<V: AvxVector256> AvxArray256<V> for [Complex<V::ScalarType>] {
    #[inline(always)]
    unsafe fn load_partial1_complex(&self, index: usize) -> V::HalfVector {
        debug_assert!(self.len() >= index + 1);
        V::load_partial1_complex(self.get_unchecked(index) as *const Complex<_>)
    }
    #[inline(always)]
    unsafe fn load_partial2_complex(&self, index: usize) -> V::HalfVector {
        debug_assert!(self.len() >= index + 2);
        V::load_partial2_complex(self.get_unchecked(index) as *const Complex<_>)
    }
    #[inline(always)]
    unsafe fn load_partial3_complex(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + 3);
        V::load_partial3_complex(self.get_unchecked(index) as *const Complex<_>)
    }
}
impl<V: AvxVector256> AvxArray256<V> for RawSlice<Complex<V::ScalarType>> {
    #[inline(always)]
    unsafe fn load_partial1_complex(&self, index: usize) -> V::HalfVector {
        debug_assert!(self.len() >= index + 1);
        V::load_partial1_complex(self.as_ptr().add(index) as *const Complex<_>)
    }
    #[inline(always)]
    unsafe fn load_partial2_complex(&self, index: usize) -> V::HalfVector {
        debug_assert!(self.len() >= index + 2);
        V::load_partial2_complex(self.as_ptr().add(index) as *const Complex<_>)
    }
    #[inline(always)]
    unsafe fn load_partial3_complex(&self, index: usize) -> V {
        debug_assert!(self.len() >= index + 3);
        V::load_partial3_complex(self.as_ptr().add(index) as *const Complex<_>)
    }
}

impl<V: AvxVector256> AvxArray256Mut<V> for [Complex<V::ScalarType>] {
    #[inline(always)]
    unsafe fn store_partial1_complex(&mut self, data: V::HalfVector, index: usize) {
        debug_assert!(self.len() >= index + 1);
        V::store_partial1_complex(self.get_unchecked_mut(index) as *mut Complex<_>, data)
    }
    #[inline(always)]
    unsafe fn store_partial2_complex(&mut self, data: V::HalfVector, index: usize) {
        debug_assert!(self.len() >= index + 2);
        V::store_partial2_complex(self.get_unchecked_mut(index) as *mut Complex<_>, data)
    }
    #[inline(always)]
    unsafe fn store_partial3_complex(&mut self, data: V, index: usize){
        debug_assert!(self.len() >= index + 3);
        V::store_partial3_complex(self.get_unchecked_mut(index) as *mut Complex<_>, data)
    }
}
impl<V: AvxVector256> AvxArray256Mut<V> for RawSliceMut<Complex<V::ScalarType>> {
    #[inline(always)]
    unsafe fn store_partial1_complex(&mut self, data: V::HalfVector, index: usize) {
        debug_assert!(self.len() >= index + 1);
        V::store_partial1_complex(self.as_mut_ptr().add(index) as *mut Complex<_>, data)
    }
    #[inline(always)]
    unsafe fn store_partial2_complex(&mut self, data: V::HalfVector, index: usize) {
        debug_assert!(self.len() >= index + 2);
        V::store_partial2_complex(self.as_mut_ptr().add(index) as *mut Complex<_>, data)
    }
    #[inline(always)]
    unsafe fn store_partial3_complex(&mut self, data: V, index: usize) {
        debug_assert!(self.len() >= index + 3);
        V::store_partial3_complex(self.as_mut_ptr().add(index) as *mut Complex<_>, data)
    }
}