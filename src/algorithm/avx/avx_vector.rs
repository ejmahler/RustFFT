use std::arch::x86_64::*;
use std::fmt::Debug;

use num_complex::Complex;
use num_traits::Zero;

use crate::common::FFTnum;
use crate::array_utils::{RawSlice, RawSliceMut};

/// A SIMD vector of complex numbers, stored with the real values and imaginary values interleaved. 
/// Implemented for __m128, __m128d, __m256, __m256d, but these all require the AVX instruction set.
///
/// The goal of this trait is to reduce code duplication by letting code be generic over the vector type 
pub trait AvxVector : Copy + Debug + Send + Sync {
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
    unsafe fn fmsubadd(left: Self, right: Self, add: Self) -> Self;

    // More basic operations that end up being implemented in 1-2 intrinsics, but unlike the ones above, these have higher-level meaning than just arithmetic
    /// Swap each real number with its corresponding imaginary number
    unsafe fn swap_complex_components(self) -> Self;

    /// first return is the reals duplicated into the imaginaries, second return is the imaginaries duplicated into the reals
    unsafe fn duplicate_complex_components(self) -> (Self, Self);

    /// Reverse the order of complex numbers in the vector, so that the last is the first and the first is the last
    unsafe fn reverse_complex_elements(self) -> Self;

    /// Copies the even elements of rows[1] into the corresponding odd elements of rows[0] and returns the result.
    unsafe fn unpacklo_complex(rows: [Self; 2]) -> Self;
    /// Copies the odd elements of rows[0] into the corresponding even elements of rows[1] and returns the result.
    unsafe fn unpackhi_complex(rows: [Self; 2]) -> Self;

    #[inline(always)]
    unsafe fn unpack_complex(rows: [Self; 2]) -> [Self; 2] {
        [Self::unpacklo_complex(rows), Self::unpackhi_complex(rows)]
    }

    /// Fill a vector by computing a twiddle factor and repeating it across the whole vector
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self;

    /// create a Rotator90 instance to rotate complex numbers either 90 or 270 degrees, based on the value of `inverse`
    unsafe fn make_rotation90(inverse: bool) -> Rotation90<Self>;

    /// Generates a chunk of twiddle factors starting at (X,Y) and incrementing X `COMPLEX_PER_VECTOR` times.
    /// The result will be [twiddle(x*y, len), twiddle((x+1)*y, len), twiddle((x+2)*y, len), ...] for as many complex numbers fit in a vector
    unsafe fn make_mixedradix_twiddle_chunk(x: usize, y: usize, len: usize, inverse: bool) -> Self;

    /// Packed transposes. Used by mixed radix. These all take a NxC array, where C is COMPLEX_PER_VECTOR, and transpose it to a CxN array.
    /// But they also pack the result into as few vectors as possible, with the goal of writing the transposed data out contiguously.
    unsafe fn transpose2_packed(rows: [Self; 2]) -> [Self; 2];
    unsafe fn transpose3_packed(rows: [Self; 3]) -> [Self; 3];
    unsafe fn transpose4_packed(rows: [Self; 4]) -> [Self; 4];
    unsafe fn transpose5_packed(rows: [Self; 5]) -> [Self; 5];
    unsafe fn transpose6_packed(rows: [Self; 6]) -> [Self; 6];
    unsafe fn transpose7_packed(rows: [Self; 7]) -> [Self; 7];
    unsafe fn transpose8_packed(rows: [Self; 8]) -> [Self; 8];
    unsafe fn transpose9_packed(rows: [Self; 9]) -> [Self; 9];
    unsafe fn transpose12_packed(rows: [Self; 12]) -> [Self; 12];
    unsafe fn transpose16_packed(rows: [Self; 16]) -> [Self; 16];

    /// Pairwise multiply the complex numbers in `left` with the complex numbers in `right`.
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
    unsafe fn column_butterfly5(rows: [Self; 5], twiddles: [Self; 2]) -> [Self; 5] {
        // This algorithm is derived directly from the definition of the DFT of size 5
        // We'd theoretically have to do 16 complex multiplications for the DFT, but many of the twiddles we'd be multiplying by are conjugates of each other
        // By doing some algebra to expand the complex multiplications and factor out the real multiplications, we get this faster formula where we only do the equivalent of 4 multiplications

        // do some prep work before we can start applying twiddle factors
        let [sum1, diff4] = Self::column_butterfly2([rows[1], rows[4]]);
        let [sum2, diff3] = Self::column_butterfly2([rows[2], rows[3]]);
        
        let rotation = Self::make_rotation90(true);
        let rotated4 = Self::rotate90(diff4, rotation);
        let rotated3 = Self::rotate90(diff3, rotation);

        // to compute the first output, compute the sum of all elements. sum1 and sum2 already have the sum of 1+4 and 2+3 respectively, so if we add them, we'll get the sum of all 4
        let sum1234 = Self::add(sum1, sum2);
        let output0 = Self::add(rows[0], sum1234);

        // apply twiddle factors
        let (twiddles0_re, twiddles0_im) = Self::duplicate_complex_components(twiddles[0]);
        let (twiddles1_re, twiddles1_im) = Self::duplicate_complex_components(twiddles[1]);
        let twiddled1_mid = Self::fmadd(twiddles0_re, sum1, rows[0]);
        let twiddled2_mid = Self::fmadd(twiddles1_re, sum1, rows[0]);
        let twiddled3_mid = Self::mul(twiddles1_im, rotated4);
        let twiddled4_mid = Self::mul(twiddles0_im, rotated4);
        let twiddled1 = Self::fmadd(twiddles1_re, sum2, twiddled1_mid);
        let twiddled2 = Self::fmadd(twiddles0_re, sum2, twiddled2_mid);
        let twiddled3 = Self::fnmadd(twiddles0_im, rotated3, twiddled3_mid); // fnmadd instead of fmadd because we're actually re-using twiddle0 here. remember that this algorithm is all about factoring out conjugated multiplications -- this negation of the twiddle0 imaginaries is a reflection of one of those conugations
        let twiddled4 = Self::fmadd(twiddles1_im, rotated3, twiddled4_mid);

        // Post-processing to mix the twiddle factors between the rest of the output
        let [output1, output4] = Self::column_butterfly2([twiddled1, twiddled4]);
        let [output2, output3] = Self::column_butterfly2([twiddled2, twiddled3]);

        [output0, output1, output2, output3, output4]
    }

    #[inline(always)]
    unsafe fn column_butterfly7(rows: [Self; 7], twiddles: [Self; 3]) -> [Self; 7] {
        // This algorithm is derived directly from the definition of the DFT of size 7
        // We'd theoretically have to do 36 complex multiplications for the DFT, but many of the twiddles we'd be multiplying by are conjugates of each other
        // By doing some algebra to expand the complex multiplications and factor out the real multiplications, we get this faster formula where we only do the equivalent of 9 multiplications

        // do some prep work before we can start applying twiddle factors
        let [sum1, diff6] = Self::column_butterfly2([rows[1], rows[6]]);
        let [sum2, diff5] = Self::column_butterfly2([rows[2], rows[5]]);
        let [sum3, diff4] = Self::column_butterfly2([rows[3], rows[4]]);
        
        let rotation = Self::make_rotation90(true);
        let rotated4 = Self::rotate90(diff4, rotation);
        let rotated5 = Self::rotate90(diff5, rotation);
        let rotated6 = Self::rotate90(diff6, rotation);

        // to compute the first output, compute the sum of all elements. sum1, sum2, and sum3 already have the sum of 1+6 and 2+5 and 3+4 respectively, so if we add them, we'll get the sum of all 6
        let output0_left = Self::add(sum1, sum2);
        let output0_right = Self::add(sum3, rows[0]);
        let output0 = Self::add(output0_left, output0_right);

        // apply twiddle factors. This is probably pushing the limit of how much we should do with this technique.
        // We probably shouldn't do a size-11 FFT with this technique, for example, because this block of multiplies would grow quadratically
        let (twiddles0_re, twiddles0_im) = Self::duplicate_complex_components(twiddles[0]);
        let (twiddles1_re, twiddles1_im) = Self::duplicate_complex_components(twiddles[1]);
        let (twiddles2_re, twiddles2_im) = Self::duplicate_complex_components(twiddles[2]);

        let twiddled1_mid = Self::fmadd(twiddles0_re, sum1, rows[0]);
        let twiddled2_mid = Self::fmadd(twiddles1_re, sum1, rows[0]);
        let twiddled3_mid = Self::fmadd(twiddles2_re, sum1, rows[0]);
        let twiddled4_mid = Self::mul(twiddles2_im, rotated6);
        let twiddled5_mid = Self::mul(twiddles1_im, rotated6);
        let twiddled6_mid = Self::mul(twiddles0_im, rotated6);

        let twiddled1_mid2 = Self::fmadd(twiddles1_re, sum2, twiddled1_mid);
        let twiddled2_mid2 = Self::fmadd(twiddles2_re, sum2, twiddled2_mid);
        let twiddled3_mid2 = Self::fmadd(twiddles0_re, sum2, twiddled3_mid);
        let twiddled4_mid2 = Self::fnmadd(twiddles0_im, rotated5, twiddled4_mid); // fnmadd instead of fmadd because we're actually re-using twiddle0 here. remember that this algorithm is all about factoring out conjugated multiplications -- this negation of the twiddle0 imaginaries is a reflection of one of those conugations
        let twiddled5_mid2 = Self::fnmadd(twiddles2_im, rotated5, twiddled5_mid);
        let twiddled6_mid2 = Self::fmadd(twiddles1_im, rotated5, twiddled6_mid);

        let twiddled1 = Self::fmadd(twiddles2_re, sum3, twiddled1_mid2);
        let twiddled2 = Self::fmadd(twiddles0_re, sum3, twiddled2_mid2);
        let twiddled3 = Self::fmadd(twiddles1_re, sum3, twiddled3_mid2);
        let twiddled4 = Self::fmadd(twiddles1_im, rotated4, twiddled4_mid2);
        let twiddled5 = Self::fnmadd(twiddles0_im, rotated4, twiddled5_mid2);
        let twiddled6 = Self::fmadd(twiddles2_im, rotated4, twiddled6_mid2);

        // Post-processing to mix the twiddle factors between the rest of the output
        let [output1, output6] = Self::column_butterfly2([twiddled1, twiddled6]);
        let [output2, output5] = Self::column_butterfly2([twiddled2, twiddled5]);
        let [output3, output4] = Self::column_butterfly2([twiddled3, twiddled4]);

        [output0, output1, output2, output3, output4, output5, output6]
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
/// Implemented for __m256, __m256d
///
/// This trait implements things specific to 256-types, like splitting a 256 vector into 128 vectors
/// For compiler-placation reasons, all interactions/awareness the scalar type go here
pub trait AvxVector256 : AvxVector {
    type HalfVector : AvxVector128<FullVector=Self>;
    type ScalarType : FFTnum<AvxType=Self>;

    unsafe fn lo(self) -> Self::HalfVector;
    unsafe fn hi(self) -> Self::HalfVector;
    unsafe fn split(self) -> (Self::HalfVector, Self::HalfVector) {
        (self.lo(), self.hi())
    }
    unsafe fn merge(lo: Self::HalfVector, hi: Self::HalfVector) -> Self;

    /// Fill a vector by repeating the provided complex number as many times as possible
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self;

    // loads/stores of complex numbers
    unsafe fn load_complex(ptr: *const Complex<Self::ScalarType>) -> Self;
    unsafe fn store_complex(ptr: *mut Complex<Self::ScalarType>, data: Self);

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
/// Implemented for __m128, __m128d, but these are all oriented around AVX, so don't call methods on these from a SSE-only context
///
/// This trait implements things specific to 128-types, like merging 2 128 vectors into a 256 vector
pub trait AvxVector128 : AvxVector {
    type FullVector : AvxVector256<HalfVector=Self>;

    unsafe fn merge(lo: Self, hi: Self) -> Self::FullVector;
    unsafe fn zero_extend(self) -> Self::FullVector;

    unsafe fn lo(input: Self::FullVector) -> Self;
    unsafe fn hi(input: Self::FullVector) -> Self;
    unsafe fn split(input: Self::FullVector) -> (Self, Self) {
        (Self::lo(input), Self::hi(input))
    }
    unsafe fn lo_rotation(input: Rotation90<Self::FullVector>) -> Rotation90<Self>;

    /// Fill a vector by repeating the provided complex number as many times as possible
    unsafe fn broadcast_complex_elements(value: Complex<<<Self as AvxVector128>::FullVector as AvxVector256>::ScalarType>) -> Self;

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
    unsafe fn fmsubadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmsubadd_ps(left, right, add)
    }
    #[inline(always)]
    unsafe fn reverse_complex_elements(self) -> Self {
        // swap the elements in-lane
        let permuted = _mm256_permute_ps(self, 0x4E);
        // swap the lanes
        _mm256_permute2f128_ps(permuted, permuted, 0x01)
    }
    #[inline(always)]
    unsafe fn unpacklo_complex(rows: [Self; 2]) -> Self {
        let row0_double = _mm256_castps_pd(rows[0]);
        let row1_double = _mm256_castps_pd(rows[1]);
        let unpacked = _mm256_unpacklo_pd(row0_double, row1_double);
        _mm256_castpd_ps(unpacked)
    }
    #[inline(always)]
    unsafe fn unpackhi_complex(rows: [Self; 2]) -> Self {
        let row0_double = _mm256_castps_pd(rows[0]);
        let row1_double = _mm256_castps_pd(rows[1]);
        let unpacked = _mm256_unpackhi_pd(row0_double, row1_double);
        _mm256_castpd_ps(unpacked)
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

        twiddle_chunk.load_complex(0)
    }

    #[inline(always)]
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self {
        Self::broadcast_complex_elements(f32::generate_twiddle_factor(index, len, inverse))
    }

    #[inline(always)]
    unsafe fn transpose2_packed(rows: [Self; 2]) -> [Self;2] {
        let unpacked = Self::unpack_complex(rows);
        let output0 = _mm256_permute2f128_ps(unpacked[0], unpacked[1], 0x20);
        let output1 = _mm256_permute2f128_ps(unpacked[0], unpacked[1], 0x31);

        [output0, output1]
    }
    #[inline(always)]
    unsafe fn transpose3_packed(rows: [Self; 3]) -> [Self;3] {
        let unpacked0 = Self::unpacklo_complex([rows[0], rows[1]]);
        let unpacked2 = Self::unpackhi_complex([rows[1], rows[2]]);
        
        // output0 and output2 each need to swap some elements. thankfully we can blend those elements into the same intermediate value, and then do a permute 128 from there
        let blended = _mm256_blend_ps(rows[0], rows[2], 0x33);
        
        let output1 = _mm256_permute2f128_ps(unpacked0, unpacked2, 0x12);
        
        let output0 = _mm256_permute2f128_ps(unpacked0, blended, 0x20);
        let output2 = _mm256_permute2f128_ps(unpacked2, blended, 0x13);

        [output0, output1, output2]
    }
    #[inline(always)]
    unsafe fn transpose4_packed(rows: [Self; 4]) -> [Self;4] {
        let permute0 = _mm256_permute2f128_ps(rows[0], rows[2], 0x20);
        let permute1 = _mm256_permute2f128_ps(rows[1], rows[3], 0x20);
        let permute2 = _mm256_permute2f128_ps(rows[0], rows[2], 0x31);
        let permute3 = _mm256_permute2f128_ps(rows[1], rows[3], 0x31);

        let [unpacked0, unpacked1] = Self::unpack_complex([permute0, permute1]);
        let [unpacked2, unpacked3] = Self::unpack_complex([permute2, permute3]);

        [unpacked0, unpacked1, unpacked2, unpacked3]
    }
    #[inline(always)]
    unsafe fn transpose5_packed(rows: [Self; 5]) -> [Self; 5] {
        let unpacked0 = Self::unpacklo_complex([rows[0], rows[1]]);
        let unpacked1 = Self::unpackhi_complex([rows[1], rows[2]]);
        let unpacked2 = Self::unpacklo_complex([rows[2], rows[3]]);
        let unpacked3 = Self::unpackhi_complex([rows[3], rows[4]]);
        let blended04  = _mm256_blend_ps(rows[0], rows[4], 0x33);

        [
            _mm256_permute2f128_ps(unpacked0, unpacked2, 0x20),
            _mm256_permute2f128_ps(blended04, unpacked1, 0x20),
            _mm256_blend_ps(unpacked0, unpacked3, 0x0f),
            _mm256_permute2f128_ps(unpacked2, blended04, 0x31),
            _mm256_permute2f128_ps(unpacked1, unpacked3, 0x31),
        ]
    }
    #[inline(always)]
    unsafe fn transpose6_packed(rows: [Self; 6]) -> [Self;6] {
        let [unpacked0, unpacked1] = Self::unpack_complex([rows[0], rows[1]]);
        let [unpacked2, unpacked3] = Self::unpack_complex([rows[2], rows[3]]);
        let [unpacked4, unpacked5] = Self::unpack_complex([rows[4], rows[5]]);

        [
            _mm256_permute2f128_ps(unpacked0, unpacked2, 0x20),
            _mm256_permute2f128_ps(unpacked1, unpacked4, 0x02),
            _mm256_permute2f128_ps(unpacked3, unpacked5, 0x20),
            _mm256_permute2f128_ps(unpacked0, unpacked2, 0x31),
            _mm256_permute2f128_ps(unpacked1, unpacked4, 0x13),
            _mm256_permute2f128_ps(unpacked3, unpacked5, 0x31),
        ]
    }
    #[inline(always)]
    unsafe fn transpose7_packed(rows: [Self; 7]) -> [Self;7] {
        let unpacked0 = Self::unpacklo_complex([rows[0], rows[1]]);
        let unpacked1 = Self::unpackhi_complex([rows[1], rows[2]]);
        let unpacked2 = Self::unpacklo_complex([rows[2], rows[3]]);
        let unpacked3 = Self::unpackhi_complex([rows[3], rows[4]]);
        let unpacked4 = Self::unpacklo_complex([rows[4], rows[5]]);
        let unpacked5 = Self::unpackhi_complex([rows[5], rows[6]]);
        let blended06  = _mm256_blend_ps(rows[0], rows[6], 0x33);

        [
            _mm256_permute2f128_ps(unpacked0, unpacked2, 0x20),
            _mm256_permute2f128_ps(unpacked4, blended06, 0x20),
            _mm256_permute2f128_ps(unpacked1, unpacked3, 0x20),
            _mm256_blend_ps(unpacked0, unpacked5, 0x0f),
            _mm256_permute2f128_ps(unpacked2, unpacked4, 0x31),
            _mm256_permute2f128_ps(blended06, unpacked1, 0x31),
            _mm256_permute2f128_ps(unpacked3, unpacked5, 0x31),
        ]
    }
    #[inline(always)]
    unsafe fn transpose8_packed(rows: [Self; 8]) -> [Self;8] {
        let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3]];
        let chunk1 = [rows[4],  rows[5],  rows[6],  rows[7]];

        let output0 = Self::transpose4_packed(chunk0);
        let output1 = Self::transpose4_packed(chunk1);

        [output0[0], output1[0], output0[1], output1[1], output0[2], output1[2], output0[3], output1[3]]
    }
    #[inline(always)]
    unsafe fn transpose9_packed(rows: [Self; 9]) -> [Self;9] {
        let unpacked0 = Self::unpacklo_complex([rows[0], rows[1]]);
        let unpacked1 = Self::unpackhi_complex([rows[1], rows[2]]);
        let unpacked2 = Self::unpacklo_complex([rows[2], rows[3]]);
        let unpacked3 = Self::unpackhi_complex([rows[3], rows[4]]);
        let unpacked5 = Self::unpacklo_complex([rows[4], rows[5]]);
        let unpacked6 = Self::unpackhi_complex([rows[5], rows[6]]);
        let unpacked7 = Self::unpacklo_complex([rows[6], rows[7]]);
        let unpacked8 = Self::unpackhi_complex([rows[7], rows[8]]);
        let blended9  = _mm256_blend_ps(rows[0], rows[8], 0x33);

        [
            _mm256_permute2f128_ps(unpacked0, unpacked2, 0x20),
            _mm256_permute2f128_ps(unpacked5, unpacked7, 0x20),
            _mm256_permute2f128_ps(blended9,  unpacked1, 0x20),
            _mm256_permute2f128_ps(unpacked3, unpacked6, 0x20),
            _mm256_blend_ps(unpacked0, unpacked8, 0x0f),
            _mm256_permute2f128_ps(unpacked2, unpacked5, 0x31),
            _mm256_permute2f128_ps(unpacked7, blended9,  0x31),
            _mm256_permute2f128_ps(unpacked1, unpacked3, 0x31),
            _mm256_permute2f128_ps(unpacked6, unpacked8, 0x31),
        ]
    }
    #[inline(always)]
    unsafe fn transpose12_packed(rows: [Self; 12]) -> [Self;12] {
        let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3]];
        let chunk1 = [rows[4],  rows[5],  rows[6],  rows[7]];
        let chunk2 = [rows[8],  rows[9],  rows[10], rows[11]];

        let output0 = Self::transpose4_packed(chunk0);
        let output1 = Self::transpose4_packed(chunk1);
        let output2 = Self::transpose4_packed(chunk2);

        [output0[0], output1[0], output2[0], output0[1], output1[1], output2[1], output0[2], output1[2], output2[2], output0[3], output1[3], output2[3]]
    }
    #[inline(always)]
    unsafe fn transpose16_packed(rows: [Self; 16]) -> [Self;16] {
        let chunk0 = [rows[0], rows[1], rows[2],  rows[3],  rows[4],  rows[5],  rows[6],  rows[7]];
        let chunk1 = [rows[8], rows[9], rows[10], rows[11], rows[12], rows[13], rows[14], rows[15]];

        let output0 = Self::transpose8_packed(chunk0);
        let output1 = Self::transpose8_packed(chunk1);

        [
            output0[0], output0[1], output1[0], output1[1], output0[2], output0[3], output1[2], output1[3],
            output0[4], output0[5], output1[4], output1[5], output0[6], output0[7], output1[6], output1[7],
        ]
    }
}
impl AvxVector256 for __m256 {
    type ScalarType = f32;
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
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm256_set_ps(value.im, value.re, value.im, value.re, value.im, value.re, value.im, value.re)
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
    unsafe fn fmsubadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fmsubadd_ps(left, right, add)
    }

    #[inline(always)]
    unsafe fn reverse_complex_elements(self) -> Self {
        // swap the elements in-lane
        _mm_permute_ps(self, 0x4E)
    }

    #[inline(always)]
    unsafe fn unpacklo_complex(rows: [Self; 2]) -> Self {
        let row0_double = _mm_castps_pd(rows[0]);
        let row1_double = _mm_castps_pd(rows[1]);
        let unpacked = _mm_unpacklo_pd(row0_double, row1_double);
        _mm_castpd_ps(unpacked)
    }
    #[inline(always)]
    unsafe fn unpackhi_complex(rows: [Self; 2]) -> Self {
        let row0_double = _mm_castps_pd(rows[0]);
        let row1_double = _mm_castps_pd(rows[1]);
        let unpacked = _mm_unpackhi_pd(row0_double, row1_double);
        _mm_castpd_ps(unpacked)
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

        _mm_loadu_ps(twiddle_chunk.as_ptr() as *const f32)
    }
    #[inline(always)]
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self {
        Self::broadcast_complex_elements(f32::generate_twiddle_factor(index, len, inverse))
    }

    #[inline(always)]
    unsafe fn transpose2_packed(rows: [Self; 2]) -> [Self;2] {
        Self::unpack_complex(rows)
    }
    #[inline(always)]
    unsafe fn transpose3_packed(rows: [Self; 3]) -> [Self;3] {
        let unpacked0 = Self::unpacklo_complex([rows[0], rows[1]]);
        let blended = _mm_blend_ps(rows[0], rows[2], 0x33);
        let unpacked2 = Self::unpackhi_complex([rows[1], rows[2]]);

        [unpacked0, blended, unpacked2]
    }
    #[inline(always)]
    unsafe fn transpose4_packed(rows: [Self; 4]) -> [Self;4] {
        let [unpacked0, unpacked1] = Self::unpack_complex([rows[0], rows[1]]);
        let [unpacked2, unpacked3] = Self::unpack_complex([rows[2], rows[3]]);

        [unpacked0, unpacked2, unpacked1, unpacked3]
    }
    #[inline(always)]
    unsafe fn transpose5_packed(rows: [Self; 5]) -> [Self; 5] {
        [
            Self::unpacklo_complex([rows[0], rows[1]]),
            Self::unpacklo_complex([rows[2], rows[3]]),
            _mm_blend_ps(rows[0], rows[4], 0x33),
            Self::unpackhi_complex([rows[1], rows[2]]),
            Self::unpackhi_complex([rows[3], rows[4]]),
        ]
    }
    #[inline(always)]
    unsafe fn transpose6_packed(rows: [Self; 6]) -> [Self;6] {
        let [unpacked0, unpacked1] = Self::unpack_complex([rows[0], rows[1]]);
        let [unpacked2, unpacked3] = Self::unpack_complex([rows[2], rows[3]]);
        let [unpacked4, unpacked5] = Self::unpack_complex([rows[4], rows[5]]);

        [unpacked0, unpacked2, unpacked4, unpacked1, unpacked3, unpacked5]
    }
    #[inline(always)]
    unsafe fn transpose7_packed(rows: [Self; 7]) -> [Self;7] {
        [
            Self::unpacklo_complex([rows[0], rows[1]]),
            Self::unpacklo_complex([rows[2], rows[3]]),
            Self::unpacklo_complex([rows[4], rows[5]]),
            _mm_shuffle_ps(rows[6], rows[0], 0xE4),
            Self::unpackhi_complex([rows[1], rows[2]]),
            Self::unpackhi_complex([rows[3], rows[4]]),
            Self::unpackhi_complex([rows[5], rows[6]]),
        ]
    }
    #[inline(always)]
    unsafe fn transpose8_packed(rows: [Self; 8]) -> [Self;8] {
        let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3]];
        let chunk1 = [rows[4],  rows[5],  rows[6],  rows[7]];

        let output0 = Self::transpose4_packed(chunk0);
        let output1 = Self::transpose4_packed(chunk1);

        [output0[0], output0[1], output1[0], output1[1], output0[2], output0[3], output1[2], output1[3]]
    }
    #[inline(always)]
    unsafe fn transpose9_packed(rows: [Self; 9]) -> [Self;9] {
        [
            Self::unpacklo_complex([rows[0], rows[1]]),
            Self::unpacklo_complex([rows[2], rows[3]]),
            Self::unpacklo_complex([rows[4], rows[5]]),
            Self::unpacklo_complex([rows[6], rows[7]]),
            _mm_shuffle_ps(rows[8], rows[0], 0xE4),
            Self::unpackhi_complex([rows[1], rows[2]]),
            Self::unpackhi_complex([rows[3], rows[4]]),
            Self::unpackhi_complex([rows[5], rows[6]]),
            Self::unpackhi_complex([rows[7], rows[8]]),
        ]
    }
    #[inline(always)]
    unsafe fn transpose12_packed(rows: [Self; 12]) -> [Self;12] {
        let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3]];
        let chunk1 = [rows[4],  rows[5],  rows[6],  rows[7]];
        let chunk2 = [rows[8],  rows[9],  rows[10], rows[11]];

        let output0 = Self::transpose4_packed(chunk0);
        let output1 = Self::transpose4_packed(chunk1);
        let output2 = Self::transpose4_packed(chunk2);

        [output0[0], output0[1], output1[0], output1[1], output2[0], output2[1], output0[2], output0[3], output1[2], output1[3], output2[2], output2[3]]
    }
    #[inline(always)]
    unsafe fn transpose16_packed(rows: [Self; 16]) -> [Self;16] {
        let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3], rows[4],  rows[5],  rows[6],  rows[7]];
        let chunk1 = [rows[8],  rows[9],  rows[10], rows[11], rows[12], rows[13], rows[14], rows[15]];

        let output0 = Self::transpose8_packed(chunk0);
        let output1 = Self::transpose8_packed(chunk1);

        [
            output0[0], output0[1], output0[2], output0[3], output1[0], output1[1], output1[2], output1[3],
            output0[4], output0[5], output0[6], output0[7], output1[4], output1[5], output1[6], output1[7],
        ]
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
    unsafe fn zero_extend(self) -> Self::FullVector {
        _mm256_zextps128_ps256(self)
    }
    #[inline(always)]
    unsafe fn lo_rotation(input: Rotation90<Self::FullVector>) -> Rotation90<Self> {
        input.lo()
    }
    #[inline(always)]
    unsafe fn broadcast_complex_elements(value: Complex<f32>) -> Self {
        _mm_set_ps(value.im, value.re, value.im, value.re)
    }
}


impl AvxVector for __m256d {
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
    unsafe fn fmsubadd(left: Self, right: Self, add: Self) -> Self {
        _mm256_fmsubadd_pd(left, right, add)
    }

    #[inline(always)]
    unsafe fn reverse_complex_elements(self) -> Self {
        _mm256_permute2f128_pd(self, self, 0x01)
    }
    #[inline(always)]
    unsafe fn unpacklo_complex(rows: [Self; 2]) -> Self {
        _mm256_permute2f128_pd(rows[0], rows[1], 0x20)
    }
    #[inline(always)]
    unsafe fn unpackhi_complex(rows: [Self; 2]) -> Self {
        _mm256_permute2f128_pd(rows[0], rows[1], 0x31)
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

        twiddle_chunk.load_complex(0)
    }
    #[inline(always)]
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self {
        Self::broadcast_complex_elements(f64::generate_twiddle_factor(index, len, inverse))
    }

    #[inline(always)]
    unsafe fn transpose2_packed(rows: [Self; 2]) -> [Self;2] {
        Self::unpack_complex(rows)
    }
    #[inline(always)]
    unsafe fn transpose3_packed(rows: [Self; 3]) -> [Self;3] {
        let unpacked0 = Self::unpacklo_complex([rows[0], rows[1]]);
        let blended = _mm256_blend_pd(rows[0], rows[2], 0x33);
        let unpacked2 = Self::unpackhi_complex([rows[1], rows[2]]);

        [unpacked0, blended, unpacked2]
    }
    #[inline(always)]
    unsafe fn transpose4_packed(rows: [Self; 4]) -> [Self;4] {
        let [unpacked0, unpacked1] = Self::unpack_complex([rows[0], rows[1]]);
        let [unpacked2, unpacked3] = Self::unpack_complex([rows[2], rows[3]]);

        [unpacked0, unpacked2, unpacked1, unpacked3]
    }
    #[inline(always)]
    unsafe fn transpose5_packed(rows: [Self; 5]) -> [Self; 5] {
        [
            Self::unpacklo_complex([rows[0], rows[1]]),
            Self::unpacklo_complex([rows[2], rows[3]]),
            _mm256_blend_pd(rows[0], rows[4], 0x33),
            Self::unpackhi_complex([rows[1], rows[2]]),
            Self::unpackhi_complex([rows[3], rows[4]]),
        ]
    }
    #[inline(always)]
    unsafe fn transpose6_packed(rows: [Self; 6]) -> [Self;6] {
        let [unpacked0, unpacked1] = Self::unpack_complex([rows[0], rows[1]]);
        let [unpacked2, unpacked3] = Self::unpack_complex([rows[2], rows[3]]);
        let [unpacked4, unpacked5] = Self::unpack_complex([rows[4], rows[5]]);

        [unpacked0, unpacked2, unpacked4, unpacked1, unpacked3, unpacked5]
    }
    #[inline(always)]
    unsafe fn transpose7_packed(rows: [Self; 7]) -> [Self;7] {
        [
            Self::unpacklo_complex([rows[0], rows[1]]),
            Self::unpacklo_complex([rows[2], rows[3]]),
            Self::unpacklo_complex([rows[4], rows[5]]),
            _mm256_blend_pd(rows[0], rows[6], 0x33),
            Self::unpackhi_complex([rows[1], rows[2]]),
            Self::unpackhi_complex([rows[3], rows[4]]),
            Self::unpackhi_complex([rows[5], rows[6]]),
        ]
    }
    #[inline(always)]
    unsafe fn transpose8_packed(rows: [Self; 8]) -> [Self;8] {
        let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3]];
        let chunk1 = [rows[4],  rows[5],  rows[6],  rows[7]];

        let output0 = Self::transpose4_packed(chunk0);
        let output1 = Self::transpose4_packed(chunk1);

        [output0[0], output0[1], output1[0], output1[1], output0[2], output0[3], output1[2], output1[3]]
    }
    #[inline(always)]
    unsafe fn transpose9_packed(rows: [Self; 9]) -> [Self;9] {
        [
            _mm256_permute2f128_pd(rows[0], rows[1], 0x20),
            _mm256_permute2f128_pd(rows[2], rows[3], 0x20),
            _mm256_permute2f128_pd(rows[4], rows[5], 0x20),
            _mm256_permute2f128_pd(rows[6], rows[7], 0x20),
            _mm256_permute2f128_pd(rows[8], rows[0], 0x30),
            _mm256_permute2f128_pd(rows[1], rows[2], 0x31),
            _mm256_permute2f128_pd(rows[3], rows[4], 0x31),
            _mm256_permute2f128_pd(rows[5], rows[6], 0x31),
            _mm256_permute2f128_pd(rows[7], rows[8], 0x31),
        ]
    }
    #[inline(always)]
    unsafe fn transpose12_packed(rows: [Self; 12]) -> [Self;12] {
        let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3]];
        let chunk1 = [rows[4],  rows[5],  rows[6],  rows[7]];
        let chunk2 = [rows[8],  rows[9],  rows[10], rows[11]];

        let output0 = Self::transpose4_packed(chunk0);
        let output1 = Self::transpose4_packed(chunk1);
        let output2 = Self::transpose4_packed(chunk2);

        [output0[0], output0[1], output1[0], output1[1], output2[0], output2[1], output0[2], output0[3], output1[2], output1[3], output2[2], output2[3]]
    }
    #[inline(always)]
    unsafe fn transpose16_packed(rows: [Self; 16]) -> [Self;16] {
        let chunk0 = [rows[0],  rows[1],  rows[2],  rows[3], rows[4],  rows[5],  rows[6],  rows[7]];
        let chunk1 = [rows[8],  rows[9],  rows[10], rows[11], rows[12], rows[13], rows[14], rows[15]];

        let output0 = Self::transpose8_packed(chunk0);
        let output1 = Self::transpose8_packed(chunk1);

        [
            output0[0], output0[1], output0[2], output0[3], output1[0], output1[1], output1[2], output1[3],
            output0[4], output0[5], output0[6], output0[7], output1[4], output1[5], output1[6], output1[7],
        ]
    }
}
impl AvxVector256 for __m256d {
    type ScalarType = f64;
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
    unsafe fn broadcast_complex_elements(value: Complex<Self::ScalarType>) -> Self {
        _mm256_set_pd(value.im, value.re, value.im, value.re)
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
    unsafe fn fmsubadd(left: Self, right: Self, add: Self) -> Self {
        _mm_fmsubadd_pd(left, right, add)
    }
    
    #[inline(always)]
    unsafe fn reverse_complex_elements(self) -> Self {
        // nothing to reverse
        self
    }
    #[inline(always)]
    unsafe fn unpacklo_complex(_rows: [Self; 2]) -> Self {
        unimplemented!(); // this operation doesn't make sense with one element. TODO: I don't know if it would be more useful to error here or to just return the inputs unchanged. If returning the inputs is useful, do that.
    }
    #[inline(always)]
    unsafe fn unpackhi_complex(_rows: [Self; 2]) -> Self {
        unimplemented!(); // this operation doesn't make sense with one element. TODO: I don't know if it would be more useful to error here or to just return the inputs unchanged. If returning the inputs is useful, do that.
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

        _mm_loadu_pd(twiddle_chunk.as_ptr() as *const f64)
    }
    #[inline(always)]
    unsafe fn broadcast_twiddle(index: usize, len: usize, inverse: bool) -> Self {
        Self::broadcast_complex_elements(f64::generate_twiddle_factor(index, len, inverse))
    }

    #[inline(always)] unsafe fn transpose2_packed(rows: [Self;2]) -> [Self;2] { rows }
    #[inline(always)] unsafe fn transpose3_packed(rows: [Self;3]) -> [Self;3] { rows }
    #[inline(always)] unsafe fn transpose4_packed(rows: [Self;4]) -> [Self;4] { rows }
    #[inline(always)] unsafe fn transpose5_packed(rows: [Self;5]) -> [Self;5] { rows }
    #[inline(always)] unsafe fn transpose6_packed(rows: [Self;6]) -> [Self;6] { rows }
    #[inline(always)] unsafe fn transpose7_packed(rows: [Self;7]) -> [Self;7] { rows }
    #[inline(always)] unsafe fn transpose8_packed(rows: [Self;8]) -> [Self;8] { rows }
    #[inline(always)] unsafe fn transpose9_packed(rows: [Self;9]) -> [Self;9] { rows }
    #[inline(always)] unsafe fn transpose12_packed(rows: [Self;12]) -> [Self;12] { rows }
    #[inline(always)] unsafe fn transpose16_packed(rows: [Self;16]) -> [Self;16] { rows }
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
    unsafe fn zero_extend(self) -> Self::FullVector {
        _mm256_zextpd128_pd256(self)
    }
    #[inline(always)]
    unsafe fn lo_rotation(input: Rotation90<Self::FullVector>) -> Rotation90<Self> {
        input.lo()
    }
    #[inline(always)]
    unsafe fn broadcast_complex_elements(value: Complex<f64>) -> Self {
        _mm_set_pd(value.im, value.re)
    }
}

pub trait AvxArray<T: FFTnum> {
    unsafe fn load_complex(&self, index: usize) -> T::AvxType;
    unsafe fn load_partial1_complex(&self, index: usize) -> <T::AvxType as AvxVector256>::HalfVector;
    unsafe fn load_partial2_complex(&self, index: usize) -> <T::AvxType as AvxVector256>::HalfVector;
    unsafe fn load_partial3_complex(&self, index: usize) -> T::AvxType;
}
pub trait AvxArrayMut<T: FFTnum> {
    unsafe fn store_complex(&mut self, data: T::AvxType, index: usize);
    unsafe fn store_partial1_complex(&mut self, data: <T::AvxType as AvxVector256>::HalfVector, index: usize);
    unsafe fn store_partial2_complex(&mut self, data: <T::AvxType as AvxVector256>::HalfVector, index: usize);
    unsafe fn store_partial3_complex(&mut self, data: T::AvxType, index: usize);
}

impl<T: FFTnum> AvxArray<T> for [Complex<T>] {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> T::AvxType {
        debug_assert!(self.len() >= index + T::AvxType::COMPLEX_PER_VECTOR);
        T::AvxType::load_complex(self.as_ptr().add(index))
    }
    #[inline(always)]
    unsafe fn load_partial1_complex(&self, index: usize) -> <T::AvxType as AvxVector256>::HalfVector {
        debug_assert!(self.len() >= index + 1);
        T::AvxType::load_partial1_complex(self.as_ptr().add(index))
    }
    #[inline(always)]
    unsafe fn load_partial2_complex(&self, index: usize) -> <T::AvxType as AvxVector256>::HalfVector {
        debug_assert!(self.len() >= index + 2);
        T::AvxType::load_partial2_complex(self.as_ptr().add(index))
    }
    #[inline(always)]
    unsafe fn load_partial3_complex(&self, index: usize) -> T::AvxType {
        debug_assert!(self.len() >= index + 3);
        T::AvxType::load_partial3_complex(self.as_ptr().add(index))
    }
}
impl<T: FFTnum> AvxArray<T> for RawSlice<Complex<T>> {
    #[inline(always)]
    unsafe fn load_complex(&self, index: usize) -> T::AvxType {
        debug_assert!(self.len() >= index + T::AvxType::COMPLEX_PER_VECTOR);
        T::AvxType::load_complex(self.as_ptr().add(index))
    }
    #[inline(always)]
    unsafe fn load_partial1_complex(&self, index: usize) -> <T::AvxType as AvxVector256>::HalfVector {
        debug_assert!(self.len() >= index + 1);
        T::AvxType::load_partial1_complex(self.as_ptr().add(index))
    }
    #[inline(always)]
    unsafe fn load_partial2_complex(&self, index: usize) -> <T::AvxType as AvxVector256>::HalfVector {
        debug_assert!(self.len() >= index + 2);
        T::AvxType::load_partial2_complex(self.as_ptr().add(index))
    }
    #[inline(always)]
    unsafe fn load_partial3_complex(&self, index: usize) -> T::AvxType {
        debug_assert!(self.len() >= index + 3);
        T::AvxType::load_partial3_complex(self.as_ptr().add(index))
    }
}

impl<T: FFTnum> AvxArrayMut<T> for [Complex<T>] {
    #[inline(always)]
    unsafe fn store_complex(&mut self, data: T::AvxType, index: usize) {
        debug_assert!(self.len() >= index + T::AvxType::COMPLEX_PER_VECTOR);
        T::AvxType::store_complex(self.as_mut_ptr().add(index), data);
    }
    #[inline(always)]
    unsafe fn store_partial1_complex(&mut self, data: <T::AvxType as AvxVector256>::HalfVector, index: usize) {
        debug_assert!(self.len() >= index + 1);
        T::AvxType::store_partial1_complex(self.as_mut_ptr().add(index), data)
    }
    #[inline(always)]
    unsafe fn store_partial2_complex(&mut self, data: <T::AvxType as AvxVector256>::HalfVector, index: usize) {
        debug_assert!(self.len() >= index + 2);
        T::AvxType::store_partial2_complex(self.as_mut_ptr().add(index), data)
    }
    #[inline(always)]
    unsafe fn store_partial3_complex(&mut self, data: T::AvxType, index: usize){
        debug_assert!(self.len() >= index + 3);
        T::AvxType::store_partial3_complex(self.as_mut_ptr().add(index), data)
    }
}
impl<T: FFTnum> AvxArrayMut<T> for RawSliceMut<Complex<T>> {
    #[inline(always)]
    unsafe fn store_complex(&mut self, data: T::AvxType, index: usize) {
        debug_assert!(self.len() >= index + T::AvxType::COMPLEX_PER_VECTOR);
        T::AvxType::store_complex(self.as_mut_ptr().add(index), data);
    }
    #[inline(always)]
    unsafe fn store_partial1_complex(&mut self, data: <T::AvxType as AvxVector256>::HalfVector, index: usize) {
        debug_assert!(self.len() >= index + 1);
        T::AvxType::store_partial1_complex(self.as_mut_ptr().add(index), data)
    }
    #[inline(always)]
    unsafe fn store_partial2_complex(&mut self, data: <T::AvxType as AvxVector256>::HalfVector, index: usize) {
        debug_assert!(self.len() >= index + 2);
        T::AvxType::store_partial2_complex(self.as_mut_ptr().add(index), data)
    }
    #[inline(always)]
    unsafe fn store_partial3_complex(&mut self, data: T::AvxType, index: usize) {
        debug_assert!(self.len() >= index + 3);
        T::AvxType::store_partial3_complex(self.as_mut_ptr().add(index), data)
    }
}