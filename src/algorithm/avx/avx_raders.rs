use std::sync::Arc;
use std::arch::x86_64::*;
use std::convert::TryInto;

use num_integer::{Integer, div_ceil};
use num_complex::Complex;
use num_traits::Zero;
use strength_reduce::StrengthReducedUsize;
use primal_check::miller_rabin;

use crate::math_utils;
use crate::{Length, IsInverse, Fft};

use super::{AvxNum, avx_vector::{AvxVector, AvxVector256, AvxVector128, AvxArray, AvxArrayMut}};
use super::avx_vector;

// This struct wraps the necessary data to compute (a * b) % divisor, where b and divisor are determined at runtime but rarely change, and a changes on every call.
// It's written using AVX2 instructions and assumes the input a are 64-bit integers, and has a restriction that each a, b, and divisor must be 31-bit numbers or smaller.
#[derive(Clone)]
struct VectorizedMultiplyMod {
    b: __m256i,
    divisor: __m256i,
    intermediate: __m256i,
}

impl VectorizedMultiplyMod {
    #[target_feature(enable = "avx")]
    unsafe fn new(b: u32, divisor: u32) -> Self {
        assert!(divisor.leading_zeros() > 0, "divisor must be less than {}, got {}", 1 << 31, divisor);

        let b = b % divisor;
        let intermediate = ((b as i64) << 32) / divisor as i64;

        Self {
            b: _mm256_set1_epi64x(b as i64),
            divisor: _mm256_set1_epi64x(divisor as i64),
            intermediate: _mm256_set1_epi64x(intermediate),
        }
    }

    
    // Input: 4 unsigned 64-bit numbers, each less than 2^30
    // Output: (x * multiplier) % divisor for each x in input
    #[allow(unused)]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_rem(&self, a: __m256i) -> __m256i {
        // Pretty hacky, but we need to prove to the compiler that each entry of the divisor is a 32-bit number, by blending the divisor vector with zeroes in the upper bits of each number.
        // If we don't do this manually, the compiler will do it anyways, but only for _mm256_mul_epu32, not for the _mm256_sub_epi64 correction step at the end
        // That inconstistency results in sub-optimal codegen where the compiler inserts extra code to handle the case where divisor is 64-bit. It also results in using one more register than necessary.
        // Since we know that can't happen, we can placate the compiler by explicitly zeroing the upper 32 bit of each divisor and relying on the compiler to lift it out of the loop.
        let masked_divisor = _mm256_blend_epi32(self.divisor, _mm256_setzero_si256(), 0xAA);

        // compute the integer quotient of (a * b) / divisor. Our precomputed intermediate value lets us skip the expensive division via arithmetic strength reduction 
        let quotient = _mm256_srli_epi64(_mm256_mul_epu32(a, self.intermediate), 32);

        // Now we can compute numerator - quotient * divisor to get the remanider
        let numerator = _mm256_mul_epu32(a, self.b);
        let quotient_product = _mm256_mul_epu32(quotient, masked_divisor);

        // Standard remainder formula: remainder = numerator - quotient * divisor
        let remainder = _mm256_sub_epi64(numerator, quotient_product);

        // it's possible for the "remainder" to end up between divisor and 2 * divisor. so we'll subtract divisor from remainder, which will make some of the result negative
        // We can then use the subtracted result as the input to a blendv. Sadly avx doesn't have a blendv_epi32 or blendv_epi64, so we're gonna do blendv_pd instead
        // this works because blendv looks at the uppermost bit to decide which variable to use, and for a two's complement i64, the upper most bit is 1 when the number is negative!
        // So when the subtraction result is negative, the uppermost bit is 1, which means the blend will choose the second param, which is the unsubtracted remainder
        let casted_remainder = _mm256_castsi256_pd(remainder);
        let subtracted_remainder = _mm256_castsi256_pd(_mm256_sub_epi64(remainder, masked_divisor));
        let wrapped_remainder = _mm256_castpd_si256(_mm256_blendv_pd(subtracted_remainder, casted_remainder, subtracted_remainder));
        wrapped_remainder
    }
}

/// Implementation of Rader's Algorithm
///
/// This algorithm computes a prime-sized FFT in O(nlogn) time. It does this by converting this size n FFT into a
/// size (n - 1) which is guaranteed to be composite.
///
/// The worst case for this algorithm is when (n - 1) is 2 * prime, resulting in a
/// [Cunningham Chain](https://en.wikipedia.org/wiki/Cunningham_chain)
///
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Rader's Algorithm
/// use rustfft::algorithm::avx::RadersAvx2;
/// use rustfft::{Fft, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1201];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1201];
///
/// // plan a FFT of size n - 1 = 1200
/// let mut planner = FFTplanner::new(false);
/// let inner_fft = planner.plan_fft(1200);
///
/// let fft = RadersAvx2::new(inner_fft).unwrap();
/// fft.process(&mut input, &mut output);
/// ~~~
///
/// Rader's Algorithm is relatively expensive compared to other FFT algorithms. Benchmarking shows that it is up to
/// an order of magnitude slower than similar composite sizes. In the example size above of 1201, benchmarking shows
/// that it takes 2.5x more time to compute than a FFT of size 1200.
pub struct RadersAvx2<T: AvxNum> {
    input_index_multiplier: VectorizedMultiplyMod,
    input_index_init: __m256i,
    
    output_index_mapping: Box<[__m128i]>,
    twiddles: Box<[T::VectorType]>,

    inner_fft: Arc<dyn Fft<T>>,

    len: usize,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    inverse: bool,
}

impl<T: AvxNum> RadersAvx2<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the FFT
    /// Returns Ok(instance) if this machine has the required instruction sets ("avx", "fma", and "avx2"), Err() if some instruction sets are missing
    ///
    /// # Panics
    /// Panics if `inner_fft_len() + 1` is not a prime number.
    #[inline]
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_avx2 && has_fma {
            // Safety: new_with_avx2 requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();
        let len = inner_fft_len + 1;
        assert!(miller_rabin(len as u64), "For raders algorithm, inner_fft.len() + 1 must be prime. Expected prime number, got {} + 1 = {}", inner_fft_len, len);

        let inverse = inner_fft.is_inverse();
        let reduced_len = StrengthReducedUsize::new(len);

        // compute the primitive root and its inverse for this size
        let primitive_root = math_utils::primitive_root(len as u64).unwrap() as usize;

        // compute the multiplicative inverse of primative_root mod len and vice versa. 
        // i64::extended_gcd will compute both the inverse of left mod right, and the inverse of right mod left, but we're only goingto use one of them
        // the primtive root inverse might be negative, if so make it positive by wrapping
        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(len as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 { gcd_data.x } else { gcd_data.x + len as i64 } as usize;

        // precompute the coefficients to use inside the process method
        let unity_scale = T::from_f64(1f64 / inner_fft_len as f64).unwrap();
        let mut inner_fft_input = vec![Complex::zero(); inner_fft_len];
        let mut twiddle_input = 1;
        for input_cell in &mut inner_fft_input {
            let twiddle = T::generate_twiddle_factor(twiddle_input, len, inverse);
            *input_cell = twiddle * unity_scale;

            twiddle_input = (twiddle_input * primitive_root_inverse) % reduced_len;
        }

        let required_inner_scratch = inner_fft.get_inplace_scratch_len();
        let extra_inner_scratch = if required_inner_scratch <= inner_fft_len { 0 } else { required_inner_scratch };

        //precompute a FFT of our reordered twiddle factors
        let mut inner_fft_scratch = vec![Zero::zero(); required_inner_scratch];
        inner_fft.process_inplace_with_scratch(&mut inner_fft_input, &mut inner_fft_scratch);

        // When computing the FFT, we'll want this array to be pre-conjugated, so conjugate it. at the same time, convert it to vectors for convenient use later.
        let conjugation_mask = AvxVector256::broadcast_complex_elements(Complex::new(T::zero(), -T::zero()));
        let inner_fft_multiplier : Box<[_]> = inner_fft_input.chunks(T::VectorType::COMPLEX_PER_VECTOR).map(|chunk| {
            let chunk_vector = match chunk.len() {
                1 =>  chunk.load_partial1_complex(0).zero_extend(),
                2 => if chunk.len() == T::VectorType::COMPLEX_PER_VECTOR { chunk.load_complex(0) } else {chunk.load_partial2_complex(0).zero_extend()},
                3 => chunk.load_partial3_complex(0),
                4 => chunk.load_complex(0),
                _ => unreachable!()
            };
            AvxVector::xor(chunk_vector, conjugation_mask) // compute our conjugation by xoring our data with a precomputed mask
        }).collect();

        // Set up the data for our input index remapping computation
        const NUM_POWERS : usize = 5;
        let mut root_powers = [0; NUM_POWERS];
        let mut current_power = 1;
        for i in 0..NUM_POWERS {
            root_powers[i] = current_power;
            current_power = (current_power * primitive_root) % reduced_len;
        }

        let (input_index_multiplier, input_index_init) = if T::VectorType::COMPLEX_PER_VECTOR == 4 {
            (VectorizedMultiplyMod::new(root_powers[4] as u32, len as u32), _mm256_loadu_si256(root_powers.as_ptr().add(1) as *const __m256i))
        } else {
            let duplicated_powers = [root_powers[1],root_powers[1],root_powers[2],root_powers[2],];
            (VectorizedMultiplyMod::new(root_powers[2] as u32, len as u32), _mm256_loadu_si256(duplicated_powers.as_ptr() as *const __m256i))
        };

        // Set up our output index remapping. Ideally we could compute the output indexes on the fly, but the output reindexing requires scatter, which doesn't exist until avx-512
        // Instead, we can invert the scatter indexes to be gather indexes. But if there's an algorithmic way to compute this, I don't know what it is --
        // so we won't be able to compute it on the fly with some sort of VectorizedMultiplyMod thing. Instead, we're going to precompute the inverted mapping and gather from that mapping.
        // We want enough elements in our array to fill out an entire set of vectors so that we don't have to deal with any partial indexes etc.
        let mapping_size = 1 + div_ceil(len, T::VectorType::COMPLEX_PER_VECTOR) * T::VectorType::COMPLEX_PER_VECTOR;
        let mut output_mapping_inverse: Vec<i32> = vec![0; mapping_size];
        let mut output_index = 1;
        for i in 1..len {
            output_index = (output_index * primitive_root_inverse) % reduced_len;
            output_mapping_inverse[output_index] = i.try_into().unwrap();
        }

        // the actual vector of indexes depends on whether we're f32 or f64
        let output_index_mapping = if T::VectorType::COMPLEX_PER_VECTOR == 4 {
            (&output_mapping_inverse[1..]).chunks_exact(T::VectorType::COMPLEX_PER_VECTOR).map(|chunk| _mm_loadu_si128(chunk.as_ptr() as *const __m128i)).collect::<Box<[__m128i]>>()
        } else {
            (&output_mapping_inverse[1..]).chunks_exact(T::VectorType::COMPLEX_PER_VECTOR).map(|chunk| {
                let duplicated_indexes = [chunk[0], chunk[0], chunk[1], chunk[1]];
                _mm_loadu_si128(duplicated_indexes.as_ptr() as *const __m128i)
            }).collect::<Box<[__m128i]>>()
        };
        Self {
            input_index_multiplier,
            input_index_init,

            output_index_mapping,

            inner_fft: inner_fft,
            twiddles: inner_fft_multiplier,

            len,

            inplace_scratch_len: len + extra_inner_scratch,
            outofplace_scratch_len: extra_inner_scratch,
            inverse,
        }
    }

    // Do the necessary setup for rader's algorithm: Reorder the inputs into the output buffer, gather a sum of all inputs. Return the first input, and the aum of all inputs
    #[target_feature(enable = "avx2", enable = "avx", enable = "fma")]
    unsafe fn prepare_raders(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> (Complex<T>, Complex<T>) {
        let mut vector_sum = T::VectorType::zero();
        let mut indexes = self.input_index_init;
        let first_element = input[0];

        let index_multiplier = self.input_index_multiplier.clone();

        // loop over the output array and use AVX gathers to reorder data from the input
        let mut chunks_iter = (&mut output[1..]).chunks_exact_mut(T::VectorType::COMPLEX_PER_VECTOR);
        for chunk in chunks_iter.by_ref() {
            let gathered_elements = T::VectorType::gather_complex_avx2_index64(input.as_ptr(), indexes);

            // advance our indexes
            indexes = index_multiplier.mul_rem(indexes);

            // keep the sum of data updated
            vector_sum = AvxVector::add(vector_sum, gathered_elements);

            // Store this chunk
            chunk.store_complex(gathered_elements, 0);
        }

        // at this point, we either have 0 or 2 remaining elements to gather. because we know our length ends in 1 or 3. so when we subtract 1 for the inner FFT, that gives us 0 or 2
        let output_remainder = chunks_iter.into_remainder();
        if output_remainder.len() == 2 {
            let half_data = AvxVector128::gather64_complex_avx2(input.as_ptr(), _mm256_castsi256_si128(indexes));

            // add this last chunk to our sum
            vector_sum = AvxVector::add(vector_sum, AvxVector128::zero_extend(half_data));

            // store the remainder in the last chunk
            output_remainder.store_partial2_complex(half_data, 0);
        }

        (first_element, vector_sum.hadd_complex() + first_element)
    }

    // Do the necessary finalization for rader's algorithm: Reorder the inputs into the output buffer, conjugating the input as we go, and add the first input value to every output value
    #[target_feature(enable = "avx2", enable = "avx", enable = "fma")]
    unsafe fn finalize_raders(&self, input: &[Complex<T>], output: &mut [Complex<T>], first_input: Complex<T>) {
        let output_add = AvxVector256::broadcast_complex_elements(first_input);

        // We need to conjugate elements as a part of the finalization step, and sadly we can't roll it into any other instructions. So we'll do it via an xor.
        let conjugation_mask = AvxVector256::broadcast_complex_elements(Complex::new(T::zero(), -T::zero()));

        let mut chunks_iter = (&mut output[1..]).chunks_exact_mut(T::VectorType::COMPLEX_PER_VECTOR);
        for (i, chunk) in chunks_iter.by_ref().enumerate() {
            let index_chunk = *self.output_index_mapping.get_unchecked(i);
            let gathered_elements = T::VectorType::gather_complex_avx2_index32(input.as_ptr(), index_chunk);
            let conjugated_elements = AvxVector::xor(gathered_elements, conjugation_mask);

            // Add the first input value to each output value, then store
            let added_elements = AvxVector::add(output_add, conjugated_elements);
            chunk.store_complex(added_elements, 0);
        }

        // at this point, we either have 0 or 2 remaining elements to gather. because we know our length ends in 1 or 3. so when we subtract 1 for the inner FFT, that gives us 0 or 2
        let output_remainder = chunks_iter.into_remainder();
        if output_remainder.len() == 2 {
            let index_chunk = *self.output_index_mapping.get_unchecked(self.output_index_mapping.len() - 1);
            let half_data = AvxVector128::gather32_complex_avx2(input.as_ptr(), index_chunk);

            let conjugated_elements = AvxVector::xor(half_data, conjugation_mask.lo());

            // Add the first input value to each output value, then store
            let added_elements = AvxVector::add(output_add.lo(), conjugated_elements);
            output_remainder.store_partial2_complex(added_elements, 0);
        }
    }

    fn perform_fft_out_of_place(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (first_input, first_output) = unsafe { self.prepare_raders(input, output) };

        let inner_input = &mut input[1..];
        let inner_output = &mut output[1..];

        // perform the first of two inner FFTs
        let inner_scratch = if scratch.len() > 0 { &mut scratch[..] } else { &mut inner_input[..] };
        self.inner_fft.process_inplace_with_scratch(inner_output, inner_scratch);

        // multiply the inner result with our cached setup data
        // also conjugate every entry. this sets us up to do an inverse FFT
        // (because an inverse FFT is equivalent to a normal FFT where you conjugate both the inputs and outputs)
        unsafe { avx_vector::pairwise_complex_mul_conjugated(&mut inner_output[..], &mut inner_input[..], &self.twiddles) };

        // execute the second FFT
        let inner_scratch = if scratch.len() > 0 { scratch } else { &mut inner_output[..] };
        self.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // copy the final values into the output, reordering as we go
        output[0] = first_output;
        unsafe { self.finalize_raders(input, output, first_input); }
    }
    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (scratch, extra_scratch) = scratch.split_at_mut(self.len());
        let (first_input, first_output) = unsafe { self.prepare_raders(buffer, scratch) };

        let truncated_scratch = &mut scratch[1..];

        // perform the first of two inner FFTs
        let inner_scratch = if extra_scratch.len() > 0 { extra_scratch } else { &mut buffer[..] };
        self.inner_fft.process_inplace_with_scratch(truncated_scratch, inner_scratch);

        // multiply the inner result with our cached setup data
        // also conjugate every entry. this sets us up to do an inverse FFT
        // (because an inverse FFT is equivalent to a normal FFT where you conjugate both the inputs and outputs)
        unsafe { avx_vector::pairwise_complex_mul_assign_conjugated(truncated_scratch, &self.twiddles) };

        // execute the second FFT
        self.inner_fft.process_inplace_with_scratch(truncated_scratch, inner_scratch);

        // copy the final values into the output, reordering as we go
        buffer[0] = first_output;
        unsafe { self.finalize_raders(scratch, buffer, first_input); }
    }
}
boilerplate_avx_fft!(RadersAvx2, 
    |this: &RadersAvx2<_>| this.len,
    |this: &RadersAvx2<_>| this.inplace_scratch_len,
    |this: &RadersAvx2<_>| this.outofplace_scratch_len
);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use crate::test_utils::check_fft_algorithm;
    use crate::algorithm::DFT;

    #[test]
    fn test_raders_avx_f32() {
        for len in 3..100 {
            if miller_rabin(len as u64) {
                test_raders_with_length::<f32>(len, false);
                test_raders_with_length::<f32>(len, true);
            }
        }
    }

    #[test]
    fn test_raders_avx_f64() {
        for len in 3..100 {
            if miller_rabin(len as u64) {
                test_raders_with_length::<f64>(len, false);
                test_raders_with_length::<f64>(len, true);
            }
        }
    }

    fn test_raders_with_length<T: AvxNum + num_traits::Float>(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len - 1, inverse));
        let fft = RadersAvx2::new(inner_fft).unwrap();

        check_fft_algorithm::<T>(&fft, len, inverse);
    }
}