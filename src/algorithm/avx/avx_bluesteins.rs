use std::sync::Arc;

use num_integer::div_ceil;
use num_complex::Complex;
use num_traits::Zero;

use crate::common::FFTnum;
use crate::{Length, IsInverse, Fft};

use super::avx_vector::{AvxVector, AvxVector128, AvxVector256, AvxArray, AvxArrayMut};
use super::CommonSimdData;

/// Implementation of Bluestein's Algorithm
///
/// This algorithm computes an arbitrary-sized FFT in O(nlogn) time. It does this by converting this size n FFT into a
/// size M where M >= 2N - 1. M is usually a power of two, although that isn't a requirement.
///
/// It requires a large scratch space, so it's probably inconvenient to use as an inner FFT to other algorithms.
///
/// ~~~
/// // Computes a forward FFT of size 1201 (prime number), using Bluestein's Algorithm
/// use rustfft::algorithm::RadersAlgorithm;
/// use rustfft::{FFTplanner, Fft};
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
/// let fft = RadersAlgorithm::new(1201, inner_fft);
/// fft.process(&mut input, &mut output);
/// ~~~
///
/// Bluestein's Algorithm is relatively expensive compared to other FFT algorithms. Benchmarking shows that it is up to
/// an order of magnitude slower than similar composite sizes. In the example size above of 1201, benchmarking shows
/// that it takes 2.5x more time to compute than a FFT of size 1200.

pub struct BluesteinsAvx<T: FFTnum> {
    inner_fft_multiplier: Box<[T::AvxType]>,
    common_data: CommonSimdData<T, T::AvxType>,
}
boilerplate_fft_commondata!(BluesteinsAvx);

impl<T: FFTnum> BluesteinsAvx<T> {
    fn compute_bluesteins_twiddle(index: usize, len: usize, inverse: bool) -> Complex<T> {
        let index_float = index as f64;
        let index_squared = index_float * index_float;

        T::generate_twiddle_factor_floatindex(index_squared, len*2, !inverse)
    }

    /// Pairwise multiply the complex numbers in `left` with the complex numbers in `right`.
    /// This is exactly the same as `mul_complex` in `AvxVector`, but this implementation also conjugates the `left` input before multiplying
    #[inline(always)]
    unsafe fn mul_complex_conjugated<V: AvxVector>(left: V, right: V) -> V {
        // Extract the real and imaginary components from left into 2 separate registers
        let (left_real, left_imag) = V::duplicate_complex_components(left);

        // create a shuffled version of right where the imaginary values are swapped with the reals
        let right_shuffled = V::swap_complex_components(right);

        // multiply our duplicated imaginary left vector by our shuffled right vector. that will give us the right side of the traditional complex multiplication formula
        let output_right = V::mul(left_imag, right_shuffled);

        // use a FMA instruction to multiply together left side of the complex multiplication formula, then alternatingly add and subtract the left side from the right
        // By using subadd instead of addsub, we can conjugate the left side for free.
        V::fmsubadd(left_real, right, output_right)
    }


    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(len: usize, inner_fft: Arc<dyn Fft<T>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(len, inner_fft) })
        } else {
            Err(())
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(len: usize, inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();
        assert!(len * 2 - 1 <= inner_fft_len, "Bluestein's algorithm requires inner_fft.len() >= self.len() * 2 - 1. Expected >= {}, got {}", len * 2 - 1, inner_fft_len);
        assert_eq!(inner_fft_len % T::AvxType::COMPLEX_PER_VECTOR, 0, "BluesteinsAvx requires its inner_fft.len() to be a multiple of {} (IE the number of complex numbers in a single vector) inner_fft.len() = {}", T::AvxType::COMPLEX_PER_VECTOR, inner_fft_len);

        // when computing FFTs, we're going to run our inner multiply pairwise by some precomputed data, then run an inverse inner FFT. We need to precompute that inner data here
        let inner_len_float = T::from_usize(inner_fft_len).unwrap();
        let inverse = inner_fft.is_inverse();

        // Compute twiddle factors that we'll run our inner FFT on
        let mut inner_fft_input = vec![Complex::zero(); inner_fft_len];
        for i in 0..len {
            inner_fft_input[i] = Self::compute_bluesteins_twiddle(i, len, inverse) / inner_len_float;
        }
        for i in 1..len {
            inner_fft_input[inner_fft_len - i] = inner_fft_input[i];
        }

        //Compute the inner fft
        let mut inner_fft_scratch = vec![Complex::zero(); inner_fft.get_inplace_scratch_len()];
        inner_fft.process_inplace_with_scratch(&mut inner_fft_input, &mut inner_fft_scratch);

        // When computing the FFT, we'll want this array to be pre-conjugated, so conjugate it now
        let conjugation_mask = AvxVector256::broadcast_complex_elements(Complex::new(T::zero(), -T::zero()));
        let inner_fft_multiplier = inner_fft_input.chunks_exact(T::AvxType::COMPLEX_PER_VECTOR).map(|chunk| {
            let chunk_vector = chunk.load_complex(0);
            AvxVector::xor(chunk_vector, conjugation_mask) // compute our conjugation by xoring our data with a precomputed mask
        }).collect::<Vec<_>>().into_boxed_slice();

        // also compute some more mundane twiddle factors to start and end with.
        let chunk_count = div_ceil(len, T::AvxType::COMPLEX_PER_VECTOR);
        let twiddles : Vec<_> = (0..chunk_count).map(|x| {
            let mut twiddle_chunk = [Complex::zero();4]; // can't give this a length of T::AvxType::COMPLEX_PER_VECTOR because arrays obnoxiously can't have generic lengths

            for i in 0..T::AvxType::COMPLEX_PER_VECTOR {
                twiddle_chunk[i] = Self::compute_bluesteins_twiddle(x*T::AvxType::COMPLEX_PER_VECTOR+i, len, !inverse);
            }
            twiddle_chunk.load_complex(0)
        }).collect();

        let required_scratch = inner_fft_input.len() + inner_fft_scratch.len();

        Self {
            inner_fft_multiplier,
            common_data: CommonSimdData {
                inner_fft,
                twiddles: twiddles.into_boxed_slice(),

                len,

                inplace_scratch_len: required_scratch,
                outofplace_scratch_len: required_scratch,

                inverse,
            }
        }
    }

    // Do the necessary setup for bluestein's algorithm: copy the data to the inner buffers, apply some twiddle factors, zero out the rest of the inner buffer
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn prepare_bluesteins(&self, input: &[Complex<T>], inner_fft_buffer: &mut [Complex<T>]) {
        let chunk_count = self.common_data.twiddles.len() - 1;
        let remainder = self.len() - chunk_count * T::AvxType::COMPLEX_PER_VECTOR;

        // Copy the buffer into our inner FFT input, applying twiddle factors as we go. the buffer will only fill part of the FFT input, so zero fill the rest
        for (i, twiddle) in self.common_data.twiddles[..chunk_count].iter().enumerate() {
            let index = i * T::AvxType::COMPLEX_PER_VECTOR;
            let input_vector = input.load_complex(index);
            let product_vector = AvxVector::mul_complex(input_vector, *twiddle);
            inner_fft_buffer.store_complex(product_vector, index);
        }

        // the buffer will almost certainly have a remainder. it's so likely, in fact, that we're just going to apply a remainder unconditionally
        // it uses a couple more instructions in the rare case when our FFT size is a multiple of 4, but wastes instructions when it's not
        {
            let remainder_twiddle = self.common_data.twiddles[chunk_count];

            let remainder_index = chunk_count * T::AvxType::COMPLEX_PER_VECTOR;
            let remainder_data = match remainder {
                1 => input.load_partial1_complex(remainder_index).zero_extend(),
                2 => if T::AvxType::COMPLEX_PER_VECTOR == 2 {
                    input.load_complex(remainder_index)
                } else {
                    input.load_partial2_complex(remainder_index).zero_extend()
                },
                3 => input.load_partial3_complex(remainder_index),
                4 => input.load_complex(remainder_index),
                _ => unreachable!(),
            };
            
            let twiddled_remainder = AvxVector::mul_complex(remainder_twiddle, remainder_data);
            inner_fft_buffer.store_complex(twiddled_remainder, remainder_index);
        }

        // zero fill the rest of the `inner` array
        let zerofill_start = chunk_count + 1;
        for i in zerofill_start..(inner_fft_buffer.len()/T::AvxType::COMPLEX_PER_VECTOR) {
            let index = i * T::AvxType::COMPLEX_PER_VECTOR;
            inner_fft_buffer.store_complex(AvxVector::zero(), index);
        }
    }

    // Do the necessary finalization for bluestein's algorithm: Conjugate the inner FFT buffer, apply some twiddle factors, zero out the rest of the inner buffer
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn finalize_bluesteins(&self, inner_fft_buffer: &[Complex<T>], output: &mut [Complex<T>]) {
        let chunk_count = self.common_data.twiddles.len() - 1;
        let remainder = self.len() - chunk_count * T::AvxType::COMPLEX_PER_VECTOR;

        // copy our data to the output, applying twiddle factors again as we go. Also conjugate inner_fft_buffer to complete the inverse FFT
        for (i, twiddle) in self.common_data.twiddles[..chunk_count].iter().enumerate() {
            let index = i * T::AvxType::COMPLEX_PER_VECTOR;
            let inner_vector = inner_fft_buffer.load_complex(index);
            let product_vector = Self::mul_complex_conjugated(inner_vector, *twiddle);
            output.store_complex(product_vector, index);
        }

        // again, unconditionally apply a remainder
        {
            let remainder_twiddle = self.common_data.twiddles[chunk_count];

            let remainder_index = chunk_count * T::AvxType::COMPLEX_PER_VECTOR;
            let inner_vector = inner_fft_buffer.load_complex(remainder_index);
            let product_vector = Self::mul_complex_conjugated(inner_vector, remainder_twiddle);

            match remainder {
                1 => output.store_partial1_complex(product_vector.lo(), remainder_index),
                2 => if T::AvxType::COMPLEX_PER_VECTOR == 2 {
                    output.store_complex(product_vector, remainder_index)
                } else {
                    output.store_partial2_complex(product_vector.lo(), remainder_index)
                },
                3 => output.store_partial3_complex(product_vector, remainder_index),
                4 => output.store_complex(product_vector, remainder_index),
                _ => unreachable!(),
            };
        }
    }

    // compute buffer[i] = buffer[i].conj() * multiplier[i] pairwise complex multiplication for each element.
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pairwise_complex_multiply_conjugated(buffer: &mut [Complex<T>], multiplier: &[T::AvxType]) {
        for (i, right) in multiplier.iter().enumerate() {
            let left = buffer.load_complex(i*T::AvxType::COMPLEX_PER_VECTOR);

            // Do a complex multiplication between `left` and `right`
            let product = Self::mul_complex_conjugated(left, *right);

            // Store the result
            buffer.store_complex(product, i*T::AvxType::COMPLEX_PER_VECTOR);
        }
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_multiplier.len()*T::AvxType::COMPLEX_PER_VECTOR);

        // do the necessary setup for bluestein's algorithm: copy the data to the inner buffers, apply some twiddle factors, zero out the rest of the inner buffer
        unsafe { self.prepare_bluesteins(buffer, inner_input) };

        // run our inner forward FFT
        self.common_data.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT. 
        // We can conjugate the result of multiplication by conjugating both inputs. We pre-conjugated the multiplier array, so we just need to conjugate inner_input
        unsafe { Self::pairwise_complex_multiply_conjugated(inner_input, &self.inner_fft_multiplier) };

        // inverse FFT. we're computing a forward but we're massaging it into an inverse by conjugating the inputs and outputs
        self.common_data.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // conjugate the inner FFT again to finalize the 
        unsafe { self.finalize_bluesteins(inner_input, buffer) };
    }

    fn perform_fft_out_of_place(&self, input: &[Complex<T>], output: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.inner_fft_multiplier.len()*T::AvxType::COMPLEX_PER_VECTOR);

        // do the necessary setup for bluestein's algorithm: copy the data to the inner buffers, apply some twiddle factors, zero out the rest of the inner buffer
        unsafe { self.prepare_bluesteins(input, inner_input) };

        // run our inner forward FFT
        self.common_data.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT. 
        // We can conjugate the result of multiplication by conjugating both inputs. We pre-conjugated the multiplier array, so we just need to conjugate inner_input
        unsafe { Self::pairwise_complex_multiply_conjugated(inner_input, &self.inner_fft_multiplier) };

        // inverse FFT. we're computing a forward but we're massaging it into an inverse by conjugating the inputs and outputs
        self.common_data.inner_fft.process_inplace_with_scratch(inner_input, inner_scratch);

        // conjugate the inner FFT again to finalize the 
        unsafe { self.finalize_bluesteins(inner_input, output) };
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use crate::test_utils::check_fft_algorithm;
    use crate::algorithm::DFT;

    #[test]
    fn test_bluesteins_avx_f32() {
        for len in 2..16 {
            // for this len, compute the range of inner FFT lengths we'll use.
            // Bluesteins AVX f32 requires a multiple of 4 for the inner FFT, so we need to go up to the next multiple of 4 from the minimum
            let minimum_inner : usize = len * 2 - 1;
            let remainder = minimum_inner % 4;

            // remainder will never be 0, because "n * 2 - 1" is guaranteed to be odd. so we can just subtract the remainder and add 4.
            let next_multiple_of_4 = minimum_inner - remainder + 4;
            let maximum_inner = minimum_inner.checked_next_power_of_two().unwrap() + 1;

            // start at the next multiple of 4, and increment by 4 unti lwe get to the next power of 2.
            for inner_len in (next_multiple_of_4..maximum_inner).step_by(4) {
                test_bluesteins_avx_with_length::<f32>(len, inner_len, false);
                test_bluesteins_avx_with_length::<f32>(len, inner_len, true);
            }
        }
    }

    #[test]
    fn test_bluesteins_avx_f64() {
        for len in 2..16 {
            // for this len, compute the range of inner FFT lengths we'll use.
            // Bluesteins AVX f64 requires a multiple of 2 for the inner FFT, so we need to go up to the next multiple of 2 from the minimum
            let minimum_inner : usize = len * 2 - 1;
            let remainder = minimum_inner % 2;

            let next_multiple_of_2 = minimum_inner + remainder;
            let maximum_inner = minimum_inner.checked_next_power_of_two().unwrap() + 1;

            // start at the next multiple of 2, and increment by 2 unti lwe get to the next power of 2.
            for inner_len in (next_multiple_of_2..maximum_inner).step_by(2) {
                test_bluesteins_avx_with_length::<f64>(len, inner_len, false);
                test_bluesteins_avx_with_length::<f64>(len, inner_len, true);
            }
        }
    }

    fn test_bluesteins_avx_with_length<T: FFTnum + num_traits::Float>(len: usize, inner_len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(inner_len, inverse));
        let fft : BluesteinsAvx<T> = BluesteinsAvx::new(len, inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_fft_algorithm(&fft, len, inverse);
    }
}
