use std::{any::TypeId, sync::Arc};

use num_complex::Complex;
use num_integer::div_ceil;

use crate::array_utils::{into_complex_mut, workaround_transmute_mut};
use crate::{Fft, FftDirection, FftNum, FftRealToComplex, Length};

use super::{
    avx_vector::{AvxArray, AvxArrayMut, AvxVector, AvxVector256},
    AvxNum,
};

/// Processes FFTs with real-only inputs. Restricted to even input sizes.
pub struct RealToComplexEvenAvx<A: AvxNum, T> {
    inner_fft: Arc<dyn Fft<T>>,
    twiddles: Box<[A::VectorType]>,

    len: usize,
    required_scratch: usize,
    direction: FftDirection,
}
impl<A: AvxNum, T: FftNum> RealToComplexEvenAvx<A, T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[allow(unused)]
    #[inline]
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Result<Self, ()> {
        // Internal sanity check: Make sure that A == T.
        // This struct has two generic parameters A and T, but they must always be the same, and are only kept separate to help work around the lack of specialization.
        // It would be cool if we could do this as a static_assert instead
        let id_a = TypeId::of::<A>();
        let id_t = TypeId::of::<T>();
        assert_eq!(id_a, id_t);

        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }

    /// Creates a FFT instance which will process forward FFTs with real-only inputs of size `inner_fft.len() * 2`
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();
        let inner_buffer_len = inner_fft_len + 1;
        let len = inner_fft_len * 2;
        let direction = inner_fft.fft_direction();

        // Compute our twiddle factors. We only need half as many twiddle factors as our FFT length,
        // and keep in mind that we're baking a multiply by half into the twiddles
        let first_twiddle = if (inner_buffer_len / 2) % 2 == 0 {
            0
        } else {
            1
        };

        let twiddle_count = if inner_fft_len % 2 == 0 {
            inner_fft_len / 2 - first_twiddle
        } else {
            inner_fft_len / 2 + 1 - first_twiddle
        };

        let full_twiddle_chunks = div_ceil(twiddle_count, A::VectorType::COMPLEX_PER_VECTOR);

        let twiddles: Box<[A::VectorType]> = (0..full_twiddle_chunks)
            .map(|i| first_twiddle + i * A::VectorType::COMPLEX_PER_VECTOR)
            .map(|twiddle_base_index| {
                let twiddle_chunk = A::VectorType::make_mixedradix_twiddle_chunk(
                    twiddle_base_index,
                    1,
                    len,
                    direction,
                );
                AvxVector::mul(twiddle_chunk, AvxVector::half())
            })
            .collect();

        Self {
            required_scratch: 2 * inner_fft.get_outofplace_scratch_len(),

            inner_fft,
            twiddles: twiddles,

            len,
            direction,
        }
    }

    #[inline(always)]
    unsafe fn postprocess_chunk<V: AvxVector>(val_fwd: V, val_rev: V, twiddle: V) -> (V, V) {
        let val_rev = AvxVector::reverse_complex_elements(val_rev);

        let sum = AvxVector::add(val_fwd, val_rev);
        let diff = AvxVector::sub(val_fwd, val_rev);

        let (twiddle_re, twiddle_im) = AvxVector::duplicate_complex_components(twiddle);
        let twiddle_re = twiddle_re.conj();

        // This is unusual - we want to swap the imaginaries in 'sum' with the imaginaries in 'diff'
        // 'sumdiff' will contain (sum.re, diff.im), and 'diffsum' will contain (diff.re, sum.im)
        let sumdiff_blended = AvxVector::blend_real_imaginary(sum, diff);
        let diffsum_blended = AvxVector::blend_real_imaginary(diff, sum);
        let diffsum_swapped = AvxVector::swap_complex_components(diffsum_blended);

        // Apply twiddle factors. Theoretically we'd have to load 2 separate twiddle factors here, one for the beginning
        // and one for the end. But the twiddle factor for the end is jsut the twiddle for the beginning, with the
        // real part negated. Since it's the same twiddle, we can factor out a ton of math ops and cut the number of
        // multiplications in half
        let twiddled_diffsum_blended = AvxVector::mul(diffsum_blended, twiddle_im);
        let twiddled_diffsum_swapped = AvxVector::mul(diffsum_swapped, twiddle_re);
        let half_sumdiff = AvxVector::mul(sumdiff_blended, AvxVector::half());

        let twiddled_output = AvxVector::add(twiddled_diffsum_blended, twiddled_diffsum_swapped);

        let out_fwd = AvxVector::add(half_sumdiff, twiddled_output);
        let out_rev = AvxVector::sub(half_sumdiff, twiddled_output).conj();
        let out_rev = AvxVector::reverse_complex_elements(out_rev);

        (out_fwd, out_rev)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn postprocess_output(&self, buffer: &mut [Complex<A>]) {
        // Next step is to apply twiddle factors to our output array, in-place.
        // The process works by loading 2 elements from opposite ends of the array,
        // combining them, and writing them back where we found them.
        // To support reading/writing from opposite ends of the array simultaneously, split the output array in half
        let (mut output_left, mut output_right) = buffer.split_at_mut(buffer.len() / 2);

        let special_first_element = output_left.len() % 2 == 1;

        // The first and last element don't require any twiddle factors, so skip that work
        match (output_left.first_mut(), output_right.last_mut()) {
            (Some(first_element), Some(last_element)) => {
                if special_first_element {
                    // The first and last elements are just a sum and difference of the first value's real and imaginary values
                    let first_value = *first_element;
                    *first_element = Complex {
                        re: first_value.re + first_value.im,
                        im: A::zero(),
                    };
                    *last_element = Complex {
                        re: first_value.re - first_value.im,
                        im: A::zero(),
                    };

                    // Chop the first and last element off of our slices so that the loop below doesn't have to deal with them
                    output_left = &mut output_left[1..];
                    let right_len = output_right.len();
                    output_right = &mut output_right[..right_len - 1];
                } else {
                    // Copy the first element to the last element to prepare for the main postprocessing loop below
                    *last_element = *first_element;
                }
            }
            _ => {
                return;
            }
        }

        let chunk_count = output_left.len() / A::VectorType::COMPLEX_PER_VECTOR;
        let remainder_count = output_left.len() % A::VectorType::COMPLEX_PER_VECTOR;

        // Loop over the remaining elements and apply twiddle factors on them.
        // This algorithm is ripped directly from the scalar version of the same struct. Thankfully it maps very cleanly.
        for (i, twiddle) in (&self.twiddles[..chunk_count]).iter().enumerate() {
            // We need to load a bunch of elements from the beginning of the array, and a bunch from the end.
            // In the scalar version, we load the ones fro mthe end in reverse order, so we have to reverse them here
            let val_rev = output_right
                .load_complex(output_right.len() - (i + 1) * A::VectorType::COMPLEX_PER_VECTOR);
            let val_fwd = output_left.load_complex(i * A::VectorType::COMPLEX_PER_VECTOR);

            let (out_fwd, out_rev) = Self::postprocess_chunk(val_fwd, val_rev, *twiddle);

            output_left.store_complex(out_fwd, i * A::VectorType::COMPLEX_PER_VECTOR);
            output_right.store_complex(
                out_rev,
                output_right.len() - (i + 1) * A::VectorType::COMPLEX_PER_VECTOR,
            );
        }

        // Our tricks with the first/last element up above meant that we will never have to deal with remainders of 1 or 3. But we may have to deal with a remainder of 2.
        if remainder_count == 2 {
            // We need to load a bunch of elements from the beginning of the array, and a bunch from the end.
            // In the scalar version, we load the ones fro mthe end in reverse order, so we have to reverse them here
            let val_rev = output_right.load_partial2_complex(
                output_right.len() - chunk_count * A::VectorType::COMPLEX_PER_VECTOR - 2,
            );
            let val_fwd = output_left.load_partial2_complex(output_left.len() - 2);
            let twiddle = self.twiddles.last().unwrap().lo();

            let (out_fwd, out_rev) = Self::postprocess_chunk(val_fwd, val_rev, twiddle);

            output_left.store_partial2_complex(out_fwd, output_left.len() - 2);
            output_right.store_partial2_complex(
                out_rev,
                output_right.len() - chunk_count * A::VectorType::COMPLEX_PER_VECTOR - 2,
            );
        }

        // If the output len is odd, the loop above can't postprocess the centermost element, so handle that separately
        if self.direction == FftDirection::Forward && buffer.len() % 2 == 1 {
            if let Some(center_element) = buffer.get_mut(buffer.len() / 2) {
                center_element.im = -center_element.im;
            }
        }
    }
}
impl<A: AvxNum, T: FftNum> FftRealToComplex<T> for RealToComplexEvenAvx<A, T> {
    fn process(&self, input: &mut [T], output: &mut [Complex<T>], scratch: &mut [T]) {
        if self.len() == 0 {
            return;
        }

        // The simplest part of the process is computing the inner FFT. Just transmute the input and forward it to the FFT
        {
            let inner_fft_len = self.len() / 2;

            let input_complex = into_complex_mut(input);
            let chopped_output = &mut output[..inner_fft_len];
            let scratch_complex = into_complex_mut(scratch);

            self.inner_fft.process_outofplace_with_scratch(
                input_complex,
                chopped_output,
                scratch_complex,
            );
        }

        let output_avx: &mut [Complex<A>] = unsafe { workaround_transmute_mut(output) };

        unsafe { self.postprocess_output(output_avx) };
    }

    fn get_scratch_len(&self) -> usize {
        self.required_scratch
    }
}
impl<A: AvxNum, T: FftNum> Length for RealToComplexEvenAvx<A, T> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::{
        algorithm::Dft,
        test_utils::{compare_vectors, random_real_signal},
        FftDirection,
    };
    use num_traits::{Float, Zero};
    use rand::distributions::uniform::SampleUniform;

    #[test]
    fn test_r2c_even_avx() {
        for inner_len in 0..10 {
            test_r2c_even_with_inner::<f32>(inner_len, FftDirection::Forward);
            test_r2c_even_with_inner::<f64>(inner_len, FftDirection::Forward);

            // Note: Even though R2C is usually used with a forward FFT, it was pretty trivial to make it support inverse FFTs.
            // If there's a compelling performance reason to drop inverse support, go ahead.
            test_r2c_even_with_inner::<f32>(inner_len, FftDirection::Inverse);
            test_r2c_even_with_inner::<f64>(inner_len, FftDirection::Inverse);
        }
    }

    fn test_r2c_even_with_inner<A: AvxNum + SampleUniform + Float>(
        inner_len: usize,
        direction: FftDirection,
    ) {
        let inner_fft: Arc<Dft<A>> = Arc::new(Dft::new(inner_len, direction));
        let fft = RealToComplexEvenAvx::<A, A>::new(inner_fft).unwrap();

        let control = Dft::new(inner_len * 2, direction);

        let mut real_input = random_real_signal(control.len());
        let mut complex_input: Vec<Complex<A>> = real_input.iter().map(Complex::from).collect();

        control.process(&mut complex_input);

        let mut real_output = vec![Complex::zero(); inner_len + 1];
        let mut scratch = vec![A::zero(); fft.get_scratch_len()];
        fft.process(&mut real_input, &mut real_output, &mut scratch);

        if inner_len > 0 {
            assert!(
                compare_vectors(&complex_input[..inner_len + 1], &real_output),
                "process() failed, len = {}, direction = {}",
                inner_len * 2,
                direction,
            );
        }
    }
}
