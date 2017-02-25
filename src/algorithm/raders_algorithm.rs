
use num::{Complex, Zero, FromPrimitive, Signed, Num};
use std::f32;

use butterflies::{butterfly_2_single, butterfly_2_inverse, butterfly_2_dif};
use math_utils;
use twiddles;
use super::FFTAlgorithm;

pub struct RadersAlgorithm<T> {
    len: usize,

    primitive_root: u64,
    root_inverse: u64,

    unity_fft_result: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,

    inner_fft: InnerFFT<T>,
}

impl<T> RadersAlgorithm<T>
    where T: Signed + FromPrimitive + Copy
{
    pub fn new(len: usize, inverse: bool) -> Self {

        // we can theoretically just always do n - 1 as the inner FFT size
        // BUT the code will be much simpler if we can just always call radix 2
        // so we only use n - 1 if it's a power of two
        // otherwise we'll pad it out to the next power of two
        let inner_fft_size = if (len - 1).is_power_of_two() {
            len - 1
        } else {
            (2 * len - 3).next_power_of_two()
        };

        // compute the primitive root and its inverse for this size
        let primitive_root = math_utils::primitive_root(len as u64).unwrap();
        let root_inverse = math_utils::multiplicative_inverse(primitive_root, len as u64);

        // precompute the coefficients to use inside the process method
        let unity_scale = 1f32 / inner_fft_size as f32;
        let dir = if inverse {
            1
        } else {
            -1
        };
        let mut unity_fft_data: Vec<Complex<T>> = (0..len - 1)
            .map(|i| math_utils::modular_exponent(root_inverse, i as u64, len as u64))
            .map(|i| dir as f32 * i as f32 * 2.0 * f32::consts::PI / len as f32)
            .map(|phase| Complex::from_polar(&unity_scale, &phase))
            .map(|c| {
                Complex {
                    re: FromPrimitive::from_f32(c.re).unwrap(),
                    im: FromPrimitive::from_f32(c.im).unwrap(),
                }
            })
            .collect();

        // pad out the fft input if necessary by repeating the values
        unity_fft_data.reserve(inner_fft_size);
        let mut index = 0;
        while unity_fft_data.len() < inner_fft_size {
            let element = unsafe { *unity_fft_data.get_unchecked(index) };
            unity_fft_data.push(element);
            index += 1;
        }

        let inner_fft = InnerFFT::new(inner_fft_size, inverse);
        inner_fft.process(unity_fft_data.as_mut_slice());

        RadersAlgorithm {
            len: len,
            primitive_root: primitive_root,
            root_inverse: root_inverse,
            unity_fft_result: unity_fft_data,
            scratch: vec![Zero::zero(); inner_fft_size],
            inner_fft: inner_fft,
        }
    }

    fn setup_inner_fft(&mut self, input: &[Complex<T>]) {
        // it's not just a straight copy from the input to the scratch, we have
        // to compute the input index based on the scratch index and primitive root
        let get_input_val = |base: u64, exponent: u64, modulo: u64| {
            let input_index = math_utils::modular_exponent(base, exponent, modulo) as usize;
            unsafe { *input.get_unchecked(input_index) }
        };

        // copy the input into the scratch space
        if self.len - 1 == self.scratch.len() {
            for (scratch_index, scratch_element) in self.scratch.iter_mut().enumerate() {
                *scratch_element =
                    get_input_val(self.primitive_root, scratch_index as u64, self.len as u64);
            }
        } else {
            // we have to zero-pad the input in a very specific way. input[1]
            // goes at the beginning of the scratch, and the rest is packed at the end
            // the rest is zeroes
            unsafe {
                *self.scratch.get_unchecked_mut(0) = *input.get_unchecked(1);
            };

            // zero fill the middle
            let zero_end = self.scratch.len() - (self.len - 2);
            zero_fill(&mut self.scratch[1..zero_end]);

            for (scratch_index, scratch_element) in self.scratch[zero_end..]
                .iter_mut()
                .enumerate() {
                *scratch_element = get_input_val(self.primitive_root,
                                                 (scratch_index + 1) as u64,
                                                 self.len as u64);
            }
        }
    }

    fn copy_to_output(&mut self, output: &mut [Complex<T>], first_element: Complex<T>) {
        // copy the data back into the input vector, but again it's not just a straight copy
        for (scratch_index, scratch_element) in self.scratch[..self.len - 1].iter().enumerate() {
            let output_index =
                math_utils::modular_exponent(self.root_inverse,
                                             scratch_index as u64,
                                             self.len as u64) as usize;

            *unsafe { output.get_unchecked_mut(output_index) } = first_element + *scratch_element;
        }
    }
}

impl<T> FFTAlgorithm<T> for RadersAlgorithm<T>
    where T: Signed + FromPrimitive + Copy
{
    /// Runs the FFT on the input `signal` array, placing the output in the 'spectrum' array
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        self.setup_inner_fft(signal);

        // use radix 2 to run a FFT on the data now in the scratch space
        self.inner_fft.process(self.scratch.as_mut_slice());

        // multiply the result pointwise with the cached unity FFT
        for (scratch_item, unity) in self.scratch.iter_mut().zip(self.unity_fft_result.iter()) {
            *scratch_item = *scratch_item * unity;
        }

        // execute the inverse FFT
        self.inner_fft.process_inverse(self.scratch.as_mut_slice());

        // the first output element is equal to the sum of the whole input array
        let sum = signal.iter().fold(Zero::zero(), |acc, &x| acc + x);
        unsafe { *spectrum.get_unchecked_mut(0) = sum };

        // copy the rest of the output from the scratch space
        let first_input = unsafe { *signal.get_unchecked(0) };
        self.copy_to_output(spectrum, first_input);
    }
}

fn zero_fill<T: Num + Clone>(input: &mut [Complex<T>]) {
    for element in input.iter_mut() {
        *element = Zero::zero();
    }
}



/// This ineer FFT is designed to be used in situations where you need to perform a forward FFT
/// followed by some processing, immediately followed by an inverse FFT on the same data
///
/// In these cases, it speeds up the process by skipping the reordering step on the FFT output
/// and skipping the reordering step on the inverse FFT input
///
/// So the in-between data will be in bit-reversed order.
/// This is acceptable for raders algorithm because all we're doing is multiplying pointwise
/// with another FFT'ed array. the output is the same if both are in natural order
/// vs both being in bit reversed order.
///
/// so this inner FFT class enables skipping the unnecessary work of reordering
struct InnerFFT<T> {
    twiddles: Vec<Complex<T>>,
}

impl<T> InnerFFT<T>
    where T: Signed + FromPrimitive + Copy
{
    pub fn new(len: usize, inverse: bool) -> Self {
        InnerFFT {
            twiddles: twiddles::generate_twiddle_factors(len, inverse),
        }
    }

    /// Performs a DIF radix 2 FFT in the provided sequence
    /// The DIF formula takes natural order input and outputs bit-reversed
    pub fn process(&self, spectrum: &mut [Complex<T>]) {

        // perform all the cross-FFTs, one "layer" at a time
        // the innermost for loop is basically the "butterfly_2" function, except
        // butterfly_2 is designed for a DIT FFT, and we need DIF
        let num_layers = spectrum.len().trailing_zeros() as usize;
        for layer in 0..num_layers - 1 {
            let num_groups = 1 << layer;
            let group_size = spectrum.len() / num_groups;

            for chunk in spectrum.chunks_mut(group_size) {
                unsafe {
                    butterfly_2_dif(chunk, num_groups, self.twiddles.as_slice(), group_size / 2)
                }
            }
        }

        // perform the butterflies, with a stride of size / 2
        for chunk in spectrum.chunks_mut(2) {
            unsafe { butterfly_2_single(chunk, 1) }
        }
    }

    /// Performs a DIT radix 2 FFT in the provided sequence
    /// The DIT formula takes bit-reversed input and outputs in natural order
    /// Also makes this an actual inverse fft by conjugating the twiddle factors
    pub fn process_inverse(&self, spectrum: &mut [Complex<T>]) {
        // perform the butterflies, with a stride of size / 2
        for chunk in spectrum.chunks_mut(2) {
            unsafe { butterfly_2_single(chunk, 1) }
        }

        // now, perform all the cross-FFTs, one "layer" at a time
        // the innermost for loop is basically the "butterfly_2" step, except
        // we're calling ".conj()" on each twiddle factor before using it
        let num_layers = spectrum.len().trailing_zeros() as usize;
        for layer in (0..num_layers - 1).rev() {
            let num_groups = 1 << layer;
            let group_size = spectrum.len() / num_groups;

            for chunk in spectrum.chunks_mut(group_size) {
                unsafe {
                    butterfly_2_inverse(chunk, num_groups, self.twiddles.as_slice(), group_size / 2)
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::InnerFFT;

    use ::dft;

    use num::Complex;

    #[test]
    fn test_inner_fft_forward() {
        let len = 8;
        let input: Vec<Complex<f32>> = (0..len).map(|i| Complex::from(i as f32)).collect();

        let fft = InnerFFT::new(len, false);

        let mut midpoint_expected = input.clone();
        dft(input.as_slice(), midpoint_expected.as_mut_slice());

        let mut midpoint_actual = input.clone();
        fft.process(midpoint_actual.as_mut_slice());

        reorder(midpoint_actual.as_mut_slice());
        assert!(compare_vectors(midpoint_expected.as_slice(), midpoint_actual.as_slice()));
    }

    #[test]
    fn test_inner_fft_backward() {
        let len = 4;
        let input: Vec<Complex<f32>> = (0..len).map(|i| Complex::from(i as f32)).collect();

        let fft = InnerFFT::new(len, false);

        // to set up for the inverse FFT, use the correct FFT to get the midpoint, then reorder it
        let mut result = input.clone();
        dft(input.as_slice(), result.as_mut_slice());
        reorder(result.as_mut_slice());

        fft.process_inverse(result.as_mut_slice());

        // we have to scale by 1/n to get back to the input vector
        let result_scale = 1f32 / len as f32;
        for element in result.iter_mut() {
            *element = *element * result_scale;
        }

        assert!(compare_vectors(input.as_slice(), result.as_slice()));
    }

    fn reorder<T: Copy>(input: &mut [T]) {
        let num_bits = input.len().trailing_zeros();

        for i in 0..input.len() {
            let swap_index = reverse_bits(i, num_bits);

            if swap_index > i {
                input.swap(i, swap_index);
            }
        }
    }

    fn reverse_bits(mut n: usize, num_bits: u32) -> usize {
        let mut result = 0;
        for _ in 0..num_bits {
            result <<= 1;
            result |= n & 1;
            n >>= 1;
        }
        result
    }

    /// Returns true if the mean difference in the elements of the two vectors
    /// is small
    fn compare_vectors(vec1: &[Complex<f32>], vec2: &[Complex<f32>]) -> bool {
        assert_eq!(vec1.len(), vec2.len());
        let mut sse = 0f32;
        for (&a, &b) in vec1.iter().zip(vec2.iter()) {
            sse = sse + (a - b).norm();
        }
        return (sse / vec1.len() as f32) < 0.1f32;
    }
}
