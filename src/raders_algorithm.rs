
use num::{Complex, Zero, FromPrimitive, Signed, Num};
use std::f32;

use radix4::process_radix2_inplace;
use math_utils;

pub struct RadersAlgorithm<T> {
	len: usize,

	primitive_root: u64,
    root_inverse: u64,

	twiddles: Vec<Complex<T>>,
	unity_fft_result: Vec<Complex<T>>,
	scratch: Vec<Complex<T>>,
}

impl<T> RadersAlgorithm<T> where T: Signed + FromPrimitive + Copy {

	pub fn new(len: usize, inverse: bool) -> Self {

		// we can theoretically just always do n - 1 as the inner FFT size
		// BUT the code will be much simpler if we can just always call radix 4 -- because it doesn't need any extra scratch space
		// so we only use n - 1 if it's a power of two. otherwise we'll pad it out to the next power of two
        let inner_fft_size = if (len - 1).is_power_of_two() {
        	len - 1
        } else {
        	(2 * len - 3).next_power_of_two()
        };

        let dir = if inverse { 1 } else { -1 };

        //compute the primitive root and its inverse for this size
        let primitive_root = math_utils::primitive_root(len as u64).unwrap();
        let root_inverse = math_utils::multiplicative_inverse(primitive_root, len as u64);

        //compute the twiddles for the inner fft
        let inner_twiddles: Vec<Complex<T>> = (0..inner_fft_size)
            .map(|i| dir as f32 * i as f32 * 2.0 * f32::consts::PI / inner_fft_size as f32)
            .map(|phase| Complex::from_polar(&1.0, &phase))
            .map(|c| {
                Complex {
                    re: FromPrimitive::from_f32(c.re).unwrap(),
                    im: FromPrimitive::from_f32(c.im).unwrap(),
                }
            })
            .collect();

        let size_float: f32 = FromPrimitive::from_usize(inner_fft_size).unwrap();
        let length_scale = 1f32 / size_float;

        //precompute the coefficients to use inside the process method
        let mut unity_fft_data: Vec<Complex<T>> = (0..len - 1)
            .map(|i| math_utils::modular_exponent(root_inverse, i as u64, len as u64))
            .map(|i| { dir as f32 * i as f32 * 2.0 * f32::consts::PI / len as f32})
            .map(|phase| Complex::from_polar(&length_scale, &phase))
            .map(|c| {
                Complex {
                    re: FromPrimitive::from_f32(c.re).unwrap(),
                    im: FromPrimitive::from_f32(c.im).unwrap(),
                }
            })
            .collect();

        //pad out the fft input if necessary by repeating the values
        unity_fft_data.reserve(inner_fft_size);
        let mut index = 0;
        while unity_fft_data.len() < inner_fft_size {
            let element = unsafe { *unity_fft_data.get_unchecked(index) };
            unity_fft_data.push(element);
            index += 1;
        }

        //FFT the unity fft data
        process_radix2_inplace(inner_fft_size, unity_fft_data.as_mut_slice(), 1, inner_twiddles.as_slice());

        RadersAlgorithm {
        	len: len,
        	primitive_root: primitive_root,
            root_inverse: root_inverse,
            unity_fft_result: unity_fft_data,
            twiddles: inner_twiddles,
            scratch: vec![Zero::zero(); inner_fft_size],
        }
    }

    /// Runs the FFT on the input `input` buffer, replacing it with the FFT result
    pub fn process(&mut self, input: &mut [Complex<T>], stride: usize) {
    	assert!(input.len() == self.len);

        self.setup_inner_fft(input, stride);

    	//use radix 2 to run a FFT on the data now in the scratch space
    	process_radix2_inplace(self.scratch.len(), self.scratch.as_mut_slice(), 1, self.twiddles.as_slice());

    	//multiply the result pointwise with the cached unity FFT
    	for (scratch_item, unity) in self.scratch.iter_mut().zip(self.unity_fft_result.iter()) {
    		*scratch_item = *scratch_item * unity;
    	}

    	//prepare for the inverse FFT by conjugating all our twiddle factors
    	self.conjugate_twiddles();

    	//execute the inverse FFT
    	process_radix2_inplace(self.scratch.len(), self.scratch.as_mut_slice(), 1, self.twiddles.as_slice());

    	//make sure we'll be ready for the next call by undoing the twiddle conjugation
    	self.conjugate_twiddles();

        // the first output element is equal to the sum of the others. but we need the first input element, so store it before computing the output
        let first_input = unsafe { *input.get_unchecked(0) };
        for i in 1..self.len {
            unsafe { *input.get_unchecked_mut(0) = input.get_unchecked(0) + input.get_unchecked(i * stride); }
        }

        //copy the data back into the input vector, but again it's not just a straight copy
    	for scratch_index in 0..self.len-1 {
            let output_index = math_utils::modular_exponent(self.root_inverse, scratch_index as u64, self.len as u64) as usize;

            unsafe {
                *input.get_unchecked_mut (stride * output_index) = 
                first_input + *self.scratch.get_unchecked_mut(scratch_index);
            }
        }
    }

    fn setup_inner_fft(&mut self, input: &[Complex<T>], stride: usize) {
        //it's not just a straight copy from the input to the scratch, we have
        //to compute the input index based on the scratch index and primitive root
        let get_input_val = |base: u64, exponent: u64, modulo: u64| {
            let input_index = math_utils::modular_exponent(base, exponent, modulo) as usize;
            unsafe { *input.get_unchecked(stride * input_index) }
        };

        // copy the input into the scratch space
        if self.len - 1 == self.scratch.len() {
            for scratch_index in 0..self.scratch.len() {
                unsafe { *self.scratch.get_unchecked_mut(scratch_index) = get_input_val(self.primitive_root, scratch_index as u64, self.len as u64) };
            }
        } else {
            //we have to zero-pad the input in a very specific way. input[1] goes at the beginning of the scratch, and the rest is packed at the end
            //the rest is zeroes
            unsafe { *self.scratch.get_unchecked_mut(0) = *input.get_unchecked(stride); };

            //zero fill the middle
            let zero_end = self.scratch.len() - (self.len - 2);
            zero_fill(&mut self.scratch[1..zero_end]);

            for scratch_index in 1..self.len-1 {
                unsafe { *self.scratch.get_unchecked_mut(scratch_index + zero_end - 1)
                    = get_input_val(self.primitive_root, scratch_index as u64, self.len as u64) };
            }
        }
    }

    fn conjugate_twiddles(&mut self) {
    	for item in &mut self.twiddles {
    		*item = item.conj();
    	}
    }
}

fn zero_fill<T: Num + Clone>(input: &mut [Complex<T>]) {
	for element in input.iter_mut() {
		*element = Zero::zero();
	}
}
