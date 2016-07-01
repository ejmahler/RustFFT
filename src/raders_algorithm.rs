
use num::{Complex, Zero, One, Float, FromPrimitive, Signed, Num, Integer, PrimInt};
use num::traits::cast;
use num::CheckedSub;
use std::f32;
use std::mem::swap;

use std::ops::Add;
use radix4::execute_radix2;
use math_utils;

pub struct RadersAlgorithm<T> {
	len: usize,
	primitive_root: usize,
	twiddles: Vec<Complex<T>>,
	unity_fft_result: Vec<Complex<T>>,
	scratch: Vec<Complex<T>>,
	inverse: bool,
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

        let dir = if inverse { 1 } else { -1 } as usize;

        RadersAlgorithm {
        	len: len,
        	primitive_root: math_utils::primitive_root(len as u64).unwrap() as usize,
            twiddles: (0..inner_fft_size)
                .map(|i| dir as f32 * i as f32 * 2.0 * f32::consts::PI / len as f32)
                .map(|phase| Complex::from_polar(&1.0, &phase))
                .map(|c| {
                    Complex {
                        re: FromPrimitive::from_f32(c.re).unwrap(),
                        im: FromPrimitive::from_f32(c.im).unwrap(),
                    }
                })
                .collect(),
            unity_fft_result: Vec::new(),
            scratch: vec![Zero::zero(); inner_fft_size],
            inverse: inverse,
        }
    }

    /// Runs the FFT on the input `input` buffer, replacing it with the FFT result
    pub fn process(&mut self, input: &mut [Complex<T>], stride: usize) {
    	assert!(input.len() == self.len);

    	// the first output element is equal to the sum of the others. but we need the first input element, so sore it before computing the output
    	let first_input = unsafe { *input.get_unchecked(0) };
    	for i in 1..self.len {
    		unsafe { *input.get_unchecked_mut(0) = input.get_unchecked(0) + input.get_unchecked(i * stride); }
    	}

    	// copy the input into the scratch space
    	if self.len - 1 == self.scratch.len() {
    		//ignore input[0] because we already computed it
    		unsafe { copy_data(&input[stride..], self.scratch.as_mut_slice(), self.len - 1, stride, 1) }
    	} else {
    		//we have to zero-pad the input in a very specific way. input[1] goes at the beginning of the scratch, and the rest is packed at the end
    		//the rest is zeroes
    		unsafe { *self.scratch.get_unchecked_mut(0) = *input.get_unchecked(stride); }

    		//zero fill the middle
    		let zero_end = self.scratch.len() - (self.len - 2);
    		zero_fill(&mut self.scratch[1..zero_end]);

    		//fill in the rest wih the rest of the input array
    		unsafe { copy_data(&input[2*stride..], &mut self.scratch[zero_end..], self.len - 2, stride, 1) }
    	}

    	//use radix 2 to run a FFT on the data now in the scratch space. but explicitly avoid the preparation step!
    	//all the preparation would do is reorder the inputs. but we're about to do an inverse FFT
    	//and said inverse FFT will do our job of reordering for us
    	execute_radix2(self.scratch.len(), self.scratch.as_mut_slice(), 1, self.twiddles.as_slice());

    	//multiply the result pointwise with the cached unity FFT
    	for (scratch_item, unity) in self.scratch.iter_mut().zip(self.unity_fft_result.iter()) {
    		*scratch_item = *scratch_item * unity;
    	}

    	//prepare for the inverse FFT by conjugating all our twiddle factors
    	self.conjugate_twiddles();

    	//execute the inverse FFT
    	execute_radix2(self.scratch.len(), self.scratch.as_mut_slice(), 1, self.twiddles.as_slice());

    	//make sure we'll be ready for the next call by undoing the twiddle conjugation
    	self.conjugate_twiddles();

    	unsafe { copy_data(&self.scratch[..self.len - 1], &mut input[stride..], self.len - 1, 1, stride) }
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

unsafe fn copy_data<T: Copy>(source: &[T], dest: &mut [T], count: usize, source_stride: usize, dest_stride: usize) {
	for i in 0..count {
		*dest.get_unchecked_mut(i * dest_stride) = *source.get_unchecked(i * source_stride);
	}
}
