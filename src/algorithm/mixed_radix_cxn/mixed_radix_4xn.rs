use std::sync::Arc;

use num_complex::Complex;
use transpose;

use common::{FFTnum, verify_length, verify_length_divisible};

use ::{Length, IsInverse, FFT};
use twiddles;
use super::ColumnChunksExactMut4;


/// Specialized implementation of the Mixed-Radix FFT algorithm where the FFT size is a multiple of 4
///
/// This algorithm factors a FFT into 4 * N, computes several inner FFTs of size 4 and N, then combines the 
/// results to get the final answer. N must greater than or equal to 4.
///
/// ~~~
/// // Computes a forward FFT of size 1200, using the Mixed-Radix 2xN Algorithm
/// use rustfft::algorithm::MixedRadix2xN;
/// use rustfft::{FFT, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1200];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1200];
///
/// // we need to find a N such that 4 * N == 1200
/// // N = 300 satisfies this
/// let mut planner = FFTplanner::new(false);
/// let inner_fft = planner.plan_fft(300);
///
/// // the mixed radix FFT length will be 4 * inner_fft.len() = 1200
/// let fft = MixedRadix4xN::new(inner_fft);
/// fft.process(&mut input, &mut output);
/// ~~~

pub struct MixedRadix4xN<T> {
    inner_fft: Arc<FFT<T>>,
    twiddles: Box<[Complex<T>]>,

    chunk_size: usize,
    inner_len: usize,

    inverse: bool,
}

impl<T: FFTnum> MixedRadix4xN<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `4 * inner_fft.len()`
    pub fn new(inner_fft: Arc<FFT<T>>) -> Self {
    	let inner_len = inner_fft.len();
        let len = 4 * inner_len;

        let chunk_size = Self::get_chunk_size();
        let chunk_count = inner_len / chunk_size;

        let inverse = inner_fft.is_inverse();

        let mut twiddles = Vec::with_capacity(len / 4 * 3);

        // Chunk twiddles
        for x_chunk in 0..chunk_count {
        	for y in 1..4 {
        		for x_idx in 0..chunk_size {
        			let x = x_chunk * chunk_size + x_idx;
	                twiddles.push(twiddles::single_twiddle(x * y, len, inverse));
	            }
	        }
        }

        // Remainder twiddles
        for y in 1..4 {
        	for x in chunk_count*chunk_size..inner_len {
        		twiddles.push(twiddles::single_twiddle(x * y, len, inverse));
        	}
        }

        Self {
            inner_fft,
            twiddles: twiddles.into_boxed_slice(),
            chunk_size,
            inner_len,
            inverse,
        }
    }

    // When doing our FFTs of size 4, we're actually going to do them in chunks of this size.
    // The simplest thing to do would be a chunk size of 1, but benchmarking shows that more is decidedly faster
    // We will store our twiddles in rows of this chunk size for easy SIMD layout
    fn get_chunk_size() -> usize {
    	8
    }


    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        // six step FFT, just like the main mixed radix implementation. But swe know the eact size of one of the two FFTs, it's pretty safe to skip the transposes, and we can be a little smarter with our twiddle factors 

        // STEP 1: Transpose -- skipped because we will do the first set of FFTs non-contiguously

        // STEP 2 and 3 combined: Perform FFTs of size 4 and apply our twiddle factors.
        if !self.inverse {
	        column_butterfly4s_forward(input, &self.twiddles, self.chunk_size);
	    } else {
	    	column_butterfly4s_inverse(input, &self.twiddles, self.chunk_size);
	    }
	    output.copy_from_slice(input);
       
       	// STEP 4: Transpose -- skipped because we will do the first set of FFTs non-contiguously

        // STEP 5: perform FFTs of size `inner_len`
        self.inner_fft.process_multi(output, input);

        // STEP 6: transpose again
        transpose::transpose(input, output, self.inner_len, 4);
    }
}
impl<T: FFTnum> FFT<T> for MixedRadix4xN<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

        self.perform_fft(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            self.perform_fft(in_chunk, out_chunk);
        }
    }
}
impl<T> Length for MixedRadix4xN<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.twiddles.len() / 3 * 4
    }
}
impl<T> IsInverse for MixedRadix4xN<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub(crate) unsafe fn process_butterfly4_chunks_naive<'a, T: FFTnum>(column_chunks: [&'a mut [Complex<T>];4], outer_twiddle_chunk: &[Complex<T>], chunk_size: usize, inner_twiddle: impl Fn(Complex<T>) -> Complex<T>) {
	for i in 0..chunk_size {
		// first: column butterfly 2s
		let sum0 = *column_chunks[0].get_unchecked(i) + *column_chunks[2].get_unchecked(i);
		*column_chunks[2].get_unchecked_mut(i) = *column_chunks[0].get_unchecked(i) - *column_chunks[2].get_unchecked(i);
		*column_chunks[0].get_unchecked_mut(i) = sum0;
		let sum1 = *column_chunks[1].get_unchecked(i) + *column_chunks[3].get_unchecked(i);
		let diff3 = *column_chunks[1].get_unchecked(i) - *column_chunks[3].get_unchecked(i);
		*column_chunks[1].get_unchecked_mut(i) = sum1;
		*column_chunks[3].get_unchecked_mut(i) = inner_twiddle(diff3); // Apply butterfly 4 twiddle factor here


		// Next: Row butterfly 2s. Also apply twiddle factors.
		let sum0 = *column_chunks[0].get_unchecked(i) + *column_chunks[1].get_unchecked(i);
		let diff1 = *column_chunks[0].get_unchecked(i) - *column_chunks[1].get_unchecked(i);
		let sum2 = *column_chunks[2].get_unchecked(i) + *column_chunks[3].get_unchecked(i);
		let diff3 = *column_chunks[2].get_unchecked(i) - *column_chunks[3].get_unchecked(i);

		*column_chunks[0].get_unchecked_mut(i) = sum0;
		*column_chunks[1].get_unchecked_mut(i) = sum2 * *outer_twiddle_chunk.get_unchecked(i);
		*column_chunks[2].get_unchecked_mut(i) = diff1 * *outer_twiddle_chunk.get_unchecked(i + chunk_size);
		*column_chunks[3].get_unchecked_mut(i) = diff3 * *outer_twiddle_chunk.get_unchecked(i + chunk_size * 2);
	}
}

pub(crate) fn column_butterfly4s_forward<T: FFTnum>(buffer: &mut [Complex<T>], outer_twiddles: &[Complex<T>], chunk_size: usize) {
	assert_eq!(buffer.len() / 4 * 3, outer_twiddles.len());

	let mut column_chunk_iter = ColumnChunksExactMut4::new(buffer, chunk_size);
	let mut outer_twiddle_chunks = outer_twiddles.chunks_exact(chunk_size * 3);
	let twiddle_fn = |c: Complex<T>| Complex{ re: c.im, im: -c.re };

	for (column_chunks, outer_twiddle_chunk) in column_chunk_iter.by_ref().zip(outer_twiddle_chunks.by_ref()) {
		unsafe { process_butterfly4_chunks_naive(column_chunks, outer_twiddle_chunk, chunk_size, twiddle_fn) };
	}

	let column_chunks = column_chunk_iter.into_remainder();
	let outer_twiddle_chunk = outer_twiddle_chunks.remainder();
	let remainder_len = column_chunks[0].len();
	unsafe { process_butterfly4_chunks_naive(column_chunks, outer_twiddle_chunk, remainder_len, twiddle_fn) };
}

pub(crate) fn column_butterfly4s_inverse<T: FFTnum>(buffer: &mut [Complex<T>], outer_twiddles: &[Complex<T>], chunk_size: usize) {
	assert_eq!(buffer.len() / 4 * 3, outer_twiddles.len());

	let mut column_chunk_iter = ColumnChunksExactMut4::new(buffer, chunk_size);
	let mut outer_twiddle_chunks = outer_twiddles.chunks_exact(chunk_size * 3);
	let twiddle_fn = |c: Complex<T>| Complex{ re: -c.im, im: c.re };

	for (column_chunks, outer_twiddle_chunk) in column_chunk_iter.by_ref().zip(outer_twiddle_chunks.by_ref()) {
		unsafe { process_butterfly4_chunks_naive(column_chunks, outer_twiddle_chunk, chunk_size, twiddle_fn) };
	}

	let column_chunks = column_chunk_iter.into_remainder();
	let outer_twiddle_chunk = outer_twiddle_chunks.remainder();
	let remainder_len = column_chunks[0].len();
	unsafe { process_butterfly4_chunks_naive(column_chunks, outer_twiddle_chunk, remainder_len, twiddle_fn) };
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use test_utils::check_fft_algorithm;
    use algorithm::DFT;

    #[test]
    fn test_mixedx4_radix() {
        for width in 8..9 {
            test_mixed_radix_with_length(width, false);
            //test_mixed_radix_with_length(width, true);
        }
    }

    fn test_mixed_radix_with_length(inner_len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(inner_len, inverse)) as Arc<FFT<f32>>;

        let fft = MixedRadix4xN::new(inner_fft);

        check_fft_algorithm(&fft, inner_len * 4, inverse);
    }
}
