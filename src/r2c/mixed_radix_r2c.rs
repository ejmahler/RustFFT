use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use crate::{Fft, FftDirection, FftNum, FftRealToComplex, Length, array_utils::{self, into_complex_mut, into_real_mut}, common::r2c_error_outofplace, twiddles};



pub struct MixedRadixR2C<T> {
    inner_r2c: Arc<dyn FftRealToComplex<T>>,
    inner_r2c_len: usize,
    inner_r2c_complex_len: usize,

    inner_fft: Arc<dyn Fft<T>>,
    inner_fft_len: usize,

    len: usize,

    twiddles: Box<[Complex<T>]>,
}
impl<T: FftNum> MixedRadixR2C<T> {
    pub fn new(inner_r2c: Arc<dyn FftRealToComplex<T>>, inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();
        let inner_r2c_len = inner_r2c.len();

        let direction = inner_fft.fft_direction();

        let inner_r2c_complex_len = inner_r2c_len / 2 + 1;

        let len = inner_fft_len * inner_r2c_len;

        let mut twiddles = Vec::with_capacity(inner_r2c_complex_len * inner_fft_len);

        for k in 0..inner_fft_len {
            for i in 0..inner_r2c_complex_len {
                twiddles.push(twiddles::compute_twiddle(i*k, len, direction))
            }
        }
        Self {
            len: inner_r2c_len * inner_fft_len,

            inner_r2c_len,
            inner_r2c_complex_len,
            inner_r2c,

            inner_fft_len,
            inner_fft,

            twiddles: twiddles.into_boxed_slice(),

        }
    }
    #[inline(never)]
    fn perform_r2c(&self, input: &[T], output: &mut [Complex<T>], scratch: &mut [T]) {
        let (scratch1, scratch2) = into_complex_mut(scratch).split_at_mut(self.inner_fft_len * self.inner_r2c_complex_len);

        // Step 1: Transpose. The output is guaranteed to be big enough to hold the input, so it's a great place to transpose. Alternatively we could transpose into scratch1.
        let temp_transpose = &mut into_real_mut(output)[..input.len()];
        
        transpose::transpose(&input, temp_transpose, self.inner_fft_len, self.inner_r2c_len);

        // Step 2: Compute the inner R2C FFTs
        self.inner_r2c.process(temp_transpose, scratch1, &mut []);

        // Step 3: Apply twiddle factors
        for (element, twiddle) in scratch1.iter_mut().zip(self.twiddles.iter()) {
            *element = *element * *twiddle;
        }

        // Step 4: Transpose
        transpose::transpose(&scratch1, scratch2, self.inner_r2c_complex_len, self.inner_fft_len);

        // Step 5: Compute the inner full-complex FFT of size width. Even though we're computing full-complex FFTs,
        // we're still only doing half the work, because we only need to compute half of them! The other half, per the definition of the R2C, will contain transposed redundant data
        self.inner_fft.process_with_scratch(scratch2, scratch1);

        // step 6: transpose. Slightly different than the normal transpose, since we have to work around the fact that some of the data is missing
        self.transpose_r2c_final_small(scratch2, output);
    }

    #[allow(unused)]
    #[inline(never)]
    fn transpose_r2c_final_small(&self, source: &[Complex<T>], destination: &mut [Complex<T>]) {
        let forward_column_count = self.inner_r2c_complex_len;
        let reverse_column_count = self.inner_r2c_len - self.inner_r2c_complex_len;

        let column_stride = self.inner_fft_len;

        let mut indexes_gather = vec![0; destination.len()];
        let mut indexes_scatter = vec![0; source.len()];

        // The easy part: the forward data is just a standard transpose
        let mut chunks_iter = destination.chunks_exact_mut(self.inner_r2c_len);
        for (y, chunk) in chunks_iter.by_ref().enumerate() {
            for (x, destination_element) in (&mut chunk[..forward_column_count]).iter_mut().enumerate() {
                *destination_element = source[x * column_stride + y];
                indexes_gather[y * self.inner_r2c_len + x] = (x * column_stride + y) as isize;
                indexes_scatter[x * column_stride + y] = (y * self.inner_r2c_len + x) as isize;
            }
        }
        // We'll have one remainder row at the end of the short section that can be processed entirely with the forward logic
        for (x, destination_element) in chunks_iter.into_remainder().iter_mut().enumerate() {
            *destination_element = source[x * column_stride + column_stride / 2];
            indexes_gather[column_stride / 2 * self.inner_r2c_len + x] = (x * column_stride + column_stride / 2) as isize;
            indexes_scatter[x * column_stride + column_stride / 2] = (column_stride / 2 * self.inner_r2c_len + x) as isize;
        }

        // the hard part: the "reverse data", where we have to pull data from a weird part of the array and conjugate it
        // we're making up for the fact that we're trying to pull data from a part of the array that we didn't compute
        // Because we're computing a R2C, all of that missing data is available in another spot, conjugated
        if reverse_column_count > 0 {
            let reverse_base = self.len() - 1 + column_stride;
            for (y, chunk) in destination.chunks_exact_mut(self.inner_r2c_len).enumerate() {
                for (x, destination_element) in (forward_column_count..).zip((&mut chunk[forward_column_count..]).iter_mut()) {
                    *destination_element = source[reverse_base - x * column_stride - y].conj();
                    indexes_gather[y * self.inner_r2c_len + x] = -((reverse_base - x * column_stride - y) as isize);
                    indexes_scatter[reverse_base - x * column_stride - y] = -((y * self.inner_r2c_len + x) as isize);
                }
            }
        }

        println!();
        println!("len={}, r2c len={}, r2c complex len={}, fft len={}", self.len, self.inner_r2c_len, self.inner_r2c_complex_len, self.inner_fft_len);
        println!("Gather indexes: ");
        for chunk in indexes_gather.chunks(self.inner_r2c_len) {
            for e in chunk.iter() {
                print!("{:>4}", e);
            }
            println!();
        }

        println!("Scatter indexes: ");
        for chunk in indexes_scatter.chunks(self.inner_fft_len) {
            for e in chunk.iter() {
                print!("{:>4}", e);
            }
            println!();
        }
    }

    #[allow(unused)]
    #[inline(never)]
    fn transpose_r2c_final(&self, source: &[Complex<T>], destination: &mut [Complex<T>]) {
        let forward_column_count = self.inner_r2c_complex_len;
        let reverse_column_count = self.inner_r2c_len - self.inner_r2c_complex_len;
        let bare_columns = forward_column_count - reverse_column_count;
        
        let column_stride = self.inner_fft_len;
        let rev_offset = if bare_columns % 2 == 1 { 0 } else { column_stride };

        // we will have a small remainder chunk at the end that will need special handling - slice it off now
        let remainder_len = destination.len() % self.inner_r2c_len;
        let (destination, destination_remainder) = destination.split_at_mut(destination.len() - remainder_len);
        let tile_count = destination.len() / (self.inner_r2c_len * TILE_SIZE);

        // The easy part: the forward data is just a standard transpose
        // we're going to iterate over multiple rows at once in order to implement "loop tiling", which is a cache-friendliness improvement to transposing
        // Intentionally use chunks_mut() instead of chunks_exact_mut() so that we get the final 
        const TILE_SIZE : usize = 8;
        let mut tiles_iter = destination.chunks_exact_mut(self.inner_r2c_len * TILE_SIZE);
        for (tile_index, tile) in tiles_iter.by_ref().enumerate() {
            for x in 0..forward_column_count {
                for y in 0..TILE_SIZE {
                    unsafe { *tile.get_unchecked_mut(x + y * self.inner_r2c_len) = source[x * column_stride + tile_index * TILE_SIZE + y] };
                }
            }
        }

        // The last section of loop tiling - not big enough for an entire tile
        let final_tile = tiles_iter.into_remainder();
        for x in 0..forward_column_count {
            for (y, chunk) in final_tile.chunks_exact_mut(self.inner_r2c_len).enumerate() {
                chunk[x] = source[x * column_stride + tile_count + y];
            }
        }

        // We'll have one remainder row at the end of the short section that can be processed entirely with the forward logic
        for (x, destination_element) in destination_remainder.iter_mut().enumerate() {
            *destination_element = source[x * column_stride + column_stride / 2];
        }

        // the hard part: the "reverse data", where we have to pull data from a weird part of the array and conjugate it
        // we're making up for the fact that we're trying to pull data from a part of the array that we didn't compute
        // Because we're computing a R2C, all of that missing data is available in another spot, conjugated
        if reverse_column_count > 0 {
            let reverse_base = self.len() - 1 + column_stride;
            for (y, chunk) in destination.chunks_exact_mut(self.inner_r2c_len).enumerate() {
                for (x, destination_element) in (forward_column_count..).zip((&mut chunk[forward_column_count..]).iter_mut()) {
                    *destination_element = source[reverse_base - x * column_stride - y].conj();
                }
            }
        }
    }
}
impl<T: FftNum> FftRealToComplex<T> for MixedRadixR2C<T> {
    fn process(&self, input: &mut [T], output: &mut [Complex<T>], scratch: &mut [T]) {
        if self.len() == 0 {
            return;
        }

        let output_blocksize = self.len() / 2 + 1;

        let required_scratch = self.get_scratch_len();
        if scratch.len() < required_scratch
            || input.len() < self.len()
            || output.len() < output_blocksize
        {
            // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
            r2c_error_outofplace(
                self.len(),
                input.len(),
                output.len(),
                self.get_scratch_len(),
                scratch.len(),
            );
            return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
        }

        let scratch = &mut scratch[..required_scratch];
        let result = array_utils::iter_chunks_zipped_r2c(
            input,
            self.len(),
            output,
            output_blocksize,
            |in_chunk, out_chunk| {
                self.perform_r2c(in_chunk, out_chunk, scratch)
            },
        );

        if result.is_err() {
            // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
            // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
            r2c_error_outofplace(
                self.len(),
                input.len(),
                output.len(),
                self.get_scratch_len(),
                scratch.len(),
            );
        }
    }

    fn get_scratch_len(&self) -> usize {
        4 * self.inner_fft_len * self.inner_r2c_complex_len
    }
}
impl<T> Length for MixedRadixR2C<T> {
    fn len(&self) -> usize {
        self.len
    }
}

pub struct DftR2C<T> {
    twiddles: Box<[Complex<T>]>,
}
impl<T: FftNum> DftR2C<T> {
    pub fn new(len: usize, direction: FftDirection) -> Self {
        Self {
            twiddles: (0..len).map(|i| twiddles::compute_twiddle(i, len, direction)).collect()
        }
    }

    fn perform_r2c(&self, input: &mut [T], output: &mut [Complex<T>], _: &mut [T]) {
        assert_eq!(input.len(), self.len());
        for i in 0..output.len() {
            let mut output_value = Zero::zero();
            for k in 0..input.len() {
                output_value = output_value + self.twiddles[(i*k)%self.len()] * input[k];
            }
            output[i] = output_value;
        }
    }
}
impl<T: FftNum> FftRealToComplex<T> for DftR2C<T> {
    fn process(&self, input: &mut [T], output: &mut [Complex<T>], scratch: &mut [T]) {
        if self.len() == 0 {
            return;
        }

        let output_blocksize = self.len() / 2 + 1;

        let required_scratch = self.get_scratch_len();
        if scratch.len() < required_scratch
            || input.len() < self.len()
            || output.len() < output_blocksize
        {
            // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
            r2c_error_outofplace(
                self.len(),
                input.len(),
                output.len(),
                self.get_scratch_len(),
                scratch.len(),
            );
            return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
        }

        let scratch = &mut scratch[..required_scratch];
        let result = array_utils::iter_chunks_zipped_r2c(
            input,
            self.len(),
            output,
            output_blocksize,
            |in_chunk, out_chunk| {
                self.perform_r2c(in_chunk, out_chunk, scratch)
            },
        );

        if result.is_err() {
            // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
            // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
            r2c_error_outofplace(
                self.len(),
                input.len(),
                output.len(),
                self.get_scratch_len(),
                scratch.len(),
            );
        }
    }

    fn get_scratch_len(&self) -> usize {
        0
    }
}
impl<T> Length for DftR2C<T> {
    fn len(&self) -> usize {
        self.twiddles.len()
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
    use num_traits::Zero;

    #[test]
    fn test_r2c_mixedradix_scalar() {
        for width in 1..10 {
            for height in 1..10 {
                test_r2c_mixedradix(width, height, FftDirection::Forward);

                // Note: Even though R2C is usually used with a forward FFT, it was pretty trivial to make it support inverse FFTs.
                // If there's a compelling performance reason to drop inverse support, go ahead.
                //test_r2c_even_with_inner(inner_len, FftDirection::Inverse);
            }
        }
    }

    fn test_r2c_mixedradix(width: usize, height: usize, direction: FftDirection) {
        let inner_r2c: Arc<DftR2C<f64>> = Arc::new(DftR2C::new(height, direction));
        let inner_fft: Arc<Dft<f64>> = Arc::new(Dft::new(width, direction));
        let fft = MixedRadixR2C::new(inner_r2c, inner_fft);

        let control = DftR2C::new(width * height, direction);

        let mut real_input = random_real_signal(control.len());
        let mut control_input = random_real_signal(control.len());

        let mut real_output = vec![Complex::zero(); control.len()/2 + 1];
        let mut control_output = vec![Complex::zero(); control.len()/2 + 1];

        control.process(&mut control_input, &mut control_output, &mut []);

        let mut scratch = vec![0.0; fft.get_scratch_len()];
        fft.process(&mut real_input, &mut real_output, &mut scratch);

        assert!(
            compare_vectors(&real_output, &control_output),
            "process() failed, inner_fft_len = {}, inner_r2c_len = {}, direction = {}",
            width, height,
            direction,
        );
    }

}