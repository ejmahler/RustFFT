use num::{Complex, FromPrimitive, Signed};

use butterflies::{butterfly_2_single, butterfly_4_single, butterfly_2, butterfly_4};

pub fn process_radix4<T>(size: usize,
                         signal: &[Complex<T>],
                         spectrum: &mut [Complex<T>],
                         stride: usize,
                         twiddles: &[Complex<T>],
                         inverse: bool)
    where T: Signed + FromPrimitive + Copy
{

    prepare_radix4(size, signal, spectrum, stride);

    // first, perform the butterflies. the butterfly size depends on the input size
    let num_bits = (size - 1).count_ones();
    let mut current_size = if num_bits % 2 > 0 {
        // the size is a power of 2, so we need to do size-2 butterflies, with a stride of size / 2
        for chunk in spectrum.chunks_mut(2 * stride) {
            unsafe { butterfly_2_single(chunk, stride) }
        }

        // for the cross-ffts we want to to start off with a size of 8 (2 * 4)
        8
    } else {
        for chunk in spectrum.chunks_mut(4 * stride) {
            unsafe { butterfly_4_single(chunk, stride, inverse) }
        }

        // for the cross-ffts we want to to start off with a size of 16 (4 * 4)
        16
    };

    // now, perform all the cross-FFTs, one "layer" at a time
    while current_size <= size {
        let group_stride = size / current_size;

        for i in 0..group_stride {
            unsafe {
                butterfly_4(&mut spectrum[i * current_size..],
                            group_stride,
                            twiddles,
                            current_size / 4,
                            inverse)
            }
        }
        current_size *= 4;
    }
}

// after testing an iterative bit reversal algorithm, this recursive algorithm
// was almost an order of magnitude faster at setting up
fn prepare_radix4<T: Copy>(size: usize,
                           signal: &[Complex<T>],
                           spectrum: &mut [Complex<T>],
                           stride: usize) {
    match size {
        4 => unsafe {
            for i in 0..4 {
                *spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        },
        2 => unsafe {
            for i in 0..2 {
                *spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        },
        _ => {
            for i in 0..4 {
                prepare_radix4(size / 4,
                               &signal[i * stride..],
                               &mut spectrum[i * (size / 4)..],
                               stride * 4);
            }
        }
    }
}

pub fn process_radix2_inplace<T>(size: usize,
                         spectrum: &mut [Complex<T>],
                         stride: usize,
                         twiddles: &[Complex<T>])
    where T: Signed + FromPrimitive + Copy
{
    // perform the bit reversal swap
    let num_bits = size.trailing_zeros();

    for i in 0..size {
        let swap_index = reverse_bits(i, num_bits);

        if swap_index > i {
            spectrum.swap(i * stride, swap_index * stride);
        }
    }

    // perform the butterflies, with a stride of size / 2
    for chunk in spectrum.chunks_mut(2 * stride) {
        unsafe { butterfly_2_single(chunk, stride) }
    }

    // for the cross-ffts we want to to start off with a size of 4 (2 * 2)
    let mut current_size = 4;

    // now, perform all the cross-FFTs, one "layer" at a time
    while current_size <= size {
        let group_stride = size / current_size;

        for i in 0..group_stride {
            unsafe {
                butterfly_2(&mut spectrum[i * current_size..],
                            group_stride,
                            twiddles,
                            current_size / 2)
            }
        }
        current_size *= 2;
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