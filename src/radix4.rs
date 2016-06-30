use num::{Complex, FromPrimitive, Signed};

use butterflies::{butterfly_2_single, butterfly_4_single, butterfly_4};

pub fn process_radix4<T>(size: usize,
              signal: &[Complex<T>],
              spectrum: &mut [Complex<T>],
              stride: usize,
              twiddles: &[Complex<T>],
              inverse: bool) where T: Signed + FromPrimitive + Copy {

    let num_bits = (size - 1).count_ones();

    //first, perform the butterflies. the butterfly size depends on the input size
    let mut current_size = if num_bits % 2 == 1 {
        //the size is a power of 2, so we need to do size-2 butterflies, with a stride of size / 2
        for (chunk_index, chunk) in spectrum.chunks_mut(2 * stride).enumerate() {
            //copy the data from the signal into our chunk -- we improve cache friendliness by copying it right when we need it
            for i in 0..2 {
                let signal_index = reverse_bits_radix4(num_bits, chunk_index * 2 + i);
                unsafe { *chunk.get_unchecked_mut(i * stride) =
                    *signal.get_unchecked(signal_index * stride) }
            }

            //perform the butterly on our newly copied data
            unsafe { butterfly_2_single(chunk, stride) }
        }

        //for the cross-ffts we want to to start off with a size of 8 (2 * 4)
        8
    }
    else {
        for (chunk_index, chunk) in spectrum.chunks_mut(4 * stride).enumerate() {
            //copy the data from the signal into our chunk -- we improve cache friendliness by copying it right when we need it
            for i in 0..4 {
                let signal_index = reverse_bits_radix4(num_bits, chunk_index * 4 + i);
                 unsafe { *chunk.get_unchecked_mut(i * stride) =
                    *signal.get_unchecked(signal_index * stride) }
            }

            unsafe { butterfly_4_single(chunk, stride, inverse) }
        }

        //for the cross-ffts we want to to start off with a size of 16 (4 * 4)
        16
    };

    //now, perform all the cross-FFTs, one "layer" at a time
    while current_size <= size {
        let group_stride = size / current_size;

        for i in 0..group_stride {
            unsafe { butterfly_4(&mut spectrum[i * current_size..], group_stride, twiddles, current_size / 4, inverse) }
        }
        current_size *= 4;
    }
}

fn reverse_bits_radix4(num_bits: u32, mut n: usize) -> usize {
    let mut output = 0;

    //for radix 4, we want to reverse the bits of n, in blocks of 2 bits at a time
    //so 110110 becomes 100111

    //for radix 2, num_bits will be odd. in this case, we want to reverse just a single bit in the last iteration
    if num_bits %2 == 1 {
        output |= n & 1;
        n >>= 1;
    }

    //now copy 2 bits at a time from n into 
    for _ in 0..num_bits/2 {
        output <<= 2;
        output |= n & 3;
        n >>= 2;
    }

    output
}

#[cfg(test)]
mod test {
    use super::reverse_bits_radix4;

    #[test]
    fn test_bit_reversal() {
        //first tuple element is num bits, second is expected output
        let test_list = vec![
            (1, vec![0,1]),
            (2, vec![0,1,2,3]),
            (3, vec![0,4,1,5,2,6,3,7]),
            (4, vec![0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]),
            (5, vec![0,16,4,20,8,24,12,28,1,17,5,21,9,25,13,29,   2,18,6,22,10,26,14,30,3,19,7,23,11,27,15,31]),
        ];

        for (num_bits, expected_list) in test_list {

            // verify that the bit reversal function reorders the range 0..size into the expected output
            for (result, e) in
                    (0..expected_list.len())
                    .map(|i| reverse_bits_radix4(num_bits, i))
                    .zip(expected_list.iter()) {
                assert_eq!(result, *e);
            }
        }
    }
}