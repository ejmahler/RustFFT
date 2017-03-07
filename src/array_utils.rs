const BLOCK_SIZE: usize = 16;

use common::verify_length;

#[inline(always)]
unsafe fn transpose_block<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize, block_x: usize, block_y: usize) {
    for inner_x in 0..BLOCK_SIZE {
        for inner_y in 0..BLOCK_SIZE {
            let x = block_x * BLOCK_SIZE + inner_x;
            let y = block_y * BLOCK_SIZE + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

#[inline(always)]
unsafe fn transpose_endcap_block<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize, block_x: usize, block_y: usize, block_width: usize, block_height: usize) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = block_x * BLOCK_SIZE + inner_x;
            let y = block_y * BLOCK_SIZE + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output
// Use "Loop tiling" to improve cache-friendliness
pub fn transpose<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T]) {
    verify_length(input, output, width * height);

    let x_block_count = width / BLOCK_SIZE;
    let y_block_count = height / BLOCK_SIZE;

    let remainder_x = width - x_block_count * BLOCK_SIZE;
    let remainder_y = height - y_block_count * BLOCK_SIZE;

    for y_block in 0..y_block_count {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block(
                    input, output,
                    width, height, 
                    x_block, y_block);
            }
        }

        //if the width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_endcap_block(
                    input, output, 
                    width, height, 
                    x_block_count, y_block, 
                    remainder_x, BLOCK_SIZE);
            }
        }
    }

    //if the height is not cleanly divisible by BLOCK_SIZE, there are still a few columns that haven't been transposed
    if remainder_y > 0 {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_endcap_block(
                    input, output,
                    width, height,
                    x_block, y_block_count,
                    BLOCK_SIZE, remainder_y,
                    );
            }
        }

        //if the width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_endcap_block(
                    input, output,
                    width, height, 
                    x_block_count, y_block_count, 
                    remainder_x, remainder_y);
            }
        }
    } 
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output
/// benchmarking shows that loop tiling isn't effective for small arrays (in the range of 50x50 or smaller)
pub unsafe fn transpose_small<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T]) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::random_signal;
    use num::{Complex, Zero};

    #[test]
    fn test_transpose() {
        let sizes = [1, BLOCK_SIZE - 1, BLOCK_SIZE * 4, BLOCK_SIZE * 4 + 3];

        for &width in &sizes {
            for &height in &sizes {
                let len = width * height;

                let input: Vec<Complex<f32>> = random_signal(len);
                let mut output = vec![Zero::zero(); len];

                transpose(width, height, &input, &mut output);

                for x in 0..width {
                    for y in 0..height {
                        assert_eq!(input[x + y * width], output[y + x * height], "x = {}, y = {}", x, y);
                    }
                }

                unsafe { transpose_small(width, height, &input, &mut output) };

                for x in 0..width {
                    for y in 0..height {
                        assert_eq!(input[x + y * width], output[y + x * height], "x = {}, y = {}", x, y);
                    }
                }

            }
        }
    }
}
