use num_complex::Complex;

use crate::simd::simd_vector::SimdVector;

/// Transpose U rows from input to output, using specified reverse and forward x indexes
/// Safety:
///     - input.len() == ooutput.len()
///     - input.len() == width * height
///     - every element of forward_chunk must be < width
///     - every element of reverse_chunk must be < width
#[inline(always)]
unsafe fn table_transpose_strip<V: SimdVector, const U: usize>(
    height: usize,
    width: usize,
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
    forward_chunk: &[usize; U],
    reverse_chunk: &[usize; U],
) {
    for y in 0..height {
        for (&x_fwd, &x_rev) in forward_chunk.into_iter().zip(reverse_chunk.iter()) {
            // Both indexes below are unchecked, so here is why neither can leave the slices.
            // Both slices are `width * height` long, asserted above, so an index is in range
            // as long as it stays below `width * height`.
            //
            // The read is from `row + i`, which is `group * D + y * width + i`. The loops cap
            // each term: `group` reaches `width / D - 1` because `chunks_exact(D)` over a
            // slice of length `width` yields `width / D` chunks, `y` reaches `height - 1`,
            // and `i` reaches `D - 1`. Substituting all three gives
            // `(width / D - 1) * D + (height - 1) * width + D - 1`, and since D divides
            // `width` that simplifies to `width * height - 1`.
            let input_index = y * width + x_fwd;
            let value = unsafe { *input.get_unchecked(input_index) };

            // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
            // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
            // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
            // largest index is `height - 1 + (width - 1) * height`, which is
            // `width * height - 1`.
            let output_index = y + x_rev * height;
            unsafe {
                *output.get_unchecked_mut(output_index) = value;
            }
        }
    }
}

/// `factor_transpose` with the reversed column indices looked up instead of recomputed.
///
/// `reversed_columns[x]` is the output column of input column `x`, so its length is the width and
/// nothing here needs to divide. Every entry must be below the width, which `SimdRadixN::new`
/// asserts, and `D` must divide the width.
#[target_feature(enable = "neon")]
pub fn table_transpose<V: SimdVector>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    assert!(input.len() == width * height && output.len() == input.len());

    const UNROLL_FACTOR: usize = 4;
    let (chunks, remainder) = reversed_columns.as_chunks::<UNROLL_FACTOR>();

    for (tile_x, reversed_chunk) in chunks.iter().enumerate() {
        let forward_chunk = std::array::from_fn(|i| tile_x * UNROLL_FACTOR + i);
        unsafe { table_transpose_strip::<V, UNROLL_FACTOR>(height, width, input, output, &forward_chunk, reversed_chunk) };
    }

    // Process the remainder by doing fewer than UNROLL_FACTOR rows
    let final_tile_x = chunks.len();
    match remainder.len() {
        /*30 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 30>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        29 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 29>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        28 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 28>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        27 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 27>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        26 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 26>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        25 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 25>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        24 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 24>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        23 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 23>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        22 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 22>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        21 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 21>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        20 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 20>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        19 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 19>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        18 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 18>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        17 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 17>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        16 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 16>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        15 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 15>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        14 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 14>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        13 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 13>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        12 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 12>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        11 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 11>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        10 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 10>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        9 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 9>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        8 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 8>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        7 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 7>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        6 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 6>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        5 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 5>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        4 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 4>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },*/
        3 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 3>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        2 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 2>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        1 => {
            let forward_chunk = std::array::from_fn(|i| final_tile_x * UNROLL_FACTOR + i);
            let reverse_chunk = remainder.try_into().unwrap();
            unsafe { table_transpose_strip::<V, 1>(height, width, input, output, &forward_chunk, reverse_chunk) };
        },
        0 => {},

        // No idea why the compiler can't see this is impossible
        _ => unreachable!(),
    }
}
