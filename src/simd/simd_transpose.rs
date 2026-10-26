use num_complex::Complex;

use crate::simd::simd_array::{SimdComplexArray, SimdComplexArrayMut};
use crate::simd::simd_vector::SimdVector;

/// Transpose UNROLL_FACTOR_X columns from input, starting at `base_x`, to rows in the output, combined with reversing column indexes using `reversed_columns` as a lookup table
/// Safety:
///     - width * height == input.len() == output.len()
//      - reversed_columns.len() == width
//      - every element of reversed_columns < width
///     - base_x + UNROLL_FACTOR_X <= width
///     - base_y + UNROLL_FACTOR_Y <= height
#[inline(always)]
unsafe fn table_transpose_tile<
    V: SimdVector,
    const UNROLL_FACTOR_X: usize,
    const UNROLL_FACTOR_Y: usize,
>(
    height: usize,
    width: usize,
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
    base_x: usize,
    base_y: usize,
    reversed_columns: &[usize],
) {
    for x in base_x..(base_x + UNROLL_FACTOR_X) {
        let x_rev = unsafe { *reversed_columns.get_unchecked(x) };
        for y in base_y..(base_y + UNROLL_FACTOR_Y) {
            let input_index = y * width + x;
            let value: V = unsafe { input.load(input_index) };

            let output_index = y + x_rev * height;
            unsafe {
                output.store(value, output_index);
            }
        }
    }
}

/// Transpose UNROLL_FACTOR_X columns from input, starting at `base_x`, to rows in the output, combined with reversing column indexes using `reversed_columns` as a lookup table
/// Safety:
///     - width * height == input.len() == output.len()
///     - base_x + UNROLL_FACTOR_X <= reversed_columns.len()
#[inline(always)]
unsafe fn table_transpose_strip<
    V: SimdVector,
    const UNROLL_FACTOR_X: usize,
    const UNROLL_FACTOR_Y: usize,
>(
    height: usize,
    width: usize,
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
    base_x: usize,
    reversed_columns: &[usize],
) {
    let height_remainder = height % UNROLL_FACTOR_Y;
    match height_remainder {
        3 => unsafe {
            table_transpose_tile::<V, UNROLL_FACTOR_X, 3>(
                height,
                width,
                input,
                output,
                base_x,
                0,
                reversed_columns,
            )
        },
        2 => unsafe {
            table_transpose_tile::<V, UNROLL_FACTOR_X, 2>(
                height,
                width,
                input,
                output,
                base_x,
                0,
                reversed_columns,
            )
        },
        1 => unsafe {
            table_transpose_tile::<V, UNROLL_FACTOR_X, 1>(
                height,
                width,
                input,
                output,
                base_x,
                0,
                reversed_columns,
            )
        },
        0 => {}
        _ => unreachable!(),
    }

    let mut base_y = height_remainder;
    while base_y < height {
        unsafe {
            table_transpose_tile::<V, UNROLL_FACTOR_X, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                base_x,
                base_y,
                reversed_columns,
            )
        };
        base_y += UNROLL_FACTOR_Y;
    }
}

/// `factor_transpose` with the reversed column indices looked up instead of recomputed.
///
/// `reversed_columns[x]` is the output column of input column `x`, so its length is the width and
/// nothing here needs to divide. Every entry must be below the width, which `SimdRadixN::new`
/// asserts, and `D` must divide the width.
#[target_feature(enable = "simd128")]
pub unsafe fn table_transpose<
    V: SimdVector,
    const UNROLL_FACTOR: usize,
    const UNROLL_FACTOR_Y: usize,
>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    assert!(input.len() == width * height && output.len() == input.len());

    let width_remainder = width % UNROLL_FACTOR;
    match width_remainder {
        7 => unsafe {
            table_transpose_strip::<V, 7, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                0,
                reversed_columns,
            )
        },
        6 => unsafe {
            table_transpose_strip::<V, 6, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                0,
                reversed_columns,
            )
        },
        5 => unsafe {
            table_transpose_strip::<V, 5, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                0,
                reversed_columns,
            )
        },
        4 => unsafe {
            table_transpose_strip::<V, 4, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                0,
                reversed_columns,
            )
        },
        3 => unsafe {
            table_transpose_strip::<V, 3, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                0,
                reversed_columns,
            )
        },
        2 => unsafe {
            table_transpose_strip::<V, 2, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                0,
                reversed_columns,
            )
        },
        1 => unsafe {
            table_transpose_strip::<V, 1, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                0,
                reversed_columns,
            )
        },
        0 => {}

        // No idea why the compiler can't see this is impossible
        _ => unreachable!(),
    }

    let mut base_x = width_remainder;
    while base_x < width {
        unsafe {
            table_transpose_strip::<V, UNROLL_FACTOR, UNROLL_FACTOR_Y>(
                height,
                width,
                input,
                output,
                base_x,
                reversed_columns,
            )
        };
        base_x += UNROLL_FACTOR;
    }
}

#[target_feature(enable = "simd128")]
pub unsafe fn full_bitreverse<V: SimdVector>(
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
    permutation: &[usize],
) {
    for (output_element, input_index) in output.iter_mut().zip(permutation.iter()) {
        *output_element = *unsafe {
            input.get_unchecked(input_index / std::mem::size_of::<Complex<V::ScalarType>>())
        };
    }
}

#[target_feature(enable = "simd128")]
pub unsafe fn naive_transpose_height<V: SimdVector>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    assert!(input.len() == width * height && output.len() == input.len());

    for (x_fwd, &x_rev) in reversed_columns.iter().enumerate() {
        for y in 0..height {
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
            let value = unsafe { *input.get_unchecked(y * width + x_fwd) };

            // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
            // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
            // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
            // largest index is `height - 1 + (width - 1) * height`, which is
            // `width * height - 1`.
            unsafe {
                *output.get_unchecked_mut(y + x_rev * height) = value;
            }
        }
    }
}

#[target_feature(enable = "simd128")]
pub unsafe fn naive_transpose_width<V: SimdVector>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    assert!(input.len() == width * height && output.len() == input.len());

    for y in 0..height {
        for (x_fwd, &x_rev) in reversed_columns.iter().enumerate() {
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
            let value = *unsafe { input.get_unchecked(y * width + x_fwd) };

            // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
            // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
            // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
            // largest index is `height - 1 + (width - 1) * height`, which is
            // `width * height - 1`.
            *unsafe { output.get_unchecked_mut(y + x_rev * height) } = value;
        }
    }
}

#[target_feature(enable = "simd128")]
pub unsafe fn gather_transpose_height<V: SimdVector>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    assert!(input.len() == width * height && output.len() == input.len());

    for (x_rev, &x_fwd) in reversed_columns.iter().enumerate() {
        for y in 0..height {
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
            let value = unsafe { *input.get_unchecked(y * width + x_fwd) };

            // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
            // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
            // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
            // largest index is `height - 1 + (width - 1) * height`, which is
            // `width * height - 1`.
            unsafe {
                *output.get_unchecked_mut(y + x_rev * height) = value;
            }
        }
    }
}

#[target_feature(enable = "simd128")]
pub unsafe fn gather_transpose_width<V: SimdVector>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    assert!(input.len() == width * height && output.len() == input.len());

    for y in 0..height {
        for (x_rev, &x_fwd) in reversed_columns.iter().enumerate() {
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
            let value = *unsafe { input.get_unchecked(y * width + x_fwd) };

            // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
            // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
            // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
            // largest index is `height - 1 + (width - 1) * height`, which is
            // `width * height - 1`.
            *unsafe { output.get_unchecked_mut(y + x_rev * height) } = value;
        }
    }
}

#[target_feature(enable = "simd128")]
pub unsafe fn swapped_naive_transpose_width<V: SimdVector, const D: usize>(
    reversed_rows: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = D;
    let height = reversed_rows.len();
    assert!(input.len() == width * height && output.len() == input.len());

    for (y_fwd, &y_rev) in reversed_rows.iter().enumerate() {
        for x in 0..D {
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
            let value = unsafe { *input.get_unchecked(y_fwd * width + x) };

            // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
            // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
            // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
            // largest index is `height - 1 + (width - 1) * height`, which is
            // `width * height - 1`.
            unsafe {
                *output.get_unchecked_mut(y_rev + x * height) = value;
            }
        }
    }
}

#[inline(always)]
unsafe fn transpose_tile<V: SimdVector, const UNROLL_X: usize, const UNROLL_Y: usize>(
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
    width: usize,
    height: usize,
    y_rev: &[usize; UNROLL_Y],
    base_y: usize,
) {
    for x in 0..UNROLL_X {
        for i in 0..UNROLL_Y {
            let y = base_y + i;

            let value = unsafe { *input.get_unchecked(y_rev[i] * width + x) };
            *unsafe { output.get_unchecked_mut(y + x * height) } = value;
        }
    }
}

#[target_feature(enable = "simd128")]
pub unsafe fn swapped_gather_transpose_width<V: SimdVector, const UNROLL_X: usize>(
    reversed_rows: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = UNROLL_X;
    let height = reversed_rows.len();
    assert!(input.len() == width * height && output.len() == input.len());

    const UNROLL_Y: usize = 4;
    let (chunks, remainder) = reversed_rows.as_chunks::<UNROLL_Y>();

    for (i, y_rev) in chunks.iter().enumerate() {
        unsafe {
            transpose_tile::<V, UNROLL_X, UNROLL_Y>(
                input,
                output,
                width,
                height,
                y_rev,
                i * UNROLL_Y,
            )
        };
    }
    match remainder.len() {
        3 => unsafe {
            transpose_tile::<V, UNROLL_X, 3>(
                input,
                output,
                width,
                height,
                remainder.try_into().unwrap(),
                chunks.len() * UNROLL_Y,
            )
        },
        2 => unsafe {
            transpose_tile::<V, UNROLL_X, 2>(
                input,
                output,
                width,
                height,
                remainder.try_into().unwrap(),
                chunks.len() * UNROLL_Y,
            )
        },
        1 => unsafe {
            transpose_tile::<V, UNROLL_X, 1>(
                input,
                output,
                width,
                height,
                remainder.try_into().unwrap(),
                chunks.len() * UNROLL_Y,
            )
        },
        0 => {}
        _ => unreachable!(),
    }
}

/// `factor_transpose` with the reversed column indices looked up instead of recomputed.
///
/// `reversed_columns[x]` is the output column of input column `x`, so its length is the width and
/// nothing here needs to divide. Every entry must be below the width, which `SimdRadixN::new`
/// asserts, and `D` must divide the width.
#[target_feature(enable = "simd128")]
pub unsafe fn table_transpose_original<T: Copy, const D: usize>(
    height: usize,
    reversed_columns: &[usize],
    input: &[T],
    output: &mut [T],
) {
    let width = reversed_columns.len();
    assert!(width % D == 0 && input.len() == width * height && output.len() == input.len());

    for (group, rev) in reversed_columns.chunks_exact(D).enumerate() {
        let x = group * D;
        let rev: &[usize; D] = rev.try_into().unwrap();
        for y in 0..height {
            let row = x + y * width;
            for (i, &r) in rev.iter().enumerate() {
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
                let value = unsafe { *input.get_unchecked(row + i) };

                // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
                // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
                // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
                // largest index is `height - 1 + (width - 1) * height`, which is
                // `width * height - 1`.
                unsafe {
                    *output.get_unchecked_mut(y + r * height) = value;
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn transpose_original_tile<V: SimdVector, const UNROLL_X: usize, const UNROLL_Y: usize>(
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
    rev: &[usize; UNROLL_X],
    width: usize,
    height: usize,
    base_x: usize,
    base_y: usize,
) {
    for (i, &x_rev) in rev.iter().enumerate() {
        for j in 0..UNROLL_Y {
            let x = base_x + i;
            let y = base_y + j;

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
            let value = unsafe { *input.get_unchecked(x + y * width) };

            // The write is to `y + r * height`. `r` comes out of `reversed_columns`, and
            // `SimdRadixN::new` asserts that every entry there is below the width, so `r`
            // reaches at most `width - 1`. With `y` capped at `height - 1` as above, the
            // largest index is `height - 1 + (width - 1) * height`, which is
            // `width * height - 1`.
            unsafe {
                *output.get_unchecked_mut(y + x_rev * height) = value;
            }
        }
    }
}

/// `factor_transpose` with the reversed column indices looked up instead of recomputed.
///
/// `reversed_columns[x]` is the output column of input column `x`, so its length is the width and
/// nothing here needs to divide. Every entry must be below the width, which `SimdRadixN::new`
/// asserts, and `D` must divide the width.
#[target_feature(enable = "simd128")]
pub unsafe fn table_transpose_original_tiled<V: SimdVector, const UNROLL_X: usize>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    let (reversed_column_chunks, remainder_x) = reversed_columns.as_chunks::<UNROLL_X>();
    assert!(remainder_x.is_empty() && input.len() == width * height && output.len() == input.len());

    const UNROLL_Y: usize = 4;
    let (chunks_y, remainder_y) = (height / UNROLL_Y, height % UNROLL_Y);

    for (group_x, rev) in reversed_column_chunks.iter().enumerate() {
        let base_x = group_x * UNROLL_X;
        for group_y in 0..chunks_y {
            let base_y = group_y * UNROLL_Y;
            unsafe {
                transpose_original_tile::<V, UNROLL_X, UNROLL_Y>(
                    input, output, rev, width, height, base_x, base_y,
                )
            };
        }
        match remainder_y {
            3 => unsafe {
                transpose_original_tile::<V, UNROLL_X, 3>(
                    input,
                    output,
                    rev,
                    width,
                    height,
                    base_x,
                    chunks_y * UNROLL_Y,
                )
            },
            2 => unsafe {
                transpose_original_tile::<V, UNROLL_X, 2>(
                    input,
                    output,
                    rev,
                    width,
                    height,
                    base_x,
                    chunks_y * UNROLL_Y,
                )
            },
            1 => unsafe {
                transpose_original_tile::<V, UNROLL_X, 1>(
                    input,
                    output,
                    rev,
                    width,
                    height,
                    base_x,
                    chunks_y * UNROLL_Y,
                )
            },
            0 => {}
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn transpose_original_tile_simd_1complex<
    V: SimdVector,
    const UNROLL_X: usize,
    const UNROLL_Y: usize,
>(
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
    rev: &[usize; UNROLL_X],
    width: usize,
    height: usize,
    base_x: usize,
    base_y: usize,
) {
    for i in 0..UNROLL_X {
        let chunk_x = i * V::COMPLEX_PER_VECTOR;
        let x_fwd = chunk_x + base_x;
        let x_rev = rev[chunk_x];

        for j in 0..UNROLL_Y {
            let y = base_y + j;
            let value: V = unsafe { input.load(x_fwd + y * width) };
            unsafe {
                output.store(value, y + x_rev * height);
            }
        }
    }
}

#[inline(always)]
unsafe fn transpose_original_tile_simd_2complex<
    V: SimdVector,
    const UNROLL_X: usize,
    const UNROLL_Y: usize,
>(
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
    rev: &[usize; UNROLL_X],
    width: usize,
    height: usize,
    base_x: usize,
    base_y: usize,
) {
    // We're assuming that we can get each y in at most one vector, so if we can't it's going to break. If const blocks were stable we could make this a static assert.
    assert!(UNROLL_Y <= V::COMPLEX_PER_VECTOR);

    let (vectors_x, remainder_x) = (
        UNROLL_X / V::COMPLEX_PER_VECTOR,
        UNROLL_X % V::COMPLEX_PER_VECTOR,
    );

    for i in 0..vectors_x {
        let chunk_x = i * V::COMPLEX_PER_VECTOR;

        let mut values = [unsafe { V::zero_vector() }; 2];
        for v in 0..UNROLL_Y {
            values[v] = unsafe { input.load(chunk_x + base_x + (base_y + v) * width) };
        }
        let values = unsafe { V::transpose_square(values) };

        for v in 0..V::COMPLEX_PER_VECTOR {
            match UNROLL_Y {
                2 => unsafe { output.store(values[v], base_y + rev[chunk_x + v] * height) },
                1 => unsafe { output.store1_lo(values[v], base_y + rev[chunk_x + v] * height) },
                _ => unreachable!(),
            }
        }
    }
    if remainder_x == 1 {
        let chunk_x = vectors_x * V::COMPLEX_PER_VECTOR;

        match UNROLL_Y {
            2 => {
                let mut values = [unsafe { V::zero_vector() }; 2];
                for v in 0..UNROLL_Y {
                    values[v] = unsafe { input.load1_lo(chunk_x + base_x + (base_y + v) * width) };
                }
                let values = unsafe { V::transpose_square(values) };

                unsafe { output.store(values[0], base_y + rev[chunk_x] * height) };
            }
            1 => {
                let value: V = unsafe { input.load1_lo(chunk_x + base_x + base_y * width) };
                unsafe { output.store1_lo(value, base_y + rev[chunk_x] * height) };
            }
            _ => unreachable!(),
        }
    }
}

/// `factor_transpose` with the reversed column indices looked up instead of recomputed.
///
/// `reversed_columns[x]` is the output column of input column `x`, so its length is the width and
/// nothing here needs to divide. Every entry must be below the width, which `SimdRadixN::new`
/// asserts, and `D` must divide the width.
#[target_feature(enable = "simd128")]
pub unsafe fn table_transpose_original_tiled_simd<V: SimdVector, const UNROLL_X: usize>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    let (reversed_column_chunks, remainder_x) = reversed_columns.as_chunks::<UNROLL_X>();
    assert!(remainder_x.is_empty() && input.len() == width * height && output.len() == input.len());

    // I'd greatly prefer to have this all be one generic loop, but rustc doesn't support using V::COMPLEX_PER_VECTOR as a generic parameter to the inner function
    match V::COMPLEX_PER_VECTOR {
        1 => {
            for (group_x, rev) in reversed_column_chunks.iter().enumerate() {
                let base_x = group_x * UNROLL_X;
                for y in 0..height {
                    unsafe {
                        transpose_original_tile_simd_1complex::<V, UNROLL_X, 1>(
                            input, output, rev, width, height, base_x, y,
                        )
                    };
                }
            }
        }
        2 => {
            const UNROLL_Y: usize = 2;
            let (chunks_y, remainder_y) = (height / UNROLL_Y, height % UNROLL_Y);

            for (group_x, rev) in reversed_column_chunks.iter().enumerate() {
                let base_x = group_x * UNROLL_X;
                let mut base_y = 0;
                while base_y < chunks_y * UNROLL_Y {
                    unsafe {
                        transpose_original_tile_simd_2complex::<V, UNROLL_X, UNROLL_Y>(
                            input, output, rev, width, height, base_x, base_y,
                        )
                    };
                    base_y += UNROLL_Y;
                }
                match remainder_y {
                    1 => unsafe {
                        transpose_original_tile_simd_2complex::<V, UNROLL_X, 1>(
                            input,
                            output,
                            rev,
                            width,
                            height,
                            base_x,
                            chunks_y * UNROLL_Y,
                        )
                    },
                    0 => {}
                    _ => unreachable!(),
                }
            }
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
unsafe fn simd_transpose_strip_1complex<V: SimdVector, const UNROLL_X: usize>(
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
    rev: &[usize; UNROLL_X],
    width: usize,
    height: usize,
    base_x: usize,
) {
    for y in 0..height {
        for (i, &x_rev) in rev.iter().enumerate() {
            let value = unsafe { *input.get_unchecked(y * width + base_x + i) };
            unsafe {
                *output.get_unchecked_mut(y + x_rev * height) = value;
            }
        }
    }
}

#[inline(always)]
unsafe fn simd_transpose_1complex<V: SimdVector>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    assert!(input.len() == width * height && output.len() == input.len());

    const UNROLL_X: usize = 4;
    let (reverse_chunks, reverse_remainder) = reversed_columns.as_chunks::<UNROLL_X>();

    for (group, rev) in reverse_chunks.iter().enumerate() {
        unsafe {
            simd_transpose_strip_1complex::<V, UNROLL_X>(
                input,
                output,
                rev,
                width,
                height,
                group * UNROLL_X,
            )
        };
    }
    match reverse_remainder.len() {
        3 => unsafe {
            simd_transpose_strip_1complex::<V, 3>(
                input,
                output,
                reverse_remainder.try_into().unwrap(),
                width,
                height,
                reverse_chunks.len() * UNROLL_X,
            )
        },
        2 => unsafe {
            simd_transpose_strip_1complex::<V, 2>(
                input,
                output,
                reverse_remainder.try_into().unwrap(),
                width,
                height,
                reverse_chunks.len() * UNROLL_X,
            )
        },
        1 => unsafe {
            simd_transpose_strip_1complex::<V, 1>(
                input,
                output,
                reverse_remainder.try_into().unwrap(),
                width,
                height,
                reverse_chunks.len() * UNROLL_X,
            )
        },
        0 => {}
        _ => unreachable!(),
    }
}

#[inline(always)]
unsafe fn simd_transpose_2complex_4cols<V: SimdVector>(
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
    rev: &[usize; 4],
    width: usize,
    height: usize,
    base_x: usize,
) {
    const UNROLL_Y: usize = 2;
    let (chunks_y, remainder_y) = (height / UNROLL_Y, height % UNROLL_Y);

    for j in 0..chunks_y {
        unsafe {
            let base_y = j * UNROLL_Y;
            let values = [
                input.load((base_y + 0) * width + base_x + 0),
                input.load((base_y + 0) * width + base_x + 2),
                input.load((base_y + 1) * width + base_x + 0),
                input.load((base_y + 1) * width + base_x + 2),
            ];

            let transposed = V::transpose_contiguous_to_strided(values);

            output.store(transposed[0], base_y + rev[0] * height);
            output.store(transposed[1], base_y + rev[1] * height);
            output.store(transposed[2], base_y + rev[2] * height);
            output.store(transposed[3], base_y + rev[3] * height);
        }
    }
    if remainder_y == 1 {
        unsafe {
            let base_y = chunks_y * UNROLL_Y;
            let values = [
                input.load((base_y + 0) * width + base_x + 0),
                input.load((base_y + 0) * width + base_x + 2),
                V::zero_vector(),
                V::zero_vector(),
            ];

            let transposed = V::transpose_contiguous_to_strided(values);

            output.store1_lo(transposed[0], base_y + rev[0] * height);
            output.store1_lo(transposed[1], base_y + rev[1] * height);
            output.store1_lo(transposed[2], base_y + rev[2] * height);
            output.store1_lo(transposed[3], base_y + rev[3] * height);
        }
    }
}

#[inline(always)]
unsafe fn simd_transpose_2complex_3cols<V: SimdVector>(
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
    rev: &[usize; 3],
    width: usize,
    height: usize,
    base_x: usize,
) {
    const UNROLL_Y: usize = 2;
    let (chunks_y, remainder_y) = (height / UNROLL_Y, height % UNROLL_Y);

    for j in 0..chunks_y {
        unsafe {
            let base_y = j * UNROLL_Y;
            let values = [
                input.load((base_y + 0) * width + base_x + 0),
                input.load1_lo((base_y + 0) * width + base_x + 2),
                input.load((base_y + 1) * width + base_x + 0),
                input.load1_lo((base_y + 1) * width + base_x + 2),
            ];

            let transposed = V::transpose_contiguous_to_strided(values);

            output.store(transposed[0], base_y + rev[0] * height);
            output.store(transposed[1], base_y + rev[1] * height);
            output.store(transposed[2], base_y + rev[2] * height);
        }
    }
    if remainder_y == 1 {
        unsafe {
            let base_y = chunks_y * UNROLL_Y;
            let values = [
                input.load((base_y + 0) * width + base_x + 0),
                input.load1_lo((base_y + 0) * width + base_x + 2),
                V::zero_vector(),
                V::zero_vector(),
            ];

            let transposed = V::transpose_contiguous_to_strided(values);

            output.store1_lo(transposed[0], base_y + rev[0] * height);
            output.store1_lo(transposed[1], base_y + rev[1] * height);
            output.store1_lo(transposed[2], base_y + rev[2] * height);
        }
    }
}

#[inline(always)]
unsafe fn simd_transpose_2complex_2cols<V: SimdVector>(
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
    rev: &[usize; 2],
    width: usize,
    height: usize,
    base_x: usize,
) {
    const UNROLL_Y: usize = 2;
    let (chunks_y, remainder_y) = (height / UNROLL_Y, height % UNROLL_Y);

    for j in 0..chunks_y {
        unsafe {
            let base_y = j * UNROLL_Y;
            let values = [
                input.load((base_y + 0) * width + base_x + 0),
                input.load((base_y + 1) * width + base_x + 0),
            ];

            let transposed = V::transpose_contiguous_to_strided(values);

            output.store(transposed[0], base_y + rev[0] * height);
            output.store(transposed[1], base_y + rev[1] * height);
        }
    }
    if remainder_y == 1 {
        unsafe {
            let base_y = chunks_y * UNROLL_Y;
            let values = [
                input.load((base_y + 0) * width + base_x + 0),
                V::zero_vector(),
            ];

            let transposed = V::transpose_contiguous_to_strided(values);

            output.store1_lo(transposed[0], base_y + rev[0] * height);
            output.store1_lo(transposed[1], base_y + rev[1] * height);
        }
    }
}

#[inline(always)]
unsafe fn simd_transpose_2complex_1col<V: SimdVector>(
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
    rev: &[usize; 1],
    width: usize,
    height: usize,
    base_x: usize,
) {
    const UNROLL_Y: usize = 2;
    let (chunks_y, remainder_y) = (height / UNROLL_Y, height % UNROLL_Y);

    for j in 0..chunks_y {
        unsafe {
            let base_y = j * UNROLL_Y;
            let values = [
                input.load1_lo((base_y + 0) * width + base_x + 0),
                input.load1_lo((base_y + 1) * width + base_x + 0),
            ];

            let transposed = V::transpose_contiguous_to_strided(values);

            output.store(transposed[0], base_y + rev[0] * height);
        }
    }
    if remainder_y == 1 {
        unsafe {
            let base_y = chunks_y * UNROLL_Y;
            let value: V = input.load1_lo((base_y + 0) * width + base_x + 0);
            output.store1_lo(value, base_y + rev[0] * height);
        }
    }
}

#[inline(always)]
unsafe fn simd_transpose_2complex<V: SimdVector>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    let width = reversed_columns.len();
    assert!(input.len() == width * height && output.len() == input.len());

    const UNROLL_X: usize = 4;
    let (reverse_chunks, reverse_remainder) = reversed_columns.as_chunks::<UNROLL_X>();

    for (group, rev) in reverse_chunks.iter().enumerate() {
        unsafe {
            simd_transpose_2complex_4cols::<V>(input, output, rev, width, height, group * UNROLL_X)
        };
    }
    match reverse_remainder.len() {
        3 => unsafe {
            simd_transpose_2complex_3cols::<V>(
                input,
                output,
                reverse_remainder.try_into().unwrap(),
                width,
                height,
                reverse_chunks.len() * UNROLL_X,
            )
        },
        2 => unsafe {
            simd_transpose_2complex_2cols::<V>(
                input,
                output,
                reverse_remainder.try_into().unwrap(),
                width,
                height,
                reverse_chunks.len() * UNROLL_X,
            )
        },
        1 => unsafe {
            simd_transpose_2complex_1col::<V>(
                input,
                output,
                reverse_remainder.try_into().unwrap(),
                width,
                height,
                reverse_chunks.len() * UNROLL_X,
            )
        },
        0 => {}
        _ => unreachable!(),
    }
}

#[target_feature(enable = "simd128")]
pub unsafe fn simd_transpose<V: SimdVector>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    // It'd be really cool to be able to write this all with just a single const generic set of functions, but rust's type system just isn't there yet
    // For example, you can't use an associated constant of a generic parameter as an array size (IE you can't write [V::zero_vector; V::COMPLEX_PER_VECTOR])
    // Until then it's easiest to just split ut up by COMPLEX_PER_VECTOR
    match V::COMPLEX_PER_VECTOR {
        1 => unsafe { simd_transpose_1complex::<V>(height, reversed_columns, input, output) },
        2 => unsafe { simd_transpose_2complex::<V>(height, reversed_columns, input, output) },
        _ => unreachable!(),
    }
}
