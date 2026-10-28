use num_complex::Complex;

use crate::simd::simd_array::{SimdComplexArray, SimdComplexArrayMut};
use crate::simd::simd_vector::SimdVector;
use crate::common::RadixFactor;

pub enum TransposeAlgorithm {
    TableTranspose(RadixFactor, Box<[usize]>),
    UnrolledTranspose(RadixFactor),
    Copy,
}

#[inline(always)]
unsafe fn table_transpose_1complex<V: SimdVector, const D: usize>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
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
unsafe fn table_transpose_strip_2complex<V: SimdVector, const UNROLL_X: usize>(
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
    rev: &[usize; UNROLL_X],
    width: usize,
    height: usize,
    base_x: usize,
) {
    const UNROLL_Y: usize = 2;
    let (chunks_y, remainder_y) = (height / UNROLL_Y, height % UNROLL_Y);
    let (chunks_x, remainder_x) = (UNROLL_X / 2, UNROLL_X % 2);

    for iy in 0..chunks_y {
        let y = iy * UNROLL_Y;
        for ix in 0..chunks_x {
            let x = ix * UNROLL_Y;
            unsafe {
                let values = [
                    input.load((y + 0) * width + base_x + x),
                    input.load((y + 1) * width + base_x + x),
                ];

                let transposed = V::transpose_contiguous_to_strided(values);

                output.store(transposed[0], y + rev[x + 0] * height);
                output.store(transposed[1], y + rev[x + 1] * height);
            }
        }
        if remainder_x == 1 {
            let x = chunks_x * UNROLL_Y;
            unsafe {
                let values = [
                    input.load1_lo((y + 0) * width + base_x + x),
                    input.load1_lo((y + 1) * width + base_x + x),
                ];

                let transposed = V::transpose_contiguous_to_strided(values);

                output.store(transposed[0], y + rev[x] * height);
            }
        }
    }
    if remainder_y == 1 {
        let y = chunks_y * UNROLL_Y;
        for ix in 0..chunks_x {
            let x = ix * UNROLL_Y;
            unsafe {
                let values = [
                    input.load(y * width + base_x + x),
                    V::zero_vector(),
                ];

                let transposed = V::transpose_contiguous_to_strided(values);

                output.store1_lo(transposed[0], y + rev[x + 0] * height);
                output.store1_lo(transposed[1], y + rev[x + 1] * height);
            }
        }
        if remainder_x == 1 {
            let x = chunks_x * UNROLL_Y;
            unsafe {
                let value : V = input.load1_lo(y * width + base_x + x);
                output.store1_lo(value, y + rev[x] * height);
            }
        }
    }
}

#[inline(always)]
unsafe fn table_transpose_2complex<V: SimdVector, const EXTERNAL_UNROLL_FACTOR: usize>(
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
            table_transpose_strip_2complex::<V, 4>(input, output, rev, width, height, group * UNROLL_X)
        };
    }

    // For the most part we ignore the external unroll factor for this algorithm.
    // The only thing we use it for is to know if we can skip remainder checks.
    if EXTERNAL_UNROLL_FACTOR != UNROLL_X {
        match reverse_remainder.len() {
            3 => unsafe {
                table_transpose_strip_2complex::<V, 3>(
                    input,
                    output,
                    reverse_remainder.try_into().unwrap(),
                    width,
                    height,
                    reverse_chunks.len() * UNROLL_X,
                )
            },
            2 => unsafe {
                table_transpose_strip_2complex::<V, 2>(
                    input,
                    output,
                    reverse_remainder.try_into().unwrap(),
                    width,
                    height,
                    reverse_chunks.len() * UNROLL_X,
                )
            },
            1 => unsafe {
                table_transpose_strip_2complex::<V, 1>(
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
}

#[inline(always)]
pub unsafe fn table_transpose<V: SimdVector, const EXTERNAL_UNROLL_FACTOR: usize>(
    height: usize,
    reversed_columns: &[usize],
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    // It'd be really cool to be able to write this all with just a single const generic set of functions, but rust's type system just isn't there yet
    // For example, you can't use an associated constant of a generic parameter as an array size (IE you can't write [V::zero_vector; V::COMPLEX_PER_VECTOR])
    // Until then it's easiest to just split ut up by COMPLEX_PER_VECTOR
    match V::COMPLEX_PER_VECTOR {
        1 => unsafe { table_transpose_1complex::<V, EXTERNAL_UNROLL_FACTOR>(height, reversed_columns, input, output) },
        2 => unsafe { table_transpose_2complex::<V, EXTERNAL_UNROLL_FACTOR>(height, reversed_columns, input, output) },
        _ => unreachable!(),
    }
}



#[inline(always)]
unsafe fn unrolled_transpose_1complex<V: SimdVector, const WIDTH: usize>(
    height: usize,
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    assert!(input.len() == WIDTH * height && output.len() == input.len());

    for y in 0..height {
        for x in 0..WIDTH {
            let value = unsafe { *input.get_unchecked(y * WIDTH + x) };
            unsafe {
                *output.get_unchecked_mut(y + x * height) = value;
            }
        }
    }
}


#[inline(always)]
unsafe fn unrolled_transpose_2complex<V: SimdVector, const WIDTH: usize>(
    height: usize,
    input: &[Complex<V::ScalarType>],
    mut output: &mut [Complex<V::ScalarType>],
) {
    assert!(input.len() == WIDTH * height && output.len() == input.len());

    const UNROLL_Y: usize = 2;
    const UNROLL_X: usize = 2;
    let (chunks_y, remainder_y) = (height / UNROLL_Y, height % UNROLL_Y);
    let (chunks_x, remainder_x) = (WIDTH / UNROLL_X, WIDTH % UNROLL_X);

    for iy in 0..chunks_y {
        let y = iy * UNROLL_Y;
        for ix in 0..chunks_x {
            let x = ix * UNROLL_X;
            unsafe {
                let values = [
                    input.load((y + 0) * WIDTH + x),
                    input.load((y + 1) * WIDTH + x),
                ];

                let transposed = V::transpose_contiguous_to_strided(values);

                output.store(transposed[0], y + (x + 0) * height);
                output.store(transposed[1], y + (x + 1) * height);
            }
        }
        if remainder_x == 1 {
            let x = chunks_x * UNROLL_X;
            unsafe {
                let values = [
                    input.load1_lo((y + 0) * WIDTH + x),
                    input.load1_lo((y + 1) * WIDTH + x),
                ];

                let transposed = V::transpose_contiguous_to_strided(values);

                output.store(transposed[0], y + x * height);
            }
        }
    }
    if remainder_y == 1 {
        let y = chunks_y * UNROLL_Y;
        for ix in 0..chunks_x {
            let x = ix * UNROLL_X;
            unsafe {
                let values = [
                    input.load(y * WIDTH + x),
                    V::zero_vector(),
                ];

                let transposed = V::transpose_contiguous_to_strided(values);

                output.store1_lo(transposed[0], y + (x + 0) * height);
                output.store1_lo(transposed[1], y + (x + 1) * height);
            }
        }
        if remainder_x == 1 {
            let x = chunks_x * UNROLL_X;
            unsafe {
                let value : V = input.load1_lo(y * WIDTH + x);
                output.store1_lo(value, y + x * height);
            }
        }
    }
}

#[inline(always)]
pub unsafe fn unrolled_transpose<V: SimdVector, const WIDTH: usize>(
    height: usize,
    input: &[Complex<V::ScalarType>],
    output: &mut [Complex<V::ScalarType>],
) {
    // It'd be really cool to be able to write this all with just a single const generic set of functions, but rust's type system just isn't there yet
    // For example, you can't use an associated constant of a generic parameter as an array size (IE you can't write [V::zero_vector; V::COMPLEX_PER_VECTOR])
    // Until then it's easiest to just split ut up by COMPLEX_PER_VECTOR
    match V::COMPLEX_PER_VECTOR {
        1 => unsafe { unrolled_transpose_1complex::<V, WIDTH>(height, input, output) },
        2 => unsafe { unrolled_transpose_2complex::<V, WIDTH>(height, input, output) },
        _ => unreachable!(),
    }
}
