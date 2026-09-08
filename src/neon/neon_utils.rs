use core::arch::aarch64::*;
use num_complex::Complex;
use crate::FftNum;
use crate::neon::neon_vector::{NeonArray, NeonArrayMut, NeonVector};

//  __  __       _   _               _________  _     _ _
// |  \/  | __ _| |_| |__           |___ /___ \| |__ (_) |_
// | |\/| |/ _` | __| '_ \   _____    |_ \ __) | '_ \| | __|
// | |  | | (_| | |_| | | | |_____|  ___) / __/| |_) | | |_
// |_|  |_|\__,_|\__|_| |_|         |____/_____|_.__/|_|\__|
//

pub struct Rotate90F32 {
    //sign_lo: float32x4_t,
    sign_hi: float32x2_t,
    sign_both: float32x4_t,
}

impl Rotate90F32 {
    pub fn new(positive: bool) -> Self {
        // There doesn't seem to be any need for rotating just the first element, but let's keep the code just in case
        //let sign_lo = unsafe {
        //    if positive {
        //        _mm_set_ps(0.0, 0.0, 0.0, -0.0)
        //    }
        //    else {
        //        _mm_set_ps(0.0, 0.0, -0.0, 0.0)
        //    }
        //};
        let sign_hi = unsafe {
            if positive {
                vld1_f32([-0.0, 0.0].as_ptr())
            } else {
                vld1_f32([0.0, -0.0].as_ptr())
            }
        };
        let sign_both = unsafe {
            if positive {
                vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr())
            } else {
                vld1q_f32([0.0, -0.0, 0.0, -0.0].as_ptr())
            }
        };
        Self {
            //sign_lo,
            sign_hi,
            sign_both,
        }
    }

    #[inline(always)]
    pub unsafe fn rotate_hi(&self, values: float32x4_t) -> float32x4_t {
        vcombine_f32(
            vget_low_f32(values),
            vreinterpret_f32_u32(veor_u32(
                vrev64_u32(vreinterpret_u32_f32(vget_high_f32(values))),
                vreinterpret_u32_f32(self.sign_hi),
            )),
        )
    }

    // There doesn't seem to be any need for rotating just the first element, but let's keep the code just in case
    //#[inline(always)]
    //pub unsafe fn rotate_lo(&self, values: __m128) -> __m128 {
    //    let temp = _mm_shuffle_ps(values, values, 0xE1);
    //    _mm_xor_ps(temp, self.sign_lo)
    //}

    #[inline(always)]
    pub unsafe fn rotate_both(&self, values: float32x4_t) -> float32x4_t {
        let temp = vrev64q_f32(values);
        vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(temp),
            vreinterpretq_u32_f32(self.sign_both),
        ))
    }

    #[inline(always)]
    pub unsafe fn rotate_both_45(&self, values: float32x4_t) -> float32x4_t {
        let rotated = self.rotate_both(values);
        let sum = vaddq_f32(rotated, values);
        vmulq_f32(sum, vmovq_n_f32(0.5f32.sqrt()))
    }

    #[inline(always)]
    pub unsafe fn rotate_both_135(&self, values: float32x4_t) -> float32x4_t {
        let rotated = self.rotate_both(values);
        let diff = vsubq_f32(rotated, values);
        vmulq_f32(diff, vmovq_n_f32(0.5f32.sqrt()))
    }

    #[inline(always)]
    pub unsafe fn rotate_both_225(&self, values: float32x4_t) -> float32x4_t {
        let rotated = self.rotate_both(values);
        let diff = vaddq_f32(rotated, values);
        vmulq_f32(diff, vmovq_n_f32(-(0.5f32.sqrt())))
    }
}

// Pack low (1st) complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r1.re, r1.im, l1.re, l1.im
#[inline(always)]
pub unsafe fn extract_lo_lo_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    //_mm_shuffle_ps(left, right, 0x44)
    vreinterpretq_f32_f64(vtrn1q_f64(
        vreinterpretq_f64_f32(left),
        vreinterpretq_f64_f32(right),
    ))
}

// Pack high (2nd) complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r2.re, r2.im, l2.re, l2.im
#[inline(always)]
pub unsafe fn extract_hi_hi_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    vreinterpretq_f32_f64(vtrn2q_f64(
        vreinterpretq_f64_f32(left),
        vreinterpretq_f64_f32(right),
    ))
}

// Pack low (1st) and high (2nd) complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r1.re, r1.im, l2.re, l2.im
#[inline(always)]
pub unsafe fn extract_lo_hi_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    vcombine_f32(vget_low_f32(left), vget_high_f32(right))
}

// Pack  high (2nd) and low (1st) complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r2.re, r2.im, l1.re, l1.im
#[inline(always)]
pub unsafe fn extract_hi_lo_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    vcombine_f32(vget_high_f32(left), vget_low_f32(right))
}

// Reverse complex
// values: a.re, a.im, b.re, b.im
// --> b.re, b.im, a.re, a.im
#[inline(always)]
pub unsafe fn reverse_complex_elements_f32(values: float32x4_t) -> float32x4_t {
    vcombine_f32(vget_high_f32(values), vget_low_f32(values))
}

// Reverse complex and then negate hi complex
// values: a.re, a.im, b.re, b.im
// --> b.re, b.im, -a.re, -a.im
#[inline(always)]
pub unsafe fn reverse_complex_and_negate_hi_f32(values: float32x4_t) -> float32x4_t {
    vcombine_f32(vget_high_f32(values), vneg_f32(vget_low_f32(values)))
}

// Invert sign of high (2nd) complex
// values: a.re, a.im, b.re, b.im
// -->  a.re, a.im, -b.re, -b.im
//#[inline(always)]
//pub unsafe fn negate_hi_f32(values: float32x4_t) -> float32x4_t {
//    vcombine_f32(vget_low_f32(values), vneg_f32(vget_high_f32(values)))
//}

// Duplicate low (1st) complex
// values: a.re, a.im, b.re, b.im
// --> a.re, a.im, a.re, a.im
#[inline(always)]
pub unsafe fn duplicate_lo_f32(values: float32x4_t) -> float32x4_t {
    vreinterpretq_f32_f64(vtrn1q_f64(
        vreinterpretq_f64_f32(values),
        vreinterpretq_f64_f32(values),
    ))
}

// Duplicate high (2nd) complex
// values: a.re, a.im, b.re, b.im
// --> b.re, b.im, b.re, b.im
#[inline(always)]
pub unsafe fn duplicate_hi_f32(values: float32x4_t) -> float32x4_t {
    vreinterpretq_f32_f64(vtrn2q_f64(
        vreinterpretq_f64_f32(values),
        vreinterpretq_f64_f32(values),
    ))
}

// Transpose a 2x2 complex f32 matrix in NEON registers:
// left:  [a0.re, a0.im, a1.re, a1.im] (row 0: a0 = col 0, a1 = col 1)
// right: [b0.re, b0.im, b1.re, b1.im] (row 1: b0 = col 0, b1 = col 1)
// Returns:
// [0]: [a0.re, a0.im, b0.re, b0.im] (col 0: a0 = row 0, b0 = row 1)
// [1]: [a1.re, a1.im, b1.re, b1.im] (col 1: a1 = row 0, b1 = row 1)
#[inline(always)]
pub unsafe fn transpose_complex_2x2_f32(left: float32x4_t, right: float32x4_t) -> [float32x4_t; 2] {
    let temp02 = extract_lo_lo_f32(left, right);
    let temp13 = extract_hi_hi_f32(left, right);
    [temp02, temp13]
}

//  __  __       _   _                __   _  _   _     _ _
// |  \/  | __ _| |_| |__            / /_ | || | | |__ (_) |_
// | |\/| |/ _` | __| '_ \   _____  | '_ \| || |_| '_ \| | __|
// | |  | | (_| | |_| | | | |_____| | (_) |__   _| |_) | | |_
// |_|  |_|\__,_|\__|_| |_|          \___/   |_| |_.__/|_|\__|
//

pub(crate) struct Rotate90F64 {
    sign: float64x2_t,
}

impl Rotate90F64 {
    pub fn new(positive: bool) -> Self {
        let sign = unsafe {
            if positive {
                vld1q_f64([-0.0, 0.0].as_ptr())
            } else {
                vld1q_f64([0.0, -0.0].as_ptr())
            }
        };
        Self { sign }
    }

    #[inline(always)]
    pub unsafe fn rotate(&self, values: float64x2_t) -> float64x2_t {
        let temp = vcombine_f64(vget_high_f64(values), vget_low_f64(values));
        vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(temp),
            vreinterpretq_u64_f64(self.sign),
        ))
    }

    #[inline(always)]
    pub unsafe fn rotate_45(&self, values: float64x2_t) -> float64x2_t {
        let rotated = self.rotate(values);
        let sum = vaddq_f64(rotated, values);
        vmulq_f64(sum, vmovq_n_f64(0.5f64.sqrt()))
    }

    #[inline(always)]
    pub unsafe fn rotate_135(&self, values: float64x2_t) -> float64x2_t {
        let rotated = self.rotate(values);
        let diff = vsubq_f64(rotated, values);
        vmulq_f64(diff, vmovq_n_f64(0.5f64.sqrt()))
    }

    #[inline(always)]
    pub unsafe fn rotate_225(&self, values: float64x2_t) -> float64x2_t {
        let rotated = self.rotate(values);
        let diff = vaddq_f64(rotated, values);
        vmulq_f64(diff, vmovq_n_f64(-(0.5f64.sqrt())))
    }
}

/// Fused complex twiddle multiplication and matrix transpose for small f64 sizes.
///
/// Computes `output[y + x * height] = input[x + y * width] * twiddles[x + y * width]`.
/// Processes 2x2 complex blocks with NEON SIMD vector instructions,
/// followed by remainder column and row handling.
#[inline(always)]
pub unsafe fn transpose_small_twiddle_f64(
    input: impl NeonArray<f64>,
    mut output: impl NeonArrayMut<f64>,
    twiddles: impl NeonArray<f64>,
    width: usize,
    height: usize,
) {
    // Loop 1: process 2 rows at a time
    let mut y = 0;
    while y + 2 <= height {
        let in_r0 = y * width;
        let in_r1 = (y + 1) * width;

        // Loop 2: process 2 columns at a time (2x2 complex block)
        let mut x = 0;
        while x + 2 <= width {
            let out_c0 = y + x * height;
            let out_c1 = y + (x + 1) * height;

            let a0 = input.load_complex(in_r0 + x);
            let tw_a0 = twiddles.load_complex(in_r0 + x);
            let a1 = input.load_complex(in_r0 + x + 1);
            let tw_a1 = twiddles.load_complex(in_r0 + x + 1);

            let b0 = input.load_complex(in_r1 + x);
            let tw_b0 = twiddles.load_complex(in_r1 + x);
            let b1 = input.load_complex(in_r1 + x + 1);
            let tw_b1 = twiddles.load_complex(in_r1 + x + 1);

            let res_a0 = NeonVector::mul_complex(a0, tw_a0);
            let res_a1 = NeonVector::mul_complex(a1, tw_a1);
            let res_b0 = NeonVector::mul_complex(b0, tw_b0);
            let res_b1 = NeonVector::mul_complex(b1, tw_b1);

            output.store_complex(res_a0, out_c0);
            output.store_complex(res_b0, out_c0 + 1);
            output.store_complex(res_a1, out_c1);
            output.store_complex(res_b1, out_c1 + 1);

            x += 2;
        }

        // Remainder column when width is odd (2 rows x 1 column)
        if x < width {
            let out_c0 = y + x * height;

            let a0 = input.load_complex(in_r0 + x);
            let tw_a0 = twiddles.load_complex(in_r0 + x);
            let b0 = input.load_complex(in_r1 + x);
            let tw_b0 = twiddles.load_complex(in_r1 + x);

            let res_a0 = NeonVector::mul_complex(a0, tw_a0);
            let res_b0 = NeonVector::mul_complex(b0, tw_b0);

            output.store_complex(res_a0, out_c0);
            output.store_complex(res_b0, out_c0 + 1);
        }

        y += 2;
    }

    // Remainder row when height is odd (1 row x width columns)
    if y < height {
        let in_r = y * width;
        for x in 0..width {
            let in_idx = in_r + x;
            let out_idx = y + x * height;
            let a = input.load_complex(in_idx);
            let tw = twiddles.load_complex(in_idx);
            let res = NeonVector::mul_complex(a, tw);
            output.store_complex(res, out_idx);
        }
    }
}

/// Fused complex twiddle multiplication and matrix transpose for small f32 sizes.
///
/// Computes `output[y + x * height] = input[x + y * width] * twiddles[x + y * width]`.
/// Processes 2x2 complex blocks with NEON SIMD vector instructions,
/// followed by remainder column and row handling.
#[inline(always)]
pub unsafe fn transpose_small_twiddle_f32(
    input: impl NeonArray<f32>,
    mut output: impl NeonArrayMut<f32>,
    twiddles: impl NeonArray<f32>,
    width: usize,
    height: usize,
) {
    // Loop 1: process 2 rows at a time
    let mut y = 0;
    while y + 2 <= height {
        let in_r0 = y * width;
        let in_r1 = (y + 1) * width;

        // Loop 2: process 2 columns at a time (2x2 complex block)
        let mut x = 0;
        while x + 2 <= width {
            let out_c0 = y + x * height;
            let out_c1 = y + (x + 1) * height;

            let row0 = input.load_complex(in_r0 + x);
            let tw_row0 = twiddles.load_complex(in_r0 + x);
            let row1 = input.load_complex(in_r1 + x);
            let tw_row1 = twiddles.load_complex(in_r1 + x);

            let res0 = NeonVector::mul_complex(row0, tw_row0);
            let res1 = NeonVector::mul_complex(row1, tw_row1);

            let [col0, col1] = transpose_complex_2x2_f32(res0, res1);

            output.store_complex(col0, out_c0);
            output.store_complex(col1, out_c1);

            x += 2;
        }

        // Remainder column when width is odd (2 rows x 1 column)
        if x < width {
            let out_c0 = y + x * height;

            let a0 = vget_low_f32(input.load_partial_lo_complex(in_r0 + x));
            let b0 = vget_low_f32(input.load_partial_lo_complex(in_r1 + x));
            let val = vcombine_f32(a0, b0);

            let tw_a0 = vget_low_f32(twiddles.load_partial_lo_complex(in_r0 + x));
            let tw_b0 = vget_low_f32(twiddles.load_partial_lo_complex(in_r1 + x));
            let tw = vcombine_f32(tw_a0, tw_b0);

            let res = NeonVector::mul_complex(val, tw);

            output.store_complex(res, out_c0);
        }

        y += 2;
    }

    // Remainder row when height is odd (1 row x width columns)
    if y < height {
        let in_r = y * width;
        for x in 0..width {
            let in_idx = in_r + x;
            let out_idx = y + x * height;
            let a = input.load_partial_lo_complex(in_idx);
            let tw = twiddles.load_partial_lo_complex(in_idx);
            let res = NeonVector::mul_complex(a, tw);
            output.store_partial_lo_complex(res, out_idx);
        }
    }
}

pub unsafe fn transpose_small_twiddle<T: FftNum>(
    width: usize,
    height: usize,
    input: &[Complex<T>],
    output: &mut [Complex<T>],
    twiddles: &[Complex<T>],
) -> bool {
    debug_assert!(input.len() >= width * height);
    debug_assert!(output.len() >= width * height);
    debug_assert!(twiddles.len() >= width * height);

    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input: &[Complex<f64>] = crate::array_utils::workaround_transmute(input);
        let output: &mut [Complex<f64>] = crate::array_utils::workaround_transmute_mut(output);
        let twiddles: &[Complex<f64>] = crate::array_utils::workaround_transmute(twiddles);
        transpose_small_twiddle_f64(input, output, twiddles, width, height);
        return true;
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input: &[Complex<f32>] = crate::array_utils::workaround_transmute(input);
        let output: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(output);
        let twiddles: &[Complex<f32>] = crate::array_utils::workaround_transmute(twiddles);
        transpose_small_twiddle_f32(input, output, twiddles, width, height);
        return true;
    }
    false
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::neon::NeonVector;
    use num_complex::Complex;

    #[test]
    fn test_mul_complex_f64() {
        unsafe {
            let right = vld1q_f64([1.0, 2.0].as_ptr());
            let left = vld1q_f64([5.0, 7.0].as_ptr());
            let res = NeonVector::mul_complex(left, right);
            let expected = vld1q_f64([1.0 * 5.0 - 2.0 * 7.0, 1.0 * 7.0 + 2.0 * 5.0].as_ptr());
            assert_eq!(
                std::mem::transmute::<float64x2_t, Complex<f64>>(res),
                std::mem::transmute::<float64x2_t, Complex<f64>>(expected)
            );
        }
    }

    #[test]
    fn test_mul_complex_f32() {
        unsafe {
            let val1 = Complex::<f32>::new(1.0, 2.5);
            let val2 = Complex::<f32>::new(3.2, 4.75);
            let val3 = Complex::<f32>::new(5.75, 6.25);
            let val4 = Complex::<f32>::new(7.4, 8.5);

            let nbr2 = vld1q_f32([val3, val4].as_ptr() as *const f32);
            let nbr1 = vld1q_f32([val1, val2].as_ptr() as *const f32);
            let res = NeonVector::mul_complex(nbr1, nbr2);
            let res = std::mem::transmute::<float32x4_t, [Complex<f32>; 2]>(res);
            let expected = [val1 * val3, val2 * val4];
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn test_pack() {
        unsafe {
            let nbr2 = vld1q_f32([5.0, 6.0, 7.0, 8.0].as_ptr());
            let nbr1 = vld1q_f32([1.0, 2.0, 3.0, 4.0].as_ptr());
            let first = extract_lo_lo_f32(nbr1, nbr2);
            let second = extract_hi_hi_f32(nbr1, nbr2);
            let first = std::mem::transmute::<float32x4_t, [Complex<f32>; 2]>(first);
            let second = std::mem::transmute::<float32x4_t, [Complex<f32>; 2]>(second);
            let first_expected = [Complex::new(1.0, 2.0), Complex::new(5.0, 6.0)];
            let second_expected = [Complex::new(3.0, 4.0), Complex::new(7.0, 8.0)];
            assert_eq!(first, first_expected);
            assert_eq!(second, second_expected);
        }
    }

    #[test]
    fn test_transpose_small_twiddle_neon() {
        use num_traits::Zero;
        for width in [1, 2, 3, 4, 5, 7, 8, 16, 31, 32, 33] {
            for height in [1, 2, 3, 4, 5, 7, 8, 16, 31, 32, 33] {
                let len = width * height;

                // f32
                let input_f32: Vec<Complex<f32>> = (0..len)
                    .map(|i| Complex::new((i % 7) as f32, (i % 5) as f32))
                    .collect();
                let twiddles_f32: Vec<Complex<f32>> = (0..len)
                    .map(|i| Complex::new((i % 3) as f32, (i % 4) as f32))
                    .collect();
                let mut out_f32 = vec![Complex::zero(); len];
                unsafe {
                    transpose_small_twiddle(width, height, &input_f32, &mut out_f32, &twiddles_f32);
                }
                for y in 0..height {
                    for x in 0..width {
                        let expected = input_f32[x + y * width] * twiddles_f32[x + y * width];
                        let actual = out_f32[y + x * height];
                        assert!(
                            (actual.re - expected.re).abs() < 1e-5 && (actual.im - expected.im).abs() < 1e-5,
                            "f32 twiddle mismatch at ({}, {}) for {}x{}: expected {:?}, got {:?}",
                            x, y, width, height, expected, actual
                        );
                    }
                }

                // f64
                let input_f64: Vec<Complex<f64>> = (0..len)
                    .map(|i| Complex::new((i % 7) as f64, (i % 5) as f64))
                    .collect();
                let twiddles_f64: Vec<Complex<f64>> = (0..len)
                    .map(|i| Complex::new((i % 3) as f64, (i % 4) as f64))
                    .collect();
                let mut out_f64 = vec![Complex::zero(); len];
                unsafe {
                    transpose_small_twiddle(width, height, &input_f64, &mut out_f64, &twiddles_f64);
                }
                for y in 0..height {
                    for x in 0..width {
                        let expected = input_f64[x + y * width] * twiddles_f64[x + y * width];
                        let actual = out_f64[y + x * height];
                        assert!(
                            (actual.re - expected.re).abs() < 1e-10 && (actual.im - expected.im).abs() < 1e-10,
                            "f64 twiddle mismatch at ({}, {}) for {}x{}: expected {:?}, got {:?}",
                            x, y, width, height, expected, actual
                        );
                    }
                }
            }
        }
    }
}
