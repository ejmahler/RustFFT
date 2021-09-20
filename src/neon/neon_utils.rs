use core::arch::aarch64::*;

//  __  __       _   _               _________  _     _ _
// |  \/  | __ _| |_| |__           |___ /___ \| |__ (_) |_
// | |\/| |/ _` | __| '_ \   _____    |_ \ __) | '_ \| | __|
// | |  | | (_| | |_| | | | |_____|  ___) / __/| |_) | | |_
// |_|  |_|\__,_|\__|_| |_|         |____/_____|_.__/|_|\__|
//

pub struct Rotate90F32 {
    //sign_lo: float32x4_t,
    sign_hi: float32x4_t,
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
                vld1q_f32([0.0, 0.0, -0.0, 0.0].as_ptr())
            } else {
                vld1q_f32([0.0, 0.0, 0.0, -0.0].as_ptr())
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
        let idx = vld1q_u8([0, 1, 2, 3, 4 ,5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11].as_ptr());
        let temp = vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(values), idx));
        vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(temp), vreinterpretq_u32_f32(self.sign_hi)))
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
        vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(temp), vreinterpretq_u32_f32(self.sign_both)))
    }
}

// Pack low (1st) complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r1.re, r1.im, l1.re, l1.im
#[inline(always)]
pub unsafe fn extract_lo_lo_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    //_mm_shuffle_ps(left, right, 0x44)
    vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(left), vreinterpretq_f64_f32(right)))
}

// Pack high (2nd) complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r2.re, r2.im, l2.re, l2.im
#[inline(always)]
pub unsafe fn extract_hi_hi_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(left), vreinterpretq_f64_f32(right)))
}

// Pack low (1st) and high (2nd) complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r1.re, r1.im, l2.re, l2.im
#[inline(always)]
pub unsafe fn extract_lo_hi_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    let idx = vld1q_u8([0, 1, 2, 3, 4 ,5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31].as_ptr());
    vreinterpretq_f32_u8(vqtbl2q_u8(uint8x16x2_t(vreinterpretq_u8_f32(left), vreinterpretq_u8_f32(right)), idx))
}


// Pack  high (2nd) and low (1st) complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r2.re, r2.im, l1.re, l1.im
#[inline(always)]
pub unsafe fn extract_hi_lo_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    vextq_f32::<2>(left, right)
}

// Reverse complex
// values: a.re, a.im, b.re, b.im
// --> b.re, b.im, a.re, a.im
#[inline(always)]
pub unsafe fn reverse_complex_elements_f32(values: float32x4_t) -> float32x4_t {
    vextq_f32::<2>(values, values)
}

// Invert sign of high (2nd) complex
// values: a.re, a.im, b.re, b.im
// -->  a.re, a.im, -b.re, -b.im
#[inline(always)]
pub unsafe fn negate_hi_f32(values: float32x4_t) -> float32x4_t {
    let neg = vld1q_f32([0.0, 0.0, -0.0, -0.0].as_ptr());
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(values), vreinterpretq_u32_f32(neg)))
}

// Duplicate low (1st) complex
// values: a.re, a.im, b.re, b.im
// --> a.re, a.im, a.re, a.im
#[inline(always)]
pub unsafe fn duplicate_lo_f32(values: float32x4_t) -> float32x4_t {
    vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(values), vreinterpretq_f64_f32(values)))
}

// Duplicate high (2nd) complex
// values: a.re, a.im, b.re, b.im
// --> b.re, b.im, b.re, b.im
#[inline(always)]
pub unsafe fn duplicate_hi_f32(values: float32x4_t) -> float32x4_t {
    vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(values), vreinterpretq_f64_f32(values)))
}

// transpose a 2x2 complex matrix given as [x0, x1], [x2, x3]
// result is [x0, x2], [x1, x3]
#[inline(always)]
pub unsafe fn transpose_complex_2x2_f32(left: float32x4_t, right: float32x4_t) -> [float32x4_t; 2] {
    let temp02 = extract_lo_lo_f32(left, right);
    let temp13 = extract_hi_hi_f32(left, right);
    [temp02, temp13]
}

// Complex multiplication.
// Each input contains two complex values, which are multiplied in parallel.
#[inline(always)]
pub unsafe fn mul_complex_f32(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    // Workaround since vcmulq_f32 and vcmlaq_f32 intrinsics are not yet available.
    let mut temp1 = vtrn1q_f32(right, right);
    let mut temp2 = vtrn2q_f32(right, right);
    temp1 = vmulq_f32(temp1, left);
    temp2 = vmulq_f32(temp2, left);
    temp2 = vrev64q_f32(temp2);
    let flip = vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr());
    temp2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(temp2), vreinterpretq_u32_f32(flip)));
    vaddq_f32(temp1, temp2)
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
        let temp = vextq_f64(values, values, 0x01);
        vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(temp), vreinterpretq_u64_f64(self.sign)))
    }
}

#[inline(always)]
pub unsafe fn mul_complex_f64(left: float64x2_t, right: float64x2_t) -> float64x2_t {
    // Workaround since vcmulq_f64 and vcmlaq_f64 intrinsics are not yet available.
    let mut temp1 = vtrn1q_f64(right, right);
    let mut temp2 = vtrn2q_f64(right, right);
    temp1 = vmulq_f64(temp1, left);
    temp2 = vmulq_f64(temp2, left);
    temp2 = vextq_f64(temp2, temp2, 0x01);
    let flip = vld1q_f64([-0.0, 0.0].as_ptr());
    temp2 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(temp2), vreinterpretq_u64_f64(flip)));
    vaddq_f64(temp1, temp2)
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_mul_complex_f64() {
        unsafe {
            let right = vld1q_f64([1.0, 2.0].as_ptr());
            let left = vld1q_f64([5.0, 7.0].as_ptr());
            let res = mul_complex_f64(left, right);
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
            let val2 = Complex::<f32>::new(3.2, 4.2);
            let val3 = Complex::<f32>::new(5.6, 6.2);
            let val4 = Complex::<f32>::new(7.4, 8.3);

            let nbr2 = vld1q_f32([val3, val4].as_ptr() as *const f32);
            let nbr1 = vld1q_f32([val1, val2].as_ptr() as *const f32);
            let res = mul_complex_f32(nbr1, nbr2);
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

}
