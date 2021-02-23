use core::arch::x86_64::*;

//  __  __       _   _               _________  _     _ _   
// |  \/  | __ _| |_| |__           |___ /___ \| |__ (_) |_ 
// | |\/| |/ _` | __| '_ \   _____    |_ \ __) | '_ \| | __|
// | |  | | (_| | |_| | | | |_____|  ___) / __/| |_) | | |_ 
// |_|  |_|\__,_|\__|_| |_|         |____/_____|_.__/|_|\__|
//                                                          



pub struct Rotate90_32 {
    //sign_1st: __m128,
    sign_2nd: __m128,
    sign_both: __m128,
}

impl Rotate90_32 {
    pub fn new(positive: bool) -> Self {
        //let sign_1st = unsafe {
        //    if positive {
        //        _mm_set_ps(0.0, 0.0, 0.0, -0.0)
        //    }
        //    else {
        //        _mm_set_ps(0.0, 0.0, -0.0, 0.0)
        //    }
        //};
        let sign_2nd = unsafe {
            if positive {
                _mm_set_ps(0.0, -0.0, 0.0, 0.0)
            }
            else {
                _mm_set_ps(-0.0, 0.0, 0.0, 0.0)
            }
        };
        let sign_both = unsafe {
            if positive {
                _mm_set_ps(0.0, -0.0, 0.0, -0.0)
            }
            else {
                _mm_set_ps(-0.0, 0.0, -0.0, 0.0)
            }
        };
        Self {
            //sign_1st,
            sign_2nd,
            sign_both,
        }
    }

    #[inline(always)]
    pub unsafe fn rotate_2nd(&self, values: __m128) -> __m128 {
        let temp = _mm_shuffle_ps(values, values, 0xB4);
        _mm_xor_ps(temp, self.sign_2nd)
    }

    //#[inline(always)]
    //pub unsafe fn rotate_1st(&self, values: __m128) -> __m128 {
    //    let temp = _mm_shuffle_ps(values, values, 0xE1);
    //    _mm_xor_ps(temp, self.sign_1st)
    //}

    #[inline(always)]
    pub unsafe fn rotate_both(&self, values: __m128) -> __m128 {
        let temp = _mm_shuffle_ps(values, values, 0xB1);
        _mm_xor_ps(temp, self.sign_both)
    }
}

#[inline(always)]
pub unsafe fn pack_1st_32(left: __m128, right: __m128) -> __m128 {
    _mm_shuffle_ps(left, right, 0x44)
}

#[inline(always)]
pub unsafe fn pack_2nd_32(left: __m128, right: __m128) -> __m128 {
    _mm_shuffle_ps(left, right, 0xEE)
}


#[inline(always)]
pub unsafe fn complex_double_mul_32(left: __m128, right: __m128) -> __m128 {
    // SSE2
    // http://microperf.blogspot.com/2016/12/multiplying-two-complex-numbers-by-two.html
    // ar_ai_cr_ci: __m128, br_bi_dr_di: __m128
    //let sign = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
    //let ar_ar_cr_cr = _mm_shuffle_ps(ar_ai_cr_ci, ar_ai_cr_ci, 0xA0);
    //let ai_ai_ci_ci = _mm_shuffle_ps(ar_ai_cr_ci, ar_ai_cr_ci, 0xF5);
    //let bi_br_di_dr = _mm_shuffle_ps(br_bi_dr_di, br_bi_dr_di, 0xB1);
    //let arbr_arbi_crdr_crdi = _mm_mul_ps(ar_ar_cr_cr, br_bi_dr_di);
    //let aibi_aibr_cidi_cidr = _mm_mul_ps(ai_ai_ci_ci, bi_br_di_dr);
    //let naibi_aibr_ncidi_cidr = _mm_xor_ps(aibi_aibr_cidi_cidr, sign);
    //_mm_add_ps(arbr_arbi_crdr_crdi, naibi_aibr_ncidi_cidr)

    //SSE3, Intel manual
    let mut temp1 = _mm_shuffle_ps(right, right,0xA0);
    let mut temp2 = _mm_shuffle_ps(right, right,0xF5);
    temp1 = _mm_mul_ps(temp1, left);
    temp2 = _mm_mul_ps(temp2, left);
    temp2 = _mm_shuffle_ps(temp2, temp2, 0xB1);
    _mm_addsub_ps(temp1, temp2)
}


//  __  __       _   _                __   _  _   _     _ _   
// |  \/  | __ _| |_| |__            / /_ | || | | |__ (_) |_ 
// | |\/| |/ _` | __| '_ \   _____  | '_ \| || |_| '_ \| | __|
// | |  | | (_| | |_| | | | |_____| | (_) |__   _| |_) | | |_ 
// |_|  |_|\__,_|\__|_| |_|          \___/   |_| |_.__/|_|\__|
//                                                            



pub(crate) struct Rotate90_64 {
    sign: __m128d,
}

impl Rotate90_64 {
    pub fn new(positive: bool) -> Self {
        let sign = unsafe {
            if positive {
                _mm_set_pd(0.0, -0.0)
            }
            else {
                _mm_set_pd(-0.0, 0.0)
            }
        };
        Self {
            sign,
        }
    }

    #[inline(always)]
    pub unsafe fn rotate(&self, values: __m128d) -> __m128d {
        let temp = _mm_shuffle_pd(values, values, 0x01);
        _mm_xor_pd(temp, self.sign)
    }
}

#[inline(always)]
pub unsafe fn complex_mul_64(left: __m128d, right: __m128d) -> __m128d {
    // SSE2
    //let mul1 = _mm_mul_pd(left, right);
    //let right_flipped = _mm_shuffle_pd(right, right, 0x01);
    //let mul2 = _mm_mul_pd(left, right_flipped);
    //let sign = _mm_set_pd(-0.0, 0.0);
    //let mul1 = _mm_xor_pd(mul1, sign);
    //let temp1 = _mm_shuffle_pd(mul1, mul2, 0x00);
    //let temp2 = _mm_shuffle_pd(mul1, mul2, 0x03);
    //_mm_add_pd(temp1, temp2)

    // SSE3
    let mut temp1 = _mm_unpacklo_pd(right, right);
    let mut temp2 = _mm_unpackhi_pd(right, right);
    temp1 = _mm_mul_pd(temp1, left);
    temp2 = _mm_mul_pd(temp2, left);
    temp2 = _mm_shuffle_pd(temp2, temp2, 0x01);
    _mm_addsub_pd(temp1, temp2)


}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use num_complex::Complex;

    #[inline(always)]
    unsafe fn complex_mul_64(left: __m128d, right: __m128d) -> __m128d {
        let mul1 = _mm_mul_pd(left, right);
        let right_flipped = _mm_shuffle_pd(right, right, 0x01);
        let mul2 = _mm_mul_pd(left, right_flipped);
        let sign = _mm_set_pd(-0.0, 0.0);
        let mul1 = _mm_xor_pd(mul1, sign);
        let temp1 = _mm_shuffle_pd(mul1, mul2, 0x00);
        let temp2 = _mm_shuffle_pd(mul1, mul2, 0x03);
        _mm_add_pd(temp1, temp2)
    }


    #[test]
    fn test_complex_mul_64() {
        unsafe {
            let right = _mm_set_pd(1.0, 2.0);
            let left = _mm_set_pd(5.0, 7.0);
            println!("left: {:?}", left);
            println!("right: {:?}", right);
            let res = complex_mul_64(left, right);
            println!("res: {:?}", res);
            let expected = _mm_set_pd(2.0*5.0 + 1.0*7.0, 2.0*7.0 - 1.0*5.0);
            assert_eq!(std::mem::transmute::<__m128d, Complex<f64>>(res), std::mem::transmute::<__m128d, Complex<f64>>(expected));
        }
    }

    #[test]
    fn test_complex_double_mul_32() {
        unsafe {
            let val1 = Complex::<f32>::new(1.0, 2.5);
            let val2 = Complex::<f32>::new(3.2, 4.2);
            let val3 = Complex::<f32>::new(5.6, 6.2);
            let val4 = Complex::<f32>::new(7.4, 8.3);

            let nbr2 = _mm_set_ps(val4.im, val4.re, val3.im, val3.re);
            let nbr1 = _mm_set_ps(val2.im, val2.re, val1.im, val1.re);
            println!("left: {:?}", nbr1);
            println!("right: {:?}", nbr2);
            let res = complex_double_mul_32(nbr1, nbr2);
            let res = std::mem::transmute::<__m128, [Complex<f32>;2]>(res);
            println!("res: {:?}", res);
            let expected = [val1*val3, val2*val4];
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn test_pack() {
        unsafe {
            let nbr2 = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
            let nbr1 = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
            println!("nbr1: {:?}", nbr1);
            println!("nbr2: {:?}", nbr2);
            let first = pack_1st_32(nbr1, nbr2);
            let second = pack_2nd_32(nbr1, nbr2);
            println!("first: {:?}", first);
            println!("second: {:?}", first);
            let first = std::mem::transmute::<__m128, [Complex<f32>;2]>(first);
            let second = std::mem::transmute::<__m128, [Complex<f32>;2]>(second);
            let first_expected = [Complex::new(1.0, 2.0), Complex::new(5.0, 6.0)];
            let second_expected = [Complex::new(3.0, 4.0), Complex::new(7.0, 8.0)];
            assert_eq!(first, first_expected);
            assert_eq!(second, second_expected);
        }
    }
}