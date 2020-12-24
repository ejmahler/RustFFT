use crate::common::FFTnum;
use num_complex::Complex;

pub fn generate_twiddle_factors<T: FFTnum>(fft_len: usize, inverse: bool) -> Vec<Complex<T>> {
    (0..fft_len)
        .map(|i| T::generate_twiddle_factor(i, fft_len, inverse))
        .collect()
}

pub fn rotate_90<T: FFTnum>(value: Complex<T>, inverse: bool) -> Complex<T> {
    if inverse {
        Complex {
            re: -value.im,
            im: value.re,
        }
    } else {
        Complex {
            re: value.im,
            im: -value.re,
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::compare_vectors;
    use std::f32;

    #[test]
    fn test_generate() {
        //test the length-0 case
        let zero_twiddles: Vec<Complex<f32>> = generate_twiddle_factors(0, false);
        assert_eq!(0, zero_twiddles.len());

        let constant = -2f32 * f32::consts::PI;

        for len in 1..10 {
            let actual: Vec<Complex<f32>> = generate_twiddle_factors(len, false);
            let expected: Vec<Complex<f32>> = (0..len)
                .map(|i| Complex::from_polar(1f32, constant * i as f32 / len as f32))
                .collect();

            assert!(compare_vectors(&actual, &expected), "len = {}", len)
        }

        //for each len, verify that each element in the inverse is the conjugate of the non-inverse
        for len in 1..10 {
            let twiddles: Vec<Complex<f32>> = generate_twiddle_factors(len, false);
            let mut twiddles_inverse: Vec<Complex<f32>> = generate_twiddle_factors(len, true);

            for value in twiddles_inverse.iter_mut() {
                *value = value.conj();
            }

            assert!(
                compare_vectors(&twiddles, &twiddles_inverse),
                "len = {}",
                len
            )
        }
    }
}
