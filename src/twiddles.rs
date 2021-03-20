use crate::{common::FftNum, FftDirection};
use num_complex::Complex;

pub fn compute_twiddle<T: FftNum>(
    index: usize,
    fft_len: usize,
    direction: FftDirection,
) -> Complex<T> {
    let constant = -T::TAU() / T::from_usize(fft_len).unwrap();
    let angle = constant * T::from_usize(index).unwrap();

    let result = Complex {
        re: angle.cos(),
        im: angle.sin(),
    };

    match direction {
        FftDirection::Forward => result,
        FftDirection::Inverse => result.conj(),
    }
}

pub fn compute_twiddle_floatindex<T: FftNum>(
    index: T,
    fft_len: usize,
    direction: FftDirection,
) -> Complex<T> {
    let constant = -T::TAU() / T::from_usize(fft_len).unwrap();
    let angle = constant * index;

    let result = Complex {
        re: angle.cos(),
        im: angle.sin(),
    };

    match direction {
        FftDirection::Forward => result,
        FftDirection::Inverse => result.conj(),
    }
}

pub fn rotate_90<T: FftNum>(value: Complex<T>, direction: FftDirection) -> Complex<T> {
    match direction {
        FftDirection::Forward => Complex {
            re: value.im,
            im: -value.re,
        },
        FftDirection::Inverse => Complex {
            re: -value.im,
            im: value.re,
        },
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_rotate() {
        // Verify that the rotate90 function does the same thing as multiplying by twiddle(1,4), in the forward direction
        let value = Complex { re: 9.1, im: 2.2 };
        let rotated_forward = rotate_90(value, FftDirection::Forward);
        let twiddled_forward = value * compute_twiddle(1, 4, FftDirection::Forward);

        assert_eq!(value.re, -rotated_forward.im);
        assert_eq!(value.im, rotated_forward.re);

        assert!(value.re + twiddled_forward.im < 0.0001);
        assert!(value.im - rotated_forward.re < 0.0001);

        // Verify that the rotate90 function does the same thing as multiplying by twiddle(1,4), in the inverse direction
        let rotated_forward = rotate_90(value, FftDirection::Inverse);
        let twiddled_forward = value * compute_twiddle(1, 4, FftDirection::Inverse);

        assert_eq!(value.re, rotated_forward.im);
        assert_eq!(value.im, -rotated_forward.re);

        assert!(value.re - twiddled_forward.im < 0.0001);
        assert!(value.im + rotated_forward.re < 0.0001);
    }
}
