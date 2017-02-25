
use std::f64;
use num::{Complex, FromPrimitive, Signed, One};

pub fn generate_twiddle_factors<T>(fft_len: usize, inverse: bool) -> Vec<Complex<T>> where T: Signed + FromPrimitive + Copy
{
	let constant = if inverse {
        2f64 * f64::consts::PI
    } else {
        -2f64  * f64::consts::PI
    };

	(0..fft_len)
        .map(|i| constant * i as f64 / fft_len as f64)
        .map(|phase| Complex::from_polar(&One::one(), &phase))
        .map(|c| {
            Complex {
                re: FromPrimitive::from_f64(c.re).unwrap(),
                im: FromPrimitive::from_f64(c.im).unwrap(),
            }
        })
        .collect()
}

#[cfg(test)]
mod test {
	use super::*;
	use std::f32;
    use ::test::{compare_vectors};

    #[test]
    fn test_twiddles()
    {
    	//test the length-0 case
    	let zero_twiddles: Vec<Complex<f32>> = generate_twiddle_factors(0, false);
    	assert_eq!(0, zero_twiddles.len());

    	let constant = -2f32 * f32::consts::PI;

    	for len in 1..10 {
    		let actual: Vec<Complex<f32>> = generate_twiddle_factors(len, false);
    		let expected: Vec<Complex<f32>> = (0..len).map(|i| Complex::from_polar(&1f32, &(constant * i as f32 / len as f32))).collect();

    		assert!(compare_vectors(&actual, &expected), "len = {}", len)
    	}
    }

    #[test]
    fn test_inverse()
    {
    	//for each len, verify that each element in the inverse is the conjugate of the non-inverse
    	for len in 1..10 {
    		let twiddles: Vec<Complex<f32>> = generate_twiddle_factors(len, false);
    		let mut twiddles_inverse: Vec<Complex<f32>> = generate_twiddle_factors(len, true);

    		for value in twiddles_inverse.iter_mut()
    		{
    			*value = value.conj();
    		}

    		assert!(compare_vectors(&twiddles, &twiddles_inverse), "len = {}", len)
    	}
    }
}