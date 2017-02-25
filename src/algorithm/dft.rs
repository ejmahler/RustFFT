use std::f32;

use num::{Complex, FromPrimitive, Signed, Zero};

use algorithm::FFTAlgorithm;

pub struct DFTAlgorithm<T> {
    twiddles: Vec<Complex<T>>,
}

impl<T> DFTAlgorithm<T>
    where T: Signed + FromPrimitive + Copy
{
    pub fn new(len: usize) -> Self {

        Self {
            twiddles: (0..len)
                .map(|i| -2f32 * i as f32 * f32::consts::PI / len as f32)
                .map(|phase| Complex::from_polar(&1.0, &phase))
                .map(|c| {
                    Complex {
                        re: FromPrimitive::from_f32(c.re).unwrap(),
                        im: FromPrimitive::from_f32(c.im).unwrap(),
                    }
                })
                .collect(),
        }
    }
}

impl<T> FFTAlgorithm<T> for DFTAlgorithm<T>
    where T: Signed + FromPrimitive + Copy
{
    fn process(&mut self, signal: &[Complex<T>], spectrum: &mut [Complex<T>]) {
        for (k, output_cell) in spectrum.iter_mut().enumerate() {
            let mut sum = Zero::zero();
            for (i, &input_cell) in signal.iter().enumerate() {
                let twiddle = unsafe { *self.twiddles.get_unchecked((k * i) % self.twiddles.len()) };
                sum = sum + twiddle * input_cell;
            }
            *output_cell = sum;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ::test::{random_signal, compare_vectors};
    use ::dft;

    #[test]
    fn test_matches_dft() {
        for len in 1..50 {
            let input = random_signal(len);
            
            let mut expected = input.clone();
            dft(&input, &mut expected);

            let mut actual = input.clone();
            let mut wrapper = DFTAlgorithm::new(len);
            wrapper.process(&input, &mut actual);

            assert!(compare_vectors(&expected, &actual), "length = {}", len);
        }
    }
}