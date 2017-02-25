use num::{Complex, FromPrimitive, Signed, Zero};

use algorithm::FFTAlgorithm;
use twiddles;

pub struct DFTAlgorithm<T> {
    twiddles: Vec<Complex<T>>,
}

impl<T> DFTAlgorithm<T>
    where T: Signed + FromPrimitive + Copy
{
    pub fn new(len: usize) -> Self {
        Self {
            twiddles: twiddles::generate_twiddle_factors(len, false),
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