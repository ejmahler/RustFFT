use std::collections::HashMap;
use std::rc::Rc;

use common::FFTnum;

use FFT;
use algorithm::*;
use algorithm::butterflies::*;

use math_utils;


const MIN_RADIX4_BITS: u32 = 8; // minimum size to consider radix 4 an option is 2^8 = 256
const BUTTERFLIES: [usize; 8] = [2, 3, 4, 5, 6, 7, 8, 16];
const COMPOSITE_BUTTERFLIES: [usize; 4] = [4, 6, 8, 16];

pub struct FFTplanner<T> {
    inverse: bool,
    algorithm_cache: HashMap<usize, Rc<FFT<T>>>,
    butterfly_cache: HashMap<usize, Rc<FFTButterfly<T>>>,
}

impl<T: FFTnum> FFTplanner<T> {
    pub fn new(inverse: bool) -> Self {
        FFTplanner {
            inverse: inverse,
            algorithm_cache: HashMap::new(),
            butterfly_cache: HashMap::new(),
        }
    }

    pub fn plan_fft(&mut self, len: usize) -> Rc<FFT<T>> {
        if len < 2 {
            Rc::new(DFT::new(len, self.inverse)) as Rc<FFT<T>>
        } else {
            let factors = math_utils::prime_factors(len);
            self.plan_fft_with_factors(len, &factors)
        }
    }

    fn plan_butterfly(&mut self, len: usize) -> Rc<FFTButterfly<T>> {
        let inverse = self.inverse;
        self.butterfly_cache.entry(len).or_insert_with(|| 
            match len {
                2 => Rc::new(Butterfly2 {}),
                3 => Rc::new(Butterfly3::new(inverse)),
                4 => Rc::new(Butterfly4::new(inverse)),
                5 => Rc::new(Butterfly5::new(inverse)),
                6 => Rc::new(Butterfly6::new(inverse)),
                7 => Rc::new(Butterfly7::new(inverse)),
                8 => Rc::new(Butterfly8::new(inverse)),
                16 => Rc::new(Butterfly16::new(inverse)),
                _ => panic!("Invalid butterfly size: {}", len),
            }
        ).clone()
    }
    
    fn plan_fft_with_factors(&mut self, len: usize, factors: &[usize]) -> Rc<FFT<T>> {
        if self.algorithm_cache.contains_key(&len) {
            self.algorithm_cache.get(&len).unwrap().clone()
        } else {
            let result = if factors.len() == 1 || COMPOSITE_BUTTERFLIES.contains(&len) {
                self.plan_fft_single_factor(len)

            } else if len.trailing_zeros() >= MIN_RADIX4_BITS {
                //the number of trailing zeroes in len is the number of `2` factors
                //ie if len = 2048 * n, len.trailing_zeros() will equal 11 because 2^11 == 2048

                if len.is_power_of_two() {
                    Rc::new(Radix4::new(len, self.inverse))
                } else {
                    let left_len = 1 << len.trailing_zeros();
                    let right_len = len / left_len;

                    let (left_factors, right_factors) = factors.split_at(len.trailing_zeros() as usize);

                    self.plan_mixed_radix(left_len, left_factors, right_len, right_factors)
                }

            } else {
                let sqrt = (len as f32).sqrt() as usize;
                if sqrt * sqrt == len {
                    // since len is a perfect square, each of its prime factors is duplicated.
                    // since we know they're sorted, we can loop through them in chunks of 2 and keep one out of each chunk
                    // if the stride iterator ever becomes stabilized, it'll be cleaner to use that instead of chunks
                    let mut sqrt_factors = Vec::with_capacity(factors.len() / 2);
                    for chunk in factors.chunks(2) {
                        sqrt_factors.push(chunk[0]);
                    }

                    self.plan_mixed_radix(sqrt, &sqrt_factors, sqrt, &sqrt_factors)
                } else {
                    //len isn't a perfect square. greedily take factors from the list until both sides are as close as possible to sqrt(len)
                    //TODO: We can probably make this more optimal by using a more sophisticated non-greedy algorithm
                    let mut product = 1;
                    let mut second_half_index = 1;
                    for (i, factor) in factors.iter().enumerate() {
                        if product * *factor > sqrt {
                            second_half_index = i;
                            break;
                        } else {
                            product = product * *factor;
                        }
                    }

                    //we now know that product is the largest it can be without being greater than len / product
                    //there's one more thing we can try to make them closer together -- if product * factors[index] < len / product,
                    if product * factors[second_half_index] < len / product {
                        product = product * factors[second_half_index];
                        second_half_index = second_half_index + 1;
                    }

                    //we now have our two FFT sizes: product and product / len
                    let (left_factors, right_factors) = factors.split_at(second_half_index);
                    self.plan_mixed_radix(product, left_factors, len / product, right_factors)
                }
            };
            self.algorithm_cache.insert(len, result.clone());
            result
        }
    }

    fn plan_mixed_radix(&mut self,
                        left_len: usize,
                        left_factors: &[usize],
                        right_len: usize,
                        right_factors: &[usize])
                        -> Rc<FFT<T>> {

        let left_is_butterfly = BUTTERFLIES.contains(&left_len);
        let right_is_butterfly = BUTTERFLIES.contains(&right_len);

        //if both left_len and right_len are butterflies, use a mixed radix implementation specialized for butterfly sub-FFTs
        if left_is_butterfly && right_is_butterfly {
            let left_fft = self.plan_butterfly(left_len);
            let right_fft = self.plan_butterfly(right_len);

            Rc::new(MixedRadixDoubleButterfly::new(left_fft, right_fft, self.inverse)) as
            Rc<FFT<T>>
        } else {
            //neither size is a butterfly, so go with the normal algorithm
            let left_fft = self.plan_fft_with_factors(left_len, left_factors);
            let right_fft = self.plan_fft_with_factors(right_len, right_factors);

            Rc::new(MixedRadix::new(left_fft, right_fft, self.inverse)) as Rc<FFT<T>>
        }
    }


    fn plan_fft_single_factor(&mut self, len: usize) -> Rc<FFT<T>> {
        match len {
            0...1 => Rc::new(DFT::new(len, self.inverse)) as Rc<FFT<T>>,
            2 => Rc::new(butterflies::Butterfly2 {}) as Rc<FFT<T>>,
            3 => Rc::new(butterflies::Butterfly3::new(self.inverse)) as Rc<FFT<T>>,
            4 => Rc::new(butterflies::Butterfly4::new(self.inverse)) as Rc<FFT<T>>,
            5 => Rc::new(butterflies::Butterfly5::new(self.inverse)) as Rc<FFT<T>>,
            6 => Rc::new(butterflies::Butterfly6::new(self.inverse)) as Rc<FFT<T>>,
            7 => Rc::new(butterflies::Butterfly7::new(self.inverse)) as Rc<FFT<T>>,
            8 => Rc::new(butterflies::Butterfly8::new(self.inverse)) as Rc<FFT<T>>,
            16 => Rc::new(butterflies::Butterfly16::new(self.inverse)) as Rc<FFT<T>>,
            _ => self.plan_prime(len),
        }
    }

    fn plan_prime(&mut self, len: usize) -> Rc<FFT<T>> {
        let inner_fft_len = len - 1;
        let factors = math_utils::prime_factors(inner_fft_len);

        let inner_fft = self.plan_fft_with_factors(inner_fft_len, &factors);

        Rc::new(RadersAlgorithm::new(len, inner_fft, self.inverse)) as Rc<FFT<T>>
    }
}
