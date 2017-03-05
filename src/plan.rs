use std::collections::HashMap;
use std::rc::Rc;

use common::FFTnum;

use algorithm::*;
use math_utils;


const MIN_RADERS_SIZE: usize = 100;
const BUTTERFLIES: [usize; 6] = [2, 3, 4, 5, 6, 7];
const COMPOSITE_BUTTERFLIES: [usize; 2] = [4, 6];

pub struct Planner<T> {
    inverse: bool,
    algorithm_cache: HashMap<usize, Rc<FFTAlgorithm<T>>>,
}

impl<T: FFTnum> Planner<T> {
    pub fn new(inverse: bool) -> Self {
        Self {
            inverse: inverse,
            algorithm_cache: HashMap::new(),
        }
    }

    pub fn plan_fft(&mut self, len: usize) -> Rc<FFTAlgorithm<T>> {
        if len < 2 {
            Rc::new(NoopAlgorithm { len: len }) as Rc<FFTAlgorithm<T>>
        } else {
            let factors = math_utils::prime_factors(len);
            self.plan_fft_with_factors(len, &factors)
        }
    }

    fn plan_fft_with_factors(&mut self, len: usize, factors: &[usize]) -> Rc<FFTAlgorithm<T>> {
        if self.algorithm_cache.contains_key(&len) {
            self.algorithm_cache.get(&len).unwrap().clone()
        } else {
            let result = if factors.len() == 1 || COMPOSITE_BUTTERFLIES.contains(&len) {
                self.plan_fft_single_factor(len)
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
                        -> Rc<FFTAlgorithm<T>> {

        let left_is_butterfly = BUTTERFLIES.contains(&left_len);
        let right_is_butterfly = BUTTERFLIES.contains(&right_len);

        //if both left_len and right_len are butterflies, use a mixed radix implementation specialized for butterfly sub-FFTs
        if left_is_butterfly && right_is_butterfly {
            let left_fft = self.plan_butterfly(left_len);
            let right_fft = self.plan_butterfly(right_len);

            Rc::new(MixedRadixDoubleButterfly::new(left_fft, right_fft, self.inverse)) as
            Rc<FFTAlgorithm<T>>
        } else {
            //neither size is a butterfly, so go with the normal algorithm
            let left_fft = self.plan_fft_with_factors(left_len, left_factors);
            let right_fft = self.plan_fft_with_factors(right_len, right_factors);

            Rc::new(MixedRadix::new(left_fft, right_fft, self.inverse)) as Rc<FFTAlgorithm<T>>
        }
    }


    fn plan_fft_single_factor(&mut self, len: usize) -> Rc<FFTAlgorithm<T>> {
        match len {
            0 => Rc::new(NoopAlgorithm { len: 0 }) as Rc<FFTAlgorithm<T>>,
            1 => Rc::new(NoopAlgorithm { len: 1 }) as Rc<FFTAlgorithm<T>>,
            2 => Rc::new(butterflies::Butterfly2 {}) as Rc<FFTAlgorithm<T>>,
            3 => Rc::new(butterflies::Butterfly3::new(self.inverse)) as Rc<FFTAlgorithm<T>>,
            4 => Rc::new(butterflies::Butterfly4::new(self.inverse)) as Rc<FFTAlgorithm<T>>,
            5 => Rc::new(butterflies::Butterfly5::new(self.inverse)) as Rc<FFTAlgorithm<T>>,
            6 => Rc::new(butterflies::Butterfly6::new(self.inverse)) as Rc<FFTAlgorithm<T>>,
            7 => Rc::new(butterflies::Butterfly7::new(self.inverse)) as Rc<FFTAlgorithm<T>>,
            _ => {
                if len >= MIN_RADERS_SIZE {
                    self.plan_prime(len)
                } else {
                    //we have a prime factor, but it's not large enough for raders algorithm to be worth it
                    Rc::new(DFTAlgorithm::new(len, self.inverse)) as Rc<FFTAlgorithm<T>>
                }
            }
        }
    }

    fn plan_prime(&mut self, len: usize) -> Rc<FFTAlgorithm<T>> {
        let inner_fft_len = len - 1;
        let factors = math_utils::prime_factors(inner_fft_len);

        let inner_fft = self.plan_fft_with_factors(inner_fft_len, &factors);

        Rc::new(RadersAlgorithm::new(len, inner_fft, self.inverse)) as Rc<FFTAlgorithm<T>>
    }

    fn plan_butterfly(&self, len: usize) -> ButterflyEnum<T> {
        match len {
            2 => ButterflyEnum::Butterfly2(butterflies::Butterfly2 {}),
            3 => ButterflyEnum::Butterfly3(butterflies::Butterfly3::new(self.inverse)),
            4 => ButterflyEnum::Butterfly4(butterflies::Butterfly4::new(self.inverse)),
            5 => ButterflyEnum::Butterfly5(butterflies::Butterfly5::new(self.inverse)),
            6 => ButterflyEnum::Butterfly6(butterflies::Butterfly6::new(self.inverse)),
            7 => ButterflyEnum::Butterfly7(butterflies::Butterfly7::new(self.inverse)),
            _ => panic!("Invalid butterfly size: {}", len),
        }
    }
}
