use common::FFTnum;

use algorithm::*;
use math_utils;

pub fn plan_fft<T: FFTnum>(len: usize, inverse: bool) -> Box<FFTAlgorithm<T>> {
    if len < 2 {
        Box::new(NoopAlgorithm {}) as Box<FFTAlgorithm<T>>
    } else if len.is_power_of_two() {
        Box::new(Radix4::new(len, inverse)) as Box<FFTAlgorithm<T>>
    } else {
        let factors = math_utils::prime_factors(len);
        plan_fft_with_factors(len, &factors, inverse)
    }
}


const MIN_RADERS_SIZE: usize = 100;
const BUTTERFLIES: [usize; 5] = [2, 3, 4, 5, 6];
const COMPOSITE_BUTTERFLIES: [usize; 2] = [4, 6];

fn plan_fft_with_factors<T: FFTnum>(len: usize, factors: &[usize], inverse: bool) -> Box<FFTAlgorithm<T>> {
    if factors.len() == 1 || COMPOSITE_BUTTERFLIES.contains(&len) {
        plan_fft_single_factor(len, inverse)
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

            plan_mixed_radix(sqrt, &sqrt_factors, sqrt, &sqrt_factors, inverse)
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
            plan_mixed_radix(product, left_factors, len / product, right_factors, inverse)
        }
    }
}

fn plan_mixed_radix<T: FFTnum>(left_len: usize, left_factors: &[usize], right_len: usize, right_factors: &[usize], inverse: bool) -> Box<FFTAlgorithm<T>> {
    let left_is_butterfly = BUTTERFLIES.contains(&left_len);
    let right_is_butterfly = BUTTERFLIES.contains(&right_len);

    //if both left_len and right_len are butterflies, use a mixed radix implementation specialized for butterfly sub-FFTs
    if left_is_butterfly && right_is_butterfly {
        let left_fft = plan_butterfly(left_len, inverse);
        let right_fft = plan_butterfly(right_len, inverse);

        Box::new(MixedRadixDoubleButterfly::new(left_len, left_fft, right_len, right_fft, inverse)) as Box<FFTAlgorithm<T>>
    } else if !left_is_butterfly && ! right_is_butterfly {
        //neither size is a butterfly, so go with the normal algorithm
        let left_fft = plan_fft_with_factors(left_len, left_factors, inverse);
        let right_fft = plan_fft_with_factors(right_len, right_factors, inverse);

        Box::new(MixedRadix::new(left_len, left_fft, right_len, right_fft, inverse)) as Box<FFTAlgorithm<T>>
    } else if left_is_butterfly {
        //the left is a butterfly, but not the right
        let butterfly_len = left_len;
        let inner_len = right_len;
        let inner_factors = right_factors;

        let butterfly_fft = plan_butterfly(butterfly_len, inverse);
        let inner_fft = plan_fft_with_factors(inner_len, inner_factors, inverse);

        Box::new(MixedRadixSingleButterfly::new(inner_len, inner_fft, butterfly_len, butterfly_fft, inverse)) as Box<FFTAlgorithm<T>>
    } else {
        //the right is a butterfly, but not the left
        let butterfly_len = right_len;
        let inner_len = left_len;
        let inner_factors = left_factors;

        let butterfly_fft = plan_butterfly(butterfly_len, inverse);
        let inner_fft = plan_fft_with_factors(inner_len, inner_factors, inverse);

        Box::new(MixedRadixSingleButterfly::new(inner_len, inner_fft, butterfly_len, butterfly_fft, inverse)) as Box<FFTAlgorithm<T>>
    }
}

fn plan_fft_single_factor<T: FFTnum>(len: usize, inverse: bool) -> Box<FFTAlgorithm<T>> {
    match len {
        0 => Box::new(NoopAlgorithm {}) as Box<FFTAlgorithm<T>>,
        1 => Box::new(NoopAlgorithm {}) as Box<FFTAlgorithm<T>>,
        2 => Box::new(butterflies::Butterfly2 {}) as Box<FFTAlgorithm<T>>,
        3 => Box::new(butterflies::Butterfly3::new(inverse)) as Box<FFTAlgorithm<T>>,
        4 => Box::new(butterflies::Butterfly4::new(inverse)) as Box<FFTAlgorithm<T>>,
        5 => Box::new(butterflies::Butterfly5::new(inverse)) as Box<FFTAlgorithm<T>>,
        6 => Box::new(butterflies::Butterfly6::new(inverse)) as Box<FFTAlgorithm<T>>,
        _ => if len >= MIN_RADERS_SIZE {
                Box::new(RadersAlgorithm::new(len, inverse)) as Box<FFTAlgorithm<T>>
            } else {
                //we have a prime factor, but it's not large enough for raders algorithm to be worth it
                Box::new(DFTAlgorithm::new(len, inverse)) as Box<FFTAlgorithm<T>>
            },
    }
}

fn plan_butterfly<T: FFTnum>(len: usize, inverse: bool) -> ButterflyEnum<T> {
    match len {
            2 => ButterflyEnum::Butterfly2(butterflies::Butterfly2{}),
            3 => ButterflyEnum::Butterfly3(butterflies::Butterfly3::new(inverse)),
            4 => ButterflyEnum::Butterfly4(butterflies::Butterfly4::new(inverse)),
            5 => ButterflyEnum::Butterfly5(butterflies::Butterfly5::new(inverse)),
            6 => ButterflyEnum::Butterfly6(butterflies::Butterfly6::new(inverse)),
            _ => panic!("Invalid butterfly size: {}", len)
        }
}