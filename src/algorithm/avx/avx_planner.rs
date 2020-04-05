use std::sync::Arc;
use std::collections::HashMap;

use ::algorithm::butterflies::*;
use ::algorithm::*;
use ::math_utils::{PrimeFactor, prime_factors};
use ::common::FFTnum;
use ::Fft;

use super::*;

fn wrap_fft<T: FFTnum>(butterfly: impl Fft<T> + 'static) -> Arc<dyn Fft<T>> {
    Arc::new(butterfly) as Arc<dyn Fft<T>>
}

fn wrap_fft_some<T: FFTnum>(butterfly: impl Fft<T> + 'static) -> Option<Arc<dyn Fft<T>>> {
    Some(Arc::new(butterfly) as Arc<dyn Fft<T>>)
}

pub struct FftPlannerAvx<T: FFTnum> {
    algorithm_cache: HashMap<usize, Arc<Fft<T>>>,
    inverse: bool,
}
impl<T: FFTnum> FftPlannerAvx<T> {
    pub fn new(inverse: bool) -> Result<Self, ()> {
        // Eventually we might make AVX algorithms that don't also require FMA.
        // If that happens, we can only check for AVX here? seems like a pretty low-priority addition
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            Ok(Self {
                algorithm_cache: HashMap::new(),
                inverse,
            })
        } else {
            Err(())
        }
    }

    pub fn plan_fft(&mut self, len: usize) -> Arc<Fft<T>> {
        if let Some(instance) = self.algorithm_cache.get(&len) {
            Arc::clone(instance)
        } else {
            let instance = self.plan_fft_with_factors(len, prime_factors(len));
            self.algorithm_cache.insert(len, Arc::clone(&instance));
            instance
        }
    }

    fn plan_fft_with_factors(&mut self, len: usize, factors: Vec<PrimeFactor>) -> Arc<Fft<T>> {
        if let Some(butterfly) = self.plan_new_butterfly(len) {
            println!("planned butterfly for len={}", len);
            butterfly
        }
        else if factors.len() == 1 && factors[0].count == 1 {
            self.plan_new_prime(len)
        }
        else {
            self.plan_new_composite(len, factors)
        }
    }
}

trait MakeFftAvx<T: FFTnum> {
    fn plan_new_butterfly(&self, len: usize) -> Option<Arc<Fft<T>>>;
    fn plan_new_prime(&mut self, len: usize) -> Arc<Fft<T>>;
    fn plan_new_composite(&mut self, len: usize, factors: Vec<PrimeFactor>) -> Arc<Fft<T>>;
}

impl<T: FFTnum> MakeFftAvx<T> for FftPlannerAvx<T> {
    default fn plan_new_butterfly(&self, _len: usize) -> Option<Arc<Fft<T>>> {
        unimplemented!();
    }
    default fn plan_new_prime(&mut self, _len: usize) -> Arc<Fft<T>> {
        unimplemented!();
    }
    default fn plan_new_composite(&mut self, _len: usize, _factors: Vec<PrimeFactor>) -> Arc<Fft<T>> {
        unimplemented!();
    }
}
impl MakeFftAvx<f32> for FftPlannerAvx<f32> {
    fn plan_new_butterfly(&self, len: usize) -> Option<Arc<Fft<f32>>> {
        match len {
            0|1 =>  wrap_fft_some(DFT::new(len, self.inverse)),
            2 =>    wrap_fft_some(Butterfly2::new(self.inverse)),
            3 =>    wrap_fft_some(Butterfly3::new(self.inverse)),
            4 =>    wrap_fft_some(Butterfly4::new(self.inverse)),
            5 =>    wrap_fft_some(Butterfly5::new(self.inverse)),
            6 =>    wrap_fft_some(Butterfly6::new(self.inverse)),
            7 =>    wrap_fft_some(Butterfly7::new(self.inverse)),
            8 =>    wrap_fft_some(MixedRadixAvx4x2::new(self.inverse).unwrap()),
            16 =>   wrap_fft_some(MixedRadixAvx4x4::new(self.inverse).unwrap()),
            32 =>   wrap_fft_some(MixedRadixAvx4x8::new(self.inverse).unwrap()),
            64 =>   wrap_fft_some(MixedRadixAvx8x8::new(self.inverse).unwrap()),
            _ => None
        }
    }

    fn plan_new_prime(&mut self, len: usize) -> Arc<Fft<f32>> {
        // for prime numbers, we can either use rader's algorithm, which computes an inner FFT if size len - 1
        // or bluestein's algorithm, which computes an inner FFT of any size at least len * 2 - 1. (usually, we pick a power of two)

        // rader's algorithm is faster if len - 1 is very composite, but bluestein's algorithm is faster if len - 1 has very few prime factors 
        // Compute the prime factors of our hpothetical inner FFT. if they're very composite, use rader's algorithm
        let inner_fft_len = len - 1;
        let inner_factors = prime_factors(inner_fft_len);

        // similar to the main planner, we're going 
        let total_factor_count : u32 = inner_factors.iter().map(|factor| factor.count).sum();
        if (total_factor_count as f32) < (len as f32).log(3.0) {
            println!("planning bluestein's for len={}", len);
            // the inner FFT isn't composite enough, so we're doing bluestein's algorithm instead
            // TODO: instead of unconditionally using a power of 2, investigate 3x2^n butterflies like 24 and 48, and use them to build 
            let inner_fft_len = (len * 2 - 1).checked_next_power_of_two().unwrap();
            let inner_fft = self.plan_fft_with_factors(inner_fft_len, vec![PrimeFactor { value: 2, count: inner_fft_len.trailing_zeros() }]);

            wrap_fft(BluesteinsAvx::new(len, inner_fft).unwrap())
        } else {
            println!("planning rader's for len={}", len);
            let inner_fft = self.plan_fft_with_factors(inner_fft_len, inner_factors);
            wrap_fft(RadersAlgorithm::new(len, inner_fft))
        }
    }
    fn plan_new_composite(&mut self, len: usize, mut factors: Vec<PrimeFactor>) -> Arc<Fft<f32>> {
        // If we have factors of 2, split them off. Otherwise, use bluestein's algorithm
        // TODO: eventually we should be able to incorporate factors of 3 and 5
        let trailing_zeros = len.trailing_zeros();
        if trailing_zeros > 0 {
            println!("Planning power of two for len = {}", len);
            // we will recursively call power-of-two algorithms (16xn, 8xn, 4xn, 2xn) until we either hit a butterfly or a bluestein's step
            // we can handle any power of two, but we want to avoid 2xn if possible, because it's slower than the others
            // so we're going to try to line it up so that we only need to plan a 2xn when we have no other options
            let fft_power : u32 = if len.is_power_of_two() {
                if trailing_zeros <= 6 {
                    return self.plan_new_butterfly(len).unwrap();
                }

                match trailing_zeros - 6 {
                    1 => 2,
                    2 => 2,
                    3 => 3,
                    4 => 4,
                    _ => 3, // 8xn is the fastest for larger sizes, so if we're not "in range" of the bottom of the stack, use 8xn
                }
            } else {
                match trailing_zeros {
                    1 => 1,
                    2 => 2,
                    3 => 3,
                    4 => 4,
                    _ => 3,  // 8xn is the fastest for larger sizes, so if we're not "in range" of the bottom of the stack, use 8xn
                }
            };

            println!("FFT power: {}", fft_power);

            // update our factors to account for the factors of 2 we're going to strip away, and pass the updated factors to the planner to compute the inner FFT
            factors[0].count -= fft_power;
            if factors[0].count == 0 {
                factors.remove(0);
            }
            dbg!(&factors);
            dbg!(len >> fft_power);
            let inner_fft = self.plan_fft_with_factors(len >> fft_power, factors);

            // construct the outer FFT with the inner one
            match fft_power {
                1 => wrap_fft(MixedRadix2xnAvx::new(inner_fft).unwrap()),
                2 => wrap_fft(MixedRadix4xnAvx::new(inner_fft).unwrap()),
                3 => wrap_fft(MixedRadix8xnAvx::new(inner_fft).unwrap()),
                4 => wrap_fft(MixedRadix16xnAvx::new(inner_fft).unwrap()),
                _ => panic!(),
            }
        }
        else {
            println!("COMPOSITE: Planning bluestein's for len = {}", len);
            let inner_fft_len = (len * 2 - 1).checked_next_power_of_two().unwrap();
            let inner_fft = self.plan_fft_with_factors(inner_fft_len, vec![PrimeFactor { value: 2, count: inner_fft_len.trailing_zeros() }]);

            wrap_fft(BluesteinsAvx::new(len, inner_fft).unwrap())
        }
    }
}