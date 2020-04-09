use std::sync::Arc;
use std::collections::HashMap;

use ::algorithm::butterflies::*;
use ::algorithm::*;
use ::math_utils::{PrimeFactor, PrimeFactors };
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
            let instance = self.plan_fft_with_factors(len, PrimeFactors::compute(len));
            self.algorithm_cache.insert(len, Arc::clone(&instance));
            instance
        }
    }

    fn plan_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Arc<Fft<T>> {
        if let Some(butterfly) = self.plan_new_butterfly(len) {
            butterfly
        } else if factors.is_prime() {
            self.plan_new_prime(len)
        } else {
            self.plan_new_composite(len, factors)
        }
    }
}

impl FftPlannerAvx<f32> {
    fn plan_new_power2_f32(&mut self, len: usize) -> Arc<Fft<f32>> {
        assert!(len.is_power_of_two() && len >= 64); //internal consistency check: we must be a power of two, and len should be more than our largest butterfly

        // We have several multiple-of-two algorithms to choose from. We can use the 2xn, 4x, 8x, and 6xn algorithms
        // if we're in-range to land exactly on the size-64 butterfly, intentionally do so by choosing whichever algorithm will have an inner FFT size of 64
        let power : u32 = match len {
            128 => 2, // 2xn is really slow, so we'd rather use 4xn to go down to 32
            256 => 2,
            512 => 3,
            1024 => 4,
            _ => 3,  // 8xn is the fastest for larger sizes, so if we're not "in range" to hit a butterfly with a single step, use 8xn
        };

        let inner_len = len >> power;
        let inner_fft = self.plan_fft_with_factors(inner_len, PrimeFactors::compute(inner_len));

        // construct the outer FFT with the inner one
        match power {
            2 => wrap_fft(MixedRadix4xnAvx::new_f32(inner_fft).unwrap()),
            3 => wrap_fft(MixedRadix8xnAvx::new(inner_fft).unwrap()),
            4 => wrap_fft(MixedRadix16xnAvx::new(inner_fft).unwrap()),
            _ => panic!(),
        }
    }
    fn plan_new_3xpower2_f32(&mut self, len: usize) -> Arc<Fft<f32>> {
        assert!(len >> len.trailing_zeros() == 3); //internal consistency check. we must be 2^n times 3, and len should be larger than our largest butterfly

        // We have several multiple-of-two algorithms to choose from. We can use the 2xn, 4x, 8x, and 6xn algorithms
        // if we're in-range to land exactly on the size-48 butterfly, intentionally do so by choosing whichever algorithm will have an inner FFT size of 48
        let power : u32 = match len {
            96 => 1,
            192 => 2,
            384 => 3,
            768 => 4,
            _ => 3,  // 8xn is the fastest for larger sizes, so if we're not "in range" to hit a butterfly with a single step, use 8xn
        };

        let inner_len = len >> power;
        let inner_fft = self.plan_fft_with_factors(inner_len, PrimeFactors::compute(inner_len));

        // construct the outer FFT with the inner one
        match power {
            1 => wrap_fft(MixedRadix2xnAvx::new_f32(inner_fft).unwrap()),
            2 => wrap_fft(MixedRadix4xnAvx::new_f32(inner_fft).unwrap()),
            3 => wrap_fft(MixedRadix8xnAvx::new(inner_fft).unwrap()),
            4 => wrap_fft(MixedRadix16xnAvx::new(inner_fft).unwrap()),
            _ => panic!(),
        }
    }
    fn plan_new_bluesteins_f32(&mut self, len: usize) -> Arc<Fft<f32>> {
        assert!(len > 1); // Internal consistency check: The logic in this method doesn't work for a length of 1

        // Plan a step of Bluestein's Algorithm
        // Bluestein's computes a FFT of size `len` by reorganizing it as a FFT of ANY size greter than or equal to len * 2 - 1
        // an obvious choice is the next power of two larger than  len * 2 - 1, but if we can find a smaller FFT that will go faster, we can save a lot of time!
        // We can very efficiently compute any 3 * 2^n, so we can take the next power of 2, divide it by 4, then multiply it by 3. If the result >= len*2 - 1, use it!

        // TODO: if we get the ability to compute arbitrary powers of 3 on the fast path, we can also try max / 16 * 9, max / 32 * 27, max / 128 * 81, to give alternative sizes

        // One caveat is that the size-12 blutterfly is slower than size-16, so we only want to do this if the next power of two is greater than 16
        let min_size = len*2 - 1;
        let max_size = min_size.checked_next_power_of_two().unwrap();

        let potential_3x = max_size / 4 * 3;
        let inner_fft_len = if max_size > 16 && potential_3x >= min_size {
            potential_3x
        } else {
            max_size
        };

        let inner_fft = self.plan_fft_with_factors(inner_fft_len, PrimeFactors::compute(inner_fft_len));
        wrap_fft(BluesteinsAvx::new(len, inner_fft).unwrap())
    }
}

trait MakeFftAvx<T: FFTnum> {
    fn plan_new_butterfly(&self, len: usize) -> Option<Arc<Fft<T>>>;
    fn plan_new_prime(&mut self, len: usize) -> Arc<Fft<T>>;
    fn plan_new_composite(&mut self, len: usize, factors: PrimeFactors) -> Arc<Fft<T>>;
}

impl<T: FFTnum> MakeFftAvx<T> for FftPlannerAvx<T> {
    default fn plan_new_butterfly(&self, _len: usize) -> Option<Arc<Fft<T>>> { unimplemented!(); }
    default fn plan_new_prime(&mut self, _len: usize) -> Arc<Fft<T>> {  unimplemented!(); }
    default fn plan_new_composite(&mut self, _len: usize, _factors: PrimeFactors) -> Arc<Fft<T>> { unimplemented!(); }
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
            12 =>   wrap_fft_some(MixedRadixAvx4x3::new(self.inverse).unwrap()),
            16 =>   wrap_fft_some(MixedRadixAvx4x4::new(self.inverse).unwrap()),
            24 =>   wrap_fft_some(MixedRadixAvx4x6::new(self.inverse).unwrap()),
            32 =>   wrap_fft_some(MixedRadixAvx4x8::new(self.inverse).unwrap()),
            48 =>   wrap_fft_some(MixedRadixAvx4x12::new(self.inverse).unwrap()),
            64 =>   wrap_fft_some(MixedRadixAvx8x8::new(self.inverse).unwrap()),
            _ => None
        }
    }

    fn plan_new_prime(&mut self, len: usize) -> Arc<Fft<f32>> {
        // for prime numbers, we can either use rader's algorithm, which computes an inner FFT if size len - 1
        // or bluestein's algorithm, which computes an inner FFT of any size at least len * 2 - 1. (usually, we pick a power of two)

        // rader's algorithm is faster if len - 1 is very composite, but bluestein's algorithm is faster if len - 1 has very few prime factors 
        // Compute the prime factors of our hpothetical inner FFT. if they're very composite, use rader's algorithm
        let raders_fft_len = len - 1;
        let raders_factors = PrimeFactors::compute(raders_fft_len);

        if (raders_factors.get_total_factor_count() as f32) < (len as f32).log(3.0) {
            // the inner FFT isn't composite enough, so we're doing bluestein's algorithm instead
            self.plan_new_bluesteins_f32(len)
        } else {
            let inner_fft = self.plan_fft_with_factors(raders_fft_len, raders_factors);
            wrap_fft(RadersAlgorithm::new(len, inner_fft))
        }
    }
    fn plan_new_composite(&mut self, len: usize, factors: PrimeFactors) -> Arc<Fft<f32>> {
        // First up: If this is a power of 2, we have a fast path  that goes straight down to butterflies
        if len.is_power_of_two() {
            self.plan_new_power2_f32(len)
        }

        // If this is 3x2^n, we also have a fast path that goes straight down to butterflies
        else if factors.get_power_of_two() > 0 && factors.get_power_of_three() == 1 && factors.get_other_factors().len() == 0 {
            self.plan_new_3xpower2_f32(len)
        }
        
        // If we get to this point, we will need to use bluestein's algorithm or rader's algorithm. before do that, see if there are any powers of 2 we can split off
        // TODO: is it ever worth it to use 2xn here, as opposed to just passing 2xwhatever into bluestein's? needs benchmarking
        else if len.trailing_zeros() > 0 {
            // we will recursively call power-of-two algorithms (16xn, 8xn, 4xn, 2xn) until we either hit a butterfly or all of our powers of two are gone
            // we can handle any power of two, but we want to avoid 2xn if possible, because it's slower than the others
            // so we're going to try to line it up so that we only need to plan a 2xn when we have no other options
            let power : u32 = match len.trailing_zeros() {
                1 => 1,
                2 => 2,
                3 => 3,
                4 => 4,
                _ => 3,  // 8xn is the fastest for larger sizes, so if we're not "in range" of the bottom of the stack, use 8xn
            };

            // update our factors to account for the factors of 2 we're going to strip away, and pass the updated factors to the planner to compute the inner FFT
            let factors = factors.remove_factors(PrimeFactor { value: 2, count: power }).unwrap();
            let inner_fft = self.plan_fft_with_factors(len >> power, factors);

            // construct the outer FFT with the inner one
            match power {
                1 => wrap_fft(MixedRadix2xnAvx::new_f32(inner_fft).unwrap()),
                2 => wrap_fft(MixedRadix4xnAvx::new_f32(inner_fft).unwrap()),
                3 => wrap_fft(MixedRadix8xnAvx::new(inner_fft).unwrap()),
                4 => wrap_fft(MixedRadix16xnAvx::new(inner_fft).unwrap()),
                _ => panic!(),
            }
        }
        else {
            // because this is composite, rader's algorithm isn't an option. so unconditionally apply bluestein's
            self.plan_new_bluesteins_f32(len)
        }
    }
}


impl MakeFftAvx<f64> for FftPlannerAvx<f64> {
    fn plan_new_butterfly(&self, len: usize) -> Option<Arc<Fft<f64>>> {
        match len {
            0|1 =>  wrap_fft_some(DFT::new(len, self.inverse)),
            2 =>    wrap_fft_some(Butterfly2::new(self.inverse)),
            3 =>    wrap_fft_some(Butterfly3::new(self.inverse)),
            4 =>    wrap_fft_some(Butterfly4::new(self.inverse)),
            5 =>    wrap_fft_some(Butterfly5::new(self.inverse)),
            6 =>    wrap_fft_some(Butterfly6::new(self.inverse)),
            7 =>    wrap_fft_some(Butterfly7::new(self.inverse)),
            8 =>    wrap_fft_some(MixedRadix64Avx4x2::new(self.inverse).unwrap()),
            16 =>   wrap_fft_some(MixedRadix64Avx4x4::new(self.inverse).unwrap()),
            32 =>   wrap_fft_some(MixedRadix64Avx4x8::new(self.inverse).unwrap()),
            _ => None
        }
    }

    fn plan_new_prime(&mut self, _len: usize) -> Arc<Fft<f64>> {
        unimplemented!();
    }
    fn plan_new_composite(&mut self, len: usize, factors: PrimeFactors) -> Arc<Fft<f64>> {
        if len.is_power_of_two() {
            if len.trailing_zeros() - 5 <= 1 {
                let factors = factors.remove_factors(PrimeFactor { value: 2, count: 1 }).unwrap();
                let inner_fft = self.plan_fft_with_factors(len >> 1, factors);
                wrap_fft(MixedRadix2xnAvx::new_f64(inner_fft).unwrap())
            } else {
                let factors = factors.remove_factors(PrimeFactor { value: 2, count: 2 }).unwrap();
                let inner_fft = self.plan_fft_with_factors(len >> 2, factors);
                wrap_fft(MixedRadix4xnAvx::new_f64(inner_fft).unwrap())
            }
        }
        else {
            dbg!(len);
            panic!();
        }
    }
}