use std::collections::HashMap;
use std::sync::Arc;
use num_integer::gcd;

use common::FFTnum;

use Fft;
use algorithm::*;
use algorithm::butterflies::*;
use algorithm::avx::avx_planner::FftPlannerAvx;

use math_utils::{ PrimeFactors, PrimeFactor };

const MIN_RADIX4_BITS: u32 = 5; // smallest size to consider radix 4 an option is 2^5 = 32
const MAX_RADIX4_BITS: u32 = 16; // largest size to consider radix 4 an option is 2^16 = 65536

/// The Fft planner is used to make new Fft algorithm instances.
///
/// RustFFT has several Fft algorithms available; For a given Fft size, the FFTplanner decides which of the
/// available Fft algorithms to use and then initializes them.
///
/// ~~~
/// // Perform a forward Fft of size 1234
/// use std::sync::Arc;
/// use rustfft::FFTplanner;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1234];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1234];
///
/// let mut planner = FFTplanner::new(false);
/// let fft = planner.plan_fft(1234);
/// fft.process(&mut input, &mut output);
/// 
/// // The fft instance returned by the planner is stored behind an `Arc`, so it's cheap to clone
/// let fft_clone = Arc::clone(&fft);
/// ~~~
///
/// If you plan on creating multiple Fft instances, it is recommnded to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across Fft instances wherever possible, saving memory and reducing
/// setup time. (Fft instances created with one planner will never re-use data and buffers with Fft instances created
/// by a different planner)
///
/// Each Fft instance owns `Arc`s to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating Fft instances.
pub struct FFTplanner<T: FFTnum> {
    algorithm_cache: HashMap<usize, Arc<Fft<T>>>,
    inverse: bool,

    // None if this machine doesn't support avx
    avx_planner: Option<FftPlannerAvx<T>>,
}

impl<T: FFTnum> FFTplanner<T> {
    /// Creates a new Fft planner.
    ///
    /// If `inverse` is false, this planner will plan forward FFTs. If `inverse` is true, it will plan inverse FFTs.
    pub fn new(inverse: bool) -> Self {
        Self {
            inverse: inverse,
            algorithm_cache: HashMap::new(),
            avx_planner: None//FftPlannerAvx::new(inverse).ok()
        }
    }

    /// Returns a Fft instance which processes signals of size `len`
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_fft(&mut self, len: usize) -> Arc<Fft<T>> {
        if let Some(avx_planner) = &mut self.avx_planner {
            // If we have an AVX planner, defer to that for all construction needs
            // TODO: eventually, "FFTplanner" could be an enum of different planner types? "ScalarPlanner" etc
            // That way, we wouldn't need to waste memoery storing the scalar planner's algorithm cache when we're not gonna use it
            avx_planner.plan_fft(len)
        } else if let Some(instance) = self.algorithm_cache.get(&len) {
            Arc::clone(instance)
        } else {
            let instance = self.plan_new_fft_with_factors(len, PrimeFactors::compute(len));
            self.algorithm_cache.insert(len, Arc::clone(&instance));
            instance
        }
    }

    fn plan_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Arc<Fft<T>> {
        if let Some(instance) = self.algorithm_cache.get(&len) {
            Arc::clone(instance)
        } else {
            let instance = self.plan_new_fft_with_factors(len, factors);
            self.algorithm_cache.insert(len, Arc::clone(&instance));
            instance
        }
    }
    
    fn plan_new_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Arc<Fft<T>> {
        if let Some(fft_instance) = self.plan_butterfly_algorithm(len) {
            fft_instance
        } else if factors.is_prime() {
            self.plan_prime(len)
        } else if len.trailing_zeros() <= MAX_RADIX4_BITS && len.trailing_zeros() >= MIN_RADIX4_BITS {
            if len.is_power_of_two() {
                Arc::new(Radix4::new(len, self.inverse))
            } else {
                let non_power_of_two = factors.remove_factors(PrimeFactor { value: 2, count: len.trailing_zeros()}).unwrap();
                let power_of_two = PrimeFactors::compute(1 << len.trailing_zeros());
                self.plan_mixed_radix(power_of_two, non_power_of_two)
            }
        } else {
            let (left_factors, right_factors) = factors.partition_factors();
            self.plan_mixed_radix(left_factors, right_factors)
        }
    }

    fn plan_mixed_radix(&mut self, left_factors: PrimeFactors, right_factors: PrimeFactors) -> Arc<Fft<T>> {
        let left_len = left_factors.get_product();
        let right_len = right_factors.get_product();

        //neither size is a butterfly, so go with the normal algorithm
        let left_fft = self.plan_fft_with_factors(left_len, left_factors);
        let right_fft = self.plan_fft_with_factors(right_len, right_factors);

        //if both left_len and right_len are small, use algorithms optimized for small FFTs
        if left_len < 31 && right_len < 31 {
            // for small FFTs, if gcd is 1, good-thomas is faster
            if gcd(left_len, right_len) == 1 {
                Arc::new(GoodThomasAlgorithmSmall::new(left_fft, right_fft)) as Arc<Fft<T>>
            } else {
                Arc::new(MixedRadixSmall::new(left_fft, right_fft)) as Arc<Fft<T>>
            }
        } else {
            

            Arc::new(MixedRadix::new(left_fft, right_fft)) as Arc<Fft<T>>
        }
    }


    // Returns Some(instance) if we have a butterfly available for this size. Returns None if there is no butterfly available for this size
    fn plan_butterfly_algorithm(&mut self, len: usize) -> Option<Arc<Fft<T>>>{
        fn wrap_butterfly<N: FFTnum>(butterfly: impl Fft<N> + 'static) -> Option<Arc<dyn Fft<N>>> {
            Some(Arc::new(butterfly) as Arc<dyn Fft<N>>)
        }

        match len {
            0|1 => wrap_butterfly(DFT::new(len, self.inverse)),
            2 => wrap_butterfly(Butterfly2::new(self.inverse)),
            3 => wrap_butterfly(Butterfly3::new(self.inverse)),
            4 => wrap_butterfly(Butterfly4::new(self.inverse)),
            5 => wrap_butterfly(Butterfly5::new(self.inverse)),
            6 => wrap_butterfly(Butterfly6::new(self.inverse)),
            7 => wrap_butterfly(Butterfly7::new(self.inverse)),
            8 => wrap_butterfly(Butterfly8::new(self.inverse)),
            16 => wrap_butterfly(Butterfly16::new(self.inverse)),
            32 => wrap_butterfly(Butterfly32::new(self.inverse)),
            _ => None,
        }
    }

    fn plan_prime(&mut self, len: usize) -> Arc<Fft<T>> {
        // for prime numbers, we can either use rader's algorithm, which computes an inner FFT if size len - 1
        // or bluestein's algorithm, which computes an inner FFT of any size at least len * 2 - 1. (usually, we pick a power of two)

        // rader's algorithm is faster if len - 1 is very composite, but bluestein's algorithm is faster if len - 1 has very few prime factors 
        // Compute the prime factors of our hpothetical inner FFT. if they're very composite, use rader's algorithm
        let raders_fft_len = len - 1;
        let raders_factors = PrimeFactors::compute(raders_fft_len);

        // "very composite" obviously isn't well-defined criteria, but we can form a pretty good heuristic by comparing the number of prime factors to log5(len)
        // if the number of factors is less, that means there are fewer factors than there would have been if len was a power of 5, and we'll call that "not very composite"
        if (raders_factors.get_total_factor_count() as f32) < (len as f32).log(5.0) {
            // the inner FFT isn't composite enough, so we're doing bluestein's algorithm instead
            let inner_fft_len = (len * 2 - 1).checked_next_power_of_two().unwrap();
            let inner_fft = self.plan_fft_with_factors(inner_fft_len, PrimeFactors::compute(inner_fft_len));

            Arc::new(BluesteinsAlgorithm::new(len, inner_fft)) as Arc<Fft<T>>
        } else {
            let inner_fft = self.plan_fft_with_factors(raders_fft_len, raders_factors);
            Arc::new(RadersAlgorithm::new(len, inner_fft)) as Arc<Fft<T>>
        }
    }
}
