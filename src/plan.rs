use std::collections::HashMap;
use std::sync::Arc;
use num_integer::gcd;

use crate::common::FFTnum;

use crate::Fft;
use crate::algorithm::*;
use crate::algorithm::butterflies::*;
use crate::algorithm::avx::avx_planner::FftPlannerAvx;

use crate::math_utils::{ PrimeFactors, PrimeFactor };

const MIN_RADIX4_BITS: u32 = 5; // smallest size to consider radix 4 an option is 2^5 = 32
const MAX_RADIX4_BITS: u32 = 16; // largest size to consider radix 4 an option is 2^16 = 65536
const MAX_RADER_PRIME_FACTOR: usize = 7; // don't use Raders if the inner fft length has prime factor larger than this
const MIN_BLUESTEIN_MIXED_RADIX_LEN: usize = 90; // only use mixed radix for the inner fft of Bluestein if length is larger than this

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
    algorithm_cache: HashMap<usize, Arc<dyn Fft<T>>>,
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
            avx_planner: FftPlannerAvx::new(inverse).ok()
        }
    }

    /// Returns a Fft instance which processes signals of size `len`
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_fft(&mut self, len: usize) -> Arc<dyn Fft<T>> {
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

    fn plan_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Arc<dyn Fft<T>> {
        if let Some(instance) = self.algorithm_cache.get(&len) {
            Arc::clone(instance)
        } else {
            let instance = self.plan_new_fft_with_factors(len, factors);
            self.algorithm_cache.insert(len, Arc::clone(&instance));
            instance
        }
    }
    
    fn plan_new_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Arc<dyn Fft<T>> {
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

    fn plan_mixed_radix(&mut self, left_factors: PrimeFactors, right_factors: PrimeFactors) -> Arc<dyn Fft<T>> {
        let left_len = left_factors.get_product();
        let right_len = right_factors.get_product();

        //neither size is a butterfly, so go with the normal algorithm
        let left_fft = self.plan_fft_with_factors(left_len, left_factors);
        let right_fft = self.plan_fft_with_factors(right_len, right_factors);

        //if both left_len and right_len are small, use algorithms optimized for small FFTs
        if left_len < 31 && right_len < 31 {
            // for small FFTs, if gcd is 1, good-thomas is faster
            if gcd(left_len, right_len) == 1 {
                Arc::new(GoodThomasAlgorithmSmall::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            } else {
                Arc::new(MixedRadixSmall::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
        } else {
            

            Arc::new(MixedRadix::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
        }
    }


    // Returns Some(instance) if we have a butterfly available for this size. Returns None if there is no butterfly available for this size
    fn plan_butterfly_algorithm(&mut self, len: usize) -> Option<Arc<dyn Fft<T>>>{
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

    fn plan_prime(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        let inner_fft_len_rader = len - 1;
        let factors = crate::math_utils::distinct_prime_factors(inner_fft_len_rader as u64);
        // If any of the prime factors is too large, Rader's gets slow and Bluestein's is the better choice
        if factors.iter().any(|val| *val as usize > MAX_RADER_PRIME_FACTOR) {
            let inner_fft_len_pow2 = (2 * len - 1).checked_next_power_of_two().unwrap();
            // for long ffts a mixed radix inner fft is faster than a longer radix4
            let min_inner_len = 2 * len - 1;
            let mixed_radix_len = 3*inner_fft_len_pow2/4;
            let inner_fft = if mixed_radix_len >= min_inner_len && len >= MIN_BLUESTEIN_MIXED_RADIX_LEN {
                let inner_factors = crate::math_utils::PrimeFactors::compute(mixed_radix_len);
                self.plan_fft_with_factors(mixed_radix_len, inner_factors)
            }
            else {
                Arc::new(Radix4::new(inner_fft_len_pow2, self.inverse))
            };
            Arc::new(BluesteinsAlgorithm::new(len, inner_fft)) as Arc<dyn Fft<T>>
        }
        else {
            let factors = crate::math_utils::PrimeFactors::compute(inner_fft_len_rader);
            let inner_fft = self.plan_fft_with_factors(inner_fft_len_rader, factors);
            Arc::new(RadersAlgorithm::new(inner_fft)) as Arc<dyn Fft<T>>
        }
    }
}
