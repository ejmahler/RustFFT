use num_integer::gcd;
use std::collections::HashMap;
use std::sync::Arc;

use crate::common::FFTnum;

use crate::algorithm::butterflies::*;
use crate::algorithm::*;
use crate::Fft;

use crate::FftPlannerAvx;

use crate::math_utils::{PrimeFactor, PrimeFactors};

/// The FFT planner is used to make new FFT algorithm instances.
///
/// RustFFT has several FFT algorithms available. For a given FFT size, the `FftPlanner` decides which of the
/// available FFT algorithms to use and then initializes them.
///
/// ~~~
/// // Perform a forward Fft of size 1234
/// use std::sync::Arc;
/// use rustfft::{FftPlanner, num_complex::Complex};
///
/// let mut planner = FftPlanner::new(false);
/// let fft = planner.plan_fft(1234);
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
/// fft.process_inplace(&mut buffer);
///
/// // The FFT instance returned by the planner has the type `Arc<dyn Fft<T>>`,
/// // where T is the numeric type, ie f32 or f64, so it's cheap to clone
/// let fft_clone = Arc::clone(&fft);
/// ~~~
///
/// If you plan on creating multiple FFT instances, it is recommnded to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
/// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
/// by a different planner)
///
/// Each FFT instance owns [`Arc`s](std::sync::Arc) to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating Fft instances.
pub enum FftPlanner<T: FFTnum> {
    Scalar(FftPlannerScalar<T>),
    Avx(FftPlannerAvx<T>),
}
impl<T: FFTnum> FftPlanner<T> {
    /// Creates a new `FftPlanner` instance. It detects if AVX is supported on the current machine. If it is, it will plan AVX-accelerated FFTs.
    /// If AVX isn't supported, it will seamlessly fall back to planning non-SIMD FFTs.
    ///
    /// If `inverse` is false, this planner will plan forward FFTs. If `inverse` is true, it will plan inverse FFTs.
    pub fn new(inverse: bool) -> Self {
        if let Ok(avx_planner) = FftPlannerAvx::new(inverse) {
            Self::Avx(avx_planner)
        } else {
            Self::Scalar(FftPlannerScalar::new(inverse))
        }
    }

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        match self {
            Self::Scalar(scalar_planner) => scalar_planner.plan_fft(len),
            Self::Avx(avx_planner) => avx_planner.plan_fft(len),
        }
    }
}

const MIN_RADIX4_BITS: u32 = 5; // smallest size to consider radix 4 an option is 2^5 = 32
const MAX_RADIX4_BITS: u32 = 16; // largest size to consider radix 4 an option is 2^16 = 65536
const MAX_RADER_PRIME_FACTOR: usize = 23; // don't use Raders if the inner fft length has prime factor larger than this
const MIN_BLUESTEIN_MIXED_RADIX_LEN: usize = 90; // only use mixed radix for the inner fft of Bluestein if length is larger than this

<<<<<<< HEAD
/// The Scalar FFT planner creates new FFT algorithm instances using non-SIMD algorithms.
=======
#[derive(Debug, std::cmp::PartialEq)]
pub enum Recipe {
    DFT(usize),
    MixedRadix { left_fft: Box<Recipe>, right_fft: Box<Recipe>},
    GoodThomas { left_fft: Box<Recipe>, right_fft: Box<Recipe>},
    MixedRadixDoubleButterfly(usize, usize),
    GoodThomasDoubleButterfly(usize, usize),
    Rader { len: usize, inner_fft: Box<Recipe>},
    Bluestein { len: usize, inner_fft: Box<Recipe>},
    Radix4(usize),
    Butterfly(usize),
}

macro_rules! butterfly {
    ($len:expr, $inverse:expr) => {
        match $len {
            2 => Arc::new(Butterfly2::new($inverse)),
            3 => Arc::new(Butterfly3::new($inverse)),
            4 => Arc::new(Butterfly4::new($inverse)),
            5 => Arc::new(Butterfly5::new($inverse)),
            6 => Arc::new(Butterfly6::new($inverse)),
            7 => Arc::new(Butterfly7::new($inverse)),
            8 => Arc::new(Butterfly8::new($inverse)),
            11 => Arc::new(Butterfly11::new($inverse)),
            13 => Arc::new(Butterfly13::new($inverse)),
            16 => Arc::new(Butterfly16::new($inverse)),
            17 => Arc::new(Butterfly17::new($inverse)),
            19 => Arc::new(Butterfly19::new($inverse)),
            23 => Arc::new(Butterfly23::new($inverse)),
            29 => Arc::new(Butterfly29::new($inverse)),
            31 => Arc::new(Butterfly31::new($inverse)),
            32 => Arc::new(Butterfly32::new($inverse)),
            _ => panic!("Invalid butterfly size: {}", $len),
        }
    };
}

/// The FFT planner is used to make new FFT algorithm instances.
>>>>>>> db2c653... WIP adapt new planner
///
/// RustFFT has several FFT algorithms available. For a given FFT size, the `FftPlannerScalar` decides which of the
/// available FFT algorithms to use and then initializes them.
///
/// Use `FftPlannerScalar` instead of [`FftPlanner`](crate::FftPlanner) or [`FftPlannerAvx`](crate::FftPlannerAvx) when you want to explicitly opt out of using any SIMD-accelerated algorithms.
///
/// ~~~
/// // Perform a forward Fft of size 1234
/// use std::sync::Arc;
/// use rustfft::{FftPlannerScalar, num_complex::Complex};
///
/// let mut planner = FftPlannerScalar::new(false);
/// let fft = planner.plan_fft(1234);
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
/// fft.process_inplace(&mut buffer);
///
/// // The FFT instance returned by the planner has the type `Arc<dyn Fft<T>>`,
/// // where T is the numeric type, ie f32 or f64, so it's cheap to clone
/// let fft_clone = Arc::clone(&fft);
/// ~~~
///
/// If you plan on creating multiple FFT instances, it is recommnded to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
/// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
/// by a different planner)
///
/// Each FFT instance owns [`Arc`s](std::sync::Arc) to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating Fft instances.
pub struct FftPlannerScalar<T: FFTnum> {
    algorithm_cache: HashMap<usize, Arc<dyn Fft<T>>>,
    inverse: bool,
}

impl<T: FFTnum> FftPlannerScalar<T> {
    /// Creates a new `FftPlannerScalar` instance.
    ///
    /// If `inverse` is false, this planner will plan forward FFTs. If `inverse` is true, it will plan inverse FFTs.
    pub fn new(inverse: bool) -> Self {
        Self {
            inverse,
            algorithm_cache: HashMap::new(),
        }
    }

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        if let Some(instance) = self.algorithm_cache.get(&len) {
            Arc::clone(instance)
        } else {
            let recipe = self.make_recipe_for_len(len);
            let fft = self.construct_fft(plan);
            self.algorithm_cache.insert(len, Arc::clone(&fft));
            fft
        }
    }

    // Make a recipe for a length
    fn make_recipe_for_len(&mut self, len: usize) -> Recipe {
        if len < 2 {
            Recipe::DFT(len)
        } else {    
            let factors = PrimeFactors::compute(len);
            self.make_recipe_from_factors(len, &factors)
        }
    }

    // Make a recipe for the given prime factors
    fn make_recipe_from_factors(&mut self, len: usize, factors: &[usize]) -> Recipe {
        if factors.len() == 1 || COMPOSITE_BUTTERFLIES.contains(&len) {
            //the length is either a prime or matches a butterfly
            self.make_recipe_for_single_factor(len)
        } else if len.trailing_zeros() <= MAX_RADIX4_BITS && len.trailing_zeros() >= MIN_RADIX4_BITS && len.is_power_of_two(){
            //the length is a power of two in the range where Radix4 is the fastest option.
            Recipe::Radix4(len)
        } else {
            self.make_recipe_for_mixed_radix(len, &factors)
        }
    }

    fn make_recipe_for_mixed_radix(&mut self, len: usize, factors: &[usize]) -> Recipe {
        if len.trailing_zeros() <= MAX_RADIX4_BITS && len.trailing_zeros() >= MIN_RADIX4_BITS {
            //the number of trailing zeroes in len is the number of `2` factors
            //ie if len = 2048 * n, len.trailing_zeros() will equal 11 because 2^11 == 2048
            let left_len = 1 << len.trailing_zeros();
            let right_len = len / left_len;
            let (left_factors, right_factors) = factors.split_at(len.trailing_zeros() as usize);
            self.make_recipe_for_mixed_radix_from_factor_lists(left_len, left_factors, right_len, right_factors)
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
                self.make_recipe_for_mixed_radix_from_factor_lists(sqrt, &sqrt_factors, sqrt, &sqrt_factors)
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
                self.make_recipe_for_mixed_radix_from_factor_lists(product, left_factors, len / product, right_factors)
            }
        }
    }

    // Make a recipe using mixed radix
    fn make_recipe_for_mixed_radix_from_factor_lists(&mut self,
        left_len: usize,
        left_factors: &[usize],
        right_len: usize,
        right_factors: &[usize])
        -> Recipe {

        let left_is_butterfly = BUTTERFLIES.contains(&left_len);
        let right_is_butterfly = BUTTERFLIES.contains(&right_len);

        //if both left_len and right_len are butterflies, use a mixed radix implementation specialized for butterfly sub-FFTs
        if left_is_butterfly && right_is_butterfly {            
            // for butterflies, if gcd is 1, we always want to use good-thomas
            if gcd(left_len, right_len) == 1 {
                Recipe::GoodThomasDoubleButterfly(left_len, right_len)
            } else {
                Recipe::MixedRadixDoubleButterfly(left_len, right_len)
            }
        } else {
            //neither size is a butterfly, so go with the normal algorithm
            let left_fft = Box::new(self.make_recipe_from_factors(left_len, left_factors));
            let right_fft = Box::new(self.make_recipe_from_factors(right_len, right_factors));
            //if gcd(left_len, right_len) == 1 {
            //    Recipe::GoodThomas{left_fft, right_fft}
            //} else {
                Recipe::MixedRadix{left_fft, right_fft}
            //}
        }
    }

    // Make a recipe for a single factor
    fn make_recipe_for_single_factor(&mut self, len: usize) -> Recipe {
        match len {
            0|1=> Recipe::DFT(len),
            2|3|4|5|6|7|8|16|32 => Recipe::Butterfly(len),
            _ => self.make_recipe_for_prime(len),
        }
    }

    // Make a recipe for a prime factor
    fn make_recipe_for_prime(&mut self, len: usize) -> Recipe {
        let inner_fft_len_rader = len - 1;
        let factors = math_utils::prime_factors(inner_fft_len_rader);
        // If any of the prime factors is too large, Rader's gets slow and Bluestein's is the better choice
        if factors.iter().any(|val| *val > MAX_RADER_PRIME_FACTOR) {
            let inner_fft_len_pow2 = (2 * len - 1).checked_next_power_of_two().unwrap();
            // over a certain length, a shorter mixed radix inner fft is faster than a longer radix4
            let min_inner_len = 2 * len - 1;
            let mixed_radix_len = 3*inner_fft_len_pow2/4;
            let inner_fft = if mixed_radix_len >= min_inner_len && len >= MIN_BLUESTEIN_MIXED_RADIX_LEN {
                let inner_factors = math_utils::prime_factors(mixed_radix_len);
                self.make_recipe_from_factors(mixed_radix_len, &inner_factors)
            }
            else {
                Recipe::Radix4(inner_fft_len_pow2)
            };
            Recipe::Bluestein{len, inner_fft: Box::new(inner_fft)}
        }
        else {
            let inner_fft = self.make_recipe_from_factors(inner_fft_len_rader, &factors);
            Recipe::Rader{len, inner_fft: Box::new(inner_fft)}
        }
    }

    // Create the fft from a recipe
    fn construct_fft(&mut self, plan: Recipe) -> Arc<FFT<T>> {
        match plan {
            Recipe::DFT(len) => Arc::new(DFT::new(len, self.inverse)) as Arc<FFT<T>>,
            Recipe::Radix4(len) => Arc::new(Radix4::new(len, self.inverse)) as Arc<FFT<T>>,
            Recipe::Butterfly(len) => {
                butterfly!(len, self.inverse)
            }
            Recipe::MixedRadix { left_fft, right_fft } => {
                let left_fft = self.construct_fft(*left_fft);
                let right_fft = self.construct_fft(*right_fft);
                Arc::new(MixedRadix::new(left_fft, right_fft)) as Arc<FFT<T>>
            },
            Recipe::GoodThomas { left_fft, right_fft } => {
                let left_fft = self.construct_fft(*left_fft);
                let right_fft = self.construct_fft(*right_fft);
                Arc::new(GoodThomasAlgorithm::new(left_fft, right_fft)) as Arc<FFT<T>>
            },
            Recipe::MixedRadixDoubleButterfly(left_len, right_len) => {
                let left_fft = self.construct_butterfly(left_len);
                let right_fft = self.construct_butterfly(right_len);
                Arc::new(MixedRadixDoubleButterfly::new(left_fft, right_fft)) as Arc<FFT<T>>
            },
            Recipe::GoodThomasDoubleButterfly(left_len, right_len) => {
                let left_fft = self.construct_butterfly(left_len);
                let right_fft = self.construct_butterfly(right_len);
                Arc::new(GoodThomasAlgorithmDoubleButterfly::new(left_fft, right_fft)) as Arc<FFT<T>>
            },
            Recipe::Rader { len, inner_fft } => {
                let inner_fft = self.construct_fft(*inner_fft);
                Arc::new(RadersAlgorithm::new(len, inner_fft)) as Arc<FFT<T>>
            },
            Recipe::Bluestein { len , inner_fft } => {
                let inner_fft = self.construct_fft(*inner_fft);
                Arc::new(Bluesteins::new(len, inner_fft)) as Arc<FFT<T>>
            },
        }
    }

    // Create a butterfly
    fn construct_butterfly(&mut self, len: usize) -> Arc<FFTButterfly<T>> {
        let inverse = self.inverse;
        let instance = self.butterfly_cache.entry(len).or_insert_with(|| 
            match len {
                0 | 1 => wrap_butterfly(DFT::new(len, self.inverse)),
                2 => wrap_butterfly(Butterfly2::new(self.inverse)),
                3 => wrap_butterfly(Butterfly3::new(self.inverse)),
                4 => wrap_butterfly(Butterfly4::new(self.inverse)),
                5 => wrap_butterfly(Butterfly5::new(self.inverse)),
                6 => wrap_butterfly(Butterfly6::new(self.inverse)),
                7 => wrap_butterfly(Butterfly7::new(self.inverse)),
                8 => wrap_butterfly(Butterfly8::new(self.inverse)),
                11 => wrap_butterfly(Butterfly11::new(self.inverse)),
                13 => wrap_butterfly(Butterfly13::new(self.inverse)),
                16 => wrap_butterfly(Butterfly16::new(self.inverse)),
                17 => wrap_butterfly(Butterfly17::new(self.inverse)),
                19 => wrap_butterfly(Butterfly19::new(self.inverse)),
                23 => wrap_butterfly(Butterfly23::new(self.inverse)),
                29 => wrap_butterfly(Butterfly29::new(self.inverse)),
                31 => wrap_butterfly(Butterfly31::new(self.inverse)),
                32 => wrap_butterfly(Butterfly32::new(self.inverse)),
            }
        );
        Arc::clone(instance)
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
        } else if len.trailing_zeros() <= MAX_RADIX4_BITS && len.trailing_zeros() >= MIN_RADIX4_BITS
        {
            if len.is_power_of_two() {
                Arc::new(Radix4::new(len, self.inverse))
            } else {
                dbg!(len);
                dbg!(len.trailing_zeros());
                dbg!(&factors);
                let non_power_of_two = factors
                    .remove_factors(PrimeFactor {
                        value: 2,
                        count: len.trailing_zeros(),
                    })
                    .unwrap();
                let power_of_two = PrimeFactors::compute(1 << len.trailing_zeros());
                self.plan_mixed_radix(power_of_two, non_power_of_two)
            }
        } else {
            let (left_factors, right_factors) = factors.partition_factors();
            self.plan_mixed_radix(left_factors, right_factors)
        }
    }

    fn plan_mixed_radix(
        &mut self,
        left_factors: PrimeFactors,
        right_factors: PrimeFactors,
    ) -> Arc<dyn Fft<T>> {
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
    fn plan_butterfly_algorithm(&mut self, len: usize) -> Option<Arc<dyn Fft<T>>> {
        fn wrap_butterfly<N: FFTnum>(butterfly: impl Fft<N> + 'static) -> Option<Arc<dyn Fft<N>>> {
            Some(Arc::new(butterfly) as Arc<dyn Fft<N>>)
        }

        match len {
            0 | 1 => wrap_butterfly(DFT::new(len, self.inverse)),
            2 => wrap_butterfly(Butterfly2::new(self.inverse)),
            3 => wrap_butterfly(Butterfly3::new(self.inverse)),
            4 => wrap_butterfly(Butterfly4::new(self.inverse)),
            5 => wrap_butterfly(Butterfly5::new(self.inverse)),
            6 => wrap_butterfly(Butterfly6::new(self.inverse)),
            7 => wrap_butterfly(Butterfly7::new(self.inverse)),
            8 => wrap_butterfly(Butterfly8::new(self.inverse)),
            11 => wrap_butterfly(Butterfly11::new(self.inverse)),
            13 => wrap_butterfly(Butterfly13::new(self.inverse)),
            16 => wrap_butterfly(Butterfly16::new(self.inverse)),
            17 => wrap_butterfly(Butterfly17::new(self.inverse)),
            19 => wrap_butterfly(Butterfly19::new(self.inverse)),
            23 => wrap_butterfly(Butterfly23::new(self.inverse)),
            29 => wrap_butterfly(Butterfly29::new(self.inverse)),
            31 => wrap_butterfly(Butterfly31::new(self.inverse)),
            32 => wrap_butterfly(Butterfly32::new(self.inverse)),
            _ => None,
        }
    }

    fn plan_prime(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        let inner_fft_len_rader = len - 1;
        let raders_factors = PrimeFactors::compute(inner_fft_len_rader);
        // If any of the prime factors is too large, Rader's gets slow and Bluestein's is the better choice
        if raders_factors
            .get_other_factors()
            .iter()
            .any(|val| val.value > MAX_RADER_PRIME_FACTOR)
        {
            let inner_fft_len_pow2 = (2 * len - 1).checked_next_power_of_two().unwrap();
            // for long ffts a mixed radix inner fft is faster than a longer radix4
            let min_inner_len = 2 * len - 1;
            let mixed_radix_len = 3 * inner_fft_len_pow2 / 4;
            let inner_fft =
                if mixed_radix_len >= min_inner_len && len >= MIN_BLUESTEIN_MIXED_RADIX_LEN {
                    let mixed_radix_factors = PrimeFactors::compute(mixed_radix_len);
                    self.plan_fft_with_factors(mixed_radix_len, mixed_radix_factors)
                } else {
                    Arc::new(Radix4::new(inner_fft_len_pow2, self.inverse))
                };
            Arc::new(BluesteinsAlgorithm::new(len, inner_fft)) as Arc<dyn Fft<T>>
        } else {
            let inner_fft = self.plan_fft_with_factors(inner_fft_len_rader, raders_factors);
            Arc::new(RadersAlgorithm::new(inner_fft)) as Arc<dyn Fft<T>>
        }
    }
}


#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_plan_trivial() {
        // Length 0 and 1 should use DFT
        let mut planner = FFTplanner::<f64>::new(false);
        for len in 0..2 {
            let plan = planner.make_recipe_for_len(len);
            assert_eq!(plan, Recipe::DFT(len));
        }
    }

    #[test]
    fn test_plan_mediumpoweroftwo() {
        // Powers of 2 between 64 and 32768 should use Radix4
        let mut planner = FFTplanner::<f64>::new(false);
        for pow in 6..16 {
            let len = 1 << pow;
            let plan = planner.make_recipe_for_len(len);
            assert_eq!(plan, Recipe::Radix4(len));
        }
    }

    #[test]
    fn test_plan_largepoweroftwo() {
        // Powers of 2 from 65536 and up should use MixedRadix
        let mut planner = FFTplanner::<f64>::new(false);
        for pow in 17..32 {
            let len = 1 << pow;
            let plan = planner.make_recipe_for_len(len);
            assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
        }
    }

    #[test]
    fn test_plan_butterflies() {
        // Check that all butterflies are used
        let mut planner = FFTplanner::<f64>::new(false);
        for len in [2,3,4,5,6,7,8,16,32].iter() {
            let plan = planner.make_recipe_for_len(*len);
            assert_eq!(plan, Recipe::Butterfly(*len));
        }
    }

    #[test]
    fn test_plan_mixedradix() {
        // Products of several different primes should become MixedRadix
        let mut planner = FFTplanner::<f64>::new(false);
        for pow2 in 1..3 {
            for pow3 in 1..3 {
                for pow5 in 1..3 {
                    for pow7 in 1..3 {
                        let len = 2usize.pow(pow2) * 3usize.pow(pow3) * 5usize.pow(pow5) * 7usize.pow(pow7);
                        let plan = planner.make_recipe_for_len(len);
                        assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
                    }
                }
            }
        }
    }

    fn is_mixedradix(plan: &Recipe) -> bool {
        match plan {
            &Recipe::MixedRadix{..} => true,
            _ => false,
        }
    }

    #[test]
    fn test_plan_mixedradixbutterfly() {
        // Products of two existing butterfly lengths that have a common divisor >1, and isn't a power of 2 should be MixedRadixDoubleButterfly
        let mut planner = FFTplanner::<f64>::new(false);
        for len in [4*6, 3*6, 3*3].iter() {
            let plan = planner.make_recipe_for_len(*len);
            assert!(is_mixedradixbutterfly(&plan), "Expected MixedRadixDoubleButterfly, got {:?}", plan);
        }
    }

    fn is_mixedradixbutterfly(plan: &Recipe) -> bool {
        match plan {
            &Recipe::MixedRadixDoubleButterfly{..} => true,
            _ => false,
        }
    }

    #[test]
    fn test_plan_goodthomasbutterfly() {
        let mut planner = FFTplanner::<f64>::new(false);
        for len in [3*4, 3*5, 3*7, 5*7].iter() {
            let plan = planner.make_recipe_for_len(*len);
            assert!(is_goodthomasbutterfly(&plan), "Expected GoodThomasDoubleButterfly, got {:?}", plan);
        }
    }

    fn is_goodthomasbutterfly(plan: &Recipe) -> bool {
        match plan {
            &Recipe::GoodThomasDoubleButterfly{..} => true,
            _ => false,
        }
    }

    #[test]
    fn test_plan_bluestein() {
        let primes: [usize; 6] = [89, 179, 359, 719, 1439, 2879];

        let mut planner = FFTplanner::<f64>::new(false);
        for len in primes.iter() {
            let plan = planner.make_recipe_for_len(*len);
            assert!(is_bluesteins(&plan), "Expected Bluesteins, got {:?}", plan);
        }
    }

    fn is_bluesteins(plan: &Recipe) -> bool {
        match plan {
            &Recipe::Bluestein{..} => true,
            _ => false,
        }
    }
}