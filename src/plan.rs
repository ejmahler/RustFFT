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

#[derive(Debug, std::cmp::PartialEq, Clone)]
#[allow(dead_code)]
pub enum Recipe {
    DFT(usize),
    MixedRadix {
        left_fft: Box<Recipe>,
        right_fft: Box<Recipe>,
    },
    GoodThomasAlgorithm {
        left_fft: Box<Recipe>,
        right_fft: Box<Recipe>,
    },
    MixedRadixSmall {
        left_fft: Box<Recipe>,
        right_fft: Box<Recipe>,
    },
    GoodThomasAlgorithmSmall {
        left_fft: Box<Recipe>,
        right_fft: Box<Recipe>,
    },
    RadersAlgorithm {
        inner_fft: Box<Recipe>,
    },
    BluesteinsAlgorithm {
        len: usize,
        inner_fft: Box<Recipe>,
    },
    Radix4(usize),
    Butterfly2,
    Butterfly3,
    Butterfly4,
    Butterfly5,
    Butterfly6,
    Butterfly7,
    Butterfly8,
    Butterfly11,
    Butterfly13,
    Butterfly16,
    Butterfly17,
    Butterfly19,
    Butterfly23,
    Butterfly29,
    Butterfly31,
    Butterfly32,
}

/// The Scalar FFT planner creates new FFT algorithm instances using non-SIMD algorithms.
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
    recipe_cache: HashMap<usize, Recipe>,
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
            recipe_cache: HashMap::new(),
        }
    }

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        if let Some(instance) = self.algorithm_cache.get(&len) {
            Arc::clone(instance)
        } else {
            let recipe = self.design_fft_for_len(len);
            let fft = self.build_fft(recipe);
            self.algorithm_cache.insert(len, Arc::clone(&fft));
            fft
        }
    }

    // Make a recipe for a length
    fn design_fft_for_len(&mut self, len: usize) -> Recipe {
        if len < 2 {
            Recipe::DFT(len)
        } else {
            let factors = PrimeFactors::compute(len);
            self.design_fft_with_factors(len, factors)
        }
    }

    // Create the fft from a recipe
    fn build_fft(&mut self, plan: Recipe) -> Arc<dyn Fft<T>> {
        match plan {
            Recipe::DFT(len) => Arc::new(DFT::new(len, self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Radix4(len) => Arc::new(Radix4::new(len, self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly2 => Arc::new(Butterfly2::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly3 => Arc::new(Butterfly3::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly4 => Arc::new(Butterfly4::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly5 => Arc::new(Butterfly5::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly6 => Arc::new(Butterfly6::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly7 => Arc::new(Butterfly7::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly8 => Arc::new(Butterfly8::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly11 => Arc::new(Butterfly11::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly13 => Arc::new(Butterfly13::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly16 => Arc::new(Butterfly16::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly17 => Arc::new(Butterfly17::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly19 => Arc::new(Butterfly19::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly23 => Arc::new(Butterfly23::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly29 => Arc::new(Butterfly29::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly31 => Arc::new(Butterfly31::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly32 => Arc::new(Butterfly32::new(self.inverse)) as Arc<dyn Fft<T>>,
            Recipe::MixedRadix {
                left_fft,
                right_fft,
            } => {
                let left_fft = self.build_fft(*left_fft);
                let right_fft = self.build_fft(*right_fft);
                Arc::new(MixedRadix::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::GoodThomasAlgorithm {
                left_fft,
                right_fft,
            } => {
                let left_fft = self.build_fft(*left_fft);
                let right_fft = self.build_fft(*right_fft);
                Arc::new(GoodThomasAlgorithm::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::MixedRadixSmall {
                left_fft,
                right_fft,
            } => {
                let left_fft = self.build_fft(*left_fft);
                let right_fft = self.build_fft(*right_fft);
                Arc::new(MixedRadixSmall::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::GoodThomasAlgorithmSmall {
                left_fft,
                right_fft,
            } => {
                let left_fft = self.build_fft(*left_fft);
                let right_fft = self.build_fft(*right_fft);
                Arc::new(GoodThomasAlgorithmSmall::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::RadersAlgorithm { inner_fft } => {
                let inner_fft = self.build_fft(*inner_fft);
                Arc::new(RadersAlgorithm::new(inner_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::BluesteinsAlgorithm { len, inner_fft } => {
                let inner_fft = self.build_fft(*inner_fft);
                Arc::new(BluesteinsAlgorithm::new(len, inner_fft)) as Arc<dyn Fft<T>>
            }
        }
    }

    fn design_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Recipe {
        if let Some(recipe) = self.recipe_cache.get(&len) {
            recipe.clone()
        } else {
            let recipe = self.design_new_fft_with_factors(len, factors);
            self.recipe_cache.insert(len, recipe.clone());
            recipe
        }
        
    }

    fn design_new_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Recipe {
        if let Some(fft_instance) = self.design_butterfly_algorithm(len) {
            fft_instance
        } else if factors.is_prime() {
            self.design_prime(len)
        } else if len.trailing_zeros() <= MAX_RADIX4_BITS && len.trailing_zeros() >= MIN_RADIX4_BITS
        {
            if len.is_power_of_two() {
                //Arc::new(Radix4::new(len, self.inverse))
                Recipe::Radix4(len)
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
                self.design_mixed_radix(power_of_two, non_power_of_two)
            }
        } else {
            let (left_factors, right_factors) = factors.partition_factors();
            self.design_mixed_radix(left_factors, right_factors)
        }
    }

    fn design_mixed_radix(
        &mut self,
        left_factors: PrimeFactors,
        right_factors: PrimeFactors,
    ) -> Recipe {
        let left_len = left_factors.get_product();
        let right_len = right_factors.get_product();

        //neither size is a butterfly, so go with the normal algorithm
        let left_fft = Box::new(self.design_fft_with_factors(left_len, left_factors));
        let right_fft = Box::new(self.design_fft_with_factors(right_len, right_factors));

        //if both left_len and right_len are small, use algorithms optimized for small FFTs
        if left_len < 31 && right_len < 31 {
            // for small FFTs, if gcd is 1, good-thomas is faster
            if gcd(left_len, right_len) == 1 {
                //Arc::new(GoodThomasAlgorithmSmall::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
                Recipe::GoodThomasAlgorithmSmall {
                    left_fft,
                    right_fft,
                }
            } else {
                //Arc::new(MixedRadixSmall::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
                Recipe::MixedRadixSmall {
                    left_fft,
                    right_fft,
                }
            }
        } else {
            //Arc::new(MixedRadix::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            Recipe::MixedRadix {
                left_fft,
                right_fft,
            }
        }
    }

    // Returns Some(instance) if we have a butterfly available for this size. Returns None if there is no butterfly available for this size
    fn design_butterfly_algorithm(&mut self, len: usize) -> Option<Recipe> {
        match len {
            2 => Some(Recipe::Butterfly2),
            3 => Some(Recipe::Butterfly3),
            4 => Some(Recipe::Butterfly4),
            5 => Some(Recipe::Butterfly5),
            6 => Some(Recipe::Butterfly6),
            7 => Some(Recipe::Butterfly7),
            8 => Some(Recipe::Butterfly8),
            11 => Some(Recipe::Butterfly11),
            13 => Some(Recipe::Butterfly13),
            16 => Some(Recipe::Butterfly16),
            17 => Some(Recipe::Butterfly17),
            19 => Some(Recipe::Butterfly19),
            23 => Some(Recipe::Butterfly23),
            29 => Some(Recipe::Butterfly29),
            31 => Some(Recipe::Butterfly31),
            32 => Some(Recipe::Butterfly32),
            _ => None,
        }
    }

    fn design_prime(&mut self, len: usize) -> Recipe {
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
                    Box::new(self.design_fft_with_factors(mixed_radix_len, mixed_radix_factors))
                } else {
                    Box::new(Recipe::Radix4(inner_fft_len_pow2))
                    //Arc::new(Radix4::new(inner_fft_len_pow2, self.inverse))
                };
            //Arc::new(BluesteinsAlgorithm::new(len, inner_fft)) as Arc<dyn Fft<T>>
            Recipe::BluesteinsAlgorithm { len, inner_fft }
        } else {
            let inner_fft =
                Box::new(self.design_fft_with_factors(inner_fft_len_rader, raders_factors));
            //Arc::new(RadersAlgorithm::new(inner_fft)) as Arc<dyn Fft<T>>
            Recipe::RadersAlgorithm { inner_fft }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn is_mixedradix(plan: &Recipe) -> bool {
        match plan {
            &Recipe::MixedRadix { .. } => true,
            _ => false,
        }
    }

    fn is_mixedradixsmall(plan: &Recipe) -> bool {
        match plan {
            &Recipe::MixedRadixSmall { .. } => true,
            _ => false,
        }
    }

    fn is_goodthomassmall(plan: &Recipe) -> bool {
        match plan {
            &Recipe::GoodThomasAlgorithmSmall { .. } => true,
            _ => false,
        }
    }

    fn is_raders(plan: &Recipe) -> bool {
        match plan {
            &Recipe::RadersAlgorithm { .. } => true,
            _ => false,
        }
    }

    fn is_bluesteins(plan: &Recipe) -> bool {
        match plan {
            &Recipe::BluesteinsAlgorithm { .. } => true,
            _ => false,
        }
    }

    #[test]
    fn test_plan_scalar_trivial() {
        // Length 0 and 1 should use DFT
        let mut planner = FftPlannerScalar::<f64>::new(false);
        for len in 0..2 {
            let plan = planner.design_fft_for_len(len);
            assert_eq!(plan, Recipe::DFT(len));
        }
    }

    #[test]
    fn test_plan_scalar_mediumpoweroftwo() {
        // Powers of 2 between 64 and 32768 should use Radix4
        let mut planner = FftPlannerScalar::<f64>::new(false);
        for pow in 6..16 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert_eq!(plan, Recipe::Radix4(len));
        }
    }

    #[test]
    fn test_plan_scalar_largepoweroftwo() {
        // Powers of 2 from 65536 and up should use MixedRadix
        let mut planner = FftPlannerScalar::<f64>::new(false);
        for pow in 17..32 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
        }
    }

    #[test]
    fn test_plan_scalar_butterflies() {
        // Check that all butterflies are used
        let mut planner = FftPlannerScalar::<f64>::new(false);
        assert_eq!(planner.design_fft_for_len(2), Recipe::Butterfly2);
        assert_eq!(planner.design_fft_for_len(3), Recipe::Butterfly3);
        assert_eq!(planner.design_fft_for_len(4), Recipe::Butterfly4);
        assert_eq!(planner.design_fft_for_len(5), Recipe::Butterfly5);
        assert_eq!(planner.design_fft_for_len(6), Recipe::Butterfly6);
        assert_eq!(planner.design_fft_for_len(7), Recipe::Butterfly7);
        assert_eq!(planner.design_fft_for_len(8), Recipe::Butterfly8);
        assert_eq!(planner.design_fft_for_len(11), Recipe::Butterfly11);
        assert_eq!(planner.design_fft_for_len(13), Recipe::Butterfly13);
        assert_eq!(planner.design_fft_for_len(16), Recipe::Butterfly16);
        assert_eq!(planner.design_fft_for_len(17), Recipe::Butterfly17);
        assert_eq!(planner.design_fft_for_len(19), Recipe::Butterfly19);
        assert_eq!(planner.design_fft_for_len(23), Recipe::Butterfly23);
        assert_eq!(planner.design_fft_for_len(29), Recipe::Butterfly29);
        assert_eq!(planner.design_fft_for_len(31), Recipe::Butterfly31);
        assert_eq!(planner.design_fft_for_len(32), Recipe::Butterfly32);
    }

    #[test]
    fn test_plan_scalar_mixedradix() {
        // Products of several different primes should become MixedRadix
        let mut planner = FftPlannerScalar::<f64>::new(false);
        for pow2 in 2..6 {
            for pow3 in 2..6 {
                for pow5 in 2..6 {
                    for pow7 in 2..6 {
                        let len = 2usize.pow(pow2)
                            * 3usize.pow(pow3)
                            * 5usize.pow(pow5)
                            * 7usize.pow(pow7);
                        let plan = planner.design_fft_for_len(len);
                        assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
                    }
                }
            }
        }
    }

    #[test]
    fn test_plan_scalar_mixedradixsmall() {
        // Products of two "small" lengths < 31 that have a common divisor >1, and isn't a power of 2 should be MixedRadixSmall
        let mut planner = FftPlannerScalar::<f64>::new(false);
        for len in [5 * 20, 5 * 25].iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(
                is_mixedradixsmall(&plan),
                "Expected MixedRadixSmall, got {:?}",
                plan
            );
        }
    }

    #[test]
    fn test_plan_scalar_goodthomasbutterfly() {
        let mut planner = FftPlannerScalar::<f64>::new(false);
        for len in [3 * 4, 3 * 5, 3 * 7, 5 * 7, 11 * 13].iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(
                is_goodthomassmall(&plan),
                "Expected GoodThomasAlgorithmSmall, got {:?}",
                plan
            );
        }
    }

    #[test]
    fn test_plan_scalar_bluestein_vs_rader() {
        let difficultprimes: [usize; 11] = [59, 83, 107, 149, 167, 173, 179, 359, 719, 1439, 2879];
        let easyprimes: [usize; 24] = [
            53, 61, 67, 71, 73, 79, 89, 97, 101, 103, 109, 113, 127, 131, 137, 139, 151, 157, 163,
            181, 191, 193, 197, 199,
        ];

        let mut planner = FftPlannerScalar::<f64>::new(false);
        for len in difficultprimes.iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(
                is_bluesteins(&plan),
                "Expected BluesteinsAlgorithm, got {:?}",
                plan
            );
        }
        for len in easyprimes.iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(is_raders(&plan), "Expected RadersAlgorithm, got {:?}", plan);
        }
    }
}


#[cfg(test)]
mod unit_tests {
    use super::*;

    fn is_mixedradix(plan: &Recipe) -> bool {
        match plan {
            &Recipe::MixedRadix{..} => true,
            _ => false,
        }
    }

    fn is_mixedradixsmall(plan: &Recipe) -> bool {
        match plan {
            &Recipe::MixedRadixSmall{..} => true,
            _ => false,
        }
    }

    fn is_goodthomassmall(plan: &Recipe) -> bool {
        match plan {
            &Recipe::GoodThomasAlgorithmSmall{..} => true,
            _ => false,
        }
    }

    fn is_raders(plan: &Recipe) -> bool {
        match plan {
            &Recipe::RadersAlgorithm{..} => true,
            _ => false,
        }
    }

    fn is_bluesteins(plan: &Recipe) -> bool {
        match plan {
            &Recipe::BluesteinsAlgorithm{..} => true,
            _ => false,
        }
    }

    #[test]
    fn test_plan_trivial() {
        // Length 0 and 1 should use DFT
        let mut planner = FftPlanner::<f64>::new(false);
        for len in 0..2 {
            let plan = planner.design_fft_for_len(len);
            assert_eq!(plan, Recipe::DFT(len));
        }
    }

    #[test]
    fn test_plan_mediumpoweroftwo() {
        // Powers of 2 between 64 and 32768 should use Radix4
        let mut planner = FftPlanner::<f64>::new(false);
        for pow in 6..16 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert_eq!(plan, Recipe::Radix4(len));
        }
    }

    #[test]
    fn test_plan_largepoweroftwo() {
        // Powers of 2 from 65536 and up should use MixedRadix
        let mut planner = FftPlanner::<f64>::new(false);
        for pow in 17..32 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
        }
    }

    #[test]
    fn test_plan_butterflies() {
        // Check that all butterflies are used
        let mut planner = FftPlanner::<f64>::new(false);
        assert_eq!(planner.design_fft_for_len(2), Recipe::Butterfly2);
        assert_eq!(planner.design_fft_for_len(3), Recipe::Butterfly3);
        assert_eq!(planner.design_fft_for_len(4), Recipe::Butterfly4);
        assert_eq!(planner.design_fft_for_len(5), Recipe::Butterfly5);
        assert_eq!(planner.design_fft_for_len(6), Recipe::Butterfly6);
        assert_eq!(planner.design_fft_for_len(7), Recipe::Butterfly7);
        assert_eq!(planner.design_fft_for_len(8), Recipe::Butterfly8);
        assert_eq!(planner.design_fft_for_len(11), Recipe::Butterfly11);
        assert_eq!(planner.design_fft_for_len(13), Recipe::Butterfly13);
        assert_eq!(planner.design_fft_for_len(16), Recipe::Butterfly16);
        assert_eq!(planner.design_fft_for_len(17), Recipe::Butterfly17);
        assert_eq!(planner.design_fft_for_len(19), Recipe::Butterfly19);
        assert_eq!(planner.design_fft_for_len(23), Recipe::Butterfly23);
        assert_eq!(planner.design_fft_for_len(29), Recipe::Butterfly29);
        assert_eq!(planner.design_fft_for_len(31), Recipe::Butterfly31);
        assert_eq!(planner.design_fft_for_len(32), Recipe::Butterfly32);
    }

    #[test]
    fn test_plan_mixedradix() {
        // Products of several different primes should become MixedRadix
        let mut planner = FftPlanner::<f64>::new(false);
        for pow2 in 2..6 {
            for pow3 in 2..6 {
                for pow5 in 2..6 {
                    for pow7 in 2..6 {
                        let len = 2usize.pow(pow2) * 3usize.pow(pow3) * 5usize.pow(pow5) * 7usize.pow(pow7);
                        let plan = planner.design_fft_for_len(len);
                        assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
                    }
                }
            }
        }
    }



    #[test]
    fn test_plan_mixedradixsmall() {
        // Products of two "small" lengths < 31 that have a common divisor >1, and isn't a power of 2 should be MixedRadixSmall
        let mut planner = FftPlanner::<f64>::new(false);
        for len in [5*20, 5*25].iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(is_mixedradixsmall(&plan), "Expected MixedRadixSmall, got {:?}", plan);
        }
    }



    #[test]
    fn test_plan_goodthomasbutterfly() {
        let mut planner = FftPlanner::<f64>::new(false);
        for len in [3*4, 3*5, 3*7, 5*7, 11*13].iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(is_goodthomassmall(&plan), "Expected GoodThomasAlgorithmSmall, got {:?}", plan);
        }
    }



    #[test]
    fn test_plan_bluestein_vs_rader() {
        let difficultprimes: [usize; 11] = [59, 83, 107, 149, 167, 173, 179, 359, 719, 1439, 2879];
        let easyprimes: [usize; 24] = [53, 61, 67, 71, 73, 79, 89, 97, 101, 103, 109, 113, 127, 131, 137, 139, 151, 157, 163, 181, 191, 193, 197, 199];

        let mut planner = FftPlanner::<f64>::new(false);
        for len in difficultprimes.iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(is_bluesteins(&plan), "Expected BluesteinsAlgorithm, got {:?}", plan);
        }
        for len in easyprimes.iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(is_raders(&plan), "Expected RadersAlgorithm, got {:?}", plan);
        }
    }

}