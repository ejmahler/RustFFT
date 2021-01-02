use num_integer::gcd;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::{common::FftNum, fft_cache::FftCache, FftDirection};

use crate::algorithm::butterflies::*;
use crate::algorithm::*;
use crate::Fft;

use crate::FftPlannerAvx;

use crate::math_utils::{PrimeFactor, PrimeFactors};

enum ChosenFftPlanner<T: FftNum> {
    Scalar(FftPlannerScalar<T>),
    Avx(FftPlannerAvx<T>),
    // todo: If we add NEON, SSE, avx-512 etc support, add more enum variants for them here
}

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
/// let mut planner = FftPlanner::new();
/// let fft = planner.plan_fft_forward(1234);
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
/// fft.process_inplace(&mut buffer);
///
/// // The FFT instance returned by the planner has the type `Arc<dyn Fft<T>>`,
/// // where T is the numeric type, ie f32 or f64, so it's cheap to clone
/// let fft_clone = Arc::clone(&fft);
/// ~~~
///
/// If you plan on creating multiple FFT instances, it is recommended to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
/// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
/// by a different planner)
///
/// Each FFT instance owns [`Arc`s](std::sync::Arc) to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating Fft instances.
pub struct FftPlanner<T: FftNum> {
    chosen_planner: ChosenFftPlanner<T>,
}
impl<T: FftNum> FftPlanner<T> {
    /// Creates a new `FftPlanner` instance. It detects if AVX is supported on the current machine. If it is, it will plan AVX-accelerated FFTs.
    /// If AVX isn't supported, it will seamlessly fall back to planning non-SIMD FFTs.
    pub fn new() -> Self {
        if let Ok(avx_planner) = FftPlannerAvx::new() {
            Self {
                chosen_planner: ChosenFftPlanner::Avx(avx_planner),
            }
        } else {
            Self {
                chosen_planner: ChosenFftPlanner::Scalar(FftPlannerScalar::new()),
            }
        }
    }

    /// Returns a `Fft` instance which computes FFTs of size `len`.
    ///
    /// If the provided `direction` is `FftDirection::Forward`, the returned instance will compute forward FFTs. If it's `FftDirection::Inverse`, it will compute inverse FFTs.
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        match &mut self.chosen_planner {
            ChosenFftPlanner::Scalar(scalar_planner) => scalar_planner.plan_fft(len, direction),
            ChosenFftPlanner::Avx(avx_planner) => avx_planner.plan_fft(len, direction),
        }
    }

    /// Returns a `Fft` instance which computes forward FFTs of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    /// Returns a `Fft` instance which computes inverse FFTs of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }
}

const MIN_RADIX4_BITS: u32 = 5; // smallest size to consider radix 4 an option is 2^5 = 32
const MAX_RADIX4_BITS: u32 = 16; // largest size to consider radix 4 an option is 2^16 = 65536
const MAX_RADER_PRIME_FACTOR: usize = 23; // don't use Raders if the inner fft length has prime factor larger than this
const MIN_BLUESTEIN_MIXED_RADIX_LEN: usize = 90; // only use mixed radix for the inner fft of Bluestein if length is larger than this

/// A Recipe is a structure that describes the design of a FFT, without actually creating it.
/// It is used as a middle step in the planning process.
#[derive(Debug, std::cmp::PartialEq, Clone)]
pub enum Recipe {
    DFT(usize),
    MixedRadix {
        left_fft: Rc<Recipe>,
        right_fft: Rc<Recipe>,
    },
    #[allow(dead_code)]
    GoodThomasAlgorithm {
        left_fft: Rc<Recipe>,
        right_fft: Rc<Recipe>,
    },
    MixedRadixSmall {
        left_fft: Rc<Recipe>,
        right_fft: Rc<Recipe>,
    },
    GoodThomasAlgorithmSmall {
        left_fft: Rc<Recipe>,
        right_fft: Rc<Recipe>,
    },
    RadersAlgorithm {
        inner_fft: Rc<Recipe>,
    },
    BluesteinsAlgorithm {
        len: usize,
        inner_fft: Rc<Recipe>,
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

impl Recipe {
    pub fn len(&self) -> usize {
        match self {
            Recipe::DFT(length) => *length,
            Recipe::Radix4(length) => *length,
            Recipe::Butterfly2 => 2,
            Recipe::Butterfly3 => 3,
            Recipe::Butterfly4 => 4,
            Recipe::Butterfly5 => 5,
            Recipe::Butterfly6 => 6,
            Recipe::Butterfly7 => 7,
            Recipe::Butterfly8 => 8,
            Recipe::Butterfly11 => 11,
            Recipe::Butterfly13 => 13,
            Recipe::Butterfly16 => 16,
            Recipe::Butterfly17 => 17,
            Recipe::Butterfly19 => 19,
            Recipe::Butterfly23 => 23,
            Recipe::Butterfly29 => 29,
            Recipe::Butterfly31 => 31,
            Recipe::Butterfly32 => 32,
            Recipe::MixedRadix {
                left_fft,
                right_fft,
            } => left_fft.len() * right_fft.len(),
            Recipe::GoodThomasAlgorithm {
                left_fft,
                right_fft,
            } => left_fft.len() * right_fft.len(),
            Recipe::MixedRadixSmall {
                left_fft,
                right_fft,
            } => left_fft.len() * right_fft.len(),
            Recipe::GoodThomasAlgorithmSmall {
                left_fft,
                right_fft,
            } => left_fft.len() * right_fft.len(),
            Recipe::RadersAlgorithm { inner_fft } => inner_fft.len() + 1,
            Recipe::BluesteinsAlgorithm { len, .. } => *len,
        }
    }
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
/// let mut planner = FftPlannerScalar::new();
/// let fft = planner.plan_fft_forward(1234);
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
/// fft.process_inplace(&mut buffer);
///
/// // The FFT instance returned by the planner has the type `Arc<dyn Fft<T>>`,
/// // where T is the numeric type, ie f32 or f64, so it's cheap to clone
/// let fft_clone = Arc::clone(&fft);
/// ~~~
///
/// If you plan on creating multiple FFT instances, it is recommended to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
/// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
/// by a different planner)
///
/// Each FFT instance owns [`Arc`s](std::sync::Arc) to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating Fft instances.
pub struct FftPlannerScalar<T: FftNum> {
    algorithm_cache: FftCache<T>,
    recipe_cache: HashMap<usize, Rc<Recipe>>,
}

impl<T: FftNum> FftPlannerScalar<T> {
    /// Creates a new `FftPlannerScalar` instance.
    pub fn new() -> Self {
        Self {
            algorithm_cache: FftCache::new(),
            recipe_cache: HashMap::new(),
        }
    }

    /// Returns a `Fft` instance which computes FFTs of size `len`.
    ///
    /// If the provided `direction` is `FftDirection::Forward`, the returned instance will compute forward FFTs. If it's `FftDirection::Inverse`, it will compute inverse FFTs.
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        // Step 1: Create a "recipe" for this FFT, which will tell us exactly which combination of algorithms to use
        let recipe = self.design_fft_for_len(len);

        // Step 2: Use our recipe to construct a Fft trait object
        self.build_fft(&recipe, direction)
    }

    /// Returns a `Fft` instance which computes forward FFTs of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    /// Returns a `Fft` instance which computes inverse FFTs of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }

    // Make a recipe for a length
    fn design_fft_for_len(&mut self, len: usize) -> Rc<Recipe> {
        if len < 2 {
            Rc::new(Recipe::DFT(len))
        } else if let Some(recipe) = self.recipe_cache.get(&len) {
            Rc::clone(&recipe)
        } else {
            let factors = PrimeFactors::compute(len);
            let recipe = self.design_fft_with_factors(len, factors);
            self.recipe_cache.insert(len, Rc::clone(&recipe));
            recipe
        }
    }

    // Create the fft from a recipe, take from cache if possible
    fn build_fft(&mut self, recipe: &Recipe, direction: FftDirection) -> Arc<dyn Fft<T>> {
        let len = recipe.len();
        if let Some(instance) = self.algorithm_cache.get(len, direction) {
            instance
        } else {
            let fft = self.build_new_fft(recipe, direction);
            self.algorithm_cache.insert(&fft);
            fft
        }
    }

    // Create a new fft from a recipe
    fn build_new_fft(&mut self, recipe: &Recipe, direction: FftDirection) -> Arc<dyn Fft<T>> {
        match recipe {
            Recipe::DFT(len) => Arc::new(DFT::new(*len, direction)) as Arc<dyn Fft<T>>,
            Recipe::Radix4(len) => Arc::new(Radix4::new(*len, direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly2 => Arc::new(Butterfly2::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly3 => Arc::new(Butterfly3::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly4 => Arc::new(Butterfly4::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly5 => Arc::new(Butterfly5::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly6 => Arc::new(Butterfly6::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly7 => Arc::new(Butterfly7::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly8 => Arc::new(Butterfly8::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly11 => Arc::new(Butterfly11::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly13 => Arc::new(Butterfly13::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly16 => Arc::new(Butterfly16::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly17 => Arc::new(Butterfly17::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly19 => Arc::new(Butterfly19::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly23 => Arc::new(Butterfly23::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly29 => Arc::new(Butterfly29::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly31 => Arc::new(Butterfly31::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly32 => Arc::new(Butterfly32::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::MixedRadix {
                left_fft,
                right_fft,
            } => {
                let left_fft = self.build_fft(&left_fft, direction);
                let right_fft = self.build_fft(&right_fft, direction);
                Arc::new(MixedRadix::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::GoodThomasAlgorithm {
                left_fft,
                right_fft,
            } => {
                let left_fft = self.build_fft(&left_fft, direction);
                let right_fft = self.build_fft(&right_fft, direction);
                Arc::new(GoodThomasAlgorithm::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::MixedRadixSmall {
                left_fft,
                right_fft,
            } => {
                let left_fft = self.build_fft(&left_fft, direction);
                let right_fft = self.build_fft(&right_fft, direction);
                Arc::new(MixedRadixSmall::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::GoodThomasAlgorithmSmall {
                left_fft,
                right_fft,
            } => {
                let left_fft = self.build_fft(&left_fft, direction);
                let right_fft = self.build_fft(&right_fft, direction);
                Arc::new(GoodThomasAlgorithmSmall::new(left_fft, right_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::RadersAlgorithm { inner_fft } => {
                let inner_fft = self.build_fft(&inner_fft, direction);
                Arc::new(RadersAlgorithm::new(inner_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::BluesteinsAlgorithm { len, inner_fft } => {
                let inner_fft = self.build_fft(&inner_fft, direction);
                Arc::new(BluesteinsAlgorithm::new(*len, inner_fft)) as Arc<dyn Fft<T>>
            }
        }
    }

    fn design_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Rc<Recipe> {
        if let Some(fft_instance) = self.design_butterfly_algorithm(len) {
            fft_instance
        } else if factors.is_prime() {
            self.design_prime(len)
        } else if len.trailing_zeros() <= MAX_RADIX4_BITS && len.trailing_zeros() >= MIN_RADIX4_BITS
        {
            if len.is_power_of_two() {
                Rc::new(Recipe::Radix4(len))
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
    ) -> Rc<Recipe> {
        let left_len = left_factors.get_product();
        let right_len = right_factors.get_product();

        //neither size is a butterfly, so go with the normal algorithm
        let left_fft = self.design_fft_with_factors(left_len, left_factors);
        let right_fft = self.design_fft_with_factors(right_len, right_factors);

        //if both left_len and right_len are small, use algorithms optimized for small FFTs
        if left_len < 31 && right_len < 31 {
            // for small FFTs, if gcd is 1, good-thomas is faster
            if gcd(left_len, right_len) == 1 {
                Rc::new(Recipe::GoodThomasAlgorithmSmall {
                    left_fft,
                    right_fft,
                })
            } else {
                Rc::new(Recipe::MixedRadixSmall {
                    left_fft,
                    right_fft,
                })
            }
        } else {
            Rc::new(Recipe::MixedRadix {
                left_fft,
                right_fft,
            })
        }
    }

    // Returns Some(instance) if we have a butterfly available for this size. Returns None if there is no butterfly available for this size
    fn design_butterfly_algorithm(&mut self, len: usize) -> Option<Rc<Recipe>> {
        match len {
            2 => Some(Rc::new(Recipe::Butterfly2)),
            3 => Some(Rc::new(Recipe::Butterfly3)),
            4 => Some(Rc::new(Recipe::Butterfly4)),
            5 => Some(Rc::new(Recipe::Butterfly5)),
            6 => Some(Rc::new(Recipe::Butterfly6)),
            7 => Some(Rc::new(Recipe::Butterfly7)),
            8 => Some(Rc::new(Recipe::Butterfly8)),
            11 => Some(Rc::new(Recipe::Butterfly11)),
            13 => Some(Rc::new(Recipe::Butterfly13)),
            16 => Some(Rc::new(Recipe::Butterfly16)),
            17 => Some(Rc::new(Recipe::Butterfly17)),
            19 => Some(Rc::new(Recipe::Butterfly19)),
            23 => Some(Rc::new(Recipe::Butterfly23)),
            29 => Some(Rc::new(Recipe::Butterfly29)),
            31 => Some(Rc::new(Recipe::Butterfly31)),
            32 => Some(Rc::new(Recipe::Butterfly32)),
            _ => None,
        }
    }

    fn design_prime(&mut self, len: usize) -> Rc<Recipe> {
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
                    self.design_fft_with_factors(mixed_radix_len, mixed_radix_factors)
                } else {
                    Rc::new(Recipe::Radix4(inner_fft_len_pow2))
                };
            Rc::new(Recipe::BluesteinsAlgorithm { len, inner_fft })
        } else {
            let inner_fft = self.design_fft_with_factors(inner_fft_len_rader, raders_factors);
            Rc::new(Recipe::RadersAlgorithm { inner_fft })
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
        let mut planner = FftPlannerScalar::<f64>::new();
        for len in 0..2 {
            let plan = planner.design_fft_for_len(len);
            assert_eq!(*plan, Recipe::DFT(len));
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_scalar_mediumpoweroftwo() {
        // Powers of 2 between 64 and 32768 should use Radix4
        let mut planner = FftPlannerScalar::<f64>::new();
        for pow in 6..16 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert_eq!(*plan, Recipe::Radix4(len));
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_scalar_largepoweroftwo() {
        // Powers of 2 from 65536 and up should use MixedRadix
        let mut planner = FftPlannerScalar::<f64>::new();
        for pow in 17..32 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_scalar_butterflies() {
        // Check that all butterflies are used
        let mut planner = FftPlannerScalar::<f64>::new();
        assert_eq!(*planner.design_fft_for_len(2), Recipe::Butterfly2);
        assert_eq!(*planner.design_fft_for_len(3), Recipe::Butterfly3);
        assert_eq!(*planner.design_fft_for_len(4), Recipe::Butterfly4);
        assert_eq!(*planner.design_fft_for_len(5), Recipe::Butterfly5);
        assert_eq!(*planner.design_fft_for_len(6), Recipe::Butterfly6);
        assert_eq!(*planner.design_fft_for_len(7), Recipe::Butterfly7);
        assert_eq!(*planner.design_fft_for_len(8), Recipe::Butterfly8);
        assert_eq!(*planner.design_fft_for_len(11), Recipe::Butterfly11);
        assert_eq!(*planner.design_fft_for_len(13), Recipe::Butterfly13);
        assert_eq!(*planner.design_fft_for_len(16), Recipe::Butterfly16);
        assert_eq!(*planner.design_fft_for_len(17), Recipe::Butterfly17);
        assert_eq!(*planner.design_fft_for_len(19), Recipe::Butterfly19);
        assert_eq!(*planner.design_fft_for_len(23), Recipe::Butterfly23);
        assert_eq!(*planner.design_fft_for_len(29), Recipe::Butterfly29);
        assert_eq!(*planner.design_fft_for_len(31), Recipe::Butterfly31);
        assert_eq!(*planner.design_fft_for_len(32), Recipe::Butterfly32);
    }

    #[test]
    fn test_plan_scalar_mixedradix() {
        // Products of several different primes should become MixedRadix
        let mut planner = FftPlannerScalar::<f64>::new();
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
                        assert_eq!(plan.len(), len, "Recipe reports wrong length");
                    }
                }
            }
        }
    }

    #[test]
    fn test_plan_scalar_mixedradixsmall() {
        // Products of two "small" lengths < 31 that have a common divisor >1, and isn't a power of 2 should be MixedRadixSmall
        let mut planner = FftPlannerScalar::<f64>::new();
        for len in [5 * 20, 5 * 25].iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(
                is_mixedradixsmall(&plan),
                "Expected MixedRadixSmall, got {:?}",
                plan
            );
            assert_eq!(plan.len(), *len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_scalar_goodthomasbutterfly() {
        let mut planner = FftPlannerScalar::<f64>::new();
        for len in [3 * 4, 3 * 5, 3 * 7, 5 * 7, 11 * 13].iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(
                is_goodthomassmall(&plan),
                "Expected GoodThomasAlgorithmSmall, got {:?}",
                plan
            );
            assert_eq!(plan.len(), *len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_scalar_bluestein_vs_rader() {
        let difficultprimes: [usize; 11] = [59, 83, 107, 149, 167, 173, 179, 359, 719, 1439, 2879];
        let easyprimes: [usize; 24] = [
            53, 61, 67, 71, 73, 79, 89, 97, 101, 103, 109, 113, 127, 131, 137, 139, 151, 157, 163,
            181, 191, 193, 197, 199,
        ];

        let mut planner = FftPlannerScalar::<f64>::new();
        for len in difficultprimes.iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(
                is_bluesteins(&plan),
                "Expected BluesteinsAlgorithm, got {:?}",
                plan
            );
            assert_eq!(plan.len(), *len, "Recipe reports wrong length");
        }
        for len in easyprimes.iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(is_raders(&plan), "Expected RadersAlgorithm, got {:?}", plan);
            assert_eq!(plan.len(), *len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_scalar_fft_cache() {
        {
            // Check that FFTs are reused if they're both forward
            let mut planner = FftPlannerScalar::<f64>::new();
            let fft_a = planner.plan_fft(1234, FftDirection::Forward);
            let fft_b = planner.plan_fft(1234, FftDirection::Forward);
            assert!(Arc::ptr_eq(&fft_a, &fft_b), "Existing fft was not reused");
        }
        {
            // Check that FFTs are reused if they're both inverse
            let mut planner = FftPlannerScalar::<f64>::new();
            let fft_a = planner.plan_fft(1234, FftDirection::Inverse);
            let fft_b = planner.plan_fft(1234, FftDirection::Inverse);
            assert!(Arc::ptr_eq(&fft_a, &fft_b), "Existing fft was not reused");
        }
        {
            // Check that FFTs are NOT resued if they don't both have the same direction
            let mut planner = FftPlannerScalar::<f64>::new();
            let fft_a = planner.plan_fft(1234, FftDirection::Forward);
            let fft_b = planner.plan_fft(1234, FftDirection::Inverse);
            assert!(
                !Arc::ptr_eq(&fft_a, &fft_b),
                "Existing fft was reused, even though directions don't match"
            );
        }
    }

    #[test]
    fn test_scalar_recipe_cache() {
        // Check that all butterflies are used
        let mut planner = FftPlannerScalar::<f64>::new();
        let fft_a = planner.design_fft_for_len(1234);
        let fft_b = planner.design_fft_for_len(1234);
        assert!(Rc::ptr_eq(&fft_a, &fft_b), "Existing recipe was not reused");
    }
}
