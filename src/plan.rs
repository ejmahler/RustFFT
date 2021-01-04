use num_integer::gcd;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::{common::FftNum, fft_cache::FftCache, FftDirection};

use crate::algorithm::butterflies::*;
use crate::algorithm::*;
use crate::Fft;

use crate::FftPlannerAvx;

use crate::math_utils::PrimeFactors;

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
/// If you plan on creating multiple FFT instances, it is recommnded to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
/// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
/// by a different planner)
///
/// Each FFT instance owns [`Arc`s](std::sync::Arc) to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating Fft instances.
pub enum FftPlanner<T: FftNum> {
    Scalar(FftPlannerScalar<T>),
    Avx(FftPlannerAvx<T>),
}
impl<T: FftNum> FftPlanner<T> {
    /// Creates a new `FftPlanner` instance. It detects if AVX is supported on the current machine. If it is, it will plan AVX-accelerated FFTs.
    /// If AVX isn't supported, it will seamlessly fall back to planning non-SIMD FFTs.
    pub fn new() -> Self {
        if let Ok(avx_planner) = FftPlannerAvx::new() {
            Self::Avx(avx_planner)
        } else {
            Self::Scalar(FftPlannerScalar::new())
        }
    }

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        match self {
            Self::Scalar(scalar_planner) => scalar_planner.plan_fft(len, direction),
            Self::Avx(avx_planner) => avx_planner.plan_fft(len, direction),
        }
    }

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }
}

const SLOWDOWN_LEN: usize = 100000; // lenght where we run out of cpu cache and algorithms tend to slow down
const SMALL_LEN: usize = 32; // limit of "small" length for mixed radix algos

/// A Recipe is a structure that describes the design of a FFT, without actually creating it.
/// It is used as a middle step in the planning process.
#[derive(Debug, std::cmp::PartialEq, Clone)]
pub enum Recipe {
    DFT(usize),
    MixedRadix {
        left_fft: Rc<Recipe>,
        right_fft: Rc<Recipe>,
    },
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
    Butterfly1,
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
            Recipe::Butterfly1 => 1,
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

    pub fn cost(&self) -> f32 {
        match self {
            Recipe::DFT(len) => 0.9 * (*len as f32).powf(2.2),
            Recipe::Radix4(len) => {
                let mut cost = 1.2 * (*len as f32).powf(1.2);
                if *len > SLOWDOWN_LEN {
                    cost += 10.0 * (*len - SLOWDOWN_LEN) as f32;
                }
                cost
            }
            Recipe::Butterfly1 => 0.3,
            Recipe::Butterfly2 => 0.625,
            Recipe::Butterfly3 => 0.848,
            Recipe::Butterfly4 => 1.36,
            Recipe::Butterfly5 => 2.65,
            Recipe::Butterfly6 => 2.11,
            Recipe::Butterfly7 => 6.46,
            Recipe::Butterfly8 => 5.45,
            Recipe::Butterfly11 => 12.9,
            Recipe::Butterfly13 => 17.6,
            Recipe::Butterfly16 => 25.8,
            Recipe::Butterfly17 => 31.0,
            Recipe::Butterfly19 => 39.6,
            Recipe::Butterfly23 => 98.0,
            Recipe::Butterfly29 => 198.0,
            Recipe::Butterfly31 => 233.0,
            Recipe::Butterfly32 => 86.3,
            Recipe::MixedRadix {
                left_fft,
                right_fft,
            }
            | Recipe::MixedRadixSmall {
                left_fft,
                right_fft,
            } => {
                let len = self.len();
                let inners_cost = left_fft.cost() * right_fft.len() as f32
                    + right_fft.cost() * left_fft.len() as f32;
                let mut twiddle_cost = 15.0 + 1.3 * len as f32;
                if len > SLOWDOWN_LEN {
                    twiddle_cost += 10.0 * (len - SLOWDOWN_LEN) as f32;
                }
                inners_cost + twiddle_cost
            }
            Recipe::GoodThomasAlgorithm {
                left_fft,
                right_fft,
            } => {
                let len = self.len();
                let inners_cost = left_fft.cost() * right_fft.len() as f32
                    + right_fft.cost() * left_fft.len() as f32;
                // This means this algo will never be used, good since there currently a bug somewhere if one inner is Bluestein
                let mut twiddle_cost = 10.0 + 1.4 * len as f32;
                if len > SLOWDOWN_LEN {
                    twiddle_cost += 10.0 * (len - SLOWDOWN_LEN) as f32;
                }
                inners_cost + twiddle_cost
            }
            Recipe::GoodThomasAlgorithmSmall {
                left_fft,
                right_fft,
            } => {
                let len = self.len();
                let inners_cost = left_fft.cost() * right_fft.len() as f32
                    + right_fft.cost() * left_fft.len() as f32;
                let mut twiddle_cost = 10.0 + 1.3 * len as f32;
                if len > SLOWDOWN_LEN {
                    twiddle_cost += 10.0 * (len - SLOWDOWN_LEN) as f32;
                }
                inners_cost + twiddle_cost
            }
            Recipe::RadersAlgorithm { inner_fft } => {
                5.0 * self.len() as f32 + 2.0 * inner_fft.cost()
            }
            Recipe::BluesteinsAlgorithm { len, inner_fft } => {
                4.0 * *len as f32 + 2.0 * inner_fft.cost()
            }
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
/// If you plan on creating multiple FFT instances, it is recommnded to reuse the same planner for all of them. This
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

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        // Step 1: Create a "recipe" for this FFT, which will tell us exactly which combination of algorithms to use
        let recipe = self.design_fft_for_len(len);

        // Step 2: Use our recipe to construct a Fft trait object
        self.build_fft(&recipe, direction)
    }

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    /// Returns a `Fft` instance which processes signals of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }

    // Make a recipe for a length
    fn design_fft_for_len(&mut self, len: usize) -> Rc<Recipe> {
        if len == 0 {
            Rc::new(Recipe::DFT(0))
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
            Recipe::Butterfly1 => Arc::new(Butterfly2::new(direction)) as Arc<dyn Fft<T>>,
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
        let recipes = self.design_all_ffts_with_factors(len, factors);
        let fastest = recipes
            .iter()
            .min_by(|x, y| {
                match x.cost().partial_cmp(&y.cost()) {
                    Some(Ordering::Equal) => Ordering::Equal,
                    Some(Ordering::Less) => Ordering::Less,
                    Some(Ordering::Greater) => Ordering::Greater,
                    None => Ordering::Equal, //This shoud never happen
                }
            })
            .unwrap();
        Rc::clone(fastest)
    }

    fn design_all_ffts_with_factors(
        &mut self,
        len: usize,
        factors: PrimeFactors,
    ) -> Vec<Rc<Recipe>> {
        let mut recipes = Vec::new();
        if let Some(butterfly) = self.design_butterfly_algorithm(len) {
            recipes.push(butterfly);
            // We can't beat the butterflies, stop here
            return recipes;
        }
        if let Some(mixedradix) = self.design_mixedradix(&factors) {
            recipes.push(mixedradix);
        }
        if let Some(radix4) = self.design_radix4(&factors) {
            recipes.push(radix4);
            // We have a power of two, return here since nothing below here can be faster
            return recipes;
        }
        if let Some(mixedradixpow2) = self.design_mixedradix_separate_twos(&factors) {
            recipes.push(mixedradixpow2);
        }
        if let Some(goodthomas) = self.design_goodthomas(&factors) {
            recipes.push(goodthomas);
        }
        if let Some(raders) = self.design_raders(&factors) {
            recipes.push(raders);
        }
        if let Some(bluestein) = self.design_bluesteins(&factors) {
            recipes.push(bluestein);
        }
        recipes
    }

    fn design_mixedradix(&mut self, factors: &PrimeFactors) -> Option<Rc<Recipe>> {
        if factors.is_prime() {
            None
        } else {
            let factors = factors.clone();
            let (left_factors, right_factors) = factors.partition_factors();
            let left_len = left_factors.get_product();
            let right_len = right_factors.get_product();
            let left_fft = self.design_fft_for_len(left_len);
            let right_fft = self.design_fft_for_len(right_len);

            // if total length is small, use algorithms optimized for small FFTs
            if right_len <= SMALL_LEN && left_len <= SMALL_LEN {
                Some(Rc::new(Recipe::MixedRadixSmall {
                    left_fft,
                    right_fft,
                }))
            } else {
                Some(Rc::new(Recipe::MixedRadix {
                    left_fft,
                    right_fft,
                }))
            }
        }
    }

    fn design_mixedradix_separate_twos(&mut self, factors: &PrimeFactors) -> Option<Rc<Recipe>> {
        let len = factors.get_product();
        if factors.is_prime() || len.is_power_of_two() || len.trailing_zeros() < 1 {
            None
        } else {
            let power_of_two = 1 << len.trailing_zeros();
            let non_power_of_two = len / power_of_two;
            let left_fft = self.design_fft_for_len(power_of_two);
            let right_fft = self.design_fft_for_len(non_power_of_two);

            // if total length is small, use algorithms optimized for small FFTs
            if power_of_two <= SMALL_LEN && non_power_of_two <= SMALL_LEN {
                Some(Rc::new(Recipe::MixedRadixSmall {
                    left_fft,
                    right_fft,
                }))
            } else {
                Some(Rc::new(Recipe::MixedRadix {
                    left_fft,
                    right_fft,
                }))
            }
        }
    }

    fn design_goodthomas(&mut self, factors: &PrimeFactors) -> Option<Rc<Recipe>> {
        if factors.is_prime() {
            None
        } else {
            let factors = factors.clone();
            let (left_factors, right_factors) = factors.partition_factors();
            let left_len = left_factors.get_product();
            let right_len = right_factors.get_product();
            if gcd(left_len, right_len) > 1 {
                None
            } else {
                let left_fft = self.design_fft_for_len(left_len);
                let right_fft = self.design_fft_for_len(right_len);

                //if both left_len and right_len are small, use algorithms optimized for small FFTs
                if right_len <= SMALL_LEN && left_len <= SMALL_LEN {
                    Some(Rc::new(Recipe::GoodThomasAlgorithmSmall {
                        left_fft,
                        right_fft,
                    }))
                } else {
                    Some(Rc::new(Recipe::GoodThomasAlgorithm {
                        left_fft,
                        right_fft,
                    }))
                }
            }
        }
    }

    fn design_radix4(&mut self, factors: &PrimeFactors) -> Option<Rc<Recipe>> {
        let len = factors.get_product();
        if !len.is_power_of_two() || len < 4 {
            None
        } else {
            Some(Rc::new(Recipe::Radix4(len)))
        }
    }

    // Returns Some(instance) if we have a butterfly available for this size. Returns None if there is no butterfly available for this size
    fn design_butterfly_algorithm(&mut self, len: usize) -> Option<Rc<Recipe>> {
        match len {
            1 => Some(Rc::new(Recipe::Butterfly1)),
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

    fn design_raders(&mut self, factors: &PrimeFactors) -> Option<Rc<Recipe>> {
        if !factors.is_prime() {
            // For Raders the length must be a prime
            None
        } else {
            let len = factors.get_product();
            let inner_fft = self.design_fft_for_len(len - 1);
            Some(Rc::new(Recipe::RadersAlgorithm { inner_fft }))
        }
    }

    fn design_bluesteins(&mut self, factors: &PrimeFactors) -> Option<Rc<Recipe>> {
        if factors.get_other_factors().is_empty() {
            // Don't propose a recipe for simple lengths. 
            // This is mostly to stop a Bluestein from trying a Bluestein inner, which will try a Bluestein inner and so on forever.
            None
        } else {
            let len = factors.get_product();

            // for long ffts a mixed radix inner fft is faster than a longer radix4
            let min_inner_len = 2 * len - 1;
            let inner_fft_len_pow2 = min_inner_len.checked_next_power_of_two().unwrap();
            let mixed_radix_len = 3 * inner_fft_len_pow2 / 4;
            let inner_fft_len = if mixed_radix_len >= min_inner_len {
                mixed_radix_len
            } else {
                inner_fft_len_pow2
            };
            let inner_fft = self.design_fft_for_len(inner_fft_len);
            Some(Rc::new(Recipe::BluesteinsAlgorithm { len, inner_fft }))
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
        // Length 0 should use DFT
        let mut planner = FftPlannerScalar::<f64>::new();
        for len in 0..1 {
            let plan = planner.design_fft_for_len(len);
            assert_eq!(*plan, Recipe::DFT(len));
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_scalar_mediumpoweroftwo() {
        // Powers of 2 between 64 and 4096 should use Radix4
        let mut planner = FftPlannerScalar::<f64>::new();
        for pow in 6..12 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert_eq!(
                *plan,
                Recipe::Radix4(len),
                "Length: {}, expected Radix4, got {:?}",
                len,
                plan
            );
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
            assert!(
                is_mixedradix(&plan),
                "Length: {}, expected MixedRadix, got {:?}",
                len,
                plan
            );
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_scalar_butterflies() {
        // Check that all butterflies are used
        let mut planner = FftPlannerScalar::<f64>::new();
        assert_eq!(*planner.design_fft_for_len(1), Recipe::Butterfly1);
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
                        assert!(
                            is_mixedradix(&plan),
                            "Length: {}, expected MixedRadix, got {:?}",
                            len,
                            plan
                        );
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
                "Length: {}, expected MixedRadixSmall, got {:?}",
                len,
                plan
            );
            assert_eq!(plan.len(), *len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_scalar_goodthomassmall() {
        let mut planner = FftPlannerScalar::<f64>::new();
        for prime1 in &[5, 7, 11, 13, 17, 19, 23, 29, 31] {
            for prime2 in &[5, 7, 11, 13, 17, 19, 23, 29, 31] {
                if prime1 != prime2 {
                    let len = prime1*prime2;
                    let plan = planner.design_fft_for_len(len);
                    assert!(is_goodthomassmall(&plan), "Len: {}, expected GoodThomasAlgorithmSmall, got {:?}", len ,plan);
                    assert_eq!(plan.len(), len, "Recipe reports wrong length");
                }
            }
        }
    }

    #[test]
    fn test_plan_scalar_bluestein_vs_rader() {
        let difficultprimes: [usize; 10] = [59, 83, 107, 167, 173, 179, 359, 719, 1439, 2879];
        let easyprimes: [usize; 24] = [
            53, 61, 67, 71, 73, 79, 89, 97, 101, 103, 109, 113, 131, 137, 139, 149, 151, 157, 163,
            181, 191, 193, 197, 199,
        ];

        let mut planner = FftPlannerScalar::<f64>::new();
        for len in difficultprimes.iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(
                is_bluesteins(&plan),
                "Length: {}, Expected BluesteinsAlgorithm, got {:?}",
                len,
                plan
            );
            assert_eq!(plan.len(), *len, "Recipe reports wrong length");
        }
        for len in easyprimes.iter() {
            let plan = planner.design_fft_for_len(*len);
            assert!(
                is_raders(&plan),
                "Length: {}, Expected RadersAlgorithm, got {:?}",
                len,
                plan
            );
            assert_eq!(plan.len(), *len, "Recipe reports wrong length");
        }
    }

    // For these lengths a Bluesteins would be faster than the MixedRadix that currently gets chosen
    //#[test]
    //fn test_plan_scalar_bluestein_specials() {
    //    let good_for_bluesteins: [usize; 12] = [619, 807, 811, 815, 830, 835, 865, 895, 913, 919, 921, 991]; 
    //    let mut planner = FftPlannerScalar::<f64>::new();
    //    for len in good_for_bluesteins.iter() {
    //        let plan = planner.design_fft_for_len(*len);
    //        assert!(
    //            is_bluesteins(&plan),
    //            "Length: {}, Expected BluesteinsAlgorithm, got {:?}",
    //            len,
    //            plan
    //        );
    //        assert_eq!(plan.len(), *len, "Recipe reports wrong length");
    //    }
    //}

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
