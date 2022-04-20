use num_integer::gcd;
use std::any::TypeId;
use std::collections::HashMap;

use std::sync::Arc;

use crate::{common::FftNum, fft_cache::FftCache, FftDirection};

use crate::algorithm::*;
use crate::neon::neon_butterflies::*;
use crate::neon::neon_prime_butterflies::*;
use crate::neon::neon_radix4::*;
use crate::Fft;

use crate::math_utils::{PrimeFactor, PrimeFactors};

const MIN_RADIX4_BITS: u32 = 6; // smallest size to consider radix 4 an option is 2^6 = 64
const MAX_RADER_PRIME_FACTOR: usize = 23; // don't use Raders if the inner fft length has prime factor larger than this
const MIN_BLUESTEIN_MIXED_RADIX_LEN: usize = 90; // only use mixed radix for the inner fft of Bluestein if length is larger than this

/// A Recipe is a structure that describes the design of a FFT, without actually creating it.
/// It is used as a middle step in the planning process.
#[derive(Debug, PartialEq, Clone)]
pub enum Recipe {
    Dft(usize),
    MixedRadix {
        left_fft: Arc<Recipe>,
        right_fft: Arc<Recipe>,
    },
    #[allow(dead_code)]
    GoodThomasAlgorithm {
        left_fft: Arc<Recipe>,
        right_fft: Arc<Recipe>,
    },
    MixedRadixSmall {
        left_fft: Arc<Recipe>,
        right_fft: Arc<Recipe>,
    },
    GoodThomasAlgorithmSmall {
        left_fft: Arc<Recipe>,
        right_fft: Arc<Recipe>,
    },
    RadersAlgorithm {
        inner_fft: Arc<Recipe>,
    },
    BluesteinsAlgorithm {
        len: usize,
        inner_fft: Arc<Recipe>,
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
    Butterfly9,
    Butterfly10,
    Butterfly11,
    Butterfly12,
    Butterfly13,
    Butterfly15,
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
            Recipe::Dft(length) => *length,
            Recipe::Radix4(length) => *length,
            Recipe::Butterfly1 => 1,
            Recipe::Butterfly2 => 2,
            Recipe::Butterfly3 => 3,
            Recipe::Butterfly4 => 4,
            Recipe::Butterfly5 => 5,
            Recipe::Butterfly6 => 6,
            Recipe::Butterfly7 => 7,
            Recipe::Butterfly8 => 8,
            Recipe::Butterfly9 => 9,
            Recipe::Butterfly10 => 10,
            Recipe::Butterfly11 => 11,
            Recipe::Butterfly12 => 12,
            Recipe::Butterfly13 => 13,
            Recipe::Butterfly15 => 15,
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

/// The Neon FFT planner creates new FFT algorithm instances using a mix of scalar and Neon accelerated algorithms.
/// It is supported when using the 64-bit AArch64 instruction set.
///
/// RustFFT has several FFT algorithms available. For a given FFT size, the `FftPlannerNeon` decides which of the
/// available FFT algorithms to use and then initializes them.
///
/// ~~~
/// // Perform a forward Fft of size 1234
/// use std::sync::Arc;
/// use rustfft::{FftPlannerNeon, num_complex::Complex};
///
/// if let Ok(mut planner) = FftPlannerNeon::new() {
///   let fft = planner.plan_fft_forward(1234);
///
///   let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
///   fft.process(&mut buffer);
///
///   // The FFT instance returned by the planner has the type `Arc<dyn Fft<T>>`,
///   // where T is the numeric type, ie f32 or f64, so it's cheap to clone
///   let fft_clone = Arc::clone(&fft);
/// }
/// ~~~
///
/// If you plan on creating multiple FFT instances, it is recommended to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
/// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
/// by a different planner)
///
/// Each FFT instance owns [`Arc`s](std::sync::Arc) to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating Fft instances.
pub struct FftPlannerNeon<T: FftNum> {
    algorithm_cache: FftCache<T>,
    recipe_cache: HashMap<usize, Arc<Recipe>>,
}

impl<T: FftNum> FftPlannerNeon<T> {
    /// Creates a new `FftPlannerNeon` instance.
    ///
    /// Returns `Ok(planner_instance)` if this machine has the required instruction sets.
    /// Returns `Err(())` if some instruction sets are missing.
    pub fn new() -> Result<Self, ()> {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Ideally, we would implement the planner with specialization.
            // Specialization won't be on stable rust for a long time though, so in the meantime, we can hack around it.
            //
            // We use TypeID to determine if T is f32, f64, or neither. If neither, we don't want to do any Neon acceleration
            // If it's f32 or f64, then construct and return a Neon planner instance.
            //
            // All Neon accelerated algorithms come in separate versions for f32 and f64. The type is checked when a new one is created, and if it does not
            // match the type the FFT is meant for, it will panic. This will never be a problem if using a planner to construct the FFTs.
            //
            // An annoying snag with this setup is that we frequently have to transmute buffers from &mut [Complex<T>] to &mut [Complex<f32 or f64>] or vice versa.
            // We know this is safe because we assert everywhere that Type(f32 or f64)==Type(T), so it's just a matter of "doing it right" every time.
            // These transmutes are required because the FFT algorithm's input will come through the FFT trait, which may only be bounded by FftNum.
            // So the buffers will have the type &mut [Complex<T>].
            let id_f32 = TypeId::of::<f32>();
            let id_f64 = TypeId::of::<f64>();
            let id_t = TypeId::of::<T>();

            if id_t == id_f32 || id_t == id_f64 {
                return Ok(Self {
                    algorithm_cache: FftCache::new(),
                    recipe_cache: HashMap::new(),
                });
            }
        }
        Err(())
    }

    /// Returns a `Fft` instance which uses Neon instructions to compute FFTs of size `len`.
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

    /// Returns a `Fft` instance which uses Neon instructions to compute forward FFTs of size `len`
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    /// Returns a `Fft` instance which uses Neon instructions to compute inverse FFTs of size `len.
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }

    // Make a recipe for a length
    fn design_fft_for_len(&mut self, len: usize) -> Arc<Recipe> {
        if len < 1 {
            Arc::new(Recipe::Dft(len))
        } else if let Some(recipe) = self.recipe_cache.get(&len) {
            Arc::clone(&recipe)
        } else {
            let factors = PrimeFactors::compute(len);
            let recipe = self.design_fft_with_factors(len, factors);
            self.recipe_cache.insert(len, Arc::clone(&recipe));
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
        let id_f32 = TypeId::of::<f32>();
        let id_f64 = TypeId::of::<f64>();
        let id_t = TypeId::of::<T>();

        match recipe {
            Recipe::Dft(len) => Arc::new(Dft::new(*len, direction)) as Arc<dyn Fft<T>>,
            Recipe::Radix4(len) => {
                if id_t == id_f32 {
                    Arc::new(Neon32Radix4::new(*len, direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(Neon64Radix4::new(*len, direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly1 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly1::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly1::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly2 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly2::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly2::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly3 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly3::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly3::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly4 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly4::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly4::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly5 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly5::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly5::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly6 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly6::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly6::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly7 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly7::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly7::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly8 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly8::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly8::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly9 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly9::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly9::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly10 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly10::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly10::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly11 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly11::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly11::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly12 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly12::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly12::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly13 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly13::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly13::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly15 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly15::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly15::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly16 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly16::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly16::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly17 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly17::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly17::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly19 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly19::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly19::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly23 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly23::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly23::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly29 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly29::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly29::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly31 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly31::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly31::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
            Recipe::Butterfly32 => {
                if id_t == id_f32 {
                    Arc::new(NeonF32Butterfly32::new(direction)) as Arc<dyn Fft<T>>
                } else if id_t == id_f64 {
                    Arc::new(NeonF64Butterfly32::new(direction)) as Arc<dyn Fft<T>>
                } else {
                    panic!("Not f32 or f64");
                }
            }
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

    fn design_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Arc<Recipe> {
        if let Some(fft_instance) = self.design_butterfly_algorithm(len) {
            fft_instance
        } else if factors.is_prime() {
            self.design_prime(len)
        } else if len.trailing_zeros() >= MIN_RADIX4_BITS {
            if len.is_power_of_two() {
                Arc::new(Recipe::Radix4(len))
            } else {
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
            // Can we do this as a mixed radix with just two butterflies?
            // Loop through and find all combinations
            // If more than one is found, keep the one where the factors are closer together.
            // For example length 20 where 10x2 and 5x4 are possible, we use 5x4.
            let butterflies: [usize; 20] = [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 23, 29, 31, 32,
            ];
            let mut bf_left = 0;
            let mut bf_right = 0;
            // If the length is below 14, or over 1024 we don't need to try this.
            if len > 13 && len <= 1024 {
                for (n, bf_l) in butterflies.iter().enumerate() {
                    if len % bf_l == 0 {
                        let bf_r = len / bf_l;
                        if butterflies.iter().skip(n).any(|&m| m == bf_r) {
                            bf_left = *bf_l;
                            bf_right = bf_r;
                        }
                    }
                }
                if bf_left > 0 {
                    let fact_l = PrimeFactors::compute(bf_left);
                    let fact_r = PrimeFactors::compute(bf_right);
                    return self.design_mixed_radix(fact_l, fact_r);
                }
            }
            // Not possible with just butterflies, go with the general solution.
            let (left_factors, right_factors) = factors.partition_factors();
            self.design_mixed_radix(left_factors, right_factors)
        }
    }

    fn design_mixed_radix(
        &mut self,
        left_factors: PrimeFactors,
        right_factors: PrimeFactors,
    ) -> Arc<Recipe> {
        let left_len = left_factors.get_product();
        let right_len = right_factors.get_product();

        //neither size is a butterfly, so go with the normal algorithm
        let left_fft = self.design_fft_with_factors(left_len, left_factors);
        let right_fft = self.design_fft_with_factors(right_len, right_factors);

        //if both left_len and right_len are small, use algorithms optimized for small FFTs
        if left_len < 33 && right_len < 33 {
            // for small FFTs, if gcd is 1, good-thomas is faster
            if gcd(left_len, right_len) == 1 {
                Arc::new(Recipe::GoodThomasAlgorithmSmall {
                    left_fft,
                    right_fft,
                })
            } else {
                Arc::new(Recipe::MixedRadixSmall {
                    left_fft,
                    right_fft,
                })
            }
        } else {
            Arc::new(Recipe::MixedRadix {
                left_fft,
                right_fft,
            })
        }
    }

    // Returns Some(instance) if we have a butterfly available for this size. Returns None if there is no butterfly available for this size
    fn design_butterfly_algorithm(&mut self, len: usize) -> Option<Arc<Recipe>> {
        match len {
            1 => Some(Arc::new(Recipe::Butterfly1)),
            2 => Some(Arc::new(Recipe::Butterfly2)),
            3 => Some(Arc::new(Recipe::Butterfly3)),
            4 => Some(Arc::new(Recipe::Butterfly4)),
            5 => Some(Arc::new(Recipe::Butterfly5)),
            6 => Some(Arc::new(Recipe::Butterfly6)),
            7 => Some(Arc::new(Recipe::Butterfly7)),
            8 => Some(Arc::new(Recipe::Butterfly8)),
            9 => Some(Arc::new(Recipe::Butterfly9)),
            10 => Some(Arc::new(Recipe::Butterfly10)),
            11 => Some(Arc::new(Recipe::Butterfly11)),
            12 => Some(Arc::new(Recipe::Butterfly12)),
            13 => Some(Arc::new(Recipe::Butterfly13)),
            15 => Some(Arc::new(Recipe::Butterfly15)),
            16 => Some(Arc::new(Recipe::Butterfly16)),
            17 => Some(Arc::new(Recipe::Butterfly17)),
            19 => Some(Arc::new(Recipe::Butterfly19)),
            23 => Some(Arc::new(Recipe::Butterfly23)),
            29 => Some(Arc::new(Recipe::Butterfly29)),
            31 => Some(Arc::new(Recipe::Butterfly31)),
            32 => Some(Arc::new(Recipe::Butterfly32)),
            _ => None,
        }
    }

    fn design_prime(&mut self, len: usize) -> Arc<Recipe> {
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
                    Arc::new(Recipe::Radix4(inner_fft_len_pow2))
                };
            Arc::new(Recipe::BluesteinsAlgorithm { len, inner_fft })
        } else {
            let inner_fft = self.design_fft_with_factors(inner_fft_len_rader, raders_factors);
            Arc::new(Recipe::RadersAlgorithm { inner_fft })
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
    fn test_plan_neon_trivial() {
        // Length 0 and 1 should use Dft
        let mut planner = FftPlannerNeon::<f64>::new().unwrap();
        for len in 0..1 {
            let plan = planner.design_fft_for_len(len);
            assert_eq!(*plan, Recipe::Dft(len));
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_neon_largepoweroftwo() {
        // Powers of 2 above 6 should use Radix4
        let mut planner = FftPlannerNeon::<f64>::new().unwrap();
        for pow in 6..32 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert_eq!(*plan, Recipe::Radix4(len));
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[test]
    fn test_plan_neon_butterflies() {
        // Check that all butterflies are used
        let mut planner = FftPlannerNeon::<f64>::new().unwrap();
        assert_eq!(*planner.design_fft_for_len(2), Recipe::Butterfly2);
        assert_eq!(*planner.design_fft_for_len(3), Recipe::Butterfly3);
        assert_eq!(*planner.design_fft_for_len(4), Recipe::Butterfly4);
        assert_eq!(*planner.design_fft_for_len(5), Recipe::Butterfly5);
        assert_eq!(*planner.design_fft_for_len(6), Recipe::Butterfly6);
        assert_eq!(*planner.design_fft_for_len(7), Recipe::Butterfly7);
        assert_eq!(*planner.design_fft_for_len(8), Recipe::Butterfly8);
        assert_eq!(*planner.design_fft_for_len(9), Recipe::Butterfly9);
        assert_eq!(*planner.design_fft_for_len(10), Recipe::Butterfly10);
        assert_eq!(*planner.design_fft_for_len(11), Recipe::Butterfly11);
        assert_eq!(*planner.design_fft_for_len(12), Recipe::Butterfly12);
        assert_eq!(*planner.design_fft_for_len(13), Recipe::Butterfly13);
        assert_eq!(*planner.design_fft_for_len(15), Recipe::Butterfly15);
        assert_eq!(*planner.design_fft_for_len(16), Recipe::Butterfly16);
        assert_eq!(*planner.design_fft_for_len(17), Recipe::Butterfly17);
        assert_eq!(*planner.design_fft_for_len(19), Recipe::Butterfly19);
        assert_eq!(*planner.design_fft_for_len(23), Recipe::Butterfly23);
        assert_eq!(*planner.design_fft_for_len(29), Recipe::Butterfly29);
        assert_eq!(*planner.design_fft_for_len(31), Recipe::Butterfly31);
        assert_eq!(*planner.design_fft_for_len(32), Recipe::Butterfly32);
    }

    #[test]
    fn test_plan_neon_mixedradix() {
        // Products of several different primes should become MixedRadix
        let mut planner = FftPlannerNeon::<f64>::new().unwrap();
        for pow2 in 2..5 {
            for pow3 in 2..5 {
                for pow5 in 2..5 {
                    for pow7 in 2..5 {
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
    fn test_plan_neon_mixedradixsmall() {
        // Products of two "small" lengths < 31 that have a common divisor >1, and isn't a power of 2 should be MixedRadixSmall
        let mut planner = FftPlannerNeon::<f64>::new().unwrap();
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
    fn test_plan_neon_goodthomasbutterfly() {
        let mut planner = FftPlannerNeon::<f64>::new().unwrap();
        for len in [3 * 7, 5 * 7, 11 * 13, 2 * 29].iter() {
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
    fn test_plan_neon_bluestein_vs_rader() {
        let difficultprimes: [usize; 11] = [59, 83, 107, 149, 167, 173, 179, 359, 719, 1439, 2879];
        let easyprimes: [usize; 24] = [
            53, 61, 67, 71, 73, 79, 89, 97, 101, 103, 109, 113, 127, 131, 137, 139, 151, 157, 163,
            181, 191, 193, 197, 199,
        ];

        let mut planner = FftPlannerNeon::<f64>::new().unwrap();
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
    fn test_neon_fft_cache() {
        {
            // Check that FFTs are reused if they're both forward
            let mut planner = FftPlannerNeon::<f64>::new().unwrap();
            let fft_a = planner.plan_fft(1234, FftDirection::Forward);
            let fft_b = planner.plan_fft(1234, FftDirection::Forward);
            assert!(Arc::ptr_eq(&fft_a, &fft_b), "Existing fft was not reused");
        }
        {
            // Check that FFTs are reused if they're both inverse
            let mut planner = FftPlannerNeon::<f64>::new().unwrap();
            let fft_a = planner.plan_fft(1234, FftDirection::Inverse);
            let fft_b = planner.plan_fft(1234, FftDirection::Inverse);
            assert!(Arc::ptr_eq(&fft_a, &fft_b), "Existing fft was not reused");
        }
        {
            // Check that FFTs are NOT resued if they don't both have the same direction
            let mut planner = FftPlannerNeon::<f64>::new().unwrap();
            let fft_a = planner.plan_fft(1234, FftDirection::Forward);
            let fft_b = planner.plan_fft(1234, FftDirection::Inverse);
            assert!(
                !Arc::ptr_eq(&fft_a, &fft_b),
                "Existing fft was reused, even though directions don't match"
            );
        }
    }

    #[test]
    fn test_neon_recipe_cache() {
        // Check that all butterflies are used
        let mut planner = FftPlannerNeon::<f64>::new().unwrap();
        let fft_a = planner.design_fft_for_len(1234);
        let fft_b = planner.design_fft_for_len(1234);
        assert!(
            Arc::ptr_eq(&fft_a, &fft_b),
            "Existing recipe was not reused"
        );
    }
}
