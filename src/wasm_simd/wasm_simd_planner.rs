use crate::{fft_cache::FftCache, math_utils::PrimeFactors, Fft, FftDirection, FftNum};
use std::{any::TypeId, collections::HashMap, sync::Arc};

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

/// The WASM FFT planner creates new FFT algorithm instances using a mix of scalar and WASM SIMD accelerated algorithms.
/// It is supported when using fairly recent browser versions as outlined in [the WebAssembly roadmap](https://webassembly.org/roadmap/).
///
/// RustFFT has several FFT algorithms available. For a given FFT size, `FftPlannerWasmSimd` decides which of the
/// available FFT algorithms to use and then initializes them.
///
/// ~~~
/// // Perform a forward Fft of size 1234
/// use std::sync::Arc;
/// use rustfft::{FftPlannerWasmSimd, num_complex::Complex};
///
/// if let Ok(mut planner) = FftPlannerWasmSimd::new() {
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
pub struct FftPlannerWasmSimd<T: FftNum> {
    algorithm_cache: FftCache<T>,
    recipe_cache: HashMap<usize, Arc<Recipe>>,
}
impl<T: FftNum> FftPlannerWasmSimd<T> {
    /// Creates a new `FftPlannerNeon` instance.
    ///
    /// Returns `Ok(planner_instance)` if this machine has the required instruction sets.
    /// Returns `Err(())` if some instruction sets are missing.
    pub fn new() -> Result<Self, ()> {
        let id_f32 = TypeId::of::<f32>();
        let id_f64 = TypeId::of::<f64>();
        let id_t = TypeId::of::<T>();

        if id_t != id_f32 && id_t != id_f64 {
            return Err(());
        }

        Ok(Self {
            algorithm_cache: FftCache::new(),
            recipe_cache: HashMap::new(),
        })
    }
    /// Returns a `Fft` instance which uses WebAssembly SIMD instructions to compute FFTs of size `len`.
    ///
    /// If the provided `direction` is `FftDirection::Forward`, the returned instance will compute forward FFTs. If it's `FftDirection::Inverse`, it will compute inverse FFTs.
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        unreachable!()
    }
    /// Returns a `Fft` instance which uses WebAssembly SIMD instructions to compute forward FFTs of size `len`.
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }
    /// Returns a `Fft` instance which uses Neon instructions to compute inverse FFTs of size `len.
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft_inverse(&mut self, _len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(_len, FftDirection::Inverse)
    }
}

impl<T: FftNum> FftPlannerWasmSimd<T> {
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

    fn build_new_fft(&mut self, recipe: &Recipe, direction: FftDirection) -> Arc<dyn Fft<T>> {
        todo!()
    }

    fn design_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Arc<Recipe> {
        todo!()
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use wasm_bindgen_test::*;

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

    #[wasm_bindgen_test]
    fn test_plan_sse_trivial() {
        // Length 0 and 1 should use Dft
        let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
        for len in 0..1 {
            let plan = planner.design_fft_for_len(len);
            assert_eq!(*plan, Recipe::Dft(len));
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[wasm_bindgen_test]
    fn test_plan_sse_largepoweroftwo() {
        // Powers of 2 above 6 should use Radix4
        let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
        for pow in 6..32 {
            let len = 1 << pow;
            let plan = planner.design_fft_for_len(len);
            assert_eq!(*plan, Recipe::Radix4(len));
            assert_eq!(plan.len(), len, "Recipe reports wrong length");
        }
    }

    #[wasm_bindgen_test]
    fn test_plan_sse_butterflies() {
        // Check that all butterflies are used
        let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
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

    #[wasm_bindgen_test]
    fn test_plan_sse_mixedradix() {
        // Products of several different primes should become MixedRadix
        let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
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

    #[wasm_bindgen_test]
    fn test_plan_sse_mixedradixsmall() {
        // Products of two "small" lengths < 31 that have a common divisor >1, and isn't a power of 2 should be MixedRadixSmall
        let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
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

    #[wasm_bindgen_test]
    fn test_plan_sse_goodthomasbutterfly() {
        let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
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

    #[wasm_bindgen_test]
    fn test_plan_sse_bluestein_vs_rader() {
        let difficultprimes: [usize; 11] = [59, 83, 107, 149, 167, 173, 179, 359, 719, 1439, 2879];
        let easyprimes: [usize; 24] = [
            53, 61, 67, 71, 73, 79, 89, 97, 101, 103, 109, 113, 127, 131, 137, 139, 151, 157, 163,
            181, 191, 193, 197, 199,
        ];

        let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
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

    #[wasm_bindgen_test]
    fn test_sse_fft_cache() {
        {
            // Check that FFTs are reused if they're both forward
            let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
            let fft_a = planner.plan_fft(1234, FftDirection::Forward);
            let fft_b = planner.plan_fft(1234, FftDirection::Forward);
            assert!(Arc::ptr_eq(&fft_a, &fft_b), "Existing fft was not reused");
        }
        {
            // Check that FFTs are reused if they're both inverse
            let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
            let fft_a = planner.plan_fft(1234, FftDirection::Inverse);
            let fft_b = planner.plan_fft(1234, FftDirection::Inverse);
            assert!(Arc::ptr_eq(&fft_a, &fft_b), "Existing fft was not reused");
        }
        {
            // Check that FFTs are NOT resued if they don't both have the same direction
            let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
            let fft_a = planner.plan_fft(1234, FftDirection::Forward);
            let fft_b = planner.plan_fft(1234, FftDirection::Inverse);
            assert!(
                !Arc::ptr_eq(&fft_a, &fft_b),
                "Existing fft was reused, even though directions don't match"
            );
        }
    }

    #[wasm_bindgen_test]
    fn test_sse_recipe_cache() {
        // Check that all butterflies are used
        let mut planner = FftPlannerWasmSimd::<f64>::new().unwrap();
        let fft_a = planner.design_fft_for_len(1234);
        let fft_b = planner.design_fft_for_len(1234);
        assert!(
            Arc::ptr_eq(&fft_a, &fft_b),
            "Existing recipe was not reused"
        );
    }
}
