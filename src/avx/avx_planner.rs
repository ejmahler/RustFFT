use std::sync::Arc;
use std::collections::HashMap;
use std::cmp::min;

use primal_check::miller_rabin;

use crate::algorithm::butterflies::*;
use crate::algorithm::*;
use crate::math_utils::PartialFactors;
use crate::common::FFTnum;
use crate::Fft;

use super::*;

fn wrap_fft<T: FFTnum>(butterfly: impl Fft<T> + 'static) -> Arc<dyn Fft<T>> {
    Arc::new(butterfly) as Arc<dyn Fft<T>>
}

/// repreesnts a FFT plan, stored as a base FFT and a stack of MixedRadix*xn on top of it.
#[derive(Debug)]
pub struct MixedRadixPlan {
    len: usize, // product of base and radixes
    base: usize, // either a butterfly, or a bluesteins/raders step
    radixes: Vec<u8>, // stored from smallest to largest
}
impl MixedRadixPlan {
    fn new(base: usize, radixes: &[u8]) -> Self {
        Self {
            len: base * radixes.iter().map(|r| *r as usize).product::<usize>(),
            base,
            radixes: radixes.to_vec(),
        }
    }
    fn push_radix(&mut self, radix: u8) {
        self.radixes.push(radix);
        self.len *= radix as usize;
    }
    fn push_radix_power(&mut self, radix: u8, power: u32) {
        self.radixes.extend(std::iter::repeat(radix).take(power as usize));
        self.len *= (radix as usize).pow(power);
    }
    fn is_base_only(&self) -> bool {
        self.radixes.len() == 0
    }
}

/// The FFT planner is used to make new FFT algorithm instances. Specifically, `FftPlannerAvx` generates FFT algorithms
/// that are designed with the AVX instruction set in mind.
///
/// Creating an instance of `FftPlannerAvx` requires the `avx` and `fma` instructions to be available on the current machine. A few algorithms will
/// use `avx2` if it's available, but it isn't required.
///
/// For the time being, AVX acceleration is black box, and AVX accelerated algorithms are not available without a planner. This may change in the future.
///
/// ~~~
/// // Perform a forward Fft of size 1234
/// use std::sync::Arc;
/// use rustfft::{FftPlannerAvx, num_complex::Complex};
///
/// // If FftPlannerAvx::new() returns Ok(), we'll know AVX algorithms are available on this machine
/// if let Ok(mut planner) = FftPlannerAvx::new(false) {
///     let fft = planner.plan_fft(1234);
///
///     let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
///     fft.process_inplace(&mut buffer);
/// 
///     // The FFT instance returned by the planner has the type `Arc<dyn Fft<T>>`,
///     // where T is the numeric type, ie f32 or f64, so it's cheap to clone
///     let fft_clone = Arc::clone(&fft);
/// }
/// ~~~
///
/// If you plan on creating multiple FFT instances, it is recommnded to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
/// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
/// by a different planner)
///
/// Each FFT instance owns [`Arc`s](std::sync::Arc) to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating Fft instances.
pub struct FftPlannerAvx<T: FFTnum> {
    algorithm_cache: HashMap<usize, Arc<dyn Fft<T>>>,
    inverse: bool,
}
impl<T: FFTnum> FftPlannerAvx<T> {
    /// Constructs a new `FftPlannerAvx` instance.
    ///
    /// Returns `Ok(planner_instance)` if this machine has the required instruction sets, Err() if some instruction sets are missing.
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

    /// Returns a `Fft` instance which processes signals of size `len` using AVX instructions.
    ///
    /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
    pub fn plan_fft(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_with_cache(len, Self::plan_and_construct_new_fft)
    }

    // If we already have the provided FFT len cached, return the cached value. If not, call `create_fn`to create it, and cache the results
    fn plan_with_cache(&mut self, len: usize, create_fn: impl FnOnce(&mut Self, usize) -> Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> {
        if let Some(instance) = self.algorithm_cache.get(&len) {
            Arc::clone(instance)
        } else {
            let instance = create_fn(self, len);
            self.algorithm_cache.insert(len, Arc::clone(&instance));
            instance
        }
    }

    fn plan_and_construct_new_fft(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        let factors = PartialFactors::compute(len);
        let base = self.plan_mixed_radix_base(len, &factors);

        // it's possible that the base planner plans out the whole FFT. it's guaranteed if `len` is a prime number, or if it's a butterfly, for example
        let plan = if base.len == len {
            base
        } else {
            let radix_factors = factors.divide_by(&PartialFactors::compute(base.len)).unwrap_or_else(|| panic!("Invalid base for FFT length={}, base={:?}, base radixes={:?}", len, base.base, base.radixes));
            self.plan_mixed_radix(radix_factors, base)
        };
        self.construct_plan(plan)
    }

    // given a set of factors, compute how many iterations of 12xn and 16xn we should plan for. Returns (k, j) for 12^k and 6^j
    fn plan_power12_power6(radix_factors: &PartialFactors) -> (u32, u32) {
        // it's helpful to think of this process as rewriting the FFT length as powers of our radixes
        // the fastest FFT we could possibly compute is 8^n, because the 8xn algorithm is blazing fast. 9xn and 12xn are also in the top tier for speed, so those 3 algorithms are what we will aim for
        // Specifically, we want to find a combination of 8, 9, and 12, that will "consume" all factors of 2 and 3, without having any leftovers
        
        // Unfortunately, most FFTs don't come in the form 8^n * 9^m * 12^k
        // Thankfully, 6xn is also reasonably fast, so we can use 6xn to strip away factors.
        // This function's job will be to divide radix_factors into 8^n * 9^m * 12^k * 6^j, which minimizes j, then maximizes k

        // we're going to hypothetically add as many 12's to our plan as possible, keeping track of how many 6's were required to balance things out
        // we can also compute this analytically with modular arithmetic, but that technique only works when the FFT is above a minimum size, but this loop+array technique always works
        let max_twelves = min(radix_factors.get_power2() / 2, radix_factors.get_power3());
        let mut required_sixes = [None; 4]; // only track 6^0 through 6^3. 6^4 can be converted into 12^2 * 9, and 6^5 can be converted into 12 * 8 * 9 * 9
        for hypothetical_twelve_power in 0..(max_twelves+1) {
            let hypothetical_twos = radix_factors.get_power2() - hypothetical_twelve_power * 2;
            let hypothetical_threes = radix_factors.get_power3() - hypothetical_twelve_power;

            // figure out how many sixes we would need to leave our FFT at 8^n * 9^m via modular arithmetic, and write to that index of our twelves_per_sixes array
            let sixes = match (hypothetical_twos % 3, hypothetical_threes % 2) {
                (0, 0) => Some(0),
                (1, 1) => Some(1),
                (2, 0) => Some(2),
                (0, 1) => Some(3),
                (1, 0) => None, // it would take 4 sixes, which can be replaced by 2 twelves, so we'll hit it in a later loop (if we have that many factors)
                (2, 1) => None, // it would take 5 sixes, but note that 12 is literally 2^2 * 3^1, so instead of applying 5 sixes, we can apply a single 12
                (_, _) => unreachable!(),
            };

            // if we can bring the FFT into range for the fast path with sixes, record so in the required_sixes array
            // but make sure the number of sixes we're going to apply actually fits into our available factors
            if let Some(sixes) = sixes {
                if sixes <= hypothetical_twos && sixes <= hypothetical_threes {
                    required_sixes[sixes as usize] = Some(hypothetical_twelve_power)
                }
            }
        }

        // required_sixes[i] now contains the largest power of twelve that we can apply, given that we also apply 6^i
        // we want to apply as many of 12 as possible, so take the array element with the largest non-None element
        // note that it's possible (and very likely) that either power_twelve or power_six is zero, or both of them are zero! this will happen for a pure power of 2 or power of 3 FFT, for example
        let (power_twelve, mut power_six) = required_sixes
            .iter()
            .enumerate()
            .filter_map(|(i, maybe_twelve)| maybe_twelve.map(|twelve| (twelve, i as u32)))
            .fold((0, 0), |best, current| if current.0 >= best.0 { current } else { best });

        // special case: if we have exactly one factor of 2 and at least one factor of 3, unconditionally apply a factor of 6 to get rid of the 2
        if radix_factors.get_power2() == 1 && radix_factors.get_power3() > 0 {
            power_six = 1;
        }
        // special case: if we have a single factor of 3 and more than one factor of 2 (and we don't have any twelves), unconditionally apply a factor of 6 to get rid of the 3
        if radix_factors.get_power2() > 1 && radix_factors.get_power3() == 1 && power_twelve == 0 {
            power_six = 1;
        }

        (power_twelve, power_six)
    }

    fn plan_mixed_radix(&self, mut radix_factors: PartialFactors, mut plan: MixedRadixPlan) -> MixedRadixPlan {
        // if we can complete the FFT with a single radix, do it
        if [2,3,4,5,6,7,8,9,12,16].contains(&radix_factors.product()) {
            plan.push_radix(radix_factors.product() as u8)
        } else {
            // Compute how many powers of 12 and powers of 6 we want to strip away
            let (power_twelve, power_six) = Self::plan_power12_power6(&radix_factors);
            
            // divide our powers of 12 and 6 out of our radix factors
            radix_factors = radix_factors.divide_by(&PartialFactors::compute(6usize.pow(power_six) * 12usize.pow(power_twelve))).unwrap();

            // now that we know the 12 and 6 factors, the plan array can be computed in descending radix size
            if radix_factors.get_power2() % 3 == 1 && radix_factors.get_power2() > 1 {
                // our factors of 2 might not quite be a power of 8 -- our plan_power12_power6 function tried its best, but if there are very few factors of 3, it can't help. 
                // if we're 2 * 8^N, benchmarking shows that applying a 16 before our chain of 8s is very fast.
                plan.push_radix(16); 
                radix_factors = radix_factors.divide_by(&PartialFactors::compute(16)).unwrap();
            }
            plan.push_radix_power(12, power_twelve);
            plan.push_radix_power(11, radix_factors.get_power11());
            plan.push_radix_power(9, radix_factors.get_power3() / 2);
            plan.push_radix_power(8, radix_factors.get_power2() / 3);
            plan.push_radix_power(7, radix_factors.get_power7());
            plan.push_radix_power(6, power_six);
            plan.push_radix_power(5, radix_factors.get_power5());
            if radix_factors.get_power2() % 3 == 2 {
                // our factors of 2 might not quite be a power of 8 -- our plan_power12_power6 function tried its best, but if we are a power of 2, it can't help. 
                // if we're 4 * 8^N, benchmarking shows that applying a 4 to the end our chain of 8s is very fast.
                plan.push_radix(4);
            }
            if radix_factors.get_power3() % 2 == 1 {
                // our factors of 3 might not quite be a power of 9 -- our plan_power12_power6 function tried its best, but if we are a power of 3, it can't help. 
                // if we're 3 * 9^N, our only choice is to add an 8xn step
                plan.push_radix(3);
            }
            if radix_factors.get_power2() % 3 == 1 {
                // our factors of 2 might not quite be a power of 8. We tried to correct this with a 16 radix and 4 radix, but as a last resort, apply a 2. 2 is very slow, but it's better than not computing the FFT
                plan.push_radix(2);
            }

            // measurement opportunity: is it faster to let the plan_power12_power6 function put a 4 on the end instead of relying on all 8's?
            // measurement opportunity: is it faster to slap a 16 on top of the stack?
            // measurement opportunity: if our plan_power12_power6 function adds both 12s and sixes, is it faster to drop combinations of 12+6 down to 8+9?
        };
        plan
    }

    fn construct_plan(&mut self, plan: MixedRadixPlan) -> Arc<dyn Fft<T>> {
        dbg!(&plan);
        if plan.is_base_only() {
            self.construct_base(plan.base)
        } else {
            // first, see if we can find a cached FFT instance somewhere in the chain
            let (base_index, base_fft) = plan.radixes
                .iter()
                .enumerate()
                .scan(plan.base, |product, (index, radix)| { // compute a running product and pair it with each radix, so we can check the algorithm cache
                    *product = *product * *radix as usize;
                    self.algorithm_cache.get(product).map(|cached_instance| (index + 1, cached_instance))
                })
                .last()
                .map(|(base_index, cached_instance)| (base_index, Arc::clone(cached_instance)))
                .unwrap_or_else(|| (0, self.construct_base(plan.base)));

            self.construct_radix_chain(&plan.radixes[base_index..], base_fft)
        }
    }

    fn construct_base(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        if self.is_butterfly(len) {
            let butterfly = self.construct_butterfly(len);
            self.algorithm_cache.insert(len, Arc::clone(&butterfly));
            return butterfly;
        }

        // if we get to this point, we will have to use either rader's algorithm or bluestein's algorithm as our base
        // We can only use rader's if `len` is prime, so next step is to compute a full factorization
        if miller_rabin(len as u64) {
            // len is prime, so we can use Rader's Algorithm as a base. Whether or not that's a good idea is a different story
            // Rader's Algorithm is only faster in a few narrow cases. 
            // as a heuristic, only use rader's algorithm if its inner FFT can be computed entirely without bluestein's or rader's
            // We're intentionally being too conservative here. Otherwise we'd be recursively applying a heuristic, and repeated heuristic failures could stack to make a rader's chain significantly slower.
            // If we were writing a measuring planner, expanding this heuristic and measuring its effectiveness would be an opportunity for up to 2x performance gains.
            let inner_factors = PartialFactors::compute(len - 1);
            if inner_factors.get_other_factors() == 1 {
                // We only have factors of 2,3,5, and 7. If we don't have AVX2, we also have to exclude factors of 5 and 7, because avx2 gives us enough headroom for the overhead of those to not be a problem
                if is_x86_feature_detected!("avx2") || (inner_factors.get_power5() == 0 && inner_factors.get_power7() == 0) {
                    return self.plan_with_cache(len, Self::construct_raders);
                }
            }
        }

        // if we get to this point, we decided to use Bluestein's Algorithm as the base
        self.plan_with_cache(len, Self::construct_bluesteins)
    }
    
}

pub trait MakeFftAvx<T: FFTnum> {
    // todo: if we ever have a core plan type that ins't mixed radix, this could return an enum
    fn plan_mixed_radix_base(&self, len: usize, factors: &PartialFactors) -> MixedRadixPlan;

    fn is_butterfly(&self, len: usize) -> bool;

    fn construct_butterfly(&mut self, len: usize) -> Arc<dyn Fft<T>>;
    fn construct_radix_chain(&mut self, chain: &[u8], base: Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>>;
    fn construct_bluesteins(&mut self, len: usize) -> Arc<dyn Fft<T>>;
    fn construct_raders(&mut self, len: usize) -> Arc<dyn Fft<T>>;
}
impl<T: FFTnum> MakeFftAvx<T> for FftPlannerAvx<T> {
    default fn plan_mixed_radix_base(&self, _len: usize, _factors: &PartialFactors) -> MixedRadixPlan { unimplemented!(); }

    default fn is_butterfly(&self, _len: usize) -> bool { unimplemented!(); }

    default fn construct_butterfly(&mut self, _len: usize) -> Arc<dyn Fft<T>> { unimplemented!(); }
    default fn construct_radix_chain(&mut self, _chain: &[u8], _base: Arc<dyn Fft<T>>) -> Arc<dyn Fft<T>> { unimplemented!(); }
    default fn construct_bluesteins(&mut self, _len: usize) -> Arc<dyn Fft<T>> { unimplemented!(); }
    default fn construct_raders(&mut self, _len: usize) -> Arc<dyn Fft<T>> { unimplemented!(); }
}








impl MakeFftAvx<f32> for FftPlannerAvx<f32> {
    fn plan_mixed_radix_base(&self, len: usize, factors: &PartialFactors) -> MixedRadixPlan { 
        // if we have a factor that can't be computed with 2xn 3xn etc, we'll have to compute it with bluestein's or rader's, so use that as the base
        if factors.get_other_factors() > 1 {
            return MixedRadixPlan::new(factors.get_other_factors(), &[]);
        }

        // If this FFT size is a butterfly, use that
        if self.is_butterfly(len) {
            return MixedRadixPlan::new(len, &[]);
        }

        // If the power2 * power3 component of this FFT is a butterfly and not too small, return that
        let power2power3 = factors.product_power2power3();
        if power2power3 > 4 && self.is_butterfly(power2power3) {
            return MixedRadixPlan::new(power2power3, &[]);
        }

        // most of this code is heuristics assuming FFTs of a minimum size. if the FFT is below that minimum size, the heuristics break down.
        // so the first thing we're going to do is hardcode the plan for osme specific sizes where we know the heuristics won't be enough
        let hardcoded_base = match power2power3 {
            // 3 * 2^n special cases
            96 => Some(MixedRadixPlan::new(32, &[3])), // 2^5 * 3
            192 => Some(MixedRadixPlan::new(48, &[4])), // 2^6 * 3
            1536 => Some(MixedRadixPlan::new(48, &[8, 4])), // 2^8 * 3

            // 9 * 2^n special cases
            18 => Some(MixedRadixPlan::new(3, &[6])), // 2 * 3^2
            144 => Some(MixedRadixPlan::new(36, &[4])), // 2^4 * 3^2
            
            _=> None,
        };
        if let Some(hardcoded) = hardcoded_base {
            return hardcoded;
        }

        if factors.get_power2() >= 5 {
            match factors.get_power3() {
                // if this FFT is a power of 2, our strategy here is to tweak the butterfly to free us up to do an 8xn chain
                0 => match factors.get_power2() % 3 {
                    0 => MixedRadixPlan::new(512, &[]),
                    1 => MixedRadixPlan::new(256, &[]),
                    2 => MixedRadixPlan::new(256, &[]),
                    _ => unreachable!(),
                },
                // if this FFT is 3 times a power of 2, our strategy here is to tweak butterflies to make it easier to set up a 8xn chain
                1 => match factors.get_power2() % 3 {
                    0 => MixedRadixPlan::new(64, &[12, 16]),
                    1 => MixedRadixPlan::new(48, &[]),
                    2 => MixedRadixPlan::new(64, &[]),
                    _ => unreachable!(),
                },
                // if this FFT is 9 or greater times a power of 2, just use 72. As you might expect, in this vast field of options, what is optimal becomes a lot more muddy and situational
                // but across all the benchmarking i've done, 72 seems like the best default that will get us the best plan in 95% of the cases
                // 64, 54, and 48 are occasionally faster, although i haven't been able to discern a pattern.
                _ => MixedRadixPlan::new(72, &[]),
            }
        } else if factors.get_power3() >= 3 {
            // Our FFT is a power of 3 times a low power of 2. A high level summary of our strategy is that we want to pick a base that will 
            // A: consume all factors of 2, and B: leave us with an even power of 3, so that we can do a 9xn chain.
            match factors.get_power2() {
                0 => MixedRadixPlan::new(27, &[]),
                1 => MixedRadixPlan::new(54, &[]),
                2 => match factors.get_power3() % 2 {
                    0 => MixedRadixPlan::new(36, &[]),
                    1 => MixedRadixPlan::new(if len < 1000 { 36 } else { 12 }, &[]),
                    _ => unreachable!(),
                },
                3 => match factors.get_power3() % 2 {
                    0 => MixedRadixPlan::new(72, &[]),
                    1 => MixedRadixPlan::new(if factors.get_power3() > 7 { 24 } else { 72 }, &[]),
                    _ => unreachable!(),
                },
                4 => match factors.get_power3() % 2 {
                    0 => MixedRadixPlan::new(if factors.get_power3() > 6 { 16 } else { 72 }, &[]),
                    1 => MixedRadixPlan::new(if factors.get_power3() > 9 { 48 } else { 72 }, &[]),
                    _ => unreachable!(),
                },
                // if this FFT is 32 or greater times a power of 3, just use 72. As you might expect, in this vast field of options, what is optimal becomes a lot more muddy and situational
                // but across all the benchmarking i've done, 72 seems like the best default that will get us the best plan in 95% of the cases
                // 64, 54, and 48 are occasionally faster, although i haven't been able to discern a pattern.
                _ => MixedRadixPlan::new(72, &[]),
            }
        }
        // If this FFT has powers of 11, 7, or 5, use that
        else if factors.get_power11() > 0 {
            MixedRadixPlan::new(11, &[])
        }
        else if factors.get_power7() > 0 {
            MixedRadixPlan::new(7, &[])
        }
        else if factors.get_power5() > 0 {
            MixedRadixPlan::new(5, &[])
        } else {
            panic!("Couldn't find a base for FFT size {}, factors={:?}", len, factors)
        }
    }

    fn is_butterfly(&self, len: usize) -> bool {
        [0,1,2,3,4,5,6,7,8,9,11,12,13,16,17,19,23,24,27,29,31,32,36,48,54,64,72,128,256,512].contains(&len)
    }

    fn construct_butterfly(&mut self, len: usize) -> Arc<dyn Fft<f32>> {
        match len {
            0|1 =>  wrap_fft(DFT::new(len, self.inverse)),
            2 =>    wrap_fft(Butterfly2::new(self.inverse)),
            3 =>    wrap_fft(Butterfly3::new(self.inverse)),
            4 =>    wrap_fft(Butterfly4::new(self.inverse)),
            5 =>    wrap_fft(Butterfly5Avx::new(self.inverse).unwrap()),
            6 =>    wrap_fft(Butterfly6::new(self.inverse)),
            7 =>    wrap_fft(Butterfly7Avx::new(self.inverse).unwrap()),
            8 =>    wrap_fft(Butterfly8Avx::new(self.inverse).unwrap()),
            9 =>    wrap_fft(Butterfly9Avx::new(self.inverse).unwrap()),
            11 =>   wrap_fft(Butterfly11Avx::new(self.inverse).unwrap()),
            12 =>   wrap_fft(Butterfly12Avx::new(self.inverse).unwrap()),
            13 =>   wrap_fft(Butterfly13::new(self.inverse)),
            16 =>   wrap_fft(Butterfly16Avx::new(self.inverse).unwrap()),
            17 =>   wrap_fft(Butterfly17::new(self.inverse)),
            19 =>   wrap_fft(Butterfly19::new(self.inverse)),
            23 =>   wrap_fft(Butterfly23::new(self.inverse)),
            24 =>   wrap_fft(Butterfly24Avx::new(self.inverse).unwrap()),
            27 =>   wrap_fft(Butterfly27Avx::new(self.inverse).unwrap()),
            29 =>   wrap_fft(Butterfly29::new(self.inverse)),
            31 =>   wrap_fft(Butterfly31::new(self.inverse)),
            32 =>   wrap_fft(Butterfly32Avx::new(self.inverse).unwrap()),
            36 =>   wrap_fft(Butterfly36Avx::new(self.inverse).unwrap()),
            48 =>   wrap_fft(Butterfly48Avx::new(self.inverse).unwrap()),
            54 =>   wrap_fft(Butterfly54Avx::new(self.inverse).unwrap()),
            64 =>   wrap_fft(Butterfly64Avx::new(self.inverse).unwrap()),
            72 =>   wrap_fft(Butterfly72Avx::new(self.inverse).unwrap()),
            128 =>  wrap_fft(Butterfly128Avx::new(self.inverse).unwrap()),
            256 =>  wrap_fft(Butterfly256Avx::new(self.inverse).unwrap()),
            512 =>  wrap_fft(Butterfly512Avx::new(self.inverse).unwrap()),
            _ => panic!("Invalid butterfly len: {}", len)
        }
    }
    fn construct_radix_chain(&mut self, chain: &[u8], base: Arc<dyn Fft<f32>>) -> Arc<dyn Fft<f32>> { 
        let mut fft = base;
        for radix in chain.iter() {
            fft = match radix {
                2  => wrap_fft(MixedRadix2xnAvx::new(fft).unwrap()),
                3  => wrap_fft(MixedRadix3xnAvx::new(fft).unwrap()),
                4  => wrap_fft(MixedRadix4xnAvx::new(fft).unwrap()),
                5  => wrap_fft(MixedRadix5xnAvx::new(fft).unwrap()),
                6  => wrap_fft(MixedRadix6xnAvx::new(fft).unwrap()),
                7  => wrap_fft(MixedRadix7xnAvx::new(fft).unwrap()),
                8  => wrap_fft(MixedRadix8xnAvx::new(fft).unwrap()),
                9  => wrap_fft(MixedRadix9xnAvx::new(fft).unwrap()),
                11 => wrap_fft(MixedRadix11xnAvx::new(fft).unwrap()),
                12 => wrap_fft(MixedRadix12xnAvx::new(fft).unwrap()),
                16 => wrap_fft(MixedRadix16xnAvx::new(fft).unwrap()),
                _ => unreachable!(),
            };

            // cache this algorithm for next time
            self.algorithm_cache.insert(fft.len(), Arc::clone(&fft));
        }
        
        fft
    }
    fn construct_bluesteins(&mut self, len: usize) -> Arc<dyn Fft<f32>> { 
        assert!(len > 1); // Internal consistency check: The logic in this method doesn't work for a length of 1

        // Bluestein's computes a FFT of size `len` by reorganizing it as a FFT of ANY size greater than or equal to len * 2 - 1
        // an obvious choice is the next power of two larger than  len * 2 - 1, but if we can find a smaller FFT that will go faster, we can save a lot of time.
        // We can very efficiently compute almost any 2^n * 3^m, so we're going to search for all numbers of the form 2^n * 3^m that lie between len * 2 - 1 and the next power of two.
        let min_len = len*2 - 1;
        let baseline_candidate = min_len.checked_next_power_of_two().unwrap();

        // our algorithm here is to start with our next power of 2, and repeatedly divide by 2 and multiply by 3, trying to keep our value in range
        let mut bluesteins_candidates = Vec::new();
        let mut candidate = baseline_candidate;
        let mut factor2 = candidate.trailing_zeros();
        let mut factor3 = 0;

        let min_factor2 = 3; // benchmarking shows that while 3^n, 2 * 3^n, and 2^2 * 3^n are fast, they're typically slower than the next-higher candidate, so don't bother generating them
        while factor2 >= min_factor2 {
            // if this candidate length isn't too small, add it to our candidates list
            if candidate >= min_len {
                bluesteins_candidates.push((candidate, factor2, factor3));
            }
            // if the candidate is too large, divide it by 2. if it's too small, divide it by 3
            if candidate >= baseline_candidate {
                candidate >>= 1;
                factor2 -= 1;
            } else {
                candidate *= 3;
                factor3 += 1;
            }
        }
        bluesteins_candidates.sort();

        // we now have a list of candidates to choosse from. some 2^n * 3^m FFTs are faster than others, so apply a filter, which will let us skip sizes that benchmarking shas shown to be slow
        let inner_len = bluesteins_candidates.iter().find_map(|(len, factor2, factor3)| {
            if *factor2 > 16 && *factor3 < 3 { 
                // surprisingly, pure powers of 2 have a pretty steep dropoff in speed after 65536. 
                // the algorithm is designed to generate candidadtes larget than baseline_candidate, so if we hit a large power of 2, there should be more after it that we can skip to
                None
            } else {
                Some(*len)
            }
        }).unwrap_or_else(|| panic!("Failed to find a bluestein's candidate for len={}, candidates: {:?}", len, bluesteins_candidates));

        let inner_fft = self.plan_fft(inner_len);
        wrap_fft(BluesteinsAvx::new(len, inner_fft).unwrap())
    }

    fn construct_raders(&mut self, len: usize) -> Arc<dyn Fft<f32>> {
        let inner_fft = self.plan_fft(len - 1);
        
        // try to construct our AVX2 rader's algorithm. If that fails (probably because the machine we're running on doesn't have AVX2), fall back to scalar
        if let Ok(raders_avx) = RadersAvx2::new(Arc::clone(&inner_fft)) {
            wrap_fft(raders_avx)
        } else {
            wrap_fft(RadersAlgorithm::new(inner_fft))
        }
    }
}



impl MakeFftAvx<f64> for FftPlannerAvx<f64> {
    fn plan_mixed_radix_base(&self, len: usize, factors: &PartialFactors) -> MixedRadixPlan {
        // if we have a factor that can't be computed with 2xn 3xn etc, we'll have to compute it with bluestein's or rader's, so use that as the base
        if factors.get_other_factors() > 1 {
            return MixedRadixPlan::new(factors.get_other_factors(), &[]);
        }

        // If this FFT size is a butterfly, use that
        if self.is_butterfly(len) {
            return MixedRadixPlan::new(len, &[]);
        }

        // If the power2 * power3 component of this FFT is a butterfly and not too small, return that
        let power2power3 = factors.product_power2power3();
        if power2power3 > 4 && self.is_butterfly(power2power3) {
            return MixedRadixPlan::new(power2power3, &[]);
        }

        // most of this code is heuristics assuming FFTs of a minimum size. if the FFT is below that minimum size, the heuristics break down.
        // so the first thing we're going to do is hardcode the plan for osme specific sizes where we know the heuristics won't be enough
        let hardcoded_base = match power2power3 {
            // 2^n special cases
            64 => Some(MixedRadixPlan::new(16, &[4])), // 2^6

            // 3 * 2^n special cases
            48 => Some(MixedRadixPlan::new(12, &[4])), // 3 * 2^4
            96 => Some(MixedRadixPlan::new(12, &[8])), // 3 * 2^5
            768 => Some(MixedRadixPlan::new(12, &[8, 8])), // 3 * 2^8

            // 9 * 2^n special cases
            72 => Some(MixedRadixPlan::new(24, &[3])), // 2^3 * 3^2
            288 => Some(MixedRadixPlan::new(32, &[9])), // 2^5 * 3^2

            // 4 * 3^n special cases
            108 => Some(MixedRadixPlan::new(18, &[6])), // 2^4 * 3^2
            _=> None,
        };
        if let Some(hardcoded) = hardcoded_base {
            return hardcoded;
        }

        if factors.get_power2() >= 4 {
            match factors.get_power3() {
                // if this FFT is a power of 2, our strategy here is to tweak the butterfly to free us up to do an 8xn chain
                0 => match factors.get_power2() % 3 {
                    0 => MixedRadixPlan::new(512, &[]),
                    1 => MixedRadixPlan::new(128, &[]),
                    2 => MixedRadixPlan::new(256, &[]),
                    _ => unreachable!(),
                },
                // if this FFT is 3 times a power of 2, our strategy here is to tweak butterflies to make it easier to set up a 8xn chain
                1 => match factors.get_power2() % 3 {
                    0 => MixedRadixPlan::new(24, &[]),
                    1 => MixedRadixPlan::new(32, &[12]),
                    2 => MixedRadixPlan::new(32, &[12, 16]),
                    _ => unreachable!(),
                },
                // if this FFT is 9 times a power of 2, our strategy here is to tweak butterflies to make it easier to set up a 8xn chain
                2 => match factors.get_power2() % 3 {
                    0 => MixedRadixPlan::new(36, &[16]),
                    1 => MixedRadixPlan::new(36, &[]),
                    2 => MixedRadixPlan::new(18, &[]),
                    _ => unreachable!(),
                },
                // this FFT is 27 or greater times a power of two. As you might expect, in this vast field of options, what is optimal becomes a lot more muddy and situational
                // but across all the benchmarking i've done, 36 seems like the best default that will get us the best plan in 95% of the cases
                // 32 is rarely faster, although i haven't been able to discern a pattern.
                _ => MixedRadixPlan::new(36, &[])
            }
        } else if factors.get_power3() >= 3 {
            // Our FFT is a power of 3 times a low power of 2
            match factors.get_power2() {
                0 => match factors.get_power3() % 2 {
                    0 => MixedRadixPlan::new(if factors.get_power3() > 10 { 9 } else { 27 }, &[]),
                    1 => MixedRadixPlan::new(27, &[]),
                    _ => unreachable!(),
                },
                1 => MixedRadixPlan::new(18, &[]),
                2 => match factors.get_power3() % 2 {
                    0 => MixedRadixPlan::new(36, &[]),
                    1 => MixedRadixPlan::new(if factors.get_power3() > 10 { 36 } else { 18 }, &[]),
                    _ => unreachable!(),
                },
                3 => MixedRadixPlan::new(18, &[]),
                // this FFT is 16 or greater times a power of three. As you might expect, in this vast field of options, what is optimal becomes a lot more muddy and situational
                // but across all the benchmarking i've done, 36 seems like the best default that will get us the best plan in 95% of the cases
                // 32 is rarely faster, although i haven't been able to discern a pattern.
                _ => MixedRadixPlan::new(36, &[])
            }
        } 
         // If this FFT has powers of 11, 7, or 5, use that
         else if factors.get_power11() > 0 {
            MixedRadixPlan::new(11, &[])
        }
        else if factors.get_power7() > 0 {
            MixedRadixPlan::new(7, &[])
        }
        else if factors.get_power5() > 0 {
            MixedRadixPlan::new(5, &[])
        } else {
            panic!("Couldn't find a base for FFT size {}, factors={:?}", len, factors)
        }
    }

    fn is_butterfly(&self, len: usize) -> bool {
        [0,1,2,3,4,5,6,7,8,9,11,12,13,16,17,18,19,23,24,27,29,31,32,36,64,128,256,512].contains(&len)
    }

    fn construct_butterfly(&mut self, len: usize) -> Arc<dyn Fft<f64>> {
        match len {
            0|1 =>  wrap_fft(DFT::new(len, self.inverse)),
            2 =>    wrap_fft(Butterfly2::new(self.inverse)),
            3 =>    wrap_fft(Butterfly3::new(self.inverse)),
            4 =>    wrap_fft(Butterfly4::new(self.inverse)),
            5 =>    wrap_fft(Butterfly5Avx64::new(self.inverse).unwrap()),
            6 =>    wrap_fft(Butterfly6::new(self.inverse)),
            7 =>    wrap_fft(Butterfly7Avx64::new(self.inverse).unwrap()),
            8 =>    wrap_fft(Butterfly8Avx64::new(self.inverse).unwrap()),
            9 =>    wrap_fft(Butterfly9Avx64::new(self.inverse).unwrap()),
            11 =>   wrap_fft(Butterfly11Avx64::new(self.inverse).unwrap()),
            12 =>   wrap_fft(Butterfly12Avx64::new(self.inverse).unwrap()),
            13 =>   wrap_fft(Butterfly13::new(self.inverse)),
            16 =>   wrap_fft(Butterfly16Avx64::new(self.inverse).unwrap()),
            17 =>   wrap_fft(Butterfly17::new(self.inverse)),
            18 =>   wrap_fft(Butterfly18Avx64::new(self.inverse).unwrap()),
            19 =>   wrap_fft(Butterfly19::new(self.inverse)),
            23 =>   wrap_fft(Butterfly23::new(self.inverse)),
            24 =>   wrap_fft(Butterfly24Avx64::new(self.inverse).unwrap()),
            27 =>   wrap_fft(Butterfly27Avx64::new(self.inverse).unwrap()),
            29 =>   wrap_fft(Butterfly29::new(self.inverse)),
            31 =>   wrap_fft(Butterfly31::new(self.inverse)),
            32 =>   wrap_fft(Butterfly32Avx64::new(self.inverse).unwrap()),
            36 =>   wrap_fft(Butterfly36Avx64::new(self.inverse).unwrap()),
            64 =>   wrap_fft(Butterfly64Avx64::new(self.inverse).unwrap()),
            128 =>   wrap_fft(Butterfly128Avx64::new(self.inverse).unwrap()),
            256 =>   wrap_fft(Butterfly256Avx64::new(self.inverse).unwrap()),
            512 =>   wrap_fft(Butterfly512Avx64::new(self.inverse).unwrap()),
            _ => panic!("Invalid butterfly len: {}", len)
        }
    }
    fn construct_radix_chain(&mut self, chain: &[u8], base: Arc<dyn Fft<f64>>) -> Arc<dyn Fft<f64>> { 
        let mut fft = base;
        for radix in chain.iter() {
            fft = match radix {
                2  => wrap_fft(MixedRadix2xnAvx::new(fft).unwrap()),
                3  => wrap_fft(MixedRadix3xnAvx::new(fft).unwrap()),
                4  => wrap_fft(MixedRadix4xnAvx::new(fft).unwrap()),
                5  => wrap_fft(MixedRadix5xnAvx::new(fft).unwrap()),
                6  => wrap_fft(MixedRadix6xnAvx::new(fft).unwrap()),
                7  => wrap_fft(MixedRadix7xnAvx::new(fft).unwrap()),
                8  => wrap_fft(MixedRadix8xnAvx::new(fft).unwrap()),
                9  => wrap_fft(MixedRadix9xnAvx::new(fft).unwrap()),
                11 => wrap_fft(MixedRadix11xnAvx::new(fft).unwrap()),
                12 => wrap_fft(MixedRadix12xnAvx::new(fft).unwrap()),
                16 => wrap_fft(MixedRadix16xnAvx::new(fft).unwrap()),
                _ => unreachable!(),
            };

            // cache this algorithm for next time
            self.algorithm_cache.insert(fft.len(), Arc::clone(&fft));
        }
        
        fft
    }
    fn construct_bluesteins(&mut self, len: usize) -> Arc<dyn Fft<f64>> { 
        assert!(len > 1); // Internal consistency check: The logic in this method doesn't work for a length of 1

        // Bluestein's computes a FFT of size `len` by reorganizing it as a FFT of ANY size greater than or equal to len * 2 - 1
        // an obvious choice is the next power of two larger than  len * 2 - 1, but if we can find a smaller FFT that will go faster, we can save a lot of time.
        // We can very efficiently compute almost any 2^n * 3^m, so we're going to search for all numbers of the form 2^n * 3^m that lie between len * 2 - 1 and the next power of two.
        let min_len = len*2 - 1;
        let baseline_candidate = min_len.checked_next_power_of_two().unwrap();

        // our algorithm here is to start with our next power of 2, and repeatedly divide by 2 and multiply by 3, trying to keep our value in range
        let mut bluesteins_candidates = Vec::new();
        let mut candidate = baseline_candidate;
        let mut factor2 = candidate.trailing_zeros();
        let mut factor3 = 0;

        let min_factor2 = 2; // benchmarking shows that while 3^n and  2 * 3^n are fast, they're typically slower than the next-higher candidate, so don't bother generating them
        while factor2 >= min_factor2 {
            // if this candidate length isn't too small, add it to our candidates list
            if candidate >= min_len {
                bluesteins_candidates.push((candidate, factor2, factor3));
            }
            // if the candidate is too large, divide it by 2. if it's too small, divide it by 3
            if candidate >= baseline_candidate {
                candidate >>= 1;
                factor2 -= 1;
            } else {
                candidate *= 3;
                factor3 += 1;
            }
        }
        bluesteins_candidates.sort();

        // we now have a list of candidates to choosse from. some 2^n * 3^m FFTs are faster than others, so apply a filter, which will let us skip sizes that benchmarking has shown to be slow
        let inner_len = bluesteins_candidates.iter().find_map(|(len, factor2, factor3)| {
            if *factor3 < 1 && *factor2 > 13 { return None; }
            if *factor3 < 4 && *factor2 > 14 { return None; }
            Some(*len)
        }).unwrap_or_else(|| panic!("Failed to find a bluestein's candidate for len={}, candidates: {:?}", len, bluesteins_candidates));

        let inner_fft = self.plan_fft(inner_len);
        wrap_fft(BluesteinsAvx::new(len, inner_fft).unwrap())
    }
    fn construct_raders(&mut self, len: usize) -> Arc<dyn Fft<f64>> {
        let inner_fft = self.plan_fft(len - 1);
        
        // try to construct our AVX2 rader's algorithm. If that fails (probably because the machine we're running on doesn't have AVX2), fall back to scalar
        if let Ok(raders_avx) = RadersAvx2::new(Arc::clone(&inner_fft)) {
            wrap_fft(raders_avx)
        } else {
            wrap_fft(RadersAlgorithm::new(inner_fft))
        }
    }
}
