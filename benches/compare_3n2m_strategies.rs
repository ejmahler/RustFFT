#![feature(test)]
#![allow(non_snake_case)]
#![allow(unused)]

extern crate test;
extern crate rustfft;
use test::Bencher;

use rustfft::algorithm::avx::*;
use rustfft::algorithm::butterflies::*;
use rustfft::algorithm::DFT;
use rustfft::{Fft, FFTnum};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;
use rustfft::algorithm::avx::avx_planner::*;

use std::sync::Arc;

/// This benchmark's purpose is to build some programmer intuition for planner heuristics
/// We have mixed radix 2xn, 3xn, 4xn, 6xn, 8xn, 9x, 12xn, and 16xn implementations -- for a given FFT of the form 2^xn * 3^m, which combination is the fastest? Is 12xn -> 4xn faster than 6xn -> 8xn?
/// Is it faster to put 9xn as an outer FFT of 8xn or as an inner FFT? this file autogenerates benchmarks that answer these questions
/// 
/// The "generate_3n2m_comparison_benchmarks" benchmark will print benchmark code to the console which should be pasted back into this file, basically a low-budget procedural macro

#[derive(Clone, Debug)]
struct FftSize {
    len: usize,
    power2: u32,
    power3: u32,
}
impl FftSize {
    fn new(len: usize) -> Self {
        let power2 = len.trailing_zeros();
        let mut remaining_factors = len >> power2;
        let mut power3 = 0;
        while remaining_factors % 3 == 0 {
            power3 += 1;
            remaining_factors /= 3;
        }

        assert!(remaining_factors == 1);
        Self { power2, power3, len }
    }

    fn divide(&self, other: &Self) -> Option<Self> {
        if self.power2 <= other.power2 && self.power3 <= other.power3 {
            Some(Self { power2: other.power2 - self.power2, power3: other.power3 - self.power3, len: other.len / self.len})
        } else {
            None
        }
    }
}

// We don't need to generate a combinatoric explosion of tests that we know will be slow. filter_radix applies some dumb heuristics to filter out the most common slow cases
fn filter_radix(current_strategy: &[usize], potential_radix: &FftSize, is_butterfly: bool) -> bool {
    // if we've seen any radix larger than this before, reject. otherwise we'll get a million reorderings of the same radixex, with benchmarking showing that smaller being higher is typically faster
    if !is_butterfly && current_strategy.iter().find(|i| **i > potential_radix.len && **i != 16).is_some() {
        return false;
    }

    // apply filters to size 2
    if potential_radix.len == 2 {
        // if our strategy already contains any 2's, 3's, or 4's, reject -- because 4, 6, or 8 will be faster, respectively
        return !current_strategy.contains(&2) && !current_strategy.contains(&3) && !current_strategy.contains(&4);
    }
    // apply filters to size 3
    if potential_radix.len == 3 {
        // if our strategy already contains any 2's, 3's or 4s, reject -- because 6 and 9 and 12 will be faster, respectively
        return !current_strategy.contains(&2) && !current_strategy.contains(&3) && !current_strategy.contains(&4);
    }
    // apply filters to size 4
    if potential_radix.len == 4 {
        // if our strategy already contains any 2's, reject -- because 8 will be faster
        // if our strategy already contains 2 4's, don't add a third, because 2 8's would have been faster
        // if our strategy already contains a 16, reject -- because 2 8's will be faster (8s are seriously fast guys)
        return !current_strategy.contains(&2) && !current_strategy.contains(&3) && !current_strategy.contains(&4) && !current_strategy.contains(&16);
    }
    if potential_radix.len == 16 {
        // if our strategy already contains a 4, reject -- because 2 8's will be faster (8s are seriously fast guys)
        // if our strategy already contains a 16, reject -- benchmarking shows that 16s are very situational, and repeating them never helps)
        return !current_strategy.contains(&4) && !current_strategy.contains(&16);
    }
    return true;
}

fn recursive_strategy_builder(strategy_list: &mut Vec<Vec<usize>>, last_ditch_strategy_list: &mut Vec<Vec<usize>>, mut current_strategy: Vec<usize>, len: FftSize, butterfly_sizes: &[usize], last_ditch_butterflies: &[usize], available_radixes: &[FftSize]) {
    if butterfly_sizes.contains(&len.len) {
        if filter_radix(&current_strategy, &len, true) {
            current_strategy.push(len.len);

            //If this strategy contains a 2 or 3, it's very unlikely to be the fastest. we don't want to rule it out, because it's required sometimes, but don't use it unless there aren't any other
            if current_strategy.contains(&2) || current_strategy.contains(&3) {
                strategy_list.push(current_strategy.clone());
            } else {
                strategy_list.push(current_strategy.clone());
            }
        }
    }
    else if last_ditch_butterflies.contains(&len.len) {
        if filter_radix(&current_strategy, &len, true) {
            current_strategy.push(len.len);
            last_ditch_strategy_list.push(current_strategy.clone());
        }
    }
    else if len.len > 1 {
        for radix in available_radixes {
            if filter_radix(&current_strategy, radix, false) {
                if let Some(inner) = radix.divide(&len) {
                    let mut cloned_strategy = current_strategy.clone();
                    cloned_strategy.push(radix.len);
                    recursive_strategy_builder(strategy_list, last_ditch_strategy_list, cloned_strategy, inner, butterfly_sizes, last_ditch_butterflies, available_radixes);
                }
            }
        }
    }
}

// it's faster to filter strategies at the radix level since we can prune entire permutations, but some can only be done once the full plan is built
fn filter_strategy(strategy: &Vec<usize>) -> bool {
    if strategy.contains(&16) {
        let index = strategy.iter().position(|s| *s == 16).unwrap();
        index == 0 || index == strategy.len() - 1 || index == strategy.len() - 2 || (strategy[index - 1] < 12 && strategy[index + 1] >= 12)
    } else {
        true
    }
}

// cargo bench generate_3n2m_comparison_benchmarks_32 -- --nocapture --ignored
#[ignore]
#[bench]
fn generate_3n2m_comparison_benchmarks_32(_: &mut test::Bencher) {
    let butterfly_sizes = [ 72, 36, 48, 54, 64 ]; 
    let last_ditch_butterflies = [ 27, 9 ]; 
    let available_radixes = [FftSize::new(3), FftSize::new(4), FftSize::new(6), FftSize::new(8), FftSize::new(9), FftSize::new(12), FftSize::new(16)];

    let max_len : usize = 1 << 18;
    let min_len = 64;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;
    
    for power3 in 2..10 {
        for power2 in 3..4 {
            let len = 3usize.pow(power3) << power2;
            if len > max_len { continue; }

            //let planned_fft : Arc<dyn Fft<f32>> = rustfft::FFTplanner::new(false).plan_fft(len);

            // we want to catalog all the different possible ways there are to compute a FFT of size `len`
            // we can do that by recursively looping over each radix, dividing our length by that radix, then recursively trying rach radix again
            let mut strategies = vec![];
            let mut last_ditch_strategies = vec![];
            recursive_strategy_builder(&mut strategies, &mut last_ditch_strategies, Vec::new(), FftSize::new(len), &butterfly_sizes, &last_ditch_butterflies, &available_radixes);

            if strategies.len() == 0 {
                strategies = last_ditch_strategies;
            }

            for mut s in strategies.into_iter().filter(filter_strategy) {
                s.reverse();
                let strategy_strings : Vec<_> = s.into_iter().map(|i| i.to_string()).collect();
                let test_id = strategy_strings.join("_");
                let strategy_array = strategy_strings.join(",");
                println!("#[bench] fn comparef32__2power{:02}__3power{:02}__len{:08}__{}(b: &mut Bencher) {{ compare_fft_f32(b, &[{}]); }}", power2, power3, len, test_id, strategy_array);
            }
        }  
    }
}

// cargo bench generate_3n2m_comparison_benchmarks_64 -- --nocapture --ignored
#[ignore]
#[bench]
fn generate_3n2m_comparison_benchmarks_64(_: &mut test::Bencher) {
    let butterfly_sizes = [ 36, 32, 27, 24, 18, 16, 12 ]; 
    let last_ditch_butterflies = [ 8, 9 ]; 
    let available_radixes = [FftSize::new(3), FftSize::new(4), FftSize::new(6), FftSize::new(8), FftSize::new(9), FftSize::new(12), FftSize::new(16)];

    let max_len : usize = 1 << 20;
    let min_len = 64;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;

    for power3 in 1..2 {
        for power2 in 3..max_power2 {
            let len = 3usize.pow(power3) << power2;
            if len > max_len { continue; }

            let planned_fft : Arc<dyn Fft<f32>> = rustfft::FFTplanner::new(false).plan_fft(len);

            let butterfly_sizes : &[usize] = &butterfly_sizes;

            // we want to catalog all the different possible ways there are to compute a FFT of size `len`
            // we can do that by recursively looping over each radix, dividing our length by that radix, then recursively trying rach radix again
            let mut strategies = vec![];
            let mut last_ditch_strategies = vec![];
            recursive_strategy_builder(&mut strategies, &mut last_ditch_strategies, Vec::new(), FftSize::new(len), &butterfly_sizes, &last_ditch_butterflies, &available_radixes);

            if strategies.len() == 0 {
                strategies = last_ditch_strategies;
            }

            for mut s in strategies.into_iter().filter(filter_strategy) {
                s.reverse();
                let strategy_strings : Vec<_> = s.into_iter().map(|i| i.to_string()).collect();
                let test_id = strategy_strings.join("_");
                let strategy_array = strategy_strings.join(",");
                println!("#[bench] fn comparef64__2power{:02}__3power{:02}__len{:08}__{}(b: &mut Bencher) {{ compare_fft_f64(b, &[{}]); }}", power2, power3, len, test_id, strategy_array);
            }
        }  
    }
}

// cargo bench generate_3n2m_planned_benchmarks_32 -- --nocapture --ignored
#[ignore]
#[bench]
fn generate_3n2m_planned_benchmarks(_: &mut test::Bencher) {
    let mut fft_sizes = vec![];

    let max_len : usize = 1 << 23;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;
    for power2 in 0..max_power2 {
        for power3 in 0..max_power3 {
            let len = 3usize.pow(power3) << power2;
            if len > max_len { continue; }
            if power3 < 2 && power2 > 16 { continue; }
            if power3 < 3 && power2 > 17 { continue; }
            if power2 < 1 { continue; }
            fft_sizes.push(len);
        }
    }

    for len in fft_sizes {
        let power2 = len.trailing_zeros();
        let mut remaining_factors = len >> power2;
        let mut power3 = 0;
        while remaining_factors % 3 == 0 {
            power3 += 1;
            remaining_factors /= 3;
        }

        println!("#[bench] fn comparef32_len{:07}_2power{:02}_3power{:02}(b: &mut Bencher) {{ bench_planned_fft_f32(b, {}); }}",len, power2, power3, len);
    }
}

// cargo bench generate_3n2m_planned_benchmarks_64 -- --nocapture --ignored
#[ignore]
#[bench]
fn generate_3n2m_planned_benchmarks_64(_: &mut test::Bencher) {
    let mut fft_sizes = vec![];

    let max_len : usize = 1 << 23;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;
    for power2 in 0..max_power2 {
        for power3 in 0..max_power3 {
            let len = 3usize.pow(power3) << power2;
            if len > max_len { continue; }
            if power3 < 1 && power2 > 13 { continue; }
            if power3 < 4 && power2 > 14 { continue; }
            if power2 < 2 { continue; }
            fft_sizes.push(len);
        }
    }

    for len in fft_sizes {
        let power2 = len.trailing_zeros();
        let mut remaining_factors = len >> power2;
        let mut power3 = 0;
        while remaining_factors % 3 == 0 {
            power3 += 1;
            remaining_factors /= 3;
        }

        println!("#[bench] fn comparef64_len{:07}_2power{:02}_3power{:02}(b: &mut Bencher) {{ bench_planned_fft_f64(b, {}); }}",len, power2, power3, len);
    }
}



#[derive(Copy, Clone, Debug)]
pub struct PartialFactors {
    power2: u32,
    power3: u32,
    other_factors: usize,
}
impl PartialFactors {
    pub fn compute(len: usize) -> Self {
        let power2 = len.trailing_zeros();
        let mut other_factors = len >> power2;
        let mut power3 = 0;
        while other_factors % 3 == 0 {
            power3 += 1;
            other_factors /= 3;
        }

        Self { power2, power3, other_factors }
    }

    pub fn get_power2(&self) -> u32 {
        self.power2
    }
    pub fn get_power3(&self) -> u32 {
        self.power3
    }
    pub fn get_other_factors(&self) -> usize {
        self.other_factors
    }
    #[allow(unused)]
    pub fn product(&self) -> usize {
        (self.other_factors * 3usize.pow(self.power3)) << self.power2
    }
    #[allow(unused)]
    pub fn divide_by(&self, divisor: &PartialFactors) -> Option<PartialFactors> {
        let two_divides = self.power2 >= divisor.power2;
        let three_divides = self.power3 >= divisor.power3;
        let other_divides = self.other_factors % divisor.other_factors == 0;
        if two_divides && three_divides && other_divides {
            Some(Self { 
                power2: self.power2 - divisor.power2,
                power3: self.power3 - divisor.power3,
                other_factors: if self.other_factors == divisor.other_factors { 1 } else { self.other_factors / divisor.other_factors }
            })
        }
        else {
            None
        }
    }
}

// cargo bench generate_raders_benchmarks -- --nocapture --ignored
#[ignore]
#[bench]
fn generate_raders_benchmarks(_: &mut test::Bencher) {
    // simple sieve of eratosthones to get all primes below N
    let max_prime = 1000000;
    let primes = {
        let mut primes : Vec<_> = (2..max_prime).collect();
        let mut index = 0;
        while index < primes.len() {
            let value = primes[index];
            primes.retain(|e| *e == value || e % value > 0);
            index += 1;
        }
        primes
    };

    for len in primes {
        let factors = PartialFactors::compute(len - 1);
        if len > 10 && factors.get_other_factors() == 1 {
            println!("#[bench] fn comparef32_len{:07}_bluesteins(b: &mut Bencher) {{ bench_planned_bluesteins_f32(b, {}); }}", len*2, len*2);
            println!("#[bench] fn comparef32_len{:07}_2xn_bluesteins(b: &mut Bencher) {{ bench_2xn_bluesteins_f32(b, {}); }}", len*2, len*2);
            println!("#[bench] fn comparef32_len{:07}_2xn_raders(b: &mut Bencher) {{ bench_planned_raders_f32(b, {}); }}", len*2, len*2);
        }
    }
}

fn wrap_fft<T: FFTnum>(fft: impl Fft<T> + 'static) -> Arc<dyn Fft<T>> {
    Arc::new(fft) as Arc<dyn Fft<T>>
}

fn compare_fft_f32(b: &mut Bencher, strategy: &[usize]) {
    let mut fft = match strategy[0] {
        1 =>    wrap_fft(DFT::new(1, false)),
        2 =>    wrap_fft(Butterfly2::new(false)),
        3 =>    wrap_fft(Butterfly3::new(false)),
        4 =>    wrap_fft(Butterfly4::new(false)),
        5 =>    wrap_fft(Butterfly5::new(false)),
        6 =>    wrap_fft(Butterfly6::new(false)),
        7 =>    wrap_fft(Butterfly7::new(false)),
        8 =>    wrap_fft(MixedRadixAvx4x2::new(false).unwrap()),
        9 =>    wrap_fft(MixedRadixAvx3x3::new(false).unwrap()),
        12 =>   wrap_fft(MixedRadixAvx4x3::new(false).unwrap()),
        16 =>   wrap_fft(MixedRadixAvx4x4::new(false).unwrap()),
        24 =>   wrap_fft(MixedRadixAvx4x6::new(false).unwrap()),
        27 =>   wrap_fft(MixedRadixAvx3x9::new(false).unwrap()),
        32 =>   wrap_fft(MixedRadixAvx4x8::new(false).unwrap()),
        36 =>   wrap_fft(MixedRadixAvx4x9::new(false).unwrap()),
        48 =>   wrap_fft(MixedRadixAvx4x12::new(false).unwrap()),
        54 =>   wrap_fft(MixedRadixAvx6x9::new(false).unwrap()),
        64 =>   wrap_fft(MixedRadixAvx8x8::new(false).unwrap()),
        72 =>   wrap_fft(MixedRadixAvx6x12::new(false).unwrap()),
        _ => panic!()
    };

    for radix in strategy.iter().skip(1) {
        fft = match radix {
            2 => wrap_fft(MixedRadix2xnAvx::new(fft).unwrap()),
            3 => wrap_fft(MixedRadix3xnAvx::new(fft).unwrap()),
            4 => wrap_fft(MixedRadix4xnAvx::new(fft).unwrap()),
            6 => wrap_fft(MixedRadix6xnAvx::new(fft).unwrap()),
            8 => wrap_fft(MixedRadix8xnAvx::new(fft).unwrap()),
            9 => wrap_fft(MixedRadix9xnAvx::new(fft).unwrap()),
            12 => wrap_fft(MixedRadix12xnAvx::new(fft).unwrap()),
            16 => wrap_fft(MixedRadix16xnAvx::new(fft).unwrap()),
            _ => panic!()
        }
    }

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

fn compare_fft_f64(b: &mut Bencher, strategy: &[usize]) {
    let mut fft = match strategy[0] {
        1 =>    wrap_fft(DFT::new(1, false)),
        2 =>    wrap_fft(Butterfly2::new(false)),
        3 =>    wrap_fft(Butterfly3::new(false)),
        4 =>    wrap_fft(Butterfly4::new(false)),
        5 =>    wrap_fft(Butterfly5::new(false)),
        6 =>    wrap_fft(Butterfly6::new(false)),
        7 =>    wrap_fft(Butterfly7::new(false)),
        8 =>    wrap_fft(MixedRadix64Avx4x2::new(false).unwrap()),
        9 =>    wrap_fft(MixedRadix64Avx3x3::new(false).unwrap()),
        12 =>   wrap_fft(MixedRadix64Avx4x3::new(false).unwrap()),
        16 =>   wrap_fft(MixedRadix64Avx4x4::new(false).unwrap()),
        18 =>   wrap_fft(MixedRadix64Avx3x6::new(false).unwrap()),
        24 =>   wrap_fft(MixedRadix64Avx4x6::new(false).unwrap()),
        27 =>   wrap_fft(MixedRadix64Avx3x9::new(false).unwrap()),
        32 =>   wrap_fft(MixedRadix64Avx4x8::new(false).unwrap()),
        36 =>   wrap_fft(MixedRadix64Avx6x6::new(false).unwrap()),
        _ => unimplemented!()
    };

    for radix in strategy.iter().skip(1) {
        fft = match radix {
            2 => wrap_fft(MixedRadix2xnAvx::new(fft).unwrap()),
            3 => wrap_fft(MixedRadix3xnAvx::new(fft).unwrap()),
            4 => wrap_fft(MixedRadix4xnAvx::new(fft).unwrap()),
            6 => wrap_fft(MixedRadix6xnAvx::new(fft).unwrap()),
            8 => wrap_fft(MixedRadix8xnAvx::new(fft).unwrap()),
            9 => wrap_fft(MixedRadix9xnAvx::new(fft).unwrap()),
            12 => wrap_fft(MixedRadix12xnAvx::new(fft).unwrap()),
            16 => wrap_fft(MixedRadix16xnAvx::new(fft).unwrap()),
            _ => panic!()
        }
    }

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

// passes the given FFT length directly to the FFT planner
fn bench_planned_fft_f32(b: &mut Bencher, len: usize) {
    let mut planner : FFTplanner<f32> = FFTplanner::new(false);
    let fft = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}


// passes the given FFT length directly to the FFT planner
fn bench_planned_fft_f64(b: &mut Bencher, len: usize) {
    let mut planner : FFTplanner<f64> = FFTplanner::new(false);
    let fft = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}





// Computes the given FFT length using Bluestein's Algorithm, using the planner to plan the inner FFT
fn bench_planned_bluesteins_f32(b: &mut Bencher, len: usize) {
    let mut planner : FftPlannerAvx<f32> = FftPlannerAvx::new(false).unwrap();
    let fft = planner.construct_bluesteins(len*2);

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}


// Computes the given FFT length using Bluestein's Algorithm, using the planner to plan the inner FFT
fn bench_2xn_bluesteins_f32(b: &mut Bencher, len: usize) {
    let mut planner : FftPlannerAvx<f32> = FftPlannerAvx::new(false).unwrap();
    let inner_fft = planner.construct_bluesteins(len);
    let fft : Arc<dyn Fft<f32>> = Arc::new(MixedRadix2xnAvx::new(inner_fft).unwrap());

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

// Computes the given FFT length using Rader's Algorithm, using the planner to plan the inner FFT
fn bench_planned_raders_f32(b: &mut Bencher, len: usize) {
    let mut planner : FftPlannerAvx<f32> = FftPlannerAvx::new(false).unwrap();
    let inner_fft = planner.construct_raders(len);
    let fft : Arc<dyn Fft<f32>> = Arc::new(MixedRadix2xnAvx::new(inner_fft).unwrap());

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

#[bench] fn comparef32__2power03__3power02__len00000072__72(b: &mut Bencher) { compare_fft_f32(b, &[72]); }
#[bench] fn comparef32__2power03__3power03__len00000216__72_3(b: &mut Bencher) { compare_fft_f32(b, &[72,3]); }
#[bench] fn comparef32__2power03__3power03__len00000216__54_4(b: &mut Bencher) { compare_fft_f32(b, &[54,4]); }
#[bench] fn comparef32__2power03__3power03__len00000216__36_6(b: &mut Bencher) { compare_fft_f32(b, &[36,6]); }
#[bench] fn comparef32__2power03__3power04__len00000648__36_6_3(b: &mut Bencher) { compare_fft_f32(b, &[36,6,3]); }
#[bench] fn comparef32__2power03__3power04__len00000648__72_9(b: &mut Bencher) { compare_fft_f32(b, &[72,9]); }
#[bench] fn comparef32__2power03__3power04__len00000648__54_12(b: &mut Bencher) { compare_fft_f32(b, &[54,12]); }
#[bench] fn comparef32__2power03__3power05__len00001944__72_9_3(b: &mut Bencher) { compare_fft_f32(b, &[72,9,3]); }
#[bench] fn comparef32__2power03__3power05__len00001944__54_12_3(b: &mut Bencher) { compare_fft_f32(b, &[54,12,3]); }
#[bench] fn comparef32__2power03__3power05__len00001944__54_9_4(b: &mut Bencher) { compare_fft_f32(b, &[54,9,4]); }
#[bench] fn comparef32__2power03__3power05__len00001944__54_6_6(b: &mut Bencher) { compare_fft_f32(b, &[54,6,6]); }
#[bench] fn comparef32__2power03__3power05__len00001944__36_9_6(b: &mut Bencher) { compare_fft_f32(b, &[36,9,6]); }
#[bench] fn comparef32__2power03__3power06__len00005832__54_6_6_3(b: &mut Bencher) { compare_fft_f32(b, &[54,6,6,3]); }
#[bench] fn comparef32__2power03__3power06__len00005832__36_9_6_3(b: &mut Bencher) { compare_fft_f32(b, &[36,9,6,3]); }
#[bench] fn comparef32__2power03__3power06__len00005832__72_9_9(b: &mut Bencher) { compare_fft_f32(b, &[72,9,9]); }
#[bench] fn comparef32__2power03__3power06__len00005832__54_12_9(b: &mut Bencher) { compare_fft_f32(b, &[54,12,9]); }
#[bench] fn comparef32__2power03__3power07__len00017496__72_9_9_3(b: &mut Bencher) { compare_fft_f32(b, &[72,9,9,3]); }
#[bench] fn comparef32__2power03__3power07__len00017496__54_12_9_3(b: &mut Bencher) { compare_fft_f32(b, &[54,12,9,3]); }
#[bench] fn comparef32__2power03__3power07__len00017496__54_9_9_4(b: &mut Bencher) { compare_fft_f32(b, &[54,9,9,4]); }
#[bench] fn comparef32__2power03__3power07__len00017496__54_9_6_6(b: &mut Bencher) { compare_fft_f32(b, &[54,9,6,6]); }
#[bench] fn comparef32__2power03__3power07__len00017496__36_9_9_6(b: &mut Bencher) { compare_fft_f32(b, &[36,9,9,6]); }
#[bench] fn comparef32__2power03__3power08__len00052488__54_9_6_6_3(b: &mut Bencher) { compare_fft_f32(b, &[54,9,6,6,3]); }
#[bench] fn comparef32__2power03__3power08__len00052488__36_9_9_6_3(b: &mut Bencher) { compare_fft_f32(b, &[36,9,9,6,3]); }
#[bench] fn comparef32__2power03__3power08__len00052488__72_9_9_9(b: &mut Bencher) { compare_fft_f32(b, &[72,9,9,9]); }
#[bench] fn comparef32__2power03__3power08__len00052488__54_12_9_9(b: &mut Bencher) { compare_fft_f32(b, &[54,12,9,9]); }
#[bench] fn comparef32__2power03__3power09__len00157464__72_9_9_9_3(b: &mut Bencher) { compare_fft_f32(b, &[72,9,9,9,3]); }
#[bench] fn comparef32__2power03__3power09__len00157464__54_12_9_9_3(b: &mut Bencher) { compare_fft_f32(b, &[54,12,9,9,3]); }
#[bench] fn comparef32__2power03__3power09__len00157464__54_9_9_9_4(b: &mut Bencher) { compare_fft_f32(b, &[54,9,9,9,4]); }
#[bench] fn comparef32__2power03__3power09__len00157464__54_9_9_6_6(b: &mut Bencher) { compare_fft_f32(b, &[54,9,9,6,6]); }
#[bench] fn comparef32__2power03__3power09__len00157464__36_9_9_9_6(b: &mut Bencher) { compare_fft_f32(b, &[36,9,9,9,6]); }