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
                last_ditch_strategy_list.push(current_strategy.clone());
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
    let butterfly_sizes_small3 = [ 36, 48, 54, 64 ]; 
    let butterfly_sizes_big3 = [ 36, 48, 54, 64 ]; 
    let last_ditch_butterflies = [ 27, 9 ]; 
    let available_radixes = [FftSize::new(3), FftSize::new(4), FftSize::new(6), FftSize::new(8), FftSize::new(9), FftSize::new(12), FftSize::new(16)];

    let max_len : usize = 1 << 20;
    let min_len = 64;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;
    
    for power3 in 3..max_power3 {
        for power2 in 10..max_power2 {
            let len = 3usize.pow(power3) << power2;
            if len > max_len { continue; }

            let planned_fft : Arc<dyn Fft<f32>> = rustfft::FFTplanner::new(false).plan_fft(len);

            let butterfly_sizes : &[usize] = if power2 > 4 { &butterfly_sizes_small3 } else { &butterfly_sizes_big3 };

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
                println!("#[bench] fn comparef32__len{:08}__2power{:02}__3power{:02}__{}(b: &mut Bencher) {{ compare_fft_f32(b, &[{}]); }}", len, power2, power3, test_id, strategy_array);
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
        _ => panic!()
    };

    for radix in strategy.iter().skip(1) {
        fft = match radix {
            2 => wrap_fft(MixedRadix2xnAvx::new_f32(fft).unwrap()),
            3 => wrap_fft(MixedRadix3xnAvx::new_f32(fft).unwrap()),
            4 => wrap_fft(MixedRadix4xnAvx::new_f32(fft).unwrap()),
            6 => wrap_fft(MixedRadix6xnAvx::new_f32(fft).unwrap()),
            8 => wrap_fft(MixedRadix8xnAvx::new_f32(fft).unwrap()),
            9 => wrap_fft(MixedRadix9xnAvx::new_f32(fft).unwrap()),
            12 => wrap_fft(MixedRadix12xnAvx::new_f32(fft).unwrap()),
            16 => wrap_fft(MixedRadix16xnAvx::new_f32(fft).unwrap()),
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
            2 => wrap_fft(MixedRadix2xnAvx::new_f64(fft).unwrap()),
            3 => wrap_fft(MixedRadix3xnAvx::new_f64(fft).unwrap()),
            4 => wrap_fft(MixedRadix4xnAvx::new_f64(fft).unwrap()),
            6 => wrap_fft(MixedRadix6xnAvx::new_f64(fft).unwrap()),
            8 => wrap_fft(MixedRadix8xnAvx::new_f64(fft).unwrap()),
            9 => wrap_fft(MixedRadix9xnAvx::new_f64(fft).unwrap()),
            12 => wrap_fft(MixedRadix12xnAvx::new_f64(fft).unwrap()),
            16 => wrap_fft(MixedRadix16xnAvx::new_f64(fft).unwrap()),
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
    let fft : Arc<dyn Fft<f32>> = Arc::new(MixedRadix2xnAvx::new_f32(inner_fft).unwrap());

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

// Computes the given FFT length using Rader's Algorithm, using the planner to plan the inner FFT
fn bench_planned_raders_f32(b: &mut Bencher, len: usize) {
    let mut planner : FftPlannerAvx<f32> = FftPlannerAvx::new(false).unwrap();
    let inner_fft = planner.construct_raders(len);
    let fft : Arc<dyn Fft<f32>> = Arc::new(MixedRadix2xnAvx::new_f32(inner_fft).unwrap());

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

#[bench] fn comparef64_len0000004_2power02_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 4); }
#[bench] fn comparef64_len0000012_2power02_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 12); }
#[bench] fn comparef64_len0000036_2power02_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 36); }
#[bench] fn comparef64_len0000108_2power02_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 108); }
#[bench] fn comparef64_len0000324_2power02_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 324); }
#[bench] fn comparef64_len0000972_2power02_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 972); }
#[bench] fn comparef64_len0002916_2power02_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 2916); }
#[bench] fn comparef64_len0008748_2power02_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 8748); }
#[bench] fn comparef64_len0026244_2power02_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 26244); }
#[bench] fn comparef64_len0078732_2power02_3power09(b: &mut Bencher) { bench_planned_fft_f64(b, 78732); }
#[bench] fn comparef64_len0236196_2power02_3power10(b: &mut Bencher) { bench_planned_fft_f64(b, 236196); }
#[bench] fn comparef64_len0708588_2power02_3power11(b: &mut Bencher) { bench_planned_fft_f64(b, 708588); }
#[bench] fn comparef64_len2125764_2power02_3power12(b: &mut Bencher) { bench_planned_fft_f64(b, 2125764); }
#[bench] fn comparef64_len6377292_2power02_3power13(b: &mut Bencher) { bench_planned_fft_f64(b, 6377292); }
#[bench] fn comparef64_len0000008_2power03_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 8); }
#[bench] fn comparef64_len0000024_2power03_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 24); }
#[bench] fn comparef64_len0000072_2power03_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 72); }
#[bench] fn comparef64_len0000216_2power03_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 216); }
#[bench] fn comparef64_len0000648_2power03_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 648); }
#[bench] fn comparef64_len0001944_2power03_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 1944); }
#[bench] fn comparef64_len0005832_2power03_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 5832); }
#[bench] fn comparef64_len0017496_2power03_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 17496); }
#[bench] fn comparef64_len0052488_2power03_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 52488); }
#[bench] fn comparef64_len0157464_2power03_3power09(b: &mut Bencher) { bench_planned_fft_f64(b, 157464); }
#[bench] fn comparef64_len0472392_2power03_3power10(b: &mut Bencher) { bench_planned_fft_f64(b, 472392); }
#[bench] fn comparef64_len1417176_2power03_3power11(b: &mut Bencher) { bench_planned_fft_f64(b, 1417176); }
#[bench] fn comparef64_len4251528_2power03_3power12(b: &mut Bencher) { bench_planned_fft_f64(b, 4251528); }
#[bench] fn comparef64_len0000016_2power04_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 16); }
#[bench] fn comparef64_len0000048_2power04_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 48); }
#[bench] fn comparef64_len0000144_2power04_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 144); }
#[bench] fn comparef64_len0000432_2power04_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 432); }
#[bench] fn comparef64_len0001296_2power04_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 1296); }
#[bench] fn comparef64_len0003888_2power04_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 3888); }
#[bench] fn comparef64_len0011664_2power04_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 11664); }
#[bench] fn comparef64_len0034992_2power04_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 34992); }
#[bench] fn comparef64_len0104976_2power04_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 104976); }
#[bench] fn comparef64_len0314928_2power04_3power09(b: &mut Bencher) { bench_planned_fft_f64(b, 314928); }
#[bench] fn comparef64_len0944784_2power04_3power10(b: &mut Bencher) { bench_planned_fft_f64(b, 944784); }
#[bench] fn comparef64_len2834352_2power04_3power11(b: &mut Bencher) { bench_planned_fft_f64(b, 2834352); }
#[bench] fn comparef64_len0000032_2power05_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 32); }
#[bench] fn comparef64_len0000096_2power05_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 96); }
#[bench] fn comparef64_len0000288_2power05_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 288); }
#[bench] fn comparef64_len0000864_2power05_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 864); }
#[bench] fn comparef64_len0002592_2power05_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 2592); }
#[bench] fn comparef64_len0007776_2power05_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 7776); }
#[bench] fn comparef64_len0023328_2power05_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 23328); }
#[bench] fn comparef64_len0069984_2power05_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 69984); }
#[bench] fn comparef64_len0209952_2power05_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 209952); }
#[bench] fn comparef64_len0629856_2power05_3power09(b: &mut Bencher) { bench_planned_fft_f64(b, 629856); }
#[bench] fn comparef64_len1889568_2power05_3power10(b: &mut Bencher) { bench_planned_fft_f64(b, 1889568); }
#[bench] fn comparef64_len5668704_2power05_3power11(b: &mut Bencher) { bench_planned_fft_f64(b, 5668704); }
#[bench] fn comparef64_len0000064_2power06_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 64); }
#[bench] fn comparef64_len0000192_2power06_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 192); }
#[bench] fn comparef64_len0000576_2power06_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 576); }
#[bench] fn comparef64_len0001728_2power06_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 1728); }
#[bench] fn comparef64_len0005184_2power06_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 5184); }
#[bench] fn comparef64_len0015552_2power06_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 15552); }
#[bench] fn comparef64_len0046656_2power06_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 46656); }
#[bench] fn comparef64_len0139968_2power06_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 139968); }
#[bench] fn comparef64_len0419904_2power06_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 419904); }
#[bench] fn comparef64_len1259712_2power06_3power09(b: &mut Bencher) { bench_planned_fft_f64(b, 1259712); }
#[bench] fn comparef64_len3779136_2power06_3power10(b: &mut Bencher) { bench_planned_fft_f64(b, 3779136); }
#[bench] fn comparef64_len0000128_2power07_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 128); }
#[bench] fn comparef64_len0000384_2power07_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 384); }
#[bench] fn comparef64_len0001152_2power07_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 1152); }
#[bench] fn comparef64_len0003456_2power07_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 3456); }
#[bench] fn comparef64_len0010368_2power07_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 10368); }
#[bench] fn comparef64_len0031104_2power07_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 31104); }
#[bench] fn comparef64_len0093312_2power07_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 93312); }
#[bench] fn comparef64_len0279936_2power07_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 279936); }
#[bench] fn comparef64_len0839808_2power07_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 839808); }
#[bench] fn comparef64_len2519424_2power07_3power09(b: &mut Bencher) { bench_planned_fft_f64(b, 2519424); }
#[bench] fn comparef64_len7558272_2power07_3power10(b: &mut Bencher) { bench_planned_fft_f64(b, 7558272); }
#[bench] fn comparef64_len0000256_2power08_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 256); }
#[bench] fn comparef64_len0000768_2power08_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 768); }
#[bench] fn comparef64_len0002304_2power08_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 2304); }
#[bench] fn comparef64_len0006912_2power08_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 6912); }
#[bench] fn comparef64_len0020736_2power08_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 20736); }
#[bench] fn comparef64_len0062208_2power08_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 62208); }
#[bench] fn comparef64_len0186624_2power08_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 186624); }
#[bench] fn comparef64_len0559872_2power08_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 559872); }
#[bench] fn comparef64_len1679616_2power08_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 1679616); }
#[bench] fn comparef64_len5038848_2power08_3power09(b: &mut Bencher) { bench_planned_fft_f64(b, 5038848); }
#[bench] fn comparef64_len0000512_2power09_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 512); }
#[bench] fn comparef64_len0001536_2power09_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 1536); }
#[bench] fn comparef64_len0004608_2power09_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 4608); }
#[bench] fn comparef64_len0013824_2power09_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 13824); }
#[bench] fn comparef64_len0041472_2power09_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 41472); }
#[bench] fn comparef64_len0124416_2power09_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 124416); }
#[bench] fn comparef64_len0373248_2power09_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 373248); }
#[bench] fn comparef64_len1119744_2power09_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 1119744); }
#[bench] fn comparef64_len3359232_2power09_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 3359232); }
#[bench] fn comparef64_len0001024_2power10_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 1024); }
#[bench] fn comparef64_len0003072_2power10_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 3072); }
#[bench] fn comparef64_len0009216_2power10_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 9216); }
#[bench] fn comparef64_len0027648_2power10_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 27648); }
#[bench] fn comparef64_len0082944_2power10_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 82944); }
#[bench] fn comparef64_len0248832_2power10_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 248832); }
#[bench] fn comparef64_len0746496_2power10_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 746496); }
#[bench] fn comparef64_len2239488_2power10_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 2239488); }
#[bench] fn comparef64_len6718464_2power10_3power08(b: &mut Bencher) { bench_planned_fft_f64(b, 6718464); }
#[bench] fn comparef64_len0002048_2power11_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 2048); }
#[bench] fn comparef64_len0006144_2power11_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 6144); }
#[bench] fn comparef64_len0018432_2power11_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 18432); }
#[bench] fn comparef64_len0055296_2power11_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 55296); }
#[bench] fn comparef64_len0165888_2power11_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 165888); }
#[bench] fn comparef64_len0497664_2power11_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 497664); }
#[bench] fn comparef64_len1492992_2power11_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 1492992); }
#[bench] fn comparef64_len4478976_2power11_3power07(b: &mut Bencher) { bench_planned_fft_f64(b, 4478976); }
#[bench] fn comparef64_len0004096_2power12_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 4096); }
#[bench] fn comparef64_len0012288_2power12_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 12288); }
#[bench] fn comparef64_len0036864_2power12_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 36864); }
#[bench] fn comparef64_len0110592_2power12_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 110592); }
#[bench] fn comparef64_len0331776_2power12_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 331776); }
#[bench] fn comparef64_len0995328_2power12_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 995328); }
#[bench] fn comparef64_len2985984_2power12_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 2985984); }
#[bench] fn comparef64_len0008192_2power13_3power00(b: &mut Bencher) { bench_planned_fft_f64(b, 8192); }
#[bench] fn comparef64_len0024576_2power13_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 24576); }
#[bench] fn comparef64_len0073728_2power13_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 73728); }
#[bench] fn comparef64_len0221184_2power13_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 221184); }
#[bench] fn comparef64_len0663552_2power13_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 663552); }
#[bench] fn comparef64_len1990656_2power13_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 1990656); }
#[bench] fn comparef64_len5971968_2power13_3power06(b: &mut Bencher) { bench_planned_fft_f64(b, 5971968); }
#[bench] fn comparef64_len0049152_2power14_3power01(b: &mut Bencher) { bench_planned_fft_f64(b, 49152); }
#[bench] fn comparef64_len0147456_2power14_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 147456); }
#[bench] fn comparef64_len0442368_2power14_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 442368); }
#[bench] fn comparef64_len1327104_2power14_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 1327104); }
#[bench] fn comparef64_len3981312_2power14_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 3981312); }
#[bench] fn comparef64_len0294912_2power15_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 294912); }
#[bench] fn comparef64_len0884736_2power15_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 884736); }
#[bench] fn comparef64_len2654208_2power15_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 2654208); }
#[bench] fn comparef64_len7962624_2power15_3power05(b: &mut Bencher) { bench_planned_fft_f64(b, 7962624); }
#[bench] fn comparef64_len0589824_2power16_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 589824); }
#[bench] fn comparef64_len1769472_2power16_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 1769472); }
#[bench] fn comparef64_len5308416_2power16_3power04(b: &mut Bencher) { bench_planned_fft_f64(b, 5308416); }
#[bench] fn comparef64_len1179648_2power17_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 1179648); }
#[bench] fn comparef64_len3538944_2power17_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 3538944); }
#[bench] fn comparef64_len2359296_2power18_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 2359296); }
#[bench] fn comparef64_len7077888_2power18_3power03(b: &mut Bencher) { bench_planned_fft_f64(b, 7077888); }
#[bench] fn comparef64_len4718592_2power19_3power02(b: &mut Bencher) { bench_planned_fft_f64(b, 4718592); }