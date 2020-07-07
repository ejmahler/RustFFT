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

use primal_check::miller_rabin;

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
    let butterfly_sizes = [ 128, 256, 512, 72, 36, 48, 54, 64 ]; 
    let last_ditch_butterflies = [ 27, 9, 32, 24 ]; 
    let available_radixes = [FftSize::new(3), FftSize::new(4), FftSize::new(6), FftSize::new(8), FftSize::new(9), FftSize::new(12), FftSize::new(16)];

    let max_len : usize = 1 << 21;
    let min_len = 64;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;
    
    for power3 in 1..2 {
        for power2 in 4..max_power2 {
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
    let butterfly_sizes = [ 512, 256, 128, 64, 36, 27, 24, 18, 12 ]; 
    let last_ditch_butterflies = [ 32, 16, 8, 9 ]; 
    let available_radixes = [FftSize::new(3), FftSize::new(4), FftSize::new(6), FftSize::new(8), FftSize::new(9), FftSize::new(12)];

    let max_len : usize = 1 << 21;
    let min_len = 64;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;

    for power3 in 0..1 {
        for power2 in 3..max_power2 {
            let len = 3usize.pow(power3) << power2;
            if len > max_len { continue; }

            //let planned_fft : Arc<dyn Fft<f32>> = rustfft::FFTplanner::new(false).plan_fft(len);

            // we want to catalog all the different possible ways there are to compute a FFT of size `len`
            // we can do that by recursively looping over each radix, dividing our length by that radix, then recursively trying rach radix again
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
    power5: u32,
    power7: u32,
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

        let mut power5 = 0;
        while other_factors % 5 == 0 {
            power5 += 1;
            other_factors /= 5;
        }

        let mut power7 = 0;
        while other_factors % 7 == 0 {
            power7 += 1;
            other_factors /= 7;
        }

        Self { power2, power3, power5, power7, other_factors }
    }

    pub fn get_power2(&self) -> u32 {
        self.power2
    }
    pub fn get_power3(&self) -> u32 {
        self.power3
    }
    pub fn get_power5(&self) -> u32 {
        self.power5
    }
    pub fn get_power7(&self) -> u32 {
        self.power7
    }
    pub fn get_other_factors(&self) -> usize {
        self.other_factors
    }
    pub fn product(&self) -> usize {
        (self.other_factors * 3usize.pow(self.power3) * 5usize.pow(self.power5) * 7usize.pow(self.power7)) << self.power2
    }
    pub fn product_power2power3(&self) -> usize {
        3usize.pow(self.power3) << self.power2
    }
    #[allow(unused)]
    pub fn divide_by(&self, divisor: &PartialFactors) -> Option<PartialFactors> {
        let two_divides = self.power2 >= divisor.power2;
        let three_divides = self.power3 >= divisor.power3;
        let five_divides = self.power5 >= divisor.power5;
        let seven_divides = self.power7 >= divisor.power7;
        let other_divides = self.other_factors % divisor.other_factors == 0;
        if two_divides && three_divides && five_divides && seven_divides && other_divides {
            Some(Self { 
                power2: self.power2 - divisor.power2,
                power3: self.power3 - divisor.power3,
                power5: self.power5 - divisor.power5,
                power7: self.power7 - divisor.power7,
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
    for len in 10usize..200 {
        if miller_rabin(len as u64) {
            let inner_other_factors = PartialFactors::compute(len - 1).get_other_factors();
            if inner_other_factors == 1 {
                println!("#[bench] fn comparef32_len{:07}_bluesteins(b: &mut Bencher) {{ bench_planned_bluesteins_f32(b, {}); }}", len, len);
                println!("#[bench] fn comparef32_len{:07}_raders(b: &mut Bencher) {{ bench_planned_raders_f32(b, {}); }}", len, len);
            }
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
        8 =>    wrap_fft(Butterfly8Avx::new(false).unwrap()),
        9 =>    wrap_fft(Butterfly9Avx::new(false).unwrap()),
        12 =>   wrap_fft(Butterfly12Avx::new(false).unwrap()),
        16 =>   wrap_fft(Butterfly16Avx::new(false).unwrap()),
        24 =>   wrap_fft(Butterfly24Avx::new(false).unwrap()),
        27 =>   wrap_fft(Butterfly27Avx::new(false).unwrap()),
        32 =>   wrap_fft(Butterfly32Avx::new(false).unwrap()),
        36 =>   wrap_fft(Butterfly36Avx::new(false).unwrap()),
        48 =>   wrap_fft(Butterfly48Avx::new(false).unwrap()),
        54 =>   wrap_fft(Butterfly54Avx::new(false).unwrap()),
        64 =>   wrap_fft(Butterfly64Avx::new(false).unwrap()),
        72 =>   wrap_fft(Butterfly72Avx::new(false).unwrap()),
        128 =>   wrap_fft(Butterfly128Avx::new(false).unwrap()),
        256 =>   wrap_fft(Butterfly256Avx::new(false).unwrap()),
        512 =>   wrap_fft(Butterfly512Avx::new(false).unwrap()),
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
        8 =>    wrap_fft(Butterfly8Avx64::new(false).unwrap()),
        9 =>    wrap_fft(Butterfly9Avx64::new(false).unwrap()),
        12 =>   wrap_fft(Butterfly12Avx64::new(false).unwrap()),
        16 =>   wrap_fft(Butterfly16Avx64::new(false).unwrap()),
        18 =>   wrap_fft(Butterfly18Avx64::new(false).unwrap()),
        24 =>   wrap_fft(Butterfly24Avx64::new(false).unwrap()),
        27 =>   wrap_fft(Butterfly27Avx64::new(false).unwrap()),
        32 =>   wrap_fft(Butterfly32Avx64::new(false).unwrap()),
        36 =>   wrap_fft(Butterfly36Avx64::new(false).unwrap()),
        64 =>   wrap_fft(Butterfly64Avx64::new(false).unwrap()),
        128=>   wrap_fft(Butterfly128Avx64::new(false).unwrap()),
        256=>   wrap_fft(Butterfly256Avx64::new(false).unwrap()),
        512=>   wrap_fft(Butterfly512Avx64::new(false).unwrap()),
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
    let fft = planner.construct_raders(len);

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}


// Computes the given FFT length using Bluestein's Algorithm, using the planner to plan the inner FFT
fn bench_planned_bluesteins_f64(b: &mut Bencher, len: usize) {
    let mut planner : FftPlannerAvx<f64> = FftPlannerAvx::new(false).unwrap();
    let fft = planner.construct_bluesteins(len*2);

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

// Computes the given FFT length using Rader's Algorithm, using the planner to plan the inner FFT
fn bench_planned_raders_f64(b: &mut Bencher, len: usize) {
    let mut planner : FftPlannerAvx<f64> = FftPlannerAvx::new(false).unwrap();
    let fft = planner.construct_raders(len);

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}

#[bench] fn comparef32_len0000011_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 11); }
#[bench] fn comparef32_len0000011_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 11); }
#[bench] fn comparef32_len0000013_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 13); }
#[bench] fn comparef32_len0000013_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 13); }
#[bench] fn comparef32_len0000017_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 17); }
#[bench] fn comparef32_len0000017_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 17); }
#[bench] fn comparef32_len0000019_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 19); }
#[bench] fn comparef32_len0000019_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 19); }
#[bench] fn comparef32_len0000029_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 29); }
#[bench] fn comparef32_len0000029_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 29); }
#[bench] fn comparef32_len0000031_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 31); }
#[bench] fn comparef32_len0000031_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 31); }
#[bench] fn comparef32_len0000037_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 37); }
#[bench] fn comparef32_len0000037_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 37); }
#[bench] fn comparef32_len0000041_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 41); }
#[bench] fn comparef32_len0000041_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 41); }
#[bench] fn comparef32_len0000043_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 43); }
#[bench] fn comparef32_len0000043_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 43); }
#[bench] fn comparef32_len0000061_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 61); }
#[bench] fn comparef32_len0000061_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 61); }
#[bench] fn comparef32_len0000071_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 71); }
#[bench] fn comparef32_len0000071_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 71); }
#[bench] fn comparef32_len0000073_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 73); }
#[bench] fn comparef32_len0000073_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 73); }
#[bench] fn comparef32_len0000097_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 97); }
#[bench] fn comparef32_len0000097_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 97); }
#[bench] fn comparef32_len0000101_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 101); }
#[bench] fn comparef32_len0000101_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 101); }
#[bench] fn comparef32_len0000109_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 109); }
#[bench] fn comparef32_len0000109_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 109); }
#[bench] fn comparef32_len0000113_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 113); }
#[bench] fn comparef32_len0000113_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 113); }
#[bench] fn comparef32_len0000127_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 127); }
#[bench] fn comparef32_len0000127_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 127); }
#[bench] fn comparef32_len0000151_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 151); }
#[bench] fn comparef32_len0000151_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 151); }
#[bench] fn comparef32_len0000163_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 163); }
#[bench] fn comparef32_len0000163_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 163); }
#[bench] fn comparef32_len0000181_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 181); }
#[bench] fn comparef32_len0000181_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 181); }
#[bench] fn comparef32_len0000193_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 193); }
#[bench] fn comparef32_len0000193_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 193); }
#[bench] fn comparef32_len0000197_bluesteins(b: &mut Bencher) { bench_planned_bluesteins_f32(b, 197); }
#[bench] fn comparef32_len0000197_raders(b: &mut Bencher) { bench_planned_raders_f32(b, 197); }