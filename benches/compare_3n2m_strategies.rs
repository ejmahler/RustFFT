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
        // if our strategy already contains any 2's, 3's, reject -- because 6 and 9 will be faster, respectively
        return !current_strategy.contains(&2) && !current_strategy.contains(&3);
    }
    // apply filters to size 4
    if potential_radix.len == 4 {
        // if our strategy already contains any 2's, reject -- because 8 will be faster
        // if our strategy already contains 2 4's, don't add a third, because 2 8's would have been faster
        // if our strategy already contains a 16, reject -- because 2 8's will be faster (8s are seriously fast guys)
        return !current_strategy.contains(&2) && current_strategy.iter().filter(|i| **i == 4).count() < 2 && !current_strategy.contains(&16);
    }
    if potential_radix.len == 16 {
         // if our strategy already contains a 4, reject -- because 2 8's will be faster (8s are seriously fast guys)
         // if our strategy already contains a 16, reject -- benchmarking shows that 16s are very situational, and repeating them never helps)
         return !current_strategy.contains(&4) && !current_strategy.contains(&16)
    }
    return true;
}

fn recursive_strategy_builder(strategy_list: &mut Vec<Vec<usize>>, last_ditch_strategy_list: &mut Vec<Vec<usize>>, mut current_strategy: Vec<usize>, len: FftSize, butterfly_sizes: &[usize], last_ditch_butterflies: &[usize], available_radixes: &[FftSize]) {
    if butterfly_sizes.contains(&len.len) {
        if filter_radix(&current_strategy, &len, true) {
            current_strategy.push(len.len);

            // If this strategy contains a 3, it's very unlikely to be the fastest. we don't want to rule it out, because it's required sometimes, but don't use it unless there aren't any other
            if current_strategy.contains(&3) {
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
        index == 0 || index == strategy.len() - 1 || (strategy[index - 1] < 12 && strategy[index + 1] >= 12)
    } else {
        true
    }
}

// cargo bench generate_3n2m_comparison_benchmarks -- --nocapture --ignored
#[ignore]
#[bench]
fn generate_3n2m_comparison_benchmarks(_: &mut test::Bencher) {
    let butterfly_sizes_small3 = [ 32, 48, 64 ]; 
    let butterfly_sizes_big3 = [ 48 ]; 
    let last_ditch_butterflies = [ 24, 12, 3 ]; 
    let available_radixes = [FftSize::new(3), FftSize::new(4), FftSize::new(6), FftSize::new(8), FftSize::new(9), FftSize::new(12), FftSize::new(16)];

    let max_len : usize = 1 << 20;
    let min_len = 64;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;
    
    for power3 in 0..max_power3 {
        for power2 in 4..max_power2 {
            let len = 3usize.pow(power3) << power2;
            if len > max_len { continue; }

            let planned_fft : Arc<dyn Fft<f32>> = rustfft::FFTplanner::new(false).plan_fft(len);

            let butterfly_sizes : &[usize] = if power3 <= 2 { &butterfly_sizes_small3 } else { &butterfly_sizes_big3 };

            // we want to catalog all the different possible ways there are to compute a FFT of size `len`
            // we can do that by recursively looping over each radix, dividing our length by that radix, then recursively trying rach radix again
            let mut strategies = vec![];
            let mut last_ditch_strategies = vec![];
            recursive_strategy_builder(&mut strategies, &mut last_ditch_strategies, Vec::new(), FftSize::new(len), &butterfly_sizes, &last_ditch_butterflies, &available_radixes);

            if strategies.len() == 0 {
                strategies = last_ditch_strategies;
            }

            for s in strategies.into_iter().filter(filter_strategy) {
                let strategy_strings : Vec<_> = s.into_iter().map(|i| i.to_string()).collect();
                let test_id = strategy_strings.join("_");
                let strategy_array = strategy_strings.join(",");
                println!("#[bench] fn comparef32__2power{:02}__3power{:02}__len{:03}__{}(b: &mut Bencher) {{ compare_fft_f32(b, &[{}]); }}", power2, power3, len, test_id, strategy_array);
            }
        }  
    }
}

// cargo bench generate_3n2m_planned_benchmarks -- --nocapture --ignored
#[ignore]
#[bench]
fn generate_3n2m_planned_benchmarks(_: &mut test::Bencher) {
    let mut fft_sizes = vec![];

    let max_len : usize = 1 << 20;
    let max_power2 = max_len.trailing_zeros();
    let max_power3 = (max_len as f32).log(3.0).ceil() as u32;
    for power2 in 1..3 {
        for power3 in 0..max_power3 {
            let len = 3usize.pow(power3) << power2;
            if len <= max_len && !(power2 == 3 && len > 24) {
                fft_sizes.push(len);
            }
        }
    }

    //fft_sizes.sort();

    for len in fft_sizes {
        let power2 = len.trailing_zeros();
        let mut remaining_factors = len >> power2;
        let mut power3 = 0;
        while remaining_factors % 3 == 0 {
            power3 += 1;
            remaining_factors /= 3;
        }

        println!("#[bench] fn comparef32_2power{:02}_3power{:02}(b: &mut Bencher) {{ bench_planned_fft_f32(b, {}); }}", power2, power3, len);
    }
}

fn wrap_fft<T: FFTnum>(fft: impl Fft<T> + 'static) -> Arc<dyn Fft<T>> {
    Arc::new(fft) as Arc<dyn Fft<T>>
}

fn compare_fft_f32(b: &mut Bencher, strategy: &[usize]) {
    let mut fft = match strategy.last().unwrap() {
        1 =>    wrap_fft(DFT::new(1, false)),
        2 =>    wrap_fft(Butterfly2::new(false)),
        3 =>    wrap_fft(Butterfly3::new(false)),
        4 =>    wrap_fft(Butterfly4::new(false)),
        5 =>    wrap_fft(Butterfly5::new(false)),
        6 =>    wrap_fft(Butterfly6::new(false)),
        7 =>    wrap_fft(Butterfly7::new(false)),
        8 =>    wrap_fft(MixedRadixAvx4x2::new(false).unwrap()),
        12 =>   wrap_fft(MixedRadixAvx4x3::new(false).unwrap()),
        16 =>   wrap_fft(MixedRadixAvx4x4::new(false).unwrap()),
        24 =>   wrap_fft(MixedRadixAvx4x6::new(false).unwrap()),
        32 =>   wrap_fft(MixedRadixAvx4x8::new(false).unwrap()),
        48 =>   wrap_fft(MixedRadixAvx4x12::new(false).unwrap()),
        64 =>   wrap_fft(MixedRadixAvx8x8::new(false).unwrap()),
        _ => panic!()
    };

    for radix in strategy.iter().rev().skip(1) {
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

fn bench_planned_fft_f32(b: &mut Bencher, len: usize) {
    let mut planner : FFTplanner<f32> = FFTplanner::new(false);
    let fft = planner.plan_fft(len);

    let mut buffer = vec![Complex::zero(); fft.len()];
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    b.iter(|| { fft.process_inplace_with_scratch(&mut buffer, &mut scratch); });
}
