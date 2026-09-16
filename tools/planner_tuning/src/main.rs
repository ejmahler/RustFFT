//! Measurement tool for RustFFT's planners.
//!
//! Works against any planner that implements `TunablePlanner`, selected with `--planner`.
//! Recipes are built through the planner's own internals, so what is measured here is exactly
//! what the planner would construct. "planner" below always means the fixed planner the
//! estimating one replaces, and "model" the estimating planner.
//!
//! Subcommands:
//!   time SPEC...          time recipes against each other; a '*reps' suffix runs a recipe
//!                         over a len*reps buffer, which is how inner FFTs are invoked
//!   regret LEN...         how far both planners' picks are from the best candidate measured
//!   sweep LEN... | A..B   time the fixed planner's pick against the estimating planner's, as TSV
//!   survey A..B           sweep over `--count` lengths drawn at random from a range
//!   verify LEN...         check every enumerated candidate against a direct DFT
//!   dump LEN...           time every candidate and write the raw timings, for offline replay
//!   score DUMP            replay a dump: regret of the cost model's pick among its candidates
//!   costs DUMP            replay a dump: the model's cost next to every measured time
//!   explain SPEC          print the cost model's cost tree for one recipe
//!   plantime LEN...       plan time of both planners against build time
//!   crossover LEN...      which recipe wins once construction cost counts

use rustfft::num_complex::Complex;
use rustfft::num_traits::{ToPrimitive, Zero};
use rustfft::tuning::{
    candidates_capped, parse, spec_cost, to_spec_string, CostModel, InstructionSet, ScalarTuner,
    Spec, TunablePlanner,
};
use rustfft::{Fft, FftDirection, FftNum};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct Subject<T: FftNum> {
    name: String,
    fft: Arc<dyn Fft<T>>,
    reps: usize,
    buffer: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    rounds: Vec<f64>,
}

impl<T: FftNum> Subject<T> {
    fn new(name: String, fft: Arc<dyn Fft<T>>, reps: usize) -> Self {
        let buffer = vec![Complex::zero(); fft.len() * reps];
        let scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
        Subject {
            name,
            fft,
            reps,
            buffer,
            scratch,
            rounds: Vec::new(),
        }
    }

    /// Wall-clock nanoseconds for `iters` passes over the whole buffer.
    fn time_block(&mut self, iters: usize) -> f64 {
        let start = Instant::now();
        for _ in 0..iters {
            self.fft
                .process_with_scratch(&mut self.buffer, &mut self.scratch);
        }
        start.elapsed().as_secs_f64() * 1e9
    }

    /// Nanoseconds per individual FFT, ie per chunk of `fft.len()`.
    fn time_per_fft(&mut self, iters: usize) -> f64 {
        let reps = self.reps;
        self.time_block(iters) / (iters * reps) as f64
    }

    fn best(&self) -> f64 {
        self.rounds.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn median(&self) -> f64 {
        let mut values = self.rounds.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[values.len() / 2]
    }
}

/// Time a set of subjects against each other, round-robin.
///
/// Round-robin rather than one at a time so any drift over the run hits every subject equally,
/// and min-of-rounds rather than a mean because the quantity of interest is the cost with
/// nothing else interfering.
fn measure<T: FftNum>(subjects: &mut [Subject<T>], rounds: usize, block_ms: f64) {
    let iters: Vec<usize> = subjects
        .iter_mut()
        .map(|subject| {
            let probe = subject.time_block(1);
            (((block_ms * 1e6) / probe.max(1.0)).ceil() as usize).clamp(1, 20_000_000)
        })
        .collect();

    for (subject, &n) in subjects.iter_mut().zip(iters.iter()) {
        subject.time_block(1 + n / 4);
    }

    for _ in 0..rounds {
        for (subject, &n) in subjects.iter_mut().zip(iters.iter()) {
            let per_fft = subject.time_per_fft(n);
            subject.rounds.push(per_fft);
        }
    }
}

/// Keep this thread on a performance core. Without it the macOS scheduler is free to park a
/// long-running thread on an efficiency core, which shows up as bimodal timings.
#[cfg(target_os = "macos")]
fn request_performance_core() {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn request_performance_core() {}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Overrides for the cost model's fitted weights. Anything left `None` keeps the planner's
/// default for its backend and element type.
#[derive(Default, Debug)]
struct Weights {
    strided: Option<f64>,
    permuted: Option<f64>,
    rader_index: Option<f64>,
    radixn_extra: Option<f64>,
    general_row: Option<f64>,
    small_row: Option<f64>,
}

impl Weights {
    fn apply(&self, mut model: CostModel) -> CostModel {
        let set = |field: &mut f64, value: Option<f64>| {
            if let Some(value) = value {
                *field = value;
            }
        };
        set(&mut model.strided, self.strided);
        set(&mut model.permuted, self.permuted);
        set(&mut model.rader_index, self.rader_index);
        set(&mut model.radixn_extra, self.radixn_extra);
        set(&mut model.general_row, self.general_row);
        set(&mut model.small_row, self.small_row);
        model
    }
}

struct Options {
    rounds: usize,
    block_ms: f64,
    cap: usize,
    verbose: bool,
    /// Where `dump` writes its rows.
    out: Option<String>,
    weights: Weights,
    f32: bool,
    /// For `survey`: how many lengths, and the seed that picks them.
    count: usize,
    seed: u64,
}

/// A tuner for `P`, with the weight overrides applied to its estimating planner.
fn tuner<T: FftNum, P: TunablePlanner<T>>(opts: &Options) -> P {
    let mut planner = P::new();
    if let Some(model) = planner.cost_model() {
        planner.set_cost_model(opts.weights.apply(model));
    }
    planner
}

/// The cost model for pricing specs offline, for a planner named on the command line or in a
/// dump header. wasm_simd borrows NEON's counts, as the planner itself does.
fn offline_model(planner_label: &str, opts: &Options) -> CostModel {
    let instruction_set = match planner_label {
        "sse" => InstructionSet::Sse,
        "neon" | "wasm_simd" => InstructionSet::Neon,
        other => {
            eprintln!(
                "the cost model has no instruction counts for planner '{}'",
                other
            );
            std::process::exit(2);
        }
    };
    opts.weights.apply(CostModel::new(
        instruction_set,
        if opts.f32 { 2 } else { 1 },
    ))
}

/// The planner a dump was taken with, from its header.
fn dump_planner(text: &str) -> String {
    text.lines()
        .find_map(|line| line.strip_prefix("# planner\t"))
        .unwrap_or("neon")
        .to_string()
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[((sorted.len() as f64 * fraction) as usize).min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

fn cmd_time<T: FftNum, P: TunablePlanner<T>>(specs: &[String], opts: &Options) {
    let mut planner = tuner::<T, P>(opts);
    let mut subjects: Vec<Subject<T>> = specs
        .iter()
        .map(|text| {
            let (spec_text, reps) = match text.rsplit_once('*') {
                Some((spec, reps)) => (spec, reps.parse().expect("bad repeat count")),
                None => (text.as_str(), 1usize),
            };
            let spec = parse(spec_text).unwrap_or_else(|e| {
                eprintln!("error in spec '{}': {}", spec_text, e);
                std::process::exit(2);
            });
            let fft = planner.build(&spec, FftDirection::Forward);
            Subject::new(text.clone(), fft, reps)
        })
        .collect();

    measure(&mut subjects, opts.rounds, opts.block_ms);

    println!(
        "{:<46} {:>9} {:>7} {:>13} {:>9}",
        "recipe", "len", "reps", "min ns", "spread"
    );
    for subject in subjects.iter() {
        let (min, med) = (subject.best(), subject.median());
        println!(
            "{:<46} {:>9} {:>7} {:>13.1} {:>8.2}%",
            subject.name,
            subject.fft.len(),
            subject.reps,
            min,
            (med - min) / min * 100.0
        );
    }

    if subjects.len() > 1 {
        let best = subjects
            .iter()
            .map(|s| s.best())
            .fold(f64::INFINITY, f64::min);
        println!("\nrelative to best:");
        for subject in subjects.iter() {
            println!("  {:<46} {:>6.3}x", subject.name, subject.best() / best);
        }
    }
}

/// Error budget for a correct FFT of this length in this precision.
///
/// Two terms, because two things are inexact.
///
/// The recipe's own error grows like `eps * sqrt(log2 len)`: an FFT accumulates rounding over its
/// `log2 len` passes, not over its elements. The margin of 20 leaves room for recipes that compose
/// several algorithms, and still catches a real defect, which shows up as orders of magnitude
/// rather than as a factor of two.
///
/// The reference's error grows like `eps * sqrt(len)`, because a naive DFT really does sum `len`
/// terms per output. The reference runs in f64, so for f32 recipes this term is negligible and the
/// budget is tight. For f64 recipes the reference is no better than the thing it judges, and this
/// term dominates: at len 100000 it is 2.8e-13 against the recipes' 1.8e-14.
fn tolerance<T: FftNum>(len: usize) -> f64 {
    let eps = if std::mem::size_of::<T>() == 4 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    let len = len as f64;
    20.0 * eps * len.log2().max(1.0).sqrt() + 4.0 * f64::EPSILON * len.sqrt()
}

/// Measure every candidate at every length and write the raw timings to a file.
///
/// This exists so that a cost model can be scored and refitted offline, against a frozen
/// dataset, instead of needing the machine for every iteration. Rows are
/// `len<TAB>spec<TAB>ns<TAB>pass`, where pass is 1 for the wide sweep and 2 for the careful
/// re-timing the fastest few get.
fn cmd_dump<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    use std::io::Write;
    const FINALISTS: usize = 8;

    let path = opts.out.clone().unwrap_or_else(|| "dump.tsv".to_string());
    let mut f =
        std::io::BufWriter::new(std::fs::File::create(&path).expect("cannot create output"));
    writeln!(f, "# planner\t{}", P::label()).unwrap();
    writeln!(
        f,
        "# rounds\t{}\tblock_ms\t{}\tcap\t{}",
        opts.rounds, opts.block_ms, opts.cap
    )
    .unwrap();
    writeln!(f, "len\tspec\tns\tpass\tplanner_pick").unwrap();

    for &len in lengths {
        let mut planner = tuner::<T, P>(opts);
        let picked = to_spec_string(&planner.plan(len));
        let specs = candidates_capped(&mut planner, len, opts.cap);
        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();
        measure(&mut subjects, opts.rounds, opts.block_ms);

        // Re-time the fastest few for longer. The minimum of many noisy draws is biased low, so
        // without this the best candidate looks faster than it is and every regret is flattered.
        let mut order: Vec<usize> = (0..subjects.len()).collect();
        order.sort_by(|a, b| {
            subjects[*a]
                .best()
                .partial_cmp(&subjects[*b].best())
                .unwrap()
        });
        let finalists: Vec<usize> = order.into_iter().take(FINALISTS).collect();
        let mut finals: Vec<Subject<T>> = finalists
            .iter()
            .map(|&i| {
                let fft = planner.build(&specs[i], FftDirection::Forward);
                Subject::new(subjects[i].name.clone(), fft, 1)
            })
            .collect();
        measure(&mut finals, opts.rounds * 4, opts.block_ms);

        for (i, s) in subjects.iter().enumerate() {
            let is_pick = if s.name == picked { "1" } else { "0" };
            writeln!(f, "{}\t{}\t{:.3}\t1\t{}", len, s.name, s.best(), is_pick).unwrap();
            let _ = i;
        }
        for s in finals.iter() {
            let is_pick = if s.name == picked { "1" } else { "0" };
            writeln!(f, "{}\t{}\t{:.3}\t2\t{}", len, s.name, s.best(), is_pick).unwrap();
        }
        println!("{:>9}  {} candidates", len, subjects.len());
    }
    println!("wrote {}", path);
}

/// Candidates for `len` with the estimating planner's pick guaranteed among them, and its index.
///
/// The exhaustive enumeration takes its inner recipes from the fixed planner, while the
/// estimating planner optimises its inners too, so its pick is often not in the list.
fn candidates_with_model_pick<T: FftNum, P: TunablePlanner<T>>(
    planner: &mut P,
    len: usize,
    cap: usize,
) -> (Vec<Arc<Spec>>, Option<usize>) {
    let mut specs = candidates_capped(planner, len, cap);
    let model_index =
        planner
            .estimate(len)
            .map(|pick| match specs.iter().position(|spec| **spec == *pick) {
                Some(index) => index,
                None => {
                    specs.push(pick);
                    specs.len() - 1
                }
            });
    (specs, model_index)
}

fn cmd_regret<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    println!(
        "{:>9} {:>8} {:>8} {:>10} {:>10}  {}",
        "len", "planner", "model", "planner ns", "best ns", "best recipe (when neither picked it)"
    );

    let mut planner_regrets = Vec::new();
    let mut model_regrets = Vec::new();

    for &len in lengths {
        let mut planner = tuner::<T, P>(opts);
        let (specs, model_index) = candidates_with_model_pick(&mut planner, len, opts.cap);

        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();
        measure(&mut subjects, opts.rounds, opts.block_ms);

        let best_index = (0..subjects.len())
            .min_by(|&a, &b| subjects[a].best().total_cmp(&subjects[b].best()))
            .unwrap();
        let best_time = subjects[best_index].best();
        let planner_regret = subjects[0].best() / best_time;
        let model_regret = model_index.map(|i| subjects[i].best() / best_time);

        println!(
            "{:>9} {:>7.3}x {:>8} {:>10.0} {:>10.0}  {}",
            len,
            planner_regret,
            model_regret
                .map(|r| format!("{:.3}x", r))
                .unwrap_or_else(|| "-".into()),
            subjects[0].best(),
            best_time,
            if best_index == 0 || Some(best_index) == model_index {
                String::new()
            } else {
                subjects[best_index].name.clone()
            }
        );
        if opts.verbose {
            let mut ranked: Vec<&Subject<T>> = subjects.iter().collect();
            ranked.sort_by(|a, b| a.best().total_cmp(&b.best()));
            for subject in ranked.iter().take(6) {
                println!(
                    "            {:>6.3}x  {}",
                    subject.best() / best_time,
                    subject.name
                );
            }
            println!("            planner: {}", subjects[0].name);
            if let Some(i) = model_index {
                println!("            model:   {}", subjects[i].name);
            }
            println!("            ({} candidates measured)", subjects.len());
        }

        planner_regrets.push(planner_regret);
        if let Some(r) = model_regret {
            model_regrets.push(r);
        }
    }

    println!("\n--- regret: pick divided by the best recipe measured ---");
    for (label, mut values) in [("planner", planner_regrets), ("model", model_regrets)] {
        if values.is_empty() {
            continue;
        }
        values.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        println!(
            "  {:<8} n={:<4} mean {:.4}  median {:.4}  p90 {:.4}  worst {:.4}  more than 2% off: {}",
            label,
            values.len(),
            mean,
            percentile(&values, 0.5),
            percentile(&values, 0.9),
            values[values.len() - 1],
            values.iter().filter(|&&r| r > 1.02).count()
        );
    }
}

/// Time the fixed planner's pick against the estimating planner's, at every length given.
///
/// Builds and times only those two recipes per length, which is what makes a thousand lengths
/// affordable. When both agree, the recipe is timed once and reported for both, so an agreement
/// shows as exactly 1.000 rather than as timing noise.
///
/// Output is TSV on stdout, one row per length, ready for `plot_sweep.py`, followed by a summary
/// in `#` comment lines. The `cands` column is kept for the plotting script and is always `-`.
fn cmd_sweep<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    let mut planner = tuner::<T, P>(opts);
    let Some(model) = planner.cost_model() else {
        eprintln!(
            "the {} planner does not estimate, so there is nothing to sweep",
            P::label()
        );
        std::process::exit(2);
    };

    println!("# planner\t{}", P::label());
    println!("# elem\t{}", if opts.f32 { "f32" } else { "f64" });
    println!("# model\t{:?}", model);
    println!("# rounds\t{}\tblock_ms\t{}", opts.rounds, opts.block_ms);
    println!(
        "len\tcands\tagree\tplanner_ns\tmodel_ns\tratio\tplanner_norm\tmodel_norm\tplanner_spec\tmodel_spec"
    );

    // Estimating over fixed, so below 1 means the estimating planner is faster.
    let mut changes = Vec::new();
    for &len in lengths {
        let planner_spec = planner.plan(len);
        let model_spec = planner.estimate(len).unwrap();
        let agree = planner_spec == model_spec;

        let mut subjects = vec![Subject::new(
            to_spec_string(&planner_spec),
            planner.build(&planner_spec, FftDirection::Forward),
            1,
        )];
        if !agree {
            subjects.push(Subject::new(
                to_spec_string(&model_spec),
                planner.build(&model_spec, FftDirection::Forward),
                1,
            ));
        }
        measure(&mut subjects, opts.rounds, opts.block_ms);

        let planner_ns = subjects[0].best();
        let model_ns = subjects[subjects.len() - 1].best();
        changes.push(model_ns / planner_ns);

        // n log2 n, the work a radix-2 FFT of this length would do. Undefined at len 1, where
        // there is no work to normalise by.
        let nlogn = (len as f64) * (len as f64).log2();
        let norm = |ns: f64| if nlogn > 0.0 { ns / nlogn } else { f64::NAN };

        println!(
            "{}\t-\t{}\t{:.2}\t{:.2}\t{:.4}\t{:.5}\t{:.5}\t{}\t{}",
            len,
            if agree { 1 } else { 0 },
            planner_ns,
            model_ns,
            planner_ns / model_ns,
            norm(planner_ns),
            norm(model_ns),
            subjects[0].name,
            subjects[subjects.len() - 1].name
        );
    }

    if changes.is_empty() {
        return;
    }
    let agreed = changes.iter().filter(|&&c| c == 1.0).count();
    let geomean = (changes.iter().map(|c| c.ln()).sum::<f64>() / changes.len() as f64).exp();
    changes.sort_by(f64::total_cmp);
    println!("# summary\truntime of the estimating planner's pick over the fixed planner's");
    println!("# lengths\t{}\tsame recipe\t{}", changes.len(), agreed);
    println!("# geomean\t{:.4}", geomean);
    for (label, fraction) in [
        ("p10", 0.1),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p90", 0.9),
    ] {
        println!("# {}\t{:.4}", label, percentile(&changes, fraction));
    }
    println!(
        "# best\t{:.4}\tworst\t{:.4}",
        changes[0],
        changes[changes.len() - 1]
    );
}

/// `count` distinct lengths drawn uniformly from `lo..=hi`, sorted, from a fixed seed.
fn random_lengths(lo: usize, hi: usize, count: usize, seed: u64) -> Vec<usize> {
    // splitmix64, so the tool needs no dependencies and a seed reproduces a survey anywhere.
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let span = (hi - lo + 1) as u64;
    let count = count.min(span as usize);
    let mut lengths = std::collections::BTreeSet::new();
    while lengths.len() < count {
        lengths.insert(lo + (next() % span) as usize);
    }
    lengths.into_iter().collect()
}

/// Check that every enumerated candidate, and the estimating planner's pick, computes a correct
/// FFT. Timing a recipe says nothing about whether it is valid, and an invalid one would happily
/// produce fast wrong answers. Each candidate is compared against a direct DFT.
fn cmd_verify<T: FftNum + ToPrimitive, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    let mut worst_overall: f64 = 0.0;
    let mut failures = 0usize;

    for &len in lengths {
        let input: Vec<Complex<T>> = (0..len)
            .map(|i| {
                let x = ((i * 2654435761usize) % 1000) as f64 / 500.0 - 1.0;
                let y = ((i * 40503usize) % 1000) as f64 / 500.0 - 1.0;
                Complex::new(T::from_f64(x).unwrap(), T::from_f64(y).unwrap())
            })
            .collect();

        // Reference: a direct DFT, which shares no code with the recipes under test.
        //
        // It is computed in f64 even when the recipes run in f32. A same-precision reference is
        // useless at large f32 lengths: the naive DFT sums `len` terms, so its own error grows
        // with length and swamps what is being measured. At len 100000 an f32 reference DFT is
        // off by 1.6e-5, which is 30x the error of the recipes it is judging.
        let reference_fft = rustfft::algorithm::Dft::<f64>::new(len, FftDirection::Forward);
        let mut reference: Vec<Complex<f64>> = input
            .iter()
            .map(|c| Complex::new(c.re.to_f64().unwrap(), c.im.to_f64().unwrap()))
            .collect();
        let mut reference_scratch = vec![Complex::zero(); reference_fft.get_inplace_scratch_len()];
        reference_fft.process_with_scratch(&mut reference, &mut reference_scratch);
        let reference_norm: f64 = reference.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

        let mut planner = tuner::<T, P>(opts);
        let (specs, _) = candidates_with_model_pick(&mut planner, len, opts.cap);
        let mut worst_here: f64 = 0.0;
        let mut worst_spec = String::new();

        for spec in specs.iter() {
            let fft = planner.build(spec, FftDirection::Forward);
            let mut buffer = input.clone();
            let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
            fft.process_with_scratch(&mut buffer, &mut scratch);

            let error: f64 = buffer
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| {
                    let d =
                        Complex::new(a.re.to_f64().unwrap() - b.re, a.im.to_f64().unwrap() - b.im);
                    d.norm_sqr()
                })
                .sum::<f64>()
                .sqrt()
                / reference_norm;
            if error > worst_here {
                worst_here = error;
                worst_spec = to_spec_string(spec);
            }
        }

        let bad = worst_here > tolerance::<T>(len);
        if bad {
            failures += 1;
        }
        println!(
            "{:>8}  {} candidates, worst relative error {:.3e} (budget {:.1e})  {}{}",
            len,
            specs.len(),
            worst_here,
            tolerance::<T>(len),
            if bad { "FAIL " } else { "" },
            worst_spec
        );
        worst_overall = worst_overall.max(worst_here);
    }

    println!(
        "\nworst relative error over all lengths: {:.3e}  ({} lengths failed)",
        worst_overall, failures
    );
    if failures > 0 {
        std::process::exit(1);
    }
}

/// Rows of a dump file: len -> [(spec, best ns, is the fixed planner's pick)].
///
/// Pass 2, the careful re-timing, overwrites pass 1. The inner list preserves the dump's own
/// order, which is the order `candidates_capped` enumerated in, so that replay breaks cost ties
/// the same way enumeration would. Keying by spec string instead would sort `gts(b10,b3)` ahead
/// of `gts(b3,b10)` and silently reverse every width/height tie.
fn read_dump(text: &str) -> std::collections::BTreeMap<usize, Vec<(String, f64, bool)>> {
    let mut data: std::collections::BTreeMap<usize, Vec<(String, f64, bool)>> = Default::default();
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("len\t") {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 {
            continue;
        }
        let (len, spec, ns, pass, pick) = (
            f[0].parse::<usize>().unwrap(),
            f[1].to_string(),
            f[2].parse::<f64>().unwrap(),
            f[3].parse::<u32>().unwrap(),
            f[4] == "1",
        );
        let rows = data.entry(len).or_default();
        match rows.iter_mut().find(|row| row.0 == spec) {
            Some(row) if pass != 1 => {
                row.1 = ns;
                row.2 = pick;
            }
            Some(_) => {}
            None => rows.push((spec, ns, pick)),
        }
    }
    data
}

/// Score the cost model against a dump file. No measurement, no planner, no machine.
///
/// This replays a one-level choice among the dumped candidates, whose inner recipes are the fixed
/// planner's. It is how the weights were fitted, and it is not the same as the estimating
/// planner's pick, which also optimises the inner recipes; `sweep` measures that.
fn cmd_score(path: &str, opts: &Options) {
    let text = std::fs::read_to_string(path).expect("cannot read dump");
    let model = offline_model(&dump_planner(&text), opts);
    println!("model: {:?}", model);
    println!(
        "{:>9} {:>9} {:>9}   {}",
        "len", "model", "planner", "model's pick (when it is not the best)"
    );

    let (mut model_regrets, mut planner_regrets) = (Vec::new(), Vec::new());
    for (len, rows) in read_dump(&text) {
        let best = rows.iter().map(|row| row.1).fold(f64::INFINITY, f64::min);
        let planner_regret = rows.iter().find(|row| row.2).map(|row| row.1 / best);

        let pick = rows
            .iter()
            .filter_map(|(text, ns, _)| {
                let spec = parse(text).ok()?;
                Some((text, spec_cost(&model, &spec)?, *ns))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((pick_name, _, pick_ns)) = pick else {
            println!("{:>9}  no candidate priced", len);
            continue;
        };
        let model_regret = pick_ns / best;
        model_regrets.push(model_regret);
        if let Some(p) = planner_regret {
            planner_regrets.push(p);
        }
        println!(
            "{:>9} {:>8.3}x {:>8}   {}",
            len,
            model_regret,
            planner_regret
                .map(|p| format!("{:.3}x", p))
                .unwrap_or_else(|| "-".into()),
            if model_regret <= 1.0001 {
                "= best"
            } else {
                pick_name
            }
        );
    }

    println!("\n--- regret, pick divided by best measured ---");
    for (label, mut values) in [("model", model_regrets), ("planner", planner_regrets)] {
        if values.is_empty() {
            continue;
        }
        values.sort_by(f64::total_cmp);
        println!(
            "  {:<9} n={:<4} mean {:.4}  median {:.4}  p90 {:.4}  worst {:.4}",
            label,
            values.len(),
            values.iter().sum::<f64>() / values.len() as f64,
            percentile(&values, 0.5),
            percentile(&values, 0.9),
            values[values.len() - 1]
        );
    }
}

/// Print the model's cost for every measured candidate, for offline analysis.
///
/// Columns: len, spec, measured ns, model cost. Pure replay, no machine needed.
fn cmd_costs(path: &str, opts: &Options) {
    let text = std::fs::read_to_string(path).expect("cannot read dump");
    let model = offline_model(&dump_planner(&text), opts);
    println!("len\tspec\tns\tcost\tplanner_pick");
    for (len, rows) in read_dump(&text) {
        for (spec_text, ns, pick) in rows {
            let cost = parse(&spec_text)
                .ok()
                .and_then(|spec| spec_cost(&model, &spec))
                .map(|c| format!("{:.1}", c))
                .unwrap_or_else(|| "NA".into());
            println!(
                "{}\t{}\t{:.3}\t{}\t{}",
                len,
                spec_text,
                ns,
                cost,
                if pick { 1 } else { 0 }
            );
        }
    }
}

/// Print the model's cost tree for one recipe, so the recursion can be checked by eye.
fn cmd_explain(spec_text: &str, planner_label: &str, opts: &Options) {
    let spec = parse(spec_text).expect("could not parse spec");
    let model = offline_model(planner_label, opts);

    fn walk(model: &CostModel, spec: &Spec, mult: f64, depth: usize, out: &mut Vec<String>) {
        let own = spec_cost(model, spec).unwrap_or(f64::NAN);
        // Each child, at the multiplicity the parent runs it.
        let kids: Vec<(&Spec, f64)> = match spec {
            Spec::MixedRadix { left, right, .. } | Spec::GoodThomas { left, right, .. } => vec![
                (left.as_ref(), right.len() as f64),
                (right.as_ref(), left.len() as f64),
            ],
            Spec::RadixN { radixes, base } => {
                vec![(base.as_ref(), radixes.iter().product::<usize>() as f64)]
            }
            Spec::Radix4 { k, base } => vec![(base.as_ref(), (1u64 << (2 * k)) as f64)],
            Spec::Raders { inner } | Spec::Bluesteins { inner, .. } => vec![(inner.as_ref(), 2.0)],
            _ => vec![],
        };
        let child_total: f64 = kids
            .iter()
            .map(|(child, m)| m * spec_cost(model, child).unwrap_or(0.0))
            .sum();
        out.push(format!(
            "{:indent$}{:<34} len {:>7}  x{:<8.0} cost {:>14.0}  own {:>12.0}",
            "",
            to_spec_string(spec),
            spec.len(),
            mult,
            mult * own,
            mult * (own - child_total),
            indent = depth * 2
        ));
        for (child, m) in kids {
            walk(model, child, mult * m, depth + 1, out);
        }
    }

    let mut out = Vec::new();
    walk(&model, &spec, 1.0, 0, &mut out);
    println!("model: {:?}", model);
    println!(
        "{:<36} {:>11}  {:<9} {:>19} {:>16}",
        "recipe", "len", "times", "total cost", "own cost"
    );
    for line in out {
        println!("{}", line);
    }
}

/// Compare the cost of *choosing* a recipe against the cost of *building* it.
///
/// Planning only produces a `Recipe`. Turning that into an `Arc<dyn Fft>` is a separate and much
/// larger job: every algorithm precomputes twiddles, and Rader's and Bluestein's both run a full
/// inner FFT inside their constructors. So the question that decides whether the estimating
/// planner is affordable is not how much slower it is than the fixed planner, but how much it
/// adds to plan-plus-build, which is what a caller actually pays before the first transform.
///
/// Every timing uses a fresh planner, so these are cold-start figures: nothing is served from a
/// cache, including the inner lengths a session planning several lengths would share.
fn cmd_plantime<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    println!(
        "{:>7} {:>12} {:>14} {:>12} {:>9} {:>11}",
        "len", "plan fixed", "plan estimate", "build", "build/plan", "extra vs"
    );
    println!(
        "{:>7} {:>12} {:>14} {:>12} {:>9} {:>11}",
        "", "ns", "ns", "ns", "", "plan+build"
    );
    let (mut tot_a, mut tot_b, mut tot_c) = (0.0, 0.0, 0.0);
    for &len in lengths {
        let reps = 50;
        let time_plan = |plan: &dyn Fn(&mut P) -> Arc<Spec>| {
            let mut planners: Vec<P> = (0..reps).map(|_| tuner::<T, P>(opts)).collect();
            let start = Instant::now();
            for planner in planners.iter_mut() {
                std::hint::black_box(plan(planner));
            }
            start.elapsed().as_secs_f64() * 1e9 / reps as f64
        };
        let a = time_plan(&|planner| planner.plan(len));
        let b = time_plan(&|planner| planner.estimate(len).unwrap());

        // Build the recipe the estimating planner chose. `build` uses a fresh planner each time.
        let mut planner = tuner::<T, P>(opts);
        let spec = planner.estimate(len).unwrap();
        let build_reps = if len > 4096 {
            5
        } else if len > 256 {
            20
        } else {
            100
        };
        let start = Instant::now();
        for _ in 0..build_reps {
            std::hint::black_box(planner.build(&spec, FftDirection::Forward));
        }
        let c = start.elapsed().as_secs_f64() * 1e9 / build_reps as f64;

        tot_a += a;
        tot_b += b;
        tot_c += c;
        println!(
            "{:>7} {:>12.0} {:>14.0} {:>12.0} {:>8.0}x {:>10.1}%",
            len,
            a,
            b,
            c,
            c / b,
            100.0 * (b - a) / (a + c)
        );
    }
    println!(
        "\n  totals: plan fixed {:.0} ns, plan estimate {:.0} ns ({:.1}x), build {:.0} ns",
        tot_a,
        tot_b,
        tot_b / tot_a,
        tot_c
    );
    println!(
        "  the estimating planner adds {:.2}% to plan-plus-build",
        100.0 * (tot_b - tot_a) / (tot_a + tot_c)
    );
}

/// How the best recipe changes once construction cost is counted.
///
/// For every candidate this measures build time and execution time, then reports which recipe
/// minimises `build + k * execute` at several values of `k`, and how many executions the
/// estimating planner's pick needs to repay any extra build cost. Plan time is deliberately
/// excluded: it is the same for every candidate at one length, so it cannot change which wins.
fn cmd_crossover<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    println!(
        "{:>7} {:>6} {:>6} {:>11} {:>11} {:>12}  {}",
        "len", "cands", "k", "build ns", "exec ns", "total ns", "recipe"
    );

    for &len in lengths {
        let mut planner = tuner::<T, P>(opts);
        let (specs, model_index) = candidates_with_model_pick(&mut planner, len, opts.cap);
        let Some(mi) = model_index else {
            eprintln!("the {} planner does not estimate", P::label());
            std::process::exit(2);
        };

        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();
        measure(&mut subjects, opts.rounds, opts.block_ms);
        let exec: Vec<f64> = subjects.iter().map(|s| s.best()).collect();

        // `build` uses a fresh planner each time, so no inner FFT is served from a cache.
        let build_reps = if len > 4096 {
            5
        } else if len > 256 {
            20
        } else {
            100
        };
        let build: Vec<f64> = specs
            .iter()
            .map(|spec| {
                let start = Instant::now();
                for _ in 0..build_reps {
                    std::hint::black_box(planner.build(spec, FftDirection::Forward));
                }
                start.elapsed().as_secs_f64() * 1e9 / build_reps as f64
            })
            .collect();

        let pick = |k: f64| -> usize {
            (0..specs.len())
                .min_by(|&a, &b| (build[a] + k * exec[a]).total_cmp(&(build[b] + k * exec[b])))
                .unwrap()
        };

        for (row, k) in [1.0, 10.0, 100.0, 1000.0].into_iter().enumerate() {
            let i = pick(k);
            println!(
                "{:>7} {:>6} {:>6} {:>11.0} {:>11.1} {:>12.0}  {}",
                if row == 0 {
                    len.to_string()
                } else {
                    String::new()
                },
                if row == 0 {
                    specs.len().to_string()
                } else {
                    String::new()
                },
                k as usize,
                build[i],
                exec[i],
                build[i] + k * exec[i],
                subjects[i].name
            );
        }

        let one = pick(1.0);
        println!(
            "{:>7} {:>6} {:>6} {:>11.0} {:>11.1} {:>12}  {}",
            "", "", "model", build[mi], exec[mi], "", subjects[mi].name
        );
        if one != mi && exec[mi] < exec[one] {
            let k = (build[mi] - build[one]) / (exec[one] - exec[mi]);
            println!(
                "{:>7} {:>6} {:>6} {:>11} {:>11} {:>12}  model repays its build cost after {:.0} executions",
                "", "", "", "", "", "", k.max(0.0)
            );
        } else if one == mi {
            println!(
                "{:>7} {:>6} {:>6} {:>11} {:>11} {:>12}  same recipe at k=1 and by cost model: nothing to choose",
                "", "", "", "", "", ""
            );
        }
        println!();
    }
}

// ---------------------------------------------------------------------------

enum Command {
    Time(Vec<String>),
    Regret(Vec<usize>),
    Sweep(Vec<usize>),
    Verify(Vec<usize>),
    Dump(Vec<usize>),
    Score(String),
    Explain(String),
    Costs(String),
    Plantime(Vec<usize>),
    Crossover(Vec<usize>),
}

fn run<T: FftNum + ToPrimitive, P: TunablePlanner<T>>(command: &Command, opts: &Options) {
    match command {
        Command::Time(specs) => cmd_time::<T, P>(specs, opts),
        Command::Regret(lengths) => cmd_regret::<T, P>(lengths, opts),
        Command::Sweep(lengths) => cmd_sweep::<T, P>(lengths, opts),
        Command::Verify(lengths) => cmd_verify::<T, P>(lengths, opts),
        Command::Dump(lengths) => cmd_dump::<T, P>(lengths, opts),
        Command::Plantime(lengths) => cmd_plantime::<T, P>(lengths, opts),
        Command::Crossover(lengths) => cmd_crossover::<T, P>(lengths, opts),
        Command::Score(path) => cmd_score(path, opts),
        Command::Costs(path) => cmd_costs(path, opts),
        Command::Explain(spec) => cmd_explain(spec, P::label(), opts),
    }
}

fn dispatch<T: FftNum + ToPrimitive>(planner: &str, command: &Command, opts: &Options) {
    match planner {
        "scalar" => run::<T, ScalarTuner<T>>(command, opts),
        // The manifest gives rustfft the SIMD feature matching the target, so architecture
        // alone decides which of these exists. A tool-crate `feature = ...` cfg would refer to
        // the tool's own features and always be false.
        #[cfg(target_arch = "aarch64")]
        "neon" => run::<T, rustfft::tuning::NeonTuner<T>>(command, opts),
        #[cfg(target_arch = "x86_64")]
        "sse" => run::<T, rustfft::tuning::SseTuner<T>>(command, opts),
        #[cfg(target_arch = "wasm32")]
        "wasm_simd" => run::<T, rustfft::tuning::WasmSimdTuner<T>>(command, opts),
        other => {
            eprintln!("unknown or unavailable planner '{}' on this build", other);
            std::process::exit(2);
        }
    }
}

/// The SIMD planner this build has, which is the default.
fn native_planner() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "neon"
    } else if cfg!(target_arch = "x86_64") {
        "sse"
    } else if cfg!(target_arch = "wasm32") {
        "wasm_simd"
    } else {
        "scalar"
    }
}

fn usage() -> ! {
    eprintln!("usage: planner_tuning <command> [options] ARGS...");
    eprintln!("commands: time SPEC... | regret LEN... | sweep LEN...|A..B | survey A..B");
    eprintln!("          verify LEN... | dump LEN... | score DUMP | costs DUMP | explain SPEC");
    eprintln!("          plantime LEN... | crossover LEN...");
    eprintln!(
        "  --planner NAME     scalar, neon, sse or wasm_simd (default: this build's SIMD one)"
    );
    eprintln!("  --f32              f32 instead of f64");
    eprintln!("  --rounds N         timing rounds per subject (default 9)");
    eprintln!("  --block-ms MS      wall-clock time per timed block (default 10)");
    eprintln!("  --cap N            max candidates per length when enumerating (default 48)");
    eprintln!("  --count N          survey: how many lengths (default 300)");
    eprintln!("  --seed N           survey: seed for picking lengths (default 1)");
    eprintln!("  --out FILE         dump: where to write (default dump.tsv)");
    eprintln!("  --verbose          regret: list the top candidates per length");
    eprintln!("  cost model weights, overriding the planner's defaults:");
    eprintln!("    --strided X --permuted X --rader-index X --radixn-extra X");
    eprintln!("    --general-row X --small-row X");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }

    let command_name = args[0].clone();
    let mut planner = native_planner().to_string();
    let mut opts = Options {
        rounds: 9,
        block_ms: 10.0,
        cap: 48,
        verbose: false,
        out: None,
        weights: Weights::default(),
        f32: false,
        count: 300,
        seed: 1,
    };
    let mut rest: Vec<String> = Vec::new();

    let mut i = 1;
    let value = |i: &mut usize, name: &str| -> String {
        *i += 1;
        args.get(*i)
            .unwrap_or_else(|| {
                eprintln!("{} wants a value", name);
                std::process::exit(2);
            })
            .clone()
    };
    let number = |text: String, name: &str| -> f64 {
        text.parse().unwrap_or_else(|_| {
            eprintln!("{} wants a number", name);
            std::process::exit(2);
        })
    };
    while i < args.len() {
        let arg = args[i].clone();
        match arg.as_str() {
            "--planner" => planner = value(&mut i, &arg),
            "--rounds" => opts.rounds = number(value(&mut i, &arg), &arg) as usize,
            "--block-ms" => opts.block_ms = number(value(&mut i, &arg), &arg),
            "--cap" => opts.cap = number(value(&mut i, &arg), &arg) as usize,
            "--count" => opts.count = number(value(&mut i, &arg), &arg) as usize,
            "--seed" => opts.seed = number(value(&mut i, &arg), &arg) as u64,
            "--out" => opts.out = Some(value(&mut i, &arg)),
            "--f32" => opts.f32 = true,
            "--verbose" => opts.verbose = true,
            "--strided" => opts.weights.strided = Some(number(value(&mut i, &arg), &arg)),
            "--permuted" => opts.weights.permuted = Some(number(value(&mut i, &arg), &arg)),
            "--rader-index" => opts.weights.rader_index = Some(number(value(&mut i, &arg), &arg)),
            "--radixn-extra" => opts.weights.radixn_extra = Some(number(value(&mut i, &arg), &arg)),
            "--general-row" => opts.weights.general_row = Some(number(value(&mut i, &arg), &arg)),
            "--small-row" => opts.weights.small_row = Some(number(value(&mut i, &arg), &arg)),
            other if other.starts_with("--") => {
                eprintln!("unknown option '{}'", other);
                usage();
            }
            other => rest.push(other.to_string()),
        }
        i += 1;
    }

    // Lengths may be a list, or `A..B` ranges, which is how a sweep takes a thousand of them.
    let lengths = |values: &[String]| -> Vec<usize> {
        let mut out = Vec::new();
        for value in values {
            match value.split_once("..") {
                Some((lo, hi)) => {
                    let lo: usize = lo.parse().expect("bad range start");
                    let hi: usize = hi.parse().expect("bad range end");
                    out.extend(lo..=hi);
                }
                None => out.push(value.parse().expect("lengths must be numbers")),
            }
        }
        out
    };
    let single =
        |values: &[String]| -> String { values.first().cloned().unwrap_or_else(|| usage()) };

    let command = match command_name.as_str() {
        "time" => Command::Time(rest.clone()),
        "regret" => Command::Regret(lengths(&rest)),
        "sweep" => Command::Sweep(lengths(&rest)),
        "survey" => {
            let range = single(&rest);
            let (lo, hi) = range.split_once("..").unwrap_or_else(|| usage());
            let (lo, hi) = (
                lo.parse().expect("bad range start"),
                hi.parse().expect("bad range end"),
            );
            println!(
                "# survey\t{} lengths from {}..{}, seed {}",
                opts.count, lo, hi, opts.seed
            );
            Command::Sweep(random_lengths(lo, hi, opts.count, opts.seed))
        }
        "verify" => Command::Verify(lengths(&rest)),
        "dump" => Command::Dump(lengths(&rest)),
        "score" => Command::Score(single(&rest)),
        "explain" => Command::Explain(single(&rest)),
        "costs" => Command::Costs(single(&rest)),
        "plantime" => Command::Plantime(lengths(&rest)),
        "crossover" => Command::Crossover(lengths(&rest)),
        other => {
            eprintln!("unknown command '{}'", other);
            usage();
        }
    };

    request_performance_core();

    if opts.f32 {
        dispatch::<f32>(&planner, &command, &opts);
    } else {
        dispatch::<f64>(&planner, &command, &opts);
    }
}
