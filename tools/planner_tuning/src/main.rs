//! Measurement tool for RustFFT's planners.
//!
//! Works against any planner that implements `TunablePlanner`, selected with `--planner`.
//! Recipes are built through the planner's own internals, so what is measured here is exactly
//! what the planner would construct.
//!
//! Subcommands:
//!   time SPEC...              time recipes against each other; a '*reps' suffix runs a recipe
//!                             over a len*reps buffer, which is how inner FFTs are invoked
//!   regret LEN...             measure how far the planner's pick is from the best available
//!   sweep LEN... | A..B      time the planner's pick against the counted model's pick, as TSV
//!   model TRAIN... 0 TEST...  calibrate a cost model and score its picks the same way
//!   residuals LEN...          show how per-element overhead varies with working set
//!   verify LEN...             check every enumerated candidate against a direct DFT
//!   emit LEN...               print the fitted model as a Rust source file

mod counted;
mod emit;
mod model;

use model::{bucket_of, overhead_scale, Model, FIT_ORDER};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{ToPrimitive, Zero};
use rustfft::tuning::{
    candidates_capped, parse, plan_candidates, to_spec_string, ScalarTuner, Spec,
    TunablePlanner,
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

fn median_of(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// Measure every primitive the model needs: each butterfly, and each valid Radix4 shape.
///
/// Each is timed over a buffer of at least 8192 elements, because primitives are almost always
/// invoked many times over a larger buffer rather than standalone, and a single cold call would
/// price in a startup cost they do not really pay in use.
fn calibrate_primitives<T: FftNum, P: TunablePlanner<T>>(
    rounds: usize,
    block_ms: f64,
    max_len: usize,
) -> Model {
    const REFERENCE_ELEMENTS: usize = 8192;
    let mut planner = P::new();
    let mut model = Model::default();

    let mut shapes: Vec<Option<(usize, u32)>> = Vec::new();
    let mut subjects: Vec<Subject<T>> = Vec::new();

    for len in P::butterfly_lens() {
        let spec = Spec::Butterfly(len);
        let fft = planner.build(&spec, FftDirection::Forward);
        let reps = (REFERENCE_ELEMENTS / len).max(1);
        shapes.push(None);
        subjects.push(Subject::new(format!("b{}", len), fft, reps));
    }

    for base in P::radix4_bases() {
        let mut k = 1u32;
        while base * (1usize << (2 * k)) <= max_len {
            let spec = Spec::Radix4 {
                k,
                base: Arc::new(Spec::Butterfly(base)),
            };
            let len = spec.len();
            let fft = planner.build(&spec, FftDirection::Forward);
            let reps = (REFERENCE_ELEMENTS / len).max(1);
            shapes.push(Some((base, k)));
            subjects.push(Subject::new(to_spec_string(&spec), fft, reps));
            k += 1;
        }
    }

    measure(&mut subjects, rounds, block_ms);

    for (shape, subject) in shapes.iter().zip(subjects.iter()) {
        match shape {
            Some(key) => {
                model.radix4.insert(*key, subject.best());
            }
            None => {
                model.butterfly.insert(subject.fft.len(), subject.best());
            }
        }
    }
    model
}

/// Fit one overhead curve for each composing algorithm.
///
/// Kinds are fitted in dependency order, since the residual of a MixedRadix that contains a
/// MixedRadixSmall only means anything once the Small's own overhead is known.
fn fit_overheads<T: FftNum, P: TunablePlanner<T>>(
    model: &mut Model,
    lengths: &[usize],
    rounds: usize,
    block_ms: f64,
    cap: usize,
    bucketed: bool,
) -> Vec<(Arc<Spec>, f64)> {
    let mut samples: Vec<(Arc<Spec>, f64)> = Vec::new();
    for &len in lengths {
        let mut planner = P::new();
        let specs = candidates_capped(&mut planner, len, cap);
        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();
        measure(&mut subjects, rounds, block_ms);
        for (spec, subject) in specs.iter().zip(subjects.iter()) {
            samples.push((Arc::clone(spec), subject.best()));
        }
    }

    let mut fitted: Vec<&'static str> = Vec::new();
    for target in FIT_ORDER {
        let usable: Vec<(u32, f64)> = samples
            .iter()
            .filter(|(spec, _)| spec.kind() == target && model.descendants_known(spec, &fitted))
            .filter_map(|(spec, measured)| {
                model
                    .inner_cost(spec)
                    .map(|inner| (bucket_of(spec), (measured - inner) / overhead_scale(spec)))
            })
            .collect();
        if usable.is_empty() {
            eprintln!("warning: no calibration samples for '{}'", target);
            continue;
        }

        // One median per log2 bucket, but only for buckets with enough samples to mean anything.
        // Sparse buckets are dropped and filled in by interpolation instead.
        let mut table: Vec<(u32, f64)> = Vec::new();
        if bucketed {
            let mut by_bucket: std::collections::BTreeMap<u32, Vec<f64>> = Default::default();
            for (bucket, residual) in usable.iter() {
                by_bucket.entry(*bucket).or_default().push(*residual);
            }
            const MIN_PER_BUCKET: usize = 3;
            table = by_bucket
                .iter()
                .filter(|(_, values)| values.len() >= MIN_PER_BUCKET)
                .map(|(bucket, values)| (*bucket, median_of(values.clone())))
                .collect();
        }

        // Too little data to describe a curve, or curves not wanted, so use one constant.
        if table.len() < 2 {
            table = vec![(0, median_of(usable.iter().map(|(_, r)| *r).collect()))];
        }

        let rendered: Vec<String> = table
            .iter()
            .map(|(bucket, value)| format!("{}:{:.2}", 1usize << bucket, value))
            .collect();
        println!(
            "  {:<5} {:>3} buckets from {:>4} samples   {}",
            target,
            table.len(),
            usable.len(),
            rendered.join(" ")
        );
        model.overhead.insert(target, table);
        fitted.push(target);
    }
    samples
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

struct Options {
    rounds: usize,
    block_ms: f64,
    cap: usize,
    verbose: bool,
    bucketed: bool,
    /// Where `dump` writes its rows.
    out: Option<String>,
    /// Weights for the counted model.
    params: counted::Params,
    backend_explicit: bool,
}

fn cmd_time<T: FftNum, P: TunablePlanner<T>>(specs: &[String], opts: &Options) {
    let mut planner = P::new();
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

fn cmd_regret<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    println!(
        "{:>9} {:>8} {:>10} {:>10}  {}",
        "len", "regret", "planner ns", "best ns", "best recipe (when it differs)"
    );

    let mut regrets: Vec<(f64, usize, String, String)> = Vec::new();

    for &len in lengths {
        let mut planner = P::new();
        let specs = candidates_capped(&mut planner, len, opts.cap);
        let planner_spec = to_spec_string(&specs[0]);

        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();

        measure(&mut subjects, opts.rounds, opts.block_ms);

        let planner_time = subjects[0].best();
        let best_index = subjects
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.best().partial_cmp(&b.1.best()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let best_time = subjects[best_index].best();
        let best_spec = subjects[best_index].name.clone();
        let regret = planner_time / best_time;

        println!(
            "{:>9} {:>7.3}x {:>10.0} {:>10.0}  {}",
            len,
            regret,
            planner_time,
            best_time,
            if best_index == 0 {
                "= planner".to_string()
            } else {
                best_spec.clone()
            }
        );
        if opts.verbose {
            let mut ranked: Vec<&Subject<T>> = subjects.iter().collect();
            ranked.sort_by(|a, b| a.best().partial_cmp(&b.best()).unwrap());
            for subject in ranked.iter().take(6) {
                println!(
                    "            {:>6.3}x  {}",
                    subject.best() / best_time,
                    subject.name
                );
            }
            println!("            ({} candidates measured)", subjects.len());
        }

        regrets.push((regret, len, planner_spec, best_spec));
    }

    regrets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = regrets.len();
    let mean = regrets.iter().map(|r| r.0).sum::<f64>() / n as f64;
    println!("\n--- regret: planner's pick divided by the best recipe measured ---");
    println!("  lengths  {}", n);
    println!("  mean     {:.4}", mean);
    println!("  median   {:.4}", regrets[n / 2].0);
    println!("  p90      {:.4}", regrets[(n * 9) / 10].0);
    println!("  worst    {:.4}", regrets[n - 1].0);
    let losing = regrets.iter().filter(|r| r.0 > 1.02).count();
    println!("  more than 2% off the best: {} of {} lengths", losing, n);
    println!("\nworst offenders:");
    for (regret, len, planner_spec, best_spec) in regrets.iter().rev().take(10) {
        println!("  {:>8}  {:.3}x", len, regret);
        println!("      planner: {}", planner_spec);
        println!("      best:    {}", best_spec);
    }
}

/// Time the shipping planner's pick against the counted model's pick, at every length in a range.
///
/// Unlike `regret`, this builds and times only two recipes per length rather than the whole
/// candidate set, which is what makes a thousand-length sweep affordable. When both agree, the
/// recipe is timed once and reported for both, so an agreement shows as exactly 1.000 rather than
/// as timing noise.
///
/// Output is TSV on stdout, one row per length, ready to plot.
fn cmd_sweep<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    let model = counted::CountedModel::new(opts.params);

    println!("# planner\t{}", P::label());
    println!("# params\t{:?}", opts.params);
    println!("# rounds\t{}\tblock_ms\t{}\tcap\t{}", opts.rounds, opts.block_ms, opts.cap);
    println!(
        "len\tcands\tagree\tplanner_ns\tmodel_ns\tratio\tplanner_norm\tmodel_norm\tplanner_spec\tmodel_spec"
    );

    for &len in lengths {
        let mut planner = P::new();
        let specs = plan_candidates(&mut planner, len, opts.cap);
        if specs.is_empty() {
            eprintln!("len {}: no candidates", len);
            continue;
        }

        let model_index = specs
            .iter()
            .enumerate()
            .filter_map(|(i, spec)| model.cost(spec).map(|c| (i, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let agree = model_index == 0;
        let planner_spec = to_spec_string(&specs[0]);
        let model_spec = to_spec_string(&specs[model_index]);

        let mut subjects: Vec<Subject<T>> = if agree {
            vec![Subject::new(
                planner_spec.clone(),
                planner.build(&specs[0], FftDirection::Forward),
                1,
            )]
        } else {
            vec![
                Subject::new(
                    planner_spec.clone(),
                    planner.build(&specs[0], FftDirection::Forward),
                    1,
                ),
                Subject::new(
                    model_spec.clone(),
                    planner.build(&specs[model_index], FftDirection::Forward),
                    1,
                ),
            ]
        };
        measure(&mut subjects, opts.rounds, opts.block_ms);

        let planner_ns = subjects[0].best();
        let model_ns = if agree { planner_ns } else { subjects[1].best() };

        // n log2 n, the work a radix-2 FFT of this length would do. Undefined at len 1, where
        // there is no work to normalise by.
        let nlogn = (len as f64) * (len as f64).log2();
        let norm = |ns: f64| if nlogn > 0.0 { ns / nlogn } else { f64::NAN };

        println!(
            "{}\t{}\t{}\t{:.2}\t{:.2}\t{:.4}\t{:.5}\t{:.5}\t{}\t{}",
            len,
            specs.len(),
            if agree { 1 } else { 0 },
            planner_ns,
            model_ns,
            planner_ns / model_ns,
            norm(planner_ns),
            norm(model_ns),
            planner_spec,
            model_spec
        );
    }
}

/// Check that every enumerated candidate actually computes a correct FFT.
///
/// Timing a recipe says nothing about whether it is valid, and an invalid one would happily
/// produce fast wrong answers. Each candidate is compared against a direct DFT.
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

        let mut planner = P::new();
        let specs = candidates_capped(&mut planner, len, opts.cap);
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
                    let d = Complex::new(
                        a.re.to_f64().unwrap() - b.re,
                        a.im.to_f64().unwrap() - b.im,
                    );
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

/// Show how each algorithm's per-element overhead varies with size.
fn cmd_residuals<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    let max_len = lengths.iter().copied().max().unwrap_or(1024) * 4;
    eprintln!("measuring primitives...");
    let mut model = calibrate_primitives::<T, P>(opts.rounds, opts.block_ms, max_len);
    eprintln!("fitting overheads...");
    let samples = fit_overheads::<T, P>(
        &mut model,
        lengths,
        opts.rounds,
        opts.block_ms,
        opts.cap,
        opts.bucketed,
    );

    let mut binned: std::collections::BTreeMap<
        &'static str,
        std::collections::BTreeMap<u32, Vec<f64>>,
    > = Default::default();
    for (spec, measured) in samples.iter() {
        if !FIT_ORDER.contains(&spec.kind()) {
            continue;
        }
        if let Some(inner) = model.inner_cost(spec) {
            let residual = (measured - inner) / overhead_scale(spec);
            binned
                .entry(spec.kind())
                .or_default()
                .entry(bucket_of(spec))
                .or_default()
                .push(residual);
        }
    }

    println!(
        "{:<5} {:>8} {:>8} {:>9} {:>7}",
        "kind", "len>=", "median", "p25..p75", "n"
    );
    for (kind, buckets) in binned {
        for (bucket, values) in buckets {
            if values.len() < 3 {
                continue;
            }
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            println!(
                "{:<5} {:>8} {:>8.3} {:>9} {:>7}",
                kind,
                1usize << bucket,
                sorted[n / 2],
                format!("{:.2}..{:.2}", sorted[n / 4], sorted[(n * 3) / 4]),
                n
            );
        }
        println!();
    }
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
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("cannot create output"));
    writeln!(f, "# planner\t{}", P::label()).unwrap();
    writeln!(f, "# rounds\t{}\tblock_ms\t{}\tcap\t{}", opts.rounds, opts.block_ms, opts.cap).unwrap();
    writeln!(f, "len\tspec\tns\tpass\tplanner_pick").unwrap();

    for &len in lengths {
        let mut planner = P::new();
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
        order.sort_by(|a, b| subjects[*a].best().partial_cmp(&subjects[*b].best()).unwrap());
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

/// Score the counted model against a dump file. No measurement, no planner, no machine.
fn cmd_score(path: &str, opts: &Options) {
    use std::collections::BTreeMap;
    let text = std::fs::read_to_string(path).expect("cannot read dump");

    // len -> [(spec, best ns, is planner pick)]. Pass 2 overwrites pass 1.
    //
    // The inner container must preserve the dump's own order, which is the order
    // `candidates_capped` enumerated in. The planner takes the first minimum in that order, so
    // replay has to break cost ties the same way or it does not model the planner. Keying by
    // spec string instead sorts `gts(b10,b3)` ahead of `gts(b3,b10)`, which is the opposite of
    // the enumeration order and silently reverses every width/height tie.
    let mut data: BTreeMap<usize, Vec<(String, f64, bool)>> = BTreeMap::new();
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
        let e: &mut Vec<(String, f64, bool)> = data.entry(len).or_default();
        match e.iter_mut().find(|r| r.0 == spec) {
            // pass 2 is the careful re-timing, so it wins
            Some(row) if pass != 1 => {
                row.1 = ns;
                row.2 = pick;
            }
            Some(_) => {}
            None => e.push((spec, ns, pick)),
        }
    }

    // Take the backend from the dump header unless it was given explicitly, so an SSE dump is
    // never priced with NEON instruction costs by accident.
    let mut params = opts.params;
    if !opts.backend_explicit {
        if let Some(label) = text
            .lines()
            .find_map(|l| l.strip_prefix("# planner\t"))
            .and_then(counted::Backend::parse)
        {
            params.backend = label;
        }
    }
    let model = counted::CountedModel::new(params);
    println!("params: {:?}", params);
    println!(
        "{:>9} {:>9} {:>9}   {}",
        "len", "counted", "planner", "counted model's pick (when it is not the best)"
    );

    let (mut mr, mut pr) = (Vec::new(), Vec::new());
    for (&len, rows) in &data {
        let best = rows.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
        let planner_ns = rows.iter().find(|v| v.2).map(|v| v.1);

        let mut scored: Vec<(String, f64, f64)> = Vec::new();
        for (spec_text, ns, _) in rows {
            let spec = match parse(spec_text) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if let Some(c) = model.cost(&spec) {
                scored.push((spec_text.clone(), c, *ns));
            }
        }
        let pick = scored
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (pick_name, pick_ns) = match pick {
            Some((n, _, ns)) => (n.clone(), *ns),
            None => {
                println!("{:>9}  no candidate priced", len);
                continue;
            }
        };
        let m = pick_ns / best;
        mr.push(m);
        let p = planner_ns.map(|n| n / best);
        if let Some(p) = p {
            pr.push(p);
        }
        let shown = if m <= 1.0001 { "= best".to_string() } else { pick_name };
        println!(
            "{:>9} {:>8.3}x {:>8}   {}",
            len,
            m,
            p.map(|p| format!("{:.3}x", p)).unwrap_or_else(|| "-".into()),
            shown
        );
    }

    let stat = |v: &mut Vec<f64>, label: &str| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let median = v[v.len() / 2];
        let p90 = v[((v.len() as f64 * 0.9) as usize).min(v.len() - 1)];
        let worst = *v.last().unwrap();
        println!(
            "  {:<9} n={:<4} mean {:.4}  median {:.4}  p90 {:.4}  worst {:.4}",
            label,
            v.len(),
            mean,
            median,
            p90,
            worst
        );
    };
    println!("\n--- regret, pick divided by best measured ---");
    stat(&mut mr, "counted");
    stat(&mut pr, "planner");
}

/// Print the counted model's cost tree for one recipe, so the recursion can be checked by eye.
fn cmd_explain(spec_text: &str, opts: &Options) {
    let spec = parse(spec_text).expect("could not parse spec");
    let model = counted::CountedModel::new(opts.params);
    let root_ws = spec.len() as f64;

    fn walk(
        model: &counted::CountedModel,
        spec: &Spec,
        root_ws: f64,
        mult: f64,
        depth: usize,
        out: &mut Vec<String>,
    ) {
        let own = model.cost(spec).unwrap_or(f64::NAN);
        // cost of the children alone, at the multiplicity the parent runs them
        let (kids, child_total): (Vec<(&Spec, f64)>, f64) = match spec {
            Spec::MixedRadix { left, right, .. } | Spec::GoodThomas { left, right, .. } => {
                let v = vec![
                    (left.as_ref(), right.len() as f64),
                    (right.as_ref(), left.len() as f64),
                ];
                let t = v
                    .iter()
                    .map(|(c, m)| m * model.cost(c).unwrap_or(0.0))
                    .sum();
                (v, t)
            }
            Spec::RadixN { radixes, base } => {
                let m = radixes.iter().product::<usize>() as f64;
                (
                    vec![(base.as_ref(), m)],
                    m * model.cost(base).unwrap_or(0.0),
                )
            }
            Spec::Radix4 { k, base } => {
                let m = (1u64 << (2 * k)) as f64;
                (
                    vec![(base.as_ref(), m)],
                    m * model.cost(base).unwrap_or(0.0),
                )
            }
            Spec::Raders { inner } | Spec::Bluesteins { inner, .. } => {
                (vec![(inner.as_ref(), 2.0)], 2.0 * model.cost(inner).unwrap_or(0.0))
            }
            _ => (vec![], 0.0),
        };
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
            walk(model, child, root_ws, mult * m, depth + 1, out);
        }
    }

    let mut out = Vec::new();
    walk(&model, &spec, root_ws, 1.0, 0, &mut out);
    println!("params: {:?}", opts.params);
    println!("{:<36} {:>11}  {:<9} {:>19} {:>16}", "recipe", "len", "times", "total cost", "own cost");
    for line in out {
        println!("{}", line);
    }
}

/// Print the model's cost for every measured candidate, for offline analysis.
///
/// Columns: len, spec, measured ns, model cost. Pure replay, no machine needed.
fn cmd_costs(path: &str, opts: &Options) {
    use std::collections::BTreeMap;
    let text = std::fs::read_to_string(path).expect("cannot read dump");
    let mut params = opts.params;
    if !opts.backend_explicit {
        if let Some(b) = text
            .lines()
            .find_map(|l| l.strip_prefix("# planner\t"))
            .and_then(counted::Backend::parse)
        {
            params.backend = b;
        }
    }
    let model = counted::CountedModel::new(params);

    let mut data: BTreeMap<usize, BTreeMap<String, (f64, bool)>> = BTreeMap::new();
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
        let e = data.entry(len).or_default();
        match e.get(&spec) {
            Some(_) if pass == 1 => {}
            _ => {
                e.insert(spec, (ns, pick));
            }
        }
    }

    println!("len\tspec\tns\tcost\tplanner_pick");
    for (len, rows) in &data {
        for (spec_text, (ns, pick)) in rows {
            let cost = parse(spec_text)
                .ok()
                .and_then(|s| model.cost(&s))
                .map(|c| format!("{:.1}", c))
                .unwrap_or_else(|| "NA".into());
            println!(
                "{}\t{}\t{:.3}\t{}\t{}",
                len,
                spec_text,
                ns,
                cost,
                if *pick { 1 } else { 0 }
            );
        }
    }
}

/// Compare the plan-time cost of the fixed planner against enumerate-and-price.
///
/// The fixed planner answers from a few integer operations. A cost model has to enumerate the
/// candidate set and price every member, which is real work the fixed planner never does. This
/// is the one axis where the fixed planner is unambiguously ahead, so it should be measured.
/// Compare the cost of *choosing* a recipe against the cost of *building* it.
///
/// Planning only produces a `Recipe`. Turning that into an `Arc<dyn Fft>` is a separate and much
/// larger job: every algorithm precomputes twiddles, and Rader's and Bluestein's both run a full
/// inner FFT inside their constructors. So the question that decides whether an estimating
/// planner is affordable is not how much slower it is than the fixed planner, but how much it
/// adds to plan-plus-build, which is what a caller actually pays before the first transform.
/// How the best recipe changes once construction cost is counted, which is the question a
/// quick-and-dirty planner for one-shot transforms exists to answer.
///
/// For every candidate this measures build time and execution time, then reports which recipe
/// minimises `build + k * execute` at several values of `k`. If the same recipe wins at every `k`
/// there is nothing for a construction-aware planner to choose, and the idea is dead. Plan time is
/// deliberately excluded: it is the same constant for every candidate at one length, so it cannot
/// change which recipe wins.
fn cmd_crossover<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    let model = counted::CountedModel::new(opts.params);
    println!(
        "{:>7} {:>6} {:>6} {:>11} {:>11} {:>12}  {}",
        "len", "cands", "k", "build ns", "exec ns", "total ns", "recipe"
    );

    for &len in lengths {
        let mut planner = P::new();
        let specs = plan_candidates(&mut planner, len, opts.cap);
        if specs.is_empty() {
            eprintln!("len {}: no candidates", len);
            continue;
        }

        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();
        measure(&mut subjects, opts.rounds, opts.block_ms);
        let exec: Vec<f64> = subjects.iter().map(|s| s.best()).collect();

        // A fresh planner per repetition, so no inner FFT is served from a cache.
        let build_reps = if len > 4096 { 5 } else if len > 256 { 20 } else { 100 };
        let build: Vec<f64> = specs
            .iter()
            .map(|spec| {
                let t = Instant::now();
                for _ in 0..build_reps {
                    let mut pl = P::new();
                    std::hint::black_box(pl.build(spec, FftDirection::Forward));
                }
                t.elapsed().as_secs_f64() * 1e9 / build_reps as f64
            })
            .collect();

        let pick = |k: f64| -> usize {
            (0..specs.len())
                .min_by(|&a, &b| {
                    (build[a] + k * exec[a])
                        .partial_cmp(&(build[b] + k * exec[b]))
                        .unwrap()
                })
                .unwrap()
        };

        let mut first = true;
        for k in [1.0, 10.0, 100.0, 1000.0] {
            let i = pick(k);
            println!(
                "{:>7} {:>6} {:>6} {:>11.0} {:>11.1} {:>12.0}  {}",
                if first { len.to_string() } else { String::new() },
                if first { specs.len().to_string() } else { String::new() },
                k as usize,
                build[i],
                exec[i],
                build[i] + k * exec[i],
                subjects[i].name
            );
            first = false;
        }

        // What the counted model picks, which optimises execution alone.
        let mi = specs
            .iter()
            .enumerate()
            .filter_map(|(i, sp)| model.cost(sp).map(|c| (i, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let one = pick(1.0);
        println!(
            "{:>7} {:>6} {:>6} {:>11.0} {:>11.1} {:>12}  {}",
            "", "", "model", build[mi], exec[mi], "", subjects[mi].name
        );
        // Crossover: how many executions before the model's pick repays its extra build cost.
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

fn cmd_plantime<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    let model = counted::CountedModel::new(opts.params);
    println!(
        "{:>7} {:>6} {:>12} {:>14} {:>12} {:>9} {:>11}",
        "len", "cands", "plan fixed", "plan+price", "build", "build/plan", "extra vs"
    );
    println!(
        "{:>7} {:>6} {:>12} {:>14} {:>12} {:>9} {:>11}",
        "", "", "ns", "ns", "ns", "", "plan+build"
    );
    let (mut tot_a, mut tot_b, mut tot_c) = (0.0, 0.0, 0.0);
    for &len in lengths {
        // fixed planner: design only, with a fresh planner each time so nothing is cached
        let reps = 200;
        let t0 = Instant::now();
        for _ in 0..reps {
            let mut pl = P::new();
            std::hint::black_box(pl.plan(len));
        }
        let a = t0.elapsed().as_secs_f64() * 1e9 / reps as f64;

        let mut pl = P::new();
        let n = plan_candidates(&mut pl, len, opts.cap).len();
        let t1 = Instant::now();
        for _ in 0..reps {
            let mut pl = P::new();
            let specs = plan_candidates(&mut pl, len, opts.cap);
            let best = specs
                .iter()
                .filter_map(|sp| model.cost(sp).map(|c| (c, sp)))
                .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            std::hint::black_box(best);
        }
        let b = t1.elapsed().as_secs_f64() * 1e9 / reps as f64;

        // Build the recipe the fixed planner chose. Construction allocates and precomputes, so
        // it is far slower than planning; use fewer repetitions and a fresh planner each time so
        // nothing is served from a cache.
        let mut pl = P::new();
        let spec = pl.plan(len);
        let build_reps = if len > 4096 { 5 } else if len > 256 { 20 } else { 100 };
        let t2 = Instant::now();
        for _ in 0..build_reps {
            let mut pl = P::new();
            std::hint::black_box(pl.build(&spec, FftDirection::Forward));
        }
        let c = t2.elapsed().as_secs_f64() * 1e9 / build_reps as f64;

        tot_a += a;
        tot_b += b;
        tot_c += c;
        println!(
            "{:>7} {:>6} {:>12.0} {:>14.0} {:>12.0} {:>8.0}x {:>10.1}%",
            len, n, a, b, c, c / a, 100.0 * (b - a) / (a + c)
        );
    }
    println!(
        "\n  totals: plan fixed {:.0} ns, plan+price {:.0} ns ({:.1}x), build {:.0} ns",
        tot_a, tot_b, tot_b / tot_a, tot_c
    );
    println!(
        "  building is {:.0}x planning; the estimating planner adds {:.2}% to plan-plus-build",
        tot_c / tot_a,
        100.0 * (tot_b - tot_a) / (tot_a + tot_c)
    );
}

fn cmd_model<T: FftNum, P: TunablePlanner<T>>(train: &[usize], test: &[usize], opts: &Options) {
    let max_len = test.iter().chain(train.iter()).copied().max().unwrap_or(1024) * 4;

    println!("planner: {}", P::label());
    println!("measuring primitives...");
    let mut model = calibrate_primitives::<T, P>(opts.rounds, opts.block_ms, max_len);

    println!("fitting overheads on {} training lengths...", train.len());
    fit_overheads::<T, P>(
        &mut model,
        train,
        opts.rounds,
        opts.block_ms,
        opts.cap,
        opts.bucketed,
    );
    println!("\n{}", model.describe());

    println!(
        "{:>9} {:>9} {:>9}   {}",
        "len", "model", "planner", "model's pick (when it is not the best)"
    );
    let mut model_regrets = Vec::new();
    let mut planner_regrets = Vec::new();

    for &len in test {
        let mut planner = P::new();
        let specs = candidates_capped(&mut planner, len, opts.cap);
        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();
        measure(&mut subjects, opts.rounds, opts.block_ms);

        // Fastest of the first pass, used only to pick which recipes deserve a careful re-timing.
        let best_index = subjects
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.best().partial_cmp(&b.1.best()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let model_index = specs
            .iter()
            .enumerate()
            .filter_map(|(i, spec)| model.cost(spec).map(|c| (i, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i);

        let model_spec = match model_index {
            Some(i) => subjects[i].name.clone(),
            None => "<no candidate priced>".to_string(),
        };

        // Second pass. The winner of a wide comparison is biased fast: with sub-percent noise,
        // the minimum of many draws lands below the true minimum, which puts a floor under any
        // regret measured against it. Re-timing just the recipes of interest, for longer,
        // removes most of that bias.
        let finalists: Vec<usize> = {
            let mut picked = vec![0usize, best_index];
            if let Some(i) = model_index {
                picked.push(i);
            }
            picked.sort_unstable();
            picked.dedup();
            picked
        };
        let mut finals: Vec<Subject<T>> = finalists
            .iter()
            .map(|&i| {
                let fft = planner.build(&specs[i], FftDirection::Forward);
                Subject::new(to_spec_string(&specs[i]), fft, 1)
            })
            .collect();
        measure(&mut finals, opts.rounds * 4, opts.block_ms);

        let time_of =
            |index: usize| -> f64 { finals[finalists.iter().position(|&i| i == index).unwrap()].best() };
        let planner_time = time_of(0);
        let best_time = finalists
            .iter()
            .map(|&i| time_of(i))
            .fold(f64::INFINITY, f64::min);
        let model_time = match model_index {
            Some(i) => time_of(i),
            None => f64::NAN,
        };

        let model_regret = model_time / best_time;
        let planner_regret = planner_time / best_time;
        model_regrets.push(model_regret);
        planner_regrets.push(planner_regret);

        println!(
            "{:>9} {:>8.3}x {:>8.3}x   {}",
            len,
            model_regret,
            planner_regret,
            if model_regret <= 1.001 {
                "= best".to_string()
            } else {
                model_spec
            }
        );
    }

    for (label, mut values) in [("model  ", model_regrets), ("planner", planner_regrets)] {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        println!(
            "{}: mean {:.4}  median {:.4}  p90 {:.4}  worst {:.4}",
            label,
            mean,
            values[n / 2],
            values[(n * 9) / 10],
            values[n - 1]
        );
    }
}

fn cmd_emit<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options, element: &str) {
    let max_len = lengths.iter().copied().max().unwrap_or(1024) * 4;
    eprintln!("measuring primitives...");
    let mut model = calibrate_primitives::<T, P>(opts.rounds, opts.block_ms, max_len);
    eprintln!("fitting overheads on {} lengths...", lengths.len());
    fit_overheads::<T, P>(
        &mut model,
        lengths,
        opts.rounds,
        opts.block_ms,
        opts.cap,
        opts.bucketed,
    );
    print!("{}", emit::emit(&model, element, P::label()));
}

// ---------------------------------------------------------------------------

enum Command {
    Time(Vec<String>),
    Regret(Vec<usize>),
    Model(Vec<usize>, Vec<usize>),
    Residuals(Vec<usize>),
    Verify(Vec<usize>),
    Emit(Vec<usize>),
    Dump(Vec<usize>),
    Score(String),
    Explain(String),
    Costs(String),
    Plantime(Vec<usize>),
    Crossover(Vec<usize>),
    Sweep(Vec<usize>),
}

fn run<T: FftNum + ToPrimitive, P: TunablePlanner<T>>(
    command: &Command,
    opts: &Options,
    element: &str,
) {
    match command {
        Command::Time(specs) => cmd_time::<T, P>(specs, opts),
        Command::Regret(lengths) => cmd_regret::<T, P>(lengths, opts),
        Command::Sweep(lengths) => cmd_sweep::<T, P>(lengths, opts),
        Command::Model(train, test) => cmd_model::<T, P>(train, test, opts),
        Command::Residuals(lengths) => cmd_residuals::<T, P>(lengths, opts),
        Command::Verify(lengths) => cmd_verify::<T, P>(lengths, opts),
        Command::Emit(lengths) => cmd_emit::<T, P>(lengths, opts, element),
        Command::Dump(lengths) => cmd_dump::<T, P>(lengths, opts),
        Command::Score(path) => cmd_score(path, opts),
        Command::Explain(spec) => cmd_explain(spec, opts),
        Command::Costs(path) => cmd_costs(path, opts),
        Command::Plantime(l) => cmd_plantime::<T, P>(l, opts),
        Command::Crossover(l) => cmd_crossover::<T, P>(l, opts),
    }
}

fn dispatch<T: FftNum + ToPrimitive>(planner: &str, command: &Command, opts: &Options, el: &str) {
    match planner {
        "scalar" => run::<T, ScalarTuner<T>>(command, opts, el),
        // The manifest gives rustfft the SIMD feature matching the target, so architecture
        // alone decides which of these exists. A tool-crate `feature = ...` cfg would refer to
        // the tool's own features and always be false.
        #[cfg(target_arch = "aarch64")]
        "neon" => run::<T, rustfft::tuning::NeonTuner<T>>(command, opts, el),
        #[cfg(target_arch = "x86_64")]
        "sse" => run::<T, rustfft::tuning::SseTuner<T>>(command, opts, el),
        #[cfg(target_arch = "wasm32")]
        "wasm_simd" => run::<T, rustfft::tuning::WasmSimdTuner<T>>(command, opts, el),
        other => {
            eprintln!(
                "unknown or unavailable planner '{}' on this build; try 'scalar'",
                other
            );
            std::process::exit(2);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: planner_tuning <command> [options] ARGS...");
        eprintln!("commands: time SPEC... | regret LEN... | model TRAIN... 0 TEST...");
        eprintln!("          residuals LEN... | verify LEN... | emit LEN...");
        eprintln!("  --planner NAME  scalar (default), neon, sse");
        eprintln!("  --rounds N      timing rounds per subject (default 9)");
        eprintln!("  --block-ms MS   wall-clock time per timed block (default 10)");
        eprintln!("  --cap N         max candidates per length (default 48)");
        eprintln!("  --f32           measure f32 instead of f64");
        eprintln!("  --permuted-vector  charge permuted passes per vector, not per element");
        eprintln!("  --bucketed      fit overhead curves over working set, not constants");
        eprintln!("  --verbose       for 'regret', list the top candidates per length");
        std::process::exit(2);
    }

    let command_name = args[0].clone();
    let mut planner = "scalar".to_string();
    let mut opts = Options {
        rounds: 9,
        block_ms: 10.0,
        cap: 48,
        verbose: false,
        bucketed: false,
        out: None,
        params: counted::Params::default(),
        backend_explicit: false,
    };
    let mut f32_mode = false;
    let mut rest: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--planner" => {
                i += 1;
                planner = args[i].clone();
            }
            "--rounds" => {
                i += 1;
                opts.rounds = args[i].parse().expect("--rounds wants a number");
            }
            "--block-ms" => {
                i += 1;
                opts.block_ms = args[i].parse().expect("--block-ms wants a number");
            }
            "--cap" => {
                i += 1;
                opts.cap = args[i].parse().expect("--cap wants a number");
            }
            "--f32" => f32_mode = true,
            "--bucketed" => opts.bucketed = true,
            "--out" => {
                i += 1;
                opts.out = Some(args[i].clone());
            }
            "--seq-l1" => { i += 1; opts.params.seq[0] = args[i].parse().unwrap(); }
            "--seq-l2" => { i += 1; opts.params.seq[1] = args[i].parse().unwrap(); }
            "--seq-dram" => { i += 1; opts.params.seq[2] = args[i].parse().unwrap(); }
            "--strided" => { i += 1; opts.params.strided_mult = args[i].parse().unwrap(); }
            "--permuted" => { i += 1; opts.params.permuted_mult = args[i].parse().unwrap(); }
            "--rader-index" => { i += 1; opts.params.rader_index = args[i].parse().unwrap(); }
            "--radixn-extra" => { i += 1; opts.params.radixn_extra = args[i].parse().unwrap(); }
            "--mul-complex" => { i += 1; opts.params.mul_complex = args[i].parse().unwrap(); }
            "--spill" => { i += 1; opts.params.spill = args[i].parse().unwrap(); }
            "--general-row" => { i += 1; opts.params.general_row = args[i].parse().unwrap(); }
            "--small-row" => { i += 1; opts.params.small_row = args[i].parse().unwrap(); }
            "--permuted-vector" => opts.params.permuted_scalar = false,
            "--f64" => opts.params.elem = counted::Elem::F64,
            "--backend" => { i += 1; opts.params.backend = counted::Backend::parse(&args[i]).expect("--backend wants neon or sse"); opts.backend_explicit = true; }
            "--l1-elems" => { i += 1; opts.params.l1_elems = args[i].parse().unwrap(); }
            "--l2-elems" => { i += 1; opts.params.l2_elems = args[i].parse().unwrap(); }
            "--verbose" => opts.verbose = true,
            other => rest.push(other.to_string()),
        }
        i += 1;
    }

    let numbers = |values: &[String]| -> Vec<usize> {
        values
            .iter()
            .map(|s| s.parse().expect("lengths must be numbers"))
            .collect()
    };

    // `sweep` takes a thousand lengths, so accept "A..B" as well as a list.
    let range_or_numbers = |values: &[String]| -> Vec<usize> {
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

    let command = match command_name.as_str() {
        "time" => Command::Time(rest.clone()),
        "regret" => Command::Regret(numbers(&rest)),
        "sweep" => Command::Sweep(range_or_numbers(&rest)),
        "residuals" => Command::Residuals(numbers(&rest)),
        "verify" => Command::Verify(numbers(&rest)),
        "emit" => Command::Emit(numbers(&rest)),
        "dump" => Command::Dump(numbers(&rest)),
        "score" => Command::Score(rest[0].clone()),
        "explain" => Command::Explain(rest[0].clone()),
        "costs" => Command::Costs(rest[0].clone()),
        "plantime" => Command::Plantime(numbers(&rest)),
        "crossover" => Command::Crossover(numbers(&rest)),
        "model" => {
            let lengths = numbers(&rest);
            let split = lengths
                .iter()
                .position(|&l| l == 0)
                .expect("model wants TRAIN... 0 TEST...");
            let (train, test) = lengths.split_at(split);
            Command::Model(train.to_vec(), test[1..].to_vec())
        }
        other => {
            eprintln!("unknown command '{}'", other);
            std::process::exit(2);
        }
    };

    request_performance_core();

    if f32_mode {
        opts.params.elem = counted::Elem::F32;
        dispatch::<f32>(&planner, &command, &opts, "f32");
    } else {
        dispatch::<f64>(&planner, &command, &opts, "f64");
    }
}
