# planner_tuning

A measurement harness for RustFFT's SIMD planners. Not part of the library and not shipped: it
exists to fit the estimating planner's cost model, and to check how its picks compare against the
fixed planner it replaces and against the fastest recipe available.

It drives the planners through the library's non-default `tuning` feature, so recipes are built
by the planners' own code and priced by the library's own cost model. What gets timed is exactly
what a planner would construct.

## Vocabulary

A **recipe** is a `Spec` tree, printed compactly: `mr(b11,rad(mrs(b6,b10)))` is a MixedRadix of
butterfly 11 against Rader's over a MixedRadixSmall of 6 and 10. The prefixes are `b` butterfly,
`r4` Radix4, `rn` RadixN, `mr`/`mrs` MixedRadix and its Small variant, `gt`/`gts` Good-Thomas and
its Small variant, `rad` Rader's, `bs` Bluestein's.

**planner** in the output is the fixed planner, which the SIMD planners keep behind the `tuning`
feature while the estimating planner is a draft. **model** is the estimating planner.

**Regret** is a pick divided by the best measured candidate at that length: 1.000 is optimal,
1.05 is five percent off. It is a *lower bound* on the distance from optimal, because the
candidate set is finite.

## Comparing the two planners

`sweep` times the fixed planner's pick against the estimating planner's at every length given.
It builds only those two recipes per length, which is what makes a thousand lengths affordable.
Where the two agree the recipe is timed once and reported for both, so agreement reads as exactly
1.000 rather than as noise.

`survey` is the same over lengths drawn at random from a range, which is how to cover sizes up
to a million. Both end with a summary of the estimating planner's runtime over the fixed
planner's: geometric mean and percentiles, so below 1 means faster.

```sh
cargo build --release
./target/release/planner_tuning sweep 1..1000 > sweep_neon_f64.tsv
./target/release/planner_tuning sweep --f32 1..1000 > sweep_neon_f32.tsv
./target/release/planner_tuning survey --count 300 --seed 1 1..1000000 > survey_neon_f64.tsv
```

The planner defaults to the SIMD one this build has: neon on aarch64, sse on x86-64. Lengths may
be a list or an `A..B` range.

## Fitting weights: measure once, replay forever

`dump` times every candidate at each length and writes a TSV. `score`, `costs` and `explain` then
replay that file offline, so iterating on the cost model needs no machine after the first run.

```sh
./target/release/planner_tuning dump --rounds 7 --out dump_neon_f64.tsv 1000 1050 1200 1296
./target/release/planner_tuning score dump_neon_f64.tsv          # regret vs best measured
./target/release/planner_tuning costs dump_neon_f64.tsv          # per-candidate cost and ns
./target/release/planner_tuning explain 'rad(rn(7.6,b24))'       # cost tree for one recipe
```

The candidates in a dump are enumerated exhaustively, one level deep, with inner recipes from the
fixed planner. So `score` replays a one-level choice, which is how the weights were fitted, and
not quite what the estimating planner does: it also optimises the inner recipes. `sweep` and
`survey` measure the real thing.

Every weight can be overridden on the command line, for `score` as well as for `sweep`:
`--strided`, `--permuted`, `--rader-index`, `--radixn-extra`, `--general-row`, `--small-row`.
The defaults per backend and element type are in `src/simd/simd_estimate.rs`.

**Pick weights by the worst machine, not the best.** Weights ship as compile-time constants per
backend and element type, so a value fitted to one machine is a regression on another. When a
weight's optimum differs between machines, sweep it on all of them and take the value whose worst
machine looks best.

## Other commands

- `regret LEN...` times every candidate plus the estimating planner's pick, and reports how far
  both planners are from the best. `--verbose` lists the top candidates.
- `verify LEN...` checks every candidate, and the estimating planner's pick, against a direct
  DFT. Run it after any change to candidate enumeration.
- `plantime LEN...` compares the cost of planning with each planner against the cost of building
  the recipe. These are cold-start figures: a planner reused across lengths shares inner lengths.
- `crossover LEN...` asks which recipe would win if construction cost counted, and how many
  executions the estimating planner's pick needs to repay any extra build cost.
- `time SPEC...` times recipes against each other. A `*reps` suffix runs a recipe over a
  `len*reps` buffer, which is how an inner FFT is invoked.

## Plotting

`plot_sweep.py` draws the population normalised by `N log2 N`, and the per-length ratio, for one
or several sweep files. It prints the same figures it draws. matplotlib is the only dependency:

```sh
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python plot_sweep.py sweep_neon_f64.tsv                 # one run
.venv/bin/python plot_sweep.py sweep_*_f64_*.tsv                  # compare runs or machines
.venv/bin/python plot_sweep.py sweep_neon_f64.tsv --out fig.png   # write instead of show
```

## Other backends

On x86-64 the tool crate selects the `sse` feature, so a plain `cargo build` gives the SSE
planner. Do not build the library with its default features for this: the planner would pick
AVX and measure the wrong backend.

wasm_simd builds for `wasm32-wasip1`, not `wasm32-unknown-unknown`, which has no clock. Run it
under node with `run_wasm.mjs`, and always with long blocks (`--block-ms 100`): V8 starts on its
baseline compiler and tiers up later, so short blocks time unoptimised code.

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1
node run_wasm.mjs sweep --planner wasm_simd --block-ms 100 1..200
```

## Traps

- **`--f32` is needed on `score` and `costs` as well as on `dump`.** The dump header records the
  planner, so the backend is recovered, but it does not record the element type.
- **Do not compile while measuring.** Build first, then run.
- **zsh does not word-split unquoted variables**, so `$LENGTHS` arrives as one argument. Inline
  the numbers or use `${=VAR}`.
- Measurement output (`*.tsv`, `*.txt`, `*.log`) and `.venv` are gitignored.

## The documents

- `COST-MODEL.md` is how the cost model works: what a cost is, how one is computed, which numbers
  are counted and which are fitted, and what has to be redone when a kernel changes.
- `OP-COUNTS.md` is where the instruction counts come from.
