# planner_tuning

A measurement harness for RustFFT's planners. Not part of the library and not shipped: it exists
to answer "which recipe is actually fastest at this length, and does the planner pick it?"

Everything here works against a `TunablePlanner`, so the same commands run on the scalar, NEON,
SSE and wasm planners. Recipes are built through the planner's own internals, so what gets timed
is exactly what the planner would construct.

## Vocabulary

A **recipe** is a `Spec` tree, printed compactly: `mr(b11,rad(mrs(b6,b10)))` is a MixedRadix of
butterfly 11 against Rader's over a MixedRadixSmall of 6 and 10. The prefixes are `b` butterfly,
`r4` Radix4, `rn` RadixN, `mr`/`mrs` MixedRadix and its Small variant, `gt`/`gts` Good-Thomas and
its Small variant, `rad` Rader's, `bs` Bluestein's.

**Regret** is a pick divided by the best measured candidate at that length: 1.000 is optimal,
1.05 is five percent off. It is a *lower bound* on the distance from optimal, because the
candidate set is finite.

## The two workflows

### Measure once, replay forever

`dump` times every candidate at each length and writes a TSV. `score`, `costs` and `explain` then
replay that file offline, so iterating on the cost model needs no machine after the first run.
This is how the weights were fitted.

```sh
cargo build --release
./target/release/planner_tuning dump --planner neon --rounds 7 --cap 48 \
    --out dump_neon_f64.tsv 1000 1050 1200 1296
./target/release/planner_tuning score dump_neon_f64.tsv          # regret vs best measured
./target/release/planner_tuning costs dump_neon_f64.tsv          # per-candidate cost and ns
./target/release/planner_tuning explain 'rad(rn(7.6,b24))'       # cost tree for one recipe
```

### Sweep a whole range

`sweep` times the planner's pick against the cost model's pick at every length in a range. Unlike
`dump` it builds only those two recipes per length, which is what makes a thousand lengths
affordable. Where the two agree the recipe is timed once and reported for both, so agreement
reads as exactly 1.000 rather than as noise.

This is the instrument that matters. Both cost-model defects fixed so far were invisible to the
33- and 44-length tuning sets and were found only by sweeping 1..1000.

```sh
./target/release/planner_tuning sweep --planner neon 1..1000 > sweep_neon_f64.tsv
./target/release/planner_tuning sweep --planner neon --f32 1..1000 > sweep_neon_f32.tsv
```

Lengths may be a list or an `A..B` range.

## Plotting

`plot_sweep.py` draws the two views worth having: the population normalised by `N log2 N`, and
the per-length ratio. It prints the same figures it draws, so it doubles as the reporting tool.
It takes several files at once, which is how cost-model stages and machines get compared.

matplotlib is the only third-party dependency in this directory, so it lives in a venv:

```sh
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python plot_sweep.py sweep_neon_f64.tsv                 # one run
.venv/bin/python plot_sweep.py sweep_*_f64_*.tsv                  # compare stages or machines
.venv/bin/python plot_sweep.py sweep_neon_f64.tsv --lo 4 --hi 128 # zoom
.venv/bin/python plot_sweep.py sweep_neon_f64.tsv --out fig.png   # write instead of show
```

## Plan time

`plantime` compares the cost of *choosing* a recipe against the cost of *building* it. Planning
only produces a `Recipe`; turning that into an `Arc<dyn Fft>` allocates and precomputes, and
Rader's and Bluestein's both run a full inner FFT inside their constructors. Building is 2x to
900x planning and grows much faster with length, so the estimating planner's overhead is a large
fraction of plan-plus-build at small lengths and a small one at large lengths.

```sh
./target/release/planner_tuning plantime --planner neon --cap 48 128 1024 1200 10007
```

Single plan-time measurements are noisy, up to 1.6x apart at length 1200. Take medians of three.
The candidate counts it prints are exact.

`crossover` goes further and asks which recipe would win if construction cost counted: it builds and
times every candidate, then reports the minimiser of `build + k * execute` at several `k`, plus how
many executions the cost model's pick needs to repay its extra build cost. Measured answer is three
to six, so this mostly documents why a construction-aware planner is not worth building.

```sh
./target/release/planner_tuning crossover --planner neon --cap 24 1260 1009 2018
```

## Two candidate sets, deliberately

- `candidates` / `candidates_capped` are **exhaustive**: every split in both orders, every
  algorithm that can express it, Bluestein's at every length. Used by `dump`, `verify` and
  `regret`, because scoring a planner's pick needs the alternatives even where no planner would
  look at them.
- `plan_candidates` is **what an estimating planner would really enumerate**. Used by `sweep` and
  `plantime`. It skips work that measurement has shown carries no decision: butterfly lengths and
  powers of two return the fixed planner's pick immediately, only the smaller-width-first ordering
  of each split is emitted, and Bluestein's is offered only where some prime factor has no
  butterfly of its own.

Keep the first set exhaustive. It is how each of those shortcuts was justified, and re-justifying
them after a kernel change needs the same breadth.

## Running on another machine

The campaign is SSH-driven. The remote checkouts are rsync copies rather than git worktrees, so
their `.git` points at a path that does not exist there; sync source only and leave the data.

```sh
rsync -az --delete src/ user@host:~/repos/RustFFT-testing/src/
rsync -az --delete tools/planner_tuning/src/ \
    user@host:~/repos/RustFFT-testing/tools/planner_tuning/src/
ssh user@host 'cd ~/repos/RustFFT-testing/tools/planner_tuning && cargo build --release'
```

On x86 the tuning crate selects the `sse` feature automatically, so a plain `cargo build` gives
the SSE planner.

## Traps

- **`sweep` and `plantime` do not infer the backend.** `score` and `costs` read it from the dump
  header, but `sweep` takes the model parameters as given, so an SSE run needs `--backend sse`
  explicitly. Without it the model prices SSE recipes with NEON instruction counts and the whole
  run looks plausible and is meaningless.
- **`--f32` is needed on `score` as well as on `dump`.** The dump header records the planner, so
  the backend is recovered, but it does not record the element type.
- **Do not compile while measuring.** Build first, then run.
- **zsh does not word-split unquoted variables**, so `$LENGTHS` arrives as one argument and the
  tool panics. Inline the numbers or use `${=VAR}`.
- **Run `verify` after any change to candidate enumeration.** It checks every enumerated candidate
  against a direct DFT, which catches an illegal spec such as a Bluestein's inner shorter than
  `2n - 1`.
- Measurement output (`*.tsv`, `*.txt`, `*.log`) and `.venv` are gitignored.

## The documents

- `COST-MODEL.md` is how the estimating planner works: what a cost is, how one is computed, which
  numbers are counted and which are fitted, and what has to be redone when a kernel changes. Start
  here.
- `RESULTS.md` is the evidence: how the cost model was built and what it scores.
- `OP-COUNTS.md` is where the instruction counts come from.
- `NEXT-STEPS.md` is the live plan, and the place new findings get written down.
