# How the estimating planner estimates

This is the "how it works" document. `OP-COUNTS.md` is where the instruction counts come from,
`RESULTS.md` is the evidence that the thing works, `NEXT-STEPS.md` is the live plan, and `README.md`
is how to run the tools. This file explains the mechanism that sits under all four: what a cost is,
how one is computed, which numbers are read off the source, which are fitted against measurements,
and what has to be redone when the kernels change.

**Status.** There is no estimating planner in the library. `src/tuning/` is a feature-gated
description of what the planners can build, and the cost model lives in the measurement tool at
`tools/planner_tuning/src/counted.rs`. What exists today is a full working prototype driven from the
command line, and `NEXT-STEPS.md` holds the open question of whether it ships whole or only as the
one scoped decision it is most clearly right about.

## 1. The loop

An estimating planner does three things, and only the middle one is new:

1. **Enumerate.** Build every recipe the planner could plausibly use at this length, as a tree of
   [`Spec`](../../src/tuning/mod.rs) nodes.
2. **Price.** Give each candidate a number.
3. **Pick the minimum.**

Step 1 is [`plan_candidates`](../../src/tuning/mod.rs#L585). Step 2 is
[`CountedModel::cost`](src/counted.rs#L409). Step 3 is a `min_by`, visible in
[`cmd_sweep`](src/main.rs#L452). That is the whole planner. Everything else in this document is
about step 2.

The enumeration deliberately answers immediately at a length with its own butterfly and at a power
of two, because measurement showed there is nothing to decide at either, and it emits only the
smaller-width-first ordering of each two-way split. The reasoning and the measured cost of each of
those shortcuts is in the doc comment on `plan_candidates`. Element zero of the list is always the
current fixed planner's pick, so pruning can never leave the estimating planner behind the planner
it replaces.

The separate exhaustive set (`candidates`, `candidates_capped`) exists because *scoring* a pick
needs alternatives that no planner would ever propose. Keep it exhaustive; it is how each shortcut
above was justified.

## 2. What a cost is

**One unit is one issued arithmetic instruction.** A cost is not nanoseconds and does not try to be:
nothing converts it to time, and nothing needs to, because only the ranking within one length is
ever used. That is why a single weight set travels across machines of different clock speeds, and it
is also the reason the absolute numbers below look large and mean nothing on their own.

Every cost is the sum of two halves:

```
cost = counted arithmetic instructions        (read off the source, see OP-COUNTS.md)
     + a memory term                          (accesses x pattern x cache level)
```

A pure operation count, which is the `FFTW_ESTIMATE` analogue, scores worse than the shipping
planner: mean regret 1.359 against 1.093. Adding the memory term takes the same op counts to 1.003.
The memory term is not a refinement, it is the entire result.

### The memory term

[`mem()`](src/counted.rs#L382) is four lines and deliberately crude:

```
accesses -> divided by complexes-per-vector, except under a permutation
level    -> L1 if the working set fits in l1_elems, else L2 if it fits l2_elems, else DRAM
cost     = accesses * seq[level] * (1.0 sequential | strided_mult | permuted_mult)
```

Three things about it are load-bearing and worth stating plainly:

1. **Sequential versus jumpy carries all of it.** Pricing every access the same degrades worst-case
   regret from 1.043 to 1.246. The finer distinction between strided and permuted does much less.
2. **The cache level cannot reorder candidates, as the model is built.** Collapsing L1/L2/DRAM to one
   flat cost produces byte-identical picks at every length measured, in cache and out. That is
   structural rather than a finding about memory: the level is chosen from the whole transform's
   working set (next section), which is the same for every candidate at one length, so it is a
   common factor. **It is not true of the hardware.** At 100003 and 100049 on the Pi 5, Rader's is
   about 3x faster than every Bluestein's candidate, because Bluestein's inner FFT is 262144 points
   (4 MB of complex f64), inside the M1's 12 MB L2 and far outside the A76's 512 KB L2 and 2 MB L3.
   No setting of the cache sizes or level weights can express that. See blind spot 4.
3. **A permuted pass is charged per complex number, not per vector.** A gather or scatter computes
   an address per element and cannot fill a vector. This is invisible at f64, where the factor is 1,
   and was worth a factor of 2 at f32; finding it is what took SSE f32 from 51 to 117 of 216 weight
   settings clearing the 20% bar. `--permuted-vector` restores the old behaviour.

### The working set is threaded down unchanged

[`cost_ws`](src/counted.rs#L416) passes the *whole transform's* length to every nested algorithm,
not the nested algorithm's own length. Every pass of every inner FFT walks the top-level buffer, so
that is what decides where the traffic is served from. Pricing an inner FFT as if it ran standalone
is the specific mistake that sank the 2021 attempt.

The exception is Bluestein's, whose inner FFT runs on a buffer of its own that is at least twice the
outer length. Threading the outer length down understates that working set, which is exactly the
Pi 5 defect above. Charging a Bluestein's node at its inner length would fix it and still keep a
node's cost a function of its own subtree, which is what length-keyed recipe memoisation needs. Not
tried yet.

One caveat on reading `explain` output: it calls `cost()` per node, so each row is priced at its
own length as working set. Since the cache level cannot reorder candidates this almost never
changes a number, but the child rows of a very large transform are informational rather than exact
contributions.

## 3. What each node costs

All of this is [`cost_ws`](src/counted.rs#L416), one match arm per `Spec` variant. `len` is the
node's own length and `ws` the whole transform's.

| node | arithmetic | memory |
|---|---|---|
| `Butterfly(len)` | counted table lookup | `2*len` sequential |
| `Radix4 { k, base }` | `reps` x base, plus per layer `len/4` butterfly4 and 3 twiddles each | one `2*len` permuted digit reversal, plus `2*len` strided per layer |
| `RadixN { radixes, base }` | `reps` x base, plus per layer `len/r` x (butterfly `r` + `r-1` twiddles) | same shape as Radix4, plus `radixn_extra` per element per layer |
| `MixedRadix` | `h` x left + `w` x right, plus `len` twiddle multiplies | three transposes, plus one sequential pass |
| `GoodThomas` | `h` x left + `w` x right, no twiddles at all | two permuted CRT passes plus one transpose |
| `Raders { inner }` | 2 x inner, `len` twiddles, `len * rader_index` | two permuted passes plus one sequential |
| `Bluesteins { len, inner }` | 2 x inner, `inner.len()` pointwise multiplies | sequential over the inner length and twice over the outer |
| `Dft(n)` | `100 * n^2` | none; a quadratic that only has to sort last |

Two of these encode a decision the model would otherwise be unable to make:

- **`small` versus general.** `MixedRadixSmall` and `GoodThomasAlgorithmSmall` call
  `array_utils::transpose_small`, a naive strided double loop, so their transposes are priced
  `Permuted`. The general variants hand the job to the `transpose` crate, which tiles the rectangle
  and gets the reuse back, and pay `general_row` per row for the privilege. That makes the general
  form cheaper per element and dearer per row, which is a crossover, and the measurements are the
  shape of one: `MixedRadix` crosses under its Small variant near length 200. Without the per-row
  term the model has only per-element costs, so it would pick the general form at every length.
- **Which ordering of a split.** `small_row * max(width - height, 0)` is the only thing that
  separates `mrs(A,B)` from `mrs(B,A)`. `transpose_small`'s outer loop runs `width` times, so both
  Small variants change by exactly `width - height` when the pair is reversed, which is why one
  weight covers both. Charging the *difference* rather than the raw count keeps a square pair free
  and leaves the better ordering at exactly the cost it had before the term existed.

### A worked example, which is also a self-check

```
$ ./target/release/planner_tuning explain 'mrs(b8,rad(rn(3.3,b9)))'
recipe                                 len  times        total cost    own cost
mrs(b8,rad(rn(3.3,b9)))                656  x1               127628       13776
  b8                                     8  x82                4428        4428
  rad(rn(3.3,b9))                       82  x8               109424       69536
    rn(3.3,b9)                          81  x16               39888       28080
      b9                                 9  x144              11808       11808
```

The `b8` row by hand: 38 counted instructions from the NEON f64 table, plus `2*8 = 16` sequential
accesses at `seq[0] = 1.0` and a sequential multiplier of 1.0, is 54. It runs 82 times, giving 4428.

The root's own cost by hand: three transposes at `3 * 2*656 * 2.5` permuted is 9840, no `small_row`
charge because the width is the smaller side already, plus `656` twiddle multiplies at 4
instructions each is 2624, plus a `2*656` sequential pass is 1312. Total 13776.

If a change to `counted.rs` is supposed to leave some family of recipes alone, `explain` on one of
them is the fastest way to confirm it did.

## 4. Where every number comes from

This is the distinction the whole approach rests on. Most of the model is read off the source and
must be maintained when the source changes; a small set of weights is fitted against measurements
and must be refitted when the machine changes.

### Read off the source, never fitted

| quantity | where it is read | notes |
|---|---|---|
| butterfly instruction counts | `src/neon/*.rs`, `src/sse/*.rs`, by hand | tables in `butterfly_compute`, derived in `OP-COUNTS.md` |
| generated prime butterflies | the generator's loop structure | closed form `(h-1)(2h+5)`, cannot drift when sizes are added |
| `mul_complex`, `column_butterfly4` | the vector trait impls | 4 on NEON f64, 6 on SSE and on NEON f32 |
| `registers` | the architecture | 32 `v` on aarch64, 16 `xmm` on x86-64 |
| access counts per pass | the algorithm source | loads plus stores, counted per pass |
| access *pattern* per pass | the algorithm source | which of sequential, strided, permuted applies |
| `l1_elems`, `l2_elems` | the machine's cache sizes | inert as the model is built: no pick depends on them. Defaults are the M1's |

### Derived from the code, then confirmed against measurement

| quantity | derivation |
|---|---|
| `rader_index` = 30 (f64), 45 (f32) | `raders_algorithm.rs` recomputes `index * root % len` per element, a loop-carried `mul -> umulh -> mul -> sub` chain of about 10 cycles. That is worth tens of instruction slots, not the 7 instructions counted, but how many depends on the core's instructions per cycle, so the optimum is per machine: about 40 on the M1, 25 to 30 on a Cortex-A76. The defaults are the values acceptable on all three measured machines (M1, Pi 5, ThinkCentre) by the 1..1000 sweep, not the optimum on any one. See `NEXT-STEPS.md`. |
| `small_row` = 10 | One outer iteration of `transpose_small`, which measures 1.48 ns on the i3 and 0.7 to 1.0 ns on the M1, or 9 to 13 instruction-equivalents at either machine's scale. Scores are byte-identical for anything from 2 to 24 on every dataset, because the term only ever separates two orderings that are otherwise exactly equal. It is a tie-break with a derivation, not a weight. |

### Fitted by comparing against measured times

Six numbers, and only these six. They convert a memory access into arithmetic-instruction
equivalents, which is the one thing that cannot be read off RustFFT's source because it is a
property of the machine.

| parameter | what it prices | fitted over |
|---|---|---|
| `seq[0]`, `seq[1]`, `seq[2]` | one access at L1, L2, DRAM | pinned at 1.0; the other two are inert, see below |
| `strided_mult` | a fixed-stride pass against a sequential one | 1.0, 1.5, 2.5, 4.0 |
| `permuted_mult` | a gather or scatter against a sequential pass | 1.5, 2.5, 4.0, 6.0 |
| `radixn_extra` | the generic `SimdRadixN` driver against the hand-written `Radix4` kernel, per element per layer | 0, 1, 2, 3, 5, 8; defaults 0 on NEON, 6 on SSE f64, 1 on SSE f32, chosen by the 1..1000 sweep |
| `general_row` | per-row setup of the blocked transpose and the on-the-fly CRT mapping | the twelve measured general-over-small ratios |

`spill` and `mul_complex` are diagnostic overrides rather than fitted weights. Both are off by
default. They exist so that a specific hypothesis can be tested (register pressure in the RadixN
cross layer, and whether SSE's shuffle-heavy complex multiply costs more than its instruction count)
without editing the model.

### The fitted values, per dataset

```
dataset      seq[1]  strided  permuted  radixn_extra   mean    worst
NEON f64      any      1.5       1.5          0        1.003   1.043
NEON f32      3.0      2.5       2.5          0        1.032   1.121
SSE  f64      2.0      2.5       1.5          5        1.052   1.152
SSE  f32      1.5      2.5       2.5          3        1.034   1.121
```

**Each of those four is one (machine, backend, element type) cell.** Every NEON number comes from
the M1 and every SSE number from the ThinkCentre, so on those two alone "the NEON weights" and "the
M1 weights" are the same column. Which of the two it really is decides whether this ships:

- **If the split is per backend**, it costs nothing. The backend is chosen at compile time plus a
  feature check, and the element type is a generic parameter, so each combination can carry its own
  constants exactly as it already carries its own op counts. Four constant sets, all clearing the
  bar.
- **If the split is per machine**, no compiled-in constant is right for anybody, and the whole
  approach needs a runtime calibration step that nothing here has designed.

**A second ARM machine, the Pi 5 (Cortex-A76), says mostly per backend.** Its 1..1000 sweeps with
the M1's NEON weights give the same picks at every length, so only the timing differs. Grouped by
which algorithm each planner chose, every class of decision lands within 1.5% of the M1 except
Rader's versus Bluestein's. That includes MixedRadix versus GoodThomas, which `RESULTS.md` had as
the one preference following the machine (wasm on the M1 within 0.004 of NEON on the M1, SSE on the
ThinkCentre 0.088 away): on the Pi those calls score 1.009 in f64 and 1.018 in f32, against 1.008
and 1.030 on the M1.

The exception is genuinely per machine. `rader_index` is a latency converted into instruction slots,
so it depends on the core: about 40 on the M1, 25 to 30 on the A76. It is handled without runtime
calibration, by choosing the default that is acceptable on all three machines rather than optimal
on one (`NEXT-STEPS.md` has the three-machine table). That is the rule for any future per-machine weight too: sweep it
everywhere and take the value whose worst machine looks best.

### The floor, if one set had to serve everything

Worth knowing because it bounds the damage. Gridding a single memory weight set jointly against all
four dumps, allowing only `radixn_extra` to differ per backend since that one is a register-file
property, the best is `strided 1.5, permuted 2.5, radixn_extra 0 on NEON and 5 on SSE`, with the
cache levels irrelevant as always:

```
              one shared set     own weights      fixed planner
NEON f64      1.003 / 1.043     1.003 / 1.043    1.093 / 1.495
NEON f32      1.032 / 1.225     1.032 / 1.121    1.171 / 1.969
SSE  f64      1.085 / 1.323     1.052 / 1.152    1.246 / 1.734
SSE  f32      1.052 / 1.250     1.034 / 1.121    1.207 / 1.927
```

(mean / worst.) Three of the four then miss the 20% target, so this is not the proposal. But it
still beats the fixed planner on both statistics on all four datasets, and NEON f64 loses nothing at
all. The weights are worth getting right; getting them wrong degrades the result rather than
inverting it.

`radixn_extra` is the one fitted weight that is not just a fudge: it is exactly 0 on NEON and
positive on SSE, which is what a register-count argument predicts, since `cross_layer` holds 2R rows
live and 2R fits 32 `v` registers at every supported radix and does not fit 16 `xmm`. It also shrinks
from 5 to 2-3 when the element type halves, because a spilled register then covers twice the
elements. Predicted 2.5, observed 2 to 3.

**That was before the RadixN transpose fix** (415a29f), which removed per-call divides the weight had
been partly absorbing. Refitted by the 1..1000 sweep on the ThinkCentre, the defaults are now 6 for
SSE f64 and 1 for SSE f32, so the halving prediction no longer holds. Zero on NEON still does. The
table above predates the fix; `NEXT-STEPS.md` has the refit.

Note that `Params::default()` is the NEON working set with `permuted_mult` at 2.5 rather than the
fitted 1.5. It makes no difference to that dataset, but a run that means to reproduce a table above
should pass the weights explicitly rather than trust the defaults.

## 5. How the weights are fitted

The procedure is a grid search against a frozen measurement, with a held-out half.

```sh
# 1. measure once. every candidate at every length, times written to a TSV.
./target/release/planner_tuning dump --planner neon --rounds 7 --cap 48 \
    --out dump_neon_f64.tsv <the 33 lengths>

# 2. split into halves. even-indexed lengths train, odd-indexed test, on sorted length,
#    so each half spans the whole size range.
python3 split.py dump_neon_f64.tsv train.tsv test.tsv

# 3. grid the weights against the training half only.
./grid.sh train.tsv                      # 216 points
./sweep.sh train.tsv                     # the older 108-point grid

# 4. score the winner on the test half, once.
./target/release/planner_tuning score --seq-l2 1.5 --strided 1.5 --permuted 1.5 \
    --rader-index 30 test.tsv
```

Steps 2 to 4 are **pure replay**. No machine is involved, no planner is built, and a full grid takes
seconds. That is the property that makes this maintainable where the old measured-table model was
not: one measurement run per machine, then unlimited model iteration anywhere.

**Fit on the training half only.** Gridding on the full dump and then quoting a held-out number is
not a held-out number. The honest SSE f32 held-out worst case is 1.163, not the 1.105 that a
fit-on-everything run reports.

The metric is **regret**: measured time of the chosen recipe divided by measured time of the best
enumerated candidate, so 1.000 is optimal. It is a lower bound on the distance from optimal, because
the candidate set is finite and the inner recipes inside each candidate come from the same planner.

Two sanity properties of the fit are worth knowing before touching a weight:

- **96 of 108 settings clear the 20% bar** on NEON f64. The result does not depend on hitting the
  weights precisely, which is the main reason to think they travel.
- **The DRAM weight is inert.** 3.0, 6.0, 10.0 and 16.0 give byte-identical results, for the
  structural reason in section 2.
- **A flat weight on the tuning set proves nothing about the sweep.** The Rader's weight is flat
  from 15 to 120 on the original 33 lengths, because none of them is a small prime. The 1..1000
  sweep moves geometric mean by up to 4% between 30 and 45.

## 6. Updating the model when the code changes

This is the cost of the approach. The model is accurate *because* it tracks the source, and that
means the source moving invalidates it. The upside is that every such update is a re-count, which is
mechanical and needs no machine, rather than a re-measurement campaign.

| what changed | what to redo |
|---|---|
| a butterfly's body | re-count it into `OP-COUNTS.md`, update the table in `butterfly_compute` |
| a butterfly length added or removed | nothing for the generated primes, they are a closed form; a table entry for a hand-written one |
| a vector primitive, for instance FMA or `vcmlaq` arriving | update `Backend::mul_complex` or `column_butterfly4`, then re-count every butterfly built from it. This is what the fcma work would trigger |
| an algorithm's pass structure, for instance `raders_precompute` landing | re-derive that node's arm in `cost_ws`. A precomputed Rader's permutation drops the per-element cost by roughly 4x and `rader_index` stops being a latency term at all |
| a transpose or index computation swapped | recheck the `Pattern` on that pass, and whether `general_row` still describes the same per-row work |
| a new algorithm in a planner | a `Spec` variant, an adapter arm, and a `cost_ws` arm |
| a new backend | a `Backend` variant, its counts, and its register file size |
| a new machine | nothing counted changes. Run `sweep 1..1000` in f64 and f32 and compare it class by class against a machine on the same backend. Where a weight's optimum moves, choose the value acceptable on every machine, never refit to the new one alone |

After any of them, in this order:

1. `verify` at a spread of lengths, first. It checks every enumerated candidate against an f64
   reference DFT, which catches an illegal spec such as a Bluestein's inner shorter than `2n - 1`.
   Never trust a timing taken before `verify` is clean.
2. `explain` on a recipe whose cost you can predict, as in the worked example above.
3. `score` against an existing dump, which is free, to see whether the ranking moved at all.
4. `sweep --planner <p> 1..1000` if it did. Both cost-model defects found so far were invisible to
   the 33- and 44-length tuning sets and showed up only in the full sweep.
5. A fresh `dump` and refit only if the counts changed enough to move the fitted weights, which is
   unusual: the weights price memory, and a kernel change usually moves arithmetic.

## 7. Known blind spots

Short version; `RESULTS.md` has the numbers and `NEXT-STEPS.md` has what to do about each.

1. **Width and height are tied.** The cost function gives `mr(A,B)` and `mr(B,A)` the same cost at
   all 415 reversed pairs, apart from the `small_row` tie-break, yet 133 to 179 of them measure more
   than 2% apart. This is the clearest unexploited improvement and it is derivable from the code.
2. **MixedRadix versus GoodThomas is a fixed preference, not a decision.** With identical inners the
   cost difference is a constant multiple of `len`, so its sign cannot vary with length. It picks
   GoodThomas at 142 of 142 pairs, which is right 121 times on the M1 and 38 times on the i3. Keep
   any correction small: a stride-aware rewrite aimed at this regressed both backends and was
   reverted.
3. **Three machines, two per backend on ARM only.** The Pi 5 separated machine from backend on
   ARM (section 4), but x86 still has only the ThinkCentre. A Zen 5 box is the instrument for that.
4. **Bluestein's working set is not priced.** The cache level is taken from the outer transform
   (section 2), so a Bluestein's inner FFT two to four times the outer length is charged as if it
   fitted where the outer one does. Invisible on the M1 and in any 1..1000 sweep. On the Pi 5 it
   picks Bluestein's at 100003 and 100049 where Rader's is about 3x faster.
5. **Plan time.** Enumerate-and-price is 20x to 1265x the fixed planner's plan time, which is the
   wrong denominator: what a caller pays is plan plus build, and building is 113x planning overall.
   Against plan-plus-build, medians on the M1 at `--cap 48`:

   ```
   len       fixed plan   plan+price    build    extra on plan+build   in FFT executions
   1260          0.6 us      62.5 us    8.0 us          +720%               ~12
   10080         0.5 us       138 us     63 us          +219%                ~3
   100800        0.5 us       284 us    537 us           +53%               ~0.5
   ```

   So the cost is about twelve executions of the transform being planned at the worst measured
   length, under one above 100k, and zero at butterfly lengths and powers of two where enumeration
   short-circuits. Memoising inner recipes across candidates is the obvious optimisation and has
   deliberately not been done.
