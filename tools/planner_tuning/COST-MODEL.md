# How the estimating planner estimates

This is the "how it works" document. `OP-COUNTS.md` is where the instruction counts come from and
`README.md` is how to run the tools. This file explains what a cost is, how one is computed, which
numbers are read off the source, which are fitted against measurements, and what has to be redone
when the kernels change.

The model lives in the library, at `src/simd/simd_estimate.rs`, and is shared by the NEON, SSE and
wasm_simd planners. The scalar and AVX planners do not use it.

## 1. The loop

An estimating planner does three things, and only the middle one is new:

1. **Enumerate.** List the recipes that could compute this length, as `Shape` values: a top-level
   algorithm plus the lengths of its inner FFTs.
2. **Price.** Give each candidate a number.
3. **Pick the minimum.**

All three are in `design_fft_for_len` in each planner, over `candidates` and `CostModel::cost`.
Ties go to the first candidate, which is always the fixed planner's pick.

Enumeration answers immediately at a length with its own butterfly and at a power of two, because
measurement showed there is nothing to decide at either, and it emits only the smaller-width-first
ordering of each two-way split. It offers Bluestein's only where some prime factor has no butterfly
of its own. The reasoning for each shortcut is in the doc comments on `has_choice` and `candidates`.

The exhaustive enumeration in `src/tuning/` (`candidates`, `candidates_capped`) is a separate thing,
used only by the measurement tools: *scoring* a pick needs alternatives no planner would propose.
Keep it exhaustive. It is how each shortcut above was justified, and re-justifying them after a
kernel change needs the same breadth.

## 2. What a cost is

**One unit is one issued arithmetic instruction.** A cost is not nanoseconds and does not try to be:
nothing converts it to time, because only the ranking within one length is ever used. That is why
one weight set travels across machines of different clock speeds, and why the absolute numbers below
are large and mean nothing on their own.

```
cost = counted arithmetic instructions        (read off the source, see OP-COUNTS.md)
     + a memory term                          (accesses x pattern x whether it fits in cache)
```

A pure operation count, the `FFTW_ESTIMATE` analogue, scores *worse* than the fixed planner. Adding
the memory term is what makes the model work; it is not a refinement.

### The memory term

`mem()` is deliberately crude. Each pass is charged per element it touches, times a multiplier for
how it walks memory:

```
sequential   1.0      a contiguous run, one cache line feeding many elements
strided      strided  a fixed stride, as in a transpose or a cross-FFT layer
permuted     permuted a computed address per element: digit reversal, CRT, Rader's
```

Sequential and strided accesses are divided by the number of complex numbers in a vector; permuted
ones are not, because a gather computes an address per element and cannot fill a vector.

Three things about it are load-bearing:

1. **Sequential versus jumpy carries most of the result.** Pricing every access the same degrades
   worst-case regret badly.
2. **A pass that no longer fits in cache costs more.** Above `cache_elems` complex numbers an access
   costs `dram_pass`, and a transpose's accesses cost `dram`. See section 4, which is where the
   machines disagree most.
3. **A node is charged at the size of the buffer it walks itself**, never at the size of the
   transform it is nested inside.

That last point is what makes the search affordable. A recipe's cost depends only on its own
subtree, so each planner caches the best cost per length beside its recipe cache and the search
recurses over the divisors of a length rather than over whole trees. **If a future term ever makes a
node's cost depend on its parent, that caching becomes wrong**, and the failure would be a subtly
bad inner recipe rather than anything that trips a test.

## 3. What each node costs

One match arm per `Shape` variant in `CostModel::cost`. `len` is the node's own length.

| node | arithmetic | memory |
|---|---|---|
| `Butterfly(len)` | counted table lookup | `2*len` sequential |
| `Radix4 { k, base_len }` | `reps` x base, plus per layer `len/4` butterfly4 and 3 twiddles each | one `2*len` permuted digit reversal, plus `2*len` strided per layer, plus `radix_call` |
| `RadixN { factors, base_len }` | `reps` x base, plus per layer `len/r` x (butterfly `r` + `r-1` twiddles) | same shape as Radix4, plus `radixn_extra` per element per layer |
| `MixedRadix` | `h` x left + `w` x right, plus `len` twiddle multiplies | three transposes, plus one sequential pass |
| `GoodThomas` | `h` x left + `w` x right, no twiddles at all | two permuted CRT passes plus one transpose |
| `Raders { len }` | 2 x inner, `len` twiddles, `len * rader_index` | two permuted passes plus one sequential |
| `Bluesteins { len, inner_len }` | 2 x inner, `inner_len` pointwise multiplies | sequential over the inner length and twice over the outer |

Three of these encode a decision the model would otherwise be unable to make:

- **`small` versus general.** `MixedRadixSmall` and `GoodThomasAlgorithmSmall` call
  `array_utils::transpose_small`, a naive strided double loop, so their transposes are priced
  `Permuted`. The general variants hand the job to the `transpose` crate, which tiles the rectangle
  and gets the reuse back, and pay `general_row` per row for it. That makes the general form cheaper
  per element and dearer per row, which is a crossover, and the measurements are the shape of one:
  `MixedRadix` crosses under its Small variant near length 200. Without the per-row term the model
  has only per-element costs and would pick the general form at every length.
- **Which ordering of a split.** `small_row * max(width - height, 0)` is the only thing separating
  `mrs(A,B)` from `mrs(B,A)`. `transpose_small`'s outer loop runs `width` times, so both Small
  variants change by exactly `width - height` when the pair is reversed, which is why one weight
  covers both. Charging the *difference* rather than the raw count keeps a square pair free and
  leaves the better ordering at exactly the cost it had before the term existed.
- **A generic driver against a hand-written one.** `radix_call` is what a RadixN or Radix4 execution
  costs regardless of length, and `radixn_extra` is what its cross layers cost per element over
  `Radix4` doing the same work. Without the first, short lengths take a RadixN that measures 1.32x
  slower than a table-driven `GoodThomasAlgorithmSmall`.

### A worked example, which is also a self-check

```
$ ./target/release/planner_tuning explain 'mrs(b8,rad(rn(3.3,b9)))'
recipe                                 len  times        total cost    own cost
mrs(b8,rad(rn(3.3,b9)))                656  x1                72812       13776
  b8                                     8  x82                4428        4428
  rad(rn(3.3,b9))                       82  x8                54608       13120
    rn(3.3,b9)                          81  x16               41488       29680
      b9                                 9  x144              11808       11808
```

The `b8` row by hand: 38 counted instructions from the NEON f64 table, plus `2*8 = 16` sequential
accesses at 1.0 each, is 54. It runs 82 times, giving 4428.

The root's own cost by hand: three transposes at `3 * 2*656 * 2.5` permuted is 9840, no `small_row`
charge because the width is the smaller side already, plus `656` twiddle multiplies at 4
instructions each is 2624, plus a `2*656` sequential pass is 1312. Total 13776.

If a change to the model is supposed to leave some family of recipes alone, `explain` on one of them
is the fastest way to confirm it did.

## 4. Where every number comes from

This is the distinction the whole approach rests on. Most of the model is read off the source and
must be maintained when the source changes; a small set of weights is fitted against measurements
and must be rechecked when the machines change.

### Read off the source, never fitted

| quantity | where it is read |
|---|---|
| butterfly instruction counts | `src/neon/*.rs`, `src/sse/*.rs`, by hand; derived in `OP-COUNTS.md` |
| generated prime butterflies | the generator's loop structure, as a closed form that cannot drift |
| `mul_complex`, `column_butterfly4` | the vector trait impls |
| access counts and patterns per pass | the algorithm source, loads plus stores per pass |

### Fitted, and where each was fitted

| weight | value | how it was chosen |
|---|---|---|
| `strided` | 1.5, or 2.5 on SSE f32 | grid against measured dumps |
| `permuted` | 2.5 | grid against measured dumps |
| `general_row` | 30 | the twelve measured general-over-small ratios |
| `small_row` | 10 | one outer iteration of `transpose_small`, 1.48 ns on an i3, 0.7 to 1.0 on an M1. A tie-break with a derivation; anything from 2 to 24 scores the same |
| `rader_index` | 2 | one load from the permutation table ejmahler#178 added. It was 30 and 45 when that index came from a loop-carried modular multiply |
| `radixn_extra` | 0 on NEON, 6 and 1 on SSE f64 and f32 | expected near zero where 2R rows fit the register file: 32 vector registers on aarch64, 16 on x86-64. **Fitted before the memory terms below and not yet rechecked** |
| `radix_call` | 100 | sized at length 14 on an M1, about 8 ns or 25 cycles: the call, the scratch split, the layer setup and the virtual call into the base FFT |
| `cache_elems` | 256 KiB worth | the smallest last-level cache worth planning for, not any one machine's. No pick below length 16385 changes at this threshold, so weights fitted on short lengths stay valid |
| `dram`, `dram_pass` | 6 and 2 | fitted on a Raspberry Pi 5 and checked on an M1; see below |

### The two weights where machines disagree

`dram_pass` and `dram` are the first weights in this model whose optimum depends on the machine
rather than the backend, because they price the memory system rather than the instruction set. The
Pi 5 has 512 KB of L2 and 2 MB of L3 against the M1's 12 MB of L2, and without them the model
computes a whole transform as one Bluestein's whose inner FFT is far larger than cache: on the Pi
that costs 4.19x in f64 and 3.81x in f32, while on the M1 the same picks are fine.

They have to move together. Charging ordinary passes without charging transposes just as hard makes
a MixedRadix wrapped around a smaller radix recipe look good, since its inner FFTs are then cache
resident, and those recipes measure worse on both machines. Over the lengths where the settings
disagree, as geomean / worst / losses beyond 5%:

| `dram_pass` / `dram` | M1 f64 | Pi f64 | M1 f32 | Pi f32 |
|---|---|---|---|---|
| 1 (off) | 0.847 / 1.27 / 1 | 0.705 / 4.19 / 7 | 0.982 / 1.49 / 3 | 1.365 / 3.81 / 10 |
| 2 / 2 | 0.981 / 1.62 / 7 | 0.710 / 2.23 / 5 | 1.251 / 1.67 / 9 | 0.916 / 2.84 / 6 |
| 2 / 4 | 0.848 / 1.04 / 0 | 0.624 / 1.29 / 2 | 0.952 / 1.05 / 1 | 0.851 / 3.63 / 2 |
| **2 / 6** | **0.854 / 1.02 / 0** | **0.629 / 1.25 / 3** | **0.949 / 1.07 / 1** | **0.848 / 3.58 / 2** |

Over the whole validation set, 6 beats 4 by halving the M1's f32 losses beyond 5%, 27 to 15, for
an unchanged loss count on the Pi and a worst case there that grows from 3.22 to 3.51 inside the
large-Bluestein class neither value solves.

## 4a. What it scores

334 lengths spread over 1 to 1,000,000 (`survey --count 350 --seed 1`), as the estimating
planner's runtime over the fixed planner's, so below 1 is faster:

| | geomean | p10 | p50 | p90 | worst | losses >5% | wins >5% |
|---|---|---|---|---|---|---|---|
| M1 f64 | 0.917 | 0.76 | 0.98 | 1.000 | 1.14 | 5 | 147 |
| M1 f32 | 0.933 | 0.79 | 1.00 | 1.005 | 1.40 | 15 | 118 |
| Pi 5 f64 | 0.881 | 0.70 | 0.96 | 1.000 | 1.26 | 3 | 161 |
| Pi 5 f32 | 0.938 | 0.78 | 1.00 | 1.000 | 3.51 | 14 | 123 |

SSE is not in this table: the machine was unreachable when it was taken, and `radixn_extra` there
was fitted before the memory terms existed.

The rule for any future weight like these: sweep it on every machine and take the value whose worst
machine looks best, rather than the optimum on any one.

## 5. What to redo when the kernels change

In this order:

1. `verify` at a spread of lengths, first. It checks every enumerated candidate against an f64
   reference DFT, which catches an illegal spec such as a Bluestein's inner shorter than `2n - 1`.
   Never trust a timing taken before `verify` is clean.
2. `explain` on a recipe whose cost you can predict, as in the worked example above.
3. `picks` before and after, which needs no machine and takes seconds over tens of thousands of
   lengths. It says how far the change reaches before anything has been measured.
4. `score` against an existing dump, also free, to see whether the ranking moved.
5. `sweep 1..1000` and a `survey` up to a million if it did. Both cost-model defects found during
   the spike were invisible to the 33- and 44-length tuning sets and showed up only in a sweep.

**A dump goes stale when the algorithm it measured changes.** Every dump taken before ejmahler#178
holds the old, slow Rader's, so it cannot judge any Rader's decision: at `rader_index` 2 the three
worst lengths in the old NEON f64 dump are all Rader's picks, which is the stale timing talking, not
the weight.

## 6. Known blind spots

1. **Bluestein's at large lengths in f32.** The worst remaining class. On the Pi 5, lengths like
   774209 and 232371 still take a Bluestein's whose inner FFT is several times the cache, at 3.2x
   and 2.5x the fixed planner. The memory terms improved these without fixing them.
2. **Width and height are tied.** The cost function gives `mr(A,B)` and `mr(B,A)` the same cost
   apart from the `small_row` tie-break, yet many reversed pairs measure more than 2% apart. This is
   the clearest unexploited improvement and it is derivable from the code.
3. **MixedRadix versus GoodThomas is a fixed preference, not a decision.** With identical inners the
   cost difference is a constant multiple of `len`, so its sign cannot vary with length. Keep any
   correction small: a stride-aware rewrite aimed at this regressed both backends and was reverted.
4. **x86 has one machine.** The Pi 5 separated machine from backend on ARM. Nothing has done that
   for SSE, and `radixn_extra` is the largest single weight fitted there.
5. **Plan time.** Enumerate-and-price costs far more than the fixed planner's plan, which is the
   wrong denominator: a caller pays plan plus build, and building dominates. Against plan-plus-build
   the estimating planner costs a few executions of the transform it is planning at small lengths,
   under one at 100k, and nothing at butterfly lengths and powers of two where enumeration
   short-circuits. These are cold-start figures; a planner reused across lengths shares inner
   lengths through its caches.
