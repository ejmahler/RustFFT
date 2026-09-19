# NEON and SSE f64 operation counts, and how they were derived

Every number here was obtained by **reading the source**, not by benchmarking. The unit is one
issued NEON instruction. The counts feed the analytic cost model as leaf costs, replacing the
measured butterfly table.

Counting was done by hand, using `grep`/`python` only as a calculator to histogram intrinsic names
inside a function body. Nothing infers cost from a timing.

## Primitive costs

From `src/neon/neon_vector.rs` (the `impl NeonVector for float64x2_t` block) and
`src/neon/neon_utils.rs` (`impl Rotate90F64`).

| primitive | instr | derivation |
|---|---|---|
| `vaddq_f64`, `vsubq_f64`, `vmulq_f64`, `vfmaq_f64`, `vnegq_f64`, `vneg_f64`, `veorq_u64`, `vcombine_f64`, `vmulq_laneq_f64`, `vfmaq_laneq_f64` | 1 | one instruction each |
| `vreinterpretq_*`, `vget_low_f64`, `vget_high_f64` | 0 | pure bit-pattern views, or fold into the consuming `vcombine` |
| `vmovq_n_f64` of a literal | 0 | loop-invariant constant, hoisted |
| `vld1q_f64` / `vst1q_f64` (`load_complex` / `store_complex`) | 1 | one load or store |
| `solo_fft2_f64` | **2** | `vaddq_f64` + `vsubq_f64` |
| `NeonVector::column_butterfly2` | **2** | same, one add and one sub |
| `Rotate90F64::rotate`, `NeonVector::apply_rotate90` | **2** | `vcombine_f64` + `veorq_u64`; the `vget_*` fold in |
| `Rotate90F64::rotate_45` / `_135` / `_225` | **4** | `rotate` (2) + one `vaddq`/`vsubq` (1) + one `vmulq` (1) |
| `NeonVector::mul_complex` | **4** | `vcombine_f64` + `vneg_f64` + `vmulq_laneq_f64` + `vfmaq_laneq_f64` |
| `NeonVector::column_butterfly4` | **10** | 4 x `column_butterfly2` (8) + 1 x `apply_rotate90` (2) |

`mul_complex` costing 4 is the single most load-bearing primitive, since it is what every twiddle
multiply in RadixN, Radix4 and MixedRadix costs. Note the source comment: ARMv8.2-A `vcmlaq_f64`
would collapse this to 1-2, which is what the `fcma_backend` branch is about.

## Load and store

Every butterfly reads each of its `len` complex inputs exactly once and writes each of its `len`
outputs exactly once, so **loads = stores = len**, giving `2 * len` memory instructions per call.
This was confirmed by inspection rather than grepped, because the larger butterflies (16, 24, 32)
issue their loads from inside a `let load = |i| {...}` closure invoked once per column, which a
naive text count gets wrong.

## Hand-written butterflies

From `src/neon/neon_butterflies.rs`, counting the body of `perform_fft_direct` (or of
`perform_fft_contiguous` for 16, 24 and 32, which have no separate `direct`), with `new()` excluded
because it is construction, not per-call work. Sub-butterfly calls are resolved bottom-up.

| len | composition | compute instr |
|---|---|---|
| 1 | nothing | 0 |
| 2 | 1 x `solo_fft2` | **2** |
| 3 | 2 `vaddq` + 1 `vsubq` + 3 `vfmaq` + 1 `rotate` | **8** |
| 4 | 4 x `solo_fft2` + 1 `rotate` | **10** |
| 5 | 11 `vaddq` + 5 `vsubq` + 8 `vmulq` + 2 `rotate` | **28** |
| 6 | 2 x bf3 + 3 x `solo_fft2` | **22** |
| 8 | 2 x bf4 + 3 `rotate` + 1 `vaddq` + 1 `vsubq` + 2 `vmulq` + 4 x `solo_fft2` | **38** |
| 9 | 3x3: 6 x bf3 + 4 x `mul_complex` | **64** |
| 10 | 5x2 Good-Thomas: 2 x bf5 + 5 x bf2, no twiddles | **66** |
| 12 | 3 x bf4 + 4 x bf3 | **62** |
| 15 | 3 x bf5 + 5 x bf3 | **124** |
| 16 | 4x4: 8 x bf4 + 4 x `mul_complex` + 1 `rotate` + 2 `rotate_45` + 2 `rotate_135` | **114** |
| 24 | 6 x bf4 + 4 x bf6 + 8 x `mul_complex` + 1 `neg` + 2 `rotate` + 2 `rotate_45` + 1 `rotate_135` + 1 `rotate_225` | **201** |
| 32 | 8x4: 8 x bf4 + 4 x bf8 + 16 x `mul_complex` + 1 `rotate` + 2 `rotate_45` + 2 `rotate_135` | **314** |

Two traps worth recording, both of which produced wrong counts on the first pass:

1. Calls are often split across lines (`self\n    .bf4\n    .perform_fft_direct(...)`), so a
   line-oriented match misses them. Flatten whitespace first.
2. Butterfly 32 calls `self.bf8.bf4.perform_fft_direct` for its eight size-4 column FFTs, a
   three-level path. Missing those made it look cheaper per element than butterfly 16, which was
   the signal that the count was wrong.

## Generated prime butterflies

From `src/neon/neon_prime_butterflies.rs`, which is emitted by
`tools/gen_simd_butterflies/src/main.rs:252-310`. Because the generator is a pair of loops, the
count is a **closed form** rather than a table, so it cannot drift when sizes are added or removed.

With `h = (len + 1) / 2`, the generator emits, for each of the `h - 1` conjugate pairs:

- one `column_butterfly2` (2), one `apply_rotate90` (2) and one `add` (1) in the input stage;
- an a-chain of `h - 1` `fmadd` (1 each);
- a b-chain of one `mul` plus `h - 2` `fmadd`/`nmadd` (1 each);
- one closing `column_butterfly2` (2).

That is `(h-1) * [1 + 1 + (2h-3) + 2 + 4] = (h-1)(2h+5)` instructions.

| len | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|
| h | 4 | 6 | 7 | 9 | 10 | 12 | 15 | 16 |
| compute | **39** | **85** | **114** | **184** | **225** | **319** | **490** | **555** |

Verified against a direct histogram of the generated source for all eight lengths: exact match.

## Resulting per-element cost

`compute / len`, the number that decides whether a bigger RadixN base pays for itself:

| len | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 15 | 16 | 17 | 19 | 23 | 24 | 29 | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| instr/elem | 1.0 | 2.7 | 2.5 | 5.6 | 3.7 | 5.6 | 4.8 | 7.1 | 6.6 | 7.7 | 5.2 | 8.8 | 8.3 | 7.1 | 10.8 | 11.8 | 13.9 | 8.4 | 16.9 | 17.9 | 9.8 |

The shape is the expected one: powers of two are cheapest per element, the prime butterflies grow
roughly linearly in `len` because they are O(n^2) kernels, and the composites sit in between.

## SSE f64

Counted the same way, from `src/sse/sse_vector.rs`, `src/sse/sse_utils.rs` and
`src/sse/sse_butterflies.rs`. The decompositions are the same as NEON's; the primitive costs are
not, because SSE4.1 has no FMA.

| primitive | NEON | SSE | SSE derivation |
|---|---|---|---|
| `fmadd` / `nmadd` | 1 | **2** | `_mm_mul_pd` plus `_mm_add_pd` / `_mm_sub_pd` |
| `mul_complex` | 4 | **6** | `_mm_unpacklo_pd`, `_mm_unpackhi_pd`, two `_mm_mul_pd`, `_mm_shuffle_pd`, `_mm_addsub_pd` |
| `apply_rotate90`, `Rotate90F64::rotate` | 2 | 2 | `_mm_shuffle_pd` + `_mm_xor_pd` |
| `rotate_45` / `_135` / `_225` | 4 | 4 | rotate, then add or sub, then mul |
| `column_butterfly2`, `solo_fft2_f64` | 2 | 2 | add + sub |
| `column_butterfly4` | 10 | 10 | four `column_butterfly2` plus one `apply_rotate90` |
| `add`, `mul`, `neg`, load, store | 1 | 1 | one instruction each |

Prime butterflies re-derive to `(h-1)(4h+2)`, since only the `fmadd` chain changes weight. Verified
exactly against a histogram of `src/sse/sse_prime_butterflies.rs` for all eight lengths: 7 -> 54,
11 -> 130, 13 -> 180, 17 -> 304, 19 -> 378, 23 -> 550, 29 -> 868, 31 -> 990.

Hand-written butterflies. **Two of these are not the NEON figure re-weighted**, which is why the
SSE source was counted rather than scaled:

| len | compute | note |
|---|---|---|
| 1 | 0 | |
| 2 | 2 | |
| 3 | **10** | written without FMA as 4 add + 2 mul + 2 sub + 1 rotate, not NEON's 8 re-weighted to 11 |
| 4 | 10 | |
| 5 | 28 | |
| 6 | 26 | 2 x bf3 + 3 x solo_fft2 |
| 8 | **38** | uses `rotate_45` and `rotate_135` where NEON uses explicit multiplies; lands on the same total by coincidence |
| 9 | 84 | 6 x bf3 + 4 x mul_complex |
| 10 | 66 | 2 x bf5 + 5 x bf2, Good-Thomas, no twiddles |
| 12 | 70 | 3 x bf4 + 4 x bf3 |
| 15 | 134 | 3 x bf5 + 5 x bf3 |
| 16 | 122 | 8 x bf4 + 4 x mul_complex + rotations |
| 24 | 233 | 6 x bf4 + 4 x bf6 + 8 x mul_complex + neg + rotations |
| 32 | 346 | 8 x bf4 + 4 x bf8 + 16 x mul_complex + rotations |

As on NEON, `src/sse/sse_radixn.rs` wires its cross-FFT layers to these same
`SseF64ButterflyN::perform_fft_direct` functions, so the table applies to RadixN directly.

## NEON f32

Counted from `perform_parallel_fft_direct`, which is what `src/neon/neon_radixn.rs` wires the
cross-FFT layers to for f32 and what the `Fft` boilerplate uses for chunk pairs. That function
computes **two** FFTs at once, so the figures below are the raw count halved, giving a per-FFT cost
directly comparable with the f64 table.

Primitive differences from f64, read from `impl NeonVector for float32x4_t`: `nmadd` costs 2 rather
than 1 (`vfmaq_f32` plus `vnegq_f32`) and `mul_complex` costs 6 rather than 4 (`vtrn1q`, `vtrn2q`,
`vnegq`, `vmulq`, `vrev64q`, `vfmaq`). The f32 rotate helpers in `neon_utils.rs` are `rotate_both`
at 2, `rotate_hi` at 3, and `rotate_both_45/135/225` at 4.

| len | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 15 | 16 | 17 | 19 | 23 | 24 | 29 | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| per FFT | 2 | 4 | 5 | 16 | 11 | 19.5 | 19 | 36 | 37 | 44.5 | 31 | 61.5 | 68 | 66 | 100 | 125.5 | 180 | 113.5 | 279.5 | 321 | 178 |

Per element these are 0.50x to 0.58x the f64 figures, except length 2 at 1.00x, where the packing
needed by `parallel_fft2_contiguous_f32` eats the whole gain. That near-uniformity is why the table
turns out not to matter; see the f32 section of `RESULTS.md`.
