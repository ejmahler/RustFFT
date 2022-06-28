# RustFFT

[![CI](https://github.com/ejmahler/RustFFT/workflows/CI/badge.svg)](https://github.com/ejmahler/RustFFT/actions?query=workflow%3ACI)
[![](https://img.shields.io/crates/v/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://img.shields.io/crates/l/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://docs.rs/rustfft/badge.svg)](https://docs.rs/rustfft/)
![minimum rustc 1.37](https://img.shields.io/badge/rustc-1.37+-red.svg)

RustFFT is a high-performance FFT library written in pure Rust. It can compute FFTs of any size, including prime-number sizes, in O(nlogn) time.

Unlike previous major versions, RustFFT 5.0 has several breaking changes compared to RustFFT 4.0. Check out the [Upgrade Guide](/UpgradeGuide4to5.md) for a walkthrough of the changes RustFFT 5.0 requires.

## SIMD acceleration
### x86_64
RustFFT supports the AVX instruction set for increased performance. No special code is needed to activate AVX: Simply plan a FFT using the FftPlanner on a machine that supports the `avx` and `fma` CPU features, and RustFFT will automatically switch to faster AVX-accelerated algorithms.

For machines that do not have AVX, it also supports the SSE4.1 instruction set. As for AVX, this is enabled automatically when using the FftPlanner.

### AArch64
RustFFT optionally supports the NEON instruction set in 64-bit Arm, AArch64. This support requires a newer rustc version, and is disabled by default. See [Features](#features) for more details.


## Usage

```rust
// Perform a forward FFT of size 1234
use rustfft::{FftPlanner, num_complex::Complex};

let mut planner = FftPlanner::<f32>::new();
let fft = planner.plan_fft_forward(1234);

let mut buffer = vec![Complex{ re: 0.0, im: 0.0 }; 1234];

fft.process(&mut buffer);
```

## Supported Rust Version

RustFFT requires rustc 1.37 or newer. Minor releases of RustFFT may upgrade the MSRV(minimum supported Rust version) to a newer version of rustc.
However, if we need to increase the MSRV, the new Rust version must have been released at least six months ago.

## Features

The features `avx` and `sse` are enabled by default. On x86_64, these features enable compilation of the AVX and SSE accelerated code. 

Disabling them reduces compile time and binary size.

On other platform than x86_64, these features do nothing and RustFFT will behave like they are not set.

On AArch64, the `neon` feature enables compilation of Neon-accelerated code. This requires rustc 1.61 or newer, and is disabled by default.

## Stability/Future Breaking Changes

Version 5.0 contains several breaking API changes. In the interest of stability, we're committing to making no more breaking changes for 3 years, aka until 2024.

This policy has one exception: We currently re-export pre-1.0 versions of the [num-complex](https://crates.io/crates/num-complex) and [num-traits](https://crates.io/crates/num-traits) crates. If those crates release new major versions, we will upgrade as soon as possible, which will require a major version change of our own. If this happens, the version increase of num-complex/num-traits will be the only breaking change.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

Before submitting a PR, please make sure to run `cargo fmt`.
