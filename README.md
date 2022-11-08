# RustFFT

[![CI](https://github.com/ejmahler/RustFFT/workflows/CI/badge.svg)](https://github.com/ejmahler/RustFFT/actions?query=workflow%3ACI)
[![](https://img.shields.io/crates/v/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://img.shields.io/crates/l/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://docs.rs/rustfft/badge.svg)](https://docs.rs/rustfft/)
![minimum rustc 1.37](https://img.shields.io/badge/rustc-1.37+-red.svg)

RustFFT is a high-performance, SIMD-accelerated FFT library written in pure Rust. It can compute FFTs of any size, including prime-number sizes, in O(nlogn) time.

## Usage

```rust
// Perform a forward FFT of size 1234
use rustfft::{FftPlanner, num_complex::Complex};

let mut planner = FftPlanner::<f32>::new();
let fft = planner.plan_fft_forward(1234);

let mut buffer = vec![Complex{ re: 0.0, im: 0.0 }; 1234];

fft.process(&mut buffer);
```

## SIMD acceleration
### x86_64
RustFFT supports the AVX instruction set for increased performance. No special code is needed to activate AVX: Simply plan a FFT using the FftPlanner on a machine that supports the `avx` and `fma` CPU features, and RustFFT will automatically switch to faster AVX-accelerated algorithms.

For machines that do not have AVX, RustFFT also supports the SSE4.1 instruction set. As for AVX, this is enabled automatically when using the FftPlanner. If both AVX and SSE4.1 support are enabled, the planner will automatically choose the fastest available instruction set.

### AArch64
RustFFT optionally supports the NEON instruction set in 64-bit Arm, AArch64. This optional feature requires a newer rustc version: Rustc 1.61. See [Cargo Features](#cargo-features) below for more details.

## Cargo Features
### x86_64
The features `avx` and `sse` are enabled by default. On x86_64, these features enable compilation of AVX and SSE accelerated code. 

Disabling them reduces compile time and binary size.

On other platforms than x86_64, these features do nothing and RustFFT will behave like they are not set.

### AArch64
On AArch64, the `neon` feature enables compilation of Neon-accelerated code. This requires rustc 1.61 or newer, and is enabled by default. If this feature is disabled, rustc 1.37 or newer is required.

On other platforms than AArch64, this feature does nothing and RustFFT will behave like it is not set.

## Stability/Future Breaking Changes

Version 5.0 contains several breaking API changes. heck out the [Upgrade Guide](/UpgradeGuide4to5.md) for a walkthrough of the changes RustFFT 5.0 requires. In the interest of stability, we're committing to making no more breaking changes for 3 years, aka until 2024.

This policy has one exception: We currently re-export pre-1.0 versions of the [num-complex](https://crates.io/crates/num-complex) and [num-traits](https://crates.io/crates/num-traits) crates. In the interest of avoiding version fragmentation, we will keep up with these crates even if it requires major version bumps. When those crates release new major versions, we will upgrade as soon as possible, which will require a major version change of our own. In this situations, the version increase of num-complex/num-traits will be the only breaking change in the release.

### Supported Rust Version

RustFFT requires rustc 1.37 or newer. Minor releases of RustFFT may upgrade the MSRV(minimum supported Rust version) to a newer version of rustc.
However, if we need to increase the MSRV, the new Rust version must have been released at least six months ago.

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
