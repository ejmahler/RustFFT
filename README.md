# RustFFT

[![CI](https://github.com/ejmahler/RustFFT/workflows/CI/badge.svg)](https://github.com/ejmahler/RustFFT/actions?query=workflow%3ACI)
[![](https://img.shields.io/crates/v/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://img.shields.io/crates/l/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://docs.rs/rustfft/badge.svg)](https://docs.rs/rustfft/)
![minimum rustc nightly](https://img.shields.io/badge/rustc-nightly-red.svg)

RustFFT is a high-performance FFT library written in pure Rust. See the [documentation](https://docs.rs/rustfft/) for more details.

This is an experimental release of RustFFT that enables AVX acceleration. It currently requires a nightly compiler,
mainly for the `min_specialization` feature. The eventual plan is to release this experimental version as version 5.0 of RustFFT,
but that will not happen until it compiles on stable Rust.

No special code is needed to activate AVX: Simply plan a FFT using the FftPlanner on a machine that supports the `avx` and `fma` features.

## Usage

```rust
// Perform a forward FFT of size 1234
use rustfft::{FftPlanner, num_complex::Complex};

let mut planner = FftPlanner::new(false);
let fft = planner.plan_fft(1234);

let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
fft.process_inplace(&mut buffer);
```

## Compatibility

This experimental version of `rustfft` crate requires nightly Rust.

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
