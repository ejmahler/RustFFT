# RustFFT

[![CI](https://github.com/ejmahler/RustFFT/workflows/CI/badge.svg)](https://github.com/ejmahler/RustFFT/actions?query=workflow%3ACI)
[![](https://img.shields.io/crates/v/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://img.shields.io/crates/l/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://docs.rs/rustfft/badge.svg)](https://docs.rs/rustfft/)
![minimum rustc 1.31](https://img.shields.io/badge/rustc-1.31+-red.svg)

RustFFT is a mixed-radix FFT implementation written in Rust. See the [documentation](https://docs.rs/rustfft/) for more details.

If you're looking for the experimental AVX-accelerated release, check out the [SIMD branch](https://github.com/ejmahler/RustFFT/tree/simd).

## Compatibility

The `rustfft` crate requires rustc 1.31 or greater.

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
