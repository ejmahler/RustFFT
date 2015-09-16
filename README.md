# RustFFT

[![Build Status](https://travis-ci.org/awelkie/RustFFT.svg?branch=master)](https://travis-ci.org/awelkie/RustFFT)
[![](https://img.shields.io/crates/v/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://img.shields.io/crates/l/rustfft.svg)](https://crates.io/crates/rustfft)

RustFFT is a mixed-radix FFT implementation written in Rust. It aims to be about as fast as [KissFFT](http://kissfft.sourceforge.net/).

## Testing
To run tests with benchmarks (requires nightly rust), use the `bench` feature:
``` sh
cargo test --features bench
```
