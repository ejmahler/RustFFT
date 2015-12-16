# RustFFT

[![Build Status](https://travis-ci.org/awelkie/RustFFT.svg?branch=master)](https://travis-ci.org/awelkie/RustFFT)
[![](https://img.shields.io/crates/v/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://img.shields.io/crates/l/rustfft.svg)](https://crates.io/crates/rustfft)

RustFFT is a mixed-radix FFT implementation written in Rust.

## Example
```rust
extern crate rustfft;
extern crate num;

// This library can handle arbitrary FFT lengths, but
// lengths that are highly composite run much faster.
let fft_len = 1234;

let mut fft = rustfft::FFT::new(fft_len, false);
let signal = vec![num::Complex{re: 0.0, im: 0.0}; fft_len];
let mut spectrum = signal.clone();
fft.process(&signal, &mut spectrum);
```
