# RustFFT

[![Build Status](https://travis-ci.org/awelkie/RustFFT.svg?branch=master)](https://travis-ci.org/awelkie/RustFFT)
[![](https://img.shields.io/crates/v/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://img.shields.io/crates/l/rustfft.svg)](https://crates.io/crates/rustfft)
[![](https://docs.rs/rustfft/badge.svg)](https://docs.rs/rustfft/)

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
