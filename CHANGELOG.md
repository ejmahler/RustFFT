## [5.0.0-experimental.2]
### Fixed
- Removed a stray debug in the AVX planner

## [5.0.0-experimental.1]
### Added
- Added support for the AVX instruction set. Plan a FFT from a machine that supports AVX, and you'll get AVX-accelerated FFTs.

### Breaking Changes
- Restructured the Fft trait to be much more flexible about scratch space
- Renamed several structs and traits to fit more in line with idiomatic Rust naming conventions
- Changed MSRV to nightly instead of stable 1.31


## [4.0.0]
### Breaking Changes
- Increased the version of the num-complex dependency to 0.3. This is a breaking change because we have a public dependency on num-complex.
See the [num-complex changelog](https://github.com/rust-num/num-complex/blob/master/RELEASES.md) for a list of breaking changes in num-complex 0.3.
- Increased the minimum required Rust version from 1.26 to 1.31. This was required by the upgrade to num-complex 0.3.


## [3.0.1]
### Fixed
- Fixed warnings regarding "dyn trait", and warnings regarding inclusive ranges
- Several documentation improvements

## [3.0.0]
### Changed
- Reduced the setup time and memory usage of GoodThomasAlgorithm
- Reduced the setup time and memory usage of RadersAlgorithm

### Breaking Changes
- Documented the minimum rustsc version. Before, none was specified. now, it's 1.26. Further increases to minimum version will be a breaking change.
- Increased the version of the num-complex dependency to 0.2. This is a breaking change because we have a public dependency on num-complex.
See the [num-complex changelog](https://github.com/rust-num/num-complex/blob/master/RELEASES.md) for a list of breaking changes in num-complex 0.2

## [2.1.0]
### Added
- Added a specialized implementation of Good Thomas Algorithm for when both inner FFTs are butterflies. (#33)

### Changed
- Documentation typo fixes (#27, #35)
- Increased minimum version of num_traits and num_complex. Notably, Complex<T> is now guaranteed to be repr(C)
- Significantly improved the performance of the Radix4 algorithm (#26)
- Reduced memory usage of prime-sized FFTs (#34)
- Incorporated the Good-Thomas Double Butterfly algorithm into the planner, improving performance for most composite and prime FFTs

## [2.0.0]
### Added
- Added implementation of Good Thomas algorithm.
- Added implementation of Raders algorithm.
- Added implementation of Radix algorithm for power-of-two lengths.
- Added `FFTPlanner` to choose the fastest algorithm for a given size.

### Changed
- Changed API to take the "signal" as mutable and use it for scratch space.

## [1.0.1]
### Changed
- Relicensed to dual MIT/Apache-2.0.

## [1.0.0]
### Added
- Added initial implementation of Cooley-Tukey.
