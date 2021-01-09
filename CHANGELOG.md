## [5.0.1]
Released 8 January 2021
### Fixed
 - Fixed the FFT planner not choosing an obviously faster plan in some rare cases (#46)
 - Documentation fixes and clarificarions (#47, #48, #51)

## [5.0.0]
Released 4 January 2021
### Breaking Changes
- Several breaking changes. See the [Upgrade Guide](/UpgradeGuide4to5.md) for details.

### Added
- Added support for the `Avx` instruction set. Plan a FFT with the `FftPlanner` on a machine that supports AVX, and you'll get a 5x-10x speedup in FFT performance.

### Changed
- Even though the main focus of this release is on AVX, most users should see moderate performance improvements due to a new internal architecture that reduces the amount of internal copies required when computing a FFT.

## [4.1.0]
Released 24 December 2020
### Added
- Added a blanket impl of `FftNum` to any type that implements the required traits (#7)
- Added butterflies for many prime sizes, up to 31, and optimized the size-3, size-5, and size-7 buitterflies (#10)
- Added an implementation of Bluestein's Algorithm (#6)

### Changed
- Improved the performance of GoodThomasAlgorithm re-indexing (#20)

## [4.0.0]
Released 8 October 2020

This release moved the home repository of RustFFT from https://github.com/awelkie/RustFFT to https://github.com/ejmahler/RustFFT

### Breaking Changes
- Increased the version of the num-complex dependency to 0.3. This is a breaking change because we have a public dependency on num-complex.
See the [num-complex changelog](https://github.com/rust-num/num-complex/blob/master/RELEASES.md) for a list of breaking changes in num-complex 0.3.
- Increased the minimum required Rust version from 1.26 to 1.31. This was required by the upgrade to num-complex 0.3.


## [3.0.1]
Released 27 December 2019
### Fixed
- Fixed warnings regarding "dyn trait", and warnings regarding inclusive ranges
- Several documentation improvements

## [3.0.0]
Released 4 January 2019
### Changed
- Reduced the setup time and memory usage of GoodThomasAlgorithm
- Reduced the setup time and memory usage of RadersAlgorithm

### Breaking Changes
- Documented the minimum rustsc version. Before, none was specified. now, it's 1.26. Further increases to minimum version will be a breaking change.
- Increased the version of the num-complex dependency to 0.2. This is a breaking change because we have a public dependency on num-complex.
See the [num-complex changelog](https://github.com/rust-num/num-complex/blob/master/RELEASES.md) for a list of breaking changes in num-complex 0.2

## [2.1.0]
Released 30 July 2018
### Added
- Added a specialized implementation of Good Thomas Algorithm for when both inner FFTs are butterflies

### Changed
- Documentation typo fixes
- Increased minimum version of num_traits and num_complex. Notably, Complex<T> is now guaranteed to be repr(C)
- Significantly improved the performance of the Radix4 algorithm
- Reduced memory usage of prime-sized FFTs
- Incorporated the Good-Thomas Double Butterfly algorithm into the planner, improving performance for most composite and prime FFTs

## [2.0.0]
Released 22 May 2017
### Added
- Added implementation of Good Thomas algorithm.
- Added implementation of Raders algorithm.
- Added implementation of Radix algorithm for power-of-two lengths.
- Added `FFTPlanner` to choose the fastest algorithm for a given size.

### Changed
- Changed API to take the "signal" as mutable and use it for scratch space.

## [1.0.1]
Released 15 January 2016
### Changed
- Relicensed to dual MIT/Apache-2.0.

## [1.0.0]
Released 4 October 2015
### Added
- Added initial implementation of Cooley-Tukey.
