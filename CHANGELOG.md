## [6.1]
Released 7th Novemeber 2022
### Added
 - Implemented a code path for Neon-optimized FFTs on AArch64 (Thanks to Henrik Enquist!) (#84 and #78)
### Changed
 - Improved performance of power-of-3 FFTs when not using SIMD-accelerated code paths (#80)
 - Reduced memory usage for some FFT sizes (#81)

## [6.0.1]
Released 10 May 2021
### Fixed
 - Fixed a compile-time divide by zero error on nightly Rust in `stdarch\crates\core_arch\src\macros.rs` (#75)
 - Increased the minimum version of `strength_reduce` to 0.2.3


## [6.0.0]
Released 16 April 2021
### Breaking Changes
- Increased the version of the num-complex dependency to 0.4.
    - This is a breaking change because we have a public dependency on num-complex.
    - See the [num-complex changelog](https://github.com/rust-num/num-complex/blob/master/RELEASES.md) for a list of breaking changes in num-complex 0.4
    - As a high-level summary, most users will not need to do anything to upgrade to RustFFT 6.0: num-complex 0.4 re-exports a newer version of `rand`, and that's num-complex's only documented breaking change.

## [5.1.1]
Released 10 May 2021
### Fixed
 - Fixed a compile-time divide by zero error on nightly Rust in `stdarch\crates\core_arch\src\macros.rs` (Backported from v6.0.1)
 - Increased the minimum version of `strength_reduce` to 0.2.3  (Backported from v6.0.1)

## [5.1.0]
Released 16 April 2021
### Added
 - Implemented a code path for SSE-optimized FFTs (Thanks to Henrik Enquist!) (#60)
     - Plan a FFT using the `FftPlanner` (or the new `FftPlannerSse`) on a machine that supports SSE4.1 (but not AVX) and you'll see a 2-3x performance improvement over the default scalar code.
### Fixed
 - Fixed underflow when planning an AVX FFT of size zero (#56)
 - Fixed the FFT planner not being Send, due to internal use of Rc<> (#55)
 - Fixed typo in documentation (#54)
 - Slightly improved numerical precision of Rader's Algorithm and Bluestein's Algorithm (#66, #68)
 - Minor optimizations to Rader's Algorithm and Bluestein's Algorithm (#59)
 - Minor optimizations to MixedRadix setup time (#57)
 - Optimized performance of Radix4 (#65)

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
