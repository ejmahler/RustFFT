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
