//! Self-sorting (Stockham) FFT for powers of two, shared by every SIMD backend.
//!
//! Each pass reads one buffer and writes another, both contiguously, so the transform needs
//! neither transposes nor a bit reversal. After the first pass every butterfly of a row shares
//! its twiddles, which are broadcast once per row.

use std::any::TypeId;
use std::marker::PhantomData;

use num_complex::Complex;
use num_traits::{One, Zero};

use crate::array_utils::{DoubleBuf, workaround_transmute, workaround_transmute_mut};
use crate::simd::parallel_fft::ParallelFft;
use crate::simd::simd_array::SimdComplexArrayMut;
use crate::simd::simd_vector::SimdVector;
use crate::twiddles::compute_twiddle;
use crate::{FftDirection, FftNum};

/// A later pass, whose butterflies span every lane: `rows` of them, each `stride` complex numbers
/// wide. Every butterfly of a row shares its twiddle factors.
struct Pass {
    radix: usize,
    rows: usize,
    stride: usize,
    /// Where this pass's twiddles start in `SimdStockham::twiddles`: `radix - 1` per row, or none
    /// if there is only one row.
    twiddle_start: usize,
}

/// A Stockham FFT whose first pass is done by `first_pass` if `FIRST_PASS` is true or a vector
/// holds more than one complex number, and by a regular pass of stride 1 otherwise. With NEON f64
/// on Apple M4, the later passes run up to 20% slower after a `first_pass`, so NEON passes false.
/// FCMA is a little faster with `first_pass`.
pub struct SimdStockham<V: SimdVector, T, const FIRST_PASS: bool> {
    /// The twiddles of `first_pass`, `radix - 1` per vector of rows. Empty if there is none.
    last_twiddles: Box<[V::Twiddle]>,
    /// The twiddles of every pass in `passes`, in one allocation.
    twiddles: Box<[Complex<V::ScalarType>]>,
    /// Every pass, the first first. If there is a `first_pass`, only its radix is used.
    passes: Box<[Pass]>,

    butterfly2: V::Butterfly2,
    butterfly4: V::Butterfly4,
    butterfly8: V::Butterfly8,
    butterfly16: V::Butterfly16,

    len: usize,
    direction: FftDirection,
    _phantom: PhantomData<T>,
}

/// The planners compute a power of two with Stockham passes alone as long as one buffer is at
/// most this many bytes, and with radix 4 above that.
///
/// Every Stockham pass reads one whole buffer and writes another, so once the two no longer fit
/// in the cache it is limited by memory traffic, and the in-place radix 4 layers have less of
/// that. Where this starts depends on the CPU. On a Raspberry Pi 5 (512 KiB L2, 2 MiB L3),
/// Stockham is ahead of radix 4 up to 256 KiB per buffer.
pub const STOCKHAM_MAX_BYTES: usize = 256 << 10;

/// The longest FFT of `T` whose buffer is at most `STOCKHAM_MAX_BYTES`, or `apple_max_bytes` on
/// Apple targets. Apple's CPUs have large caches and a lot of memory bandwidth, so each planner
/// has its own limit for them.
pub const fn stockham_max_len<T>(apple_max_bytes: usize) -> usize {
    let max_bytes = if cfg!(target_vendor = "apple") {
        apple_max_bytes
    } else {
        STOCKHAM_MAX_BYTES
    };
    max_bytes / std::mem::size_of::<Complex<T>>()
}

/// The buffers that passes go back and forth between start `GAP` bytes apart modulo
/// `ALIASING_PERIOD`, rounded down to a cache line. Half the period is the same distance in both
/// directions. With 1 KiB, which was used when every pass wrote a new buffer, NEON was up to 8%
/// slower on Apple M4 once passes went back and forth between two buffers.
///
/// A pass reads and writes its two buffers in lockstep. If both start at the same offset into a
/// 4 KiB page, as large allocations tend to, every load and the store next to it map onto the
/// same L1 cache sets, and on many CPUs a load whose low 12 address bits match those of a pending
/// store also waits for that store ("4K aliasing"). Offsetting the buffers avoids both. Without
/// it, NEON and FCMA are 5 to 9% slower from 2^14 points up on Apple M4.
const ALIASING_PERIOD: usize = 4096;
const GAP: usize = ALIASING_PERIOD / 2;
const CACHE_LINE: usize = 64;

/// How many elements `place()` may skip at most, so how much longer than the buffer it places
/// the slice it's given has to be.
const fn placement_padding<C>() -> usize {
    ALIASING_PERIOD / std::mem::size_of::<C>()
}

/// The first `len` elements of `slice` from `gap` bytes, modulo `ALIASING_PERIOD`, after the
/// start of `other`, a buffer the passes read it from or write it to.
fn place<'a, C>(slice: &'a mut [C], other: &[C], gap: usize, len: usize) -> &'a mut [C] {
    let target = (other.as_ptr() as usize + gap) % ALIASING_PERIOD / CACHE_LINE * CACHE_LINE;
    let at = slice.as_ptr() as usize % ALIASING_PERIOD;
    let skip = (target + ALIASING_PERIOD - at) % ALIASING_PERIOD;
    &mut slice[skip / std::mem::size_of::<C>()..][..len]
}

/// Where the data is during `SimdStockham::run`. `Output`, `Spare` and `Extra` are its three buffers.
#[derive(Copy, Clone, PartialEq)]
enum Slot {
    Input,
    Output,
    Spare,
    Extra,
}

impl<V: SimdVector, T: FftNum, const FIRST_PASS: bool> SimdStockham<V, T, FIRST_PASS> {
    const HAS_LAST_PASS: bool = FIRST_PASS || V::COMPLEX_PER_VECTOR > 1;

    /// Preallocates the twiddles for a `len`-point FFT computed with passes of `radixes`, the
    /// first pass first. `len` is a power of two of at least 16 and the product of `radixes`.
    pub fn new(len: usize, direction: FftDirection, radixes: &[usize]) -> Self {
        assert_eq!(TypeId::of::<V::ScalarType>(), TypeId::of::<T>());
        assert!(
            len.is_power_of_two() && len >= 16,
            "invalid Stockham length {len}"
        );
        assert_eq!(
            radixes.iter().product::<usize>(),
            len,
            "radixes {radixes:?}"
        );
        assert!(radixes.len() >= 2, "radixes {radixes:?}");
        assert!(V::COMPLEX_PER_VECTOR <= 2);

        let mut last_twiddles = Vec::new();
        let mut twiddles = Vec::new();
        let mut passes = Vec::with_capacity(radixes.len());
        let (mut span, mut stride) = (1, len);
        for (index, &radix) in radixes.iter().enumerate().rev() {
            let rows = span;
            span *= radix;
            stride /= radix;

            let twiddle_start = twiddles.len();

            // Handle the final pass separately
            if index == 0 && Self::HAS_LAST_PASS {
                // Each lane holds its own row, so each needs its own twiddle
                for row in (0..rows).step_by(V::COMPLEX_PER_VECTOR) {
                    for output in 1..radix {
                        let lane_twiddles: [Complex<V::ScalarType>; 2] =
                            std::array::from_fn(|lane| {
                                compute_twiddle((row + lane) * output, len, direction)
                            });
                        last_twiddles.push(unsafe {
                            V::make_twiddle(&lane_twiddles[..V::COMPLEX_PER_VECTOR])
                        });
                    }
                }
            } else if rows > 1 {
                for row in 0..rows {
                    for output in 1..radix {
                        twiddles.push(compute_twiddle(row * output, span, direction));
                    }
                }
            }
            passes.push(Pass {
                radix,
                rows,
                stride,
                twiddle_start,
            });
        }
        assert_eq!(span, len);

        unsafe {
            Self {
                last_twiddles: last_twiddles.into_boxed_slice(),
                twiddles: twiddles.into_boxed_slice(),
                passes: passes.into_boxed_slice(),
                butterfly2: V::make_butterfly2(direction),
                butterfly4: V::make_butterfly4(direction),
                butterfly8: V::make_butterfly8(direction),
                butterfly16: V::make_butterfly16(direction),
                len,
                direction,
                _phantom: PhantomData,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn fft_direction(&self) -> FftDirection {
        self.direction
    }

    /// Scratch for `buffers` buffers, each placed by `place()`.
    fn scratch_len(&self, buffers: usize) -> usize {
        buffers * (self.len + placement_padding::<Complex<T>>())
    }

    fn odd_pass_count(&self) -> bool {
        self.passes.len() % 2 == 1
    }

    /// In place, the passes go back and forth between the buffer and the scratch. That ends in
    /// the buffer only for an even number of passes, so an odd number needs a second scratch
    /// buffer, which `run` uses once.
    pub fn get_inplace_scratch_len(&self) -> usize {
        self.scratch_len(if self.odd_pass_count() { 2 } else { 1 })
    }

    /// Out of place, the first pass reads the input and writes either the output or the scratch,
    /// so that the passes going back and forth between the two end in the output.
    ///
    /// The input could be the second buffer instead of the scratch, which would need no scratch
    /// for an odd number of passes. But a caller's input and output usually start at the same
    /// offset into a page, and `place()` can't move them apart. With both page-aligned, that was
    /// 5 to 14% slower from 2^14 points up on Apple M4.
    pub fn get_outofplace_scratch_len(&self) -> usize {
        self.scratch_len(1)
    }

    pub fn get_immutable_scratch_len(&self) -> usize {
        self.scratch_len(1)
    }

    #[inline(never)]
    pub fn process_single(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let buffer: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(buffer) };
        let scratch: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(scratch) };
        let (spare, extra) = scratch.split_at_mut(self.scratch_len(1));
        let spare = place(spare, buffer, GAP, self.len);
        // `extra` is written from `buffer` and read into `spare`, so it goes half way between
        let extra = match self.odd_pass_count() {
            true => place(extra, buffer, GAP / 2, self.len),
            false => &mut [],
        };
        self.run(Slot::Output, &[], buffer, spare, extra);
    }

    #[inline(never)]
    pub fn process_single_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.process_single_immutable(input, output, scratch);
    }

    #[inline(never)]
    pub fn process_single_immutable(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let input: &[Complex<V::ScalarType>] = unsafe { workaround_transmute(input) };
        let output: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(output) };
        let scratch: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(scratch) };
        let spare = place(scratch, output, GAP, self.len);
        self.run(Slot::Input, input, output, spare, &mut []);
    }

    /// Runs every pass on the data in `start`, ending in `output`. The passes go back and forth
    /// between `output` and `spare`. In place with an odd number of passes that would end in
    /// `spare`, so the first pass writes `extra` instead, which only that pass and the next use.
    /// `extra` may be empty if it's not needed.
    ///
    /// Every pass goes through the one `run_pass` call here, so each pass function is only
    /// generated once per caller.
    #[inline(never)]
    fn run(
        &self,
        start: Slot,
        input: &[Complex<V::ScalarType>],
        output: &mut [Complex<V::ScalarType>],
        spare: &mut [Complex<V::ScalarType>],
        extra: &mut [Complex<V::ScalarType>],
    ) {
        let mut source = start;
        for index in 0..self.passes.len() {
            // Ending in `output` needs an odd number of passes left when coming from elsewhere
            let left = self.passes.len() - index;
            let destination = match (left % 2 == 1, source) {
                (true, Slot::Output) => Slot::Extra,
                (true, _) => Slot::Output,
                (false, _) => Slot::Spare,
            };
            let (src, dst): (&[_], &mut [_]) = match (source, destination) {
                (Slot::Input, Slot::Output) => (input, &mut *output),
                (Slot::Input, Slot::Spare) => (input, &mut *spare),
                (Slot::Output, Slot::Spare) => (&*output, &mut *spare),
                (Slot::Output, Slot::Extra) => (&*output, &mut *extra),
                (Slot::Spare, Slot::Output) => (&*spare, &mut *output),
                (Slot::Extra, Slot::Spare) => (&*extra, &mut *spare),
                _ => unreachable!(),
            };
            self.run_pass(
                index,
                DoubleBuf {
                    input: src,
                    output: dst,
                },
            );
            source = destination;
        }
        debug_assert!(source == Slot::Output);
    }

    #[inline(never)]
    fn run_pass(&self, index: usize, buffer: DoubleBuf<V::ScalarType>) {
        let pass = &self.passes[index];
        unsafe {
            if index == 0 {
                match pass.radix {
                    2 => base_pass(&self.butterfly2, pass, buffer),
                    4 => base_pass(&self.butterfly4, pass, buffer),
                    8 => base_pass(&self.butterfly8, pass, buffer),
                    16 => base_pass(&self.butterfly16, pass, buffer),
                    _ => unreachable!(),
                }
            } else if Self::HAS_LAST_PASS && index == self.passes.len() - 1 {
                let twiddles = &self.last_twiddles[..];
                match pass.radix {
                    2 => last_pass(&self.butterfly2, self.len, twiddles, buffer),
                    4 => last_pass(&self.butterfly4, self.len, twiddles, buffer),
                    8 => last_pass(&self.butterfly8, self.len, twiddles, buffer),
                    16 => last_pass(&self.butterfly16, self.len, twiddles, buffer),
                    _ => unreachable!(),
                }
            } else {
                let twiddles = &self.twiddles[pass.twiddle_start..];
                match pass.radix {
                    2 => main_pass(&self.butterfly2, pass, twiddles, buffer),
                    4 => main_pass(&self.butterfly4, pass, twiddles, buffer),
                    8 => main_pass(&self.butterfly8, pass, twiddles, buffer),
                    16 => main_pass(&self.butterfly16, pass, twiddles, buffer),
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// Radixes for CPUs with fused multiply-add and FCMA, the first pass first, for vectors of
/// `lanes` complex numbers. Sizes up to 2^20 use the fastest plan measured on Apple M4.
#[cfg_attr(not(all(target_arch = "aarch64", feature = "fcma")), allow(dead_code))]
pub fn fused_radixes(len: usize, lanes: usize) -> Vec<usize> {
    let bits = len.trailing_zeros() as usize;
    let measured: &[usize] = match (bits, lanes) {
        (4, _) => &[4, 4],
        (5, _) => &[4, 8],
        (6, _) => &[8, 8],
        (7, 1) => &[8, 4, 4],
        (7, _) => &[4, 4, 8],
        (8, 1) => &[8, 4, 8],
        (8, _) => &[16, 16],
        (9, _) => &[8, 8, 8],
        (10, _) => &[8, 8, 16],
        (11, 1) => &[8, 8, 4, 8],
        (11, _) => &[8, 4, 8, 8],
        (12, 1) => &[8, 8, 8, 8],
        (12, _) => &[16, 16, 16],
        (13, _) => &[4, 8, 8, 4, 8],
        (14, 1) => &[8, 8, 8, 8, 4],
        (14, _) => &[4, 8, 8, 8, 8],
        (15, _) => &[8, 8, 8, 8, 8],
        (16, _) => &[4, 8, 8, 8, 8, 4],
        (17, 1) => &[8, 8, 8, 8, 4, 8],
        (17, _) => &[4, 4, 8, 8, 8, 4, 4],
        (18, 1) => &[4, 8, 8, 8, 8, 4, 4],
        (18, _) => &[4, 4, 8, 8, 8, 4, 8],
        (19, 1) => &[8, 8, 8, 8, 8, 4, 4],
        (19, _) => &[4, 4, 8, 8, 8, 4, 4, 4],
        (20, 1) => &[8, 8, 8, 8, 4, 4, 4, 4],
        (20, _) => &[4, 4, 8, 8, 8, 8, 4, 4],
        _ => &[],
    };
    if measured.is_empty() {
        large_radixes(bits)
    } else {
        measured.to_vec()
    }
}

/// Radixes for NEON without FCMA, the first pass first, for vectors of `lanes` complex numbers.
/// Sizes up to 2^20 use the fastest plan measured on Apple M4.
#[cfg_attr(not(all(target_arch = "aarch64", feature = "neon")), allow(dead_code))]
pub fn neon_radixes(len: usize, lanes: usize) -> Vec<usize> {
    let bits = len.trailing_zeros() as usize;
    let measured: &[usize] = match (bits, lanes) {
        (4, _) => &[4, 4],
        (5, _) => &[4, 8],
        (6, _) => &[8, 8],
        (7, 1) => &[8, 16],
        (7, _) => &[4, 4, 8],
        (8, 1) => &[8, 4, 8],
        (8, _) => &[16, 16],
        (9, _) => &[8, 4, 16],
        (10, _) => &[8, 8, 16],
        (11, 1) => &[8, 8, 2, 16],
        (11, _) => &[16, 8, 16],
        (12, 1) => &[8, 8, 4, 16],
        (12, _) => &[16, 4, 8, 8],
        (13, 1) => &[8, 8, 4, 8, 4],
        (13, _) => &[16, 8, 8, 8],
        (14, _) => &[8, 4, 8, 8, 8],
        (15, _) => &[8, 8, 8, 8, 8],
        (16, 1) => &[4, 8, 8, 8, 8, 4],
        (16, _) => &[8, 8, 8, 8, 4, 4],
        (17, _) => &[4, 4, 4, 8, 8, 4, 8],
        (18, 1) => &[4, 4, 4, 4, 8, 4, 4, 8],
        (18, _) => &[4, 8, 8, 4, 8, 4, 8],
        (19, 1) => &[4, 4, 8, 4, 8, 4, 4, 8],
        (19, _) => &[4, 4, 4, 8, 8, 4, 4, 8],
        (20, _) => &[4, 4, 4, 8, 8, 4, 4, 4, 4],
        _ => &[],
    };
    if measured.is_empty() {
        large_radixes(bits)
    } else {
        measured.to_vec()
    }
}

/// Radixes for sizes above those with their own plan: two radix 4 passes, then radix 8 ones,
/// then at least two radix 4 ones. With NEON that's ahead of or even with radix 4 from 2^21 to
/// 2^26 on Apple M1 and M4.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn large_radixes(bits: usize) -> Vec<usize> {
    let fours = (2..)
        .find(|fours| (bits - 4 - 2 * fours).is_multiple_of(3))
        .unwrap();
    let mut radixes = vec![4, 4];
    radixes.extend(std::iter::repeat_n(8, (bits - 4 - 2 * fours) / 3));
    radixes.extend(std::iter::repeat_n(4, fours));
    radixes
}

/// Radixes for SIMD without fused multiply-add, where radix 4 passes are cheaper relative to
/// radix 8 ones: radix 8 first and last, radix 4 between. Measured with Wasm SIMD on Apple M4.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub fn unfused_radixes(len: usize) -> Vec<usize> {
    let bits = len.trailing_zeros() as usize;
    match bits {
        4 => return vec![4, 4],
        5 => return vec![4, 8],
        _ => {}
    }
    let mut middle = bits - 6;
    let mut radixes = vec![8];
    if middle % 2 == 1 {
        let radix: usize = if middle >= 3 { 8 } else { 2 };
        radixes.push(radix);
        middle -= radix.trailing_zeros() as usize;
    }
    radixes.extend(std::iter::repeat_n(4, middle / 2));
    radixes.push(8);
    radixes
}

// The first pass, whose twiddles are all 1
#[inline(always)]
unsafe fn base_pass<V: SimdVector, const R: usize>(
    butterfly: &impl ParallelFft<V, R>,
    pass: &Pass,
    mut buffer: impl SimdComplexArrayMut<V>,
) {
    unsafe {
        let stride = pass.stride;
        for q in (0..stride).step_by(V::COMPLEX_PER_VECTOR) {
            let mut values = [V::zero_vector(); R];
            for (k, value) in values.iter_mut().enumerate() {
                *value = buffer.load(q + stride * k);
            }
            let values = butterfly.parallel_fft(values);
            for (j, value) in values.into_iter().enumerate() {
                buffer.store(value, q + stride * j);
            }
        }
    }
}

/// Butterflies reading `x[q + s * (p + k * rows)]` and writing `y[q + s * (R * p + j)]` for row
/// `p` and lanes `q` of each stride `s`. `twiddles` starts with the pass's twiddles.
#[inline(always)]
unsafe fn main_pass<V: SimdVector, const R: usize>(
    butterfly: &impl ParallelFft<V, R>,
    pass: &Pass,
    twiddles: &[Complex<V::ScalarType>],
    mut buffer: impl SimdComplexArrayMut<V>,
) {
    unsafe {
        let (rows, stride) = (pass.rows, pass.stride);
        let twiddles = &twiddles[..rows * (R - 1)];

        for (row, row_twiddles) in twiddles.chunks_exact(R - 1).enumerate() {
            // Every butterfly of a row shares its twiddles, so broadcast them once per row. The
            // first output's is always 1, so it's neither stored nor applied.
            let one = Complex::new(V::ScalarType::one(), V::ScalarType::zero());
            let mut broadcast = [V::broadcast_twiddle(one); R];
            for (j, &twiddle) in row_twiddles.iter().enumerate() {
                broadcast[j + 1] = V::broadcast_twiddle(twiddle);
            }
            let input = stride * R * row;
            let output = stride * row;
            for q in (0..stride).step_by(V::COMPLEX_PER_VECTOR) {
                let mut values = [V::zero_vector(); R];
                values[0] = buffer.load(input + q);
                for k in 1..R {
                    let input_value = buffer.load(input + q + stride * k);
                    values[k] = V::mul_twiddle(input_value, broadcast[k]);
                }
                let values = butterfly.parallel_fft(values);
                for j in 0..R {
                    buffer.store(values[j], output + q + stride * rows * j);
                }
            }
        }
    }
}

/// `last_pass()` exists as essentially remainder handling for the `COMPLEX_PER_VECTOR > 1` case.
/// On the last pass, `stride == 1`, and thus `q` is only ever 0. If we examine the load indexing in `main_pass()`
/// we have (ignoring twiddle factors):
///
/// ```
/// for k in 0..R {
///    values[k] = buffer.load(input + q + stride * k);
/// }
/// ```
///
/// Since `q == 0` and `stride == 1`, this collapses to:
///
/// ```
/// for k in 0..R {
///    values[k] = buffer.load(input + k);
/// }
/// ```
///
/// Because `COMPLEX_PER_VECTOR > 1` and we're incrementing `k` by 1, the naive algorithm would have us doing overlapping loads, which is clearly wrong.
/// The problem is: Because `stride == 1`, we don't have parallel FFTs to load, because the FFT data is in contiguous format.
///
/// `last_pass()` circumvents this by unrolling the load loop and loading `COMPLEX_PER_VECTOR` consecutive FFTs contiguous-style,
/// and transposing them into the parallel FFTs we want.
///
/// There's also a second reason this function exists: In other passes, we use twiddle factors multiple times, so it's fast enough to store it in
/// packed format and unpack it on the fly. For this pass, we only use each twiddle factor once, so it's faster to store them in pre-unpacked format.
#[inline(always)]
unsafe fn last_pass<V: SimdVector, const R: usize>(
    butterfly: &impl ParallelFft<V, R>,
    len: usize,
    twiddles: &[V::Twiddle],
    mut buffer: impl SimdComplexArrayMut<V>,
) {
    unsafe {
        let rows = len / R;
        for (index, twiddles) in twiddles.chunks_exact(R - 1).enumerate() {
            let row = V::COMPLEX_PER_VECTOR * index;

            // Hardcode 1 to be 0 and stride to be 1 as documented above, so we can keep the indexing identical to `main_pass()`
            let q = 0;
            let stride = 1;
            let input = stride * R * row;
            let output = stride * row;

            let mut values = [V::zero_vector(); R];
            match V::COMPLEX_PER_VECTOR {
                1 => {
                    for k in 0..R {
                        values[k] = buffer.load(input + q + stride * k);
                    }
                }
                2 => {
                    for k in (0..R).step_by(2) {
                        let local_values = [
                            buffer.load(input + q + stride * k + 0),
                            buffer.load(input + q + stride * k + R),
                        ];
                        let transposed = V::transpose_complex_2x2(local_values);
                        values[k + 0] = transposed[0];
                        values[k + 1] = transposed[1];
                    }
                }
                c => unimplemented!(
                    "SimdStockham::last_pass() not implemented for V::COMPLEX_PER_VECTOR == {c}"
                ),
            }

            for (value, twiddle) in values[1..].iter_mut().zip(twiddles) {
                *value = V::mul_twiddle(*value, *twiddle);
            }
            let values = butterfly.parallel_fft(values);

            for (j, value) in values.into_iter().enumerate() {
                buffer.store(value, output + q + stride * rows * j);
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn radixes_cover_every_bit() {
        for bits in 4..=20 {
            for lanes in [1, 2] {
                let fused = fused_radixes(1 << bits, lanes);
                assert_eq!(fused.iter().product::<usize>(), 1 << bits, "{fused:?}");
                let neon = neon_radixes(1 << bits, lanes);
                assert_eq!(neon.iter().product::<usize>(), 1 << bits, "{neon:?}");
            }

            let unfused = unfused_radixes(1 << bits);
            assert_eq!(unfused.iter().product::<usize>(), 1 << bits, "{unfused:?}");
        }
        for bits in 21..=40 {
            let large = large_radixes(bits);
            let large_bits: u32 = large.iter().map(|radix| radix.trailing_zeros()).sum();
            assert_eq!(large_bits as usize, bits, "{large:?}");
        }
    }
}
