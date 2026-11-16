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

/// The first pass, whose butterflies each own a lane and so need a twiddle factor per lane.
struct FirstPass<V: SimdVector> {
    radix: usize,
    /// `radix - 1` per vector of rows.
    twiddles: Box<[V::Twiddle]>,
}

/// A later pass, whose butterflies span every lane: `rows` of them, each `stride` complex numbers
/// wide. Every butterfly of a row shares its twiddle factors.
struct Pass<T> {
    radix: usize,
    rows: usize,
    stride: usize,
    /// `radix - 1` per row, or none if there is only one row.
    twiddles: Box<[Complex<T>]>,
}

pub struct SimdStockham<V: SimdVector, T> {
    /// A `first_pass` for the first pass, or none if it's the first of `passes` (stride 1).
    first: Option<FirstPass<V>>,
    passes: Box<[Pass<V::ScalarType>]>,

    butterfly2: V::Butterfly2,
    butterfly4: V::Butterfly4,
    butterfly8: V::Butterfly8,
    butterfly16: V::Butterfly16,

    len: usize,
    direction: FftDirection,
    _phantom: PhantomData<T>,
}

/// Consecutive passes read and write buffers whose addresses differ by `GAP` bytes modulo
/// `ALIASING_PERIOD`, rounded down to a cache line.
///
/// A pass reads and writes its two buffers in lockstep. If both start at the same offset into a
/// 4 KiB page, as large allocations tend to, every load and the store next to it map onto the
/// same L1 cache sets, and on many CPUs a load whose low 12 address bits match those of a pending
/// store also waits for that store ("4K aliasing"). Offsetting the buffers avoids both. Without
/// it, NEON and FCMA are 5 to 9% slower from 2^14 points up on Apple M4.
const ALIASING_PERIOD: usize = 4096;
const GAP: usize = 1024;
const CACHE_LINE: usize = 64;

/// How many elements `place()` may skip at most, so how much longer than the buffer it places
/// the slice it's given has to be.
const fn placement_padding<C>() -> usize {
    ALIASING_PERIOD / std::mem::size_of::<C>()
}

/// The tail of `slice` starting `GAP` bytes, modulo `ALIASING_PERIOD`, after the start of
/// `previous`, the buffer read by the pass that writes this one.
fn place<'a, C>(slice: &'a mut [C], previous: &[C]) -> &'a mut [C] {
    let target = (previous.as_ptr() as usize + GAP) % ALIASING_PERIOD / CACHE_LINE * CACHE_LINE;
    let at = slice.as_ptr() as usize % ALIASING_PERIOD;
    let skip = (target + ALIASING_PERIOD - at) % ALIASING_PERIOD;
    &mut slice[skip / std::mem::size_of::<C>()..]
}

impl<V: SimdVector, T: FftNum> SimdStockham<V, T> {
    /// Preallocates the twiddles for a `len`-point FFT computed with passes of `radixes`, the
    /// first pass first. `len` is a power of two of at least 16 and the product of `radixes`.
    ///
    /// Vectors of two complex numbers always run the first pass with `first_pass`. For vectors of
    /// one, `lane_first_pass` chooses between `first_pass` and a regular pass of stride 1. With NEON
    /// f64 on Apple M4, the later passes run up to 20% slower after a `first_pass`, so NEON passes
    /// false. FCMA is a little faster with `first_pass`.
    pub fn new(
        len: usize,
        direction: FftDirection,
        radixes: &[usize],
        lane_first_pass: bool,
    ) -> Self {
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

        let lanes = V::COMPLEX_PER_VECTOR;
        let (first, rest, mut span, mut stride) = if lanes > 1 || lane_first_pass {
            let radix = radixes[0];
            let rows = len / radix;
            let mut twiddles = Vec::with_capacity(rows / lanes * (radix - 1));
            for row in (0..rows).step_by(lanes) {
                for output in 1..radix {
                    let lane_twiddles: [Complex<V::ScalarType>; 2] = std::array::from_fn(|lane| {
                        compute_twiddle((row + lane) * output, len, direction)
                    });
                    twiddles.push(unsafe { V::make_twiddle(&lane_twiddles[..lanes]) });
                }
            }
            let first = FirstPass {
                radix,
                twiddles: twiddles.into_boxed_slice(),
            };
            (Some(first), &radixes[1..], rows, radix)
        } else {
            (None, radixes, len, 1)
        };

        let mut passes = Vec::with_capacity(rest.len());
        for &radix in rest {
            let rows = span / radix;
            let mut twiddles = Vec::new();
            if rows > 1 {
                twiddles.reserve(rows * (radix - 1));
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
                twiddles: twiddles.into_boxed_slice(),
            });
            span = rows;
            stride *= radix;
        }
        assert_eq!(span, 1);

        unsafe {
            Self {
                first,
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

    fn scratch_len(&self) -> usize {
        2 * (self.len + placement_padding::<Complex<T>>())
    }

    pub fn get_inplace_scratch_len(&self) -> usize {
        self.scratch_len()
    }

    pub fn get_outofplace_scratch_len(&self) -> usize {
        self.scratch_len()
    }

    pub fn get_immutable_scratch_len(&self) -> usize {
        self.scratch_len()
    }

    #[inline(always)]
    pub fn process_single(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let buffer: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(buffer) };
        let scratch: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(scratch) };
        let src = self.run_all_but_last(buffer, scratch);
        self.run_pass(self.pass_count() - 1, src, buffer);
    }

    #[inline(always)]
    pub fn process_single_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.process_single_immutable(input, output, scratch);
    }

    #[inline(always)]
    pub fn process_single_immutable(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let input: &[Complex<V::ScalarType>] = unsafe { workaround_transmute(input) };
        let output: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(output) };
        let scratch: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(scratch) };
        let src = self.run_all_but_last(input, scratch);
        self.run_pass(self.pass_count() - 1, src, output);
    }

    fn pass_count(&self) -> usize {
        self.passes.len() + usize::from(self.first.is_some())
    }

    /// Runs every pass but the last: the first reading `input`, the others going back and forth
    /// between two buffers placed in `scratch`. Returns the buffer the last pass reads.
    #[inline(always)]
    fn run_all_but_last<'a>(
        &self,
        input: &[Complex<V::ScalarType>],
        scratch: &'a mut [Complex<V::ScalarType>],
    ) -> &'a [Complex<V::ScalarType>] {
        let (mut src, rest) = place(scratch, input).split_at_mut(self.len);
        let mut dst = &mut place(rest, src)[..self.len];
        self.run_pass(0, input, src);
        for index in 1..self.pass_count() - 1 {
            self.run_pass(index, src, dst);
            std::mem::swap(&mut src, &mut dst);
        }
        src
    }

    #[inline(always)]
    fn run_pass(
        &self,
        index: usize,
        input: &[Complex<V::ScalarType>],
        output: &mut [Complex<V::ScalarType>],
    ) {
        let buffer = DoubleBuf { input, output };
        let index = match &self.first {
            Some(first) if index == 0 => {
                let (len, twiddles) = (self.len, &first.twiddles);
                unsafe {
                    match first.radix {
                        2 => first_pass(&self.butterfly2, len, twiddles, buffer),
                        4 => first_pass(&self.butterfly4, len, twiddles, buffer),
                        8 => first_pass(&self.butterfly8, len, twiddles, buffer),
                        16 => first_pass(&self.butterfly16, len, twiddles, buffer),
                        _ => unreachable!(),
                    }
                }
                return;
            }
            Some(_) => index - 1,
            None => index,
        };
        let pass = &self.passes[index];
        unsafe {
            match pass.radix {
                2 => later_pass(&self.butterfly2, pass, buffer),
                4 => later_pass(&self.butterfly4, pass, buffer),
                8 => later_pass(&self.butterfly8, pass, buffer),
                16 => later_pass(&self.butterfly16, pass, buffer),
                _ => unreachable!(),
            }
        }
    }
}

/// Radixes for CPUs with fused multiply-add and FCMA, the first pass first, for vectors of
/// `lanes` complex numbers. Sizes up to 2^16 use the fastest plan measured on Apple M4.
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
        _ => &[],
    };
    if !measured.is_empty() {
        return measured.to_vec();
    }
    let mut radixes = vec![8; bits / 3];
    match bits % 3 {
        1 => radixes.insert(0, 2),
        2 => radixes.insert(0, 4),
        _ => {}
    }
    radixes
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
        (17, 1) => &[8, 4, 8, 8, 8, 8],
        (17, _) => &[4, 8, 8, 8, 8, 8],
        (18, 1) => &[8, 4, 8, 8, 8, 2, 8],
        (18, _) => &[4, 2, 8, 8, 8, 8, 8],
        (19, 1) => &[8, 4, 8, 8, 8, 4, 8],
        (19, _) => &[8, 8, 8, 8, 8, 4, 4],
        (20, 1) => &[8, 4, 8, 8, 8, 8, 8],
        (20, _) => &[8, 8, 8, 8, 8, 4, 8],
        _ => &[],
    };
    if !measured.is_empty() {
        return measured.to_vec();
    }
    // Larger sizes: radix 8, with a radix 4 second and, if needed, another second to last.
    let mut radixes = vec![8; bits / 3];
    match bits % 3 {
        1 => {
            radixes[1] = 4;
            let last = radixes.len() - 1;
            radixes.insert(last, 4);
        }
        2 => radixes.insert(1, 4),
        _ => {}
    }
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

/// The first pass: butterflies over rows `p` (one per lane of a vector of rows), reading
/// `x[p + k * len / R]` and writing `y[R * p + j]`.
#[inline(always)]
unsafe fn first_pass<V: SimdVector, const R: usize>(
    butterfly: &impl ParallelFft<V, R>,
    len: usize,
    twiddles: &[V::Twiddle],
    mut buffer: impl SimdComplexArrayMut<V>,
) {
    unsafe {
        let rows = len / R;
        let lanes = V::COMPLEX_PER_VECTOR;
        for (index, twiddles) in twiddles.chunks_exact(R - 1).enumerate() {
            let row = lanes * index;
            let mut values = [V::zero_vector(); R];
            for (k, value) in values.iter_mut().enumerate() {
                *value = buffer.load(row + k * rows);
            }
            let mut values = butterfly.parallel_fft(values);
            for (value, twiddle) in values[1..].iter_mut().zip(twiddles) {
                *value = V::mul_twiddle(*value, *twiddle);
            }
            if lanes == 1 {
                for (j, value) in values.into_iter().enumerate() {
                    buffer.store(value, R * row + j);
                }
            } else {
                // The lanes hold consecutive rows, whose outputs are R apart
                for j in (0..R).step_by(2) {
                    let [lo, hi] = V::transpose_complex_2x2([values[j], values[j + 1]]);
                    buffer.store(lo, R * row + j);
                    buffer.store(hi, R * row + R + j);
                }
            }
        }
    }
}

/// Butterflies reading `x[q + s * (p + k * rows)]` and writing `y[q + s * (R * p + j)]` for row
/// `p` and lanes `q` of each stride `s`.
#[inline(always)]
unsafe fn later_pass<V: SimdVector, const R: usize>(
    butterfly: &impl ParallelFft<V, R>,
    pass: &Pass<V::ScalarType>,
    mut buffer: impl SimdComplexArrayMut<V>,
) {
    unsafe {
        let (rows, stride) = (pass.rows, pass.stride);
        let lanes = V::COMPLEX_PER_VECTOR;
        if rows == 1 {
            // The last pass, whose twiddles are all 1
            for q in (0..stride).step_by(lanes) {
                let mut values = [V::zero_vector(); R];
                for (k, value) in values.iter_mut().enumerate() {
                    *value = buffer.load(q + stride * k);
                }
                let values = butterfly.parallel_fft(values);
                for (j, value) in values.into_iter().enumerate() {
                    buffer.store(value, q + stride * j);
                }
            }
            return;
        }
        let twiddles: &[Complex<V::ScalarType>] = &pass.twiddles;
        for (row, row_twiddles) in twiddles.chunks_exact(R - 1).enumerate() {
            // Every butterfly of a row shares its twiddles, so broadcast them once per row. The
            // first output's is always 1, so it's neither stored nor applied.
            let one = Complex::new(V::ScalarType::one(), V::ScalarType::zero());
            let mut broadcast = [V::broadcast_twiddle(one); R];
            for (j, &twiddle) in row_twiddles.iter().enumerate() {
                broadcast[j + 1] = V::broadcast_twiddle(twiddle);
            }
            let input = stride * row;
            let output = stride * R * row;
            for q in (0..stride).step_by(lanes) {
                let mut values = [V::zero_vector(); R];
                for (k, value) in values.iter_mut().enumerate() {
                    *value = buffer.load(input + q + stride * rows * k);
                }
                let values = butterfly.parallel_fft(values);
                buffer.store(values[0], output + q);
                for j in 1..R {
                    let value = V::mul_twiddle(values[j], broadcast[j]);
                    buffer.store(value, output + q + stride * j);
                }
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn radixes_cover_every_bit() {
        for bits in 4..=26 {
            for lanes in [1, 2] {
                let fused = fused_radixes(1 << bits, lanes);
                assert_eq!(fused.iter().product::<usize>(), 1 << bits, "{fused:?}");
            }
            let unfused = unfused_radixes(1 << bits);
            assert_eq!(unfused.iter().product::<usize>(), 1 << bits, "{unfused:?}");
        }
    }
}
