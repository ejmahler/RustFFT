//! Self-sorting (Stockham) FFT for powers of two, shared by every SIMD backend.
//!
//! Each pass reads one buffer and writes another, both contiguously, so the transform needs
//! neither transposes nor a bit reversal. After the first pass every butterfly of a row shares
//! its twiddles, which are broadcast once per row.

use std::any::TypeId;

use num_complex::Complex;
use num_traits::{FromPrimitive, One, Zero};

use crate::array_utils::{workaround_transmute, workaround_transmute_mut};
use crate::simd::simd_vector::SimdVector;
use crate::twiddles::compute_twiddle;
use crate::{FftDirection, FftNum};

/// A pass over the whole buffer: `rows` butterflies of `radix` per lane, `stride` lanes.
struct Pass<T> {
    radix: usize,
    rows: usize,
    stride: usize,
    /// Per row and output after the first: `(re, re)`, `(-im, im)` and `(re, im)`, broadcast to
    /// every lane.
    twiddles: Box<[Complex<T>]>,
}

pub struct SimdStockham<V: SimdVector, T> {
    /// For vectors of two or more complex numbers, the first pass, whose butterflies each own a
    /// lane. Its twiddles are whole vectors: `re` parts, signed `im` parts, then whole ones.
    first: Option<(usize, Box<[V]>)>,
    passes: Box<[Pass<T>]>,
    len: usize,
    direction: FftDirection,
}

/// How far apart, modulo 4 KiB, the buffers of consecutive passes start.
const GAP: usize = 1024;

impl<V: SimdVector, T: FftNum> SimdStockham<V, T> {
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
        let lanes = V::COMPLEX_PER_VECTOR;
        let (first, rest, mut span, mut stride) = if lanes > 1 {
            let radix = radixes[0];
            let rows = len / radix;
            let mut twiddles = Vec::with_capacity(3 * rows / lanes * (radix - 1));
            for row in (0..rows).step_by(lanes) {
                for output in 1..radix {
                    let mut re = [Complex::zero(); 4];
                    let mut im = [Complex::zero(); 4];
                    let mut whole = [Complex::zero(); 4];
                    for lane in 0..lanes {
                        let w = compute_twiddle::<T>((row + lane) * output, len, direction);
                        re[lane] = Complex::new(w.re, w.re);
                        im[lane] = Complex::new(-w.im, w.im);
                        whole[lane] = w;
                    }
                    for parts in [re, im, whole] {
                        let parts = parts.as_ptr() as *const Complex<V::ScalarType>;
                        twiddles.push(unsafe { V::load_complex(parts) });
                    }
                }
            }
            (
                Some((radix, twiddles.into_boxed_slice())),
                &radixes[1..],
                rows,
                radix,
            )
        } else {
            (None, radixes, len, 1)
        };

        let mut passes = Vec::with_capacity(rest.len());
        for &radix in rest {
            let rows = span / radix;
            let mut twiddles = Vec::new();
            if rows > 1 {
                twiddles.reserve(3 * rows * (radix - 1));
                for row in 0..rows {
                    for output in 1..radix {
                        let w = compute_twiddle::<T>(row * output, span, direction);
                        twiddles.extend([Complex::new(w.re, w.re), Complex::new(-w.im, w.im), w]);
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

        Self {
            first,
            passes: passes.into_boxed_slice(),
            len,
            direction,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn fft_direction(&self) -> FftDirection {
        self.direction
    }

    fn scratch_len(&self) -> usize {
        2 * (self.len + 4096 / std::mem::size_of::<Complex<T>>())
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
        self.process(buffer.as_ptr(), buffer.as_mut_ptr(), scratch);
    }

    #[inline(always)]
    pub fn process_single_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.process(input.as_ptr(), output.as_mut_ptr(), scratch);
    }

    #[inline(always)]
    pub fn process_single_immutable(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.process(input.as_ptr(), output.as_mut_ptr(), scratch);
    }

    /// Runs every pass: the first from `input`, the last into `output`, the others back and forth
    /// between two halves of `scratch` placed apart modulo 4 KiB.
    #[inline(always)]
    fn process(
        &self,
        input: *const Complex<T>,
        output: *mut Complex<T>,
        scratch: &mut [Complex<T>],
    ) {
        let scratch: &mut [Complex<V::ScalarType>] = unsafe { workaround_transmute_mut(scratch) };
        let input = input as *const Complex<V::ScalarType>;
        let output = output as *mut Complex<V::ScalarType>;
        let (first, rest) = place(scratch, input as usize + GAP).split_at_mut(self.len);
        let second = place(rest, first.as_ptr() as usize + GAP).as_mut_ptr();
        let first = first.as_mut_ptr();
        match self.direction {
            FftDirection::Forward => unsafe { self.run::<true>(input, output, [first, second]) },
            FftDirection::Inverse => unsafe { self.run::<false>(input, output, [first, second]) },
        }
    }

    #[inline(always)]
    unsafe fn run<const FORWARD: bool>(
        &self,
        input: *const Complex<V::ScalarType>,
        output: *mut Complex<V::ScalarType>,
        halves: [*mut Complex<V::ScalarType>; 2],
    ) {
        let count = self.passes.len() + usize::from(self.first.is_some());
        let target = |index: usize| {
            if index + 1 == count {
                output
            } else {
                halves[index % 2]
            }
        };
        let mut src = input;
        let mut index = 0;
        if let Some((radix, twiddles)) = &self.first {
            let dst = target(0);
            unsafe {
                match radix {
                    2 => first_pass::<V, 2, FORWARD>(self.len, twiddles, src, dst),
                    4 => first_pass::<V, 4, FORWARD>(self.len, twiddles, src, dst),
                    8 => first_pass::<V, 8, FORWARD>(self.len, twiddles, src, dst),
                    16 => first_pass::<V, 16, FORWARD>(self.len, twiddles, src, dst),
                    _ => unreachable!(),
                }
            }
            src = dst;
            index = 1;
        }
        for pass in self.passes.iter() {
            let dst = target(index);
            let twiddles: &[Complex<V::ScalarType>] =
                unsafe { workaround_transmute(&pass.twiddles) };
            unsafe {
                match pass.radix {
                    2 => later_pass::<V, 2, FORWARD>(pass.rows, pass.stride, twiddles, src, dst),
                    4 => later_pass::<V, 4, FORWARD>(pass.rows, pass.stride, twiddles, src, dst),
                    8 => later_pass::<V, 8, FORWARD>(pass.rows, pass.stride, twiddles, src, dst),
                    16 => later_pass::<V, 16, FORWARD>(pass.rows, pass.stride, twiddles, src, dst),
                    _ => unreachable!(),
                }
            }
            src = dst;
            index += 1;
        }
    }
}

/// The tail of `slice` starting at the first element whose address is `target` modulo 4 KiB,
/// rounded down to 64 bytes.
fn place<C>(slice: &mut [C], target: usize) -> &mut [C] {
    let size = std::mem::size_of::<C>();
    let target = target % 4096 / 64 * 64;
    let at = slice.as_ptr() as usize % 4096;
    &mut slice[(target + 4096 - at) % 4096 / size..]
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

#[derive(Clone, Copy)]
struct Constants<V> {
    /// `(1, -1)` for forward FFTs, `(-1, 1)` for inverse ones: `swap(b) * rotation` is `b`
    /// times `-i` or `i`.
    rotation: V,
    half_root2: V,
    /// The 16th roots of unity to the powers 1, 3 and 9, as `(re, re)`, `(-im, im)` and whole.
    w16: [[V; 3]; 3],
}

impl<V: SimdVector> Constants<V> {
    #[inline(always)]
    unsafe fn new<const FORWARD: bool>() -> Self {
        let one = V::ScalarType::one();
        let pair = if FORWARD {
            Complex::new(one, -one)
        } else {
            Complex::new(-one, one)
        };
        let half = V::ScalarType::from_f64(std::f64::consts::FRAC_1_SQRT_2).unwrap_or(one);
        let direction = if FORWARD {
            FftDirection::Forward
        } else {
            FftDirection::Inverse
        };
        let w16 = [1, 3, 9].map(|k| {
            let w = compute_twiddle::<V::ScalarType>(k, 16, direction);
            let parts = [Complex::new(w.re, w.re), Complex::new(-w.im, w.im), w];
            parts.map(|part| unsafe { dup::<V>(&part) })
        });
        unsafe {
            Self {
                rotation: dup::<V>(&pair),
                half_root2: V::broadcast_scalar(half),
                w16,
            }
        }
    }
}

/// The complex number at `ptr` in every slot of a vector.
#[inline(always)]
unsafe fn dup<V: SimdVector>(ptr: *const Complex<V::ScalarType>) -> V {
    unsafe {
        if V::COMPLEX_PER_VECTOR == 1 {
            V::load_complex(ptr)
        } else {
            V::load1_dup_complex(ptr)
        }
    }
}

/// `a + r * b`, where `r` is `-i` for forward FFTs and `i` for inverse ones.
#[inline(always)]
unsafe fn rotate_add<V: SimdVector, const FORWARD: bool>(a: V, b: V, c: Constants<V>) -> V {
    unsafe { V::rotate_add::<FORWARD>(a, b, c.rotation) }
}

/// `a - r * b`.
#[inline(always)]
unsafe fn rotate_sub<V: SimdVector, const FORWARD: bool>(a: V, b: V, c: Constants<V>) -> V {
    unsafe { V::rotate_sub::<FORWARD>(a, b, c.rotation) }
}

#[inline(always)]
unsafe fn dft4<V: SimdVector, const F: bool>(x: [V; 4], c: Constants<V>) -> [V; 4] {
    unsafe {
        let t0 = V::add(x[0], x[2]);
        let t1 = V::sub(x[0], x[2]);
        let t2 = V::add(x[1], x[3]);
        let t3 = V::sub(x[1], x[3]);
        [
            V::add(t0, t2),
            rotate_add::<V, F>(t1, t3, c),
            V::sub(t0, t2),
            rotate_sub::<V, F>(t1, t3, c),
        ]
    }
}

#[inline(always)]
unsafe fn dft8<V: SimdVector, const F: bool>(a: [V; 8], c: Constants<V>) -> [V; 8] {
    unsafe {
        let u = [
            V::add(a[0], a[4]),
            V::add(a[1], a[5]),
            V::add(a[2], a[6]),
            V::add(a[3], a[7]),
        ];
        let v = [
            V::sub(a[0], a[4]),
            V::sub(a[1], a[5]),
            V::sub(a[2], a[6]),
            V::sub(a[3], a[7]),
        ];
        let even = dft4::<V, F>(u, c);
        let v1 = V::mul(rotate_add::<V, F>(v[1], v[1], c), c.half_root2);
        let v3 = V::mul(rotate_sub::<V, F>(v[3], v[3], c), V::neg(c.half_root2));
        let t0 = rotate_add::<V, F>(v[0], v[2], c);
        let t1 = rotate_sub::<V, F>(v[0], v[2], c);
        let t2 = V::add(v1, v3);
        let t3 = V::sub(v1, v3);
        let odd = [
            V::add(t0, t2),
            rotate_add::<V, F>(t1, t3, c),
            V::sub(t0, t2),
            rotate_sub::<V, F>(t1, t3, c),
        ];
        [
            even[0], odd[0], even[1], odd[1], even[2], odd[2], even[3], odd[3],
        ]
    }
}

/// A 16-point DFT as four 4-point DFTs, twiddles, then four more.
#[inline(always)]
unsafe fn dft16<V: SimdVector, const F: bool>(a: [V; 16], c: Constants<V>) -> [V; 16] {
    unsafe {
        let mut b = [[c.rotation; 4]; 4];
        for (k1, column) in b.iter_mut().enumerate() {
            *column = dft4::<V, F>([a[k1], a[k1 + 4], a[k1 + 8], a[k1 + 12]], c);
        }
        let general = |v: V, w: [V; 3]| V::mul_twiddle(v, w[0], w[1], w[2]);
        b[1][1] = general(b[1][1], c.w16[0]);
        b[1][2] = V::mul(rotate_add::<V, F>(b[1][2], b[1][2], c), c.half_root2);
        b[1][3] = general(b[1][3], c.w16[1]);
        b[2][1] = V::mul(rotate_add::<V, F>(b[2][1], b[2][1], c), c.half_root2);
        b[2][2] = rotate_add::<V, F>(V::zero_vector(), b[2][2], c);
        b[2][3] = V::mul(
            rotate_sub::<V, F>(b[2][3], b[2][3], c),
            V::neg(c.half_root2),
        );
        b[3][1] = general(b[3][1], c.w16[1]);
        b[3][2] = V::mul(
            rotate_sub::<V, F>(b[3][2], b[3][2], c),
            V::neg(c.half_root2),
        );
        b[3][3] = general(b[3][3], c.w16[2]);
        let mut out = [c.rotation; 16];
        for j1 in 0..4 {
            let column = dft4::<V, F>([b[0][j1], b[1][j1], b[2][j1], b[3][j1]], c);
            for (j2, value) in column.into_iter().enumerate() {
                out[j1 + 4 * j2] = value;
            }
        }
        out
    }
}

#[inline(always)]
unsafe fn butterfly<V: SimdVector, const R: usize, const F: bool>(
    rows: [V; R],
    c: Constants<V>,
) -> [V; R] {
    unsafe {
        let mut out = rows;
        match R {
            2 => {
                out[0] = V::add(rows[0], rows[1]);
                out[1] = V::sub(rows[0], rows[1]);
            }
            4 => {
                let r = dft4::<V, F>([rows[0], rows[1], rows[2], rows[3]], c);
                out[..4].copy_from_slice(&r);
            }
            8 => {
                let r = dft8::<V, F>(
                    [
                        rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7],
                    ],
                    c,
                );
                out[..8].copy_from_slice(&r);
            }
            16 => {
                let r = dft16::<V, F>(std::array::from_fn(|i| rows[i]), c);
                out[..16].copy_from_slice(&r);
            }
            _ => unreachable!(),
        }
        out
    }
}

/// The first pass for vectors of two complex numbers: butterflies over rows `p` and `p + 1`
/// (one per lane), reading `x[p + k * len / R]` and writing `y[R * p + j]`.
#[inline(always)]
unsafe fn first_pass<V: SimdVector, const R: usize, const FORWARD: bool>(
    len: usize,
    twiddles: &[V],
    x: *const Complex<V::ScalarType>,
    y: *mut Complex<V::ScalarType>,
) {
    let c = unsafe { Constants::<V>::new::<FORWARD>() };
    let rows = len / R;
    for (pair, twiddles) in twiddles.chunks_exact(3 * (R - 1)).enumerate() {
        let row = 2 * pair;
        unsafe {
            let mut inputs = [c.rotation; R];
            for (k, slot) in inputs.iter_mut().enumerate() {
                *slot = V::load_complex(x.add(row + k * rows));
            }
            let mut outputs = butterfly::<V, R, FORWARD>(inputs, c);
            for j in 1..R {
                let w = &twiddles[3 * j - 3..3 * j];
                outputs[j] = V::mul_twiddle(outputs[j], w[0], w[1], w[2]);
            }
            for j in (0..R).step_by(2) {
                let [lo, hi] = V::transpose_complex_2x2([outputs[j], outputs[j + 1]]);
                V::store_complex(y.add(R * row + j), lo);
                V::store_complex(y.add(R * row + R + j), hi);
            }
        }
    }
}

/// Butterflies reading `x[q + s * (p + k * rows)]` and writing `y[q + s * (R * p + j)]` for row
/// `p` and lanes `q` of each stride `s`.
#[inline(always)]
unsafe fn later_pass<V: SimdVector, const R: usize, const FORWARD: bool>(
    rows: usize,
    stride: usize,
    twiddles: &[Complex<V::ScalarType>],
    x: *const Complex<V::ScalarType>,
    y: *mut Complex<V::ScalarType>,
) {
    let c = unsafe { Constants::<V>::new::<FORWARD>() };
    let lanes = V::COMPLEX_PER_VECTOR;
    if rows == 1 {
        for q in (0..stride).step_by(lanes) {
            unsafe {
                let mut inputs = [c.rotation; R];
                for (k, slot) in inputs.iter_mut().enumerate() {
                    *slot = V::load_complex(x.add(q + stride * k));
                }
                let outputs = butterfly::<V, R, FORWARD>(inputs, c);
                for (j, &output) in outputs.iter().enumerate() {
                    V::store_complex(y.add(q + stride * j), output);
                }
            }
        }
        return;
    }
    for (row, twiddles) in twiddles.chunks_exact(3 * (R - 1)).enumerate() {
        let mut split = [[c.rotation; 3]; R];
        for (j, slot) in split.iter_mut().enumerate().skip(1) {
            for (part, value) in slot.iter_mut().enumerate() {
                *value = unsafe { dup::<V>(&twiddles[3 * j - 3 + part]) };
            }
        }
        let input = unsafe { x.add(stride * row) };
        let output = unsafe { y.add(stride * R * row) };
        for q in (0..stride).step_by(lanes) {
            unsafe {
                let mut inputs = [c.rotation; R];
                for (k, slot) in inputs.iter_mut().enumerate() {
                    *slot = V::load_complex(input.add(q + stride * k * rows));
                }
                let outputs = butterfly::<V, R, FORWARD>(inputs, c);
                V::store_complex(output.add(q), outputs[0]);
                for j in 1..R {
                    let [re, im, whole] = split[j];
                    let value = V::mul_twiddle(outputs[j], re, im, whole);
                    V::store_complex(output.add(q + stride * j), value);
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
