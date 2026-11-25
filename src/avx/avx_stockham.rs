use std::any::TypeId;
use std::arch::x86_64::*;
use std::marker::PhantomData;

use num_complex::Complex;
use num_traits::One;

use crate::array_utils::{workaround_transmute, workaround_transmute_mut};
use crate::twiddles::compute_twiddle;
use crate::{Direction, Fft, FftDirection, FftNum, Length};

use super::AvxNum;
use super::avx_vector::{AvxVector, AvxVector256};

/// The first pass. It reads `radix` streams `len / radix` apart, one lane per butterfly, and
/// transposes its results so they land interleaved. Its twiddles are split into real and
/// imaginary vectors.
struct FirstPass<V> {
    radix: usize,
    twiddles: Box<[[V; 2]]>,
}

/// A later pass. Every butterfly of a row shares its twiddles, so they are stored once per row
/// as scalar pairs and broadcast.
struct Pass<A> {
    radix: usize,
    rows: usize,
    stride: usize,
    twiddles: Box<[A]>,
}

/// Self-sorting (Stockham) FFT for powers of two, built from radix 8, 4 and 2 passes.
///
/// Each pass reads from one buffer and writes to another, both contiguously, so the transform
/// needs neither transposes nor a bit-reversal permutation.
pub struct StockhamAvx<A: AvxNum, T> {
    first: FirstPass<A::VectorType>,
    passes: Box<[Pass<A>]>,
    wide: bool,
    len: usize,
    direction: FftDirection,
    _phantom: PhantomData<T>,
}

impl<A: AvxNum, T: FftNum> StockhamAvx<A, T> {
    /// Preallocates the twiddles for a `len`-point FFT, `len` a power of two of at least 64.
    /// Returns `Err(())` if the CPU lacks AVX, AVX2 or FMA.
    pub fn new(len: usize, direction: FftDirection) -> Result<Self, ()> {
        assert_eq!(TypeId::of::<A>(), TypeId::of::<T>());
        assert!(
            len.is_power_of_two() && len >= 64,
            "invalid Stockham length {len}"
        );
        let supported = is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma");
        if supported {
            Ok(unsafe { Self::new_with_avx(len, direction) })
        } else {
            Err(())
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn new_with_avx(len: usize, direction: FftDirection) -> Self {
        let radixes = radixes(len, A::VectorType::COMPLEX_PER_VECTOR);
        let lanes = A::VectorType::COMPLEX_PER_VECTOR;
        let first_radix = radixes[0];
        let rows = len / first_radix;
        let mut twiddles = Vec::with_capacity(rows / lanes * (first_radix - 1));
        for row in (0..rows).step_by(lanes) {
            for output in 1..first_radix {
                let chunk = unsafe {
                    A::VectorType::make_mixedradix_twiddle_chunk(row, output, len, direction)
                };
                let (re, im) = unsafe { chunk.duplicate_complex_components() };
                twiddles.push([re, im]);
            }
        }
        let first = FirstPass {
            radix: first_radix,
            twiddles: twiddles.into_boxed_slice(),
        };

        let mut passes = Vec::with_capacity(radixes.len() - 1);
        let (mut span, mut stride) = (rows, first_radix);
        for &radix in &radixes[1..] {
            let rows = span / radix;
            let mut twiddles = Vec::new();
            if rows > 1 {
                twiddles.reserve(2 * rows * (radix - 1));
                for row in 0..rows {
                    for output in 1..radix {
                        let twiddle = compute_twiddle::<A>(row * output, span, direction);
                        twiddles.extend([twiddle.re, twiddle.im]);
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

        Self {
            first,
            passes: passes.into_boxed_slice(),
            wide: is_x86_feature_detected!("avx512f"),
            len,
            direction,
            _phantom: PhantomData,
        }
    }

    /// Two `len`-element halves of `scratch`, 64-byte aligned, the first a fixed distance past
    /// `near` and the second past the first, modulo 4 KiB. Equal low address bits make loads from
    /// one buffer wait on stores to the other (4K aliasing), which radix 8 passes suffer from most.
    fn halves<'a>(
        &self,
        scratch: &'a mut [Complex<A>],
        near: *const Complex<A>,
    ) -> (&'a mut [Complex<A>], &'a mut [Complex<A>]) {
        let (first, rest) = place(scratch, near as usize + GAP).split_at_mut(self.len);
        let second = place(rest, first.as_ptr() as usize + GAP);
        (first, &mut second[..self.len])
    }

    /// Runs every pass: the first from `input`, the last into `output`, the others back and forth
    /// between the two halves of `scratch`.
    fn process(
        &self,
        input: *const Complex<A>,
        output: *mut Complex<A>,
        scratch: &mut [Complex<A>],
    ) {
        let (first, second) = self.halves(scratch, input);
        unsafe { self.run(input, output, first.as_mut_ptr(), second.as_mut_ptr()) };
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn run(
        &self,
        input: *const Complex<A>,
        output: *mut Complex<A>,
        first: *mut Complex<A>,
        second: *mut Complex<A>,
    ) {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.run_in::<true>(input, output, first, second),
                FftDirection::Inverse => self.run_in::<false>(input, output, first, second),
            }
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn run_in<const FORWARD: bool>(
        &self,
        input: *const Complex<A>,
        output: *mut Complex<A>,
        first: *mut Complex<A>,
        second: *mut Complex<A>,
    ) {
        let len = self.len;
        let last = self.passes.len() - 1;
        let start = &self.first;
        unsafe {
            match start.radix {
                4 => first_pass::<A, 4, FORWARD>(len, &start.twiddles, input, first),
                8 => first_pass::<A, 8, FORWARD>(len, &start.twiddles, input, first),
                _ => unreachable!(),
            }
        }
        let (mut src, mut spare) = (first, second);
        for (index, pass) in self.passes.iter().enumerate() {
            let dst = if index == last { output } else { spare };
            unsafe {
                if !self.wide_pass(pass, src, dst) {
                    match pass.radix {
                        2 => later_pass::<A, 2, FORWARD>(pass, src, dst),
                        4 => later_pass::<A, 4, FORWARD>(pass, src, dst),
                        8 => later_pass::<A, 8, FORWARD>(pass, src, dst),
                        _ => unreachable!(),
                    }
                }
            }
            spare = src;
            src = dst;
        }
    }

    /// Runs `pass` with AVX-512 vectors if the CPU has them and the pass is wide enough.
    unsafe fn wide_pass(&self, pass: &Pass<A>, src: *mut Complex<A>, dst: *mut Complex<A>) -> bool {
        if !self.wide {
            return false;
        }
        let (rows, stride) = (pass.rows, pass.stride);
        let forward = self.direction == FftDirection::Forward;
        macro_rules! dispatch {
            ($vector:ty, $scalar:ty) => {{
                if stride % <$vector as Lanes>::COMPLEX != 0 {
                    return false;
                }
                let twiddles: &[$scalar] = unsafe { workaround_transmute(&pass.twiddles) };
                let (x, y) = (src as *const $scalar, dst as *mut $scalar);
                unsafe {
                    match (pass.radix, forward) {
                        (2, true) => {
                            later_pass_wide::<$vector, 2, true>(rows, stride, twiddles, x, y)
                        }
                        (4, true) => {
                            later_pass_wide::<$vector, 4, true>(rows, stride, twiddles, x, y)
                        }
                        (8, true) => {
                            later_pass_wide::<$vector, 8, true>(rows, stride, twiddles, x, y)
                        }
                        (2, false) => {
                            later_pass_wide::<$vector, 2, false>(rows, stride, twiddles, x, y)
                        }
                        (4, false) => {
                            later_pass_wide::<$vector, 4, false>(rows, stride, twiddles, x, y)
                        }
                        (8, false) => {
                            later_pass_wide::<$vector, 8, false>(rows, stride, twiddles, x, y)
                        }
                        _ => unreachable!(),
                    }
                }
                true
            }};
        }
        if TypeId::of::<A>() == TypeId::of::<f32>() {
            dispatch!(__m512, f32)
        } else {
            dispatch!(__m512d, f64)
        }
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let buffer: &mut [Complex<A>] = unsafe { workaround_transmute_mut(buffer) };
        let scratch: &mut [Complex<A>] = unsafe { workaround_transmute_mut(scratch) };
        self.process(buffer.as_ptr(), buffer.as_mut_ptr(), scratch);
    }

    fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.perform_fft_immut(input, output, scratch);
    }

    fn perform_fft_immut(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let input: &[Complex<A>] = unsafe { workaround_transmute(input) };
        let output: &mut [Complex<A>] = unsafe { workaround_transmute_mut(output) };
        let scratch: &mut [Complex<A>] = unsafe { workaround_transmute_mut(scratch) };
        self.process(input.as_ptr(), output.as_mut_ptr(), scratch);
    }

    fn scratch_len(&self) -> usize {
        2 * (self.len + 4096 / std::mem::size_of::<Complex<A>>())
    }
}

/// How far apart, modulo 4 KiB, the buffers of consecutive passes start.
const GAP: usize = 1024;

/// The tail of `slice` starting at the first element whose address is `target` modulo 4 KiB,
/// rounded down to 64 bytes.
fn place<C>(slice: &mut [C], target: usize) -> &mut [C] {
    let size = std::mem::size_of::<C>();
    let target = target % 4096 / 64 * 64;
    let at = slice.as_ptr() as usize % 4096;
    &mut slice[(target + 4096 - at) % 4096 / size..]
}

boilerplate_avx_fft!(
    StockhamAvx,
    |this: &StockhamAvx<_, _>| this.len,
    |this: &StockhamAvx<_, _>| this.scratch_len(),
    |this: &StockhamAvx<_, _>| this.scratch_len(),
    |this: &StockhamAvx<_, _>| this.scratch_len()
);

/// The shortest power of two the AVX planner computes with [`StockhamAvx`], or `None` on CPUs
/// where the mixed radix plans are faster.
///
/// Stockham passes stream the whole buffer and scratch through the cache. Zen cores refill L1
/// fast enough for that to beat the mixed radix plans; Skylake-derived Intel cores stall on the
/// refills and are faster with the cache-resident butterflies of the mixed radix plans.
pub fn stockham_min_len() -> Option<usize> {
    is_amd().then_some(1024)
}

fn is_amd() -> bool {
    use std::sync::OnceLock;
    static AMD: OnceLock<bool> = OnceLock::new();
    *AMD.get_or_init(|| {
        let leaf = std::arch::x86_64::__cpuid(0);
        let vendor = [leaf.ebx, leaf.edx, leaf.ecx]
            .map(u32::to_le_bytes)
            .concat();
        vendor == *b"AuthenticAMD" || vendor == *b"HygonGenuine"
    })
}

/// The radix of each pass for a `len`-point FFT, the first pass first, for vectors of `lanes`
/// complex numbers. Sizes up to 2^18 use the fastest plan measured on Zen 3 and Zen 4.
fn radixes(len: usize, lanes: usize) -> Vec<usize> {
    let bits = len.trailing_zeros() as usize;
    let measured: &[usize] = match (bits, lanes) {
        (10, 4) => &[8, 4, 2, 4, 4],
        (10, _) => &[4, 4, 4, 4, 4],
        (11, 4) => &[8, 4, 2, 4, 8],
        (11, _) => &[4, 4, 4, 4, 8],
        (12, 4) => &[4, 4, 8, 4, 8],
        (12, _) => &[4, 4, 4, 8, 8],
        (13, 4) => &[4, 4, 4, 4, 4, 8],
        (13, _) => &[4, 4, 8, 8, 8],
        (14, 4) => &[4, 4, 4, 4, 8, 8],
        (14, _) => &[4, 8, 8, 8, 8],
        (15, _) => &[8, 8, 8, 8, 8],
        (16, 4) => &[4, 4, 8, 8, 8, 8],
        (16, _) => &[4, 8, 8, 8, 8, 4],
        (17, _) => &[4, 8, 8, 8, 8, 8],
        (18, _) => &[8, 8, 8, 8, 8, 8],
        _ => &[],
    };
    if !measured.is_empty() {
        return measured.to_vec();
    }
    let mut radixes = vec![8; bits / 3];
    match bits % 3 {
        1 => radixes.push(2),
        2 => radixes.push(4),
        _ => {}
    }
    radixes
}

/// The vector operations the passes need, for AVX vectors and for AVX-512 ones. Pointers address
/// the real part of a complex number.
trait Lanes: Copy {
    type Scalar: Copy + One;
    const COMPLEX: usize;
    unsafe fn load(ptr: *const Self::Scalar) -> Self;
    unsafe fn store(ptr: *mut Self::Scalar, value: Self);
    unsafe fn splat(ptr: *const Self::Scalar) -> Self;
    unsafe fn add(a: Self, b: Self) -> Self;
    unsafe fn sub(a: Self, b: Self) -> Self;
    unsafe fn mul(a: Self, b: Self) -> Self;
    unsafe fn fmaddsub(a: Self, b: Self, c: Self) -> Self;
    unsafe fn fmsubadd(a: Self, b: Self, c: Self) -> Self;
    unsafe fn swap(self) -> Self;
    unsafe fn half_root2() -> Self;
    unsafe fn neg(self) -> Self;
}

impl<V: AvxVector256> Lanes for V {
    type Scalar = V::ScalarType;
    const COMPLEX: usize = V::COMPLEX_PER_VECTOR;
    #[inline(always)]
    unsafe fn load(ptr: *const Self::Scalar) -> Self {
        unsafe { V::load_complex(ptr as *const Complex<Self::Scalar>) }
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Scalar, value: Self) {
        unsafe { V::store_complex(ptr as *mut Complex<Self::Scalar>, value) }
    }
    #[inline(always)]
    unsafe fn splat(ptr: *const Self::Scalar) -> Self {
        unsafe { V::broadcast_scalar(ptr) }
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        unsafe { <V as AvxVector>::add(a, b) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        unsafe { <V as AvxVector>::sub(a, b) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        unsafe { <V as AvxVector>::mul(a, b) }
    }
    #[inline(always)]
    unsafe fn fmaddsub(a: Self, b: Self, c: Self) -> Self {
        unsafe { <V as AvxVector>::fmaddsub(a, b, c) }
    }
    #[inline(always)]
    unsafe fn fmsubadd(a: Self, b: Self, c: Self) -> Self {
        unsafe { <V as AvxVector>::fmsubadd(a, b, c) }
    }
    #[inline(always)]
    unsafe fn swap(self) -> Self {
        unsafe { self.swap_complex_components() }
    }
    #[inline(always)]
    unsafe fn half_root2() -> Self {
        unsafe { <V as AvxVector>::half_root2() }
    }
    #[inline(always)]
    unsafe fn neg(self) -> Self {
        unsafe { <V as AvxVector>::neg(self) }
    }
}

impl Lanes for __m512 {
    type Scalar = f32;
    const COMPLEX: usize = 8;
    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        unsafe { _mm512_loadu_ps(ptr) }
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut f32, value: Self) {
        unsafe { _mm512_storeu_ps(ptr, value) }
    }
    #[inline(always)]
    unsafe fn splat(ptr: *const f32) -> Self {
        unsafe { _mm512_set1_ps(*ptr) }
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        unsafe { _mm512_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        unsafe { _mm512_sub_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        unsafe { _mm512_mul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn fmaddsub(a: Self, b: Self, c: Self) -> Self {
        unsafe { _mm512_fmaddsub_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn fmsubadd(a: Self, b: Self, c: Self) -> Self {
        unsafe { _mm512_fmsubadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn swap(self) -> Self {
        unsafe { _mm512_permute_ps::<0xB1>(self) }
    }
    #[inline(always)]
    unsafe fn half_root2() -> Self {
        unsafe { _mm512_set1_ps(std::f32::consts::FRAC_1_SQRT_2) }
    }
    #[inline(always)]
    unsafe fn neg(self) -> Self {
        unsafe { _mm512_sub_ps(_mm512_setzero_ps(), self) }
    }
}

impl Lanes for __m512d {
    type Scalar = f64;
    const COMPLEX: usize = 4;
    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> Self {
        unsafe { _mm512_loadu_pd(ptr) }
    }
    #[inline(always)]
    unsafe fn store(ptr: *mut f64, value: Self) {
        unsafe { _mm512_storeu_pd(ptr, value) }
    }
    #[inline(always)]
    unsafe fn splat(ptr: *const f64) -> Self {
        unsafe { _mm512_set1_pd(*ptr) }
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        unsafe { _mm512_add_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        unsafe { _mm512_sub_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        unsafe { _mm512_mul_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn fmaddsub(a: Self, b: Self, c: Self) -> Self {
        unsafe { _mm512_fmaddsub_pd(a, b, c) }
    }
    #[inline(always)]
    unsafe fn fmsubadd(a: Self, b: Self, c: Self) -> Self {
        unsafe { _mm512_fmsubadd_pd(a, b, c) }
    }
    #[inline(always)]
    unsafe fn swap(self) -> Self {
        unsafe { _mm512_permute_pd::<0x55>(self) }
    }
    #[inline(always)]
    unsafe fn half_root2() -> Self {
        unsafe { _mm512_set1_pd(std::f64::consts::FRAC_1_SQRT_2) }
    }
    #[inline(always)]
    unsafe fn neg(self) -> Self {
        unsafe { _mm512_sub_pd(_mm512_setzero_pd(), self) }
    }
}

#[inline(always)]
unsafe fn mul_split<V: Lanes>(value: V, [re, im]: [V; 2]) -> V {
    unsafe { V::fmaddsub(value, re, V::mul(value.swap(), im)) }
}

/// `a + r * b` for `b` given as `swapped = b.swap()`, where `r` is `-i` for forward FFTs and `i`
/// for inverse ones. One fused instruction instead of a sign flip and an add.
#[inline(always)]
unsafe fn rotate_add<V: Lanes, const FORWARD: bool>(a: V, swapped: V, one: V) -> V {
    unsafe {
        if FORWARD {
            V::fmsubadd(a, one, swapped)
        } else {
            V::fmaddsub(a, one, swapped)
        }
    }
}

/// `a - r * b`, the counterpart of [`rotate_add`].
#[inline(always)]
unsafe fn rotate_sub<V: Lanes, const FORWARD: bool>(a: V, swapped: V, one: V) -> V {
    unsafe {
        if FORWARD {
            V::fmaddsub(a, one, swapped)
        } else {
            V::fmsubadd(a, one, swapped)
        }
    }
}

#[derive(Clone, Copy)]
struct Constants<V> {
    one: V,
    half_root2: V,
    neg_half_root2: V,
}

impl<V: Lanes> Constants<V> {
    #[inline(always)]
    unsafe fn new() -> Self {
        unsafe {
            let one = V::splat(&V::Scalar::one());
            let half_root2 = V::half_root2();
            Self {
                one,
                half_root2,
                neg_half_root2: half_root2.neg(),
            }
        }
    }
}

#[inline(always)]
unsafe fn dft4<V: Lanes, const FORWARD: bool>(x: [V; 4], one: V) -> [V; 4] {
    unsafe {
        let t0 = V::add(x[0], x[2]);
        let t1 = V::sub(x[0], x[2]);
        let t2 = V::add(x[1], x[3]);
        let t3 = V::sub(x[1], x[3]).swap();
        [
            V::add(t0, t2),
            rotate_add::<V, FORWARD>(t1, t3, one),
            V::sub(t0, t2),
            rotate_sub::<V, FORWARD>(t1, t3, one),
        ]
    }
}

#[inline(always)]
unsafe fn dft8<V: Lanes, const FORWARD: bool>(a: [V; 8], c: Constants<V>) -> [V; 8] {
    unsafe {
        let one = c.one;
        let u: [V; 4] = std::array::from_fn(|k| V::add(a[k], a[k + 4]));
        let v: [V; 4] = std::array::from_fn(|k| V::sub(a[k], a[k + 4]));
        let even = dft4::<V, FORWARD>(u, one);

        let v1 = V::mul(
            rotate_add::<V, FORWARD>(v[1], v[1].swap(), one),
            c.half_root2,
        );
        let v3 = V::mul(
            rotate_sub::<V, FORWARD>(v[3], v[3].swap(), one),
            c.neg_half_root2,
        );
        let v2 = v[2].swap();
        let t0 = rotate_add::<V, FORWARD>(v[0], v2, one);
        let t1 = rotate_sub::<V, FORWARD>(v[0], v2, one);
        let t2 = V::add(v1, v3);
        let t3 = V::sub(v1, v3).swap();
        let odd = [
            V::add(t0, t2),
            rotate_add::<V, FORWARD>(t1, t3, one),
            V::sub(t0, t2),
            rotate_sub::<V, FORWARD>(t1, t3, one),
        ];
        [
            even[0], odd[0], even[1], odd[1], even[2], odd[2], even[3], odd[3],
        ]
    }
}

#[inline(always)]
unsafe fn butterfly<V: Lanes, const R: usize, const FORWARD: bool>(
    rows: [V; R],
    c: Constants<V>,
) -> [V; R] {
    unsafe {
        match R {
            2 => std::array::from_fn(|i| {
                if i == 0 {
                    V::add(rows[0], rows[1])
                } else {
                    V::sub(rows[0], rows[1])
                }
            }),
            4 => {
                let out = dft4::<V, FORWARD>(std::array::from_fn(|i| rows[i]), c.one);
                std::array::from_fn(|i| out[i])
            }
            8 => {
                let out = dft8::<V, FORWARD>(std::array::from_fn(|i| rows[i]), c);
                std::array::from_fn(|i| out[i])
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn transpose<V: AvxVector, const R: usize>(rows: [V; R]) -> [V; R] {
    unsafe {
        match R {
            4 => {
                let out = V::transpose4_packed(std::array::from_fn(|i| rows[i]));
                std::array::from_fn(|i| out[i])
            }
            8 => {
                let out = V::transpose8_packed(std::array::from_fn(|i| rows[i]));
                std::array::from_fn(|i| out[i])
            }
            _ => unreachable!(),
        }
    }
}

/// Butterflies over rows `p, p + 1, ...` (one per lane), reading `x[p + k * len / R]` and writing
/// `y[R * p + j]`.
#[target_feature(enable = "avx", enable = "fma")]
unsafe fn first_pass<A: AvxNum, const R: usize, const FORWARD: bool>(
    len: usize,
    twiddles: &[[A::VectorType; 2]],
    x: *const Complex<A>,
    y: *mut Complex<A>,
) {
    let constants = unsafe { Constants::<A::VectorType>::new() };
    let lanes = A::VectorType::COMPLEX_PER_VECTOR;
    let rows = len / R;
    let mut twiddles = twiddles.chunks_exact(R - 1);
    for row in (0..rows).step_by(lanes) {
        let Some(twiddles) = twiddles.next() else {
            return;
        };
        unsafe {
            let inputs: [A::VectorType; R] =
                std::array::from_fn(|k| A::VectorType::load_complex(x.add(row + k * rows)));
            let mut outputs = butterfly::<_, R, FORWARD>(inputs, constants);
            for (output, &twiddle) in outputs[1..].iter_mut().zip(twiddles) {
                *output = mul_split(*output, twiddle);
            }
            let packed = transpose::<_, R>(outputs);
            for (index, &vector) in packed.iter().enumerate() {
                A::VectorType::store_complex(y.add(R * row + index * lanes), vector);
            }
        }
    }
}

/// Butterflies for `pass`, reading `x[q + s * (p + k * rows)]` and writing `y[q + s * (R * p + j)]`
/// for row `p` and lanes `q` of each stride `s`. Indexes count complex numbers.
#[inline(always)]
unsafe fn later_body<V: Lanes, const R: usize, const FORWARD: bool>(
    rows: usize,
    stride: usize,
    twiddles: &[V::Scalar],
    x: *const V::Scalar,
    y: *mut V::Scalar,
) {
    let constants = unsafe { Constants::<V>::new() };
    let lanes = V::COMPLEX;
    if rows == 1 {
        for q in (0..stride).step_by(lanes) {
            unsafe {
                let inputs: [V; R] = std::array::from_fn(|k| V::load(x.add(2 * (q + stride * k))));
                let outputs = butterfly::<_, R, FORWARD>(inputs, constants);
                for (j, &output) in outputs.iter().enumerate() {
                    V::store(y.add(2 * (q + stride * j)), output);
                }
            }
        }
        return;
    }
    for (row, twiddles) in twiddles.chunks_exact(2 * (R - 1)).enumerate() {
        let twiddles: [[V; 2]; R] = std::array::from_fn(|j| unsafe {
            let pair = twiddles.as_ptr().add(2 * j.max(1) - 2);
            [V::splat(pair), V::splat(pair.add(1))]
        });
        let input = unsafe { x.add(2 * stride * row) };
        let output = unsafe { y.add(2 * stride * R * row) };
        for q in (0..stride).step_by(lanes) {
            unsafe {
                let inputs: [V; R] =
                    std::array::from_fn(|k| V::load(input.add(2 * (q + stride * k * rows))));
                let outputs = butterfly::<_, R, FORWARD>(inputs, constants);
                V::store(output.add(2 * q), outputs[0]);
                for j in 1..R {
                    let value = mul_split(outputs[j], twiddles[j]);
                    V::store(output.add(2 * (q + stride * j)), value);
                }
            }
        }
    }
}

#[target_feature(enable = "avx", enable = "fma")]
unsafe fn later_pass<A: AvxNum, const R: usize, const FORWARD: bool>(
    pass: &Pass<A>,
    x: *const Complex<A>,
    y: *mut Complex<A>,
) {
    unsafe {
        later_body::<A::VectorType, R, FORWARD>(
            pass.rows,
            pass.stride,
            &pass.twiddles,
            x as *const A,
            y as *mut A,
        )
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn later_pass_wide<V: Lanes, const R: usize, const FORWARD: bool>(
    rows: usize,
    stride: usize,
    twiddles: &[V::Scalar],
    x: *const V::Scalar,
    y: *mut V::Scalar,
) {
    unsafe { later_body::<V, R, FORWARD>(rows, stride, twiddles, x, y) }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    #[test]
    fn test_stockham_avx_f32() {
        for bits in 6..=13 {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let fft = StockhamAvx::<f32, f32>::new(1 << bits, direction).unwrap();
                check_fft_algorithm::<f32>(&fft, 1 << bits, direction);
            }
        }
    }

    #[test]
    fn test_stockham_avx_f64() {
        for bits in 6..=13 {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let fft = StockhamAvx::<f64, f64>::new(1 << bits, direction).unwrap();
                check_fft_algorithm::<f64>(&fft, 1 << bits, direction);
            }
        }
    }

    #[test]
    fn test_stockham_avx_without_avx512() {
        for bits in [6, 9, 12] {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let mut fft = StockhamAvx::<f32, f32>::new(1 << bits, direction).unwrap();
                fft.wide = false;
                check_fft_algorithm::<f32>(&fft, 1 << bits, direction);
                let mut fft = StockhamAvx::<f64, f64>::new(1 << bits, direction).unwrap();
                fft.wide = false;
                check_fft_algorithm::<f64>(&fft, 1 << bits, direction);
            }
        }
    }

    #[test]
    fn radixes_cover_every_bit() {
        for bits in 6..=24 {
            for lanes in [2, 4] {
                let radixes = radixes(1 << bits, lanes);
                assert_eq!(radixes.iter().product::<usize>(), 1 << bits);
                assert_ne!(radixes[0], 2);
            }
        }
    }
}
