use std::sync::Arc;

use num_complex::Complex;

use crate::array_utils::{
    ArcSlice, DoubleBuf, choose_scratch, workaround_transmute, workaround_transmute_mut,
};
use crate::simd::parallel_fft::ParallelFft;
use crate::simd::simd_array::{SimdComplexArray, SimdComplexArrayMut};
use crate::simd::simd_vector::SimdVector;
use crate::{Fft, FftDirection, FftNum};

use super::radix_factor::RadixFactor;

pub trait LayerFactory<V: SimdVector, T: FftNum> {
    fn construct_layer(
        &self,
        factor: RadixFactor,
        child_layer: Arc<dyn CrossLayer<V, T>>,
        cross_fft_len: usize,
        twiddles: ArcSlice<V>,
        direction: FftDirection,
    ) -> Arc<dyn CrossLayer<V, T>>;
}

pub trait CrossLayer<V: SimdVector, T: FftNum>: Sync + Send {
    // SAFETY:
    // - The required CPU features for `V` must be present
    unsafe fn process_layer(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    // SAFETY:
    // - The required CPU features for `V` must be present
    unsafe fn process_layer_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );
}

pub struct BaseLayer<T: FftNum>(pub Arc<dyn Fft<T>>);
impl<V: SimdVector, T: FftNum> CrossLayer<V, T> for BaseLayer<T> {
    #[inline(always)]
    unsafe fn process_layer(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        self.0.process_with_scratch(buffer, scratch);
    }

    #[inline(always)]
    unsafe fn process_layer_outofplace(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.0
            .process_outofplace_with_scratch(input, output, scratch);
    }
}

pub struct LayerInstance<V: SimdVector, T: FftNum, const RADIX: usize, B: ParallelFft<V, RADIX>> {
    twiddles: ArcSlice<V>,
    child_fft: Arc<dyn CrossLayer<V, T>>,
    butterfly: B,
    cross_fft_len: usize,
}

impl<V: SimdVector, T: FftNum, const RADIX: usize, B: ParallelFft<V, RADIX>>
    LayerInstance<V, T, RADIX, B>
{
    pub fn new(
        twiddles: ArcSlice<V>,
        child_fft: Arc<dyn CrossLayer<V, T>>,
        butterfly: B,
        cross_fft_len: usize,
    ) -> Self {
        Self {
            twiddles,
            child_fft,
            butterfly,
            cross_fft_len,
        }
    }

    #[inline(always)]
    pub fn process_layer(&self, mut buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let inner_fft_len = self.cross_fft_len / RADIX;
        while let Some((chunk, remainder)) = buffer.split_at_mut_checked(self.cross_fft_len) {
            buffer = remainder;

            unsafe { self.child_fft.process_layer(chunk, scratch) };

            // Now that the children are done, handle this layer's cross ffts
            let chunk = unsafe { workaround_transmute_mut(chunk) };
            unsafe {
                cross_layer::<V, RADIX>(chunk, &self.twiddles, inner_fft_len, &self.butterfly)
            }
        }
    }

    #[inline(always)]
    pub fn process_layer_outofplace(
        &self,
        mut input: &mut [Complex<T>],
        mut output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let inner_fft_len = self.cross_fft_len / RADIX;
        while let (Some((chunk_in, remainder_in)), Some((chunk_out, remainder_out))) = (
            input.split_at_mut_checked(self.cross_fft_len),
            output.split_at_mut_checked(self.cross_fft_len),
        ) {
            input = remainder_in;
            output = remainder_out;

            unsafe {
                self.child_fft
                    .process_layer(chunk_in, choose_scratch(output, scratch))
            };

            // Now that the children are done, handle this layer's cross ffts
            let buffer = DoubleBuf {
                input: unsafe { workaround_transmute(chunk_in) },
                output: unsafe { workaround_transmute_mut(chunk_out) },
            };
            unsafe {
                cross_layer::<V, RADIX>(buffer, &self.twiddles, inner_fft_len, &self.butterfly)
            }
        }
    }
}

macro_rules! boilerplate_simd_radixn_layer {
    ($layer_struct_name:ident, $(#[$attr:meta])*) => {
        impl<V: SimdVector, T: FftNum> CrossLayer<V, T> for $layer_struct_name<V, T> {
            $(#[$attr])*
            unsafe fn process_layer(
                &self,
                buffer: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                self.0.process_layer(buffer, scratch)
            }

            $(#[$attr])*
            unsafe fn process_layer_outofplace(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
            ) {
                self.0.process_layer_outofplace(input, output, scratch)
            }
        }
    };
}
pub(crate) use boilerplate_simd_radixn_layer;

// Boilerplate macro for RadixN that implements the layer struct, implements the CrossLayer trait for it,
// and declares wrapper functions with the appropriate target feature attributes
macro_rules! boilerplate_simd_radixn {
    ($layer_factory:ident, $(#[$attr:meta])*) => {
        pub struct $layer_factory;
        impl<V: SimdVector, T: FftNum> LayerFactory<V, T> for $layer_factory {
            fn construct_layer(
                &self,
                radix: RadixFactor,
                child: Arc<dyn CrossLayer<V, T>>,
                cross_fft_len: usize,
                twiddles: ArcSlice<V>,
                direction: FftDirection
            ) -> Arc<dyn CrossLayer<V, T>> {
                match radix {
                    RadixFactor::Factor2 => Arc::new(ImplLayer2(LayerInstance::new(twiddles, child, unsafe { V::make_butterfly2(direction) }, cross_fft_len))),
                    RadixFactor::Factor3 => Arc::new(ImplLayer3(LayerInstance::new(twiddles, child, unsafe { V::make_butterfly3(direction) }, cross_fft_len))),
                    RadixFactor::Factor4 => Arc::new(ImplLayer4(LayerInstance::new(twiddles, child, unsafe { V::make_butterfly4(direction) }, cross_fft_len))),
                    RadixFactor::Factor5 => Arc::new(ImplLayer5(LayerInstance::new(twiddles, child, unsafe { V::make_butterfly5(direction) }, cross_fft_len))),
                    RadixFactor::Factor6 => Arc::new(ImplLayer6(LayerInstance::new(twiddles, child, unsafe { V::make_butterfly6(direction) }, cross_fft_len))),
                    RadixFactor::Factor7 => Arc::new(ImplLayer7(LayerInstance::new(twiddles, child, unsafe { V::make_butterfly7(direction) }, cross_fft_len))),
                }
            }
        }

        // These following types aren't intended to be referenced outside of this macro, and they don't need to be.
        struct ImplLayer2<V: SimdVector, T: FftNum>(LayerInstance<V, T, 2, V::Butterfly2>);
        struct ImplLayer3<V: SimdVector, T: FftNum>(LayerInstance<V, T, 3, V::Butterfly3>);
        struct ImplLayer4<V: SimdVector, T: FftNum>(LayerInstance<V, T, 4, V::Butterfly4>);
        struct ImplLayer5<V: SimdVector, T: FftNum>(LayerInstance<V, T, 5, V::Butterfly5>);
        struct ImplLayer6<V: SimdVector, T: FftNum>(LayerInstance<V, T, 6, V::Butterfly6>);
        struct ImplLayer7<V: SimdVector, T: FftNum>(LayerInstance<V, T, 7, V::Butterfly7>);
        boilerplate_simd_radixn_layer!(ImplLayer2, $(#[$attr])*);
        boilerplate_simd_radixn_layer!(ImplLayer3, $(#[$attr])*);
        boilerplate_simd_radixn_layer!(ImplLayer4, $(#[$attr])*);
        boilerplate_simd_radixn_layer!(ImplLayer5, $(#[$attr])*);
        boilerplate_simd_radixn_layer!(ImplLayer6, $(#[$attr])*);
        boilerplate_simd_radixn_layer!(ImplLayer7, $(#[$attr])*);
    };
}
pub(crate) use boilerplate_simd_radixn;

// It would be ideal to declare gather_and_twiddle() as a lambda and use std::array::from_fn inside,
// but that approach failed to inline on wasm simd, causing a horrible performance regression
#[inline(always)]
unsafe fn gather_and_twiddle<V: SimdVector, const RADIX: usize>(
    data: &impl SimdComplexArray<V>,
    twiddles: &[V],
    num_columns: usize,
    idx: usize,
    tw_base: usize,
) -> [V; RADIX] {
    unsafe {
        let mut arr = [V::zero_vector(); RADIX];

        // The row-0 twiddle is always 1, so it's neither stored nor applied.
        arr[0] = data.load(idx);

        for r in 1..RADIX {
            let v = data.load(idx + r * num_columns);
            arr[r] = V::mul_complex(v, *twiddles.get_unchecked(tw_base + r - 1));
        }
        arr
    }
}

/// One cross-FFT layer: for each vector of columns, gather RADIX rows strided by `num_columns`,
/// apply the twiddles, run the column butterfly, scatter back.
///
/// Unrolled two vectors at a time, which is what the SIMD Radix4's `butterfly_4` does and is what
/// gets the two independent dependency chains needed to keep the FMA pipeline busy.
#[inline(always)]
pub unsafe fn cross_layer<V: SimdVector, const RADIX: usize>(
    mut data: impl SimdComplexArrayMut<V>,
    twiddles: &[V],
    num_columns: usize,
    butterfly: &impl ParallelFft<V, RADIX>,
) {
    unsafe {
        let complex_per_vector = V::COMPLEX_PER_VECTOR;
        // the whole vector columns, and then the column an inexact count leaves over
        let num_vector_columns = num_columns / complex_per_vector;
        let partial_columns = num_columns % complex_per_vector;
        let tw_stride = RADIX - 1;

        // A vector holds one or two complex numbers, so the leftover is at most a single column
        debug_assert!(partial_columns <= 1);
        debug_assert!(twiddles.len() >= num_columns.div_ceil(complex_per_vector) * tw_stride);

        if V::RADIXN_CROSS_LAYER_UNROLL {
            let (unroll_count, unroll_remainder) = (num_vector_columns / 2, num_vector_columns % 2);
            for i in 0..unroll_count {
                let vcol = i * 2;
                let idx = vcol * complex_per_vector;

                let a = gather_and_twiddle(&data, twiddles, num_columns, idx, vcol * tw_stride);
                let b = gather_and_twiddle(
                    &data,
                    twiddles,
                    num_columns,
                    idx + complex_per_vector,
                    (vcol + 1) * tw_stride,
                );

                let a = butterfly.parallel_fft(a);
                let b = butterfly.parallel_fft(b);

                for (r, (a_row, b_row)) in a.iter().zip(b.iter()).enumerate() {
                    data.store(*a_row, idx + r * num_columns);
                    data.store(*b_row, idx + complex_per_vector + r * num_columns);
                }
            }

            // an odd vector column count leaves one behind
            if unroll_remainder > 0 {
                let vcol = unroll_count * 2;
                let idx = vcol * complex_per_vector;
                let a = butterfly.parallel_fft(gather_and_twiddle(
                    &data,
                    twiddles,
                    num_columns,
                    idx,
                    vcol * tw_stride,
                ));
                for (r, a_row) in a.iter().enumerate() {
                    data.store(*a_row, idx + r * num_columns);
                }
            }
        } else {
            for i in 0..num_vector_columns {
                let vcol = i;
                let idx = vcol * complex_per_vector;

                let a = gather_and_twiddle(&data, twiddles, num_columns, idx, vcol * tw_stride);

                let a = butterfly.parallel_fft(a);

                for (r, a_row) in a.iter().enumerate() {
                    data.store(*a_row, idx + r * num_columns);
                }
            }
        };

        // A column count that isn't a whole number of vectors leaves a single column, so it can't be
        // loaded or stored as a vector: a full load of the last row would run past the end of the
        // buffer, and a full store would overwrite the first column of the next row. Load and store
        // only the low half instead. The butterfly is the same one, since it works down the columns
        // and the halves of a vector never interact. Only f32 ever gets here, since a vector holds a
        // single complex f64 and a count of those is always exact.
        if partial_columns > 0 {
            let vcol = num_vector_columns;
            let idx = vcol * complex_per_vector;
            let tw_base = vcol * tw_stride;
            let rows: [V; RADIX] = std::array::from_fn(|r| {
                let v: V = data.load1_lo(idx + r * num_columns);
                if r == 0 {
                    v
                } else {
                    V::mul_complex(v, *twiddles.get_unchecked(tw_base + r - 1))
                }
            });
            let a = butterfly.parallel_fft(rows);
            for (r, a_row) in a.iter().enumerate() {
                data.store1_lo(*a_row, idx + r * num_columns);
            }
        }
    }
}
