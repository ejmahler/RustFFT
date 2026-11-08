//! A radix-8 column butterfly built from the radix-2 and radix-4 butterflies every backend already
//! has, so a RadixN layer can be radix 8 on any SIMD backend. Backends with a faster native one
//! use that instead.

use crate::simd::parallel_fft::ParallelFft;
use crate::simd::simd_vector::SimdVector;
use crate::{FftDirection, twiddles};

pub struct SimdParallelButterfly8<V: SimdVector> {
    butterfly4: V::Butterfly4,
    twiddles: [V; 3],
}

impl<V: SimdVector> SimdParallelButterfly8<V> {
    /// SAFETY: the CPU features `V` needs must be present.
    #[inline(always)]
    pub unsafe fn new(direction: FftDirection) -> Self {
        unsafe {
            let twiddle = |index| {
                let value = twiddles::compute_twiddle::<V::ScalarType>(index, 8, direction);
                if V::COMPLEX_PER_VECTOR == 1 {
                    V::load_complex(&value)
                } else {
                    V::load1_dup_complex(&value)
                }
            };
            Self {
                butterfly4: V::make_butterfly4(direction),
                twiddles: [twiddle(1), twiddle(2), twiddle(3)],
            }
        }
    }
}

impl<V: SimdVector> ParallelFft<V, 8> for SimdParallelButterfly8<V> {
    /// Decimation in frequency: a butterfly 2 across the halves, the odd half twiddled, then
    /// a butterfly 4 down each half gives the even and the odd outputs.
    #[inline(always)]
    fn parallel_fft(&self, x: [V; 8]) -> [V; 8] {
        unsafe {
            let even = [
                V::add(x[0], x[4]),
                V::add(x[1], x[5]),
                V::add(x[2], x[6]),
                V::add(x[3], x[7]),
            ];
            let odd = [
                V::sub(x[0], x[4]),
                V::mul_complex(V::sub(x[1], x[5]), self.twiddles[0]),
                V::mul_complex(V::sub(x[2], x[6]), self.twiddles[1]),
                V::mul_complex(V::sub(x[3], x[7]), self.twiddles[2]),
            ];
            let even = self.butterfly4.parallel_fft(even);
            let odd = self.butterfly4.parallel_fft(odd);
            [
                even[0], odd[0], even[1], odd[1], even[2], odd[2], even[3], odd[3],
            ]
        }
    }
}
