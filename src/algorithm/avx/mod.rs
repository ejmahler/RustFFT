use std::sync::Arc;
use ::common::FFTnum;
use ::Fft;
use plan::FFTplanner;

mod avx_utils;
mod avx_split_radix;
mod avx_mixed_radix;
mod avx_butterflies;
mod avx_bluesteins;

pub use self::avx_split_radix::SplitRadixAvx;
pub use self::avx_bluesteins::BluesteinsAvx;
pub use self::avx_mixed_radix::{MixedRadix2xnAvx, MixedRadix4xnAvx, MixedRadix8xnAvx, MixedRadix16xnAvx};
pub use self::avx_butterflies::{MixedRadixAvx4x2, MixedRadixAvx4x4, MixedRadixAvx4x8, MixedRadixAvx8x8};


// we're implementing this as a trait so that we can specialize it for f32 and f64, 
// and we're implementing it in this module to minimuze the number of "#[cfg()]" annotations we need
pub(crate) trait MakeAvxButterfly<T: FFTnum> {
    fn make_avx_butterfly(&self, len: usize, inverse: bool) -> Option<Arc<dyn Fft<T>>>;
}

impl<T: FFTnum> MakeAvxButterfly<T> for FFTplanner<T> {
    default fn make_avx_butterfly(&self, _len: usize, _inverse: bool) -> Option<Arc<dyn Fft<T>>> {
        // If we're not specialized, then we have no AVX implementations for this type
        None
    }
}
impl MakeAvxButterfly<f32> for FFTplanner<f32> {
    fn make_avx_butterfly(&self, len: usize, inverse: bool) -> Option<Arc<dyn Fft<f32>>> {
        
        fn wrap_butterfly(butterfly: impl Fft<f32> + 'static) -> Option<Arc<dyn Fft<f32>>> {
            Some(Arc::new(butterfly) as Arc<dyn Fft<f32>>)
        }

        match len {
            8 =>    MixedRadixAvx4x2::new(inverse).map_or(None, wrap_butterfly),
            16 =>   MixedRadixAvx4x4::new(inverse).map_or(None, wrap_butterfly),
            32 =>   MixedRadixAvx4x8::new(inverse).map_or(None, wrap_butterfly),
            64 =>   MixedRadixAvx8x8::new(inverse).map_or(None, wrap_butterfly),
            _ => None
        }
    }
}