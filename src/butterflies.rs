//! These are the unrolled butterflies for specific FFT lengths. They are mostly
//! translations of the butterfly functions from KissFFT, copyright Mark Borgerding.

use num::{Complex, Zero};
use common::FFTnum;

pub unsafe fn butterfly_2_single<T: FFTnum>(data: &mut [Complex<T>], stride: usize)
{
    let idx_1 = 0usize;
    let idx_2 = stride;

    let temp = *data.get_unchecked(idx_2);
    data.get_unchecked_mut(idx_2).re = data.get_unchecked(idx_1).re - temp.re;
    data.get_unchecked_mut(idx_2).im = data.get_unchecked(idx_1).im - temp.im;
    data.get_unchecked_mut(idx_1).re = data.get_unchecked(idx_1).re + temp.re;
    data.get_unchecked_mut(idx_1).im = data.get_unchecked(idx_1).im + temp.im;
}

pub unsafe fn butterfly_4<T: FFTnum>(data: &mut [Complex<T>],
                             stride: usize,
                             twiddles: &[Complex<T>],
                             num_ffts: usize,
                             inverse: bool)
{
    let mut idx = 0usize;
    let mut tw_idx_1 = 0usize;
    let mut tw_idx_2 = 0usize;
    let mut tw_idx_3 = 0usize;
    let mut scratch: [Complex<T>; 6] = [Zero::zero(); 6];
    for _ in 0..num_ffts {
        scratch[0] = data.get_unchecked(idx + 1 * num_ffts) * twiddles[tw_idx_1];
        scratch[1] = data.get_unchecked(idx + 2 * num_ffts) * twiddles[tw_idx_2];
        scratch[2] = data.get_unchecked(idx + 3 * num_ffts) * twiddles[tw_idx_3];
        scratch[5] = data.get_unchecked(idx) - scratch[1];
        *data.get_unchecked_mut(idx) = data.get_unchecked(idx) + scratch[1];
        scratch[3] = scratch[0] + scratch[2];
        scratch[4] = scratch[0] - scratch[2];
        *data.get_unchecked_mut(idx + 2 * num_ffts) = data.get_unchecked(idx) - scratch[3];
        *data.get_unchecked_mut(idx) = data.get_unchecked(idx) + scratch[3];
        if inverse {
            data.get_unchecked_mut(idx + num_ffts).re = scratch[5].re - scratch[4].im;
            data.get_unchecked_mut(idx + num_ffts).im = scratch[5].im + scratch[4].re;
            data.get_unchecked_mut(idx + 3 * num_ffts).re = scratch[5].re + scratch[4].im;
            data.get_unchecked_mut(idx + 3 * num_ffts).im = scratch[5].im - scratch[4].re;
        } else {
            data.get_unchecked_mut(idx + num_ffts).re = scratch[5].re + scratch[4].im;
            data.get_unchecked_mut(idx + num_ffts).im = scratch[5].im - scratch[4].re;
            data.get_unchecked_mut(idx + 3 * num_ffts).re = scratch[5].re - scratch[4].im;
            data.get_unchecked_mut(idx + 3 * num_ffts).im = scratch[5].im + scratch[4].re;
        }

        tw_idx_1 += 1 * stride;
        tw_idx_2 += 2 * stride;
        tw_idx_3 += 3 * stride;
        idx += 1;
    }
}

pub unsafe fn butterfly_4_single<T: FFTnum>(data: &mut [Complex<T>], stride: usize, inverse: bool) {
    let mut scratch: [Complex<T>; 6] = [Zero::zero(); 6];

    scratch[0] = *data.get_unchecked(stride);
    scratch[1] = *data.get_unchecked(2 * stride);
    scratch[2] = *data.get_unchecked(3 * stride);
    scratch[5] = *data.get_unchecked(0) - scratch[1];
    *data.get_unchecked_mut(0) = data.get_unchecked(0) + scratch[1];
    scratch[3] = scratch[0] + scratch[2];
    scratch[4] = scratch[0] - scratch[2];
    *data.get_unchecked_mut(2 * stride) = data.get_unchecked(0) - scratch[3];
    *data.get_unchecked_mut(0) = data.get_unchecked(0) + scratch[3];
    if inverse {
        data.get_unchecked_mut(stride).re = scratch[5].re - scratch[4].im;
        data.get_unchecked_mut(stride).im = scratch[5].im + scratch[4].re;
        data.get_unchecked_mut(3 * stride).re = scratch[5].re + scratch[4].im;
        data.get_unchecked_mut(3 * stride).im = scratch[5].im - scratch[4].re;
    } else {
        data.get_unchecked_mut(stride).re = scratch[5].re + scratch[4].im;
        data.get_unchecked_mut(stride).im = scratch[5].im - scratch[4].re;
        data.get_unchecked_mut(3 * stride).re = scratch[5].re - scratch[4].im;
        data.get_unchecked_mut(3 * stride).im = scratch[5].im + scratch[4].re;
    }
}

#[cfg(test)]
#[cfg(feature = "bench")]
mod benches {
    extern crate test;

    use std::f32;
    use std::iter::repeat;
    use self::test::Bencher;
    use num::{Complex, Zero};
    use super::{butterfly_2, butterfly_3, butterfly_4, butterfly_5};

    fn set_up_bench(butterfly_len: usize,
                    stride: usize,
                    num_ffts: usize)
                    -> (Vec<Complex<f32>>, Vec<Complex<f32>>) {
        let len = butterfly_len * stride * num_ffts;
        let twiddles: Vec<Complex<f32>> = (0..len)
            .map(|i| -1. * (i as f32) * 2.0 * f32::consts::PI / (len as f32))
            .map(|phase| Complex::from_polar(&1., &phase))
            .collect();

        let data: Vec<Complex<f32>> = repeat(Zero::zero()).take(len).collect();
        (data, twiddles)
    }

    #[bench]
    fn bench_butterfly_2(b: &mut Bencher) {
        let stride = 4usize;
        let num_ffts = 1000usize;
        let (mut data, twiddles) = set_up_bench(2, stride, num_ffts);

        b.iter(|| unsafe { butterfly_2(&mut data, stride, &twiddles, num_ffts) });
    }

    #[bench]
    fn bench_butterfly_3(b: &mut Bencher) {
        let stride = 4usize;
        let num_ffts = 1000usize;
        let (mut data, twiddles) = set_up_bench(3, stride, num_ffts);

        b.iter(|| unsafe { butterfly_3(&mut data, stride, &twiddles, num_ffts) });
    }

    #[bench]
    fn bench_butterfly_4(b: &mut Bencher) {
        let stride = 4usize;
        let num_ffts = 1000usize;
        let (mut data, twiddles) = set_up_bench(4, stride, num_ffts);

        b.iter(|| unsafe { butterfly_4(&mut data, stride, &twiddles, num_ffts, false) });
    }

    #[bench]
    fn bench_butterfly_5(b: &mut Bencher) {
        let stride = 4usize;
        let num_ffts = 1000usize;
        let (mut data, twiddles) = set_up_bench(5, stride, num_ffts);

        b.iter(|| unsafe { butterfly_5(&mut data, stride, &twiddles, num_ffts) });
    }
}
