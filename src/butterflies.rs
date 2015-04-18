//! These are the unrolled butterflies for specific FFT lengths. They are mostly
//! translations of the butterfly functions from KissFFT, copyright Mark Borgerding.

use num::{Complex, Zero};

pub unsafe fn butterfly_2(data: &mut [Complex<f32>], stride: usize,
                   twiddles: &[Complex<f32>], num_ffts: usize) {
    let mut idx_1 = 0usize;
    let mut idx_2 = num_ffts;
    let mut twiddle_idx = 0usize;
    for _ in 0..num_ffts {
        let twiddle = twiddles.get_unchecked(twiddle_idx);
        let temp = data.get_unchecked(idx_2) * twiddle;
        *data.get_unchecked_mut(idx_2) = data.get_unchecked(idx_1) - temp;
        *data.get_unchecked_mut(idx_1) = data.get_unchecked(idx_1) + temp;
        twiddle_idx += stride;
        idx_1 += 1;
        idx_2 += 1;
    }
}

pub unsafe fn butterfly_3(data: &mut [Complex<f32>], stride: usize,
                   twiddles: &[Complex<f32>], num_ffts: usize) {
    let mut idx = 0usize;
    let mut tw_idx_1 = 0usize;
    let mut tw_idx_2 = 0usize;
    let mut scratch: [Complex<f32>; 5] = [Zero::zero(); 5];
    for _ in 0..num_ffts {
        scratch[1] = data.get_unchecked(idx + 1 * num_ffts) * twiddles[tw_idx_1];
        scratch[2] = data.get_unchecked(idx + 2 * num_ffts) * twiddles[tw_idx_2];
        scratch[3] = scratch[1] + scratch[2];
        scratch[0] = scratch[1] - scratch[2];

        data.get_unchecked_mut(idx + num_ffts).re = data.get_unchecked(idx).re
                                                    - scratch[3].re / 2.0;
        data.get_unchecked_mut(idx + num_ffts).im = data.get_unchecked(idx).im
                                                    - scratch[3].im / 2.0;

        scratch[0].re = scratch[0].re * twiddles[stride * num_ffts].im;
        scratch[0].im = scratch[0].im * twiddles[stride * num_ffts].im;

        *data.get_unchecked_mut(idx) = data.get_unchecked(idx) + scratch[3];
        data.get_unchecked_mut(idx + 2 * num_ffts).re = data.get_unchecked(idx + num_ffts).re
                                                        + scratch[0].im;
        data.get_unchecked_mut(idx + 2 * num_ffts).im = data.get_unchecked(idx + num_ffts).im
                                                        - scratch[0].re;
        data.get_unchecked_mut(idx + num_ffts).re = data.get_unchecked(idx + num_ffts).re
                                                    - scratch[0].im;
        data.get_unchecked_mut(idx + num_ffts).im = data.get_unchecked(idx + num_ffts).im
                                                    + scratch[0].re;

        tw_idx_1 += 1 * stride;
        tw_idx_2 += 2 * stride;
        idx += 1;
    }
}

pub unsafe fn butterfly_4(data: &mut [Complex<f32>], stride: usize,
                   twiddles: &[Complex<f32>], num_ffts: usize, inverse: bool) {
    let mut idx = 0usize;
    let mut tw_idx_1 = 0usize;
    let mut tw_idx_2 = 0usize;
    let mut tw_idx_3 = 0usize;
    let mut scratch: [Complex<f32>; 6] = [Zero::zero(); 6];
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

pub unsafe fn butterfly_5(data: &mut [Complex<f32>], stride: usize,
                   twiddles: &[Complex<f32>], num_ffts: usize) {
    let mut idx_0 = 0 * num_ffts;
    let mut idx_1 = 1 * num_ffts;
    let mut idx_2 = 2 * num_ffts;
    let mut idx_3 = 3 * num_ffts;
    let mut idx_4 = 4 * num_ffts;
    let mut scratch: [Complex<f32>; 13] = [Zero::zero(); 13];
    let ya = twiddles[stride * num_ffts];
    let yb = twiddles[stride * 2 * num_ffts];
    for i in 0..num_ffts {
        scratch[0] = *data.get_unchecked(idx_0);
        scratch[1] = data.get_unchecked(idx_1) * twiddles[1 * i * stride];
        scratch[2] = data.get_unchecked(idx_2) * twiddles[2 * i * stride];
        scratch[3] = data.get_unchecked(idx_3) * twiddles[3 * i * stride];
        scratch[4] = data.get_unchecked(idx_4) * twiddles[4 * i * stride];

        scratch[7] = scratch[1] + scratch[4];
        scratch[10] = scratch[1] - scratch[4];
        scratch[8] = scratch[2] + scratch[3];
        scratch[9] = scratch[2] - scratch[3];

        data.get_unchecked_mut(idx_0).re += scratch[7].re + scratch[8].re;
        data.get_unchecked_mut(idx_0).im += scratch[7].im + scratch[8].im;

        scratch[5].re = scratch[0].re + scratch[7].re * ya.re + scratch[8].re * yb.re;
        scratch[5].im = scratch[0].im + scratch[7].im * ya.re + scratch[8].im * yb.re;

        scratch[6].re = scratch[10].im * ya.im + scratch[9].im * yb.im;
        scratch[6].im = -1.0 * scratch[10].re * ya.im - scratch[9].re * yb.im;

        *data.get_unchecked_mut(idx_1) = scratch[5] - scratch[6];
        *data.get_unchecked_mut(idx_4) = scratch[5] + scratch[6];

        scratch[11].re = scratch[0].re + scratch[7].re * yb.re + scratch[8].re * ya.re;
        scratch[11].im = scratch[0].im + scratch[7].im * yb.re + scratch[8].im * ya.re;
        scratch[12].re = -1.0 * scratch[10].im * yb.im + scratch[9].im * ya.im;
        scratch[12].im = scratch[10].re * yb.im - scratch[9].re * ya.im;

        *data.get_unchecked_mut(idx_2) = scratch[11] + scratch[12];
        *data.get_unchecked_mut(idx_3) = scratch[11] - scratch[12];

        idx_0 += 1;
        idx_1 += 1;
        idx_2 += 1;
        idx_3 += 1;
        idx_4 += 1;
    }
}

#[cfg(test)]
mod benches {
    extern crate test;

    use std::f32;
    use std::iter::repeat;
    use self::test::Bencher;
    use num::{Complex, Zero};
    use super::{butterfly_2, butterfly_3, butterfly_4, butterfly_5};

    fn set_up_bench(butterfly_len: usize, stride: usize, num_ffts: usize) -> (Vec<Complex<f32>>, Vec<Complex<f32>>) {
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

        b.iter(|| unsafe {
            butterfly_2(&mut data, stride, &twiddles, num_ffts)
        });
    }

    #[bench]
    fn bench_butterfly_3(b: &mut Bencher) {
        let stride = 4usize;
        let num_ffts = 1000usize;
        let (mut data, twiddles) = set_up_bench(3, stride, num_ffts);

        b.iter(|| unsafe {
            butterfly_3(&mut data, stride, &twiddles, num_ffts)
        });
    }

    #[bench]
    fn bench_butterfly_4(b: &mut Bencher) {
        let stride = 4usize;
        let num_ffts = 1000usize;
        let (mut data, twiddles) = set_up_bench(4, stride, num_ffts);

        b.iter(|| unsafe {
            butterfly_4(&mut data, stride, &twiddles, num_ffts, false)
        });
    }

    #[bench]
    fn bench_butterfly_5(b: &mut Bencher) {
        let stride = 4usize;
        let num_ffts = 1000usize;
        let (mut data, twiddles) = set_up_bench(5, stride, num_ffts);

        b.iter(|| unsafe {
            butterfly_5(&mut data, stride, &twiddles, num_ffts)
        });
    }
}
