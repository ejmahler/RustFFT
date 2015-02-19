use num::{Complex, Zero};
use test::Bencher;
use std::f32;
use std::iter::repeat;

pub fn butterfly_2(data: &mut [Complex<f32>], stride: usize,
                   twiddles: &[Complex<f32>], num_ffts: usize) {
    let mut idx_1 = 0us;
    let mut idx_2 = num_ffts;
    let mut twiddle_idx = 0us;
    // TODO speed: use pointers instead of pointer offsets?
    for _ in 0..num_ffts {
        unsafe {
            let twiddle = twiddles.get_unchecked(twiddle_idx);
            let temp = data.get_unchecked(idx_2) * twiddle;
            *data.get_unchecked_mut(idx_2) = data.get_unchecked(idx_1) - temp;
            *data.get_unchecked_mut(idx_1) = data.get_unchecked(idx_1) + temp;
        }
        twiddle_idx += stride;
        idx_1 += 1;
        idx_2 += 1;
    }
}

#[bench]
fn bench_butterfly_2(b: &mut Bencher) {
    let stride = 4us;
    let num_ffts = 1000us;

    let len = 2 * stride * num_ffts;
    let twiddles: Vec<Complex<f32>> = (0..len)
        .map(|i| -1. * (i as f32) * f32::consts::PI_2 / (len as f32))
        .map(|phase| Complex::from_polar(&1., &phase))
        .collect();

    let mut data: Vec<Complex<f32>> = repeat(Zero::zero()).take(len).collect();

    b.iter(|| butterfly_2(&mut data, stride, &twiddles, num_ffts));
}
