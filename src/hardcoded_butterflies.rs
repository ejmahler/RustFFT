use num::Complex;

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
