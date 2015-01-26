use num::Complex;

pub fn butterfly_2(input: &[Complex<f32>],
               input_stride: usize,
               output: &mut [Complex<f32>],
               output_stride: usize,
               twiddles: &[Complex<f32>],
               num_rows: usize) {
    let mut out_idx_1 = 0us;
    let mut out_idx_2 = num_rows * output_stride;
    let mut twiddle_idx = 0us;
    // I think the chunks method might be slower than necessary...
    for row in input.chunks(2 * input_stride) {
        unsafe {
            let twiddle = twiddles.get_unchecked(twiddle_idx);
            *output.get_unchecked_mut(out_idx_1) = 
                row.get_unchecked(0) + row.get_unchecked(input_stride) * twiddle;
            *output.get_unchecked_mut(out_idx_2) = 
                row.get_unchecked(0) - row.get_unchecked(input_stride) * twiddle;
        }
        out_idx_1 += output_stride;
        out_idx_2 += output_stride;
        twiddle_idx += input_stride;
    }
}
