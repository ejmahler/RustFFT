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
    let mut row = input.as_ptr();
    let input_stride = input_stride as isize;
    for _ in range(0, num_rows) {
        unsafe {
            let twiddle = twiddles.get_unchecked(twiddle_idx);
            *output.get_unchecked_mut(out_idx_1) = 
                *row.offset(0) + *row.offset(input_stride) * twiddle;
            *output.get_unchecked_mut(out_idx_2) = 
                *row.offset(0) - *row.offset(input_stride) * twiddle;
        }
        out_idx_1 += output_stride;
        out_idx_2 += output_stride;
        twiddle_idx += input_stride as usize;
        row = unsafe { row.offset(2 * input_stride) };
    }
}
