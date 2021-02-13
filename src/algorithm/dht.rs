use array_utils::into_real_mut;
use num_complex::Complex;
use num_traits::Zero;

use crate::{FftDirection, array_utils};

fn w(index: usize, len: usize, direction: FftDirection) -> Complex<f32> {
    let constant = -2f32 * std::f32::consts::PI / len as f32;
    let angle = index as f32 * constant;
    let twiddle = Complex::from_polar(1f32, angle);

    match direction {
        FftDirection::Forward => twiddle,
        FftDirection::Inverse => twiddle.conj(),
    }
}

#[allow(unused)]
fn compute_dft_twiddle_inverse(index: usize, len: usize) -> Complex<f32> {
    let constant = 2f32 * std::f32::consts::PI / len as f32;
    let angle = index as f32 * constant;
    let twiddle = Complex::from_polar(1f32, angle);

    twiddle
}

fn cas(index: usize, len: usize) -> f32 {
    let constant = 2f32 * std::f32::consts::PI / len as f32;
    let angle = index as f32 * constant;
    angle.cos() + angle.sin()
}

#[allow(unused)]
fn compute_dft(buffer: &mut [Complex<f32>]) {
    let mut scratch = buffer.to_vec();

    for (k, spec_bin) in buffer.iter_mut().enumerate() {
        let mut sum = Zero::zero();
        for (i, &x) in scratch.iter().enumerate() {
            let twiddle = w(i*k, scratch.len(), FftDirection::Forward);

            sum = sum + twiddle * x;
        }
        *spec_bin = sum;
    }
}

#[allow(unused)]
fn compute_r2c(input: &[f32], output: &mut [Complex<f32>]) {
    assert_eq!(output.len(), input.len() / 2 + 1);

    for (k, spec_bin) in output.iter_mut().enumerate() {
        let mut sum = Zero::zero();
        for (i, &x) in input.iter().enumerate() {
            let twiddle = w(i*k, input.len(), FftDirection::Forward);

            sum = sum + twiddle * x;
        }
        *spec_bin = sum;
    }
}
#[allow(unused)]
fn compute_c2r(input: &mut [Complex<f32>], output: &mut [f32]) {
    assert_eq!(input.len(), output.len() / 2 + 1);
    let len = output.len();

    let mut full_input = vec![Zero::zero(); len];
    (&mut full_input[..input.len()]).copy_from_slice(input);

    let unfilled_slots = full_input.len() - input.len();
    for (i, e) in input.iter().enumerate().skip(1).take(unfilled_slots) {
        full_input[len - i] = e.conj();
    }

    for (k, spec_bin) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, &x) in full_input.iter().enumerate() {
            let twiddle = w(i*k, len, FftDirection::Forward);

            sum = sum + (twiddle * x).re;
        }
        *spec_bin = sum;
    }
}
fn compute_dht(buffer: &mut [f32]) {
    let scratch = buffer.to_vec();

    for (n, output_cell) in buffer.iter_mut().enumerate() {
        let mut output_value = 0.0;
        for (k, input_cell) in scratch.iter().enumerate() {
            let twiddle = cas(n*k, scratch.len());
            
            output_value += *input_cell * twiddle;
        }
        *output_cell = output_value;
    }
}

// Computes a DFT of real-only input by converting the problem to a DHT
#[allow(unused)]
fn compute_r2c_via_dht(input: &mut [f32], output: &mut [Complex<f32>]) {
    assert_eq!(output.len(), input.len() / 2 + 1);

    compute_dht(input);

    output[0] = Complex::from(input[0]);

    for (k, output_cell) in output.iter_mut().enumerate().skip(1) {
        *output_cell = Complex {
            re: (input[input.len() - k] + input[k]) * 0.5,
            im: (input[input.len() - k] - input[k]) * 0.5,
        }
    }
}

// Computes a DFT of real-only input by converting the problem to a DHT
#[allow(unused)]
fn compute_c2r_via_dht(input: &[Complex<f32>], output: &mut [f32]) {
    assert_eq!(input.len(), output.len() / 2 + 1);

    output[0] = input[0].re;

    for (k, input_cell) in input.iter().enumerate().skip(1) {
        output[k]               = (input_cell.re + input_cell.im);
        output[output.len() - k] = (input_cell.re - input_cell.im);
    }

    compute_dht(output);
}

// Computes a DHT by converting the problem to a R2C
#[allow(unused)]
fn compute_dht_via_r2c(buffer: &mut [f32]) {
    let mut scratch = vec![Complex::zero(); buffer.len() / 2 + 1];

    compute_r2c(buffer, &mut scratch);

    buffer[0] = scratch[0].re;

    for (k, scratch_cell) in scratch.iter().enumerate().skip(1) {
        buffer[k]                   = (scratch_cell.re - scratch_cell.im);
        buffer[buffer.len() - k]    = (scratch_cell.re + scratch_cell.im);
    }
}

// Computes a DHT by converting the problem to a R2C
#[allow(unused)]
fn compute_dht_via_c2r(buffer: &mut [f32]) {
    let mut scratch = vec![Complex::zero(); buffer.len() / 2 + 1];

    scratch[0] = Complex::from(buffer[0]);

    for (k, scratch_cell) in scratch.iter_mut().enumerate().skip(1) {
        *scratch_cell = Complex {
            re: (buffer[k] + buffer[buffer.len() - k]) * 0.5,
            im: (buffer[k] - buffer[buffer.len() - k]) * 0.5,
        }
    }
    
    compute_c2r(&mut scratch, buffer);
}

#[allow(unused)]
fn subtract_mod(a: usize, b: usize, m: usize) -> usize {
    let a_wrap = a % m;
    let b_wrap = b % m;
    let result = if a_wrap >= b_wrap {
        a_wrap - b_wrap
    } else {
        m - (b_wrap - a_wrap)
    };
    assert_eq!((b + result) % m, a);
    result
}


// Computes a DHT via the six-step mixed-radix algorithm
#[allow(unused)]
fn compute_dht_mixedradix(buffer: &mut [f32], width: usize, height: usize) {
    let len = buffer.len();
    let mut scratch = vec![0.0; len];
    
    assert_eq!(len, width * height);

    // Step 1: Transpose the width x height array to height x width
    transpose::transpose(buffer, &mut scratch, width, height);

    // Step 2: Compute DHTs of size `height` down the rows of our transposed array
    for chunk in scratch.chunks_exact_mut(height) {
        compute_dht(chunk);
    }

    // Step 3: Apply twiddle factors
    for k in 0..height {
        // we need -k % height, but k is unsigned, so do it without actual negatives
        let k_rev = subtract_mod(0, k, height);
        for i in 0..width {
            // -i % radix, but i is unsigned, so do it without actual negatives
            let i_bot = subtract_mod(0, i, width);
            let top_twiddle = compute_dft_twiddle_inverse(i * k, len);
            let bot_twiddle = compute_dft_twiddle_inverse(i_bot * k, len);
            let rev_twiddle = compute_dft_twiddle_inverse(i * k_rev, len);
            let bot_rev_twiddle = compute_dft_twiddle_inverse(i_bot * k_rev, len);

            // Instead of just multiplying a single input vlaue with a single complex number like we do in the DFT,
            // we need to combine 4 numbers, determined by mirroring the input number across the horizontal and vertical axes of the array
            let top_fwd = scratch[i*height + k];
            let top_rev = scratch[i*height + k_rev];
            let bot_fwd = scratch[i_bot*height + k];
            let bot_rev = scratch[i_bot*height + k_rev];

            // Since we're overwriting data that our mirrored input values will need whenthey compute their own twiddles,
            // we currently can't apply twiddles inplace. An obvious optimization here is to compute all 4 values at once and write them all out at once.
            // That would cut down on the number of flops by 75%, and would let us do this inplace
            buffer[i*height + k] = 0.5 * (
                  top_fwd * top_twiddle.re
                - top_fwd * top_twiddle.im
                + top_rev * top_twiddle.re
                + top_rev * top_twiddle.im
                + bot_fwd * bot_twiddle.re
                + bot_fwd * bot_twiddle.im
                - bot_rev * bot_twiddle.re
                + bot_rev * bot_twiddle.im
            );
        }
    }

    // Step 4: Transpose the height x width array back to width x height
    transpose::transpose(&buffer, &mut scratch, height, width);

    // Step 5: Compute DHTs of size `width` down the rows of the array
    for chunk in scratch.chunks_exact_mut(width) {
        compute_dht(chunk);
    }

    // Step 6: Transpose the width x height array to height x width one lst time
    transpose::transpose(&scratch, buffer, width, height);
}

#[allow(unused)]
fn compute_r2c_mixedradix(input: &mut [f32], output: &mut [Complex<f32>], radix: usize) {
    assert_eq!(output.len(), input.len() / 2 + 1);

    let width = radix;
    assert!(input.len() % width == 0);
    let height = input.len() / width;
    let complex_height = height / 2 + 1;

    let mut complex_scratch = vec![Complex::zero(); complex_height * width];
    let mut complex_scratch2 = vec![Complex::zero(); complex_height * width];

    let temp_transpose = &mut into_real_mut(output)[..input.len()];
    transpose::transpose(&input, temp_transpose, width, height);

    for (i, (in_chunk, out_chunk)) in temp_transpose.chunks_exact_mut(height).zip(complex_scratch.chunks_exact_mut(complex_height)).enumerate() {
        compute_r2c(in_chunk, out_chunk);

        for k in 0..out_chunk.len() {
            out_chunk[k] = out_chunk[k] * w(i*k, input.len(), FftDirection::Forward);
        }
    }

    transpose::transpose(&complex_scratch, &mut complex_scratch2, complex_height, width);
    for chunk in complex_scratch2.chunks_exact_mut(width) {
        compute_dft(chunk);
    }

    let mut indexes = vec![0; output.len()];
    dbg!(complex_scratch2.len());

    // step 6: transpose. Slightly different than the normal transpose, since we have to work around the fact that some of the data is missing
    for i in 0..output.len() {
        let x = i % height;
        let y = i / height;

        let scratch_index = x * width + y;
        if let Some(element) = complex_scratch2.get(scratch_index) {
            output[i] = *element;
            indexes[i] = scratch_index as isize;
        } else {
            let reverse_scratch_index = (height - x) * width + (width - y - 1);
            output[i] = complex_scratch2[reverse_scratch_index].conj();
            indexes[i] = -(reverse_scratch_index as isize);
        }
    }

    for chunk in indexes.chunks(radix) {
        for index in chunk.iter() {
            print!("{:>5}", index);
        }
        println!();
    }
}






#[cfg(test)]
mod unit_tests {
    use num_complex::Complex;
    use num_traits::Zero;

    use super::*;

    use crate::{Fft, test_utils::{compare_real_vectors, random_signal}};

    use crate::{FftDirection, algorithm::Dft, test_utils::{compare_vectors, random_real_signal}};

    #[test]
    fn test_r2c() {
        for len in 1..10 {
            let control = Dft::new(len, FftDirection::Forward);
            let real_input = random_real_signal(len);
            let mut control_input: Vec<Complex<f32>> = real_input.iter().map(Complex::from).collect();

            let mut real_output = vec![Zero::zero(); len/2 + 1];

            control.process(&mut control_input);
            compute_r2c(&real_input, &mut real_output);

            assert!(compare_vectors(&control_input[..len/2 + 1], &real_output));
        }
    }

    #[test]
    fn test_c2r() {
        for len in 1..10 {
            let control = Dft::new(len, FftDirection::Forward);

            let mut real_input = random_signal(len/2 + 1);
            real_input[0].im = 0.0;
            real_input.last_mut().unwrap().im = 0.0;
            let mut complex_input = real_input.clone();

            if len%2 == 0 {
                for i in (1..len/2).rev() {
                    complex_input.push(complex_input[i].conj());
                }
            } else {
                for i in (1..len/2+1).rev() {
                    complex_input.push(complex_input[i].conj());
                }
            }

            control.process(&mut complex_input);

            let mut real_output = vec![0.0; len];
            compute_c2r(&mut real_input, &mut real_output);

            let real_output: Vec<_> = real_output.iter().map(Complex::from).collect();
            if len > 0 {
                assert!(compare_vectors(&complex_input, &real_output));
            }
        }
    }

    #[test]
    fn test_r2c_via_dht() {
        for len in 1..10 {
            let control = Dft::new(len, FftDirection::Forward);
            let mut real_input = random_real_signal(len);
            let mut control_input: Vec<Complex<f32>> = real_input.iter().map(Complex::from).collect();

            let mut real_output = vec![Zero::zero(); len/2 + 1];

            control.process(&mut control_input);
            compute_r2c_via_dht(&mut real_input, &mut real_output);

            assert!(compare_vectors(&control_input[..len/2 + 1], &real_output));
        }
    }

    #[test]
    fn test_c2r_via_dht() {
        for len in 1..10 {
            let control = Dft::new(len, FftDirection::Forward);

            let mut real_input = random_signal(len/2 + 1);
            real_input[0].im = 0.0;
            real_input.last_mut().unwrap().im = 0.0;
            let mut complex_input = real_input.clone();

            if len%2 == 0 {
                for i in (1..len/2).rev() {
                    complex_input.push(complex_input[i].conj());
                }
            } else {
                for i in (1..len/2+1).rev() {
                    complex_input.push(complex_input[i].conj());
                }
            }

            control.process(&mut complex_input);

            let mut real_output = vec![0.0; len];
            compute_c2r_via_dht(&real_input, &mut real_output);

            let real_output: Vec<_> = real_output.iter().map(Complex::from).collect();
            if len > 0 {

                assert!(compare_vectors(&complex_input, &real_output));
            }
        }
    }

    #[test]
    fn test_dht_via_r2c() {
        for len in 1..10 {
            let mut real_buffer = random_real_signal(len);
            let mut control_buffer = real_buffer.clone();

            compute_dht(&mut control_buffer);
            compute_dht_via_r2c(&mut real_buffer);

            assert!(compare_real_vectors(&control_buffer, &real_buffer));
        }
    }

    #[test]
    fn test_dht_via_c2r() {
        for len in 5..10 {
            let mut real_buffer = random_real_signal(len);
            let mut control_buffer = real_buffer.clone();

            compute_dht(&mut control_buffer);
            compute_dht_via_c2r(&mut real_buffer);

            assert!(compare_real_vectors(&control_buffer, &real_buffer));
        }
    }

    #[test]
    fn test_dht_mixedradix() {
        for width in 3..4 {
            for height in 5..6 {
                let len = height * width;
                let mut real_buffer = random_real_signal(len);
                let mut control_buffer = real_buffer.clone();

                compute_dht(&mut control_buffer);
                compute_dht_mixedradix(&mut real_buffer, width, height);

                assert!(compare_real_vectors(&control_buffer, &real_buffer), "width = {}, height = {}", width, height);
            }
        }
    }

    #[test]
    fn test_r2c_radix3() {
        for radix in 3..4 {
            for height in 10..11 {
                let len = height * radix;
                let control = Dft::new(len, FftDirection::Forward);
                let mut real_input = random_real_signal(len);
                let mut control_input: Vec<Complex<f32>> = real_input.iter().map(Complex::from).collect();

                let mut real_output = vec![Zero::zero(); len/2 + 1];

                control.process(&mut control_input);
                compute_r2c_mixedradix(&mut real_input, &mut real_output, radix);

                assert!(compare_vectors(&control_input[..len/2 + 1], &real_output));
            }
        }
    }


    fn compute_dht_splitradix(buffer: &mut [f32]) {
        let scratch = buffer.to_vec();
        let len = buffer.len();
        let half_len = len / 2;
        let quarter_len = len / 4;
        assert_eq!(len % 4, 0);
    
        for (k, output_cell) in buffer.iter_mut().enumerate() {
            let mut output_value = 0.0;
            for n in 0..half_len {
                let twiddle = cas(k*2*n, scratch.len());
                let input_value = scratch[2*n];
                
                output_value += input_value * twiddle;
            }

            for n in 0..quarter_len {
                let twiddle = cas(k*(4*n+1), scratch.len());
                let input_value = scratch[4*n + 1];
                
                output_value += input_value * twiddle;
            }

            for n in 0..quarter_len {
                let twiddle = cas(k*(4*n+3), scratch.len());
                let input_value = scratch[4*n + 3];
                
                output_value += input_value * twiddle;
            }
            *output_cell = output_value;
        }
    }

    #[test]
    fn test_dht_splitradix() {
        for height in 1..6 {
            let len = height * 4;
            let mut real_buffer = random_real_signal(len);
            let mut control_buffer = real_buffer.clone();

            compute_dht(&mut control_buffer);
            compute_dht_splitradix(&mut real_buffer);

            assert!(compare_real_vectors(&control_buffer, &real_buffer), "height = {}", height);
        }
    }
}