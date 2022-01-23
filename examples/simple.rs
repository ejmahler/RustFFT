//! A simple documented example that takes an FFT of a sine function and
//! prints the result.
//!
//! We create a sine wave with a `FREQUENCY` of 1222 Hz and a sample rate
//! of 11025 Hz.  `num_samples` are generated and saved to the buffer.  An
//! FFT is taken of the buffer.  We need to normalise the values after the
//! FFT such that `AMPLITUDE` of output matches that of the input.
//!
//! To use run the following:
//!   `cargo run --release --example simple`

use rustfft::{num_complex::Complex32, FftPlanner};

// The frequency of sine function.
const FREQUENCY: f32 = 1222.0; // Hz

// The amplitude of the sine wave.
const AMPLITUDE: f32 = 5.0;

fn create_wave(num_samples: usize, sample_rate: f32) -> Vec<Complex32> {
    // The generator function, this will create a sine wave over the
    // number of samples.
    let generator = |x| {
        let mut angle = x as f32;
        angle *= FREQUENCY / sample_rate;
        angle *= 2.0 * std::f32::consts::PI;
        Complex32::new(AMPLITUDE * angle.sin(), 0.0)
    };

    (0..num_samples).into_iter().map(generator).collect()
}

fn main() {
    // The number of frequency bins.
    let bins = 2048;

    // The number of samples to calculate in the sine function.
    let num_samples = bins * 100;
    let sample_rate = 11025; // Hz

    // The buffer is used to store the wave data and the FFT result
    let mut buffer = create_wave(num_samples, sample_rate as f32);

    // Do all the planning
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(bins);

    // Do the FFT, this will take the buffer, and write back into to.
    fft.process(&mut buffer);

    // The fft real results will only be in the first half of the buffer.
    // This is the Nyquist limit, which is half the sample frequency.
    for i in 0..bins / 2 {
        // The range goes from 0 Hz to `sample_freq/2` Hz.
        let freq = (i as f32 / bins as f32) * sample_rate as f32;

        // Need to convert to real numbers and normalise the buffer values.
        let norm_value = 2.0 * buffer[i].norm() / bins as f32;

        println!("{:.2}\t{:.6}", freq, norm_value);
    }
}
