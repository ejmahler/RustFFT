//! To test the accuracy of our FFT algorithm, we first test that our
//! naive DFT function is correct by comparing its output against several
//! known signal/spectrum relationships. Then, we generate random signals
//! for a variety of lengths, and test that our FFT algorithm matches our
//! DFT calculation for those signals.

extern crate num;
extern crate rustfft;
extern crate rand;

use num::complex::Complex;
use rand::{StdRng, SeedableRng};
use rand::distributions::{Normal, IndependentSample};
use rustfft::{FFT, dft};

/// The seed for the random number generator used to generate
/// random signals. It's defined here so that we have deterministic
/// tests
const RNG_SEED: [usize; 5] = [1910, 11431, 4984, 14828, 12226];

/// Returns true if the mean difference in the elements of the two vectors
/// is small
fn compare_vectors(vec1: &[Complex<f32>], vec2: &[Complex<f32>]) -> bool {
    assert_eq!(vec1.len(), vec2.len());
    let mut sse = 0f32;
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        sse = sse + (a - b).norm();
    }
    return (sse / vec1.len() as f32) < 0.1f32;
}

/// Returns true if our `dft` function calculates the given spectrum from the
/// given signal
fn test_dft_correct(signal: &[Complex<f32>], spectrum: &[Complex<f32>]) -> bool {
    assert_eq!(signal.len(), spectrum.len());
    let mut test_spectrum = signal.to_vec();
    dft(signal, &mut test_spectrum[..]);
    return compare_vectors(spectrum, &test_spectrum[..]);
}

#[test]
fn test_dft_known_len_2() {
    let signal = [Complex{re: 1f32, im: 0f32},
                  Complex{re:-1f32, im: 0f32}];
    let spectrum = [Complex{re: 0f32, im: 0f32},
                    Complex{re: 2f32, im: 0f32}];
    assert!(test_dft_correct(&signal[..], &spectrum[..]));
}

#[test]
fn test_dft_known_len_3() {
    let signal = [Complex{re: 1f32, im: 1f32},
                  Complex{re: 2f32, im:-3f32},
                      Complex{re:-1f32, im: 4f32}];
    let spectrum = [Complex{re: 2f32, im: 2f32},
                    Complex{re: -5.562177f32, im: -2.098076f32},
                    Complex{re: 6.562178f32, im: 3.09807f32}];
    assert!(test_dft_correct(&signal[..], &spectrum[..]));
}

#[test]
fn test_dft_known_len_4() {
    let signal = [Complex{re: 0f32, im: 1f32},
                  Complex{re: 2.5f32, im:-3f32},
                  Complex{re:-1f32, im: -1f32},
                  Complex{re: 4f32, im: 0f32}];
    let spectrum = [Complex{re: 5.5f32, im: -3f32},
                    Complex{re: -2f32, im: 3.5f32},
                    Complex{re: -7.5f32, im: 3f32},
                    Complex{re: 4f32, im: 0.5f32}];
    assert!(test_dft_correct(&signal[..], &spectrum[..]));
}

#[test]
fn test_dft_known_len_6() {
    let signal = [Complex{re: 1f32, im: 1f32},
                  Complex{re: 2f32, im: 2f32},
                  Complex{re: 3f32, im: 3f32},
                  Complex{re: 4f32, im: 4f32},
                  Complex{re: 5f32, im: 5f32},
                  Complex{re: 6f32, im: 6f32}];
    let spectrum = [Complex{re: 21f32, im: 21f32},
                    Complex{re: -8.16f32, im: 2.16f32},
                    Complex{re: -4.76f32, im: -1.24f32},
                    Complex{re: -3f32, im: -3f32},
                    Complex{re: -1.24f32, im: -4.76f32},
                    Complex{re: 2.16f32, im: -8.16f32}];
    assert!(test_dft_correct(&signal[..], &spectrum[..]));

}

fn ct_matches_dft(signal: Vec<Complex<f32>>, inverse: bool) -> bool {
    let mut spectrum_dft = signal.clone();
    let mut spectrum_ct = signal.clone();

    let mut fft = FFT::new(signal.len(), inverse);
    fft.process(&signal[..], &mut spectrum_ct[..]);

    if inverse {
        let signal_conj: Vec<Complex<f32>> = signal.iter().map(|x| x.conj()).collect();
        dft(&signal_conj[..], &mut spectrum_dft[..]);
        spectrum_dft = spectrum_dft.iter().map(|x| x.conj()).collect();
    } else {
        dft(&signal[..], &mut spectrum_dft[..]);
    };

    return compare_vectors(&spectrum_dft[..], &spectrum_ct[..]);
}

fn random_signal(length: usize) -> Vec<Complex<f32>> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Normal::new(0.0, 10.0);
    let mut rng: StdRng = SeedableRng::from_seed(&RNG_SEED[..]);
    for _ in 0..length {
        sig.push(Complex{re: (normal_dist.ind_sample(&mut rng) as f32),
                         im: (normal_dist.ind_sample(&mut rng) as f32)});
    }
    return sig;
}

/// Tests that our FFT algorithm matches the direct DFT calculation
/// for random signals.
#[test]
fn test_cooley_tukey() {
    for len in 2..100 {
        let signal = random_signal(len);
        assert!(ct_matches_dft(signal, false), "length = {}", len);
    }
}

#[test]
fn test_cooley_tukey_inverse() {
    for len in 1..100 {
        let signal = random_signal(len);
        assert!(ct_matches_dft(signal, true), "length = {}", len);
    }
}
