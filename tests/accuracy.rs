extern crate num;
extern crate rustfft;

use num::complex::Complex;
use std::rand;
use std::rand::distributions::{Normal, IndependentSample};
use rustfft::{FFT, dft};

fn compare_vectors(vec1: &[Complex<f32>], vec2: &[Complex<f32>]) -> bool
{
    let mut sse = 0f32;
    for (&a, &b) in vec1.iter().zip(vec2.iter())
    {
        sse = sse + (a - b).norm();
    }
    return (sse / vec1.len() as f32) < 0.1f32;
}

fn test_dft_correct(signal: &[Complex<f32>], spectrum: &[Complex<f32>]) -> bool
{
    assert_eq!(signal.len(), spectrum.len());
    let mut fft = FFT::new(signal.len());
    let mut test_spectrum = signal.to_vec();
    fft.process(signal, test_spectrum.as_mut_slice());
    println!("our spectrum: {:?}", test_spectrum);
    println!("yes spectrum: {:?}", spectrum);
    return compare_vectors(spectrum, test_spectrum.as_slice());
}

/*
    Tests against some known signal <-> spectrum relationships.
*/
#[test]
fn dft_test()
{

    let signal = vec![Complex{re: 1f32, im: 0f32},
                      Complex{re:-1f32, im: 0f32}];
    let spectrum = vec![Complex{re: 0f32, im: 0f32}, 
                        Complex{re: 2f32, im: 0f32}];
    assert!(test_dft_correct(signal.as_slice(), spectrum.as_slice()));

    let signal = vec![Complex{re: 1f32, im: 1f32},
                      Complex{re: 2f32, im:-3f32},
                      Complex{re:-1f32, im: 4f32}];
    let spectrum = vec![Complex{re: 2f32, im: 2f32}, 
                        Complex{re: -5.562177f32, im: -2.098076f32},
                        Complex{re: 6.562178f32, im: 3.09807f32}];
    assert!(test_dft_correct(signal.as_slice(), spectrum.as_slice()));

    let signal = vec![Complex{re: 1f32, im: 1f32},
                      Complex{re: 2f32, im: 2f32},
                      Complex{re: 3f32, im: 3f32},
                      Complex{re: 4f32, im: 4f32},
                      Complex{re: 5f32, im: 5f32},
                      Complex{re: 6f32, im: 6f32}];
    let spectrum = vec![Complex{re: 21f32, im: 21f32}, 
                        Complex{re: -8.16f32, im: 2.16f32},
                        Complex{re: -4.76f32, im: -1.24f32},
                        Complex{re: -3f32, im: -3f32},
                        Complex{re: -1.24f32, im: -4.76f32},
                        Complex{re: 2.16f32, im: -8.16f32}];
    assert!(test_dft_correct(signal.as_slice(), spectrum.as_slice()));

    let signal = vec![Complex{re: 0f32, im: 1f32},
                      Complex{re: 2.5f32, im:-3f32},
                      Complex{re:-1f32, im: -1f32},
                      Complex{re: 4f32, im: 0f32}];
    let spectrum = vec![Complex{re: 5.5f32, im: -3f32}, 
                        Complex{re: -2f32, im: 3.5f32},
                        Complex{re: -7.5f32, im: 3f32},
                        Complex{re: 4f32, im: 0.5f32}];
    assert!(test_dft_correct(signal.as_slice(), spectrum.as_slice()));

}

fn ct_matches_dft(signal: &[Complex<f32>]) -> bool
{
    let mut spectrum_dft = signal.to_vec();
    let mut spectrum_ct = signal.to_vec();
    dft(signal.iter(), spectrum_dft.iter_mut());
    test_dft_correct(signal, spectrum_dft.as_slice())
}

fn random_signal(length: usize) -> Vec<Complex<f32>>
{
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Normal::new(0.0, 10.0);
    let mut rng = rand::thread_rng();
    for _ in range(0, length)
    {
        sig.push(Complex{re: (normal_dist.ind_sample(&mut rng) as f32),
                         im: (normal_dist.ind_sample(&mut rng) as f32)});
    }
    return sig;
}

#[test]
fn test_cooley_tukey()
{
    for len in range(2us, 100)
    {
        let signal = random_signal(len);
        assert!(ct_matches_dft(signal.as_slice()), "length = {}", len);
    }
}
