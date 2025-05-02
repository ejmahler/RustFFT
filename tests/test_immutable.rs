use num_complex::Complex;
use rustfft::{FftNum, FftPlanner};

const TEST_MAX: usize = 1001;

#[test]
fn immutable_f32() {
    for i in 0..TEST_MAX {
        let input = vec![Complex::new(7.0, 8.0); i];

        let mut_output = fft_wrapper_mut::<f32>(&input);
        let immut_output = fft_wrapper_immut::<f32>(&input);

        assert_eq!(mut_output, immut_output, "{}", i);
    }
}

#[test]
fn immutable_f64() {
    for i in 0..TEST_MAX {
        let input = vec![Complex::new(7.0, 8.0); i];

        let mut_output = fft_wrapper_mut::<f64>(&input);
        let immut_output = fft_wrapper_immut::<f64>(&input);

        assert_eq!(mut_output, immut_output, "{}", i);
    }
}

fn fft_wrapper_mut<T: FftNum>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    let cz = Complex::new(T::zero(), T::zero());
    let mut plan = FftPlanner::<T>::new();
    let p = plan.plan_fft_forward(input.len());

    let mut scratch = vec![cz; p.get_inplace_scratch_len()];
    let mut output = input.to_vec();

    p.process_with_scratch(&mut output, &mut scratch);
    output
}

fn fft_wrapper_immut<T: FftNum>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    let cz = Complex::new(T::zero(), T::zero());
    let mut plan = FftPlanner::<T>::new();
    let p = plan.plan_fft_forward(input.len());

    let mut scratch = vec![cz; p.get_immutable_scratch_len()];
    let mut output = vec![cz; input.len()];

    p.process_immutable_with_scratch(input, &mut output, &mut scratch);
    output
}
