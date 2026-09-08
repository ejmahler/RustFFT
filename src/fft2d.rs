use crate::{Complex, Fft, FftNum};
use std::sync::Arc;

use num_integer::lcm;
use num_traits::Zero;

/// Wrapper struct for 2D FFTs.
pub struct FFT2D<T: FftNum> {
    fft0: Arc<dyn Fft<T>>,
    fft1: Arc<dyn Fft<T>>,
}

impl<T: FftNum> FFT2D<T> {
    /// Construct a new 2D FFT from a 1-dimensional FFT.
    pub fn new(fft: Arc<dyn Fft<T>>) -> Self {
        let fft0 = fft.clone();
        let fft1 = fft;
        Self { fft0, fft1 }
    }
    /// The length along each dimension that this FFT can process.
    pub fn len(&self) -> [usize; 2] {
        [self.fft0.len(), self.fft1.len()]
    }

    pub fn get_inplace_scratch_len(&self) -> usize {
        let [w, h] = self.len();
        self.fft0
            .get_inplace_scratch_len()
            .max(self.fft1.get_inplace_scratch_len())
            .max(w)
            .max(h)
    }
    pub fn get_outofplace_scratch_len(&self) -> usize {
        let [w, h] = self.len();
        self.fft0
            .get_outofplace_scratch_len()
            .max(self.fft1.get_outofplace_scratch_len())
            .max(w)
            .max(h)
    }
    pub fn get_immutable_scratch_len(&self) -> usize {
        let [w, h] = self.len();
        self.fft0
            .get_immutable_scratch_len()
            .max(self.fft1.get_immutable_scratch_len())
            .max(w)
            .max(h)
    }

    pub fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let [w, h] = self.len();
        assert!(scratch.len() >= w.max(h));
        self.fft0.process_with_scratch(buffer, scratch);
        transpose::transpose_inplace(buffer, scratch, w, h);
        self.fft1.process_with_scratch(buffer, scratch);
        transpose::transpose_inplace(buffer, scratch, h, w);
    }

    pub fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let [w, h] = self.len();
        assert!(scratch.len() >= w.max(h));
        self.fft0
            .process_outofplace_with_scratch(input, output, scratch);
        transpose::transpose_inplace(output, scratch, w, h);
        self.fft1
            .process_outofplace_with_scratch(output, input, scratch);
        transpose::transpose_inplace(input, scratch, h, w);
        output.copy_from_slice(input);
    }

    pub fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let [w, h] = self.len();
        assert!(scratch.len() >= w.max(h));
        self.fft0
            .process_immutable_with_scratch(input, output, scratch);
        transpose::transpose_inplace(output, scratch, w, h);
        self.fft1.process_with_scratch(output, scratch);
        transpose::transpose_inplace(output, scratch, h, w);
    }

    pub fn process(&self, buffer: &mut [Complex<T>]) {
        let [w, h] = self.len();
        // lcm is necessary so that the buffer fits for FFT along both dimensions
        let wh = lcm(w, h);
        assert_eq!(buffer.len() % wh, 0);
        let mut scratch = vec![Complex::zero(); wh];
        self.process_with_scratch(buffer, &mut scratch);
    }
}

#[cfg(test)]
pub const fn re(re: f32) -> Complex<f32> {
    Complex { re, im: 0. }
}

#[cfg(test)]
const EXAMPLE_ARR: [Complex<f32>; 16] = [
    re(1.),
    re(2.),
    re(3.),
    re(33.),
    re(5.),
    re(6.),
    re(7.),
    re(8.),
    re(9.),
    re(10.),
    re(11.),
    re(12.),
    re(13.),
    re(14.),
    re(15.),
    re(16.),
];

#[test]
fn test_basic_2d() {
    let mut planner = crate::FftPlannerScalar::<f32>::new();
    let fft = planner.plan_fft_forward(4);

    let fft2d = FFT2D::new(fft);
    let mut tmp2 = EXAMPLE_ARR.clone();
    fft2d.process(&mut tmp2);

    assert_eq!(tmp2[0], Complex::new(165., 0.));
    assert_eq!(tmp2[1], Complex::new(-8., 37.));
    assert_eq!(tmp2[2], Complex::new(-37., 0.));
    assert_eq!(tmp2[4], Complex::new(-3., 32.));
    assert_eq!(tmp2[8], Complex::new(-3., 0.));
    assert_eq!(tmp2[15], Complex::new(0., -29.));
}

#[test]
fn test_different_process_2d() {
    let mut planner = crate::FftPlannerScalar::<f32>::new();
    let fft = planner.plan_fft_forward(4);

    let fft2d = FFT2D::new(fft);
    let mut gt = EXAMPLE_ARR.clone();
    fft2d.process(&mut gt);

    // process with scratch
    let mut p_w_scratch = EXAMPLE_ARR.clone();
    assert_eq!(fft2d.get_inplace_scratch_len(), 4);
    let mut scratch = [Complex::new(0., 0.); 4];
    fft2d.process_with_scratch(&mut p_w_scratch, &mut scratch);
    assert_eq!(p_w_scratch, gt);

    // process outofplace with scratch
    let mut p_oop_w_scratch = EXAMPLE_ARR.clone();
    assert_eq!(fft2d.get_outofplace_scratch_len(), 4);
    let mut scratch = [Complex::new(0., 0.); 4];
    let mut out_oop = [Complex::new(0., 0.); 16];
    fft2d.process_outofplace_with_scratch(&mut p_oop_w_scratch, &mut out_oop, &mut scratch);
    assert_eq!(out_oop, gt);

    // process immutable with scratch
    assert_eq!(fft2d.get_immutable_scratch_len(), 4);
    let mut scratch = [Complex::new(0., 0.); 4];
    let mut out_imm = [Complex::new(0., 0.); 16];
    fft2d.process_immutable_with_scratch(&EXAMPLE_ARR, &mut out_imm, &mut scratch);
    assert_eq!(out_imm, gt);
}
