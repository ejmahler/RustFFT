use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

// Just a very simple test to help debugging
#[test]
fn test_100() {
    // let len = 102;
    for len in 37..250 {
        dbg!(len);
        let mut p = FftPlanner::<f32>::new();
        let planner: Arc<dyn Fft<f32>> = p.plan_fft_forward(len);

        let mut input: Vec<_> = (0..len).map(|i| Complex::new(i as f32, 0.0)).collect();
        // dbg!(planner.get_inplace_scratch_len(), planner.get_outofplace_scratch_len());
        let mut output = input.clone();
        let mut output2 = output.clone();
        let mut scratch = vec![Complex::<f32>::ZERO; planner.get_outofplace_scratch_len() + len];
        // planner.process_outofplace_with_scratch(&mut input, &mut output, &mut scratch);
        planner.process_outofplace_with_scratch_immut(&input, &mut output, &mut scratch);

        planner.process_outofplace_with_scratch(&mut input, &mut output2, &mut scratch);

        assert_eq!(output, output2);
    }
}
