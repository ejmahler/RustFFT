pub mod wasm_simd_planner {
    use crate::{Fft, FftDirection, FftNum};
    use std::sync::Arc;

    /// The WASM FFT planner creates new FFT algorithm instances using a mix of scalar and WASM SIMD accelerated algorithms.
    /// It is supported when using fairly recent browser versions as outlined in [the WebAssembly roadmap](https://webassembly.org/roadmap/).
    ///
    /// RustFFT has several FFT algorithms available. For a given FFT size, `FftPlannerWasmSimd` decides which of the
    /// available FFT algorithms to use and then initializes them.
    ///
    /// ~~~
    /// // Perform a forward Fft of size 1234
    /// use std::sync::Arc;
    /// use rustfft::{FftPlannerWasmSimd, num_complex::Complex};
    ///
    /// if let Ok(mut planner) = FftPlannerWasmSimd::new() {
    ///   let fft = planner.plan_fft_forward(1234);
    ///
    ///   let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 1234];
    ///   fft.process(&mut buffer);
    ///
    ///   // The FFT instance returned by the planner has the type `Arc<dyn Fft<T>>`,
    ///   // where T is the numeric type, ie f32 or f64, so it's cheap to clone
    ///   let fft_clone = Arc::clone(&fft);
    /// }
    /// ~~~
    ///
    /// If you plan on creating multiple FFT instances, it is recommended to reuse the same planner for all of them. This
    /// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
    /// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
    /// by a different planner)
    ///
    /// Each FFT instance owns [`Arc`s](std::sync::Arc) to its internal data, rather than borrowing it from the planner, so it's perfectly
    /// safe to drop the planner after creating Fft instances.
    pub struct FftPlannerWasmSimd<T: FftNum> {
        _phantom: std::marker::PhantomData<T>,
    }
    impl<T: FftNum> FftPlannerWasmSimd<T> {
        /// Creates a new `FftPlannerNeon` instance.
        ///
        /// Returns `Ok(planner_instance)` if this machine has the required instruction sets.
        /// Returns `Err(())` if some instruction sets are missing.
        pub fn new() -> Result<Self, ()> {
            Err(())
        }
        /// Returns a `Fft` instance which uses Neon instructions to compute FFTs of size `len`.
        ///
        /// If the provided `direction` is `FftDirection::Forward`, the returned instance will compute forward FFTs. If it's `FftDirection::Inverse`, it will compute inverse FFTs.
        ///
        /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
        pub fn plan_fft(&mut self, _len: usize, _direction: FftDirection) -> Arc<dyn Fft<T>> {
            unreachable!()
        }
        /// Returns a `Fft` instance which uses Neon instructions to compute forward FFTs of size `len`.
        ///
        /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
        pub fn plan_fft_forward(&mut self, _len: usize) -> Arc<dyn Fft<T>> {
            unreachable!()
        }
        /// Returns a `Fft` instance which uses Neon instructions to compute inverse FFTs of size `len.
        ///
        /// If this is called multiple times, the planner will attempt to re-use internal data between calls, reducing memory usage and FFT initialization time.
        pub fn plan_fft_inverse(&mut self, _len: usize) -> Arc<dyn Fft<T>> {
            unreachable!()
        }
    }
}
