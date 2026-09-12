use criterion::Criterion;
use std::time::Duration;

/// Keep each benchmark close to libtest's fast feedback time while still
/// giving Criterion distinct warm-up and measurement phases.
pub fn fast() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(100))
        .sample_size(10)
        .without_plots()
}

#[allow(unused_macros)]
macro_rules! register_benchmarks {
    ($criterion:expr, $($benchmark:ident),* $(,)?) => {
        $(
            $criterion.bench_function(stringify!($benchmark), $benchmark);
        )*
    };
}

#[allow(unused_imports)]
pub(crate) use register_benchmarks;
