use num::{FromPrimitive, Signed};

pub trait FFTnum: Copy + FromPrimitive + Signed + 'static {}

impl FFTnum for f32 {}
impl FFTnum for f64 {}
