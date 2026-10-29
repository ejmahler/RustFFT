use std::arch::x86_64::*;

use crate::{
    FftDirection,
    avx::{
        avx32_utils::{Rotate90F32, fmadd_ps, nmadd_ps},
        avx64_utils::{Rotate90F64, fmadd_pd, nmadd_pd},
    },
    simd::parallel_fft::ParallelFft,
    twiddles,
};

macro_rules! boilerplate_avx_parallel_butterfly {
    ($struct_name:ident, $vector_type:ident, $len:expr) => {
        impl ParallelFft<$vector_type, $len> for $struct_name {
            #[inline(always)]
            fn parallel_fft(&self, data: [$vector_type; $len]) -> [$vector_type; $len] {
                unsafe { self.process(data) }
            }
        }
    };
}

pub struct AvxF32ParallelButterfly2 {}
impl AvxF32ParallelButterfly2 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(_direction: FftDirection) -> Self {
        Self {}
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256; 2]) -> [__m256; 2] {
        [
            _mm256_add_ps(data[0], data[1]),
            _mm256_sub_ps(data[0], data[1]),
        ]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF32ParallelButterfly2, __m256, 2);

pub struct AvxF64ParallelButterfly2 {}
impl AvxF64ParallelButterfly2 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(_direction: FftDirection) -> Self {
        Self {}
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256d; 2]) -> [__m256d; 2] {
        [
            _mm256_add_pd(data[0], data[1]),
            _mm256_sub_pd(data[0], data[1]),
        ]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF64ParallelButterfly2, __m256d, 2);

pub struct AvxF32ParallelButterfly3 {
    twiddles_re: [__m256; 1],
    twiddles_im: [__m256; 1],
}
impl AvxF32ParallelButterfly3 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        let twiddles = std::array::from_fn(|i| twiddles::compute_twiddle(i + 1, 3, direction));
        Self {
            twiddles_re: twiddles.map(|t| _mm256_set1_ps(t.re)),
            twiddles_im: twiddles.map(|t| _mm256_set1_ps(t.im)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256; 3]) -> [__m256; 3] {
        let x12p = _mm256_add_ps(data[1], data[2]);
        let x12n = _mm256_sub_ps(data[1], data[2]);
        let sum = _mm256_add_ps(data[0], x12p);

        let temp_a = fmadd_ps(data[0], self.twiddles_re[0], x12p);

        let rotation = Rotate90F32::new(FftDirection::Inverse);
        let n_rot = rotation.rotate(x12n);

        let x1 = fmadd_ps(temp_a, self.twiddles_im[0], n_rot);
        let x2 = nmadd_ps(temp_a, self.twiddles_im[0], n_rot);

        [sum, x1, x2]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF32ParallelButterfly3, __m256, 3);

pub struct AvxF64ParallelButterfly3 {
    twiddles_re: [__m256d; 1],
    twiddles_im: [__m256d; 1],
}
impl AvxF64ParallelButterfly3 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        let twiddles = std::array::from_fn(|i| twiddles::compute_twiddle(i + 1, 3, direction));
        Self {
            twiddles_re: twiddles.map(|t| _mm256_set1_pd(t.re)),
            twiddles_im: twiddles.map(|t| _mm256_set1_pd(t.im)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256d; 3]) -> [__m256d; 3] {
        let x12p = _mm256_add_pd(data[1], data[2]);
        let x12n = _mm256_sub_pd(data[1], data[2]);
        let sum = _mm256_add_pd(data[0], x12p);

        let temp_a = fmadd_pd(data[0], self.twiddles_re[0], x12p);

        let rotation = Rotate90F64::new(FftDirection::Inverse);

        let n_rot = rotation.rotate(x12n);

        let x1 = fmadd_pd(temp_a, self.twiddles_im[0], n_rot);
        let x2 = nmadd_pd(temp_a, self.twiddles_im[0], n_rot);
        [sum, x1, x2]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF64ParallelButterfly3, __m256d, 3);

pub struct AvxF32ParallelButterfly4 {
    rotate: Rotate90F32,
    bf2: AvxF32ParallelButterfly2,
}
impl AvxF32ParallelButterfly4 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            rotate: Rotate90F32::new(direction),
            bf2: AvxF32ParallelButterfly2::new(direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256; 4]) -> [__m256; 4] {
        // Algorithm: 2x2 mixed radix

        // Perform the first set of size-2 FFTs.
        let [mid0, mid2] = self.bf2.process([data[0], data[2]]);
        let [mid1, mid3] = self.bf2.process([data[1], data[3]]);

        // Apply twiddle factors (in this case just a rotation)
        let mid3 = self.rotate.rotate(mid3);

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = self.bf2.process([mid0, mid1]);
        let [output2, output3] = self.bf2.process([mid2, mid3]);

        // Swap outputs 1 and 2 in the output to do a square transpose
        [output0, output2, output1, output3]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF32ParallelButterfly4, __m256, 4);

pub struct AvxF64ParallelButterfly4 {
    rotate: Rotate90F64,
    bf2: AvxF64ParallelButterfly2,
}
impl AvxF64ParallelButterfly4 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            rotate: Rotate90F64::new(direction),
            bf2: AvxF64ParallelButterfly2::new(direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256d; 4]) -> [__m256d; 4] {
        // Algorithm: 2x2 mixed radix

        // Perform the first set of size-2 FFTs.
        let [mid0, mid2] = self.bf2.process([data[0], data[2]]);
        let [mid1, mid3] = self.bf2.process([data[1], data[3]]);

        // Apply twiddle factors (in this case just a rotation)
        let mid3 = self.rotate.rotate(mid3);

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = self.bf2.process([mid0, mid1]);
        let [output2, output3] = self.bf2.process([mid2, mid3]);

        // Swap outputs 1 and 2 in the output to do a square transpose
        [output0, output2, output1, output3]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF64ParallelButterfly4, __m256d, 4);

pub struct AvxF32ParallelButterfly5 {
    bf2: AvxF32ParallelButterfly2,
    twiddles_re: [__m256; 2],
    twiddles_im: [__m256; 2],
}
impl AvxF32ParallelButterfly5 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        let twiddles = std::array::from_fn(|i| twiddles::compute_twiddle(i + 1, 5, direction));
        Self {
            bf2: AvxF32ParallelButterfly2::new(direction),
            twiddles_re: twiddles.map(|t| _mm256_set1_ps(t.re)),
            twiddles_im: twiddles.map(|t| _mm256_set1_ps(t.im)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256; 5]) -> [__m256; 5] {
        let rotate = Rotate90F32::new(FftDirection::Inverse);

        let y00 = data[0];
        let [x1p4, x1m4] = self.bf2.process([data[1], data[4]]);
        let x1m4 = rotate.rotate(x1m4);
        let y00 = _mm256_add_ps(y00, x1p4);
        let [x2p3, x2m3] = self.bf2.process([data[2], data[3]]);
        let x2m3 = rotate.rotate(x2m3);
        let y00 = _mm256_add_ps(y00, x2p3);

        let m0104a = fmadd_ps(data[0], self.twiddles_re[0], x1p4);
        let m0104a = fmadd_ps(m0104a, self.twiddles_re[1], x2p3);
        let m0104b = _mm256_mul_ps(self.twiddles_im[0], x1m4);
        let m0104b = fmadd_ps(m0104b, self.twiddles_im[1], x2m3);
        let [y01, y04] = self.bf2.process([m0104a, m0104b]);

        let m0203a = fmadd_ps(data[0], self.twiddles_re[1], x1p4);
        let m0203a = fmadd_ps(m0203a, self.twiddles_re[0], x2p3);
        let m0203b = _mm256_mul_ps(self.twiddles_im[1], x1m4);
        let m0203b = nmadd_ps(m0203b, self.twiddles_im[0], x2m3);
        let [y02, y03] = self.bf2.process([m0203a, m0203b]);

        [y00, y01, y02, y03, y04]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF32ParallelButterfly5, __m256, 5);

pub struct AvxF64ParallelButterfly5 {
    bf2: AvxF64ParallelButterfly2,
    twiddles_re: [__m256d; 2],
    twiddles_im: [__m256d; 2],
}
impl AvxF64ParallelButterfly5 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        let twiddles = std::array::from_fn(|i| twiddles::compute_twiddle(i + 1, 5, direction));
        Self {
            bf2: AvxF64ParallelButterfly2::new(direction),
            twiddles_re: twiddles.map(|t| _mm256_set1_pd(t.re)),
            twiddles_im: twiddles.map(|t| _mm256_set1_pd(t.im)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256d; 5]) -> [__m256d; 5] {
        let rotate = Rotate90F64::new(FftDirection::Inverse);

        let y00 = data[0];
        let [x1p4, x1m4] = self.bf2.process([data[1], data[4]]);
        let x1m4 = rotate.rotate(x1m4);
        let y00 = _mm256_add_pd(y00, x1p4);
        let [x2p3, x2m3] = self.bf2.process([data[2], data[3]]);
        let x2m3 = rotate.rotate(x2m3);
        let y00 = _mm256_add_pd(y00, x2p3);

        let m0104a = fmadd_pd(data[0], self.twiddles_re[0], x1p4);
        let m0104a = fmadd_pd(m0104a, self.twiddles_re[1], x2p3);
        let m0104b = _mm256_mul_pd(self.twiddles_im[0], x1m4);
        let m0104b = fmadd_pd(m0104b, self.twiddles_im[1], x2m3);
        let [y01, y04] = self.bf2.process([m0104a, m0104b]);

        let m0203a = fmadd_pd(data[0], self.twiddles_re[1], x1p4);
        let m0203a = fmadd_pd(m0203a, self.twiddles_re[0], x2p3);
        let m0203b = _mm256_mul_pd(self.twiddles_im[1], x1m4);
        let m0203b = nmadd_pd(m0203b, self.twiddles_im[0], x2m3);
        let [y02, y03] = self.bf2.process([m0203a, m0203b]);

        [y00, y01, y02, y03, y04]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF64ParallelButterfly5, __m256d, 5);

pub struct AvxF32ParallelButterfly6 {
    bf2: AvxF32ParallelButterfly2,
    bf3: AvxF32ParallelButterfly3,
}
impl AvxF32ParallelButterfly6 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            bf2: AvxF32ParallelButterfly2::new(direction),
            bf3: AvxF32ParallelButterfly3::new(direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256; 6]) -> [__m256; 6] {
        // Algorithm: 3x2 good-thomas

        // Size-3 FFTs down the columns of our reordered array
        let mid0 = self.bf3.process([data[0], data[2], data[4]]);
        let mid1 = self.bf3.process([data[3], data[5], data[1]]);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = self.bf2.process([mid0[0], mid1[0]]);
        let [output2, output3] = self.bf2.process([mid0[1], mid1[1]]);
        let [output4, output5] = self.bf2.process([mid0[2], mid1[2]]);

        // Reorder into output
        [output0, output3, output4, output1, output2, output5]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF32ParallelButterfly6, __m256, 6);

pub struct AvxF64ParallelButterfly6 {
    bf2: AvxF64ParallelButterfly2,
    bf3: AvxF64ParallelButterfly3,
}
impl AvxF64ParallelButterfly6 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            bf2: AvxF64ParallelButterfly2::new(direction),
            bf3: AvxF64ParallelButterfly3::new(direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256d; 6]) -> [__m256d; 6] {
        // Algorithm: 3x2 good-thomas

        // Size-3 FFTs down the columns of our reordered array
        let mid0 = self.bf3.process([data[0], data[2], data[4]]);
        let mid1 = self.bf3.process([data[3], data[5], data[1]]);

        // We normally would put twiddle factors right here, but since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-2 FFTs down the columns
        let [output0, output1] = self.bf2.process([mid0[0], mid1[0]]);
        let [output2, output3] = self.bf2.process([mid0[1], mid1[1]]);
        let [output4, output5] = self.bf2.process([mid0[2], mid1[2]]);

        // Reorder into output
        [output0, output3, output4, output1, output2, output5]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF64ParallelButterfly6, __m256d, 6);

pub struct AvxF32ParallelButterfly7 {
    bf2: AvxF32ParallelButterfly2,
    twiddles_re: [__m256; 3],
    twiddles_im: [__m256; 3],
}
impl AvxF32ParallelButterfly7 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        let twiddles = std::array::from_fn(|i| twiddles::compute_twiddle(i + 1, 7, direction));
        Self {
            bf2: AvxF32ParallelButterfly2::new(direction),
            twiddles_re: twiddles.map(|t| _mm256_set1_ps(t.re)),
            twiddles_im: twiddles.map(|t| _mm256_set1_ps(t.im)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256; 7]) -> [__m256; 7] {
        let rotate = Rotate90F32::new(FftDirection::Inverse);

        let y00 = data[0];
        let [x1p6, x1m6] = self.bf2.process([data[1], data[6]]);
        let x1m6 = rotate.rotate(x1m6);
        let y00 = _mm256_add_ps(y00, x1p6);
        let [x2p5, x2m5] = self.bf2.process([data[2], data[5]]);
        let x2m5 = rotate.rotate(x2m5);
        let y00 = _mm256_add_ps(y00, x2p5);
        let [x3p4, x3m4] = self.bf2.process([data[3], data[4]]);
        let x3m4 = rotate.rotate(x3m4);
        let y00 = _mm256_add_ps(y00, x3p4);

        let m0106a = fmadd_ps(data[0], self.twiddles_re[0], x1p6);
        let m0106a = fmadd_ps(m0106a, self.twiddles_re[1], x2p5);
        let m0106a = fmadd_ps(m0106a, self.twiddles_re[2], x3p4);
        let m0106b = _mm256_mul_ps(self.twiddles_im[0], x1m6);
        let m0106b = fmadd_ps(m0106b, self.twiddles_im[1], x2m5);
        let m0106b = fmadd_ps(m0106b, self.twiddles_im[2], x3m4);
        let [y01, y06] = self.bf2.process([m0106a, m0106b]);

        let m0205a = fmadd_ps(data[0], self.twiddles_re[1], x1p6);
        let m0205a = fmadd_ps(m0205a, self.twiddles_re[2], x2p5);
        let m0205a = fmadd_ps(m0205a, self.twiddles_re[0], x3p4);
        let m0205b = _mm256_mul_ps(self.twiddles_im[1], x1m6);
        let m0205b = nmadd_ps(m0205b, self.twiddles_im[2], x2m5);
        let m0205b = nmadd_ps(m0205b, self.twiddles_im[0], x3m4);
        let [y02, y05] = self.bf2.process([m0205a, m0205b]);

        let m0304a = fmadd_ps(data[0], self.twiddles_re[2], x1p6);
        let m0304a = fmadd_ps(m0304a, self.twiddles_re[0], x2p5);
        let m0304a = fmadd_ps(m0304a, self.twiddles_re[1], x3p4);
        let m0304b = _mm256_mul_ps(self.twiddles_im[2], x1m6);
        let m0304b = nmadd_ps(m0304b, self.twiddles_im[0], x2m5);
        let m0304b = fmadd_ps(m0304b, self.twiddles_im[1], x3m4);
        let [y03, y04] = self.bf2.process([m0304a, m0304b]);

        [y00, y01, y02, y03, y04, y05, y06]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF32ParallelButterfly7, __m256, 7);

pub struct AvxF64ParallelButterfly7 {
    bf2: AvxF64ParallelButterfly2,
    twiddles_re: [__m256d; 3],
    twiddles_im: [__m256d; 3],
}
impl AvxF64ParallelButterfly7 {
    #[inline]
    #[target_feature(enable = "avx,fma")]
    pub fn new(direction: FftDirection) -> Self {
        let twiddles = std::array::from_fn(|i| twiddles::compute_twiddle(i + 1, 7, direction));
        Self {
            bf2: AvxF64ParallelButterfly2::new(direction),
            twiddles_re: twiddles.map(|t| _mm256_set1_pd(t.re)),
            twiddles_im: twiddles.map(|t| _mm256_set1_pd(t.im)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx,fma")]
    fn process(&self, data: [__m256d; 7]) -> [__m256d; 7] {
        let rotate = Rotate90F64::new(FftDirection::Inverse);

        let y00 = data[0];
        let [x1p6, x1m6] = self.bf2.process([data[1], data[6]]);
        let x1m6 = rotate.rotate(x1m6);
        let y00 = _mm256_add_pd(y00, x1p6);
        let [x2p5, x2m5] = self.bf2.process([data[2], data[5]]);
        let x2m5 = rotate.rotate(x2m5);
        let y00 = _mm256_add_pd(y00, x2p5);
        let [x3p4, x3m4] = self.bf2.process([data[3], data[4]]);
        let x3m4 = rotate.rotate(x3m4);
        let y00 = _mm256_add_pd(y00, x3p4);

        let m0106a = fmadd_pd(data[0], self.twiddles_re[0], x1p6);
        let m0106a = fmadd_pd(m0106a, self.twiddles_re[1], x2p5);
        let m0106a = fmadd_pd(m0106a, self.twiddles_re[2], x3p4);
        let m0106b = _mm256_mul_pd(self.twiddles_im[0], x1m6);
        let m0106b = fmadd_pd(m0106b, self.twiddles_im[1], x2m5);
        let m0106b = fmadd_pd(m0106b, self.twiddles_im[2], x3m4);
        let [y01, y06] = self.bf2.process([m0106a, m0106b]);

        let m0205a = fmadd_pd(data[0], self.twiddles_re[1], x1p6);
        let m0205a = fmadd_pd(m0205a, self.twiddles_re[2], x2p5);
        let m0205a = fmadd_pd(m0205a, self.twiddles_re[0], x3p4);
        let m0205b = _mm256_mul_pd(self.twiddles_im[1], x1m6);
        let m0205b = nmadd_pd(m0205b, self.twiddles_im[2], x2m5);
        let m0205b = nmadd_pd(m0205b, self.twiddles_im[0], x3m4);
        let [y02, y05] = self.bf2.process([m0205a, m0205b]);

        let m0304a = fmadd_pd(data[0], self.twiddles_re[2], x1p6);
        let m0304a = fmadd_pd(m0304a, self.twiddles_re[0], x2p5);
        let m0304a = fmadd_pd(m0304a, self.twiddles_re[1], x3p4);
        let m0304b = _mm256_mul_pd(self.twiddles_im[2], x1m6);
        let m0304b = nmadd_pd(m0304b, self.twiddles_im[0], x2m5);
        let m0304b = fmadd_pd(m0304b, self.twiddles_im[1], x3m4);
        let [y03, y04] = self.bf2.process([m0304a, m0304b]);

        [y00, y01, y02, y03, y04, y05, y06]
    }
}
boilerplate_avx_parallel_butterfly!(AvxF64ParallelButterfly7, __m256d, 7);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_parallel_fft;

    macro_rules! test_parallel_butterfly_32_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly = unsafe { $struct_name::new(FftDirection::Forward) };
                check_parallel_fft::<__m256, $size>(&butterfly, FftDirection::Forward);

                let butterfly_inverse = unsafe { $struct_name::new(FftDirection::Inverse) };
                check_parallel_fft::<__m256, $size>(&butterfly_inverse, FftDirection::Inverse);
            }
        };
    }

    test_parallel_butterfly_32_func!(test_avxf32_parallel_butterfly2, AvxF32ParallelButterfly2, 2);
    test_parallel_butterfly_32_func!(test_avxf32_parallel_butterfly3, AvxF32ParallelButterfly3, 3);
    test_parallel_butterfly_32_func!(test_avxf32_parallel_butterfly4, AvxF32ParallelButterfly4, 4);
    test_parallel_butterfly_32_func!(test_avxf32_parallel_butterfly5, AvxF32ParallelButterfly5, 5);
    test_parallel_butterfly_32_func!(test_avxf32_parallel_butterfly6, AvxF32ParallelButterfly6, 6);
    test_parallel_butterfly_32_func!(test_avxf32_parallel_butterfly7, AvxF32ParallelButterfly7, 7);

    macro_rules! test_parallel_butterfly_64_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly = unsafe { $struct_name::new(FftDirection::Forward) };
                check_parallel_fft::<__m256d, $size>(&butterfly, FftDirection::Forward);

                let butterfly_inverse = unsafe { $struct_name::new(FftDirection::Inverse) };
                check_parallel_fft::<__m256d, $size>(&butterfly_inverse, FftDirection::Inverse);
            }
        };
    }

    test_parallel_butterfly_64_func!(test_avxf64_parallel_butterfly2, AvxF64ParallelButterfly2, 2);
    test_parallel_butterfly_64_func!(test_avxf64_parallel_butterfly3, AvxF64ParallelButterfly3, 3);
    test_parallel_butterfly_64_func!(test_avxf64_parallel_butterfly4, AvxF64ParallelButterfly4, 4);
    test_parallel_butterfly_64_func!(test_avxf64_parallel_butterfly5, AvxF64ParallelButterfly5, 5);
    test_parallel_butterfly_64_func!(test_avxf64_parallel_butterfly6, AvxF64ParallelButterfly6, 6);
    test_parallel_butterfly_64_func!(test_avxf64_parallel_butterfly7, AvxF64ParallelButterfly7, 7);
}
