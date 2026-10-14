use num_complex::Complex;

use crate::{simd::column_fft::ColumnFft, twiddles, FftDirection, FftNum};

pub struct ScalarColumnButterfly2<T> {
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FftNum> ScalarColumnButterfly2<T> {
    #[inline(always)]
    pub fn new(_direction: FftDirection) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T: FftNum> ColumnFft<Complex<T>, 2> for ScalarColumnButterfly2<T> {
    #[inline(always)]
    fn column_fft(&self, data: [Complex<T>; 2]) -> [Complex<T>; 2] {
        [data[0] + data[1], data[0] - data[1]]
    }
}

pub struct ScalarColumnButterfly3<T> {
    pub twiddle: Complex<T>,
}
impl<T: FftNum> ScalarColumnButterfly3<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle: twiddles::compute_twiddle(1, 3, direction),
        }
    }
}
impl<T: FftNum> ColumnFft<Complex<T>, 3> for ScalarColumnButterfly3<T> {
    #[inline(always)]
    fn column_fft(&self, data: [Complex<T>; 3]) -> [Complex<T>; 3] {
        let xp = data[1] + data[2];
        let xn = data[1] - data[2];
        let sum = data[0] + xp;

        let temp_a = data[0]
            + Complex {
                re: self.twiddle.re * xp.re,
                im: self.twiddle.re * xp.im,
            };
        let temp_b = Complex {
            re: -self.twiddle.im * xn.im,
            im: self.twiddle.im * xn.re,
        };

        [sum, temp_a + temp_b, temp_a - temp_b]
    }
}

pub struct ScalarColumnButterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FftNum> ScalarColumnButterfly4<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T: FftNum> ColumnFft<Complex<T>, 4> for ScalarColumnButterfly4<T> {
    #[inline(always)]
    fn column_fft(&self, data: [Complex<T>; 4]) -> [Complex<T>; 4] {
        let bf2 = ScalarColumnButterfly2::new(self.direction);

        // step 2: column FFTs
        let col0 = bf2.column_fft([data[0], data[2]]);
        let mut col1 = bf2.column_fft([data[1], data[3]]);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        col1[1] = twiddles::rotate_90(col1[1], self.direction);

        // step 4: transpose, (merged into the next step)

        // step 5: row FFTs
        let row0 = bf2.column_fft([col0[0], col1[0]]);
        let row1 = bf2.column_fft([col0[1], col1[1]]);
        [row0[0], row1[0], row0[1], row1[1]]
    }
}

pub struct ScalarColumnButterfly5<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}
impl<T: FftNum> ScalarColumnButterfly5<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 5, direction),
            twiddle2: twiddles::compute_twiddle(2, 5, direction),
        }
    }
}
impl<T: FftNum> ColumnFft<Complex<T>, 5> for ScalarColumnButterfly5<T> {
    #[inline(never)] // refusing to inline this code reduces code size, and doesn't hurt performance
    fn column_fft(&self, data: [Complex<T>; 5]) -> [Complex<T>; 5] {
        // let mut outer = Butterfly2::perform_fft_array([buffer.load(1), buffer.load(4)]);
        // let mut inner = Butterfly2::perform_fft_array([buffer.load(2), buffer.load(3)]);
        // let input0 = buffer.load(0);

        // buffer.store(input0 + outer[0] + inner[0], 0);

        // inner[1] = twiddles::rotate_90(inner[1], true);
        // outer[1] = twiddles::rotate_90(outer[1], true);

        // {
        //     let twiddled1 = outer[0] * self.twiddles[0].re;
        //     let twiddled2 = inner[0] * self.twiddles[1].re;
        //     let twiddled3 = inner[1] * self.twiddles[1].im;
        //     let twiddled4 = outer[1] * self.twiddles[0].im;

        //     let sum12 = twiddled1 + twiddled2;
        //     let sum34 = twiddled4 + twiddled3;

        //     let output1 = sum12 + sum34;
        //     let output4 = sum12 - sum34;

        //     buffer.store(input0 + output1, 1);
        //     buffer.store(input0 + output4, 4);
        // }

        // {
        //     let twiddled1 = outer[0] * self.twiddles[1].re;
        //     let twiddled2 = inner[0] * self.twiddles[0].re;
        //     let twiddled3 = inner[1] * self.twiddles[0].im;
        //     let twiddled4 = outer[1] * self.twiddles[1].im;
        // }

        // Let's do a plain 5-point Dft
        // |X0|   | W0 W0  W0  W0  W0  |   |x0|
        // |X1|   | W0 W1  W2  W3  W4  |   |x1|
        // |X2| = | W0 W2  W4  W6  W8  | * |x2|
        // |X3|   | W0 W3  W6  W9  W12 |   |x3|
        // |X4|   | W0 W4  W8  W12 W16 |   |x4|
        //
        // where Wn = exp(-2*pi*n/5) for a forward transform, and exp(+2*pi*n/5) for an direction.
        //
        // This can be simplified a bit since exp(-2*pi*n/5) = exp(-2*pi*n/5 + m*2*pi)
        // |X0|   | W0 W0  W0  W0  W0 |   |x0|
        // |X1|   | W0 W1  W2  W3  W4 |   |x1|
        // |X2| = | W0 W2  W4  W1  W3 | * |x2|
        // |X3|   | W0 W3  W1  W4  W2 |   |x3|
        // |X4|   | W0 W4  W3  W2  W1 |   |x4|
        //
        // Next we can use the symmetry that W3 = W2* and W4 = W1* (where * means complex conjugate), and W0 = 1
        // |X0|   | 1  1   1   1   1   |   |x0|
        // |X1|   | 1  W1  W2  W2* W1* |   |x1|
        // |X2| = | 1  W2  W1* W1  W2* | * |x2|
        // |X3|   | 1  W2* W1  W1* W2  |   |x3|
        // |X4|   | 1  W1* W2* W2  W1  |   |x4|
        //
        // Next, we write out the whole expression with real and imaginary parts.
        // X0 = x0 + x1 + x2 + x3 + x4
        // X1 = x0 + (W1.re + j*W1.im)*x1 + (W2.re + j*W2.im)*x2 + (W2.re - j*W2.im)*x3 + (W1.re - j*W1.im)*x4
        // X2 = x0 + (W2.re + j*W2.im)*x1 + (W1.re - j*W1.im)*x2 + (W1.re + j*W1.im)*x3 + (W2.re - j*W2.im)*x4
        // X3 = x0 + (W2.re - j*W2.im)*x1 + (W1.re + j*W1.im)*x2 + (W1.re - j*W1.im)*x3 + (W2.re + j*W2.im)*x4
        // X4 = x0 + (W1.re - j*W1.im)*x1 + (W2.re - j*W2.im)*x2 + (W2.re + j*W2.im)*x3 + (W1.re + j*W1.im)*x4
        //
        // Then we rearrange and sort terms.
        // X0 = x0 + x1 + x2 + x3 + x4
        // X1 = x0 + W1.re*(x1+x4) + W2.re*(x2+x3) + j*(W1.im*(x1-x4) + W2.im*(x2-x3))
        // X2 = x0 + W1.re*(x2+x3) + W2.re*(x1+x4) - j*(W1.im*(x2-x3) - W2.im*(x1-x4))
        // X3 = x0 + W1.re*(x2+x3) + W2.re*(x1+x4) + j*(W1.im*(x2-x3) - W2.im*(x1-x4))
        // X4 = x0 + W1.re*(x1+x4) + W2.re*(x2+x3) - j*(W1.im*(x1-x4) + W2.im*(x2-x3))
        //
        // Now we define x14p=x1+x4 x14n=x1-x4, x23p=x2+x3, x23n=x2-x3
        // X0 = x0 + x1 + x2 + x3 + x4
        // X1 = x0 + W1.re*(x14p) + W2.re*(x23p) + j*(W1.im*(x14n) + W2.im*(x23n))
        // X2 = x0 + W1.re*(x23p) + W2.re*(x14p) - j*(W1.im*(x23n) - W2.im*(x14n))
        // X3 = x0 + W1.re*(x23p) + W2.re*(x14p) + j*(W1.im*(x23n) - W2.im*(x14n))
        // X4 = x0 + W1.re*(x14p) + W2.re*(x23p) - j*(W1.im*(x14n) + W2.im*(x23n))
        //
        // The final step is to write out real and imaginary parts of x14n etc, and replacing using j*j=-1
        // After this it's easy to remove any repeated calculation of the same values.

        let x14p = data[1] + data[4];
        let x14n = data[1] - data[4];
        let x23p = data[2] + data[3];
        let x23n = data[2] - data[3];
        let sum = data[0] + x14p + x23p;
        let b14re_a = data[0].re + self.twiddle1.re * x14p.re + self.twiddle2.re * x23p.re;
        let b14re_b = self.twiddle1.im * x14n.im + self.twiddle2.im * x23n.im;
        let b23re_a = data[0].re + self.twiddle2.re * x14p.re + self.twiddle1.re * x23p.re;
        let b23re_b = self.twiddle2.im * x14n.im + -self.twiddle1.im * x23n.im;

        let b14im_a = data[0].im + self.twiddle1.re * x14p.im + self.twiddle2.re * x23p.im;
        let b14im_b = self.twiddle1.im * x14n.re + self.twiddle2.im * x23n.re;
        let b23im_a = data[0].im + self.twiddle2.re * x14p.im + self.twiddle1.re * x23p.im;
        let b23im_b = self.twiddle2.im * x14n.re + -self.twiddle1.im * x23n.re;

        let out1re = b14re_a - b14re_b;
        let out1im = b14im_a + b14im_b;
        let out2re = b23re_a - b23re_b;
        let out2im = b23im_a + b23im_b;
        let out3re = b23re_a + b23re_b;
        let out3im = b23im_a - b23im_b;
        let out4re = b14re_a + b14re_b;
        let out4im = b14im_a - b14im_b;

        [
            sum,
            Complex {
                re: out1re,
                im: out1im,
            },
            Complex {
                re: out2re,
                im: out2im,
            },
            Complex {
                re: out3re,
                im: out3im,
            },
            Complex {
                re: out4re,
                im: out4im,
            },
        ]
    }
}

pub struct ScalarColumnButterfly6<T> {
    bf3: ScalarColumnButterfly3<T>,
    bf2: ScalarColumnButterfly2<T>,
}
impl<T: FftNum> ScalarColumnButterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            bf3: ScalarColumnButterfly3::new(direction),
            bf2: ScalarColumnButterfly2::new(direction),
        }
    }
}
impl<T: FftNum> ColumnFft<Complex<T>, 6> for ScalarColumnButterfly6<T> {
    #[inline(always)]
    fn column_fft(&self, data: [Complex<T>; 6]) -> [Complex<T>; 6] {
        //since GCD(2,3) == 1 we're going to hardcode a step of the Good-Thomas algorithm to avoid twiddle factors

        // step 1: reorder the input directly into the scratch. normally there's a whole thing to compute this ordering
        // but thankfully we can just precompute it and hardcode it
        let col0 = [data[0], data[2], data[4]];
        let col1 = [data[3], data[5], data[1]];

        // step 2: column FFTs
        let col0 = self.bf3.column_fft(col0);
        let col1 = self.bf3.column_fft(col1);

        // step 3: apply twiddle factors -- SKIPPED because good-thomas doesn't have twiddle factors :)

        // step 4: transpose (merged into next step)

        // step 5: row FFTs
        let row0 = self.bf2.column_fft([col0[0], col1[0]]);
        let row1 = self.bf2.column_fft([col0[1], col1[1]]);
        let row2 = self.bf2.column_fft([col0[2], col1[2]]);

        // step 6: reorder the result back into the buffer. again we would normally have to do an expensive computation
        // but instead we can precompute and hardcode the ordering
        // note that we're also rolling a transpose step into this reorder
        [row0[0], row1[1], row2[0], row0[1], row1[0], row2[1]]
    }
}

pub struct ScalarColumnButterfly7<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}
impl<T: FftNum> ScalarColumnButterfly7<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 7, direction),
            twiddle2: twiddles::compute_twiddle(2, 7, direction),
            twiddle3: twiddles::compute_twiddle(3, 7, direction),
        }
    }
}
impl<T: FftNum> ColumnFft<Complex<T>, 7> for ScalarColumnButterfly7<T> {
    #[inline(never)] // refusing to inline this code reduces code size, and doesn't hurt performance
    fn column_fft(&self, data: [Complex<T>; 7]) -> [Complex<T>; 7] {
        // let mut outer = Butterfly2::perform_fft_array([buffer.load(1), buffer.load(6)]);
        // let mut mid   = Butterfly2::perform_fft_array([buffer.load(2), buffer.load(5)]);
        // let mut inner = Butterfly2::perform_fft_array([buffer.load(3), buffer.load(4)]);
        // let input0 = buffer.load(0);

        // buffer.store(input0 + outer[0] + mid[0] + inner[0], 0);

        // inner[1] = twiddles::rotate_90(inner[1], true);
        // mid[1]   = twiddles::rotate_90(mid[1],   true);
        // outer[1] = twiddles::rotate_90(outer[1], true);

        // {
        //     let twiddled1 = outer[0] * self.twiddles[0].re;
        //     let twiddled2 =   mid[0] * self.twiddles[1].re;
        //     let twiddled3 = inner[0] * self.twiddles[2].re;
        //     let twiddled4 = inner[1] * self.twiddles[2].im;
        //     let twiddled5 =   mid[1] * self.twiddles[1].im;
        //     let twiddled6 = outer[1] * self.twiddles[0].im;

        //     let sum123 = twiddled1 + twiddled2 + twiddled3;
        //     let sum456 = twiddled4 + twiddled5 + twiddled6;

        //     let output1 = sum123 + sum456;
        //     let output6 = sum123 - sum456;

        //     buffer.store(input0 + output1, 1);
        //     buffer.store(input0 + output6, 6);
        // }

        // {
        //     let twiddled1 = outer[0] * self.twiddles[1].re;
        //     let twiddled2 =   mid[0] * self.twiddles[2].re;
        //     let twiddled3 = inner[0] * self.twiddles[0].re;
        //     let twiddled4 = inner[1] * self.twiddles[0].im;
        //     let twiddled5 =   mid[1] * self.twiddles[2].im;
        //     let twiddled6 = outer[1] * self.twiddles[1].im;

        //     let sum123 = twiddled1 + twiddled2 + twiddled3;
        //     let sum456 = twiddled6 - twiddled4 - twiddled5;

        //     let output2 = sum123 + sum456;
        //     let output5 = sum123 - sum456;

        //     buffer.store(input0 + output2, 2);
        //     buffer.store(input0 + output5, 5);
        // }

        // Let's do a plain 7-point Dft
        // |X0|   | W0 W0  W0  W0  W0  W0  W0  |   |x0|
        // |X1|   | W0 W1  W2  W3  W4  W5  W6  |   |x1|
        // |X2|   | W0 W2  W4  W6  W8  W10 W12 |   |x2|
        // |X3| = | W0 W3  W6  W9  W12 W15 W18 | * |x3|
        // |X4|   | W0 W4  W8  W12 W16 W20 W24 |   |x4|
        // |X5|   | W0 W5  W10 W15 W20 W25 W30 |   |x4|
        // |X6|   | W0 W6  W12 W18 W24 W30 W36 |   |x4|
        //
        // where Wn = exp(-2*pi*n/7) for a forward transform, and exp(+2*pi*n/7) for an direction.
        //
        // Using the same logic as for the 5-point butterfly, this can be simplified to:
        // |X0|   | 1  1   1   1   1   1   1   |   |x0|
        // |X1|   | 1  W1  W2  W3  W3* W2* W1* |   |x1|
        // |X2|   | 1  W2  W3* W1* W1  W3  W2* |   |x2|
        // |X3| = | 1  W3  W1* W2  W2* W1  W3* | * |x3|
        // |X4|   | 1  W3* W1  W2* W2  W1* W3  |   |x4|
        // |X5|   | 1  W2* W3  W1  W1* W3* W2  |   |x5|
        // |X6|   | 1  W1* W2* W3* W3  W2  W1  |   |x6|
        //
        // From here it's just about eliminating repeated calculations, following the same procedure as for the 5-point butterfly.

        let x16p = data[1] + data[6];
        let x16n = data[1] - data[6];
        let x25p = data[2] + data[5];
        let x25n = data[2] - data[5];
        let x34p = data[3] + data[4];
        let x34n = data[3] - data[4];
        let sum = data[0] + x16p + x25p + x34p;

        let x16re_a = data[0].re
            + self.twiddle1.re * x16p.re
            + self.twiddle2.re * x25p.re
            + self.twiddle3.re * x34p.re;
        let x16re_b =
            self.twiddle1.im * x16n.im + self.twiddle2.im * x25n.im + self.twiddle3.im * x34n.im;
        let x25re_a = data[0].re
            + self.twiddle1.re * x34p.re
            + self.twiddle2.re * x16p.re
            + self.twiddle3.re * x25p.re;
        let x25re_b =
            -self.twiddle1.im * x34n.im + self.twiddle2.im * x16n.im - self.twiddle3.im * x25n.im;
        let x34re_a = data[0].re
            + self.twiddle1.re * x25p.re
            + self.twiddle2.re * x34p.re
            + self.twiddle3.re * x16p.re;
        let x34re_b =
            -self.twiddle1.im * x25n.im + self.twiddle2.im * x34n.im + self.twiddle3.im * x16n.im;
        let x16im_a = data[0].im
            + self.twiddle1.re * x16p.im
            + self.twiddle2.re * x25p.im
            + self.twiddle3.re * x34p.im;
        let x16im_b =
            self.twiddle1.im * x16n.re + self.twiddle2.im * x25n.re + self.twiddle3.im * x34n.re;
        let x25im_a = data[0].im
            + self.twiddle1.re * x34p.im
            + self.twiddle2.re * x16p.im
            + self.twiddle3.re * x25p.im;
        let x25im_b =
            -self.twiddle1.im * x34n.re + self.twiddle2.im * x16n.re - self.twiddle3.im * x25n.re;
        let x34im_a = data[0].im
            + self.twiddle1.re * x25p.im
            + self.twiddle2.re * x34p.im
            + self.twiddle3.re * x16p.im;
        let x34im_b =
            self.twiddle1.im * x25n.re - self.twiddle2.im * x34n.re - self.twiddle3.im * x16n.re;

        let out1re = x16re_a - x16re_b;
        let out1im = x16im_a + x16im_b;
        let out2re = x25re_a - x25re_b;
        let out2im = x25im_a + x25im_b;
        let out3re = x34re_a - x34re_b;
        let out3im = x34im_a - x34im_b;
        let out4re = x34re_a + x34re_b;
        let out4im = x34im_a + x34im_b;
        let out5re = x25re_a + x25re_b;
        let out5im = x25im_a - x25im_b;
        let out6re = x16re_a + x16re_b;
        let out6im = x16im_a - x16im_b;

        [
            sum,
            Complex {
                re: out1re,
                im: out1im,
            },
            Complex {
                re: out2re,
                im: out2im,
            },
            Complex {
                re: out3re,
                im: out3im,
            },
            Complex {
                re: out4re,
                im: out4im,
            },
            Complex {
                re: out5re,
                im: out5im,
            },
            Complex {
                re: out6re,
                im: out6im,
            },
        ]
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_column_fft;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(FftDirection::Forward);
                check_column_fft::<Complex<f32>, $size>(&butterfly, FftDirection::Forward);

                let butterfly_inverse = $struct_name::new(FftDirection::Inverse);
                check_column_fft::<Complex<f32>, $size>(&butterfly_inverse, FftDirection::Inverse);
            }
        };
    }

    test_butterfly_func!(test_scalar_column_butterfly2, ScalarColumnButterfly2, 2);
    test_butterfly_func!(test_scalar_column_butterfly3, ScalarColumnButterfly3, 3);
    test_butterfly_func!(test_scalar_column_butterfly4, ScalarColumnButterfly4, 4);
    test_butterfly_func!(test_scalar_column_butterfly5, ScalarColumnButterfly5, 5);
    test_butterfly_func!(test_scalar_column_butterfly6, ScalarColumnButterfly6, 6);
    test_butterfly_func!(test_scalar_column_butterfly7, ScalarColumnButterfly7, 7);
}
