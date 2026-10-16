use num_complex::Complex;

use crate::{simd::parallel_fft::ParallelFft, twiddles, FftDirection, FftNum};

pub struct ScalarParallelButterfly2<T> {
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FftNum> ScalarParallelButterfly2<T> {
    #[inline(always)]
    pub fn new(_direction: FftDirection) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T: FftNum> ParallelFft<Complex<T>, 2> for ScalarParallelButterfly2<T> {
    #[inline(always)]
    fn parallel_fft(&self, data: [Complex<T>; 2]) -> [Complex<T>; 2] {
        [data[0] + data[1], data[0] - data[1]]
    }
}

pub struct ScalarParallelButterfly3<T> {
    pub twiddle: Complex<T>,
}
impl<T: FftNum> ScalarParallelButterfly3<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle: twiddles::compute_twiddle(1, 3, direction),
        }
    }
}
impl<T: FftNum> ParallelFft<Complex<T>, 3> for ScalarParallelButterfly3<T> {
    #[inline(always)]
    fn parallel_fft(&self, data: [Complex<T>; 3]) -> [Complex<T>; 3] {
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

pub struct ScalarParallelButterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FftNum> ScalarParallelButterfly4<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T: FftNum> ParallelFft<Complex<T>, 4> for ScalarParallelButterfly4<T> {
    #[inline(always)]
    fn parallel_fft(&self, data: [Complex<T>; 4]) -> [Complex<T>; 4] {
        let bf2 = ScalarParallelButterfly2::new(self.direction);

        // step 2: column FFTs
        let col0 = bf2.parallel_fft([data[0], data[2]]);
        let mut col1 = bf2.parallel_fft([data[1], data[3]]);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        col1[1] = twiddles::rotate_90(col1[1], self.direction);

        // step 4: transpose, (merged into the next step)

        // step 5: row FFTs
        let row0 = bf2.parallel_fft([col0[0], col1[0]]);
        let row1 = bf2.parallel_fft([col0[1], col1[1]]);
        [row0[0], row1[0], row0[1], row1[1]]
    }
}

pub struct ScalarParallelButterfly5<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}
impl<T: FftNum> ScalarParallelButterfly5<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 5, direction),
            twiddle2: twiddles::compute_twiddle(2, 5, direction),
        }
    }
}
impl<T: FftNum> ParallelFft<Complex<T>, 5> for ScalarParallelButterfly5<T> {
    #[inline(never)] // refusing to inline this code reduces code size, and doesn't hurt performance
    fn parallel_fft(&self, data: [Complex<T>; 5]) -> [Complex<T>; 5] {
        // let mut outer = Butterfly2::perform_fft_array([data[1), data[4)]);
        // let mut inner = Butterfly2::perform_fft_array([data[2), data[3)]);
        // let input0 = data[0];

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

pub struct ScalarParallelButterfly6<T> {
    bf3: ScalarParallelButterfly3<T>,
    bf2: ScalarParallelButterfly2<T>,
}
impl<T: FftNum> ScalarParallelButterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            bf3: ScalarParallelButterfly3::new(direction),
            bf2: ScalarParallelButterfly2::new(direction),
        }
    }
}
impl<T: FftNum> ParallelFft<Complex<T>, 6> for ScalarParallelButterfly6<T> {
    #[inline(always)]
    fn parallel_fft(&self, data: [Complex<T>; 6]) -> [Complex<T>; 6] {
        //since GCD(2,3) == 1 we're going to hardcode a step of the Good-Thomas algorithm to avoid twiddle factors

        // step 1: reorder the input directly into the scratch. normally there's a whole thing to compute this ordering
        // but thankfully we can just precompute it and hardcode it
        let col0 = [data[0], data[2], data[4]];
        let col1 = [data[3], data[5], data[1]];

        // step 2: column FFTs
        let col0 = self.bf3.parallel_fft(col0);
        let col1 = self.bf3.parallel_fft(col1);

        // step 3: apply twiddle factors -- SKIPPED because good-thomas doesn't have twiddle factors :)

        // step 4: transpose (merged into next step)

        // step 5: row FFTs
        let row0 = self.bf2.parallel_fft([col0[0], col1[0]]);
        let row1 = self.bf2.parallel_fft([col0[1], col1[1]]);
        let row2 = self.bf2.parallel_fft([col0[2], col1[2]]);

        // step 6: reorder the result back into the buffer. again we would normally have to do an expensive computation
        // but instead we can precompute and hardcode the ordering
        // note that we're also rolling a transpose step into this reorder
        [row0[0], row1[1], row2[0], row0[1], row1[0], row2[1]]
    }
}

pub struct ScalarParallelButterfly7<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}
impl<T: FftNum> ScalarParallelButterfly7<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 7, direction),
            twiddle2: twiddles::compute_twiddle(2, 7, direction),
            twiddle3: twiddles::compute_twiddle(3, 7, direction),
        }
    }
}
impl<T: FftNum> ParallelFft<Complex<T>, 7> for ScalarParallelButterfly7<T> {
    #[inline(never)] // refusing to inline this code reduces code size, and doesn't hurt performance
    fn parallel_fft(&self, data: [Complex<T>; 7]) -> [Complex<T>; 7] {
        // let mut outer = Butterfly2::perform_fft_array([data[1), data[6)]);
        // let mut mid   = Butterfly2::perform_fft_array([data[2), data[5)]);
        // let mut inner = Butterfly2::perform_fft_array([data[3), data[4)]);
        // let input0 = data[0];

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

pub struct ScalarParallelButterfly11<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
}
impl<T: FftNum> ScalarParallelButterfly11<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 11, direction),
            twiddle2: twiddles::compute_twiddle(2, 11, direction),
            twiddle3: twiddles::compute_twiddle(3, 11, direction),
            twiddle4: twiddles::compute_twiddle(4, 11, direction),
            twiddle5: twiddles::compute_twiddle(5, 11, direction),
        }
    }
}
impl<T: FftNum> ParallelFft<Complex<T>,11> for ScalarParallelButterfly11<T> {
    #[inline(never)] // refusing to inline this code reduces code size, and doesn't hurt performance
    fn parallel_fft(&self, data: [Complex<T>; 11]) -> [Complex<T>; 11] {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.

        let x110p = data[1] + data[10];
        let x110n = data[1] - data[10];
        let x29p = data[2] + data[9];
        let x29n = data[2] - data[9];
        let x38p = data[3] + data[8];
        let x38n = data[3] - data[8];
        let x47p = data[4] + data[7];
        let x47n = data[4] - data[7];
        let x56p = data[5] + data[6];
        let x56n = data[5] - data[6];
        let sum = data[0] + x110p + x29p + x38p + x47p + x56p;
        let b110re_a = data[0].re
            + self.twiddle1.re * x110p.re
            + self.twiddle2.re * x29p.re
            + self.twiddle3.re * x38p.re
            + self.twiddle4.re * x47p.re
            + self.twiddle5.re * x56p.re;
        let b110re_b = self.twiddle1.im * x110n.im
            + self.twiddle2.im * x29n.im
            + self.twiddle3.im * x38n.im
            + self.twiddle4.im * x47n.im
            + self.twiddle5.im * x56n.im;
        let b29re_a = data[0].re
            + self.twiddle2.re * x110p.re
            + self.twiddle4.re * x29p.re
            + self.twiddle5.re * x38p.re
            + self.twiddle3.re * x47p.re
            + self.twiddle1.re * x56p.re;
        let b29re_b = self.twiddle2.im * x110n.im
            + self.twiddle4.im * x29n.im
            + -self.twiddle5.im * x38n.im
            + -self.twiddle3.im * x47n.im
            + -self.twiddle1.im * x56n.im;
        let b38re_a = data[0].re
            + self.twiddle3.re * x110p.re
            + self.twiddle5.re * x29p.re
            + self.twiddle2.re * x38p.re
            + self.twiddle1.re * x47p.re
            + self.twiddle4.re * x56p.re;
        let b38re_b = self.twiddle3.im * x110n.im
            + -self.twiddle5.im * x29n.im
            + -self.twiddle2.im * x38n.im
            + self.twiddle1.im * x47n.im
            + self.twiddle4.im * x56n.im;
        let b47re_a = data[0].re
            + self.twiddle4.re * x110p.re
            + self.twiddle3.re * x29p.re
            + self.twiddle1.re * x38p.re
            + self.twiddle5.re * x47p.re
            + self.twiddle2.re * x56p.re;
        let b47re_b = self.twiddle4.im * x110n.im
            + -self.twiddle3.im * x29n.im
            + self.twiddle1.im * x38n.im
            + self.twiddle5.im * x47n.im
            + -self.twiddle2.im * x56n.im;
        let b56re_a = data[0].re
            + self.twiddle5.re * x110p.re
            + self.twiddle1.re * x29p.re
            + self.twiddle4.re * x38p.re
            + self.twiddle2.re * x47p.re
            + self.twiddle3.re * x56p.re;
        let b56re_b = self.twiddle5.im * x110n.im
            + -self.twiddle1.im * x29n.im
            + self.twiddle4.im * x38n.im
            + -self.twiddle2.im * x47n.im
            + self.twiddle3.im * x56n.im;

        let b110im_a = data[0].im
            + self.twiddle1.re * x110p.im
            + self.twiddle2.re * x29p.im
            + self.twiddle3.re * x38p.im
            + self.twiddle4.re * x47p.im
            + self.twiddle5.re * x56p.im;
        let b110im_b = self.twiddle1.im * x110n.re
            + self.twiddle2.im * x29n.re
            + self.twiddle3.im * x38n.re
            + self.twiddle4.im * x47n.re
            + self.twiddle5.im * x56n.re;
        let b29im_a = data[0].im
            + self.twiddle2.re * x110p.im
            + self.twiddle4.re * x29p.im
            + self.twiddle5.re * x38p.im
            + self.twiddle3.re * x47p.im
            + self.twiddle1.re * x56p.im;
        let b29im_b = self.twiddle2.im * x110n.re
            + self.twiddle4.im * x29n.re
            + -self.twiddle5.im * x38n.re
            + -self.twiddle3.im * x47n.re
            + -self.twiddle1.im * x56n.re;
        let b38im_a = data[0].im
            + self.twiddle3.re * x110p.im
            + self.twiddle5.re * x29p.im
            + self.twiddle2.re * x38p.im
            + self.twiddle1.re * x47p.im
            + self.twiddle4.re * x56p.im;
        let b38im_b = self.twiddle3.im * x110n.re
            + -self.twiddle5.im * x29n.re
            + -self.twiddle2.im * x38n.re
            + self.twiddle1.im * x47n.re
            + self.twiddle4.im * x56n.re;
        let b47im_a = data[0].im
            + self.twiddle4.re * x110p.im
            + self.twiddle3.re * x29p.im
            + self.twiddle1.re * x38p.im
            + self.twiddle5.re * x47p.im
            + self.twiddle2.re * x56p.im;
        let b47im_b = self.twiddle4.im * x110n.re
            + -self.twiddle3.im * x29n.re
            + self.twiddle1.im * x38n.re
            + self.twiddle5.im * x47n.re
            + -self.twiddle2.im * x56n.re;
        let b56im_a = data[0].im
            + self.twiddle5.re * x110p.im
            + self.twiddle1.re * x29p.im
            + self.twiddle4.re * x38p.im
            + self.twiddle2.re * x47p.im
            + self.twiddle3.re * x56p.im;
        let b56im_b = self.twiddle5.im * x110n.re
            + -self.twiddle1.im * x29n.re
            + self.twiddle4.im * x38n.re
            + -self.twiddle2.im * x47n.re
            + self.twiddle3.im * x56n.re;

        let out1re = b110re_a - b110re_b;
        let out1im = b110im_a + b110im_b;
        let out2re = b29re_a - b29re_b;
        let out2im = b29im_a + b29im_b;
        let out3re = b38re_a - b38re_b;
        let out3im = b38im_a + b38im_b;
        let out4re = b47re_a - b47re_b;
        let out4im = b47im_a + b47im_b;
        let out5re = b56re_a - b56re_b;
        let out5im = b56im_a + b56im_b;
        let out6re = b56re_a + b56re_b;
        let out6im = b56im_a - b56im_b;
        let out7re = b47re_a + b47re_b;
        let out7im = b47im_a - b47im_b;
        let out8re = b38re_a + b38re_b;
        let out8im = b38im_a - b38im_b;
        let out9re = b29re_a + b29re_b;
        let out9im = b29im_a - b29im_b;
        let out10re = b110re_a + b110re_b;
        let out10im = b110im_a - b110im_b;
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
            Complex {
                re: out7re,
                im: out7im,
            },
            Complex {
                re: out8re,
                im: out8im,
            },
            Complex {
                re: out9re,
                im: out9im,
            },
            Complex {
                re: out10re,
                im: out10im,
            },
        ]
    }
}

pub struct ScalarParallelButterfly31<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
    twiddle9: Complex<T>,
    twiddle10: Complex<T>,
    twiddle11: Complex<T>,
    twiddle12: Complex<T>,
    twiddle13: Complex<T>,
    twiddle14: Complex<T>,
    twiddle15: Complex<T>,
}
impl<T: FftNum> ScalarParallelButterfly31<T> {
    pub fn new(direction: FftDirection) -> Self {
        let twiddle1: Complex<T> = twiddles::compute_twiddle(1, 31, direction);
        let twiddle2: Complex<T> = twiddles::compute_twiddle(2, 31, direction);
        let twiddle3: Complex<T> = twiddles::compute_twiddle(3, 31, direction);
        let twiddle4: Complex<T> = twiddles::compute_twiddle(4, 31, direction);
        let twiddle5: Complex<T> = twiddles::compute_twiddle(5, 31, direction);
        let twiddle6: Complex<T> = twiddles::compute_twiddle(6, 31, direction);
        let twiddle7: Complex<T> = twiddles::compute_twiddle(7, 31, direction);
        let twiddle8: Complex<T> = twiddles::compute_twiddle(8, 31, direction);
        let twiddle9: Complex<T> = twiddles::compute_twiddle(9, 31, direction);
        let twiddle10: Complex<T> = twiddles::compute_twiddle(10, 31, direction);
        let twiddle11: Complex<T> = twiddles::compute_twiddle(11, 31, direction);
        let twiddle12: Complex<T> = twiddles::compute_twiddle(12, 31, direction);
        let twiddle13: Complex<T> = twiddles::compute_twiddle(13, 31, direction);
        let twiddle14: Complex<T> = twiddles::compute_twiddle(14, 31, direction);
        let twiddle15: Complex<T> = twiddles::compute_twiddle(15, 31, direction);
        Self {
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            twiddle7,
            twiddle8,
            twiddle9,
            twiddle10,
            twiddle11,
            twiddle12,
            twiddle13,
            twiddle14,
            twiddle15,
        }
    }
}
impl<T: FftNum> ParallelFft<Complex<T>, 31> for ScalarParallelButterfly31<T> {
    #[inline(never)] // refusing to inline this code reduces code size, and doesn't hurt performance
    fn parallel_fft(&self, data: [Complex<T>; 31]) -> [Complex<T>; 31] {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x130p = data[1] + data[30];
        let x130n = data[1] - data[30];
        let x229p = data[2] + data[29];
        let x229n = data[2] - data[29];
        let x328p = data[3] + data[28];
        let x328n = data[3] - data[28];
        let x427p = data[4] + data[27];
        let x427n = data[4] - data[27];
        let x526p = data[5] + data[26];
        let x526n = data[5] - data[26];
        let x625p = data[6] + data[25];
        let x625n = data[6] - data[25];
        let x724p = data[7] + data[24];
        let x724n = data[7] - data[24];
        let x823p = data[8] + data[23];
        let x823n = data[8] - data[23];
        let x922p = data[9] + data[22];
        let x922n = data[9] - data[22];
        let x1021p = data[10] + data[21];
        let x1021n = data[10] - data[21];
        let x1120p = data[11] + data[20];
        let x1120n = data[11] - data[20];
        let x1219p = data[12] + data[19];
        let x1219n = data[12] - data[19];
        let x1318p = data[13] + data[18];
        let x1318n = data[13] - data[18];
        let x1417p = data[14] + data[17];
        let x1417n = data[14] - data[17];
        let x1516p = data[15] + data[16];
        let x1516n = data[15] - data[16];
        let sum = data[0]
            + x130p
            + x229p
            + x328p
            + x427p
            + x526p
            + x625p
            + x724p
            + x823p
            + x922p
            + x1021p
            + x1120p
            + x1219p
            + x1318p
            + x1417p
            + x1516p;
        let b130re_a = data[0].re
            + self.twiddle1.re * x130p.re
            + self.twiddle2.re * x229p.re
            + self.twiddle3.re * x328p.re
            + self.twiddle4.re * x427p.re
            + self.twiddle5.re * x526p.re
            + self.twiddle6.re * x625p.re
            + self.twiddle7.re * x724p.re
            + self.twiddle8.re * x823p.re
            + self.twiddle9.re * x922p.re
            + self.twiddle10.re * x1021p.re
            + self.twiddle11.re * x1120p.re
            + self.twiddle12.re * x1219p.re
            + self.twiddle13.re * x1318p.re
            + self.twiddle14.re * x1417p.re
            + self.twiddle15.re * x1516p.re;
        let b130re_b = self.twiddle1.im * x130n.im
            + self.twiddle2.im * x229n.im
            + self.twiddle3.im * x328n.im
            + self.twiddle4.im * x427n.im
            + self.twiddle5.im * x526n.im
            + self.twiddle6.im * x625n.im
            + self.twiddle7.im * x724n.im
            + self.twiddle8.im * x823n.im
            + self.twiddle9.im * x922n.im
            + self.twiddle10.im * x1021n.im
            + self.twiddle11.im * x1120n.im
            + self.twiddle12.im * x1219n.im
            + self.twiddle13.im * x1318n.im
            + self.twiddle14.im * x1417n.im
            + self.twiddle15.im * x1516n.im;
        let b229re_a = data[0].re
            + self.twiddle2.re * x130p.re
            + self.twiddle4.re * x229p.re
            + self.twiddle6.re * x328p.re
            + self.twiddle8.re * x427p.re
            + self.twiddle10.re * x526p.re
            + self.twiddle12.re * x625p.re
            + self.twiddle14.re * x724p.re
            + self.twiddle15.re * x823p.re
            + self.twiddle13.re * x922p.re
            + self.twiddle11.re * x1021p.re
            + self.twiddle9.re * x1120p.re
            + self.twiddle7.re * x1219p.re
            + self.twiddle5.re * x1318p.re
            + self.twiddle3.re * x1417p.re
            + self.twiddle1.re * x1516p.re;
        let b229re_b = self.twiddle2.im * x130n.im
            + self.twiddle4.im * x229n.im
            + self.twiddle6.im * x328n.im
            + self.twiddle8.im * x427n.im
            + self.twiddle10.im * x526n.im
            + self.twiddle12.im * x625n.im
            + self.twiddle14.im * x724n.im
            + -self.twiddle15.im * x823n.im
            + -self.twiddle13.im * x922n.im
            + -self.twiddle11.im * x1021n.im
            + -self.twiddle9.im * x1120n.im
            + -self.twiddle7.im * x1219n.im
            + -self.twiddle5.im * x1318n.im
            + -self.twiddle3.im * x1417n.im
            + -self.twiddle1.im * x1516n.im;
        let b328re_a = data[0].re
            + self.twiddle3.re * x130p.re
            + self.twiddle6.re * x229p.re
            + self.twiddle9.re * x328p.re
            + self.twiddle12.re * x427p.re
            + self.twiddle15.re * x526p.re
            + self.twiddle13.re * x625p.re
            + self.twiddle10.re * x724p.re
            + self.twiddle7.re * x823p.re
            + self.twiddle4.re * x922p.re
            + self.twiddle1.re * x1021p.re
            + self.twiddle2.re * x1120p.re
            + self.twiddle5.re * x1219p.re
            + self.twiddle8.re * x1318p.re
            + self.twiddle11.re * x1417p.re
            + self.twiddle14.re * x1516p.re;
        let b328re_b = self.twiddle3.im * x130n.im
            + self.twiddle6.im * x229n.im
            + self.twiddle9.im * x328n.im
            + self.twiddle12.im * x427n.im
            + self.twiddle15.im * x526n.im
            + -self.twiddle13.im * x625n.im
            + -self.twiddle10.im * x724n.im
            + -self.twiddle7.im * x823n.im
            + -self.twiddle4.im * x922n.im
            + -self.twiddle1.im * x1021n.im
            + self.twiddle2.im * x1120n.im
            + self.twiddle5.im * x1219n.im
            + self.twiddle8.im * x1318n.im
            + self.twiddle11.im * x1417n.im
            + self.twiddle14.im * x1516n.im;
        let b427re_a = data[0].re
            + self.twiddle4.re * x130p.re
            + self.twiddle8.re * x229p.re
            + self.twiddle12.re * x328p.re
            + self.twiddle15.re * x427p.re
            + self.twiddle11.re * x526p.re
            + self.twiddle7.re * x625p.re
            + self.twiddle3.re * x724p.re
            + self.twiddle1.re * x823p.re
            + self.twiddle5.re * x922p.re
            + self.twiddle9.re * x1021p.re
            + self.twiddle13.re * x1120p.re
            + self.twiddle14.re * x1219p.re
            + self.twiddle10.re * x1318p.re
            + self.twiddle6.re * x1417p.re
            + self.twiddle2.re * x1516p.re;
        let b427re_b = self.twiddle4.im * x130n.im
            + self.twiddle8.im * x229n.im
            + self.twiddle12.im * x328n.im
            + -self.twiddle15.im * x427n.im
            + -self.twiddle11.im * x526n.im
            + -self.twiddle7.im * x625n.im
            + -self.twiddle3.im * x724n.im
            + self.twiddle1.im * x823n.im
            + self.twiddle5.im * x922n.im
            + self.twiddle9.im * x1021n.im
            + self.twiddle13.im * x1120n.im
            + -self.twiddle14.im * x1219n.im
            + -self.twiddle10.im * x1318n.im
            + -self.twiddle6.im * x1417n.im
            + -self.twiddle2.im * x1516n.im;
        let b526re_a = data[0].re
            + self.twiddle5.re * x130p.re
            + self.twiddle10.re * x229p.re
            + self.twiddle15.re * x328p.re
            + self.twiddle11.re * x427p.re
            + self.twiddle6.re * x526p.re
            + self.twiddle1.re * x625p.re
            + self.twiddle4.re * x724p.re
            + self.twiddle9.re * x823p.re
            + self.twiddle14.re * x922p.re
            + self.twiddle12.re * x1021p.re
            + self.twiddle7.re * x1120p.re
            + self.twiddle2.re * x1219p.re
            + self.twiddle3.re * x1318p.re
            + self.twiddle8.re * x1417p.re
            + self.twiddle13.re * x1516p.re;
        let b526re_b = self.twiddle5.im * x130n.im
            + self.twiddle10.im * x229n.im
            + self.twiddle15.im * x328n.im
            + -self.twiddle11.im * x427n.im
            + -self.twiddle6.im * x526n.im
            + -self.twiddle1.im * x625n.im
            + self.twiddle4.im * x724n.im
            + self.twiddle9.im * x823n.im
            + self.twiddle14.im * x922n.im
            + -self.twiddle12.im * x1021n.im
            + -self.twiddle7.im * x1120n.im
            + -self.twiddle2.im * x1219n.im
            + self.twiddle3.im * x1318n.im
            + self.twiddle8.im * x1417n.im
            + self.twiddle13.im * x1516n.im;
        let b625re_a = data[0].re
            + self.twiddle6.re * x130p.re
            + self.twiddle12.re * x229p.re
            + self.twiddle13.re * x328p.re
            + self.twiddle7.re * x427p.re
            + self.twiddle1.re * x526p.re
            + self.twiddle5.re * x625p.re
            + self.twiddle11.re * x724p.re
            + self.twiddle14.re * x823p.re
            + self.twiddle8.re * x922p.re
            + self.twiddle2.re * x1021p.re
            + self.twiddle4.re * x1120p.re
            + self.twiddle10.re * x1219p.re
            + self.twiddle15.re * x1318p.re
            + self.twiddle9.re * x1417p.re
            + self.twiddle3.re * x1516p.re;
        let b625re_b = self.twiddle6.im * x130n.im
            + self.twiddle12.im * x229n.im
            + -self.twiddle13.im * x328n.im
            + -self.twiddle7.im * x427n.im
            + -self.twiddle1.im * x526n.im
            + self.twiddle5.im * x625n.im
            + self.twiddle11.im * x724n.im
            + -self.twiddle14.im * x823n.im
            + -self.twiddle8.im * x922n.im
            + -self.twiddle2.im * x1021n.im
            + self.twiddle4.im * x1120n.im
            + self.twiddle10.im * x1219n.im
            + -self.twiddle15.im * x1318n.im
            + -self.twiddle9.im * x1417n.im
            + -self.twiddle3.im * x1516n.im;
        let b724re_a = data[0].re
            + self.twiddle7.re * x130p.re
            + self.twiddle14.re * x229p.re
            + self.twiddle10.re * x328p.re
            + self.twiddle3.re * x427p.re
            + self.twiddle4.re * x526p.re
            + self.twiddle11.re * x625p.re
            + self.twiddle13.re * x724p.re
            + self.twiddle6.re * x823p.re
            + self.twiddle1.re * x922p.re
            + self.twiddle8.re * x1021p.re
            + self.twiddle15.re * x1120p.re
            + self.twiddle9.re * x1219p.re
            + self.twiddle2.re * x1318p.re
            + self.twiddle5.re * x1417p.re
            + self.twiddle12.re * x1516p.re;
        let b724re_b = self.twiddle7.im * x130n.im
            + self.twiddle14.im * x229n.im
            + -self.twiddle10.im * x328n.im
            + -self.twiddle3.im * x427n.im
            + self.twiddle4.im * x526n.im
            + self.twiddle11.im * x625n.im
            + -self.twiddle13.im * x724n.im
            + -self.twiddle6.im * x823n.im
            + self.twiddle1.im * x922n.im
            + self.twiddle8.im * x1021n.im
            + self.twiddle15.im * x1120n.im
            + -self.twiddle9.im * x1219n.im
            + -self.twiddle2.im * x1318n.im
            + self.twiddle5.im * x1417n.im
            + self.twiddle12.im * x1516n.im;
        let b823re_a = data[0].re
            + self.twiddle8.re * x130p.re
            + self.twiddle15.re * x229p.re
            + self.twiddle7.re * x328p.re
            + self.twiddle1.re * x427p.re
            + self.twiddle9.re * x526p.re
            + self.twiddle14.re * x625p.re
            + self.twiddle6.re * x724p.re
            + self.twiddle2.re * x823p.re
            + self.twiddle10.re * x922p.re
            + self.twiddle13.re * x1021p.re
            + self.twiddle5.re * x1120p.re
            + self.twiddle3.re * x1219p.re
            + self.twiddle11.re * x1318p.re
            + self.twiddle12.re * x1417p.re
            + self.twiddle4.re * x1516p.re;
        let b823re_b = self.twiddle8.im * x130n.im
            + -self.twiddle15.im * x229n.im
            + -self.twiddle7.im * x328n.im
            + self.twiddle1.im * x427n.im
            + self.twiddle9.im * x526n.im
            + -self.twiddle14.im * x625n.im
            + -self.twiddle6.im * x724n.im
            + self.twiddle2.im * x823n.im
            + self.twiddle10.im * x922n.im
            + -self.twiddle13.im * x1021n.im
            + -self.twiddle5.im * x1120n.im
            + self.twiddle3.im * x1219n.im
            + self.twiddle11.im * x1318n.im
            + -self.twiddle12.im * x1417n.im
            + -self.twiddle4.im * x1516n.im;
        let b922re_a = data[0].re
            + self.twiddle9.re * x130p.re
            + self.twiddle13.re * x229p.re
            + self.twiddle4.re * x328p.re
            + self.twiddle5.re * x427p.re
            + self.twiddle14.re * x526p.re
            + self.twiddle8.re * x625p.re
            + self.twiddle1.re * x724p.re
            + self.twiddle10.re * x823p.re
            + self.twiddle12.re * x922p.re
            + self.twiddle3.re * x1021p.re
            + self.twiddle6.re * x1120p.re
            + self.twiddle15.re * x1219p.re
            + self.twiddle7.re * x1318p.re
            + self.twiddle2.re * x1417p.re
            + self.twiddle11.re * x1516p.re;
        let b922re_b = self.twiddle9.im * x130n.im
            + -self.twiddle13.im * x229n.im
            + -self.twiddle4.im * x328n.im
            + self.twiddle5.im * x427n.im
            + self.twiddle14.im * x526n.im
            + -self.twiddle8.im * x625n.im
            + self.twiddle1.im * x724n.im
            + self.twiddle10.im * x823n.im
            + -self.twiddle12.im * x922n.im
            + -self.twiddle3.im * x1021n.im
            + self.twiddle6.im * x1120n.im
            + self.twiddle15.im * x1219n.im
            + -self.twiddle7.im * x1318n.im
            + self.twiddle2.im * x1417n.im
            + self.twiddle11.im * x1516n.im;
        let b1021re_a = data[0].re
            + self.twiddle10.re * x130p.re
            + self.twiddle11.re * x229p.re
            + self.twiddle1.re * x328p.re
            + self.twiddle9.re * x427p.re
            + self.twiddle12.re * x526p.re
            + self.twiddle2.re * x625p.re
            + self.twiddle8.re * x724p.re
            + self.twiddle13.re * x823p.re
            + self.twiddle3.re * x922p.re
            + self.twiddle7.re * x1021p.re
            + self.twiddle14.re * x1120p.re
            + self.twiddle4.re * x1219p.re
            + self.twiddle6.re * x1318p.re
            + self.twiddle15.re * x1417p.re
            + self.twiddle5.re * x1516p.re;
        let b1021re_b = self.twiddle10.im * x130n.im
            + -self.twiddle11.im * x229n.im
            + -self.twiddle1.im * x328n.im
            + self.twiddle9.im * x427n.im
            + -self.twiddle12.im * x526n.im
            + -self.twiddle2.im * x625n.im
            + self.twiddle8.im * x724n.im
            + -self.twiddle13.im * x823n.im
            + -self.twiddle3.im * x922n.im
            + self.twiddle7.im * x1021n.im
            + -self.twiddle14.im * x1120n.im
            + -self.twiddle4.im * x1219n.im
            + self.twiddle6.im * x1318n.im
            + -self.twiddle15.im * x1417n.im
            + -self.twiddle5.im * x1516n.im;
        let b1120re_a = data[0].re
            + self.twiddle11.re * x130p.re
            + self.twiddle9.re * x229p.re
            + self.twiddle2.re * x328p.re
            + self.twiddle13.re * x427p.re
            + self.twiddle7.re * x526p.re
            + self.twiddle4.re * x625p.re
            + self.twiddle15.re * x724p.re
            + self.twiddle5.re * x823p.re
            + self.twiddle6.re * x922p.re
            + self.twiddle14.re * x1021p.re
            + self.twiddle3.re * x1120p.re
            + self.twiddle8.re * x1219p.re
            + self.twiddle12.re * x1318p.re
            + self.twiddle1.re * x1417p.re
            + self.twiddle10.re * x1516p.re;
        let b1120re_b = self.twiddle11.im * x130n.im
            + -self.twiddle9.im * x229n.im
            + self.twiddle2.im * x328n.im
            + self.twiddle13.im * x427n.im
            + -self.twiddle7.im * x526n.im
            + self.twiddle4.im * x625n.im
            + self.twiddle15.im * x724n.im
            + -self.twiddle5.im * x823n.im
            + self.twiddle6.im * x922n.im
            + -self.twiddle14.im * x1021n.im
            + -self.twiddle3.im * x1120n.im
            + self.twiddle8.im * x1219n.im
            + -self.twiddle12.im * x1318n.im
            + -self.twiddle1.im * x1417n.im
            + self.twiddle10.im * x1516n.im;
        let b1219re_a = data[0].re
            + self.twiddle12.re * x130p.re
            + self.twiddle7.re * x229p.re
            + self.twiddle5.re * x328p.re
            + self.twiddle14.re * x427p.re
            + self.twiddle2.re * x526p.re
            + self.twiddle10.re * x625p.re
            + self.twiddle9.re * x724p.re
            + self.twiddle3.re * x823p.re
            + self.twiddle15.re * x922p.re
            + self.twiddle4.re * x1021p.re
            + self.twiddle8.re * x1120p.re
            + self.twiddle11.re * x1219p.re
            + self.twiddle1.re * x1318p.re
            + self.twiddle13.re * x1417p.re
            + self.twiddle6.re * x1516p.re;
        let b1219re_b = self.twiddle12.im * x130n.im
            + -self.twiddle7.im * x229n.im
            + self.twiddle5.im * x328n.im
            + -self.twiddle14.im * x427n.im
            + -self.twiddle2.im * x526n.im
            + self.twiddle10.im * x625n.im
            + -self.twiddle9.im * x724n.im
            + self.twiddle3.im * x823n.im
            + self.twiddle15.im * x922n.im
            + -self.twiddle4.im * x1021n.im
            + self.twiddle8.im * x1120n.im
            + -self.twiddle11.im * x1219n.im
            + self.twiddle1.im * x1318n.im
            + self.twiddle13.im * x1417n.im
            + -self.twiddle6.im * x1516n.im;
        let b1318re_a = data[0].re
            + self.twiddle13.re * x130p.re
            + self.twiddle5.re * x229p.re
            + self.twiddle8.re * x328p.re
            + self.twiddle10.re * x427p.re
            + self.twiddle3.re * x526p.re
            + self.twiddle15.re * x625p.re
            + self.twiddle2.re * x724p.re
            + self.twiddle11.re * x823p.re
            + self.twiddle7.re * x922p.re
            + self.twiddle6.re * x1021p.re
            + self.twiddle12.re * x1120p.re
            + self.twiddle1.re * x1219p.re
            + self.twiddle14.re * x1318p.re
            + self.twiddle4.re * x1417p.re
            + self.twiddle9.re * x1516p.re;
        let b1318re_b = self.twiddle13.im * x130n.im
            + -self.twiddle5.im * x229n.im
            + self.twiddle8.im * x328n.im
            + -self.twiddle10.im * x427n.im
            + self.twiddle3.im * x526n.im
            + -self.twiddle15.im * x625n.im
            + -self.twiddle2.im * x724n.im
            + self.twiddle11.im * x823n.im
            + -self.twiddle7.im * x922n.im
            + self.twiddle6.im * x1021n.im
            + -self.twiddle12.im * x1120n.im
            + self.twiddle1.im * x1219n.im
            + self.twiddle14.im * x1318n.im
            + -self.twiddle4.im * x1417n.im
            + self.twiddle9.im * x1516n.im;
        let b1417re_a = data[0].re
            + self.twiddle14.re * x130p.re
            + self.twiddle3.re * x229p.re
            + self.twiddle11.re * x328p.re
            + self.twiddle6.re * x427p.re
            + self.twiddle8.re * x526p.re
            + self.twiddle9.re * x625p.re
            + self.twiddle5.re * x724p.re
            + self.twiddle12.re * x823p.re
            + self.twiddle2.re * x922p.re
            + self.twiddle15.re * x1021p.re
            + self.twiddle1.re * x1120p.re
            + self.twiddle13.re * x1219p.re
            + self.twiddle4.re * x1318p.re
            + self.twiddle10.re * x1417p.re
            + self.twiddle7.re * x1516p.re;
        let b1417re_b = self.twiddle14.im * x130n.im
            + -self.twiddle3.im * x229n.im
            + self.twiddle11.im * x328n.im
            + -self.twiddle6.im * x427n.im
            + self.twiddle8.im * x526n.im
            + -self.twiddle9.im * x625n.im
            + self.twiddle5.im * x724n.im
            + -self.twiddle12.im * x823n.im
            + self.twiddle2.im * x922n.im
            + -self.twiddle15.im * x1021n.im
            + -self.twiddle1.im * x1120n.im
            + self.twiddle13.im * x1219n.im
            + -self.twiddle4.im * x1318n.im
            + self.twiddle10.im * x1417n.im
            + -self.twiddle7.im * x1516n.im;
        let b1516re_a = data[0].re
            + self.twiddle15.re * x130p.re
            + self.twiddle1.re * x229p.re
            + self.twiddle14.re * x328p.re
            + self.twiddle2.re * x427p.re
            + self.twiddle13.re * x526p.re
            + self.twiddle3.re * x625p.re
            + self.twiddle12.re * x724p.re
            + self.twiddle4.re * x823p.re
            + self.twiddle11.re * x922p.re
            + self.twiddle5.re * x1021p.re
            + self.twiddle10.re * x1120p.re
            + self.twiddle6.re * x1219p.re
            + self.twiddle9.re * x1318p.re
            + self.twiddle7.re * x1417p.re
            + self.twiddle8.re * x1516p.re;
        let b1516re_b = self.twiddle15.im * x130n.im
            + -self.twiddle1.im * x229n.im
            + self.twiddle14.im * x328n.im
            + -self.twiddle2.im * x427n.im
            + self.twiddle13.im * x526n.im
            + -self.twiddle3.im * x625n.im
            + self.twiddle12.im * x724n.im
            + -self.twiddle4.im * x823n.im
            + self.twiddle11.im * x922n.im
            + -self.twiddle5.im * x1021n.im
            + self.twiddle10.im * x1120n.im
            + -self.twiddle6.im * x1219n.im
            + self.twiddle9.im * x1318n.im
            + -self.twiddle7.im * x1417n.im
            + self.twiddle8.im * x1516n.im;

        let b130im_a = data[0].im
            + self.twiddle1.re * x130p.im
            + self.twiddle2.re * x229p.im
            + self.twiddle3.re * x328p.im
            + self.twiddle4.re * x427p.im
            + self.twiddle5.re * x526p.im
            + self.twiddle6.re * x625p.im
            + self.twiddle7.re * x724p.im
            + self.twiddle8.re * x823p.im
            + self.twiddle9.re * x922p.im
            + self.twiddle10.re * x1021p.im
            + self.twiddle11.re * x1120p.im
            + self.twiddle12.re * x1219p.im
            + self.twiddle13.re * x1318p.im
            + self.twiddle14.re * x1417p.im
            + self.twiddle15.re * x1516p.im;
        let b130im_b = self.twiddle1.im * x130n.re
            + self.twiddle2.im * x229n.re
            + self.twiddle3.im * x328n.re
            + self.twiddle4.im * x427n.re
            + self.twiddle5.im * x526n.re
            + self.twiddle6.im * x625n.re
            + self.twiddle7.im * x724n.re
            + self.twiddle8.im * x823n.re
            + self.twiddle9.im * x922n.re
            + self.twiddle10.im * x1021n.re
            + self.twiddle11.im * x1120n.re
            + self.twiddle12.im * x1219n.re
            + self.twiddle13.im * x1318n.re
            + self.twiddle14.im * x1417n.re
            + self.twiddle15.im * x1516n.re;
        let b229im_a = data[0].im
            + self.twiddle2.re * x130p.im
            + self.twiddle4.re * x229p.im
            + self.twiddle6.re * x328p.im
            + self.twiddle8.re * x427p.im
            + self.twiddle10.re * x526p.im
            + self.twiddle12.re * x625p.im
            + self.twiddle14.re * x724p.im
            + self.twiddle15.re * x823p.im
            + self.twiddle13.re * x922p.im
            + self.twiddle11.re * x1021p.im
            + self.twiddle9.re * x1120p.im
            + self.twiddle7.re * x1219p.im
            + self.twiddle5.re * x1318p.im
            + self.twiddle3.re * x1417p.im
            + self.twiddle1.re * x1516p.im;
        let b229im_b = self.twiddle2.im * x130n.re
            + self.twiddle4.im * x229n.re
            + self.twiddle6.im * x328n.re
            + self.twiddle8.im * x427n.re
            + self.twiddle10.im * x526n.re
            + self.twiddle12.im * x625n.re
            + self.twiddle14.im * x724n.re
            + -self.twiddle15.im * x823n.re
            + -self.twiddle13.im * x922n.re
            + -self.twiddle11.im * x1021n.re
            + -self.twiddle9.im * x1120n.re
            + -self.twiddle7.im * x1219n.re
            + -self.twiddle5.im * x1318n.re
            + -self.twiddle3.im * x1417n.re
            + -self.twiddle1.im * x1516n.re;
        let b328im_a = data[0].im
            + self.twiddle3.re * x130p.im
            + self.twiddle6.re * x229p.im
            + self.twiddle9.re * x328p.im
            + self.twiddle12.re * x427p.im
            + self.twiddle15.re * x526p.im
            + self.twiddle13.re * x625p.im
            + self.twiddle10.re * x724p.im
            + self.twiddle7.re * x823p.im
            + self.twiddle4.re * x922p.im
            + self.twiddle1.re * x1021p.im
            + self.twiddle2.re * x1120p.im
            + self.twiddle5.re * x1219p.im
            + self.twiddle8.re * x1318p.im
            + self.twiddle11.re * x1417p.im
            + self.twiddle14.re * x1516p.im;
        let b328im_b = self.twiddle3.im * x130n.re
            + self.twiddle6.im * x229n.re
            + self.twiddle9.im * x328n.re
            + self.twiddle12.im * x427n.re
            + self.twiddle15.im * x526n.re
            + -self.twiddle13.im * x625n.re
            + -self.twiddle10.im * x724n.re
            + -self.twiddle7.im * x823n.re
            + -self.twiddle4.im * x922n.re
            + -self.twiddle1.im * x1021n.re
            + self.twiddle2.im * x1120n.re
            + self.twiddle5.im * x1219n.re
            + self.twiddle8.im * x1318n.re
            + self.twiddle11.im * x1417n.re
            + self.twiddle14.im * x1516n.re;
        let b427im_a = data[0].im
            + self.twiddle4.re * x130p.im
            + self.twiddle8.re * x229p.im
            + self.twiddle12.re * x328p.im
            + self.twiddle15.re * x427p.im
            + self.twiddle11.re * x526p.im
            + self.twiddle7.re * x625p.im
            + self.twiddle3.re * x724p.im
            + self.twiddle1.re * x823p.im
            + self.twiddle5.re * x922p.im
            + self.twiddle9.re * x1021p.im
            + self.twiddle13.re * x1120p.im
            + self.twiddle14.re * x1219p.im
            + self.twiddle10.re * x1318p.im
            + self.twiddle6.re * x1417p.im
            + self.twiddle2.re * x1516p.im;
        let b427im_b = self.twiddle4.im * x130n.re
            + self.twiddle8.im * x229n.re
            + self.twiddle12.im * x328n.re
            + -self.twiddle15.im * x427n.re
            + -self.twiddle11.im * x526n.re
            + -self.twiddle7.im * x625n.re
            + -self.twiddle3.im * x724n.re
            + self.twiddle1.im * x823n.re
            + self.twiddle5.im * x922n.re
            + self.twiddle9.im * x1021n.re
            + self.twiddle13.im * x1120n.re
            + -self.twiddle14.im * x1219n.re
            + -self.twiddle10.im * x1318n.re
            + -self.twiddle6.im * x1417n.re
            + -self.twiddle2.im * x1516n.re;
        let b526im_a = data[0].im
            + self.twiddle5.re * x130p.im
            + self.twiddle10.re * x229p.im
            + self.twiddle15.re * x328p.im
            + self.twiddle11.re * x427p.im
            + self.twiddle6.re * x526p.im
            + self.twiddle1.re * x625p.im
            + self.twiddle4.re * x724p.im
            + self.twiddle9.re * x823p.im
            + self.twiddle14.re * x922p.im
            + self.twiddle12.re * x1021p.im
            + self.twiddle7.re * x1120p.im
            + self.twiddle2.re * x1219p.im
            + self.twiddle3.re * x1318p.im
            + self.twiddle8.re * x1417p.im
            + self.twiddle13.re * x1516p.im;
        let b526im_b = self.twiddle5.im * x130n.re
            + self.twiddle10.im * x229n.re
            + self.twiddle15.im * x328n.re
            + -self.twiddle11.im * x427n.re
            + -self.twiddle6.im * x526n.re
            + -self.twiddle1.im * x625n.re
            + self.twiddle4.im * x724n.re
            + self.twiddle9.im * x823n.re
            + self.twiddle14.im * x922n.re
            + -self.twiddle12.im * x1021n.re
            + -self.twiddle7.im * x1120n.re
            + -self.twiddle2.im * x1219n.re
            + self.twiddle3.im * x1318n.re
            + self.twiddle8.im * x1417n.re
            + self.twiddle13.im * x1516n.re;
        let b625im_a = data[0].im
            + self.twiddle6.re * x130p.im
            + self.twiddle12.re * x229p.im
            + self.twiddle13.re * x328p.im
            + self.twiddle7.re * x427p.im
            + self.twiddle1.re * x526p.im
            + self.twiddle5.re * x625p.im
            + self.twiddle11.re * x724p.im
            + self.twiddle14.re * x823p.im
            + self.twiddle8.re * x922p.im
            + self.twiddle2.re * x1021p.im
            + self.twiddle4.re * x1120p.im
            + self.twiddle10.re * x1219p.im
            + self.twiddle15.re * x1318p.im
            + self.twiddle9.re * x1417p.im
            + self.twiddle3.re * x1516p.im;
        let b625im_b = self.twiddle6.im * x130n.re
            + self.twiddle12.im * x229n.re
            + -self.twiddle13.im * x328n.re
            + -self.twiddle7.im * x427n.re
            + -self.twiddle1.im * x526n.re
            + self.twiddle5.im * x625n.re
            + self.twiddle11.im * x724n.re
            + -self.twiddle14.im * x823n.re
            + -self.twiddle8.im * x922n.re
            + -self.twiddle2.im * x1021n.re
            + self.twiddle4.im * x1120n.re
            + self.twiddle10.im * x1219n.re
            + -self.twiddle15.im * x1318n.re
            + -self.twiddle9.im * x1417n.re
            + -self.twiddle3.im * x1516n.re;
        let b724im_a = data[0].im
            + self.twiddle7.re * x130p.im
            + self.twiddle14.re * x229p.im
            + self.twiddle10.re * x328p.im
            + self.twiddle3.re * x427p.im
            + self.twiddle4.re * x526p.im
            + self.twiddle11.re * x625p.im
            + self.twiddle13.re * x724p.im
            + self.twiddle6.re * x823p.im
            + self.twiddle1.re * x922p.im
            + self.twiddle8.re * x1021p.im
            + self.twiddle15.re * x1120p.im
            + self.twiddle9.re * x1219p.im
            + self.twiddle2.re * x1318p.im
            + self.twiddle5.re * x1417p.im
            + self.twiddle12.re * x1516p.im;
        let b724im_b = self.twiddle7.im * x130n.re
            + self.twiddle14.im * x229n.re
            + -self.twiddle10.im * x328n.re
            + -self.twiddle3.im * x427n.re
            + self.twiddle4.im * x526n.re
            + self.twiddle11.im * x625n.re
            + -self.twiddle13.im * x724n.re
            + -self.twiddle6.im * x823n.re
            + self.twiddle1.im * x922n.re
            + self.twiddle8.im * x1021n.re
            + self.twiddle15.im * x1120n.re
            + -self.twiddle9.im * x1219n.re
            + -self.twiddle2.im * x1318n.re
            + self.twiddle5.im * x1417n.re
            + self.twiddle12.im * x1516n.re;
        let b823im_a = data[0].im
            + self.twiddle8.re * x130p.im
            + self.twiddle15.re * x229p.im
            + self.twiddle7.re * x328p.im
            + self.twiddle1.re * x427p.im
            + self.twiddle9.re * x526p.im
            + self.twiddle14.re * x625p.im
            + self.twiddle6.re * x724p.im
            + self.twiddle2.re * x823p.im
            + self.twiddle10.re * x922p.im
            + self.twiddle13.re * x1021p.im
            + self.twiddle5.re * x1120p.im
            + self.twiddle3.re * x1219p.im
            + self.twiddle11.re * x1318p.im
            + self.twiddle12.re * x1417p.im
            + self.twiddle4.re * x1516p.im;
        let b823im_b = self.twiddle8.im * x130n.re
            + -self.twiddle15.im * x229n.re
            + -self.twiddle7.im * x328n.re
            + self.twiddle1.im * x427n.re
            + self.twiddle9.im * x526n.re
            + -self.twiddle14.im * x625n.re
            + -self.twiddle6.im * x724n.re
            + self.twiddle2.im * x823n.re
            + self.twiddle10.im * x922n.re
            + -self.twiddle13.im * x1021n.re
            + -self.twiddle5.im * x1120n.re
            + self.twiddle3.im * x1219n.re
            + self.twiddle11.im * x1318n.re
            + -self.twiddle12.im * x1417n.re
            + -self.twiddle4.im * x1516n.re;
        let b922im_a = data[0].im
            + self.twiddle9.re * x130p.im
            + self.twiddle13.re * x229p.im
            + self.twiddle4.re * x328p.im
            + self.twiddle5.re * x427p.im
            + self.twiddle14.re * x526p.im
            + self.twiddle8.re * x625p.im
            + self.twiddle1.re * x724p.im
            + self.twiddle10.re * x823p.im
            + self.twiddle12.re * x922p.im
            + self.twiddle3.re * x1021p.im
            + self.twiddle6.re * x1120p.im
            + self.twiddle15.re * x1219p.im
            + self.twiddle7.re * x1318p.im
            + self.twiddle2.re * x1417p.im
            + self.twiddle11.re * x1516p.im;
        let b922im_b = self.twiddle9.im * x130n.re
            + -self.twiddle13.im * x229n.re
            + -self.twiddle4.im * x328n.re
            + self.twiddle5.im * x427n.re
            + self.twiddle14.im * x526n.re
            + -self.twiddle8.im * x625n.re
            + self.twiddle1.im * x724n.re
            + self.twiddle10.im * x823n.re
            + -self.twiddle12.im * x922n.re
            + -self.twiddle3.im * x1021n.re
            + self.twiddle6.im * x1120n.re
            + self.twiddle15.im * x1219n.re
            + -self.twiddle7.im * x1318n.re
            + self.twiddle2.im * x1417n.re
            + self.twiddle11.im * x1516n.re;
        let b1021im_a = data[0].im
            + self.twiddle10.re * x130p.im
            + self.twiddle11.re * x229p.im
            + self.twiddle1.re * x328p.im
            + self.twiddle9.re * x427p.im
            + self.twiddle12.re * x526p.im
            + self.twiddle2.re * x625p.im
            + self.twiddle8.re * x724p.im
            + self.twiddle13.re * x823p.im
            + self.twiddle3.re * x922p.im
            + self.twiddle7.re * x1021p.im
            + self.twiddle14.re * x1120p.im
            + self.twiddle4.re * x1219p.im
            + self.twiddle6.re * x1318p.im
            + self.twiddle15.re * x1417p.im
            + self.twiddle5.re * x1516p.im;
        let b1021im_b = self.twiddle10.im * x130n.re
            + -self.twiddle11.im * x229n.re
            + -self.twiddle1.im * x328n.re
            + self.twiddle9.im * x427n.re
            + -self.twiddle12.im * x526n.re
            + -self.twiddle2.im * x625n.re
            + self.twiddle8.im * x724n.re
            + -self.twiddle13.im * x823n.re
            + -self.twiddle3.im * x922n.re
            + self.twiddle7.im * x1021n.re
            + -self.twiddle14.im * x1120n.re
            + -self.twiddle4.im * x1219n.re
            + self.twiddle6.im * x1318n.re
            + -self.twiddle15.im * x1417n.re
            + -self.twiddle5.im * x1516n.re;
        let b1120im_a = data[0].im
            + self.twiddle11.re * x130p.im
            + self.twiddle9.re * x229p.im
            + self.twiddle2.re * x328p.im
            + self.twiddle13.re * x427p.im
            + self.twiddle7.re * x526p.im
            + self.twiddle4.re * x625p.im
            + self.twiddle15.re * x724p.im
            + self.twiddle5.re * x823p.im
            + self.twiddle6.re * x922p.im
            + self.twiddle14.re * x1021p.im
            + self.twiddle3.re * x1120p.im
            + self.twiddle8.re * x1219p.im
            + self.twiddle12.re * x1318p.im
            + self.twiddle1.re * x1417p.im
            + self.twiddle10.re * x1516p.im;
        let b1120im_b = self.twiddle11.im * x130n.re
            + -self.twiddle9.im * x229n.re
            + self.twiddle2.im * x328n.re
            + self.twiddle13.im * x427n.re
            + -self.twiddle7.im * x526n.re
            + self.twiddle4.im * x625n.re
            + self.twiddle15.im * x724n.re
            + -self.twiddle5.im * x823n.re
            + self.twiddle6.im * x922n.re
            + -self.twiddle14.im * x1021n.re
            + -self.twiddle3.im * x1120n.re
            + self.twiddle8.im * x1219n.re
            + -self.twiddle12.im * x1318n.re
            + -self.twiddle1.im * x1417n.re
            + self.twiddle10.im * x1516n.re;
        let b1219im_a = data[0].im
            + self.twiddle12.re * x130p.im
            + self.twiddle7.re * x229p.im
            + self.twiddle5.re * x328p.im
            + self.twiddle14.re * x427p.im
            + self.twiddle2.re * x526p.im
            + self.twiddle10.re * x625p.im
            + self.twiddle9.re * x724p.im
            + self.twiddle3.re * x823p.im
            + self.twiddle15.re * x922p.im
            + self.twiddle4.re * x1021p.im
            + self.twiddle8.re * x1120p.im
            + self.twiddle11.re * x1219p.im
            + self.twiddle1.re * x1318p.im
            + self.twiddle13.re * x1417p.im
            + self.twiddle6.re * x1516p.im;
        let b1219im_b = self.twiddle12.im * x130n.re
            + -self.twiddle7.im * x229n.re
            + self.twiddle5.im * x328n.re
            + -self.twiddle14.im * x427n.re
            + -self.twiddle2.im * x526n.re
            + self.twiddle10.im * x625n.re
            + -self.twiddle9.im * x724n.re
            + self.twiddle3.im * x823n.re
            + self.twiddle15.im * x922n.re
            + -self.twiddle4.im * x1021n.re
            + self.twiddle8.im * x1120n.re
            + -self.twiddle11.im * x1219n.re
            + self.twiddle1.im * x1318n.re
            + self.twiddle13.im * x1417n.re
            + -self.twiddle6.im * x1516n.re;
        let b1318im_a = data[0].im
            + self.twiddle13.re * x130p.im
            + self.twiddle5.re * x229p.im
            + self.twiddle8.re * x328p.im
            + self.twiddle10.re * x427p.im
            + self.twiddle3.re * x526p.im
            + self.twiddle15.re * x625p.im
            + self.twiddle2.re * x724p.im
            + self.twiddle11.re * x823p.im
            + self.twiddle7.re * x922p.im
            + self.twiddle6.re * x1021p.im
            + self.twiddle12.re * x1120p.im
            + self.twiddle1.re * x1219p.im
            + self.twiddle14.re * x1318p.im
            + self.twiddle4.re * x1417p.im
            + self.twiddle9.re * x1516p.im;
        let b1318im_b = self.twiddle13.im * x130n.re
            + -self.twiddle5.im * x229n.re
            + self.twiddle8.im * x328n.re
            + -self.twiddle10.im * x427n.re
            + self.twiddle3.im * x526n.re
            + -self.twiddle15.im * x625n.re
            + -self.twiddle2.im * x724n.re
            + self.twiddle11.im * x823n.re
            + -self.twiddle7.im * x922n.re
            + self.twiddle6.im * x1021n.re
            + -self.twiddle12.im * x1120n.re
            + self.twiddle1.im * x1219n.re
            + self.twiddle14.im * x1318n.re
            + -self.twiddle4.im * x1417n.re
            + self.twiddle9.im * x1516n.re;
        let b1417im_a = data[0].im
            + self.twiddle14.re * x130p.im
            + self.twiddle3.re * x229p.im
            + self.twiddle11.re * x328p.im
            + self.twiddle6.re * x427p.im
            + self.twiddle8.re * x526p.im
            + self.twiddle9.re * x625p.im
            + self.twiddle5.re * x724p.im
            + self.twiddle12.re * x823p.im
            + self.twiddle2.re * x922p.im
            + self.twiddle15.re * x1021p.im
            + self.twiddle1.re * x1120p.im
            + self.twiddle13.re * x1219p.im
            + self.twiddle4.re * x1318p.im
            + self.twiddle10.re * x1417p.im
            + self.twiddle7.re * x1516p.im;
        let b1417im_b = self.twiddle14.im * x130n.re
            + -self.twiddle3.im * x229n.re
            + self.twiddle11.im * x328n.re
            + -self.twiddle6.im * x427n.re
            + self.twiddle8.im * x526n.re
            + -self.twiddle9.im * x625n.re
            + self.twiddle5.im * x724n.re
            + -self.twiddle12.im * x823n.re
            + self.twiddle2.im * x922n.re
            + -self.twiddle15.im * x1021n.re
            + -self.twiddle1.im * x1120n.re
            + self.twiddle13.im * x1219n.re
            + -self.twiddle4.im * x1318n.re
            + self.twiddle10.im * x1417n.re
            + -self.twiddle7.im * x1516n.re;
        let b1516im_a = data[0].im
            + self.twiddle15.re * x130p.im
            + self.twiddle1.re * x229p.im
            + self.twiddle14.re * x328p.im
            + self.twiddle2.re * x427p.im
            + self.twiddle13.re * x526p.im
            + self.twiddle3.re * x625p.im
            + self.twiddle12.re * x724p.im
            + self.twiddle4.re * x823p.im
            + self.twiddle11.re * x922p.im
            + self.twiddle5.re * x1021p.im
            + self.twiddle10.re * x1120p.im
            + self.twiddle6.re * x1219p.im
            + self.twiddle9.re * x1318p.im
            + self.twiddle7.re * x1417p.im
            + self.twiddle8.re * x1516p.im;
        let b1516im_b = self.twiddle15.im * x130n.re
            + -self.twiddle1.im * x229n.re
            + self.twiddle14.im * x328n.re
            + -self.twiddle2.im * x427n.re
            + self.twiddle13.im * x526n.re
            + -self.twiddle3.im * x625n.re
            + self.twiddle12.im * x724n.re
            + -self.twiddle4.im * x823n.re
            + self.twiddle11.im * x922n.re
            + -self.twiddle5.im * x1021n.re
            + self.twiddle10.im * x1120n.re
            + -self.twiddle6.im * x1219n.re
            + self.twiddle9.im * x1318n.re
            + -self.twiddle7.im * x1417n.re
            + self.twiddle8.im * x1516n.re;

        let out1re = b130re_a - b130re_b;
        let out1im = b130im_a + b130im_b;
        let out2re = b229re_a - b229re_b;
        let out2im = b229im_a + b229im_b;
        let out3re = b328re_a - b328re_b;
        let out3im = b328im_a + b328im_b;
        let out4re = b427re_a - b427re_b;
        let out4im = b427im_a + b427im_b;
        let out5re = b526re_a - b526re_b;
        let out5im = b526im_a + b526im_b;
        let out6re = b625re_a - b625re_b;
        let out6im = b625im_a + b625im_b;
        let out7re = b724re_a - b724re_b;
        let out7im = b724im_a + b724im_b;
        let out8re = b823re_a - b823re_b;
        let out8im = b823im_a + b823im_b;
        let out9re = b922re_a - b922re_b;
        let out9im = b922im_a + b922im_b;
        let out10re = b1021re_a - b1021re_b;
        let out10im = b1021im_a + b1021im_b;
        let out11re = b1120re_a - b1120re_b;
        let out11im = b1120im_a + b1120im_b;
        let out12re = b1219re_a - b1219re_b;
        let out12im = b1219im_a + b1219im_b;
        let out13re = b1318re_a - b1318re_b;
        let out13im = b1318im_a + b1318im_b;
        let out14re = b1417re_a - b1417re_b;
        let out14im = b1417im_a + b1417im_b;
        let out15re = b1516re_a - b1516re_b;
        let out15im = b1516im_a + b1516im_b;
        let out16re = b1516re_a + b1516re_b;
        let out16im = b1516im_a - b1516im_b;
        let out17re = b1417re_a + b1417re_b;
        let out17im = b1417im_a - b1417im_b;
        let out18re = b1318re_a + b1318re_b;
        let out18im = b1318im_a - b1318im_b;
        let out19re = b1219re_a + b1219re_b;
        let out19im = b1219im_a - b1219im_b;
        let out20re = b1120re_a + b1120re_b;
        let out20im = b1120im_a - b1120im_b;
        let out21re = b1021re_a + b1021re_b;
        let out21im = b1021im_a - b1021im_b;
        let out22re = b922re_a + b922re_b;
        let out22im = b922im_a - b922im_b;
        let out23re = b823re_a + b823re_b;
        let out23im = b823im_a - b823im_b;
        let out24re = b724re_a + b724re_b;
        let out24im = b724im_a - b724im_b;
        let out25re = b625re_a + b625re_b;
        let out25im = b625im_a - b625im_b;
        let out26re = b526re_a + b526re_b;
        let out26im = b526im_a - b526im_b;
        let out27re = b427re_a + b427re_b;
        let out27im = b427im_a - b427im_b;
        let out28re = b328re_a + b328re_b;
        let out28im = b328im_a - b328im_b;
        let out29re = b229re_a + b229re_b;
        let out29im = b229im_a - b229im_b;
        let out30re = b130re_a + b130re_b;
        let out30im = b130im_a - b130im_b;
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
            Complex {
                re: out7re,
                im: out7im,
            },
            Complex {
                re: out8re,
                im: out8im,
            },
            Complex {
                re: out9re,
                im: out9im,
            },
            Complex {
                re: out10re,
                im: out10im,
            },
            Complex {
                re: out11re,
                im: out11im,
            },
            Complex {
                re: out12re,
                im: out12im,
            },
            Complex {
                re: out13re,
                im: out13im,
            },
            Complex {
                re: out14re,
                im: out14im,
            },
            Complex {
                re: out15re,
                im: out15im,
            },
            Complex {
                re: out16re,
                im: out16im,
            },
            Complex {
                re: out17re,
                im: out17im,
            },
            Complex {
                re: out18re,
                im: out18im,
            },
            Complex {
                re: out19re,
                im: out19im,
            },
            Complex {
                re: out20re,
                im: out20im,
            },
            Complex {
                re: out21re,
                im: out21im,
            },
            Complex {
                re: out22re,
                im: out22im,
            },
            Complex {
                re: out23re,
                im: out23im,
            },
            Complex {
                re: out24re,
                im: out24im,
            },
            Complex {
                re: out25re,
                im: out25im,
            },
            Complex {
                re: out26re,
                im: out26im,
            },
            Complex {
                re: out27re,
                im: out27im,
            },
            Complex {
                re: out28re,
                im: out28im,
            },
            Complex {
                re: out29re,
                im: out29im,
            },
            Complex {
                re: out30re,
                im: out30im,
            },
        ]
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_parallel_fft;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(FftDirection::Forward);
                check_parallel_fft::<Complex<f32>, $size>(&butterfly, FftDirection::Forward);

                let butterfly_inverse = $struct_name::new(FftDirection::Inverse);
                check_parallel_fft::<Complex<f32>, $size>(
                    &butterfly_inverse,
                    FftDirection::Inverse,
                );
            }
        };
    }

    test_butterfly_func!(test_scalar_parallel_butterfly2, ScalarParallelButterfly2, 2);
    test_butterfly_func!(test_scalar_parallel_butterfly3, ScalarParallelButterfly3, 3);
    test_butterfly_func!(test_scalar_parallel_butterfly4, ScalarParallelButterfly4, 4);
    test_butterfly_func!(test_scalar_parallel_butterfly5, ScalarParallelButterfly5, 5);
    test_butterfly_func!(test_scalar_parallel_butterfly6, ScalarParallelButterfly6, 6);
    test_butterfly_func!(test_scalar_parallel_butterfly7, ScalarParallelButterfly7, 7);
}
