use num_complex::Complex;

use crate::{common::FftNum, FftDirection};

use crate::array_utils;
use crate::array_utils::{RawSlice, RawSliceMut};
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::twiddles;
use crate::{Direction, Fft, Length};

#[allow(unused)]
macro_rules! boilerplate_fft_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            #[inline(always)]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(RawSlice::new(buffer), RawSliceMut::new(buffer));
            }
        }
        impl<T: FftNum> Fft<T> for $struct_name<T> {
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                if input.len() < self.len() || output.len() != input.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                    return; // Unreachable, because fft_error_outofplace asserts, but it helps codegen to put it here
                }

                let result = array_utils::iter_chunks_zipped(
                    input,
                    output,
                    self.len(),
                    |in_chunk, out_chunk| {
                        unsafe {
                            self.perform_fft_contiguous(
                                RawSlice::new(in_chunk),
                                RawSliceMut::new(out_chunk),
                            )
                        };
                    },
                );

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_outofplace(self.len(), input.len(), output.len(), 0, 0);
                }
            }
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                if buffer.len() < self.len() {
                    // We want to trigger a panic, but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_inplace(self.len(), buffer.len(), 0, 0);
                    return; // Unreachable, because fft_error_inplace asserts, but it helps codegen to put it here
                }

                let result = array_utils::iter_chunks(buffer, self.len(), |chunk| unsafe {
                    self.perform_fft_butterfly(chunk)
                });

                if result.is_err() {
                    // We want to trigger a panic, because the buffer sizes weren't cleanly divisible by the FFT size,
                    // but we want to avoid doing it in this function to reduce code size, so call a function marked cold and inline(never) that will do it for us
                    fft_error_inplace(self.len(), buffer.len(), 0, 0);
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                0
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len
            }
        }
        impl<T> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                $direction_fn(self)
            }
        }
    };
}

pub struct Butterfly1<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FftNum> Butterfly1<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T: FftNum> Fft<T> for Butterfly1<T> {
    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        output.copy_from_slice(&input);
    }

    fn process_with_scratch(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {}

    fn get_inplace_scratch_len(&self) -> usize {
        0
    }

    fn get_outofplace_scratch_len(&self) -> usize {
        0
    }
}
impl<T> Length for Butterfly1<T> {
    fn len(&self) -> usize {
        1
    }
}
impl<T> Direction for Butterfly1<T> {
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

pub struct Butterfly2<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_butterfly!(Butterfly2, 2, |this: &Butterfly2<_>| this.direction);
impl<T: FftNum> Butterfly2<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_strided(left: &mut Complex<T>, right: &mut Complex<T>) {
        let temp = *left + *right;

        *right = *left - *right;
        *left = temp;
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let value0 = input.load(0);
        let value1 = input.load(1);
        output.store(value0 + value1, 0);
        output.store(value0 - value1, 1);
    }
}

pub struct Butterfly3<T> {
    pub twiddle: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly3, 3, |this: &Butterfly3<_>| this.direction);
impl<T: FftNum> Butterfly3<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle: twiddles::compute_twiddle(1, 3, direction),
            direction,
        }
    }
    #[inline(always)]
    pub fn direction_of(fft: &Butterfly3<T>) -> Self {
        Self {
            twiddle: fft.twiddle.conj(),
            direction: fft.direction.opposite_direction(),
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_strided(
        &self,
        val0: &mut Complex<T>,
        val1: &mut Complex<T>,
        val2: &mut Complex<T>,
    ) {
        let xp = *val1 + *val2;
        let xn = *val1 - *val2;
        let sum = *val0 + xp;

        let temp_a = *val0
            + Complex {
                re: self.twiddle.re * xp.re,
                im: self.twiddle.re * xp.im,
            };
        let temp_b = Complex {
            re: -self.twiddle.im * xn.im,
            im: self.twiddle.im * xn.re,
        };

        *val0 = sum;
        *val1 = temp_a + temp_b;
        *val2 = temp_a - temp_b;
    }
    #[inline(always)]
    pub unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let xp = input.load(1) + input.load(2);
        let xn = input.load(1) - input.load(2);
        let sum = input.load(0) + xp;

        let temp_a = input.load(0)
            + Complex {
                re: self.twiddle.re * xp.re,
                im: self.twiddle.re * xp.im,
            };
        let temp_b = Complex {
            re: -self.twiddle.im * xn.im,
            im: self.twiddle.im * xn.re,
        };

        output.store(sum, 0);
        output.store(temp_a + temp_b, 1);
        output.store(temp_a - temp_b, 2);
    }
}

pub struct Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_butterfly!(Butterfly4, 4, |this: &Butterfly4<_>| this.direction);
impl<T: FftNum> Butterfly4<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose, which we're skipping because we're just going to perform non-contiguous FFTs
        let mut value0 = input.load(0);
        let mut value1 = input.load(1);
        let mut value2 = input.load(2);
        let mut value3 = input.load(3);

        // step 2: column FFTs
        Butterfly2::perform_fft_strided(&mut value0, &mut value2);
        Butterfly2::perform_fft_strided(&mut value1, &mut value3);

        // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
        value3 = twiddles::rotate_90(value3, self.direction);

        // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

        // step 5: row FFTs
        Butterfly2::perform_fft_strided(&mut value0, &mut value1);
        Butterfly2::perform_fft_strided(&mut value2, &mut value3);

        // step 6: transpose by swapping index 1 and 2
        output.store(value0, 0);
        output.store(value2, 1);
        output.store(value1, 2);
        output.store(value3, 3);
    }
}

pub struct Butterfly5<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly5, 5, |this: &Butterfly5<_>| this.direction);
impl<T: FftNum> Butterfly5<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 5, direction),
            twiddle2: twiddles::compute_twiddle(2, 5, direction),
            direction,
        }
    }

    #[inline(never)] // refusing to inline this code reduces code size, and doesn't hurt performance
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // let mut outer = Butterfly2::perform_fft_array([input.load(1), input.load(4)]);
        // let mut inner = Butterfly2::perform_fft_array([input.load(2), input.load(3)]);
        // let input0 = input.load(0);

        // output.store(input0 + outer[0] + inner[0], 0);

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

        //     output.store(input0 + output1, 1);
        //     output.store(input0 + output4, 4);
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

        let x14p = input.load(1) + input.load(4);
        let x14n = input.load(1) - input.load(4);
        let x23p = input.load(2) + input.load(3);
        let x23n = input.load(2) - input.load(3);
        let sum = input.load(0) + x14p + x23p;
        let b14re_a = input.load(0).re + self.twiddle1.re * x14p.re + self.twiddle2.re * x23p.re;
        let b14re_b = self.twiddle1.im * x14n.im + self.twiddle2.im * x23n.im;
        let b23re_a = input.load(0).re + self.twiddle2.re * x14p.re + self.twiddle1.re * x23p.re;
        let b23re_b = self.twiddle2.im * x14n.im + -self.twiddle1.im * x23n.im;

        let b14im_a = input.load(0).im + self.twiddle1.re * x14p.im + self.twiddle2.re * x23p.im;
        let b14im_b = self.twiddle1.im * x14n.re + self.twiddle2.im * x23n.re;
        let b23im_a = input.load(0).im + self.twiddle2.re * x14p.im + self.twiddle1.re * x23p.im;
        let b23im_b = self.twiddle2.im * x14n.re + -self.twiddle1.im * x23n.re;

        let out1re = b14re_a - b14re_b;
        let out1im = b14im_a + b14im_b;
        let out2re = b23re_a - b23re_b;
        let out2im = b23im_a + b23im_b;
        let out3re = b23re_a + b23re_b;
        let out3im = b23im_a - b23im_b;
        let out4re = b14re_a + b14re_b;
        let out4im = b14im_a - b14im_b;
        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
    }
}

pub struct Butterfly6<T> {
    butterfly3: Butterfly3<T>,
}
boilerplate_fft_butterfly!(Butterfly6, 6, |this: &Butterfly6<_>| this
    .butterfly3
    .fft_direction());
impl<T: FftNum> Butterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly3: Butterfly3::new(direction),
        }
    }
    #[inline(always)]
    pub fn direction_of(fft: &Butterfly6<T>) -> Self {
        Self {
            butterfly3: Butterfly3::direction_of(&fft.butterfly3),
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        //since GCD(2,3) == 1 we're going to hardcode a step of the Good-Thomas algorithm to avoid twiddle factors

        // step 1: reorder the input directly into the scratch. normally there's a whole thing to compute this ordering
        //but thankfully we can just precompute it and hardcode it
        let mut scratch_a = [input.load(0), input.load(2), input.load(4)];

        let mut scratch_b = [input.load(3), input.load(5), input.load(1)];

        // step 2: column FFTs
        self.butterfly3.perform_fft_butterfly(&mut scratch_a);
        self.butterfly3.perform_fft_butterfly(&mut scratch_b);

        // step 3: apply twiddle factors -- SKIPPED because good-thomas doesn't have twiddle factors :)

        // step 4: SKIPPED because the next FFTs will be non-contiguous

        // step 5: row FFTs
        Butterfly2::perform_fft_strided(&mut scratch_a[0], &mut scratch_b[0]);
        Butterfly2::perform_fft_strided(&mut scratch_a[1], &mut scratch_b[1]);
        Butterfly2::perform_fft_strided(&mut scratch_a[2], &mut scratch_b[2]);

        // step 6: reorder the result back into the buffer. again we would normally have to do an expensive computation
        // but instead we can precompute and hardcode the ordering
        // note that we're also rolling a transpose step into this reorder
        output.store(scratch_a[0], 0);
        output.store(scratch_b[1], 1);
        output.store(scratch_a[2], 2);
        output.store(scratch_b[0], 3);
        output.store(scratch_a[1], 4);
        output.store(scratch_b[2], 5);
    }
}

pub struct Butterfly7<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly7, 7, |this: &Butterfly7<_>| this.direction);
impl<T: FftNum> Butterfly7<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 7, direction),
            twiddle2: twiddles::compute_twiddle(2, 7, direction),
            twiddle3: twiddles::compute_twiddle(3, 7, direction),
            direction,
        }
    }
    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // let mut outer = Butterfly2::perform_fft_array([input.load(1), input.load(6)]);
        // let mut mid   = Butterfly2::perform_fft_array([input.load(2), input.load(5)]);
        // let mut inner = Butterfly2::perform_fft_array([input.load(3), input.load(4)]);
        // let input0 = input.load(0);

        // output.store(input0 + outer[0] + mid[0] + inner[0], 0);

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

        //     output.store(input0 + output1, 1);
        //     output.store(input0 + output6, 6);
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

        //     output.store(input0 + output2, 2);
        //     output.store(input0 + output5, 5);
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

        let x16p = input.load(1) + input.load(6);
        let x16n = input.load(1) - input.load(6);
        let x25p = input.load(2) + input.load(5);
        let x25n = input.load(2) - input.load(5);
        let x34p = input.load(3) + input.load(4);
        let x34n = input.load(3) - input.load(4);
        let sum = input.load(0) + x16p + x25p + x34p;

        let x16re_a = input.load(0).re
            + self.twiddle1.re * x16p.re
            + self.twiddle2.re * x25p.re
            + self.twiddle3.re * x34p.re;
        let x16re_b =
            self.twiddle1.im * x16n.im + self.twiddle2.im * x25n.im + self.twiddle3.im * x34n.im;
        let x25re_a = input.load(0).re
            + self.twiddle1.re * x34p.re
            + self.twiddle2.re * x16p.re
            + self.twiddle3.re * x25p.re;
        let x25re_b =
            -self.twiddle1.im * x34n.im + self.twiddle2.im * x16n.im - self.twiddle3.im * x25n.im;
        let x34re_a = input.load(0).re
            + self.twiddle1.re * x25p.re
            + self.twiddle2.re * x34p.re
            + self.twiddle3.re * x16p.re;
        let x34re_b =
            -self.twiddle1.im * x25n.im + self.twiddle2.im * x34n.im + self.twiddle3.im * x16n.im;
        let x16im_a = input.load(0).im
            + self.twiddle1.re * x16p.im
            + self.twiddle2.re * x25p.im
            + self.twiddle3.re * x34p.im;
        let x16im_b =
            self.twiddle1.im * x16n.re + self.twiddle2.im * x25n.re + self.twiddle3.im * x34n.re;
        let x25im_a = input.load(0).im
            + self.twiddle1.re * x34p.im
            + self.twiddle2.re * x16p.im
            + self.twiddle3.re * x25p.im;
        let x25im_b =
            -self.twiddle1.im * x34n.re + self.twiddle2.im * x16n.re - self.twiddle3.im * x25n.re;
        let x34im_a = input.load(0).im
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

        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        output.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        output.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
    }
}

pub struct Butterfly8<T> {
    root2: T,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly8, 8, |this: &Butterfly8<_>| this.direction);
impl<T: FftNum> Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            root2: T::from_f64(0.5f64.sqrt()).unwrap(),
            direction,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let butterfly4 = Butterfly4::new(self.direction);

        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose the input into the scratch
        let mut scratch0 = [input.load(0), input.load(2), input.load(4), input.load(6)];
        let mut scratch1 = [input.load(1), input.load(3), input.load(5), input.load(7)];

        // step 2: column FFTs
        butterfly4.perform_fft_butterfly(&mut scratch0);
        butterfly4.perform_fft_butterfly(&mut scratch1);

        // step 3: apply twiddle factors
        scratch1[1] = (twiddles::rotate_90(scratch1[1], self.direction) + scratch1[1]) * self.root2;
        scratch1[2] = twiddles::rotate_90(scratch1[2], self.direction);
        scratch1[3] = (twiddles::rotate_90(scratch1[3], self.direction) - scratch1[3]) * self.root2;

        // step 4: transpose -- skipped because we're going to do the next FFTs non-contiguously

        // step 5: row FFTs
        for i in 0..4 {
            Butterfly2::perform_fft_strided(&mut scratch0[i], &mut scratch1[i]);
        }

        // step 6: copy data to the output. we don't need to transpose, because we skipped the step 4 transpose
        for i in 0..4 {
            output.store(scratch0[i], i);
        }
        for i in 0..4 {
            output.store(scratch1[i], i + 4);
        }
    }
}

pub struct Butterfly9<T> {
    butterfly3: Butterfly3<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle4: Complex<T>,
}
boilerplate_fft_butterfly!(Butterfly9, 9, |this: &Butterfly9<_>| this
    .butterfly3
    .fft_direction());
impl<T: FftNum> Butterfly9<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly3: Butterfly3::new(direction),
            twiddle1: twiddles::compute_twiddle(1, 9, direction),
            twiddle2: twiddles::compute_twiddle(2, 9, direction),
            twiddle4: twiddles::compute_twiddle(4, 9, direction),
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // algorithm: mixed radix with width=3 and height=3

        // step 1: transpose the input into the scratch
        let mut scratch0 = [input.load(0), input.load(3), input.load(6)];
        let mut scratch1 = [input.load(1), input.load(4), input.load(7)];
        let mut scratch2 = [input.load(2), input.load(5), input.load(8)];

        // step 2: column FFTs
        self.butterfly3.perform_fft_butterfly(&mut scratch0);
        self.butterfly3.perform_fft_butterfly(&mut scratch1);
        self.butterfly3.perform_fft_butterfly(&mut scratch2);

        // step 3: apply twiddle factors
        scratch1[1] = scratch1[1] * self.twiddle1;
        scratch1[2] = scratch1[2] * self.twiddle2;
        scratch2[1] = scratch2[1] * self.twiddle2;
        scratch2[2] = scratch2[2] * self.twiddle4;

        // step 4: SKIPPED because the next FFTs will be non-contiguous

        // step 5: row FFTs
        self.butterfly3
            .perform_fft_strided(&mut scratch0[0], &mut scratch1[0], &mut scratch2[0]);
        self.butterfly3
            .perform_fft_strided(&mut scratch0[1], &mut scratch1[1], &mut scratch2[1]);
        self.butterfly3
            .perform_fft_strided(&mut scratch0[2], &mut scratch1[2], &mut scratch2[2]);

        // step 6: copy the result into the output. normally we'd need to do a transpose here, but we can skip it because we skipped the transpose in step 4
        output.store(scratch0[0], 0);
        output.store(scratch0[1], 1);
        output.store(scratch0[2], 2);
        output.store(scratch1[0], 3);
        output.store(scratch1[1], 4);
        output.store(scratch1[2], 5);
        output.store(scratch2[0], 6);
        output.store(scratch2[1], 7);
        output.store(scratch2[2], 8);
    }
}

pub struct Butterfly11<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly11, 11, |this: &Butterfly11<_>| this.direction);
impl<T: FftNum> Butterfly11<T> {
    pub fn new(direction: FftDirection) -> Self {
        let twiddle1: Complex<T> = twiddles::compute_twiddle(1, 11, direction);
        let twiddle2: Complex<T> = twiddles::compute_twiddle(2, 11, direction);
        let twiddle3: Complex<T> = twiddles::compute_twiddle(3, 11, direction);
        let twiddle4: Complex<T> = twiddles::compute_twiddle(4, 11, direction);
        let twiddle5: Complex<T> = twiddles::compute_twiddle(5, 11, direction);
        Self {
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            direction,
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.

        let x110p = input.load(1) + input.load(10);
        let x110n = input.load(1) - input.load(10);
        let x29p = input.load(2) + input.load(9);
        let x29n = input.load(2) - input.load(9);
        let x38p = input.load(3) + input.load(8);
        let x38n = input.load(3) - input.load(8);
        let x47p = input.load(4) + input.load(7);
        let x47n = input.load(4) - input.load(7);
        let x56p = input.load(5) + input.load(6);
        let x56n = input.load(5) - input.load(6);
        let sum = input.load(0) + x110p + x29p + x38p + x47p + x56p;
        let b110re_a = input.load(0).re
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
        let b29re_a = input.load(0).re
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
        let b38re_a = input.load(0).re
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
        let b47re_a = input.load(0).re
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
        let b56re_a = input.load(0).re
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

        let b110im_a = input.load(0).im
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
        let b29im_a = input.load(0).im
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
        let b38im_a = input.load(0).im
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
        let b47im_a = input.load(0).im
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
        let b56im_a = input.load(0).im
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
        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        output.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        output.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
        output.store(
            Complex {
                re: out7re,
                im: out7im,
            },
            7,
        );
        output.store(
            Complex {
                re: out8re,
                im: out8im,
            },
            8,
        );
        output.store(
            Complex {
                re: out9re,
                im: out9im,
            },
            9,
        );
        output.store(
            Complex {
                re: out10re,
                im: out10im,
            },
            10,
        );
    }
}

pub struct Butterfly13<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly13, 13, |this: &Butterfly13<_>| this.direction);
impl<T: FftNum> Butterfly13<T> {
    pub fn new(direction: FftDirection) -> Self {
        let twiddle1: Complex<T> = twiddles::compute_twiddle(1, 13, direction);
        let twiddle2: Complex<T> = twiddles::compute_twiddle(2, 13, direction);
        let twiddle3: Complex<T> = twiddles::compute_twiddle(3, 13, direction);
        let twiddle4: Complex<T> = twiddles::compute_twiddle(4, 13, direction);
        let twiddle5: Complex<T> = twiddles::compute_twiddle(5, 13, direction);
        let twiddle6: Complex<T> = twiddles::compute_twiddle(6, 13, direction);
        Self {
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            direction,
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x112p = input.load(1) + input.load(12);
        let x112n = input.load(1) - input.load(12);
        let x211p = input.load(2) + input.load(11);
        let x211n = input.load(2) - input.load(11);
        let x310p = input.load(3) + input.load(10);
        let x310n = input.load(3) - input.load(10);
        let x49p = input.load(4) + input.load(9);
        let x49n = input.load(4) - input.load(9);
        let x58p = input.load(5) + input.load(8);
        let x58n = input.load(5) - input.load(8);
        let x67p = input.load(6) + input.load(7);
        let x67n = input.load(6) - input.load(7);
        let sum = input.load(0) + x112p + x211p + x310p + x49p + x58p + x67p;
        let b112re_a = input.load(0).re
            + self.twiddle1.re * x112p.re
            + self.twiddle2.re * x211p.re
            + self.twiddle3.re * x310p.re
            + self.twiddle4.re * x49p.re
            + self.twiddle5.re * x58p.re
            + self.twiddle6.re * x67p.re;
        let b112re_b = self.twiddle1.im * x112n.im
            + self.twiddle2.im * x211n.im
            + self.twiddle3.im * x310n.im
            + self.twiddle4.im * x49n.im
            + self.twiddle5.im * x58n.im
            + self.twiddle6.im * x67n.im;
        let b211re_a = input.load(0).re
            + self.twiddle2.re * x112p.re
            + self.twiddle4.re * x211p.re
            + self.twiddle6.re * x310p.re
            + self.twiddle5.re * x49p.re
            + self.twiddle3.re * x58p.re
            + self.twiddle1.re * x67p.re;
        let b211re_b = self.twiddle2.im * x112n.im
            + self.twiddle4.im * x211n.im
            + self.twiddle6.im * x310n.im
            + -self.twiddle5.im * x49n.im
            + -self.twiddle3.im * x58n.im
            + -self.twiddle1.im * x67n.im;
        let b310re_a = input.load(0).re
            + self.twiddle3.re * x112p.re
            + self.twiddle6.re * x211p.re
            + self.twiddle4.re * x310p.re
            + self.twiddle1.re * x49p.re
            + self.twiddle2.re * x58p.re
            + self.twiddle5.re * x67p.re;
        let b310re_b = self.twiddle3.im * x112n.im
            + self.twiddle6.im * x211n.im
            + -self.twiddle4.im * x310n.im
            + -self.twiddle1.im * x49n.im
            + self.twiddle2.im * x58n.im
            + self.twiddle5.im * x67n.im;
        let b49re_a = input.load(0).re
            + self.twiddle4.re * x112p.re
            + self.twiddle5.re * x211p.re
            + self.twiddle1.re * x310p.re
            + self.twiddle3.re * x49p.re
            + self.twiddle6.re * x58p.re
            + self.twiddle2.re * x67p.re;
        let b49re_b = self.twiddle4.im * x112n.im
            + -self.twiddle5.im * x211n.im
            + -self.twiddle1.im * x310n.im
            + self.twiddle3.im * x49n.im
            + -self.twiddle6.im * x58n.im
            + -self.twiddle2.im * x67n.im;
        let b58re_a = input.load(0).re
            + self.twiddle5.re * x112p.re
            + self.twiddle3.re * x211p.re
            + self.twiddle2.re * x310p.re
            + self.twiddle6.re * x49p.re
            + self.twiddle1.re * x58p.re
            + self.twiddle4.re * x67p.re;
        let b58re_b = self.twiddle5.im * x112n.im
            + -self.twiddle3.im * x211n.im
            + self.twiddle2.im * x310n.im
            + -self.twiddle6.im * x49n.im
            + -self.twiddle1.im * x58n.im
            + self.twiddle4.im * x67n.im;
        let b67re_a = input.load(0).re
            + self.twiddle6.re * x112p.re
            + self.twiddle1.re * x211p.re
            + self.twiddle5.re * x310p.re
            + self.twiddle2.re * x49p.re
            + self.twiddle4.re * x58p.re
            + self.twiddle3.re * x67p.re;
        let b67re_b = self.twiddle6.im * x112n.im
            + -self.twiddle1.im * x211n.im
            + self.twiddle5.im * x310n.im
            + -self.twiddle2.im * x49n.im
            + self.twiddle4.im * x58n.im
            + -self.twiddle3.im * x67n.im;

        let b112im_a = input.load(0).im
            + self.twiddle1.re * x112p.im
            + self.twiddle2.re * x211p.im
            + self.twiddle3.re * x310p.im
            + self.twiddle4.re * x49p.im
            + self.twiddle5.re * x58p.im
            + self.twiddle6.re * x67p.im;
        let b112im_b = self.twiddle1.im * x112n.re
            + self.twiddle2.im * x211n.re
            + self.twiddle3.im * x310n.re
            + self.twiddle4.im * x49n.re
            + self.twiddle5.im * x58n.re
            + self.twiddle6.im * x67n.re;
        let b211im_a = input.load(0).im
            + self.twiddle2.re * x112p.im
            + self.twiddle4.re * x211p.im
            + self.twiddle6.re * x310p.im
            + self.twiddle5.re * x49p.im
            + self.twiddle3.re * x58p.im
            + self.twiddle1.re * x67p.im;
        let b211im_b = self.twiddle2.im * x112n.re
            + self.twiddle4.im * x211n.re
            + self.twiddle6.im * x310n.re
            + -self.twiddle5.im * x49n.re
            + -self.twiddle3.im * x58n.re
            + -self.twiddle1.im * x67n.re;
        let b310im_a = input.load(0).im
            + self.twiddle3.re * x112p.im
            + self.twiddle6.re * x211p.im
            + self.twiddle4.re * x310p.im
            + self.twiddle1.re * x49p.im
            + self.twiddle2.re * x58p.im
            + self.twiddle5.re * x67p.im;
        let b310im_b = self.twiddle3.im * x112n.re
            + self.twiddle6.im * x211n.re
            + -self.twiddle4.im * x310n.re
            + -self.twiddle1.im * x49n.re
            + self.twiddle2.im * x58n.re
            + self.twiddle5.im * x67n.re;
        let b49im_a = input.load(0).im
            + self.twiddle4.re * x112p.im
            + self.twiddle5.re * x211p.im
            + self.twiddle1.re * x310p.im
            + self.twiddle3.re * x49p.im
            + self.twiddle6.re * x58p.im
            + self.twiddle2.re * x67p.im;
        let b49im_b = self.twiddle4.im * x112n.re
            + -self.twiddle5.im * x211n.re
            + -self.twiddle1.im * x310n.re
            + self.twiddle3.im * x49n.re
            + -self.twiddle6.im * x58n.re
            + -self.twiddle2.im * x67n.re;
        let b58im_a = input.load(0).im
            + self.twiddle5.re * x112p.im
            + self.twiddle3.re * x211p.im
            + self.twiddle2.re * x310p.im
            + self.twiddle6.re * x49p.im
            + self.twiddle1.re * x58p.im
            + self.twiddle4.re * x67p.im;
        let b58im_b = self.twiddle5.im * x112n.re
            + -self.twiddle3.im * x211n.re
            + self.twiddle2.im * x310n.re
            + -self.twiddle6.im * x49n.re
            + -self.twiddle1.im * x58n.re
            + self.twiddle4.im * x67n.re;
        let b67im_a = input.load(0).im
            + self.twiddle6.re * x112p.im
            + self.twiddle1.re * x211p.im
            + self.twiddle5.re * x310p.im
            + self.twiddle2.re * x49p.im
            + self.twiddle4.re * x58p.im
            + self.twiddle3.re * x67p.im;
        let b67im_b = self.twiddle6.im * x112n.re
            + -self.twiddle1.im * x211n.re
            + self.twiddle5.im * x310n.re
            + -self.twiddle2.im * x49n.re
            + self.twiddle4.im * x58n.re
            + -self.twiddle3.im * x67n.re;

        let out1re = b112re_a - b112re_b;
        let out1im = b112im_a + b112im_b;
        let out2re = b211re_a - b211re_b;
        let out2im = b211im_a + b211im_b;
        let out3re = b310re_a - b310re_b;
        let out3im = b310im_a + b310im_b;
        let out4re = b49re_a - b49re_b;
        let out4im = b49im_a + b49im_b;
        let out5re = b58re_a - b58re_b;
        let out5im = b58im_a + b58im_b;
        let out6re = b67re_a - b67re_b;
        let out6im = b67im_a + b67im_b;
        let out7re = b67re_a + b67re_b;
        let out7im = b67im_a - b67im_b;
        let out8re = b58re_a + b58re_b;
        let out8im = b58im_a - b58im_b;
        let out9re = b49re_a + b49re_b;
        let out9im = b49im_a - b49im_b;
        let out10re = b310re_a + b310re_b;
        let out10im = b310im_a - b310im_b;
        let out11re = b211re_a + b211re_b;
        let out11im = b211im_a - b211im_b;
        let out12re = b112re_a + b112re_b;
        let out12im = b112im_a - b112im_b;
        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        output.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        output.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
        output.store(
            Complex {
                re: out7re,
                im: out7im,
            },
            7,
        );
        output.store(
            Complex {
                re: out8re,
                im: out8im,
            },
            8,
        );
        output.store(
            Complex {
                re: out9re,
                im: out9im,
            },
            9,
        );
        output.store(
            Complex {
                re: out10re,
                im: out10im,
            },
            10,
        );
        output.store(
            Complex {
                re: out11re,
                im: out11im,
            },
            11,
        );
        output.store(
            Complex {
                re: out12re,
                im: out12im,
            },
            12,
        );
    }
}

pub struct Butterfly16<T> {
    butterfly8: Butterfly8<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}
boilerplate_fft_butterfly!(Butterfly16, 16, |this: &Butterfly16<_>| this
    .butterfly8
    .fft_direction());
impl<T: FftNum> Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly8: Butterfly8::new(direction),
            twiddle1: twiddles::compute_twiddle(1, 16, direction),
            twiddle2: twiddles::compute_twiddle(2, 16, direction),
            twiddle3: twiddles::compute_twiddle(3, 16, direction),
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        let butterfly4 = Butterfly4::new(self.fft_direction());

        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        let mut scratch_evens = [
            input.load(0),
            input.load(2),
            input.load(4),
            input.load(6),
            input.load(8),
            input.load(10),
            input.load(12),
            input.load(14),
        ];

        let mut scratch_odds_n1 = [input.load(1), input.load(5), input.load(9), input.load(13)];
        let mut scratch_odds_n3 = [input.load(15), input.load(3), input.load(7), input.load(11)];

        // step 2: column FFTs
        self.butterfly8.perform_fft_butterfly(&mut scratch_evens);
        butterfly4.perform_fft_butterfly(&mut scratch_odds_n1);
        butterfly4.perform_fft_butterfly(&mut scratch_odds_n3);

        // step 3: apply twiddle factors
        scratch_odds_n1[1] = scratch_odds_n1[1] * self.twiddle1;
        scratch_odds_n3[1] = scratch_odds_n3[1] * self.twiddle1.conj();

        scratch_odds_n1[2] = scratch_odds_n1[2] * self.twiddle2;
        scratch_odds_n3[2] = scratch_odds_n3[2] * self.twiddle2.conj();

        scratch_odds_n1[3] = scratch_odds_n1[3] * self.twiddle3;
        scratch_odds_n3[3] = scratch_odds_n3[3] * self.twiddle3.conj();

        // step 4: cross FFTs
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[0], &mut scratch_odds_n3[0]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[1], &mut scratch_odds_n3[1]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[2], &mut scratch_odds_n3[2]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[3], &mut scratch_odds_n3[3]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.fft_direction());
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.fft_direction());
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.fft_direction());
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.fft_direction());

        //step 5: copy/add/subtract data back to buffer
        output.store(scratch_evens[0] + scratch_odds_n1[0], 0);
        output.store(scratch_evens[1] + scratch_odds_n1[1], 1);
        output.store(scratch_evens[2] + scratch_odds_n1[2], 2);
        output.store(scratch_evens[3] + scratch_odds_n1[3], 3);
        output.store(scratch_evens[4] + scratch_odds_n3[0], 4);
        output.store(scratch_evens[5] + scratch_odds_n3[1], 5);
        output.store(scratch_evens[6] + scratch_odds_n3[2], 6);
        output.store(scratch_evens[7] + scratch_odds_n3[3], 7);
        output.store(scratch_evens[0] - scratch_odds_n1[0], 8);
        output.store(scratch_evens[1] - scratch_odds_n1[1], 9);
        output.store(scratch_evens[2] - scratch_odds_n1[2], 10);
        output.store(scratch_evens[3] - scratch_odds_n1[3], 11);
        output.store(scratch_evens[4] - scratch_odds_n3[0], 12);
        output.store(scratch_evens[5] - scratch_odds_n3[1], 13);
        output.store(scratch_evens[6] - scratch_odds_n3[2], 14);
        output.store(scratch_evens[7] - scratch_odds_n3[3], 15);
    }
}

pub struct Butterfly17<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly17, 17, |this: &Butterfly17<_>| this.direction);
impl<T: FftNum> Butterfly17<T> {
    pub fn new(direction: FftDirection) -> Self {
        let twiddle1: Complex<T> = twiddles::compute_twiddle(1, 17, direction);
        let twiddle2: Complex<T> = twiddles::compute_twiddle(2, 17, direction);
        let twiddle3: Complex<T> = twiddles::compute_twiddle(3, 17, direction);
        let twiddle4: Complex<T> = twiddles::compute_twiddle(4, 17, direction);
        let twiddle5: Complex<T> = twiddles::compute_twiddle(5, 17, direction);
        let twiddle6: Complex<T> = twiddles::compute_twiddle(6, 17, direction);
        let twiddle7: Complex<T> = twiddles::compute_twiddle(7, 17, direction);
        let twiddle8: Complex<T> = twiddles::compute_twiddle(8, 17, direction);
        Self {
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            twiddle7,
            twiddle8,
            direction,
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x116p = input.load(1) + input.load(16);
        let x116n = input.load(1) - input.load(16);
        let x215p = input.load(2) + input.load(15);
        let x215n = input.load(2) - input.load(15);
        let x314p = input.load(3) + input.load(14);
        let x314n = input.load(3) - input.load(14);
        let x413p = input.load(4) + input.load(13);
        let x413n = input.load(4) - input.load(13);
        let x512p = input.load(5) + input.load(12);
        let x512n = input.load(5) - input.load(12);
        let x611p = input.load(6) + input.load(11);
        let x611n = input.load(6) - input.load(11);
        let x710p = input.load(7) + input.load(10);
        let x710n = input.load(7) - input.load(10);
        let x89p = input.load(8) + input.load(9);
        let x89n = input.load(8) - input.load(9);
        let sum = input.load(0) + x116p + x215p + x314p + x413p + x512p + x611p + x710p + x89p;
        let b116re_a = input.load(0).re
            + self.twiddle1.re * x116p.re
            + self.twiddle2.re * x215p.re
            + self.twiddle3.re * x314p.re
            + self.twiddle4.re * x413p.re
            + self.twiddle5.re * x512p.re
            + self.twiddle6.re * x611p.re
            + self.twiddle7.re * x710p.re
            + self.twiddle8.re * x89p.re;
        let b116re_b = self.twiddle1.im * x116n.im
            + self.twiddle2.im * x215n.im
            + self.twiddle3.im * x314n.im
            + self.twiddle4.im * x413n.im
            + self.twiddle5.im * x512n.im
            + self.twiddle6.im * x611n.im
            + self.twiddle7.im * x710n.im
            + self.twiddle8.im * x89n.im;
        let b215re_a = input.load(0).re
            + self.twiddle2.re * x116p.re
            + self.twiddle4.re * x215p.re
            + self.twiddle6.re * x314p.re
            + self.twiddle8.re * x413p.re
            + self.twiddle7.re * x512p.re
            + self.twiddle5.re * x611p.re
            + self.twiddle3.re * x710p.re
            + self.twiddle1.re * x89p.re;
        let b215re_b = self.twiddle2.im * x116n.im
            + self.twiddle4.im * x215n.im
            + self.twiddle6.im * x314n.im
            + self.twiddle8.im * x413n.im
            + -self.twiddle7.im * x512n.im
            + -self.twiddle5.im * x611n.im
            + -self.twiddle3.im * x710n.im
            + -self.twiddle1.im * x89n.im;
        let b314re_a = input.load(0).re
            + self.twiddle3.re * x116p.re
            + self.twiddle6.re * x215p.re
            + self.twiddle8.re * x314p.re
            + self.twiddle5.re * x413p.re
            + self.twiddle2.re * x512p.re
            + self.twiddle1.re * x611p.re
            + self.twiddle4.re * x710p.re
            + self.twiddle7.re * x89p.re;
        let b314re_b = self.twiddle3.im * x116n.im
            + self.twiddle6.im * x215n.im
            + -self.twiddle8.im * x314n.im
            + -self.twiddle5.im * x413n.im
            + -self.twiddle2.im * x512n.im
            + self.twiddle1.im * x611n.im
            + self.twiddle4.im * x710n.im
            + self.twiddle7.im * x89n.im;
        let b413re_a = input.load(0).re
            + self.twiddle4.re * x116p.re
            + self.twiddle8.re * x215p.re
            + self.twiddle5.re * x314p.re
            + self.twiddle1.re * x413p.re
            + self.twiddle3.re * x512p.re
            + self.twiddle7.re * x611p.re
            + self.twiddle6.re * x710p.re
            + self.twiddle2.re * x89p.re;
        let b413re_b = self.twiddle4.im * x116n.im
            + self.twiddle8.im * x215n.im
            + -self.twiddle5.im * x314n.im
            + -self.twiddle1.im * x413n.im
            + self.twiddle3.im * x512n.im
            + self.twiddle7.im * x611n.im
            + -self.twiddle6.im * x710n.im
            + -self.twiddle2.im * x89n.im;
        let b512re_a = input.load(0).re
            + self.twiddle5.re * x116p.re
            + self.twiddle7.re * x215p.re
            + self.twiddle2.re * x314p.re
            + self.twiddle3.re * x413p.re
            + self.twiddle8.re * x512p.re
            + self.twiddle4.re * x611p.re
            + self.twiddle1.re * x710p.re
            + self.twiddle6.re * x89p.re;
        let b512re_b = self.twiddle5.im * x116n.im
            + -self.twiddle7.im * x215n.im
            + -self.twiddle2.im * x314n.im
            + self.twiddle3.im * x413n.im
            + self.twiddle8.im * x512n.im
            + -self.twiddle4.im * x611n.im
            + self.twiddle1.im * x710n.im
            + self.twiddle6.im * x89n.im;
        let b611re_a = input.load(0).re
            + self.twiddle6.re * x116p.re
            + self.twiddle5.re * x215p.re
            + self.twiddle1.re * x314p.re
            + self.twiddle7.re * x413p.re
            + self.twiddle4.re * x512p.re
            + self.twiddle2.re * x611p.re
            + self.twiddle8.re * x710p.re
            + self.twiddle3.re * x89p.re;
        let b611re_b = self.twiddle6.im * x116n.im
            + -self.twiddle5.im * x215n.im
            + self.twiddle1.im * x314n.im
            + self.twiddle7.im * x413n.im
            + -self.twiddle4.im * x512n.im
            + self.twiddle2.im * x611n.im
            + self.twiddle8.im * x710n.im
            + -self.twiddle3.im * x89n.im;
        let b710re_a = input.load(0).re
            + self.twiddle7.re * x116p.re
            + self.twiddle3.re * x215p.re
            + self.twiddle4.re * x314p.re
            + self.twiddle6.re * x413p.re
            + self.twiddle1.re * x512p.re
            + self.twiddle8.re * x611p.re
            + self.twiddle2.re * x710p.re
            + self.twiddle5.re * x89p.re;
        let b710re_b = self.twiddle7.im * x116n.im
            + -self.twiddle3.im * x215n.im
            + self.twiddle4.im * x314n.im
            + -self.twiddle6.im * x413n.im
            + self.twiddle1.im * x512n.im
            + self.twiddle8.im * x611n.im
            + -self.twiddle2.im * x710n.im
            + self.twiddle5.im * x89n.im;
        let b89re_a = input.load(0).re
            + self.twiddle8.re * x116p.re
            + self.twiddle1.re * x215p.re
            + self.twiddle7.re * x314p.re
            + self.twiddle2.re * x413p.re
            + self.twiddle6.re * x512p.re
            + self.twiddle3.re * x611p.re
            + self.twiddle5.re * x710p.re
            + self.twiddle4.re * x89p.re;
        let b89re_b = self.twiddle8.im * x116n.im
            + -self.twiddle1.im * x215n.im
            + self.twiddle7.im * x314n.im
            + -self.twiddle2.im * x413n.im
            + self.twiddle6.im * x512n.im
            + -self.twiddle3.im * x611n.im
            + self.twiddle5.im * x710n.im
            + -self.twiddle4.im * x89n.im;

        let b116im_a = input.load(0).im
            + self.twiddle1.re * x116p.im
            + self.twiddle2.re * x215p.im
            + self.twiddle3.re * x314p.im
            + self.twiddle4.re * x413p.im
            + self.twiddle5.re * x512p.im
            + self.twiddle6.re * x611p.im
            + self.twiddle7.re * x710p.im
            + self.twiddle8.re * x89p.im;
        let b116im_b = self.twiddle1.im * x116n.re
            + self.twiddle2.im * x215n.re
            + self.twiddle3.im * x314n.re
            + self.twiddle4.im * x413n.re
            + self.twiddle5.im * x512n.re
            + self.twiddle6.im * x611n.re
            + self.twiddle7.im * x710n.re
            + self.twiddle8.im * x89n.re;
        let b215im_a = input.load(0).im
            + self.twiddle2.re * x116p.im
            + self.twiddle4.re * x215p.im
            + self.twiddle6.re * x314p.im
            + self.twiddle8.re * x413p.im
            + self.twiddle7.re * x512p.im
            + self.twiddle5.re * x611p.im
            + self.twiddle3.re * x710p.im
            + self.twiddle1.re * x89p.im;
        let b215im_b = self.twiddle2.im * x116n.re
            + self.twiddle4.im * x215n.re
            + self.twiddle6.im * x314n.re
            + self.twiddle8.im * x413n.re
            + -self.twiddle7.im * x512n.re
            + -self.twiddle5.im * x611n.re
            + -self.twiddle3.im * x710n.re
            + -self.twiddle1.im * x89n.re;
        let b314im_a = input.load(0).im
            + self.twiddle3.re * x116p.im
            + self.twiddle6.re * x215p.im
            + self.twiddle8.re * x314p.im
            + self.twiddle5.re * x413p.im
            + self.twiddle2.re * x512p.im
            + self.twiddle1.re * x611p.im
            + self.twiddle4.re * x710p.im
            + self.twiddle7.re * x89p.im;
        let b314im_b = self.twiddle3.im * x116n.re
            + self.twiddle6.im * x215n.re
            + -self.twiddle8.im * x314n.re
            + -self.twiddle5.im * x413n.re
            + -self.twiddle2.im * x512n.re
            + self.twiddle1.im * x611n.re
            + self.twiddle4.im * x710n.re
            + self.twiddle7.im * x89n.re;
        let b413im_a = input.load(0).im
            + self.twiddle4.re * x116p.im
            + self.twiddle8.re * x215p.im
            + self.twiddle5.re * x314p.im
            + self.twiddle1.re * x413p.im
            + self.twiddle3.re * x512p.im
            + self.twiddle7.re * x611p.im
            + self.twiddle6.re * x710p.im
            + self.twiddle2.re * x89p.im;
        let b413im_b = self.twiddle4.im * x116n.re
            + self.twiddle8.im * x215n.re
            + -self.twiddle5.im * x314n.re
            + -self.twiddle1.im * x413n.re
            + self.twiddle3.im * x512n.re
            + self.twiddle7.im * x611n.re
            + -self.twiddle6.im * x710n.re
            + -self.twiddle2.im * x89n.re;
        let b512im_a = input.load(0).im
            + self.twiddle5.re * x116p.im
            + self.twiddle7.re * x215p.im
            + self.twiddle2.re * x314p.im
            + self.twiddle3.re * x413p.im
            + self.twiddle8.re * x512p.im
            + self.twiddle4.re * x611p.im
            + self.twiddle1.re * x710p.im
            + self.twiddle6.re * x89p.im;
        let b512im_b = self.twiddle5.im * x116n.re
            + -self.twiddle7.im * x215n.re
            + -self.twiddle2.im * x314n.re
            + self.twiddle3.im * x413n.re
            + self.twiddle8.im * x512n.re
            + -self.twiddle4.im * x611n.re
            + self.twiddle1.im * x710n.re
            + self.twiddle6.im * x89n.re;
        let b611im_a = input.load(0).im
            + self.twiddle6.re * x116p.im
            + self.twiddle5.re * x215p.im
            + self.twiddle1.re * x314p.im
            + self.twiddle7.re * x413p.im
            + self.twiddle4.re * x512p.im
            + self.twiddle2.re * x611p.im
            + self.twiddle8.re * x710p.im
            + self.twiddle3.re * x89p.im;
        let b611im_b = self.twiddle6.im * x116n.re
            + -self.twiddle5.im * x215n.re
            + self.twiddle1.im * x314n.re
            + self.twiddle7.im * x413n.re
            + -self.twiddle4.im * x512n.re
            + self.twiddle2.im * x611n.re
            + self.twiddle8.im * x710n.re
            + -self.twiddle3.im * x89n.re;
        let b710im_a = input.load(0).im
            + self.twiddle7.re * x116p.im
            + self.twiddle3.re * x215p.im
            + self.twiddle4.re * x314p.im
            + self.twiddle6.re * x413p.im
            + self.twiddle1.re * x512p.im
            + self.twiddle8.re * x611p.im
            + self.twiddle2.re * x710p.im
            + self.twiddle5.re * x89p.im;
        let b710im_b = self.twiddle7.im * x116n.re
            + -self.twiddle3.im * x215n.re
            + self.twiddle4.im * x314n.re
            + -self.twiddle6.im * x413n.re
            + self.twiddle1.im * x512n.re
            + self.twiddle8.im * x611n.re
            + -self.twiddle2.im * x710n.re
            + self.twiddle5.im * x89n.re;
        let b89im_a = input.load(0).im
            + self.twiddle8.re * x116p.im
            + self.twiddle1.re * x215p.im
            + self.twiddle7.re * x314p.im
            + self.twiddle2.re * x413p.im
            + self.twiddle6.re * x512p.im
            + self.twiddle3.re * x611p.im
            + self.twiddle5.re * x710p.im
            + self.twiddle4.re * x89p.im;
        let b89im_b = self.twiddle8.im * x116n.re
            + -self.twiddle1.im * x215n.re
            + self.twiddle7.im * x314n.re
            + -self.twiddle2.im * x413n.re
            + self.twiddle6.im * x512n.re
            + -self.twiddle3.im * x611n.re
            + self.twiddle5.im * x710n.re
            + -self.twiddle4.im * x89n.re;

        let out1re = b116re_a - b116re_b;
        let out1im = b116im_a + b116im_b;
        let out2re = b215re_a - b215re_b;
        let out2im = b215im_a + b215im_b;
        let out3re = b314re_a - b314re_b;
        let out3im = b314im_a + b314im_b;
        let out4re = b413re_a - b413re_b;
        let out4im = b413im_a + b413im_b;
        let out5re = b512re_a - b512re_b;
        let out5im = b512im_a + b512im_b;
        let out6re = b611re_a - b611re_b;
        let out6im = b611im_a + b611im_b;
        let out7re = b710re_a - b710re_b;
        let out7im = b710im_a + b710im_b;
        let out8re = b89re_a - b89re_b;
        let out8im = b89im_a + b89im_b;
        let out9re = b89re_a + b89re_b;
        let out9im = b89im_a - b89im_b;
        let out10re = b710re_a + b710re_b;
        let out10im = b710im_a - b710im_b;
        let out11re = b611re_a + b611re_b;
        let out11im = b611im_a - b611im_b;
        let out12re = b512re_a + b512re_b;
        let out12im = b512im_a - b512im_b;
        let out13re = b413re_a + b413re_b;
        let out13im = b413im_a - b413im_b;
        let out14re = b314re_a + b314re_b;
        let out14im = b314im_a - b314im_b;
        let out15re = b215re_a + b215re_b;
        let out15im = b215im_a - b215im_b;
        let out16re = b116re_a + b116re_b;
        let out16im = b116im_a - b116im_b;
        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        output.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        output.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
        output.store(
            Complex {
                re: out7re,
                im: out7im,
            },
            7,
        );
        output.store(
            Complex {
                re: out8re,
                im: out8im,
            },
            8,
        );
        output.store(
            Complex {
                re: out9re,
                im: out9im,
            },
            9,
        );
        output.store(
            Complex {
                re: out10re,
                im: out10im,
            },
            10,
        );
        output.store(
            Complex {
                re: out11re,
                im: out11im,
            },
            11,
        );
        output.store(
            Complex {
                re: out12re,
                im: out12im,
            },
            12,
        );
        output.store(
            Complex {
                re: out13re,
                im: out13im,
            },
            13,
        );
        output.store(
            Complex {
                re: out14re,
                im: out14im,
            },
            14,
        );
        output.store(
            Complex {
                re: out15re,
                im: out15im,
            },
            15,
        );
        output.store(
            Complex {
                re: out16re,
                im: out16im,
            },
            16,
        );
    }
}

pub struct Butterfly19<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
    twiddle9: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly19, 19, |this: &Butterfly19<_>| this.direction);
impl<T: FftNum> Butterfly19<T> {
    pub fn new(direction: FftDirection) -> Self {
        let twiddle1: Complex<T> = twiddles::compute_twiddle(1, 19, direction);
        let twiddle2: Complex<T> = twiddles::compute_twiddle(2, 19, direction);
        let twiddle3: Complex<T> = twiddles::compute_twiddle(3, 19, direction);
        let twiddle4: Complex<T> = twiddles::compute_twiddle(4, 19, direction);
        let twiddle5: Complex<T> = twiddles::compute_twiddle(5, 19, direction);
        let twiddle6: Complex<T> = twiddles::compute_twiddle(6, 19, direction);
        let twiddle7: Complex<T> = twiddles::compute_twiddle(7, 19, direction);
        let twiddle8: Complex<T> = twiddles::compute_twiddle(8, 19, direction);
        let twiddle9: Complex<T> = twiddles::compute_twiddle(9, 19, direction);
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
            direction,
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x118p = input.load(1) + input.load(18);
        let x118n = input.load(1) - input.load(18);
        let x217p = input.load(2) + input.load(17);
        let x217n = input.load(2) - input.load(17);
        let x316p = input.load(3) + input.load(16);
        let x316n = input.load(3) - input.load(16);
        let x415p = input.load(4) + input.load(15);
        let x415n = input.load(4) - input.load(15);
        let x514p = input.load(5) + input.load(14);
        let x514n = input.load(5) - input.load(14);
        let x613p = input.load(6) + input.load(13);
        let x613n = input.load(6) - input.load(13);
        let x712p = input.load(7) + input.load(12);
        let x712n = input.load(7) - input.load(12);
        let x811p = input.load(8) + input.load(11);
        let x811n = input.load(8) - input.load(11);
        let x910p = input.load(9) + input.load(10);
        let x910n = input.load(9) - input.load(10);
        let sum =
            input.load(0) + x118p + x217p + x316p + x415p + x514p + x613p + x712p + x811p + x910p;
        let b118re_a = input.load(0).re
            + self.twiddle1.re * x118p.re
            + self.twiddle2.re * x217p.re
            + self.twiddle3.re * x316p.re
            + self.twiddle4.re * x415p.re
            + self.twiddle5.re * x514p.re
            + self.twiddle6.re * x613p.re
            + self.twiddle7.re * x712p.re
            + self.twiddle8.re * x811p.re
            + self.twiddle9.re * x910p.re;
        let b118re_b = self.twiddle1.im * x118n.im
            + self.twiddle2.im * x217n.im
            + self.twiddle3.im * x316n.im
            + self.twiddle4.im * x415n.im
            + self.twiddle5.im * x514n.im
            + self.twiddle6.im * x613n.im
            + self.twiddle7.im * x712n.im
            + self.twiddle8.im * x811n.im
            + self.twiddle9.im * x910n.im;
        let b217re_a = input.load(0).re
            + self.twiddle2.re * x118p.re
            + self.twiddle4.re * x217p.re
            + self.twiddle6.re * x316p.re
            + self.twiddle8.re * x415p.re
            + self.twiddle9.re * x514p.re
            + self.twiddle7.re * x613p.re
            + self.twiddle5.re * x712p.re
            + self.twiddle3.re * x811p.re
            + self.twiddle1.re * x910p.re;
        let b217re_b = self.twiddle2.im * x118n.im
            + self.twiddle4.im * x217n.im
            + self.twiddle6.im * x316n.im
            + self.twiddle8.im * x415n.im
            + -self.twiddle9.im * x514n.im
            + -self.twiddle7.im * x613n.im
            + -self.twiddle5.im * x712n.im
            + -self.twiddle3.im * x811n.im
            + -self.twiddle1.im * x910n.im;
        let b316re_a = input.load(0).re
            + self.twiddle3.re * x118p.re
            + self.twiddle6.re * x217p.re
            + self.twiddle9.re * x316p.re
            + self.twiddle7.re * x415p.re
            + self.twiddle4.re * x514p.re
            + self.twiddle1.re * x613p.re
            + self.twiddle2.re * x712p.re
            + self.twiddle5.re * x811p.re
            + self.twiddle8.re * x910p.re;
        let b316re_b = self.twiddle3.im * x118n.im
            + self.twiddle6.im * x217n.im
            + self.twiddle9.im * x316n.im
            + -self.twiddle7.im * x415n.im
            + -self.twiddle4.im * x514n.im
            + -self.twiddle1.im * x613n.im
            + self.twiddle2.im * x712n.im
            + self.twiddle5.im * x811n.im
            + self.twiddle8.im * x910n.im;
        let b415re_a = input.load(0).re
            + self.twiddle4.re * x118p.re
            + self.twiddle8.re * x217p.re
            + self.twiddle7.re * x316p.re
            + self.twiddle3.re * x415p.re
            + self.twiddle1.re * x514p.re
            + self.twiddle5.re * x613p.re
            + self.twiddle9.re * x712p.re
            + self.twiddle6.re * x811p.re
            + self.twiddle2.re * x910p.re;
        let b415re_b = self.twiddle4.im * x118n.im
            + self.twiddle8.im * x217n.im
            + -self.twiddle7.im * x316n.im
            + -self.twiddle3.im * x415n.im
            + self.twiddle1.im * x514n.im
            + self.twiddle5.im * x613n.im
            + self.twiddle9.im * x712n.im
            + -self.twiddle6.im * x811n.im
            + -self.twiddle2.im * x910n.im;
        let b514re_a = input.load(0).re
            + self.twiddle5.re * x118p.re
            + self.twiddle9.re * x217p.re
            + self.twiddle4.re * x316p.re
            + self.twiddle1.re * x415p.re
            + self.twiddle6.re * x514p.re
            + self.twiddle8.re * x613p.re
            + self.twiddle3.re * x712p.re
            + self.twiddle2.re * x811p.re
            + self.twiddle7.re * x910p.re;
        let b514re_b = self.twiddle5.im * x118n.im
            + -self.twiddle9.im * x217n.im
            + -self.twiddle4.im * x316n.im
            + self.twiddle1.im * x415n.im
            + self.twiddle6.im * x514n.im
            + -self.twiddle8.im * x613n.im
            + -self.twiddle3.im * x712n.im
            + self.twiddle2.im * x811n.im
            + self.twiddle7.im * x910n.im;
        let b613re_a = input.load(0).re
            + self.twiddle6.re * x118p.re
            + self.twiddle7.re * x217p.re
            + self.twiddle1.re * x316p.re
            + self.twiddle5.re * x415p.re
            + self.twiddle8.re * x514p.re
            + self.twiddle2.re * x613p.re
            + self.twiddle4.re * x712p.re
            + self.twiddle9.re * x811p.re
            + self.twiddle3.re * x910p.re;
        let b613re_b = self.twiddle6.im * x118n.im
            + -self.twiddle7.im * x217n.im
            + -self.twiddle1.im * x316n.im
            + self.twiddle5.im * x415n.im
            + -self.twiddle8.im * x514n.im
            + -self.twiddle2.im * x613n.im
            + self.twiddle4.im * x712n.im
            + -self.twiddle9.im * x811n.im
            + -self.twiddle3.im * x910n.im;
        let b712re_a = input.load(0).re
            + self.twiddle7.re * x118p.re
            + self.twiddle5.re * x217p.re
            + self.twiddle2.re * x316p.re
            + self.twiddle9.re * x415p.re
            + self.twiddle3.re * x514p.re
            + self.twiddle4.re * x613p.re
            + self.twiddle8.re * x712p.re
            + self.twiddle1.re * x811p.re
            + self.twiddle6.re * x910p.re;
        let b712re_b = self.twiddle7.im * x118n.im
            + -self.twiddle5.im * x217n.im
            + self.twiddle2.im * x316n.im
            + self.twiddle9.im * x415n.im
            + -self.twiddle3.im * x514n.im
            + self.twiddle4.im * x613n.im
            + -self.twiddle8.im * x712n.im
            + -self.twiddle1.im * x811n.im
            + self.twiddle6.im * x910n.im;
        let b811re_a = input.load(0).re
            + self.twiddle8.re * x118p.re
            + self.twiddle3.re * x217p.re
            + self.twiddle5.re * x316p.re
            + self.twiddle6.re * x415p.re
            + self.twiddle2.re * x514p.re
            + self.twiddle9.re * x613p.re
            + self.twiddle1.re * x712p.re
            + self.twiddle7.re * x811p.re
            + self.twiddle4.re * x910p.re;
        let b811re_b = self.twiddle8.im * x118n.im
            + -self.twiddle3.im * x217n.im
            + self.twiddle5.im * x316n.im
            + -self.twiddle6.im * x415n.im
            + self.twiddle2.im * x514n.im
            + -self.twiddle9.im * x613n.im
            + -self.twiddle1.im * x712n.im
            + self.twiddle7.im * x811n.im
            + -self.twiddle4.im * x910n.im;
        let b910re_a = input.load(0).re
            + self.twiddle9.re * x118p.re
            + self.twiddle1.re * x217p.re
            + self.twiddle8.re * x316p.re
            + self.twiddle2.re * x415p.re
            + self.twiddle7.re * x514p.re
            + self.twiddle3.re * x613p.re
            + self.twiddle6.re * x712p.re
            + self.twiddle4.re * x811p.re
            + self.twiddle5.re * x910p.re;
        let b910re_b = self.twiddle9.im * x118n.im
            + -self.twiddle1.im * x217n.im
            + self.twiddle8.im * x316n.im
            + -self.twiddle2.im * x415n.im
            + self.twiddle7.im * x514n.im
            + -self.twiddle3.im * x613n.im
            + self.twiddle6.im * x712n.im
            + -self.twiddle4.im * x811n.im
            + self.twiddle5.im * x910n.im;

        let b118im_a = input.load(0).im
            + self.twiddle1.re * x118p.im
            + self.twiddle2.re * x217p.im
            + self.twiddle3.re * x316p.im
            + self.twiddle4.re * x415p.im
            + self.twiddle5.re * x514p.im
            + self.twiddle6.re * x613p.im
            + self.twiddle7.re * x712p.im
            + self.twiddle8.re * x811p.im
            + self.twiddle9.re * x910p.im;
        let b118im_b = self.twiddle1.im * x118n.re
            + self.twiddle2.im * x217n.re
            + self.twiddle3.im * x316n.re
            + self.twiddle4.im * x415n.re
            + self.twiddle5.im * x514n.re
            + self.twiddle6.im * x613n.re
            + self.twiddle7.im * x712n.re
            + self.twiddle8.im * x811n.re
            + self.twiddle9.im * x910n.re;
        let b217im_a = input.load(0).im
            + self.twiddle2.re * x118p.im
            + self.twiddle4.re * x217p.im
            + self.twiddle6.re * x316p.im
            + self.twiddle8.re * x415p.im
            + self.twiddle9.re * x514p.im
            + self.twiddle7.re * x613p.im
            + self.twiddle5.re * x712p.im
            + self.twiddle3.re * x811p.im
            + self.twiddle1.re * x910p.im;
        let b217im_b = self.twiddle2.im * x118n.re
            + self.twiddle4.im * x217n.re
            + self.twiddle6.im * x316n.re
            + self.twiddle8.im * x415n.re
            + -self.twiddle9.im * x514n.re
            + -self.twiddle7.im * x613n.re
            + -self.twiddle5.im * x712n.re
            + -self.twiddle3.im * x811n.re
            + -self.twiddle1.im * x910n.re;
        let b316im_a = input.load(0).im
            + self.twiddle3.re * x118p.im
            + self.twiddle6.re * x217p.im
            + self.twiddle9.re * x316p.im
            + self.twiddle7.re * x415p.im
            + self.twiddle4.re * x514p.im
            + self.twiddle1.re * x613p.im
            + self.twiddle2.re * x712p.im
            + self.twiddle5.re * x811p.im
            + self.twiddle8.re * x910p.im;
        let b316im_b = self.twiddle3.im * x118n.re
            + self.twiddle6.im * x217n.re
            + self.twiddle9.im * x316n.re
            + -self.twiddle7.im * x415n.re
            + -self.twiddle4.im * x514n.re
            + -self.twiddle1.im * x613n.re
            + self.twiddle2.im * x712n.re
            + self.twiddle5.im * x811n.re
            + self.twiddle8.im * x910n.re;
        let b415im_a = input.load(0).im
            + self.twiddle4.re * x118p.im
            + self.twiddle8.re * x217p.im
            + self.twiddle7.re * x316p.im
            + self.twiddle3.re * x415p.im
            + self.twiddle1.re * x514p.im
            + self.twiddle5.re * x613p.im
            + self.twiddle9.re * x712p.im
            + self.twiddle6.re * x811p.im
            + self.twiddle2.re * x910p.im;
        let b415im_b = self.twiddle4.im * x118n.re
            + self.twiddle8.im * x217n.re
            + -self.twiddle7.im * x316n.re
            + -self.twiddle3.im * x415n.re
            + self.twiddle1.im * x514n.re
            + self.twiddle5.im * x613n.re
            + self.twiddle9.im * x712n.re
            + -self.twiddle6.im * x811n.re
            + -self.twiddle2.im * x910n.re;
        let b514im_a = input.load(0).im
            + self.twiddle5.re * x118p.im
            + self.twiddle9.re * x217p.im
            + self.twiddle4.re * x316p.im
            + self.twiddle1.re * x415p.im
            + self.twiddle6.re * x514p.im
            + self.twiddle8.re * x613p.im
            + self.twiddle3.re * x712p.im
            + self.twiddle2.re * x811p.im
            + self.twiddle7.re * x910p.im;
        let b514im_b = self.twiddle5.im * x118n.re
            + -self.twiddle9.im * x217n.re
            + -self.twiddle4.im * x316n.re
            + self.twiddle1.im * x415n.re
            + self.twiddle6.im * x514n.re
            + -self.twiddle8.im * x613n.re
            + -self.twiddle3.im * x712n.re
            + self.twiddle2.im * x811n.re
            + self.twiddle7.im * x910n.re;
        let b613im_a = input.load(0).im
            + self.twiddle6.re * x118p.im
            + self.twiddle7.re * x217p.im
            + self.twiddle1.re * x316p.im
            + self.twiddle5.re * x415p.im
            + self.twiddle8.re * x514p.im
            + self.twiddle2.re * x613p.im
            + self.twiddle4.re * x712p.im
            + self.twiddle9.re * x811p.im
            + self.twiddle3.re * x910p.im;
        let b613im_b = self.twiddle6.im * x118n.re
            + -self.twiddle7.im * x217n.re
            + -self.twiddle1.im * x316n.re
            + self.twiddle5.im * x415n.re
            + -self.twiddle8.im * x514n.re
            + -self.twiddle2.im * x613n.re
            + self.twiddle4.im * x712n.re
            + -self.twiddle9.im * x811n.re
            + -self.twiddle3.im * x910n.re;
        let b712im_a = input.load(0).im
            + self.twiddle7.re * x118p.im
            + self.twiddle5.re * x217p.im
            + self.twiddle2.re * x316p.im
            + self.twiddle9.re * x415p.im
            + self.twiddle3.re * x514p.im
            + self.twiddle4.re * x613p.im
            + self.twiddle8.re * x712p.im
            + self.twiddle1.re * x811p.im
            + self.twiddle6.re * x910p.im;
        let b712im_b = self.twiddle7.im * x118n.re
            + -self.twiddle5.im * x217n.re
            + self.twiddle2.im * x316n.re
            + self.twiddle9.im * x415n.re
            + -self.twiddle3.im * x514n.re
            + self.twiddle4.im * x613n.re
            + -self.twiddle8.im * x712n.re
            + -self.twiddle1.im * x811n.re
            + self.twiddle6.im * x910n.re;
        let b811im_a = input.load(0).im
            + self.twiddle8.re * x118p.im
            + self.twiddle3.re * x217p.im
            + self.twiddle5.re * x316p.im
            + self.twiddle6.re * x415p.im
            + self.twiddle2.re * x514p.im
            + self.twiddle9.re * x613p.im
            + self.twiddle1.re * x712p.im
            + self.twiddle7.re * x811p.im
            + self.twiddle4.re * x910p.im;
        let b811im_b = self.twiddle8.im * x118n.re
            + -self.twiddle3.im * x217n.re
            + self.twiddle5.im * x316n.re
            + -self.twiddle6.im * x415n.re
            + self.twiddle2.im * x514n.re
            + -self.twiddle9.im * x613n.re
            + -self.twiddle1.im * x712n.re
            + self.twiddle7.im * x811n.re
            + -self.twiddle4.im * x910n.re;
        let b910im_a = input.load(0).im
            + self.twiddle9.re * x118p.im
            + self.twiddle1.re * x217p.im
            + self.twiddle8.re * x316p.im
            + self.twiddle2.re * x415p.im
            + self.twiddle7.re * x514p.im
            + self.twiddle3.re * x613p.im
            + self.twiddle6.re * x712p.im
            + self.twiddle4.re * x811p.im
            + self.twiddle5.re * x910p.im;
        let b910im_b = self.twiddle9.im * x118n.re
            + -self.twiddle1.im * x217n.re
            + self.twiddle8.im * x316n.re
            + -self.twiddle2.im * x415n.re
            + self.twiddle7.im * x514n.re
            + -self.twiddle3.im * x613n.re
            + self.twiddle6.im * x712n.re
            + -self.twiddle4.im * x811n.re
            + self.twiddle5.im * x910n.re;

        let out1re = b118re_a - b118re_b;
        let out1im = b118im_a + b118im_b;
        let out2re = b217re_a - b217re_b;
        let out2im = b217im_a + b217im_b;
        let out3re = b316re_a - b316re_b;
        let out3im = b316im_a + b316im_b;
        let out4re = b415re_a - b415re_b;
        let out4im = b415im_a + b415im_b;
        let out5re = b514re_a - b514re_b;
        let out5im = b514im_a + b514im_b;
        let out6re = b613re_a - b613re_b;
        let out6im = b613im_a + b613im_b;
        let out7re = b712re_a - b712re_b;
        let out7im = b712im_a + b712im_b;
        let out8re = b811re_a - b811re_b;
        let out8im = b811im_a + b811im_b;
        let out9re = b910re_a - b910re_b;
        let out9im = b910im_a + b910im_b;
        let out10re = b910re_a + b910re_b;
        let out10im = b910im_a - b910im_b;
        let out11re = b811re_a + b811re_b;
        let out11im = b811im_a - b811im_b;
        let out12re = b712re_a + b712re_b;
        let out12im = b712im_a - b712im_b;
        let out13re = b613re_a + b613re_b;
        let out13im = b613im_a - b613im_b;
        let out14re = b514re_a + b514re_b;
        let out14im = b514im_a - b514im_b;
        let out15re = b415re_a + b415re_b;
        let out15im = b415im_a - b415im_b;
        let out16re = b316re_a + b316re_b;
        let out16im = b316im_a - b316im_b;
        let out17re = b217re_a + b217re_b;
        let out17im = b217im_a - b217im_b;
        let out18re = b118re_a + b118re_b;
        let out18im = b118im_a - b118im_b;
        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        output.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        output.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
        output.store(
            Complex {
                re: out7re,
                im: out7im,
            },
            7,
        );
        output.store(
            Complex {
                re: out8re,
                im: out8im,
            },
            8,
        );
        output.store(
            Complex {
                re: out9re,
                im: out9im,
            },
            9,
        );
        output.store(
            Complex {
                re: out10re,
                im: out10im,
            },
            10,
        );
        output.store(
            Complex {
                re: out11re,
                im: out11im,
            },
            11,
        );
        output.store(
            Complex {
                re: out12re,
                im: out12im,
            },
            12,
        );
        output.store(
            Complex {
                re: out13re,
                im: out13im,
            },
            13,
        );
        output.store(
            Complex {
                re: out14re,
                im: out14im,
            },
            14,
        );
        output.store(
            Complex {
                re: out15re,
                im: out15im,
            },
            15,
        );
        output.store(
            Complex {
                re: out16re,
                im: out16im,
            },
            16,
        );
        output.store(
            Complex {
                re: out17re,
                im: out17im,
            },
            17,
        );
        output.store(
            Complex {
                re: out18re,
                im: out18im,
            },
            18,
        );
    }
}

pub struct Butterfly23<T> {
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
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly23, 23, |this: &Butterfly23<_>| this.direction);
impl<T: FftNum> Butterfly23<T> {
    pub fn new(direction: FftDirection) -> Self {
        let twiddle1: Complex<T> = twiddles::compute_twiddle(1, 23, direction);
        let twiddle2: Complex<T> = twiddles::compute_twiddle(2, 23, direction);
        let twiddle3: Complex<T> = twiddles::compute_twiddle(3, 23, direction);
        let twiddle4: Complex<T> = twiddles::compute_twiddle(4, 23, direction);
        let twiddle5: Complex<T> = twiddles::compute_twiddle(5, 23, direction);
        let twiddle6: Complex<T> = twiddles::compute_twiddle(6, 23, direction);
        let twiddle7: Complex<T> = twiddles::compute_twiddle(7, 23, direction);
        let twiddle8: Complex<T> = twiddles::compute_twiddle(8, 23, direction);
        let twiddle9: Complex<T> = twiddles::compute_twiddle(9, 23, direction);
        let twiddle10: Complex<T> = twiddles::compute_twiddle(10, 23, direction);
        let twiddle11: Complex<T> = twiddles::compute_twiddle(11, 23, direction);
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
            direction,
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x122p = input.load(1) + input.load(22);
        let x122n = input.load(1) - input.load(22);
        let x221p = input.load(2) + input.load(21);
        let x221n = input.load(2) - input.load(21);
        let x320p = input.load(3) + input.load(20);
        let x320n = input.load(3) - input.load(20);
        let x419p = input.load(4) + input.load(19);
        let x419n = input.load(4) - input.load(19);
        let x518p = input.load(5) + input.load(18);
        let x518n = input.load(5) - input.load(18);
        let x617p = input.load(6) + input.load(17);
        let x617n = input.load(6) - input.load(17);
        let x716p = input.load(7) + input.load(16);
        let x716n = input.load(7) - input.load(16);
        let x815p = input.load(8) + input.load(15);
        let x815n = input.load(8) - input.load(15);
        let x914p = input.load(9) + input.load(14);
        let x914n = input.load(9) - input.load(14);
        let x1013p = input.load(10) + input.load(13);
        let x1013n = input.load(10) - input.load(13);
        let x1112p = input.load(11) + input.load(12);
        let x1112n = input.load(11) - input.load(12);
        let sum = input.load(0)
            + x122p
            + x221p
            + x320p
            + x419p
            + x518p
            + x617p
            + x716p
            + x815p
            + x914p
            + x1013p
            + x1112p;
        let b122re_a = input.load(0).re
            + self.twiddle1.re * x122p.re
            + self.twiddle2.re * x221p.re
            + self.twiddle3.re * x320p.re
            + self.twiddle4.re * x419p.re
            + self.twiddle5.re * x518p.re
            + self.twiddle6.re * x617p.re
            + self.twiddle7.re * x716p.re
            + self.twiddle8.re * x815p.re
            + self.twiddle9.re * x914p.re
            + self.twiddle10.re * x1013p.re
            + self.twiddle11.re * x1112p.re;
        let b122re_b = self.twiddle1.im * x122n.im
            + self.twiddle2.im * x221n.im
            + self.twiddle3.im * x320n.im
            + self.twiddle4.im * x419n.im
            + self.twiddle5.im * x518n.im
            + self.twiddle6.im * x617n.im
            + self.twiddle7.im * x716n.im
            + self.twiddle8.im * x815n.im
            + self.twiddle9.im * x914n.im
            + self.twiddle10.im * x1013n.im
            + self.twiddle11.im * x1112n.im;
        let b221re_a = input.load(0).re
            + self.twiddle2.re * x122p.re
            + self.twiddle4.re * x221p.re
            + self.twiddle6.re * x320p.re
            + self.twiddle8.re * x419p.re
            + self.twiddle10.re * x518p.re
            + self.twiddle11.re * x617p.re
            + self.twiddle9.re * x716p.re
            + self.twiddle7.re * x815p.re
            + self.twiddle5.re * x914p.re
            + self.twiddle3.re * x1013p.re
            + self.twiddle1.re * x1112p.re;
        let b221re_b = self.twiddle2.im * x122n.im
            + self.twiddle4.im * x221n.im
            + self.twiddle6.im * x320n.im
            + self.twiddle8.im * x419n.im
            + self.twiddle10.im * x518n.im
            + -self.twiddle11.im * x617n.im
            + -self.twiddle9.im * x716n.im
            + -self.twiddle7.im * x815n.im
            + -self.twiddle5.im * x914n.im
            + -self.twiddle3.im * x1013n.im
            + -self.twiddle1.im * x1112n.im;
        let b320re_a = input.load(0).re
            + self.twiddle3.re * x122p.re
            + self.twiddle6.re * x221p.re
            + self.twiddle9.re * x320p.re
            + self.twiddle11.re * x419p.re
            + self.twiddle8.re * x518p.re
            + self.twiddle5.re * x617p.re
            + self.twiddle2.re * x716p.re
            + self.twiddle1.re * x815p.re
            + self.twiddle4.re * x914p.re
            + self.twiddle7.re * x1013p.re
            + self.twiddle10.re * x1112p.re;
        let b320re_b = self.twiddle3.im * x122n.im
            + self.twiddle6.im * x221n.im
            + self.twiddle9.im * x320n.im
            + -self.twiddle11.im * x419n.im
            + -self.twiddle8.im * x518n.im
            + -self.twiddle5.im * x617n.im
            + -self.twiddle2.im * x716n.im
            + self.twiddle1.im * x815n.im
            + self.twiddle4.im * x914n.im
            + self.twiddle7.im * x1013n.im
            + self.twiddle10.im * x1112n.im;
        let b419re_a = input.load(0).re
            + self.twiddle4.re * x122p.re
            + self.twiddle8.re * x221p.re
            + self.twiddle11.re * x320p.re
            + self.twiddle7.re * x419p.re
            + self.twiddle3.re * x518p.re
            + self.twiddle1.re * x617p.re
            + self.twiddle5.re * x716p.re
            + self.twiddle9.re * x815p.re
            + self.twiddle10.re * x914p.re
            + self.twiddle6.re * x1013p.re
            + self.twiddle2.re * x1112p.re;
        let b419re_b = self.twiddle4.im * x122n.im
            + self.twiddle8.im * x221n.im
            + -self.twiddle11.im * x320n.im
            + -self.twiddle7.im * x419n.im
            + -self.twiddle3.im * x518n.im
            + self.twiddle1.im * x617n.im
            + self.twiddle5.im * x716n.im
            + self.twiddle9.im * x815n.im
            + -self.twiddle10.im * x914n.im
            + -self.twiddle6.im * x1013n.im
            + -self.twiddle2.im * x1112n.im;
        let b518re_a = input.load(0).re
            + self.twiddle5.re * x122p.re
            + self.twiddle10.re * x221p.re
            + self.twiddle8.re * x320p.re
            + self.twiddle3.re * x419p.re
            + self.twiddle2.re * x518p.re
            + self.twiddle7.re * x617p.re
            + self.twiddle11.re * x716p.re
            + self.twiddle6.re * x815p.re
            + self.twiddle1.re * x914p.re
            + self.twiddle4.re * x1013p.re
            + self.twiddle9.re * x1112p.re;
        let b518re_b = self.twiddle5.im * x122n.im
            + self.twiddle10.im * x221n.im
            + -self.twiddle8.im * x320n.im
            + -self.twiddle3.im * x419n.im
            + self.twiddle2.im * x518n.im
            + self.twiddle7.im * x617n.im
            + -self.twiddle11.im * x716n.im
            + -self.twiddle6.im * x815n.im
            + -self.twiddle1.im * x914n.im
            + self.twiddle4.im * x1013n.im
            + self.twiddle9.im * x1112n.im;
        let b617re_a = input.load(0).re
            + self.twiddle6.re * x122p.re
            + self.twiddle11.re * x221p.re
            + self.twiddle5.re * x320p.re
            + self.twiddle1.re * x419p.re
            + self.twiddle7.re * x518p.re
            + self.twiddle10.re * x617p.re
            + self.twiddle4.re * x716p.re
            + self.twiddle2.re * x815p.re
            + self.twiddle8.re * x914p.re
            + self.twiddle9.re * x1013p.re
            + self.twiddle3.re * x1112p.re;
        let b617re_b = self.twiddle6.im * x122n.im
            + -self.twiddle11.im * x221n.im
            + -self.twiddle5.im * x320n.im
            + self.twiddle1.im * x419n.im
            + self.twiddle7.im * x518n.im
            + -self.twiddle10.im * x617n.im
            + -self.twiddle4.im * x716n.im
            + self.twiddle2.im * x815n.im
            + self.twiddle8.im * x914n.im
            + -self.twiddle9.im * x1013n.im
            + -self.twiddle3.im * x1112n.im;
        let b716re_a = input.load(0).re
            + self.twiddle7.re * x122p.re
            + self.twiddle9.re * x221p.re
            + self.twiddle2.re * x320p.re
            + self.twiddle5.re * x419p.re
            + self.twiddle11.re * x518p.re
            + self.twiddle4.re * x617p.re
            + self.twiddle3.re * x716p.re
            + self.twiddle10.re * x815p.re
            + self.twiddle6.re * x914p.re
            + self.twiddle1.re * x1013p.re
            + self.twiddle8.re * x1112p.re;
        let b716re_b = self.twiddle7.im * x122n.im
            + -self.twiddle9.im * x221n.im
            + -self.twiddle2.im * x320n.im
            + self.twiddle5.im * x419n.im
            + -self.twiddle11.im * x518n.im
            + -self.twiddle4.im * x617n.im
            + self.twiddle3.im * x716n.im
            + self.twiddle10.im * x815n.im
            + -self.twiddle6.im * x914n.im
            + self.twiddle1.im * x1013n.im
            + self.twiddle8.im * x1112n.im;
        let b815re_a = input.load(0).re
            + self.twiddle8.re * x122p.re
            + self.twiddle7.re * x221p.re
            + self.twiddle1.re * x320p.re
            + self.twiddle9.re * x419p.re
            + self.twiddle6.re * x518p.re
            + self.twiddle2.re * x617p.re
            + self.twiddle10.re * x716p.re
            + self.twiddle5.re * x815p.re
            + self.twiddle3.re * x914p.re
            + self.twiddle11.re * x1013p.re
            + self.twiddle4.re * x1112p.re;
        let b815re_b = self.twiddle8.im * x122n.im
            + -self.twiddle7.im * x221n.im
            + self.twiddle1.im * x320n.im
            + self.twiddle9.im * x419n.im
            + -self.twiddle6.im * x518n.im
            + self.twiddle2.im * x617n.im
            + self.twiddle10.im * x716n.im
            + -self.twiddle5.im * x815n.im
            + self.twiddle3.im * x914n.im
            + self.twiddle11.im * x1013n.im
            + -self.twiddle4.im * x1112n.im;
        let b914re_a = input.load(0).re
            + self.twiddle9.re * x122p.re
            + self.twiddle5.re * x221p.re
            + self.twiddle4.re * x320p.re
            + self.twiddle10.re * x419p.re
            + self.twiddle1.re * x518p.re
            + self.twiddle8.re * x617p.re
            + self.twiddle6.re * x716p.re
            + self.twiddle3.re * x815p.re
            + self.twiddle11.re * x914p.re
            + self.twiddle2.re * x1013p.re
            + self.twiddle7.re * x1112p.re;
        let b914re_b = self.twiddle9.im * x122n.im
            + -self.twiddle5.im * x221n.im
            + self.twiddle4.im * x320n.im
            + -self.twiddle10.im * x419n.im
            + -self.twiddle1.im * x518n.im
            + self.twiddle8.im * x617n.im
            + -self.twiddle6.im * x716n.im
            + self.twiddle3.im * x815n.im
            + -self.twiddle11.im * x914n.im
            + -self.twiddle2.im * x1013n.im
            + self.twiddle7.im * x1112n.im;
        let b1013re_a = input.load(0).re
            + self.twiddle10.re * x122p.re
            + self.twiddle3.re * x221p.re
            + self.twiddle7.re * x320p.re
            + self.twiddle6.re * x419p.re
            + self.twiddle4.re * x518p.re
            + self.twiddle9.re * x617p.re
            + self.twiddle1.re * x716p.re
            + self.twiddle11.re * x815p.re
            + self.twiddle2.re * x914p.re
            + self.twiddle8.re * x1013p.re
            + self.twiddle5.re * x1112p.re;
        let b1013re_b = self.twiddle10.im * x122n.im
            + -self.twiddle3.im * x221n.im
            + self.twiddle7.im * x320n.im
            + -self.twiddle6.im * x419n.im
            + self.twiddle4.im * x518n.im
            + -self.twiddle9.im * x617n.im
            + self.twiddle1.im * x716n.im
            + self.twiddle11.im * x815n.im
            + -self.twiddle2.im * x914n.im
            + self.twiddle8.im * x1013n.im
            + -self.twiddle5.im * x1112n.im;
        let b1112re_a = input.load(0).re
            + self.twiddle11.re * x122p.re
            + self.twiddle1.re * x221p.re
            + self.twiddle10.re * x320p.re
            + self.twiddle2.re * x419p.re
            + self.twiddle9.re * x518p.re
            + self.twiddle3.re * x617p.re
            + self.twiddle8.re * x716p.re
            + self.twiddle4.re * x815p.re
            + self.twiddle7.re * x914p.re
            + self.twiddle5.re * x1013p.re
            + self.twiddle6.re * x1112p.re;
        let b1112re_b = self.twiddle11.im * x122n.im
            + -self.twiddle1.im * x221n.im
            + self.twiddle10.im * x320n.im
            + -self.twiddle2.im * x419n.im
            + self.twiddle9.im * x518n.im
            + -self.twiddle3.im * x617n.im
            + self.twiddle8.im * x716n.im
            + -self.twiddle4.im * x815n.im
            + self.twiddle7.im * x914n.im
            + -self.twiddle5.im * x1013n.im
            + self.twiddle6.im * x1112n.im;

        let b122im_a = input.load(0).im
            + self.twiddle1.re * x122p.im
            + self.twiddle2.re * x221p.im
            + self.twiddle3.re * x320p.im
            + self.twiddle4.re * x419p.im
            + self.twiddle5.re * x518p.im
            + self.twiddle6.re * x617p.im
            + self.twiddle7.re * x716p.im
            + self.twiddle8.re * x815p.im
            + self.twiddle9.re * x914p.im
            + self.twiddle10.re * x1013p.im
            + self.twiddle11.re * x1112p.im;
        let b122im_b = self.twiddle1.im * x122n.re
            + self.twiddle2.im * x221n.re
            + self.twiddle3.im * x320n.re
            + self.twiddle4.im * x419n.re
            + self.twiddle5.im * x518n.re
            + self.twiddle6.im * x617n.re
            + self.twiddle7.im * x716n.re
            + self.twiddle8.im * x815n.re
            + self.twiddle9.im * x914n.re
            + self.twiddle10.im * x1013n.re
            + self.twiddle11.im * x1112n.re;
        let b221im_a = input.load(0).im
            + self.twiddle2.re * x122p.im
            + self.twiddle4.re * x221p.im
            + self.twiddle6.re * x320p.im
            + self.twiddle8.re * x419p.im
            + self.twiddle10.re * x518p.im
            + self.twiddle11.re * x617p.im
            + self.twiddle9.re * x716p.im
            + self.twiddle7.re * x815p.im
            + self.twiddle5.re * x914p.im
            + self.twiddle3.re * x1013p.im
            + self.twiddle1.re * x1112p.im;
        let b221im_b = self.twiddle2.im * x122n.re
            + self.twiddle4.im * x221n.re
            + self.twiddle6.im * x320n.re
            + self.twiddle8.im * x419n.re
            + self.twiddle10.im * x518n.re
            + -self.twiddle11.im * x617n.re
            + -self.twiddle9.im * x716n.re
            + -self.twiddle7.im * x815n.re
            + -self.twiddle5.im * x914n.re
            + -self.twiddle3.im * x1013n.re
            + -self.twiddle1.im * x1112n.re;
        let b320im_a = input.load(0).im
            + self.twiddle3.re * x122p.im
            + self.twiddle6.re * x221p.im
            + self.twiddle9.re * x320p.im
            + self.twiddle11.re * x419p.im
            + self.twiddle8.re * x518p.im
            + self.twiddle5.re * x617p.im
            + self.twiddle2.re * x716p.im
            + self.twiddle1.re * x815p.im
            + self.twiddle4.re * x914p.im
            + self.twiddle7.re * x1013p.im
            + self.twiddle10.re * x1112p.im;
        let b320im_b = self.twiddle3.im * x122n.re
            + self.twiddle6.im * x221n.re
            + self.twiddle9.im * x320n.re
            + -self.twiddle11.im * x419n.re
            + -self.twiddle8.im * x518n.re
            + -self.twiddle5.im * x617n.re
            + -self.twiddle2.im * x716n.re
            + self.twiddle1.im * x815n.re
            + self.twiddle4.im * x914n.re
            + self.twiddle7.im * x1013n.re
            + self.twiddle10.im * x1112n.re;
        let b419im_a = input.load(0).im
            + self.twiddle4.re * x122p.im
            + self.twiddle8.re * x221p.im
            + self.twiddle11.re * x320p.im
            + self.twiddle7.re * x419p.im
            + self.twiddle3.re * x518p.im
            + self.twiddle1.re * x617p.im
            + self.twiddle5.re * x716p.im
            + self.twiddle9.re * x815p.im
            + self.twiddle10.re * x914p.im
            + self.twiddle6.re * x1013p.im
            + self.twiddle2.re * x1112p.im;
        let b419im_b = self.twiddle4.im * x122n.re
            + self.twiddle8.im * x221n.re
            + -self.twiddle11.im * x320n.re
            + -self.twiddle7.im * x419n.re
            + -self.twiddle3.im * x518n.re
            + self.twiddle1.im * x617n.re
            + self.twiddle5.im * x716n.re
            + self.twiddle9.im * x815n.re
            + -self.twiddle10.im * x914n.re
            + -self.twiddle6.im * x1013n.re
            + -self.twiddle2.im * x1112n.re;
        let b518im_a = input.load(0).im
            + self.twiddle5.re * x122p.im
            + self.twiddle10.re * x221p.im
            + self.twiddle8.re * x320p.im
            + self.twiddle3.re * x419p.im
            + self.twiddle2.re * x518p.im
            + self.twiddle7.re * x617p.im
            + self.twiddle11.re * x716p.im
            + self.twiddle6.re * x815p.im
            + self.twiddle1.re * x914p.im
            + self.twiddle4.re * x1013p.im
            + self.twiddle9.re * x1112p.im;
        let b518im_b = self.twiddle5.im * x122n.re
            + self.twiddle10.im * x221n.re
            + -self.twiddle8.im * x320n.re
            + -self.twiddle3.im * x419n.re
            + self.twiddle2.im * x518n.re
            + self.twiddle7.im * x617n.re
            + -self.twiddle11.im * x716n.re
            + -self.twiddle6.im * x815n.re
            + -self.twiddle1.im * x914n.re
            + self.twiddle4.im * x1013n.re
            + self.twiddle9.im * x1112n.re;
        let b617im_a = input.load(0).im
            + self.twiddle6.re * x122p.im
            + self.twiddle11.re * x221p.im
            + self.twiddle5.re * x320p.im
            + self.twiddle1.re * x419p.im
            + self.twiddle7.re * x518p.im
            + self.twiddle10.re * x617p.im
            + self.twiddle4.re * x716p.im
            + self.twiddle2.re * x815p.im
            + self.twiddle8.re * x914p.im
            + self.twiddle9.re * x1013p.im
            + self.twiddle3.re * x1112p.im;
        let b617im_b = self.twiddle6.im * x122n.re
            + -self.twiddle11.im * x221n.re
            + -self.twiddle5.im * x320n.re
            + self.twiddle1.im * x419n.re
            + self.twiddle7.im * x518n.re
            + -self.twiddle10.im * x617n.re
            + -self.twiddle4.im * x716n.re
            + self.twiddle2.im * x815n.re
            + self.twiddle8.im * x914n.re
            + -self.twiddle9.im * x1013n.re
            + -self.twiddle3.im * x1112n.re;
        let b716im_a = input.load(0).im
            + self.twiddle7.re * x122p.im
            + self.twiddle9.re * x221p.im
            + self.twiddle2.re * x320p.im
            + self.twiddle5.re * x419p.im
            + self.twiddle11.re * x518p.im
            + self.twiddle4.re * x617p.im
            + self.twiddle3.re * x716p.im
            + self.twiddle10.re * x815p.im
            + self.twiddle6.re * x914p.im
            + self.twiddle1.re * x1013p.im
            + self.twiddle8.re * x1112p.im;
        let b716im_b = self.twiddle7.im * x122n.re
            + -self.twiddle9.im * x221n.re
            + -self.twiddle2.im * x320n.re
            + self.twiddle5.im * x419n.re
            + -self.twiddle11.im * x518n.re
            + -self.twiddle4.im * x617n.re
            + self.twiddle3.im * x716n.re
            + self.twiddle10.im * x815n.re
            + -self.twiddle6.im * x914n.re
            + self.twiddle1.im * x1013n.re
            + self.twiddle8.im * x1112n.re;
        let b815im_a = input.load(0).im
            + self.twiddle8.re * x122p.im
            + self.twiddle7.re * x221p.im
            + self.twiddle1.re * x320p.im
            + self.twiddle9.re * x419p.im
            + self.twiddle6.re * x518p.im
            + self.twiddle2.re * x617p.im
            + self.twiddle10.re * x716p.im
            + self.twiddle5.re * x815p.im
            + self.twiddle3.re * x914p.im
            + self.twiddle11.re * x1013p.im
            + self.twiddle4.re * x1112p.im;
        let b815im_b = self.twiddle8.im * x122n.re
            + -self.twiddle7.im * x221n.re
            + self.twiddle1.im * x320n.re
            + self.twiddle9.im * x419n.re
            + -self.twiddle6.im * x518n.re
            + self.twiddle2.im * x617n.re
            + self.twiddle10.im * x716n.re
            + -self.twiddle5.im * x815n.re
            + self.twiddle3.im * x914n.re
            + self.twiddle11.im * x1013n.re
            + -self.twiddle4.im * x1112n.re;
        let b914im_a = input.load(0).im
            + self.twiddle9.re * x122p.im
            + self.twiddle5.re * x221p.im
            + self.twiddle4.re * x320p.im
            + self.twiddle10.re * x419p.im
            + self.twiddle1.re * x518p.im
            + self.twiddle8.re * x617p.im
            + self.twiddle6.re * x716p.im
            + self.twiddle3.re * x815p.im
            + self.twiddle11.re * x914p.im
            + self.twiddle2.re * x1013p.im
            + self.twiddle7.re * x1112p.im;
        let b914im_b = self.twiddle9.im * x122n.re
            + -self.twiddle5.im * x221n.re
            + self.twiddle4.im * x320n.re
            + -self.twiddle10.im * x419n.re
            + -self.twiddle1.im * x518n.re
            + self.twiddle8.im * x617n.re
            + -self.twiddle6.im * x716n.re
            + self.twiddle3.im * x815n.re
            + -self.twiddle11.im * x914n.re
            + -self.twiddle2.im * x1013n.re
            + self.twiddle7.im * x1112n.re;
        let b1013im_a = input.load(0).im
            + self.twiddle10.re * x122p.im
            + self.twiddle3.re * x221p.im
            + self.twiddle7.re * x320p.im
            + self.twiddle6.re * x419p.im
            + self.twiddle4.re * x518p.im
            + self.twiddle9.re * x617p.im
            + self.twiddle1.re * x716p.im
            + self.twiddle11.re * x815p.im
            + self.twiddle2.re * x914p.im
            + self.twiddle8.re * x1013p.im
            + self.twiddle5.re * x1112p.im;
        let b1013im_b = self.twiddle10.im * x122n.re
            + -self.twiddle3.im * x221n.re
            + self.twiddle7.im * x320n.re
            + -self.twiddle6.im * x419n.re
            + self.twiddle4.im * x518n.re
            + -self.twiddle9.im * x617n.re
            + self.twiddle1.im * x716n.re
            + self.twiddle11.im * x815n.re
            + -self.twiddle2.im * x914n.re
            + self.twiddle8.im * x1013n.re
            + -self.twiddle5.im * x1112n.re;
        let b1112im_a = input.load(0).im
            + self.twiddle11.re * x122p.im
            + self.twiddle1.re * x221p.im
            + self.twiddle10.re * x320p.im
            + self.twiddle2.re * x419p.im
            + self.twiddle9.re * x518p.im
            + self.twiddle3.re * x617p.im
            + self.twiddle8.re * x716p.im
            + self.twiddle4.re * x815p.im
            + self.twiddle7.re * x914p.im
            + self.twiddle5.re * x1013p.im
            + self.twiddle6.re * x1112p.im;
        let b1112im_b = self.twiddle11.im * x122n.re
            + -self.twiddle1.im * x221n.re
            + self.twiddle10.im * x320n.re
            + -self.twiddle2.im * x419n.re
            + self.twiddle9.im * x518n.re
            + -self.twiddle3.im * x617n.re
            + self.twiddle8.im * x716n.re
            + -self.twiddle4.im * x815n.re
            + self.twiddle7.im * x914n.re
            + -self.twiddle5.im * x1013n.re
            + self.twiddle6.im * x1112n.re;

        let out1re = b122re_a - b122re_b;
        let out1im = b122im_a + b122im_b;
        let out2re = b221re_a - b221re_b;
        let out2im = b221im_a + b221im_b;
        let out3re = b320re_a - b320re_b;
        let out3im = b320im_a + b320im_b;
        let out4re = b419re_a - b419re_b;
        let out4im = b419im_a + b419im_b;
        let out5re = b518re_a - b518re_b;
        let out5im = b518im_a + b518im_b;
        let out6re = b617re_a - b617re_b;
        let out6im = b617im_a + b617im_b;
        let out7re = b716re_a - b716re_b;
        let out7im = b716im_a + b716im_b;
        let out8re = b815re_a - b815re_b;
        let out8im = b815im_a + b815im_b;
        let out9re = b914re_a - b914re_b;
        let out9im = b914im_a + b914im_b;
        let out10re = b1013re_a - b1013re_b;
        let out10im = b1013im_a + b1013im_b;
        let out11re = b1112re_a - b1112re_b;
        let out11im = b1112im_a + b1112im_b;
        let out12re = b1112re_a + b1112re_b;
        let out12im = b1112im_a - b1112im_b;
        let out13re = b1013re_a + b1013re_b;
        let out13im = b1013im_a - b1013im_b;
        let out14re = b914re_a + b914re_b;
        let out14im = b914im_a - b914im_b;
        let out15re = b815re_a + b815re_b;
        let out15im = b815im_a - b815im_b;
        let out16re = b716re_a + b716re_b;
        let out16im = b716im_a - b716im_b;
        let out17re = b617re_a + b617re_b;
        let out17im = b617im_a - b617im_b;
        let out18re = b518re_a + b518re_b;
        let out18im = b518im_a - b518im_b;
        let out19re = b419re_a + b419re_b;
        let out19im = b419im_a - b419im_b;
        let out20re = b320re_a + b320re_b;
        let out20im = b320im_a - b320im_b;
        let out21re = b221re_a + b221re_b;
        let out21im = b221im_a - b221im_b;
        let out22re = b122re_a + b122re_b;
        let out22im = b122im_a - b122im_b;
        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        output.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        output.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
        output.store(
            Complex {
                re: out7re,
                im: out7im,
            },
            7,
        );
        output.store(
            Complex {
                re: out8re,
                im: out8im,
            },
            8,
        );
        output.store(
            Complex {
                re: out9re,
                im: out9im,
            },
            9,
        );
        output.store(
            Complex {
                re: out10re,
                im: out10im,
            },
            10,
        );
        output.store(
            Complex {
                re: out11re,
                im: out11im,
            },
            11,
        );
        output.store(
            Complex {
                re: out12re,
                im: out12im,
            },
            12,
        );
        output.store(
            Complex {
                re: out13re,
                im: out13im,
            },
            13,
        );
        output.store(
            Complex {
                re: out14re,
                im: out14im,
            },
            14,
        );
        output.store(
            Complex {
                re: out15re,
                im: out15im,
            },
            15,
        );
        output.store(
            Complex {
                re: out16re,
                im: out16im,
            },
            16,
        );
        output.store(
            Complex {
                re: out17re,
                im: out17im,
            },
            17,
        );
        output.store(
            Complex {
                re: out18re,
                im: out18im,
            },
            18,
        );
        output.store(
            Complex {
                re: out19re,
                im: out19im,
            },
            19,
        );
        output.store(
            Complex {
                re: out20re,
                im: out20im,
            },
            20,
        );
        output.store(
            Complex {
                re: out21re,
                im: out21im,
            },
            21,
        );
        output.store(
            Complex {
                re: out22re,
                im: out22im,
            },
            22,
        );
    }
}

pub struct Butterfly27<T> {
    butterfly9: Butterfly9<T>,
    twiddles: [Complex<T>; 12],
}
boilerplate_fft_butterfly!(Butterfly27, 27, |this: &Butterfly27<_>| this
    .butterfly9
    .fft_direction());
impl<T: FftNum> Butterfly27<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly9: Butterfly9::new(direction),
            twiddles: [
                twiddles::compute_twiddle(1, 27, direction),
                twiddles::compute_twiddle(2, 27, direction),
                twiddles::compute_twiddle(3, 27, direction),
                twiddles::compute_twiddle(4, 27, direction),
                twiddles::compute_twiddle(5, 27, direction),
                twiddles::compute_twiddle(6, 27, direction),
                twiddles::compute_twiddle(7, 27, direction),
                twiddles::compute_twiddle(8, 27, direction),
                twiddles::compute_twiddle(10, 27, direction),
                twiddles::compute_twiddle(12, 27, direction),
                twiddles::compute_twiddle(14, 27, direction),
                twiddles::compute_twiddle(16, 27, direction),
            ],
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // algorithm: mixed radix with width=9 and height=3

        // step 1: transpose the input into the scratch
        let mut scratch0 = [
            input.load(0),
            input.load(3),
            input.load(6),
            input.load(9),
            input.load(12),
            input.load(15),
            input.load(18),
            input.load(21),
            input.load(24),
        ];
        let mut scratch1 = [
            input.load(1 + 0),
            input.load(1 + 3),
            input.load(1 + 6),
            input.load(1 + 9),
            input.load(1 + 12),
            input.load(1 + 15),
            input.load(1 + 18),
            input.load(1 + 21),
            input.load(1 + 24),
        ];
        let mut scratch2 = [
            input.load(2 + 0),
            input.load(2 + 3),
            input.load(2 + 6),
            input.load(2 + 9),
            input.load(2 + 12),
            input.load(2 + 15),
            input.load(2 + 18),
            input.load(2 + 21),
            input.load(2 + 24),
        ];

        // step 2: column FFTs
        self.butterfly9.perform_fft_butterfly(&mut scratch0);
        self.butterfly9.perform_fft_butterfly(&mut scratch1);
        self.butterfly9.perform_fft_butterfly(&mut scratch2);

        // step 3: apply twiddle factors
        scratch1[1] = scratch1[1] * self.twiddles[0];
        scratch1[2] = scratch1[2] * self.twiddles[1];
        scratch1[3] = scratch1[3] * self.twiddles[2];
        scratch1[4] = scratch1[4] * self.twiddles[3];
        scratch1[5] = scratch1[5] * self.twiddles[4];
        scratch1[6] = scratch1[6] * self.twiddles[5];
        scratch1[7] = scratch1[7] * self.twiddles[6];
        scratch1[8] = scratch1[8] * self.twiddles[7];
        scratch2[1] = scratch2[1] * self.twiddles[1];
        scratch2[2] = scratch2[2] * self.twiddles[3];
        scratch2[3] = scratch2[3] * self.twiddles[5];
        scratch2[4] = scratch2[4] * self.twiddles[7];
        scratch2[5] = scratch2[5] * self.twiddles[8];
        scratch2[6] = scratch2[6] * self.twiddles[9];
        scratch2[7] = scratch2[7] * self.twiddles[10];
        scratch2[8] = scratch2[8] * self.twiddles[11];

        // step 4: SKIPPED because the next FFTs will be non-contiguous

        // step 5: row FFTs
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[0],
            &mut scratch1[0],
            &mut scratch2[0],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[1],
            &mut scratch1[1],
            &mut scratch2[1],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[2],
            &mut scratch1[2],
            &mut scratch2[2],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[3],
            &mut scratch1[3],
            &mut scratch2[3],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[4],
            &mut scratch1[4],
            &mut scratch2[4],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[5],
            &mut scratch1[5],
            &mut scratch2[5],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[6],
            &mut scratch1[6],
            &mut scratch2[6],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[7],
            &mut scratch1[7],
            &mut scratch2[7],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[8],
            &mut scratch1[8],
            &mut scratch2[8],
        );

        // step 6: copy the result into the output. normally we'd need to do a transpose here, but we can skip it because we skipped the transpose in step 4
        output.store(scratch0[0], 0);
        output.store(scratch0[1], 1);
        output.store(scratch0[2], 2);
        output.store(scratch0[3], 3);
        output.store(scratch0[4], 4);
        output.store(scratch0[5], 5);
        output.store(scratch0[6], 6);
        output.store(scratch0[7], 7);
        output.store(scratch0[8], 8);

        output.store(scratch1[0], 9 + 0);
        output.store(scratch1[1], 9 + 1);
        output.store(scratch1[2], 9 + 2);
        output.store(scratch1[3], 9 + 3);
        output.store(scratch1[4], 9 + 4);
        output.store(scratch1[5], 9 + 5);
        output.store(scratch1[6], 9 + 6);
        output.store(scratch1[7], 9 + 7);
        output.store(scratch1[8], 9 + 8);

        output.store(scratch2[0], 18 + 0);
        output.store(scratch2[1], 18 + 1);
        output.store(scratch2[2], 18 + 2);
        output.store(scratch2[3], 18 + 3);
        output.store(scratch2[4], 18 + 4);
        output.store(scratch2[5], 18 + 5);
        output.store(scratch2[6], 18 + 6);
        output.store(scratch2[7], 18 + 7);
        output.store(scratch2[8], 18 + 8);
    }
}

pub struct Butterfly29<T> {
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
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly29, 29, |this: &Butterfly29<_>| this.direction);
impl<T: FftNum> Butterfly29<T> {
    pub fn new(direction: FftDirection) -> Self {
        let twiddle1: Complex<T> = twiddles::compute_twiddle(1, 29, direction);
        let twiddle2: Complex<T> = twiddles::compute_twiddle(2, 29, direction);
        let twiddle3: Complex<T> = twiddles::compute_twiddle(3, 29, direction);
        let twiddle4: Complex<T> = twiddles::compute_twiddle(4, 29, direction);
        let twiddle5: Complex<T> = twiddles::compute_twiddle(5, 29, direction);
        let twiddle6: Complex<T> = twiddles::compute_twiddle(6, 29, direction);
        let twiddle7: Complex<T> = twiddles::compute_twiddle(7, 29, direction);
        let twiddle8: Complex<T> = twiddles::compute_twiddle(8, 29, direction);
        let twiddle9: Complex<T> = twiddles::compute_twiddle(9, 29, direction);
        let twiddle10: Complex<T> = twiddles::compute_twiddle(10, 29, direction);
        let twiddle11: Complex<T> = twiddles::compute_twiddle(11, 29, direction);
        let twiddle12: Complex<T> = twiddles::compute_twiddle(12, 29, direction);
        let twiddle13: Complex<T> = twiddles::compute_twiddle(13, 29, direction);
        let twiddle14: Complex<T> = twiddles::compute_twiddle(14, 29, direction);
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
            direction,
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x128p = input.load(1) + input.load(28);
        let x128n = input.load(1) - input.load(28);
        let x227p = input.load(2) + input.load(27);
        let x227n = input.load(2) - input.load(27);
        let x326p = input.load(3) + input.load(26);
        let x326n = input.load(3) - input.load(26);
        let x425p = input.load(4) + input.load(25);
        let x425n = input.load(4) - input.load(25);
        let x524p = input.load(5) + input.load(24);
        let x524n = input.load(5) - input.load(24);
        let x623p = input.load(6) + input.load(23);
        let x623n = input.load(6) - input.load(23);
        let x722p = input.load(7) + input.load(22);
        let x722n = input.load(7) - input.load(22);
        let x821p = input.load(8) + input.load(21);
        let x821n = input.load(8) - input.load(21);
        let x920p = input.load(9) + input.load(20);
        let x920n = input.load(9) - input.load(20);
        let x1019p = input.load(10) + input.load(19);
        let x1019n = input.load(10) - input.load(19);
        let x1118p = input.load(11) + input.load(18);
        let x1118n = input.load(11) - input.load(18);
        let x1217p = input.load(12) + input.load(17);
        let x1217n = input.load(12) - input.load(17);
        let x1316p = input.load(13) + input.load(16);
        let x1316n = input.load(13) - input.load(16);
        let x1415p = input.load(14) + input.load(15);
        let x1415n = input.load(14) - input.load(15);
        let sum = input.load(0)
            + x128p
            + x227p
            + x326p
            + x425p
            + x524p
            + x623p
            + x722p
            + x821p
            + x920p
            + x1019p
            + x1118p
            + x1217p
            + x1316p
            + x1415p;
        let b128re_a = input.load(0).re
            + self.twiddle1.re * x128p.re
            + self.twiddle2.re * x227p.re
            + self.twiddle3.re * x326p.re
            + self.twiddle4.re * x425p.re
            + self.twiddle5.re * x524p.re
            + self.twiddle6.re * x623p.re
            + self.twiddle7.re * x722p.re
            + self.twiddle8.re * x821p.re
            + self.twiddle9.re * x920p.re
            + self.twiddle10.re * x1019p.re
            + self.twiddle11.re * x1118p.re
            + self.twiddle12.re * x1217p.re
            + self.twiddle13.re * x1316p.re
            + self.twiddle14.re * x1415p.re;
        let b128re_b = self.twiddle1.im * x128n.im
            + self.twiddle2.im * x227n.im
            + self.twiddle3.im * x326n.im
            + self.twiddle4.im * x425n.im
            + self.twiddle5.im * x524n.im
            + self.twiddle6.im * x623n.im
            + self.twiddle7.im * x722n.im
            + self.twiddle8.im * x821n.im
            + self.twiddle9.im * x920n.im
            + self.twiddle10.im * x1019n.im
            + self.twiddle11.im * x1118n.im
            + self.twiddle12.im * x1217n.im
            + self.twiddle13.im * x1316n.im
            + self.twiddle14.im * x1415n.im;
        let b227re_a = input.load(0).re
            + self.twiddle2.re * x128p.re
            + self.twiddle4.re * x227p.re
            + self.twiddle6.re * x326p.re
            + self.twiddle8.re * x425p.re
            + self.twiddle10.re * x524p.re
            + self.twiddle12.re * x623p.re
            + self.twiddle14.re * x722p.re
            + self.twiddle13.re * x821p.re
            + self.twiddle11.re * x920p.re
            + self.twiddle9.re * x1019p.re
            + self.twiddle7.re * x1118p.re
            + self.twiddle5.re * x1217p.re
            + self.twiddle3.re * x1316p.re
            + self.twiddle1.re * x1415p.re;
        let b227re_b = self.twiddle2.im * x128n.im
            + self.twiddle4.im * x227n.im
            + self.twiddle6.im * x326n.im
            + self.twiddle8.im * x425n.im
            + self.twiddle10.im * x524n.im
            + self.twiddle12.im * x623n.im
            + self.twiddle14.im * x722n.im
            + -self.twiddle13.im * x821n.im
            + -self.twiddle11.im * x920n.im
            + -self.twiddle9.im * x1019n.im
            + -self.twiddle7.im * x1118n.im
            + -self.twiddle5.im * x1217n.im
            + -self.twiddle3.im * x1316n.im
            + -self.twiddle1.im * x1415n.im;
        let b326re_a = input.load(0).re
            + self.twiddle3.re * x128p.re
            + self.twiddle6.re * x227p.re
            + self.twiddle9.re * x326p.re
            + self.twiddle12.re * x425p.re
            + self.twiddle14.re * x524p.re
            + self.twiddle11.re * x623p.re
            + self.twiddle8.re * x722p.re
            + self.twiddle5.re * x821p.re
            + self.twiddle2.re * x920p.re
            + self.twiddle1.re * x1019p.re
            + self.twiddle4.re * x1118p.re
            + self.twiddle7.re * x1217p.re
            + self.twiddle10.re * x1316p.re
            + self.twiddle13.re * x1415p.re;
        let b326re_b = self.twiddle3.im * x128n.im
            + self.twiddle6.im * x227n.im
            + self.twiddle9.im * x326n.im
            + self.twiddle12.im * x425n.im
            + -self.twiddle14.im * x524n.im
            + -self.twiddle11.im * x623n.im
            + -self.twiddle8.im * x722n.im
            + -self.twiddle5.im * x821n.im
            + -self.twiddle2.im * x920n.im
            + self.twiddle1.im * x1019n.im
            + self.twiddle4.im * x1118n.im
            + self.twiddle7.im * x1217n.im
            + self.twiddle10.im * x1316n.im
            + self.twiddle13.im * x1415n.im;
        let b425re_a = input.load(0).re
            + self.twiddle4.re * x128p.re
            + self.twiddle8.re * x227p.re
            + self.twiddle12.re * x326p.re
            + self.twiddle13.re * x425p.re
            + self.twiddle9.re * x524p.re
            + self.twiddle5.re * x623p.re
            + self.twiddle1.re * x722p.re
            + self.twiddle3.re * x821p.re
            + self.twiddle7.re * x920p.re
            + self.twiddle11.re * x1019p.re
            + self.twiddle14.re * x1118p.re
            + self.twiddle10.re * x1217p.re
            + self.twiddle6.re * x1316p.re
            + self.twiddle2.re * x1415p.re;
        let b425re_b = self.twiddle4.im * x128n.im
            + self.twiddle8.im * x227n.im
            + self.twiddle12.im * x326n.im
            + -self.twiddle13.im * x425n.im
            + -self.twiddle9.im * x524n.im
            + -self.twiddle5.im * x623n.im
            + -self.twiddle1.im * x722n.im
            + self.twiddle3.im * x821n.im
            + self.twiddle7.im * x920n.im
            + self.twiddle11.im * x1019n.im
            + -self.twiddle14.im * x1118n.im
            + -self.twiddle10.im * x1217n.im
            + -self.twiddle6.im * x1316n.im
            + -self.twiddle2.im * x1415n.im;
        let b524re_a = input.load(0).re
            + self.twiddle5.re * x128p.re
            + self.twiddle10.re * x227p.re
            + self.twiddle14.re * x326p.re
            + self.twiddle9.re * x425p.re
            + self.twiddle4.re * x524p.re
            + self.twiddle1.re * x623p.re
            + self.twiddle6.re * x722p.re
            + self.twiddle11.re * x821p.re
            + self.twiddle13.re * x920p.re
            + self.twiddle8.re * x1019p.re
            + self.twiddle3.re * x1118p.re
            + self.twiddle2.re * x1217p.re
            + self.twiddle7.re * x1316p.re
            + self.twiddle12.re * x1415p.re;
        let b524re_b = self.twiddle5.im * x128n.im
            + self.twiddle10.im * x227n.im
            + -self.twiddle14.im * x326n.im
            + -self.twiddle9.im * x425n.im
            + -self.twiddle4.im * x524n.im
            + self.twiddle1.im * x623n.im
            + self.twiddle6.im * x722n.im
            + self.twiddle11.im * x821n.im
            + -self.twiddle13.im * x920n.im
            + -self.twiddle8.im * x1019n.im
            + -self.twiddle3.im * x1118n.im
            + self.twiddle2.im * x1217n.im
            + self.twiddle7.im * x1316n.im
            + self.twiddle12.im * x1415n.im;
        let b623re_a = input.load(0).re
            + self.twiddle6.re * x128p.re
            + self.twiddle12.re * x227p.re
            + self.twiddle11.re * x326p.re
            + self.twiddle5.re * x425p.re
            + self.twiddle1.re * x524p.re
            + self.twiddle7.re * x623p.re
            + self.twiddle13.re * x722p.re
            + self.twiddle10.re * x821p.re
            + self.twiddle4.re * x920p.re
            + self.twiddle2.re * x1019p.re
            + self.twiddle8.re * x1118p.re
            + self.twiddle14.re * x1217p.re
            + self.twiddle9.re * x1316p.re
            + self.twiddle3.re * x1415p.re;
        let b623re_b = self.twiddle6.im * x128n.im
            + self.twiddle12.im * x227n.im
            + -self.twiddle11.im * x326n.im
            + -self.twiddle5.im * x425n.im
            + self.twiddle1.im * x524n.im
            + self.twiddle7.im * x623n.im
            + self.twiddle13.im * x722n.im
            + -self.twiddle10.im * x821n.im
            + -self.twiddle4.im * x920n.im
            + self.twiddle2.im * x1019n.im
            + self.twiddle8.im * x1118n.im
            + self.twiddle14.im * x1217n.im
            + -self.twiddle9.im * x1316n.im
            + -self.twiddle3.im * x1415n.im;
        let b722re_a = input.load(0).re
            + self.twiddle7.re * x128p.re
            + self.twiddle14.re * x227p.re
            + self.twiddle8.re * x326p.re
            + self.twiddle1.re * x425p.re
            + self.twiddle6.re * x524p.re
            + self.twiddle13.re * x623p.re
            + self.twiddle9.re * x722p.re
            + self.twiddle2.re * x821p.re
            + self.twiddle5.re * x920p.re
            + self.twiddle12.re * x1019p.re
            + self.twiddle10.re * x1118p.re
            + self.twiddle3.re * x1217p.re
            + self.twiddle4.re * x1316p.re
            + self.twiddle11.re * x1415p.re;
        let b722re_b = self.twiddle7.im * x128n.im
            + self.twiddle14.im * x227n.im
            + -self.twiddle8.im * x326n.im
            + -self.twiddle1.im * x425n.im
            + self.twiddle6.im * x524n.im
            + self.twiddle13.im * x623n.im
            + -self.twiddle9.im * x722n.im
            + -self.twiddle2.im * x821n.im
            + self.twiddle5.im * x920n.im
            + self.twiddle12.im * x1019n.im
            + -self.twiddle10.im * x1118n.im
            + -self.twiddle3.im * x1217n.im
            + self.twiddle4.im * x1316n.im
            + self.twiddle11.im * x1415n.im;
        let b821re_a = input.load(0).re
            + self.twiddle8.re * x128p.re
            + self.twiddle13.re * x227p.re
            + self.twiddle5.re * x326p.re
            + self.twiddle3.re * x425p.re
            + self.twiddle11.re * x524p.re
            + self.twiddle10.re * x623p.re
            + self.twiddle2.re * x722p.re
            + self.twiddle6.re * x821p.re
            + self.twiddle14.re * x920p.re
            + self.twiddle7.re * x1019p.re
            + self.twiddle1.re * x1118p.re
            + self.twiddle9.re * x1217p.re
            + self.twiddle12.re * x1316p.re
            + self.twiddle4.re * x1415p.re;
        let b821re_b = self.twiddle8.im * x128n.im
            + -self.twiddle13.im * x227n.im
            + -self.twiddle5.im * x326n.im
            + self.twiddle3.im * x425n.im
            + self.twiddle11.im * x524n.im
            + -self.twiddle10.im * x623n.im
            + -self.twiddle2.im * x722n.im
            + self.twiddle6.im * x821n.im
            + self.twiddle14.im * x920n.im
            + -self.twiddle7.im * x1019n.im
            + self.twiddle1.im * x1118n.im
            + self.twiddle9.im * x1217n.im
            + -self.twiddle12.im * x1316n.im
            + -self.twiddle4.im * x1415n.im;
        let b920re_a = input.load(0).re
            + self.twiddle9.re * x128p.re
            + self.twiddle11.re * x227p.re
            + self.twiddle2.re * x326p.re
            + self.twiddle7.re * x425p.re
            + self.twiddle13.re * x524p.re
            + self.twiddle4.re * x623p.re
            + self.twiddle5.re * x722p.re
            + self.twiddle14.re * x821p.re
            + self.twiddle6.re * x920p.re
            + self.twiddle3.re * x1019p.re
            + self.twiddle12.re * x1118p.re
            + self.twiddle8.re * x1217p.re
            + self.twiddle1.re * x1316p.re
            + self.twiddle10.re * x1415p.re;
        let b920re_b = self.twiddle9.im * x128n.im
            + -self.twiddle11.im * x227n.im
            + -self.twiddle2.im * x326n.im
            + self.twiddle7.im * x425n.im
            + -self.twiddle13.im * x524n.im
            + -self.twiddle4.im * x623n.im
            + self.twiddle5.im * x722n.im
            + self.twiddle14.im * x821n.im
            + -self.twiddle6.im * x920n.im
            + self.twiddle3.im * x1019n.im
            + self.twiddle12.im * x1118n.im
            + -self.twiddle8.im * x1217n.im
            + self.twiddle1.im * x1316n.im
            + self.twiddle10.im * x1415n.im;
        let b1019re_a = input.load(0).re
            + self.twiddle10.re * x128p.re
            + self.twiddle9.re * x227p.re
            + self.twiddle1.re * x326p.re
            + self.twiddle11.re * x425p.re
            + self.twiddle8.re * x524p.re
            + self.twiddle2.re * x623p.re
            + self.twiddle12.re * x722p.re
            + self.twiddle7.re * x821p.re
            + self.twiddle3.re * x920p.re
            + self.twiddle13.re * x1019p.re
            + self.twiddle6.re * x1118p.re
            + self.twiddle4.re * x1217p.re
            + self.twiddle14.re * x1316p.re
            + self.twiddle5.re * x1415p.re;
        let b1019re_b = self.twiddle10.im * x128n.im
            + -self.twiddle9.im * x227n.im
            + self.twiddle1.im * x326n.im
            + self.twiddle11.im * x425n.im
            + -self.twiddle8.im * x524n.im
            + self.twiddle2.im * x623n.im
            + self.twiddle12.im * x722n.im
            + -self.twiddle7.im * x821n.im
            + self.twiddle3.im * x920n.im
            + self.twiddle13.im * x1019n.im
            + -self.twiddle6.im * x1118n.im
            + self.twiddle4.im * x1217n.im
            + self.twiddle14.im * x1316n.im
            + -self.twiddle5.im * x1415n.im;
        let b1118re_a = input.load(0).re
            + self.twiddle11.re * x128p.re
            + self.twiddle7.re * x227p.re
            + self.twiddle4.re * x326p.re
            + self.twiddle14.re * x425p.re
            + self.twiddle3.re * x524p.re
            + self.twiddle8.re * x623p.re
            + self.twiddle10.re * x722p.re
            + self.twiddle1.re * x821p.re
            + self.twiddle12.re * x920p.re
            + self.twiddle6.re * x1019p.re
            + self.twiddle5.re * x1118p.re
            + self.twiddle13.re * x1217p.re
            + self.twiddle2.re * x1316p.re
            + self.twiddle9.re * x1415p.re;
        let b1118re_b = self.twiddle11.im * x128n.im
            + -self.twiddle7.im * x227n.im
            + self.twiddle4.im * x326n.im
            + -self.twiddle14.im * x425n.im
            + -self.twiddle3.im * x524n.im
            + self.twiddle8.im * x623n.im
            + -self.twiddle10.im * x722n.im
            + self.twiddle1.im * x821n.im
            + self.twiddle12.im * x920n.im
            + -self.twiddle6.im * x1019n.im
            + self.twiddle5.im * x1118n.im
            + -self.twiddle13.im * x1217n.im
            + -self.twiddle2.im * x1316n.im
            + self.twiddle9.im * x1415n.im;
        let b1217re_a = input.load(0).re
            + self.twiddle12.re * x128p.re
            + self.twiddle5.re * x227p.re
            + self.twiddle7.re * x326p.re
            + self.twiddle10.re * x425p.re
            + self.twiddle2.re * x524p.re
            + self.twiddle14.re * x623p.re
            + self.twiddle3.re * x722p.re
            + self.twiddle9.re * x821p.re
            + self.twiddle8.re * x920p.re
            + self.twiddle4.re * x1019p.re
            + self.twiddle13.re * x1118p.re
            + self.twiddle1.re * x1217p.re
            + self.twiddle11.re * x1316p.re
            + self.twiddle6.re * x1415p.re;
        let b1217re_b = self.twiddle12.im * x128n.im
            + -self.twiddle5.im * x227n.im
            + self.twiddle7.im * x326n.im
            + -self.twiddle10.im * x425n.im
            + self.twiddle2.im * x524n.im
            + self.twiddle14.im * x623n.im
            + -self.twiddle3.im * x722n.im
            + self.twiddle9.im * x821n.im
            + -self.twiddle8.im * x920n.im
            + self.twiddle4.im * x1019n.im
            + -self.twiddle13.im * x1118n.im
            + -self.twiddle1.im * x1217n.im
            + self.twiddle11.im * x1316n.im
            + -self.twiddle6.im * x1415n.im;
        let b1316re_a = input.load(0).re
            + self.twiddle13.re * x128p.re
            + self.twiddle3.re * x227p.re
            + self.twiddle10.re * x326p.re
            + self.twiddle6.re * x425p.re
            + self.twiddle7.re * x524p.re
            + self.twiddle9.re * x623p.re
            + self.twiddle4.re * x722p.re
            + self.twiddle12.re * x821p.re
            + self.twiddle1.re * x920p.re
            + self.twiddle14.re * x1019p.re
            + self.twiddle2.re * x1118p.re
            + self.twiddle11.re * x1217p.re
            + self.twiddle5.re * x1316p.re
            + self.twiddle8.re * x1415p.re;
        let b1316re_b = self.twiddle13.im * x128n.im
            + -self.twiddle3.im * x227n.im
            + self.twiddle10.im * x326n.im
            + -self.twiddle6.im * x425n.im
            + self.twiddle7.im * x524n.im
            + -self.twiddle9.im * x623n.im
            + self.twiddle4.im * x722n.im
            + -self.twiddle12.im * x821n.im
            + self.twiddle1.im * x920n.im
            + self.twiddle14.im * x1019n.im
            + -self.twiddle2.im * x1118n.im
            + self.twiddle11.im * x1217n.im
            + -self.twiddle5.im * x1316n.im
            + self.twiddle8.im * x1415n.im;
        let b1415re_a = input.load(0).re
            + self.twiddle14.re * x128p.re
            + self.twiddle1.re * x227p.re
            + self.twiddle13.re * x326p.re
            + self.twiddle2.re * x425p.re
            + self.twiddle12.re * x524p.re
            + self.twiddle3.re * x623p.re
            + self.twiddle11.re * x722p.re
            + self.twiddle4.re * x821p.re
            + self.twiddle10.re * x920p.re
            + self.twiddle5.re * x1019p.re
            + self.twiddle9.re * x1118p.re
            + self.twiddle6.re * x1217p.re
            + self.twiddle8.re * x1316p.re
            + self.twiddle7.re * x1415p.re;
        let b1415re_b = self.twiddle14.im * x128n.im
            + -self.twiddle1.im * x227n.im
            + self.twiddle13.im * x326n.im
            + -self.twiddle2.im * x425n.im
            + self.twiddle12.im * x524n.im
            + -self.twiddle3.im * x623n.im
            + self.twiddle11.im * x722n.im
            + -self.twiddle4.im * x821n.im
            + self.twiddle10.im * x920n.im
            + -self.twiddle5.im * x1019n.im
            + self.twiddle9.im * x1118n.im
            + -self.twiddle6.im * x1217n.im
            + self.twiddle8.im * x1316n.im
            + -self.twiddle7.im * x1415n.im;

        let b128im_a = input.load(0).im
            + self.twiddle1.re * x128p.im
            + self.twiddle2.re * x227p.im
            + self.twiddle3.re * x326p.im
            + self.twiddle4.re * x425p.im
            + self.twiddle5.re * x524p.im
            + self.twiddle6.re * x623p.im
            + self.twiddle7.re * x722p.im
            + self.twiddle8.re * x821p.im
            + self.twiddle9.re * x920p.im
            + self.twiddle10.re * x1019p.im
            + self.twiddle11.re * x1118p.im
            + self.twiddle12.re * x1217p.im
            + self.twiddle13.re * x1316p.im
            + self.twiddle14.re * x1415p.im;
        let b128im_b = self.twiddle1.im * x128n.re
            + self.twiddle2.im * x227n.re
            + self.twiddle3.im * x326n.re
            + self.twiddle4.im * x425n.re
            + self.twiddle5.im * x524n.re
            + self.twiddle6.im * x623n.re
            + self.twiddle7.im * x722n.re
            + self.twiddle8.im * x821n.re
            + self.twiddle9.im * x920n.re
            + self.twiddle10.im * x1019n.re
            + self.twiddle11.im * x1118n.re
            + self.twiddle12.im * x1217n.re
            + self.twiddle13.im * x1316n.re
            + self.twiddle14.im * x1415n.re;
        let b227im_a = input.load(0).im
            + self.twiddle2.re * x128p.im
            + self.twiddle4.re * x227p.im
            + self.twiddle6.re * x326p.im
            + self.twiddle8.re * x425p.im
            + self.twiddle10.re * x524p.im
            + self.twiddle12.re * x623p.im
            + self.twiddle14.re * x722p.im
            + self.twiddle13.re * x821p.im
            + self.twiddle11.re * x920p.im
            + self.twiddle9.re * x1019p.im
            + self.twiddle7.re * x1118p.im
            + self.twiddle5.re * x1217p.im
            + self.twiddle3.re * x1316p.im
            + self.twiddle1.re * x1415p.im;
        let b227im_b = self.twiddle2.im * x128n.re
            + self.twiddle4.im * x227n.re
            + self.twiddle6.im * x326n.re
            + self.twiddle8.im * x425n.re
            + self.twiddle10.im * x524n.re
            + self.twiddle12.im * x623n.re
            + self.twiddle14.im * x722n.re
            + -self.twiddle13.im * x821n.re
            + -self.twiddle11.im * x920n.re
            + -self.twiddle9.im * x1019n.re
            + -self.twiddle7.im * x1118n.re
            + -self.twiddle5.im * x1217n.re
            + -self.twiddle3.im * x1316n.re
            + -self.twiddle1.im * x1415n.re;
        let b326im_a = input.load(0).im
            + self.twiddle3.re * x128p.im
            + self.twiddle6.re * x227p.im
            + self.twiddle9.re * x326p.im
            + self.twiddle12.re * x425p.im
            + self.twiddle14.re * x524p.im
            + self.twiddle11.re * x623p.im
            + self.twiddle8.re * x722p.im
            + self.twiddle5.re * x821p.im
            + self.twiddle2.re * x920p.im
            + self.twiddle1.re * x1019p.im
            + self.twiddle4.re * x1118p.im
            + self.twiddle7.re * x1217p.im
            + self.twiddle10.re * x1316p.im
            + self.twiddle13.re * x1415p.im;
        let b326im_b = self.twiddle3.im * x128n.re
            + self.twiddle6.im * x227n.re
            + self.twiddle9.im * x326n.re
            + self.twiddle12.im * x425n.re
            + -self.twiddle14.im * x524n.re
            + -self.twiddle11.im * x623n.re
            + -self.twiddle8.im * x722n.re
            + -self.twiddle5.im * x821n.re
            + -self.twiddle2.im * x920n.re
            + self.twiddle1.im * x1019n.re
            + self.twiddle4.im * x1118n.re
            + self.twiddle7.im * x1217n.re
            + self.twiddle10.im * x1316n.re
            + self.twiddle13.im * x1415n.re;
        let b425im_a = input.load(0).im
            + self.twiddle4.re * x128p.im
            + self.twiddle8.re * x227p.im
            + self.twiddle12.re * x326p.im
            + self.twiddle13.re * x425p.im
            + self.twiddle9.re * x524p.im
            + self.twiddle5.re * x623p.im
            + self.twiddle1.re * x722p.im
            + self.twiddle3.re * x821p.im
            + self.twiddle7.re * x920p.im
            + self.twiddle11.re * x1019p.im
            + self.twiddle14.re * x1118p.im
            + self.twiddle10.re * x1217p.im
            + self.twiddle6.re * x1316p.im
            + self.twiddle2.re * x1415p.im;
        let b425im_b = self.twiddle4.im * x128n.re
            + self.twiddle8.im * x227n.re
            + self.twiddle12.im * x326n.re
            + -self.twiddle13.im * x425n.re
            + -self.twiddle9.im * x524n.re
            + -self.twiddle5.im * x623n.re
            + -self.twiddle1.im * x722n.re
            + self.twiddle3.im * x821n.re
            + self.twiddle7.im * x920n.re
            + self.twiddle11.im * x1019n.re
            + -self.twiddle14.im * x1118n.re
            + -self.twiddle10.im * x1217n.re
            + -self.twiddle6.im * x1316n.re
            + -self.twiddle2.im * x1415n.re;
        let b524im_a = input.load(0).im
            + self.twiddle5.re * x128p.im
            + self.twiddle10.re * x227p.im
            + self.twiddle14.re * x326p.im
            + self.twiddle9.re * x425p.im
            + self.twiddle4.re * x524p.im
            + self.twiddle1.re * x623p.im
            + self.twiddle6.re * x722p.im
            + self.twiddle11.re * x821p.im
            + self.twiddle13.re * x920p.im
            + self.twiddle8.re * x1019p.im
            + self.twiddle3.re * x1118p.im
            + self.twiddle2.re * x1217p.im
            + self.twiddle7.re * x1316p.im
            + self.twiddle12.re * x1415p.im;
        let b524im_b = self.twiddle5.im * x128n.re
            + self.twiddle10.im * x227n.re
            + -self.twiddle14.im * x326n.re
            + -self.twiddle9.im * x425n.re
            + -self.twiddle4.im * x524n.re
            + self.twiddle1.im * x623n.re
            + self.twiddle6.im * x722n.re
            + self.twiddle11.im * x821n.re
            + -self.twiddle13.im * x920n.re
            + -self.twiddle8.im * x1019n.re
            + -self.twiddle3.im * x1118n.re
            + self.twiddle2.im * x1217n.re
            + self.twiddle7.im * x1316n.re
            + self.twiddle12.im * x1415n.re;
        let b623im_a = input.load(0).im
            + self.twiddle6.re * x128p.im
            + self.twiddle12.re * x227p.im
            + self.twiddle11.re * x326p.im
            + self.twiddle5.re * x425p.im
            + self.twiddle1.re * x524p.im
            + self.twiddle7.re * x623p.im
            + self.twiddle13.re * x722p.im
            + self.twiddle10.re * x821p.im
            + self.twiddle4.re * x920p.im
            + self.twiddle2.re * x1019p.im
            + self.twiddle8.re * x1118p.im
            + self.twiddle14.re * x1217p.im
            + self.twiddle9.re * x1316p.im
            + self.twiddle3.re * x1415p.im;
        let b623im_b = self.twiddle6.im * x128n.re
            + self.twiddle12.im * x227n.re
            + -self.twiddle11.im * x326n.re
            + -self.twiddle5.im * x425n.re
            + self.twiddle1.im * x524n.re
            + self.twiddle7.im * x623n.re
            + self.twiddle13.im * x722n.re
            + -self.twiddle10.im * x821n.re
            + -self.twiddle4.im * x920n.re
            + self.twiddle2.im * x1019n.re
            + self.twiddle8.im * x1118n.re
            + self.twiddle14.im * x1217n.re
            + -self.twiddle9.im * x1316n.re
            + -self.twiddle3.im * x1415n.re;
        let b722im_a = input.load(0).im
            + self.twiddle7.re * x128p.im
            + self.twiddle14.re * x227p.im
            + self.twiddle8.re * x326p.im
            + self.twiddle1.re * x425p.im
            + self.twiddle6.re * x524p.im
            + self.twiddle13.re * x623p.im
            + self.twiddle9.re * x722p.im
            + self.twiddle2.re * x821p.im
            + self.twiddle5.re * x920p.im
            + self.twiddle12.re * x1019p.im
            + self.twiddle10.re * x1118p.im
            + self.twiddle3.re * x1217p.im
            + self.twiddle4.re * x1316p.im
            + self.twiddle11.re * x1415p.im;
        let b722im_b = self.twiddle7.im * x128n.re
            + self.twiddle14.im * x227n.re
            + -self.twiddle8.im * x326n.re
            + -self.twiddle1.im * x425n.re
            + self.twiddle6.im * x524n.re
            + self.twiddle13.im * x623n.re
            + -self.twiddle9.im * x722n.re
            + -self.twiddle2.im * x821n.re
            + self.twiddle5.im * x920n.re
            + self.twiddle12.im * x1019n.re
            + -self.twiddle10.im * x1118n.re
            + -self.twiddle3.im * x1217n.re
            + self.twiddle4.im * x1316n.re
            + self.twiddle11.im * x1415n.re;
        let b821im_a = input.load(0).im
            + self.twiddle8.re * x128p.im
            + self.twiddle13.re * x227p.im
            + self.twiddle5.re * x326p.im
            + self.twiddle3.re * x425p.im
            + self.twiddle11.re * x524p.im
            + self.twiddle10.re * x623p.im
            + self.twiddle2.re * x722p.im
            + self.twiddle6.re * x821p.im
            + self.twiddle14.re * x920p.im
            + self.twiddle7.re * x1019p.im
            + self.twiddle1.re * x1118p.im
            + self.twiddle9.re * x1217p.im
            + self.twiddle12.re * x1316p.im
            + self.twiddle4.re * x1415p.im;
        let b821im_b = self.twiddle8.im * x128n.re
            + -self.twiddle13.im * x227n.re
            + -self.twiddle5.im * x326n.re
            + self.twiddle3.im * x425n.re
            + self.twiddle11.im * x524n.re
            + -self.twiddle10.im * x623n.re
            + -self.twiddle2.im * x722n.re
            + self.twiddle6.im * x821n.re
            + self.twiddle14.im * x920n.re
            + -self.twiddle7.im * x1019n.re
            + self.twiddle1.im * x1118n.re
            + self.twiddle9.im * x1217n.re
            + -self.twiddle12.im * x1316n.re
            + -self.twiddle4.im * x1415n.re;
        let b920im_a = input.load(0).im
            + self.twiddle9.re * x128p.im
            + self.twiddle11.re * x227p.im
            + self.twiddle2.re * x326p.im
            + self.twiddle7.re * x425p.im
            + self.twiddle13.re * x524p.im
            + self.twiddle4.re * x623p.im
            + self.twiddle5.re * x722p.im
            + self.twiddle14.re * x821p.im
            + self.twiddle6.re * x920p.im
            + self.twiddle3.re * x1019p.im
            + self.twiddle12.re * x1118p.im
            + self.twiddle8.re * x1217p.im
            + self.twiddle1.re * x1316p.im
            + self.twiddle10.re * x1415p.im;
        let b920im_b = self.twiddle9.im * x128n.re
            + -self.twiddle11.im * x227n.re
            + -self.twiddle2.im * x326n.re
            + self.twiddle7.im * x425n.re
            + -self.twiddle13.im * x524n.re
            + -self.twiddle4.im * x623n.re
            + self.twiddle5.im * x722n.re
            + self.twiddle14.im * x821n.re
            + -self.twiddle6.im * x920n.re
            + self.twiddle3.im * x1019n.re
            + self.twiddle12.im * x1118n.re
            + -self.twiddle8.im * x1217n.re
            + self.twiddle1.im * x1316n.re
            + self.twiddle10.im * x1415n.re;
        let b1019im_a = input.load(0).im
            + self.twiddle10.re * x128p.im
            + self.twiddle9.re * x227p.im
            + self.twiddle1.re * x326p.im
            + self.twiddle11.re * x425p.im
            + self.twiddle8.re * x524p.im
            + self.twiddle2.re * x623p.im
            + self.twiddle12.re * x722p.im
            + self.twiddle7.re * x821p.im
            + self.twiddle3.re * x920p.im
            + self.twiddle13.re * x1019p.im
            + self.twiddle6.re * x1118p.im
            + self.twiddle4.re * x1217p.im
            + self.twiddle14.re * x1316p.im
            + self.twiddle5.re * x1415p.im;
        let b1019im_b = self.twiddle10.im * x128n.re
            + -self.twiddle9.im * x227n.re
            + self.twiddle1.im * x326n.re
            + self.twiddle11.im * x425n.re
            + -self.twiddle8.im * x524n.re
            + self.twiddle2.im * x623n.re
            + self.twiddle12.im * x722n.re
            + -self.twiddle7.im * x821n.re
            + self.twiddle3.im * x920n.re
            + self.twiddle13.im * x1019n.re
            + -self.twiddle6.im * x1118n.re
            + self.twiddle4.im * x1217n.re
            + self.twiddle14.im * x1316n.re
            + -self.twiddle5.im * x1415n.re;
        let b1118im_a = input.load(0).im
            + self.twiddle11.re * x128p.im
            + self.twiddle7.re * x227p.im
            + self.twiddle4.re * x326p.im
            + self.twiddle14.re * x425p.im
            + self.twiddle3.re * x524p.im
            + self.twiddle8.re * x623p.im
            + self.twiddle10.re * x722p.im
            + self.twiddle1.re * x821p.im
            + self.twiddle12.re * x920p.im
            + self.twiddle6.re * x1019p.im
            + self.twiddle5.re * x1118p.im
            + self.twiddle13.re * x1217p.im
            + self.twiddle2.re * x1316p.im
            + self.twiddle9.re * x1415p.im;
        let b1118im_b = self.twiddle11.im * x128n.re
            + -self.twiddle7.im * x227n.re
            + self.twiddle4.im * x326n.re
            + -self.twiddle14.im * x425n.re
            + -self.twiddle3.im * x524n.re
            + self.twiddle8.im * x623n.re
            + -self.twiddle10.im * x722n.re
            + self.twiddle1.im * x821n.re
            + self.twiddle12.im * x920n.re
            + -self.twiddle6.im * x1019n.re
            + self.twiddle5.im * x1118n.re
            + -self.twiddle13.im * x1217n.re
            + -self.twiddle2.im * x1316n.re
            + self.twiddle9.im * x1415n.re;
        let b1217im_a = input.load(0).im
            + self.twiddle12.re * x128p.im
            + self.twiddle5.re * x227p.im
            + self.twiddle7.re * x326p.im
            + self.twiddle10.re * x425p.im
            + self.twiddle2.re * x524p.im
            + self.twiddle14.re * x623p.im
            + self.twiddle3.re * x722p.im
            + self.twiddle9.re * x821p.im
            + self.twiddle8.re * x920p.im
            + self.twiddle4.re * x1019p.im
            + self.twiddle13.re * x1118p.im
            + self.twiddle1.re * x1217p.im
            + self.twiddle11.re * x1316p.im
            + self.twiddle6.re * x1415p.im;
        let b1217im_b = self.twiddle12.im * x128n.re
            + -self.twiddle5.im * x227n.re
            + self.twiddle7.im * x326n.re
            + -self.twiddle10.im * x425n.re
            + self.twiddle2.im * x524n.re
            + self.twiddle14.im * x623n.re
            + -self.twiddle3.im * x722n.re
            + self.twiddle9.im * x821n.re
            + -self.twiddle8.im * x920n.re
            + self.twiddle4.im * x1019n.re
            + -self.twiddle13.im * x1118n.re
            + -self.twiddle1.im * x1217n.re
            + self.twiddle11.im * x1316n.re
            + -self.twiddle6.im * x1415n.re;
        let b1316im_a = input.load(0).im
            + self.twiddle13.re * x128p.im
            + self.twiddle3.re * x227p.im
            + self.twiddle10.re * x326p.im
            + self.twiddle6.re * x425p.im
            + self.twiddle7.re * x524p.im
            + self.twiddle9.re * x623p.im
            + self.twiddle4.re * x722p.im
            + self.twiddle12.re * x821p.im
            + self.twiddle1.re * x920p.im
            + self.twiddle14.re * x1019p.im
            + self.twiddle2.re * x1118p.im
            + self.twiddle11.re * x1217p.im
            + self.twiddle5.re * x1316p.im
            + self.twiddle8.re * x1415p.im;
        let b1316im_b = self.twiddle13.im * x128n.re
            + -self.twiddle3.im * x227n.re
            + self.twiddle10.im * x326n.re
            + -self.twiddle6.im * x425n.re
            + self.twiddle7.im * x524n.re
            + -self.twiddle9.im * x623n.re
            + self.twiddle4.im * x722n.re
            + -self.twiddle12.im * x821n.re
            + self.twiddle1.im * x920n.re
            + self.twiddle14.im * x1019n.re
            + -self.twiddle2.im * x1118n.re
            + self.twiddle11.im * x1217n.re
            + -self.twiddle5.im * x1316n.re
            + self.twiddle8.im * x1415n.re;
        let b1415im_a = input.load(0).im
            + self.twiddle14.re * x128p.im
            + self.twiddle1.re * x227p.im
            + self.twiddle13.re * x326p.im
            + self.twiddle2.re * x425p.im
            + self.twiddle12.re * x524p.im
            + self.twiddle3.re * x623p.im
            + self.twiddle11.re * x722p.im
            + self.twiddle4.re * x821p.im
            + self.twiddle10.re * x920p.im
            + self.twiddle5.re * x1019p.im
            + self.twiddle9.re * x1118p.im
            + self.twiddle6.re * x1217p.im
            + self.twiddle8.re * x1316p.im
            + self.twiddle7.re * x1415p.im;
        let b1415im_b = self.twiddle14.im * x128n.re
            + -self.twiddle1.im * x227n.re
            + self.twiddle13.im * x326n.re
            + -self.twiddle2.im * x425n.re
            + self.twiddle12.im * x524n.re
            + -self.twiddle3.im * x623n.re
            + self.twiddle11.im * x722n.re
            + -self.twiddle4.im * x821n.re
            + self.twiddle10.im * x920n.re
            + -self.twiddle5.im * x1019n.re
            + self.twiddle9.im * x1118n.re
            + -self.twiddle6.im * x1217n.re
            + self.twiddle8.im * x1316n.re
            + -self.twiddle7.im * x1415n.re;

        let out1re = b128re_a - b128re_b;
        let out1im = b128im_a + b128im_b;
        let out2re = b227re_a - b227re_b;
        let out2im = b227im_a + b227im_b;
        let out3re = b326re_a - b326re_b;
        let out3im = b326im_a + b326im_b;
        let out4re = b425re_a - b425re_b;
        let out4im = b425im_a + b425im_b;
        let out5re = b524re_a - b524re_b;
        let out5im = b524im_a + b524im_b;
        let out6re = b623re_a - b623re_b;
        let out6im = b623im_a + b623im_b;
        let out7re = b722re_a - b722re_b;
        let out7im = b722im_a + b722im_b;
        let out8re = b821re_a - b821re_b;
        let out8im = b821im_a + b821im_b;
        let out9re = b920re_a - b920re_b;
        let out9im = b920im_a + b920im_b;
        let out10re = b1019re_a - b1019re_b;
        let out10im = b1019im_a + b1019im_b;
        let out11re = b1118re_a - b1118re_b;
        let out11im = b1118im_a + b1118im_b;
        let out12re = b1217re_a - b1217re_b;
        let out12im = b1217im_a + b1217im_b;
        let out13re = b1316re_a - b1316re_b;
        let out13im = b1316im_a + b1316im_b;
        let out14re = b1415re_a - b1415re_b;
        let out14im = b1415im_a + b1415im_b;
        let out15re = b1415re_a + b1415re_b;
        let out15im = b1415im_a - b1415im_b;
        let out16re = b1316re_a + b1316re_b;
        let out16im = b1316im_a - b1316im_b;
        let out17re = b1217re_a + b1217re_b;
        let out17im = b1217im_a - b1217im_b;
        let out18re = b1118re_a + b1118re_b;
        let out18im = b1118im_a - b1118im_b;
        let out19re = b1019re_a + b1019re_b;
        let out19im = b1019im_a - b1019im_b;
        let out20re = b920re_a + b920re_b;
        let out20im = b920im_a - b920im_b;
        let out21re = b821re_a + b821re_b;
        let out21im = b821im_a - b821im_b;
        let out22re = b722re_a + b722re_b;
        let out22im = b722im_a - b722im_b;
        let out23re = b623re_a + b623re_b;
        let out23im = b623im_a - b623im_b;
        let out24re = b524re_a + b524re_b;
        let out24im = b524im_a - b524im_b;
        let out25re = b425re_a + b425re_b;
        let out25im = b425im_a - b425im_b;
        let out26re = b326re_a + b326re_b;
        let out26im = b326im_a - b326im_b;
        let out27re = b227re_a + b227re_b;
        let out27im = b227im_a - b227im_b;
        let out28re = b128re_a + b128re_b;
        let out28im = b128im_a - b128im_b;
        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        output.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        output.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
        output.store(
            Complex {
                re: out7re,
                im: out7im,
            },
            7,
        );
        output.store(
            Complex {
                re: out8re,
                im: out8im,
            },
            8,
        );
        output.store(
            Complex {
                re: out9re,
                im: out9im,
            },
            9,
        );
        output.store(
            Complex {
                re: out10re,
                im: out10im,
            },
            10,
        );
        output.store(
            Complex {
                re: out11re,
                im: out11im,
            },
            11,
        );
        output.store(
            Complex {
                re: out12re,
                im: out12im,
            },
            12,
        );
        output.store(
            Complex {
                re: out13re,
                im: out13im,
            },
            13,
        );
        output.store(
            Complex {
                re: out14re,
                im: out14im,
            },
            14,
        );
        output.store(
            Complex {
                re: out15re,
                im: out15im,
            },
            15,
        );
        output.store(
            Complex {
                re: out16re,
                im: out16im,
            },
            16,
        );
        output.store(
            Complex {
                re: out17re,
                im: out17im,
            },
            17,
        );
        output.store(
            Complex {
                re: out18re,
                im: out18im,
            },
            18,
        );
        output.store(
            Complex {
                re: out19re,
                im: out19im,
            },
            19,
        );
        output.store(
            Complex {
                re: out20re,
                im: out20im,
            },
            20,
        );
        output.store(
            Complex {
                re: out21re,
                im: out21im,
            },
            21,
        );
        output.store(
            Complex {
                re: out22re,
                im: out22im,
            },
            22,
        );
        output.store(
            Complex {
                re: out23re,
                im: out23im,
            },
            23,
        );
        output.store(
            Complex {
                re: out24re,
                im: out24im,
            },
            24,
        );
        output.store(
            Complex {
                re: out25re,
                im: out25im,
            },
            25,
        );
        output.store(
            Complex {
                re: out26re,
                im: out26im,
            },
            26,
        );
        output.store(
            Complex {
                re: out27re,
                im: out27im,
            },
            27,
        );
        output.store(
            Complex {
                re: out28re,
                im: out28im,
            },
            28,
        );
    }
}
pub struct Butterfly31<T> {
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
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly31, 31, |this: &Butterfly31<_>| this.direction);
impl<T: FftNum> Butterfly31<T> {
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
            direction,
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x130p = input.load(1) + input.load(30);
        let x130n = input.load(1) - input.load(30);
        let x229p = input.load(2) + input.load(29);
        let x229n = input.load(2) - input.load(29);
        let x328p = input.load(3) + input.load(28);
        let x328n = input.load(3) - input.load(28);
        let x427p = input.load(4) + input.load(27);
        let x427n = input.load(4) - input.load(27);
        let x526p = input.load(5) + input.load(26);
        let x526n = input.load(5) - input.load(26);
        let x625p = input.load(6) + input.load(25);
        let x625n = input.load(6) - input.load(25);
        let x724p = input.load(7) + input.load(24);
        let x724n = input.load(7) - input.load(24);
        let x823p = input.load(8) + input.load(23);
        let x823n = input.load(8) - input.load(23);
        let x922p = input.load(9) + input.load(22);
        let x922n = input.load(9) - input.load(22);
        let x1021p = input.load(10) + input.load(21);
        let x1021n = input.load(10) - input.load(21);
        let x1120p = input.load(11) + input.load(20);
        let x1120n = input.load(11) - input.load(20);
        let x1219p = input.load(12) + input.load(19);
        let x1219n = input.load(12) - input.load(19);
        let x1318p = input.load(13) + input.load(18);
        let x1318n = input.load(13) - input.load(18);
        let x1417p = input.load(14) + input.load(17);
        let x1417n = input.load(14) - input.load(17);
        let x1516p = input.load(15) + input.load(16);
        let x1516n = input.load(15) - input.load(16);
        let sum = input.load(0)
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
        let b130re_a = input.load(0).re
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
        let b229re_a = input.load(0).re
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
        let b328re_a = input.load(0).re
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
        let b427re_a = input.load(0).re
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
        let b526re_a = input.load(0).re
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
        let b625re_a = input.load(0).re
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
        let b724re_a = input.load(0).re
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
        let b823re_a = input.load(0).re
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
        let b922re_a = input.load(0).re
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
        let b1021re_a = input.load(0).re
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
        let b1120re_a = input.load(0).re
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
        let b1219re_a = input.load(0).re
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
        let b1318re_a = input.load(0).re
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
        let b1417re_a = input.load(0).re
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
        let b1516re_a = input.load(0).re
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

        let b130im_a = input.load(0).im
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
        let b229im_a = input.load(0).im
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
        let b328im_a = input.load(0).im
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
        let b427im_a = input.load(0).im
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
        let b526im_a = input.load(0).im
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
        let b625im_a = input.load(0).im
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
        let b724im_a = input.load(0).im
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
        let b823im_a = input.load(0).im
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
        let b922im_a = input.load(0).im
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
        let b1021im_a = input.load(0).im
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
        let b1120im_a = input.load(0).im
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
        let b1219im_a = input.load(0).im
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
        let b1318im_a = input.load(0).im
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
        let b1417im_a = input.load(0).im
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
        let b1516im_a = input.load(0).im
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
        output.store(sum, 0);
        output.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        output.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        output.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        output.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        output.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        output.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
        output.store(
            Complex {
                re: out7re,
                im: out7im,
            },
            7,
        );
        output.store(
            Complex {
                re: out8re,
                im: out8im,
            },
            8,
        );
        output.store(
            Complex {
                re: out9re,
                im: out9im,
            },
            9,
        );
        output.store(
            Complex {
                re: out10re,
                im: out10im,
            },
            10,
        );
        output.store(
            Complex {
                re: out11re,
                im: out11im,
            },
            11,
        );
        output.store(
            Complex {
                re: out12re,
                im: out12im,
            },
            12,
        );
        output.store(
            Complex {
                re: out13re,
                im: out13im,
            },
            13,
        );
        output.store(
            Complex {
                re: out14re,
                im: out14im,
            },
            14,
        );
        output.store(
            Complex {
                re: out15re,
                im: out15im,
            },
            15,
        );
        output.store(
            Complex {
                re: out16re,
                im: out16im,
            },
            16,
        );
        output.store(
            Complex {
                re: out17re,
                im: out17im,
            },
            17,
        );
        output.store(
            Complex {
                re: out18re,
                im: out18im,
            },
            18,
        );
        output.store(
            Complex {
                re: out19re,
                im: out19im,
            },
            19,
        );
        output.store(
            Complex {
                re: out20re,
                im: out20im,
            },
            20,
        );
        output.store(
            Complex {
                re: out21re,
                im: out21im,
            },
            21,
        );
        output.store(
            Complex {
                re: out22re,
                im: out22im,
            },
            22,
        );
        output.store(
            Complex {
                re: out23re,
                im: out23im,
            },
            23,
        );
        output.store(
            Complex {
                re: out24re,
                im: out24im,
            },
            24,
        );
        output.store(
            Complex {
                re: out25re,
                im: out25im,
            },
            25,
        );
        output.store(
            Complex {
                re: out26re,
                im: out26im,
            },
            26,
        );
        output.store(
            Complex {
                re: out27re,
                im: out27im,
            },
            27,
        );
        output.store(
            Complex {
                re: out28re,
                im: out28im,
            },
            28,
        );
        output.store(
            Complex {
                re: out29re,
                im: out29im,
            },
            29,
        );
        output.store(
            Complex {
                re: out30re,
                im: out30im,
            },
            30,
        );
    }
}
pub struct Butterfly32<T> {
    butterfly16: Butterfly16<T>,
    butterfly8: Butterfly8<T>,
    twiddles: [Complex<T>; 7],
}
boilerplate_fft_butterfly!(Butterfly32, 32, |this: &Butterfly32<_>| this
    .butterfly8
    .fft_direction());
impl<T: FftNum> Butterfly32<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly16: Butterfly16::new(direction),
            butterfly8: Butterfly8::new(direction),
            twiddles: [
                twiddles::compute_twiddle(1, 32, direction),
                twiddles::compute_twiddle(2, 32, direction),
                twiddles::compute_twiddle(3, 32, direction),
                twiddles::compute_twiddle(4, 32, direction),
                twiddles::compute_twiddle(5, 32, direction),
                twiddles::compute_twiddle(6, 32, direction),
                twiddles::compute_twiddle(7, 32, direction),
            ],
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(
        &self,
        input: RawSlice<Complex<T>>,
        output: RawSliceMut<Complex<T>>,
    ) {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        let mut scratch_evens = [
            input.load(0),
            input.load(2),
            input.load(4),
            input.load(6),
            input.load(8),
            input.load(10),
            input.load(12),
            input.load(14),
            input.load(16),
            input.load(18),
            input.load(20),
            input.load(22),
            input.load(24),
            input.load(26),
            input.load(28),
            input.load(30),
        ];

        let mut scratch_odds_n1 = [
            input.load(1),
            input.load(5),
            input.load(9),
            input.load(13),
            input.load(17),
            input.load(21),
            input.load(25),
            input.load(29),
        ];
        let mut scratch_odds_n3 = [
            input.load(31),
            input.load(3),
            input.load(7),
            input.load(11),
            input.load(15),
            input.load(19),
            input.load(23),
            input.load(27),
        ];

        // step 2: column FFTs
        self.butterfly16.perform_fft_butterfly(&mut scratch_evens);
        self.butterfly8.perform_fft_butterfly(&mut scratch_odds_n1);
        self.butterfly8.perform_fft_butterfly(&mut scratch_odds_n3);

        // step 3: apply twiddle factors
        scratch_odds_n1[1] = scratch_odds_n1[1] * self.twiddles[0];
        scratch_odds_n3[1] = scratch_odds_n3[1] * self.twiddles[0].conj();

        scratch_odds_n1[2] = scratch_odds_n1[2] * self.twiddles[1];
        scratch_odds_n3[2] = scratch_odds_n3[2] * self.twiddles[1].conj();

        scratch_odds_n1[3] = scratch_odds_n1[3] * self.twiddles[2];
        scratch_odds_n3[3] = scratch_odds_n3[3] * self.twiddles[2].conj();

        scratch_odds_n1[4] = scratch_odds_n1[4] * self.twiddles[3];
        scratch_odds_n3[4] = scratch_odds_n3[4] * self.twiddles[3].conj();

        scratch_odds_n1[5] = scratch_odds_n1[5] * self.twiddles[4];
        scratch_odds_n3[5] = scratch_odds_n3[5] * self.twiddles[4].conj();

        scratch_odds_n1[6] = scratch_odds_n1[6] * self.twiddles[5];
        scratch_odds_n3[6] = scratch_odds_n3[6] * self.twiddles[5].conj();

        scratch_odds_n1[7] = scratch_odds_n1[7] * self.twiddles[6];
        scratch_odds_n3[7] = scratch_odds_n3[7] * self.twiddles[6].conj();

        // step 4: cross FFTs
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[0], &mut scratch_odds_n3[0]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[1], &mut scratch_odds_n3[1]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[2], &mut scratch_odds_n3[2]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[3], &mut scratch_odds_n3[3]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[4], &mut scratch_odds_n3[4]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[5], &mut scratch_odds_n3[5]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[6], &mut scratch_odds_n3[6]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[7], &mut scratch_odds_n3[7]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.fft_direction());
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.fft_direction());
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.fft_direction());
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.fft_direction());
        scratch_odds_n3[4] = twiddles::rotate_90(scratch_odds_n3[4], self.fft_direction());
        scratch_odds_n3[5] = twiddles::rotate_90(scratch_odds_n3[5], self.fft_direction());
        scratch_odds_n3[6] = twiddles::rotate_90(scratch_odds_n3[6], self.fft_direction());
        scratch_odds_n3[7] = twiddles::rotate_90(scratch_odds_n3[7], self.fft_direction());

        //step 5: copy/add/subtract data back to buffer
        output.store(scratch_evens[0] + scratch_odds_n1[0], 0);
        output.store(scratch_evens[1] + scratch_odds_n1[1], 1);
        output.store(scratch_evens[2] + scratch_odds_n1[2], 2);
        output.store(scratch_evens[3] + scratch_odds_n1[3], 3);
        output.store(scratch_evens[4] + scratch_odds_n1[4], 4);
        output.store(scratch_evens[5] + scratch_odds_n1[5], 5);
        output.store(scratch_evens[6] + scratch_odds_n1[6], 6);
        output.store(scratch_evens[7] + scratch_odds_n1[7], 7);
        output.store(scratch_evens[8] + scratch_odds_n3[0], 8);
        output.store(scratch_evens[9] + scratch_odds_n3[1], 9);
        output.store(scratch_evens[10] + scratch_odds_n3[2], 10);
        output.store(scratch_evens[11] + scratch_odds_n3[3], 11);
        output.store(scratch_evens[12] + scratch_odds_n3[4], 12);
        output.store(scratch_evens[13] + scratch_odds_n3[5], 13);
        output.store(scratch_evens[14] + scratch_odds_n3[6], 14);
        output.store(scratch_evens[15] + scratch_odds_n3[7], 15);
        output.store(scratch_evens[0] - scratch_odds_n1[0], 16);
        output.store(scratch_evens[1] - scratch_odds_n1[1], 17);
        output.store(scratch_evens[2] - scratch_odds_n1[2], 18);
        output.store(scratch_evens[3] - scratch_odds_n1[3], 19);
        output.store(scratch_evens[4] - scratch_odds_n1[4], 20);
        output.store(scratch_evens[5] - scratch_odds_n1[5], 21);
        output.store(scratch_evens[6] - scratch_odds_n1[6], 22);
        output.store(scratch_evens[7] - scratch_odds_n1[7], 23);
        output.store(scratch_evens[8] - scratch_odds_n3[0], 24);
        output.store(scratch_evens[9] - scratch_odds_n3[1], 25);
        output.store(scratch_evens[10] - scratch_odds_n3[2], 26);
        output.store(scratch_evens[11] - scratch_odds_n3[3], 27);
        output.store(scratch_evens[12] - scratch_odds_n3[4], 28);
        output.store(scratch_evens[13] - scratch_odds_n3[5], 29);
        output.store(scratch_evens[14] - scratch_odds_n3[6], 30);
        output.store(scratch_evens[15] - scratch_odds_n3[7], 31);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(FftDirection::Forward);
                check_fft_algorithm::<f32>(&butterfly, $size, FftDirection::Forward);

                let butterfly_direction = $struct_name::new(FftDirection::Inverse);
                check_fft_algorithm::<f32>(&butterfly_direction, $size, FftDirection::Inverse);
            }
        };
    }
    test_butterfly_func!(test_butterfly2, Butterfly2, 2);
    test_butterfly_func!(test_butterfly3, Butterfly3, 3);
    test_butterfly_func!(test_butterfly4, Butterfly4, 4);
    test_butterfly_func!(test_butterfly5, Butterfly5, 5);
    test_butterfly_func!(test_butterfly6, Butterfly6, 6);
    test_butterfly_func!(test_butterfly7, Butterfly7, 7);
    test_butterfly_func!(test_butterfly8, Butterfly8, 8);
    test_butterfly_func!(test_butterfly9, Butterfly9, 9);
    test_butterfly_func!(test_butterfly11, Butterfly11, 11);
    test_butterfly_func!(test_butterfly13, Butterfly13, 13);
    test_butterfly_func!(test_butterfly16, Butterfly16, 16);
    test_butterfly_func!(test_butterfly17, Butterfly17, 17);
    test_butterfly_func!(test_butterfly19, Butterfly19, 19);
    test_butterfly_func!(test_butterfly23, Butterfly23, 23);
    test_butterfly_func!(test_butterfly27, Butterfly27, 27);
    test_butterfly_func!(test_butterfly29, Butterfly29, 29);
    test_butterfly_func!(test_butterfly31, Butterfly31, 31);
    test_butterfly_func!(test_butterfly32, Butterfly32, 32);
}
