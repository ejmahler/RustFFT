use num_complex::Complex;

use crate::common::{verify_length, verify_length_divisible, FFTnum};

use crate::twiddles;
use crate::{IsInverse, Length, FFT};

pub trait FFTButterfly<T: FFTnum>: Length + IsInverse + Sync + Send {
    /// Computes the FFT in-place in the given buffer
    ///
    /// # Safety
    /// This method performs unsafe reads/writes on `buffer`. Make sure `buffer.len()` is equal to `self.len()`
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]);

    /// Divides the given buffer into chunks of length `self.len()` and computes an in-place FFT on each chunk
    ///
    /// # Safety
    /// This method performs unsafe reads/writes on `buffer`. Make sure `buffer.len()` is a multiple of `self.len()`
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]);
}

#[inline(always)]
unsafe fn swap_unchecked<T: Copy>(buffer: &mut [T], a: usize, b: usize) {
    let temp = *buffer.get_unchecked(a);
    *buffer.get_unchecked_mut(a) = *buffer.get_unchecked(b);
    *buffer.get_unchecked_mut(b) = temp;
}

pub struct Butterfly2 {
    inverse: bool,
}
impl Butterfly2 {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly2 { inverse: inverse }
    }

    #[inline(always)]
    unsafe fn perform_fft_direct<T: FFTnum>(left: &mut Complex<T>, right: &mut Complex<T>) {
        let temp = *left + *right;

        *right = *left - *right;
        *left = temp;
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly2 {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        let temp = *buffer.get_unchecked(0) + *buffer.get_unchecked(1);

        *buffer.get_unchecked_mut(1) = *buffer.get_unchecked(0) - *buffer.get_unchecked(1);
        *buffer.get_unchecked_mut(0) = temp;
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
}
impl<T: FFTnum> FFT<T> for Butterfly2 {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl Length for Butterfly2 {
    #[inline(always)]
    fn len(&self) -> usize {
        2
    }
}
impl IsInverse for Butterfly2 {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly3<T> {
    pub twiddle: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly3<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly3 {
            twiddle: twiddles::single_twiddle(1, 3, inverse),
            inverse: inverse,
        }
    }

    #[inline(always)]
    pub fn inverse_of(fft: &Butterfly3<T>) -> Self {
        Butterfly3 {
            twiddle: fft.twiddle.conj(),
            inverse: !fft.inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly3<T> {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // Let's do a plain 3-point DFT
        // |X0|   | W0 W0  W0 |   |x0|
        // |X1| = | W0 W1  W2 | * |x1|
        // |X2|   | W0 W2  W4 |   |x2|
        //
        // where Wn = exp(-2*pi*n/3) for a forward transform, and exp(+2*pi*n/3) for an inverse.
        //
        // This can be simplified a bit since exp(-2*pi*n/3) = exp(-2*pi*n/3 + m*2*pi)
        // |X0|   | W0 W0  W0 |   |x0|
        // |X1| = | W0 W1  W2 | * |x1|
        // |X2|   | W0 W2  W1 |   |x2|
        //
        // Next we can use the symmetry that W2 = W1* and W0 = 1
        // |X0|   | 1  1   1   |   |x0|
        // |X1| = | 1  W1  W1* | * |x1|
        // |X2|   | 1  W1* W1  |   |x2|
        //
        // Next, we write out the whole expression with real and imaginary parts.
        // X0 = x0 + x1 + x2
        // X1 = x0 + (W1.re + j*W1.im)*x1 + (W1.re - j*W1.im)*x2
        // X2 = x0 + (W1.re - j*W1.im)*x1 + (W1.re + j*W1.im)*x2
        //
        // Then we rearrange and sort terms.
        // X0 = x0 + x1 + x2
        // X1 = x0 + W1.re*(x1+x2) + j*W1.im*(x1-x2)
        // X2 = x0 + W1.re*(x1+x2) - j*W1.im*(x1-x2)
        //
        // Now we define xp=x1+x2 xn=x1-x2, and write out the complex and imaginary parts
        // X0 = x0 + x1 + x2
        // X1.re = x0.re + W1.re*xp.re - W1.im*xn.im
        // X1.im = x0.im + W1.re*xp.im + W1.im*xn.re
        // X2.re = x0.re + W1.re*xp.re + W1.im*xn.im
        // X2.im = x0.im + W1.re*xp.im - W1.im*xn.re
        //
        // Finally defining:
        // temp_a = x0 + W1.re*xp.re + j*W1.re*xp.im
        // temp_b = -W1.im*xn.im + j*W1.im*xn.re
        // leads to the final result:
        // X0 = x0 + x1 + x2
        // X1 = temp_a + temp_b
        // X2 = temp_a - temp_b

        let xp = *buffer.get_unchecked(1) + *buffer.get_unchecked(2);
        let xn = *buffer.get_unchecked(1) - *buffer.get_unchecked(2);
        let sum = *buffer.get_unchecked(0) + xp;

        let temp_a = *buffer.get_unchecked(0)
            + Complex {
                re: self.twiddle.re * xp.re,
                im: self.twiddle.re * xp.im,
            };
        let temp_b = Complex {
            re: -self.twiddle.im * xn.im,
            im: self.twiddle.im * xn.re,
        };

        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = temp_a + temp_b;
        *buffer.get_unchecked_mut(2) = temp_a - temp_b;
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
}
impl<T: FFTnum> FFT<T> for Butterfly3<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly3<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        3
    }
}
impl<T> IsInverse for Butterfly3<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly4 {
    inverse: bool,
}
impl Butterfly4 {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly4 { inverse: inverse }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly4 {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        let butterfly2 = Butterfly2::new(self.inverse);

        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose, which we're skipping because we're just going to perform non-contiguous FFTs

        // step 2: column FFTs
        {
            let (a, b) = buffer.split_at_mut(2);
            Butterfly2::perform_fft_direct(a.get_unchecked_mut(0), b.get_unchecked_mut(0));
            Butterfly2::perform_fft_direct(a.get_unchecked_mut(1), b.get_unchecked_mut(1));

            // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
            *b.get_unchecked_mut(1) = twiddles::rotate_90(*b.get_unchecked(1), self.inverse);

            // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

            // step 5: row FFTs
            butterfly2.process_inplace(a);
            butterfly2.process_inplace(b);
        }

        // step 6: transpose
        swap_unchecked(buffer, 1, 2);
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
}
impl<T: FFTnum> FFT<T> for Butterfly4 {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl Length for Butterfly4 {
    #[inline(always)]
    fn len(&self) -> usize {
        4
    }
}
impl IsInverse for Butterfly4 {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly5<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly5<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 5, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 5, inverse);
        Butterfly5 {
            twiddle1,
            twiddle2,
            inverse,
        }
    }
}

impl<T: FFTnum> FFTButterfly<T> for Butterfly5<T> {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // Let's do a plain 5-point DFT
        // |X0|   | W0 W0  W0  W0  W0  |   |x0|
        // |X1|   | W0 W1  W2  W3  W4  |   |x1|
        // |X2| = | W0 W2  W4  W6  W8  | * |x2|
        // |X3|   | W0 W3  W6  W9  W12 |   |x3|
        // |X4|   | W0 W4  W8  W12 W16 |   |x4|
        //
        // where Wn = exp(-2*pi*n/5) for a forward transform, and exp(+2*pi*n/5) for an inverse.
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

        let x14p = *buffer.get_unchecked(1) + *buffer.get_unchecked(4);
        let x14n = *buffer.get_unchecked(1) - *buffer.get_unchecked(4);
        let x23p = *buffer.get_unchecked(2) + *buffer.get_unchecked(3);
        let x23n = *buffer.get_unchecked(2) - *buffer.get_unchecked(3);
        let sum = *buffer.get_unchecked(0) + x14p + x23p;
        let b14re_a =
            buffer.get_unchecked(0).re + self.twiddle1.re * x14p.re + self.twiddle2.re * x23p.re;
        let b14re_b = self.twiddle1.im * x14n.im + self.twiddle2.im * x23n.im;
        let b23re_a =
            buffer.get_unchecked(0).re + self.twiddle2.re * x14p.re + self.twiddle1.re * x23p.re;
        let b23re_b = self.twiddle2.im * x14n.im + -self.twiddle1.im * x23n.im;

        let b14im_a =
            buffer.get_unchecked(0).im + self.twiddle1.re * x14p.im + self.twiddle2.re * x23p.im;
        let b14im_b = self.twiddle1.im * x14n.re + self.twiddle2.im * x23n.re;
        let b23im_a =
            buffer.get_unchecked(0).im + self.twiddle2.re * x14p.im + self.twiddle1.re * x23p.im;
        let b23im_b = self.twiddle2.im * x14n.re + -self.twiddle1.im * x23n.re;

        let out1re = b14re_a - b14re_b;
        let out1im = b14im_a + b14im_b;
        let out2re = b23re_a - b23re_b;
        let out2im = b23im_a + b23im_b;
        let out3re = b23re_a + b23re_b;
        let out3im = b23im_a - b23im_b;
        let out4re = b14re_a + b14re_b;
        let out4im = b14im_a - b14im_b;
        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
}
impl<T: FFTnum> FFT<T> for Butterfly5<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly5<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        5
    }
}
impl<T> IsInverse for Butterfly5<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly6<T> {
    butterfly3: Butterfly3<T>,
}
impl<T: FFTnum> Butterfly6<T> {
    pub fn new(inverse: bool) -> Self {
        Butterfly6 {
            butterfly3: Butterfly3::new(inverse),
        }
    }
    pub fn inverse_of(fft: &Butterfly6<T>) -> Self {
        Butterfly6 {
            butterfly3: Butterfly3::inverse_of(&fft.butterfly3),
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly6<T> {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        //since GCD(2,3) == 1 we're going to hardcode a step of the Good-Thomas algorithm to avoid twiddle factors

        // step 1: reorder the input directly into the scratch. normally there's a whole thing to compute this ordering
        //but thankfully we can just precompute it and hardcode it
        let mut scratch_a = [
            *buffer.get_unchecked(0),
            *buffer.get_unchecked(2),
            *buffer.get_unchecked(4),
        ];

        let mut scratch_b = [
            *buffer.get_unchecked(3),
            *buffer.get_unchecked(5),
            *buffer.get_unchecked(1),
        ];

        // step 2: column FFTs
        self.butterfly3.process_inplace(&mut scratch_a);
        self.butterfly3.process_inplace(&mut scratch_b);

        // step 3: apply twiddle factors -- SKIPPED because good-thomas doesn't have twiddle factors :)

        // step 4: SKIPPED because the next FFTs will be non-contiguous

        // step 5: row FFTs
        Butterfly2::perform_fft_direct(&mut scratch_a[0], &mut scratch_b[0]);
        Butterfly2::perform_fft_direct(&mut scratch_a[1], &mut scratch_b[1]);
        Butterfly2::perform_fft_direct(&mut scratch_a[2], &mut scratch_b[2]);

        // step 6: reorder the result back into the buffer. again we would normally have to do an expensive computation
        // but instead we can precompute and hardcode the ordering
        // note that we're also rolling a transpose step into this reorder
        *buffer.get_unchecked_mut(0) = scratch_a[0];
        *buffer.get_unchecked_mut(3) = scratch_b[0];
        *buffer.get_unchecked_mut(4) = scratch_a[1];
        *buffer.get_unchecked_mut(1) = scratch_b[1];
        *buffer.get_unchecked_mut(2) = scratch_a[2];
        *buffer.get_unchecked_mut(5) = scratch_b[2];
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
}
impl<T: FFTnum> FFT<T> for Butterfly6<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly6<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        6
    }
}
impl<T> IsInverse for Butterfly6<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.butterfly3.is_inverse()
    }
}

pub struct Butterfly7<T> {
    //inner_fft: Butterfly6<T>,
    //inner_fft_multiply: [Complex<T>; 6]
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly7<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 7, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 7, inverse);
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 7, inverse);
        Butterfly7 {
            twiddle1,
            twiddle2,
            twiddle3,
            inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly7<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // Let's do a plain 7-point DFT
        // |X0|   | W0 W0  W0  W0  W0  W0  W0  |   |x0|
        // |X1|   | W0 W1  W2  W3  W4  W5  W6  |   |x1|
        // |X2|   | W0 W2  W4  W6  W8  W10 W12 |   |x2|
        // |X3| = | W0 W3  W6  W9  W12 W15 W18 | * |x3|
        // |X4|   | W0 W4  W8  W12 W16 W20 W24 |   |x4|
        // |X5|   | W0 W5  W10 W15 W20 W25 W30 |   |x4|
        // |X6|   | W0 W6  W12 W18 W24 W30 W36 |   |x4|
        //
        // where Wn = exp(-2*pi*n/7) for a forward transform, and exp(+2*pi*n/7) for an inverse.
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

        let x16p = *buffer.get_unchecked(1) + *buffer.get_unchecked(6);
        let x16n = *buffer.get_unchecked(1) - *buffer.get_unchecked(6);
        let x25p = *buffer.get_unchecked(2) + *buffer.get_unchecked(5);
        let x25n = *buffer.get_unchecked(2) - *buffer.get_unchecked(5);
        let x34p = *buffer.get_unchecked(3) + *buffer.get_unchecked(4);
        let x34n = *buffer.get_unchecked(3) - *buffer.get_unchecked(4);
        let sum = *buffer.get_unchecked(0) + x16p + x25p + x34p;

        let x16re_a = buffer.get_unchecked(0).re
            + self.twiddle1.re * x16p.re
            + self.twiddle2.re * x25p.re
            + self.twiddle3.re * x34p.re;
        let x16re_b =
            self.twiddle1.im * x16n.im + self.twiddle2.im * x25n.im + self.twiddle3.im * x34n.im;
        let x25re_a = buffer.get_unchecked(0).re
            + self.twiddle1.re * x34p.re
            + self.twiddle2.re * x16p.re
            + self.twiddle3.re * x25p.re;
        let x25re_b =
            -self.twiddle1.im * x34n.im + self.twiddle2.im * x16n.im - self.twiddle3.im * x25n.im;
        let x34re_a = buffer.get_unchecked(0).re
            + self.twiddle1.re * x25p.re
            + self.twiddle2.re * x34p.re
            + self.twiddle3.re * x16p.re;
        let x34re_b =
            -self.twiddle1.im * x25n.im + self.twiddle2.im * x34n.im + self.twiddle3.im * x16n.im;
        let x16im_a = buffer.get_unchecked(0).im
            + self.twiddle1.re * x16p.im
            + self.twiddle2.re * x25p.im
            + self.twiddle3.re * x34p.im;
        let x16im_b =
            self.twiddle1.im * x16n.re + self.twiddle2.im * x25n.re + self.twiddle3.im * x34n.re;
        let x25im_a = buffer.get_unchecked(0).im
            + self.twiddle1.re * x34p.im
            + self.twiddle2.re * x16p.im
            + self.twiddle3.re * x25p.im;
        let x25im_b =
            -self.twiddle1.im * x34n.re + self.twiddle2.im * x16n.re - self.twiddle3.im * x25n.re;
        let x34im_a = buffer.get_unchecked(0).im
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

        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
        *buffer.get_unchecked_mut(5) = Complex {
            re: out5re,
            im: out5im,
        };
        *buffer.get_unchecked_mut(6) = Complex {
            re: out6re,
            im: out6im,
        };
    }
}
impl<T: FFTnum> FFT<T> for Butterfly7<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly7<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        7
    }
}
impl<T> IsInverse for Butterfly7<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly8<T> {
    twiddle: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly8<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly8 {
            inverse: inverse,
            twiddle: twiddles::single_twiddle(1, 8, inverse),
        }
    }

    #[inline(always)]
    unsafe fn transpose_4x2_to_2x4(buffer: &mut [Complex<T>; 8]) {
        let temp1 = buffer[1];
        buffer[1] = buffer[4];
        buffer[4] = buffer[2];
        buffer[2] = temp1;

        let temp6 = buffer[6];
        buffer[6] = buffer[3];
        buffer[3] = buffer[5];
        buffer[5] = temp6;
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly8<T> {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        let butterfly2 = Butterfly2::new(self.inverse);
        let butterfly4 = Butterfly4::new(self.inverse);

        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose the input into the scratch
        let mut scratch = [
            *buffer.get_unchecked(0),
            *buffer.get_unchecked(2),
            *buffer.get_unchecked(4),
            *buffer.get_unchecked(6),
            *buffer.get_unchecked(1),
            *buffer.get_unchecked(3),
            *buffer.get_unchecked(5),
            *buffer.get_unchecked(7),
        ];

        // step 2: column FFTs
        butterfly4.process_inplace(&mut scratch[..4]);
        butterfly4.process_inplace(&mut scratch[4..]);

        // step 3: apply twiddle factors
        let twiddle1 = self.twiddle;
        let twiddle3 = Complex {
            re: -twiddle1.re,
            im: twiddle1.im,
        };

        *scratch.get_unchecked_mut(5) = scratch.get_unchecked(5) * twiddle1;
        *scratch.get_unchecked_mut(6) =
            twiddles::rotate_90(*scratch.get_unchecked(6), self.inverse);
        *scratch.get_unchecked_mut(7) = scratch.get_unchecked(7) * twiddle3;

        // step 4: transpose
        Self::transpose_4x2_to_2x4(&mut scratch);

        // step 5: row FFTs
        butterfly2.process_inplace(&mut scratch[..2]);
        butterfly2.process_inplace(&mut scratch[2..4]);
        butterfly2.process_inplace(&mut scratch[4..6]);
        butterfly2.process_inplace(&mut scratch[6..]);

        // step 6: transpose the scratch into the buffer
        *buffer.get_unchecked_mut(0) = scratch[0];
        *buffer.get_unchecked_mut(1) = scratch[2];
        *buffer.get_unchecked_mut(2) = scratch[4];
        *buffer.get_unchecked_mut(3) = scratch[6];
        *buffer.get_unchecked_mut(4) = scratch[1];
        *buffer.get_unchecked_mut(5) = scratch[3];
        *buffer.get_unchecked_mut(6) = scratch[5];
        *buffer.get_unchecked_mut(7) = scratch[7];
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
}
impl<T: FFTnum> FFT<T> for Butterfly8<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly8<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        8
    }
}
impl<T> IsInverse for Butterfly8<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly11<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly11<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 11, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 11, inverse);
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 11, inverse);
        let twiddle4: Complex<T> = twiddles::single_twiddle(4, 11, inverse);
        let twiddle5: Complex<T> = twiddles::single_twiddle(5, 11, inverse);
        Butterfly11 {
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly11<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.

        let x110p = *buffer.get_unchecked(1) + *buffer.get_unchecked(10);
        let x110n = *buffer.get_unchecked(1) - *buffer.get_unchecked(10);
        let x29p = *buffer.get_unchecked(2) + *buffer.get_unchecked(9);
        let x29n = *buffer.get_unchecked(2) - *buffer.get_unchecked(9);
        let x38p = *buffer.get_unchecked(3) + *buffer.get_unchecked(8);
        let x38n = *buffer.get_unchecked(3) - *buffer.get_unchecked(8);
        let x47p = *buffer.get_unchecked(4) + *buffer.get_unchecked(7);
        let x47n = *buffer.get_unchecked(4) - *buffer.get_unchecked(7);
        let x56p = *buffer.get_unchecked(5) + *buffer.get_unchecked(6);
        let x56n = *buffer.get_unchecked(5) - *buffer.get_unchecked(6);
        let sum = *buffer.get_unchecked(0) + x110p + x29p + x38p + x47p + x56p;
        let b110re_a = buffer.get_unchecked(0).re
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
        let b29re_a = buffer.get_unchecked(0).re
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
        let b38re_a = buffer.get_unchecked(0).re
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
        let b47re_a = buffer.get_unchecked(0).re
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
        let b56re_a = buffer.get_unchecked(0).re
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

        let b110im_a = buffer.get_unchecked(0).im
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
        let b29im_a = buffer.get_unchecked(0).im
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
        let b38im_a = buffer.get_unchecked(0).im
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
        let b47im_a = buffer.get_unchecked(0).im
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
        let b56im_a = buffer.get_unchecked(0).im
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
        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
        *buffer.get_unchecked_mut(5) = Complex {
            re: out5re,
            im: out5im,
        };
        *buffer.get_unchecked_mut(6) = Complex {
            re: out6re,
            im: out6im,
        };
        *buffer.get_unchecked_mut(7) = Complex {
            re: out7re,
            im: out7im,
        };
        *buffer.get_unchecked_mut(8) = Complex {
            re: out8re,
            im: out8im,
        };
        *buffer.get_unchecked_mut(9) = Complex {
            re: out9re,
            im: out9im,
        };
        *buffer.get_unchecked_mut(10) = Complex {
            re: out10re,
            im: out10im,
        };
    }
}
impl<T: FFTnum> FFT<T> for Butterfly11<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly11<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        11
    }
}
impl<T> IsInverse for Butterfly11<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly13<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly13<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 13, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 13, inverse);
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 13, inverse);
        let twiddle4: Complex<T> = twiddles::single_twiddle(4, 13, inverse);
        let twiddle5: Complex<T> = twiddles::single_twiddle(5, 13, inverse);
        let twiddle6: Complex<T> = twiddles::single_twiddle(6, 13, inverse);
        Butterfly13 {
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly13<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x112p = *buffer.get_unchecked(1) + *buffer.get_unchecked(12);
        let x112n = *buffer.get_unchecked(1) - *buffer.get_unchecked(12);
        let x211p = *buffer.get_unchecked(2) + *buffer.get_unchecked(11);
        let x211n = *buffer.get_unchecked(2) - *buffer.get_unchecked(11);
        let x310p = *buffer.get_unchecked(3) + *buffer.get_unchecked(10);
        let x310n = *buffer.get_unchecked(3) - *buffer.get_unchecked(10);
        let x49p = *buffer.get_unchecked(4) + *buffer.get_unchecked(9);
        let x49n = *buffer.get_unchecked(4) - *buffer.get_unchecked(9);
        let x58p = *buffer.get_unchecked(5) + *buffer.get_unchecked(8);
        let x58n = *buffer.get_unchecked(5) - *buffer.get_unchecked(8);
        let x67p = *buffer.get_unchecked(6) + *buffer.get_unchecked(7);
        let x67n = *buffer.get_unchecked(6) - *buffer.get_unchecked(7);
        let sum = *buffer.get_unchecked(0) + x112p + x211p + x310p + x49p + x58p + x67p;
        let b112re_a = buffer.get_unchecked(0).re
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
        let b211re_a = buffer.get_unchecked(0).re
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
        let b310re_a = buffer.get_unchecked(0).re
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
        let b49re_a = buffer.get_unchecked(0).re
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
        let b58re_a = buffer.get_unchecked(0).re
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
        let b67re_a = buffer.get_unchecked(0).re
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

        let b112im_a = buffer.get_unchecked(0).im
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
        let b211im_a = buffer.get_unchecked(0).im
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
        let b310im_a = buffer.get_unchecked(0).im
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
        let b49im_a = buffer.get_unchecked(0).im
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
        let b58im_a = buffer.get_unchecked(0).im
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
        let b67im_a = buffer.get_unchecked(0).im
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
        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
        *buffer.get_unchecked_mut(5) = Complex {
            re: out5re,
            im: out5im,
        };
        *buffer.get_unchecked_mut(6) = Complex {
            re: out6re,
            im: out6im,
        };
        *buffer.get_unchecked_mut(7) = Complex {
            re: out7re,
            im: out7im,
        };
        *buffer.get_unchecked_mut(8) = Complex {
            re: out8re,
            im: out8im,
        };
        *buffer.get_unchecked_mut(9) = Complex {
            re: out9re,
            im: out9im,
        };
        *buffer.get_unchecked_mut(10) = Complex {
            re: out10re,
            im: out10im,
        };
        *buffer.get_unchecked_mut(11) = Complex {
            re: out11re,
            im: out11im,
        };
        *buffer.get_unchecked_mut(12) = Complex {
            re: out12re,
            im: out12im,
        };
    }
}
impl<T: FFTnum> FFT<T> for Butterfly13<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly13<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        13
    }
}
impl<T> IsInverse for Butterfly13<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly16<T> {
    butterfly8: Butterfly8<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly16<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly16 {
            butterfly8: Butterfly8::new(inverse),
            twiddle1: twiddles::single_twiddle(1, 16, inverse),
            twiddle2: twiddles::single_twiddle(2, 16, inverse),
            twiddle3: twiddles::single_twiddle(3, 16, inverse),
            inverse: inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly16<T> {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        let butterfly4 = Butterfly4::new(self.inverse);

        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        let mut scratch_evens = [
            *buffer.get_unchecked(0),
            *buffer.get_unchecked(2),
            *buffer.get_unchecked(4),
            *buffer.get_unchecked(6),
            *buffer.get_unchecked(8),
            *buffer.get_unchecked(10),
            *buffer.get_unchecked(12),
            *buffer.get_unchecked(14),
        ];

        let mut scratch_odds_n1 = [
            *buffer.get_unchecked(1),
            *buffer.get_unchecked(5),
            *buffer.get_unchecked(9),
            *buffer.get_unchecked(13),
        ];
        let mut scratch_odds_n3 = [
            *buffer.get_unchecked(15),
            *buffer.get_unchecked(3),
            *buffer.get_unchecked(7),
            *buffer.get_unchecked(11),
        ];

        // step 2: column FFTs
        self.butterfly8.process_inplace(&mut scratch_evens);
        butterfly4.process_inplace(&mut scratch_odds_n1);
        butterfly4.process_inplace(&mut scratch_odds_n3);

        // step 3: apply twiddle factors
        scratch_odds_n1[1] = scratch_odds_n1[1] * self.twiddle1;
        scratch_odds_n3[1] = scratch_odds_n3[1] * self.twiddle1.conj();

        scratch_odds_n1[2] = scratch_odds_n1[2] * self.twiddle2;
        scratch_odds_n3[2] = scratch_odds_n3[2] * self.twiddle2.conj();

        scratch_odds_n1[3] = scratch_odds_n1[3] * self.twiddle3;
        scratch_odds_n3[3] = scratch_odds_n3[3] * self.twiddle3.conj();

        // step 4: cross FFTs
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[0], &mut scratch_odds_n3[0]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[1], &mut scratch_odds_n3[1]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[2], &mut scratch_odds_n3[2]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[3], &mut scratch_odds_n3[3]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.inverse);
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.inverse);
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.inverse);
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.inverse);

        //step 5: copy/add/subtract data back to buffer
        *buffer.get_unchecked_mut(0) = scratch_evens[0] + scratch_odds_n1[0];
        *buffer.get_unchecked_mut(1) = scratch_evens[1] + scratch_odds_n1[1];
        *buffer.get_unchecked_mut(2) = scratch_evens[2] + scratch_odds_n1[2];
        *buffer.get_unchecked_mut(3) = scratch_evens[3] + scratch_odds_n1[3];
        *buffer.get_unchecked_mut(4) = scratch_evens[4] + scratch_odds_n3[0];
        *buffer.get_unchecked_mut(5) = scratch_evens[5] + scratch_odds_n3[1];
        *buffer.get_unchecked_mut(6) = scratch_evens[6] + scratch_odds_n3[2];
        *buffer.get_unchecked_mut(7) = scratch_evens[7] + scratch_odds_n3[3];
        *buffer.get_unchecked_mut(8) = scratch_evens[0] - scratch_odds_n1[0];
        *buffer.get_unchecked_mut(9) = scratch_evens[1] - scratch_odds_n1[1];
        *buffer.get_unchecked_mut(10) = scratch_evens[2] - scratch_odds_n1[2];
        *buffer.get_unchecked_mut(11) = scratch_evens[3] - scratch_odds_n1[3];
        *buffer.get_unchecked_mut(12) = scratch_evens[4] - scratch_odds_n3[0];
        *buffer.get_unchecked_mut(13) = scratch_evens[5] - scratch_odds_n3[1];
        *buffer.get_unchecked_mut(14) = scratch_evens[6] - scratch_odds_n3[2];
        *buffer.get_unchecked_mut(15) = scratch_evens[7] - scratch_odds_n3[3];
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
}
impl<T: FFTnum> FFT<T> for Butterfly16<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly16<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        16
    }
}
impl<T> IsInverse for Butterfly16<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
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
    inverse: bool,
}
impl<T: FFTnum> Butterfly17<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 17, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 17, inverse);
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 17, inverse);
        let twiddle4: Complex<T> = twiddles::single_twiddle(4, 17, inverse);
        let twiddle5: Complex<T> = twiddles::single_twiddle(5, 17, inverse);
        let twiddle6: Complex<T> = twiddles::single_twiddle(6, 17, inverse);
        let twiddle7: Complex<T> = twiddles::single_twiddle(7, 17, inverse);
        let twiddle8: Complex<T> = twiddles::single_twiddle(8, 17, inverse);
        Butterfly17 {
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            twiddle7,
            twiddle8,
            inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly17<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x116p = *buffer.get_unchecked(1) + *buffer.get_unchecked(16);
        let x116n = *buffer.get_unchecked(1) - *buffer.get_unchecked(16);
        let x215p = *buffer.get_unchecked(2) + *buffer.get_unchecked(15);
        let x215n = *buffer.get_unchecked(2) - *buffer.get_unchecked(15);
        let x314p = *buffer.get_unchecked(3) + *buffer.get_unchecked(14);
        let x314n = *buffer.get_unchecked(3) - *buffer.get_unchecked(14);
        let x413p = *buffer.get_unchecked(4) + *buffer.get_unchecked(13);
        let x413n = *buffer.get_unchecked(4) - *buffer.get_unchecked(13);
        let x512p = *buffer.get_unchecked(5) + *buffer.get_unchecked(12);
        let x512n = *buffer.get_unchecked(5) - *buffer.get_unchecked(12);
        let x611p = *buffer.get_unchecked(6) + *buffer.get_unchecked(11);
        let x611n = *buffer.get_unchecked(6) - *buffer.get_unchecked(11);
        let x710p = *buffer.get_unchecked(7) + *buffer.get_unchecked(10);
        let x710n = *buffer.get_unchecked(7) - *buffer.get_unchecked(10);
        let x89p = *buffer.get_unchecked(8) + *buffer.get_unchecked(9);
        let x89n = *buffer.get_unchecked(8) - *buffer.get_unchecked(9);
        let sum =
            *buffer.get_unchecked(0) + x116p + x215p + x314p + x413p + x512p + x611p + x710p + x89p;
        let b116re_a = buffer.get_unchecked(0).re
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
        let b215re_a = buffer.get_unchecked(0).re
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
        let b314re_a = buffer.get_unchecked(0).re
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
        let b413re_a = buffer.get_unchecked(0).re
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
        let b512re_a = buffer.get_unchecked(0).re
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
        let b611re_a = buffer.get_unchecked(0).re
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
        let b710re_a = buffer.get_unchecked(0).re
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
        let b89re_a = buffer.get_unchecked(0).re
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

        let b116im_a = buffer.get_unchecked(0).im
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
        let b215im_a = buffer.get_unchecked(0).im
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
        let b314im_a = buffer.get_unchecked(0).im
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
        let b413im_a = buffer.get_unchecked(0).im
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
        let b512im_a = buffer.get_unchecked(0).im
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
        let b611im_a = buffer.get_unchecked(0).im
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
        let b710im_a = buffer.get_unchecked(0).im
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
        let b89im_a = buffer.get_unchecked(0).im
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
        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
        *buffer.get_unchecked_mut(5) = Complex {
            re: out5re,
            im: out5im,
        };
        *buffer.get_unchecked_mut(6) = Complex {
            re: out6re,
            im: out6im,
        };
        *buffer.get_unchecked_mut(7) = Complex {
            re: out7re,
            im: out7im,
        };
        *buffer.get_unchecked_mut(8) = Complex {
            re: out8re,
            im: out8im,
        };
        *buffer.get_unchecked_mut(9) = Complex {
            re: out9re,
            im: out9im,
        };
        *buffer.get_unchecked_mut(10) = Complex {
            re: out10re,
            im: out10im,
        };
        *buffer.get_unchecked_mut(11) = Complex {
            re: out11re,
            im: out11im,
        };
        *buffer.get_unchecked_mut(12) = Complex {
            re: out12re,
            im: out12im,
        };
        *buffer.get_unchecked_mut(13) = Complex {
            re: out13re,
            im: out13im,
        };
        *buffer.get_unchecked_mut(14) = Complex {
            re: out14re,
            im: out14im,
        };
        *buffer.get_unchecked_mut(15) = Complex {
            re: out15re,
            im: out15im,
        };
        *buffer.get_unchecked_mut(16) = Complex {
            re: out16re,
            im: out16im,
        };
    }
}
impl<T: FFTnum> FFT<T> for Butterfly17<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly17<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        17
    }
}
impl<T> IsInverse for Butterfly17<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
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
    inverse: bool,
}
impl<T: FFTnum> Butterfly19<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 19, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 19, inverse);
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 19, inverse);
        let twiddle4: Complex<T> = twiddles::single_twiddle(4, 19, inverse);
        let twiddle5: Complex<T> = twiddles::single_twiddle(5, 19, inverse);
        let twiddle6: Complex<T> = twiddles::single_twiddle(6, 19, inverse);
        let twiddle7: Complex<T> = twiddles::single_twiddle(7, 19, inverse);
        let twiddle8: Complex<T> = twiddles::single_twiddle(8, 19, inverse);
        let twiddle9: Complex<T> = twiddles::single_twiddle(9, 19, inverse);
        Butterfly19 {
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            twiddle7,
            twiddle8,
            twiddle9,
            inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly19<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x118p = *buffer.get_unchecked(1) + *buffer.get_unchecked(18);
        let x118n = *buffer.get_unchecked(1) - *buffer.get_unchecked(18);
        let x217p = *buffer.get_unchecked(2) + *buffer.get_unchecked(17);
        let x217n = *buffer.get_unchecked(2) - *buffer.get_unchecked(17);
        let x316p = *buffer.get_unchecked(3) + *buffer.get_unchecked(16);
        let x316n = *buffer.get_unchecked(3) - *buffer.get_unchecked(16);
        let x415p = *buffer.get_unchecked(4) + *buffer.get_unchecked(15);
        let x415n = *buffer.get_unchecked(4) - *buffer.get_unchecked(15);
        let x514p = *buffer.get_unchecked(5) + *buffer.get_unchecked(14);
        let x514n = *buffer.get_unchecked(5) - *buffer.get_unchecked(14);
        let x613p = *buffer.get_unchecked(6) + *buffer.get_unchecked(13);
        let x613n = *buffer.get_unchecked(6) - *buffer.get_unchecked(13);
        let x712p = *buffer.get_unchecked(7) + *buffer.get_unchecked(12);
        let x712n = *buffer.get_unchecked(7) - *buffer.get_unchecked(12);
        let x811p = *buffer.get_unchecked(8) + *buffer.get_unchecked(11);
        let x811n = *buffer.get_unchecked(8) - *buffer.get_unchecked(11);
        let x910p = *buffer.get_unchecked(9) + *buffer.get_unchecked(10);
        let x910n = *buffer.get_unchecked(9) - *buffer.get_unchecked(10);
        let sum = *buffer.get_unchecked(0)
            + x118p
            + x217p
            + x316p
            + x415p
            + x514p
            + x613p
            + x712p
            + x811p
            + x910p;
        let b118re_a = buffer.get_unchecked(0).re
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
        let b217re_a = buffer.get_unchecked(0).re
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
        let b316re_a = buffer.get_unchecked(0).re
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
        let b415re_a = buffer.get_unchecked(0).re
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
        let b514re_a = buffer.get_unchecked(0).re
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
        let b613re_a = buffer.get_unchecked(0).re
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
        let b712re_a = buffer.get_unchecked(0).re
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
        let b811re_a = buffer.get_unchecked(0).re
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
        let b910re_a = buffer.get_unchecked(0).re
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

        let b118im_a = buffer.get_unchecked(0).im
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
        let b217im_a = buffer.get_unchecked(0).im
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
        let b316im_a = buffer.get_unchecked(0).im
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
        let b415im_a = buffer.get_unchecked(0).im
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
        let b514im_a = buffer.get_unchecked(0).im
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
        let b613im_a = buffer.get_unchecked(0).im
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
        let b712im_a = buffer.get_unchecked(0).im
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
        let b811im_a = buffer.get_unchecked(0).im
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
        let b910im_a = buffer.get_unchecked(0).im
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
        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
        *buffer.get_unchecked_mut(5) = Complex {
            re: out5re,
            im: out5im,
        };
        *buffer.get_unchecked_mut(6) = Complex {
            re: out6re,
            im: out6im,
        };
        *buffer.get_unchecked_mut(7) = Complex {
            re: out7re,
            im: out7im,
        };
        *buffer.get_unchecked_mut(8) = Complex {
            re: out8re,
            im: out8im,
        };
        *buffer.get_unchecked_mut(9) = Complex {
            re: out9re,
            im: out9im,
        };
        *buffer.get_unchecked_mut(10) = Complex {
            re: out10re,
            im: out10im,
        };
        *buffer.get_unchecked_mut(11) = Complex {
            re: out11re,
            im: out11im,
        };
        *buffer.get_unchecked_mut(12) = Complex {
            re: out12re,
            im: out12im,
        };
        *buffer.get_unchecked_mut(13) = Complex {
            re: out13re,
            im: out13im,
        };
        *buffer.get_unchecked_mut(14) = Complex {
            re: out14re,
            im: out14im,
        };
        *buffer.get_unchecked_mut(15) = Complex {
            re: out15re,
            im: out15im,
        };
        *buffer.get_unchecked_mut(16) = Complex {
            re: out16re,
            im: out16im,
        };
        *buffer.get_unchecked_mut(17) = Complex {
            re: out17re,
            im: out17im,
        };
        *buffer.get_unchecked_mut(18) = Complex {
            re: out18re,
            im: out18im,
        };
    }
}
impl<T: FFTnum> FFT<T> for Butterfly19<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly19<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        19
    }
}
impl<T> IsInverse for Butterfly19<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
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
    inverse: bool,
}
impl<T: FFTnum> Butterfly23<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 23, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 23, inverse);
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 23, inverse);
        let twiddle4: Complex<T> = twiddles::single_twiddle(4, 23, inverse);
        let twiddle5: Complex<T> = twiddles::single_twiddle(5, 23, inverse);
        let twiddle6: Complex<T> = twiddles::single_twiddle(6, 23, inverse);
        let twiddle7: Complex<T> = twiddles::single_twiddle(7, 23, inverse);
        let twiddle8: Complex<T> = twiddles::single_twiddle(8, 23, inverse);
        let twiddle9: Complex<T> = twiddles::single_twiddle(9, 23, inverse);
        let twiddle10: Complex<T> = twiddles::single_twiddle(10, 23, inverse);
        let twiddle11: Complex<T> = twiddles::single_twiddle(11, 23, inverse);
        Butterfly23 {
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
            inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly23<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x122p = *buffer.get_unchecked(1) + *buffer.get_unchecked(22);
        let x122n = *buffer.get_unchecked(1) - *buffer.get_unchecked(22);
        let x221p = *buffer.get_unchecked(2) + *buffer.get_unchecked(21);
        let x221n = *buffer.get_unchecked(2) - *buffer.get_unchecked(21);
        let x320p = *buffer.get_unchecked(3) + *buffer.get_unchecked(20);
        let x320n = *buffer.get_unchecked(3) - *buffer.get_unchecked(20);
        let x419p = *buffer.get_unchecked(4) + *buffer.get_unchecked(19);
        let x419n = *buffer.get_unchecked(4) - *buffer.get_unchecked(19);
        let x518p = *buffer.get_unchecked(5) + *buffer.get_unchecked(18);
        let x518n = *buffer.get_unchecked(5) - *buffer.get_unchecked(18);
        let x617p = *buffer.get_unchecked(6) + *buffer.get_unchecked(17);
        let x617n = *buffer.get_unchecked(6) - *buffer.get_unchecked(17);
        let x716p = *buffer.get_unchecked(7) + *buffer.get_unchecked(16);
        let x716n = *buffer.get_unchecked(7) - *buffer.get_unchecked(16);
        let x815p = *buffer.get_unchecked(8) + *buffer.get_unchecked(15);
        let x815n = *buffer.get_unchecked(8) - *buffer.get_unchecked(15);
        let x914p = *buffer.get_unchecked(9) + *buffer.get_unchecked(14);
        let x914n = *buffer.get_unchecked(9) - *buffer.get_unchecked(14);
        let x1013p = *buffer.get_unchecked(10) + *buffer.get_unchecked(13);
        let x1013n = *buffer.get_unchecked(10) - *buffer.get_unchecked(13);
        let x1112p = *buffer.get_unchecked(11) + *buffer.get_unchecked(12);
        let x1112n = *buffer.get_unchecked(11) - *buffer.get_unchecked(12);
        let sum = *buffer.get_unchecked(0)
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
        let b122re_a = buffer.get_unchecked(0).re
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
        let b221re_a = buffer.get_unchecked(0).re
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
        let b320re_a = buffer.get_unchecked(0).re
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
        let b419re_a = buffer.get_unchecked(0).re
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
        let b518re_a = buffer.get_unchecked(0).re
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
        let b617re_a = buffer.get_unchecked(0).re
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
        let b716re_a = buffer.get_unchecked(0).re
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
        let b815re_a = buffer.get_unchecked(0).re
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
        let b914re_a = buffer.get_unchecked(0).re
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
        let b1013re_a = buffer.get_unchecked(0).re
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
        let b1112re_a = buffer.get_unchecked(0).re
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

        let b122im_a = buffer.get_unchecked(0).im
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
        let b221im_a = buffer.get_unchecked(0).im
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
        let b320im_a = buffer.get_unchecked(0).im
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
        let b419im_a = buffer.get_unchecked(0).im
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
        let b518im_a = buffer.get_unchecked(0).im
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
        let b617im_a = buffer.get_unchecked(0).im
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
        let b716im_a = buffer.get_unchecked(0).im
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
        let b815im_a = buffer.get_unchecked(0).im
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
        let b914im_a = buffer.get_unchecked(0).im
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
        let b1013im_a = buffer.get_unchecked(0).im
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
        let b1112im_a = buffer.get_unchecked(0).im
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
        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
        *buffer.get_unchecked_mut(5) = Complex {
            re: out5re,
            im: out5im,
        };
        *buffer.get_unchecked_mut(6) = Complex {
            re: out6re,
            im: out6im,
        };
        *buffer.get_unchecked_mut(7) = Complex {
            re: out7re,
            im: out7im,
        };
        *buffer.get_unchecked_mut(8) = Complex {
            re: out8re,
            im: out8im,
        };
        *buffer.get_unchecked_mut(9) = Complex {
            re: out9re,
            im: out9im,
        };
        *buffer.get_unchecked_mut(10) = Complex {
            re: out10re,
            im: out10im,
        };
        *buffer.get_unchecked_mut(11) = Complex {
            re: out11re,
            im: out11im,
        };
        *buffer.get_unchecked_mut(12) = Complex {
            re: out12re,
            im: out12im,
        };
        *buffer.get_unchecked_mut(13) = Complex {
            re: out13re,
            im: out13im,
        };
        *buffer.get_unchecked_mut(14) = Complex {
            re: out14re,
            im: out14im,
        };
        *buffer.get_unchecked_mut(15) = Complex {
            re: out15re,
            im: out15im,
        };
        *buffer.get_unchecked_mut(16) = Complex {
            re: out16re,
            im: out16im,
        };
        *buffer.get_unchecked_mut(17) = Complex {
            re: out17re,
            im: out17im,
        };
        *buffer.get_unchecked_mut(18) = Complex {
            re: out18re,
            im: out18im,
        };
        *buffer.get_unchecked_mut(19) = Complex {
            re: out19re,
            im: out19im,
        };
        *buffer.get_unchecked_mut(20) = Complex {
            re: out20re,
            im: out20im,
        };
        *buffer.get_unchecked_mut(21) = Complex {
            re: out21re,
            im: out21im,
        };
        *buffer.get_unchecked_mut(22) = Complex {
            re: out22re,
            im: out22im,
        };
    }
}
impl<T: FFTnum> FFT<T> for Butterfly23<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly23<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        23
    }
}
impl<T> IsInverse for Butterfly23<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
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
    inverse: bool,
}
impl<T: FFTnum> Butterfly29<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 29, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 29, inverse);
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 29, inverse);
        let twiddle4: Complex<T> = twiddles::single_twiddle(4, 29, inverse);
        let twiddle5: Complex<T> = twiddles::single_twiddle(5, 29, inverse);
        let twiddle6: Complex<T> = twiddles::single_twiddle(6, 29, inverse);
        let twiddle7: Complex<T> = twiddles::single_twiddle(7, 29, inverse);
        let twiddle8: Complex<T> = twiddles::single_twiddle(8, 29, inverse);
        let twiddle9: Complex<T> = twiddles::single_twiddle(9, 29, inverse);
        let twiddle10: Complex<T> = twiddles::single_twiddle(10, 29, inverse);
        let twiddle11: Complex<T> = twiddles::single_twiddle(11, 29, inverse);
        let twiddle12: Complex<T> = twiddles::single_twiddle(12, 29, inverse);
        let twiddle13: Complex<T> = twiddles::single_twiddle(13, 29, inverse);
        let twiddle14: Complex<T> = twiddles::single_twiddle(14, 29, inverse);
        Butterfly29 {
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
            inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly29<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x128p = *buffer.get_unchecked(1) + *buffer.get_unchecked(28);
        let x128n = *buffer.get_unchecked(1) - *buffer.get_unchecked(28);
        let x227p = *buffer.get_unchecked(2) + *buffer.get_unchecked(27);
        let x227n = *buffer.get_unchecked(2) - *buffer.get_unchecked(27);
        let x326p = *buffer.get_unchecked(3) + *buffer.get_unchecked(26);
        let x326n = *buffer.get_unchecked(3) - *buffer.get_unchecked(26);
        let x425p = *buffer.get_unchecked(4) + *buffer.get_unchecked(25);
        let x425n = *buffer.get_unchecked(4) - *buffer.get_unchecked(25);
        let x524p = *buffer.get_unchecked(5) + *buffer.get_unchecked(24);
        let x524n = *buffer.get_unchecked(5) - *buffer.get_unchecked(24);
        let x623p = *buffer.get_unchecked(6) + *buffer.get_unchecked(23);
        let x623n = *buffer.get_unchecked(6) - *buffer.get_unchecked(23);
        let x722p = *buffer.get_unchecked(7) + *buffer.get_unchecked(22);
        let x722n = *buffer.get_unchecked(7) - *buffer.get_unchecked(22);
        let x821p = *buffer.get_unchecked(8) + *buffer.get_unchecked(21);
        let x821n = *buffer.get_unchecked(8) - *buffer.get_unchecked(21);
        let x920p = *buffer.get_unchecked(9) + *buffer.get_unchecked(20);
        let x920n = *buffer.get_unchecked(9) - *buffer.get_unchecked(20);
        let x1019p = *buffer.get_unchecked(10) + *buffer.get_unchecked(19);
        let x1019n = *buffer.get_unchecked(10) - *buffer.get_unchecked(19);
        let x1118p = *buffer.get_unchecked(11) + *buffer.get_unchecked(18);
        let x1118n = *buffer.get_unchecked(11) - *buffer.get_unchecked(18);
        let x1217p = *buffer.get_unchecked(12) + *buffer.get_unchecked(17);
        let x1217n = *buffer.get_unchecked(12) - *buffer.get_unchecked(17);
        let x1316p = *buffer.get_unchecked(13) + *buffer.get_unchecked(16);
        let x1316n = *buffer.get_unchecked(13) - *buffer.get_unchecked(16);
        let x1415p = *buffer.get_unchecked(14) + *buffer.get_unchecked(15);
        let x1415n = *buffer.get_unchecked(14) - *buffer.get_unchecked(15);
        let sum = *buffer.get_unchecked(0)
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
        let b128re_a = buffer.get_unchecked(0).re
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
        let b227re_a = buffer.get_unchecked(0).re
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
        let b326re_a = buffer.get_unchecked(0).re
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
        let b425re_a = buffer.get_unchecked(0).re
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
        let b524re_a = buffer.get_unchecked(0).re
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
        let b623re_a = buffer.get_unchecked(0).re
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
        let b722re_a = buffer.get_unchecked(0).re
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
        let b821re_a = buffer.get_unchecked(0).re
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
        let b920re_a = buffer.get_unchecked(0).re
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
        let b1019re_a = buffer.get_unchecked(0).re
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
        let b1118re_a = buffer.get_unchecked(0).re
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
        let b1217re_a = buffer.get_unchecked(0).re
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
        let b1316re_a = buffer.get_unchecked(0).re
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
        let b1415re_a = buffer.get_unchecked(0).re
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

        let b128im_a = buffer.get_unchecked(0).im
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
        let b227im_a = buffer.get_unchecked(0).im
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
        let b326im_a = buffer.get_unchecked(0).im
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
        let b425im_a = buffer.get_unchecked(0).im
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
        let b524im_a = buffer.get_unchecked(0).im
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
        let b623im_a = buffer.get_unchecked(0).im
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
        let b722im_a = buffer.get_unchecked(0).im
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
        let b821im_a = buffer.get_unchecked(0).im
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
        let b920im_a = buffer.get_unchecked(0).im
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
        let b1019im_a = buffer.get_unchecked(0).im
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
        let b1118im_a = buffer.get_unchecked(0).im
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
        let b1217im_a = buffer.get_unchecked(0).im
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
        let b1316im_a = buffer.get_unchecked(0).im
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
        let b1415im_a = buffer.get_unchecked(0).im
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
        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
        *buffer.get_unchecked_mut(5) = Complex {
            re: out5re,
            im: out5im,
        };
        *buffer.get_unchecked_mut(6) = Complex {
            re: out6re,
            im: out6im,
        };
        *buffer.get_unchecked_mut(7) = Complex {
            re: out7re,
            im: out7im,
        };
        *buffer.get_unchecked_mut(8) = Complex {
            re: out8re,
            im: out8im,
        };
        *buffer.get_unchecked_mut(9) = Complex {
            re: out9re,
            im: out9im,
        };
        *buffer.get_unchecked_mut(10) = Complex {
            re: out10re,
            im: out10im,
        };
        *buffer.get_unchecked_mut(11) = Complex {
            re: out11re,
            im: out11im,
        };
        *buffer.get_unchecked_mut(12) = Complex {
            re: out12re,
            im: out12im,
        };
        *buffer.get_unchecked_mut(13) = Complex {
            re: out13re,
            im: out13im,
        };
        *buffer.get_unchecked_mut(14) = Complex {
            re: out14re,
            im: out14im,
        };
        *buffer.get_unchecked_mut(15) = Complex {
            re: out15re,
            im: out15im,
        };
        *buffer.get_unchecked_mut(16) = Complex {
            re: out16re,
            im: out16im,
        };
        *buffer.get_unchecked_mut(17) = Complex {
            re: out17re,
            im: out17im,
        };
        *buffer.get_unchecked_mut(18) = Complex {
            re: out18re,
            im: out18im,
        };
        *buffer.get_unchecked_mut(19) = Complex {
            re: out19re,
            im: out19im,
        };
        *buffer.get_unchecked_mut(20) = Complex {
            re: out20re,
            im: out20im,
        };
        *buffer.get_unchecked_mut(21) = Complex {
            re: out21re,
            im: out21im,
        };
        *buffer.get_unchecked_mut(22) = Complex {
            re: out22re,
            im: out22im,
        };
        *buffer.get_unchecked_mut(23) = Complex {
            re: out23re,
            im: out23im,
        };
        *buffer.get_unchecked_mut(24) = Complex {
            re: out24re,
            im: out24im,
        };
        *buffer.get_unchecked_mut(25) = Complex {
            re: out25re,
            im: out25im,
        };
        *buffer.get_unchecked_mut(26) = Complex {
            re: out26re,
            im: out26im,
        };
        *buffer.get_unchecked_mut(27) = Complex {
            re: out27re,
            im: out27im,
        };
        *buffer.get_unchecked_mut(28) = Complex {
            re: out28re,
            im: out28im,
        };
    }
}
impl<T: FFTnum> FFT<T> for Butterfly29<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly29<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        29
    }
}
impl<T> IsInverse for Butterfly29<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
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
    inverse: bool,
}
impl<T: FFTnum> Butterfly31<T> {
    pub fn new(inverse: bool) -> Self {
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 31, inverse);
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 31, inverse);
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 31, inverse);
        let twiddle4: Complex<T> = twiddles::single_twiddle(4, 31, inverse);
        let twiddle5: Complex<T> = twiddles::single_twiddle(5, 31, inverse);
        let twiddle6: Complex<T> = twiddles::single_twiddle(6, 31, inverse);
        let twiddle7: Complex<T> = twiddles::single_twiddle(7, 31, inverse);
        let twiddle8: Complex<T> = twiddles::single_twiddle(8, 31, inverse);
        let twiddle9: Complex<T> = twiddles::single_twiddle(9, 31, inverse);
        let twiddle10: Complex<T> = twiddles::single_twiddle(10, 31, inverse);
        let twiddle11: Complex<T> = twiddles::single_twiddle(11, 31, inverse);
        let twiddle12: Complex<T> = twiddles::single_twiddle(12, 31, inverse);
        let twiddle13: Complex<T> = twiddles::single_twiddle(13, 31, inverse);
        let twiddle14: Complex<T> = twiddles::single_twiddle(14, 31, inverse);
        let twiddle15: Complex<T> = twiddles::single_twiddle(15, 31, inverse);
        Butterfly31 {
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
            inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly31<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // This function was derived in the same manner as the butterflies for length 3, 5 and 7.
        // However, instead of doing it by hand the actual code is autogenerated
        // with the `genbutterflies.py` script in the `tools` directory.
        let x130p = *buffer.get_unchecked(1) + *buffer.get_unchecked(30);
        let x130n = *buffer.get_unchecked(1) - *buffer.get_unchecked(30);
        let x229p = *buffer.get_unchecked(2) + *buffer.get_unchecked(29);
        let x229n = *buffer.get_unchecked(2) - *buffer.get_unchecked(29);
        let x328p = *buffer.get_unchecked(3) + *buffer.get_unchecked(28);
        let x328n = *buffer.get_unchecked(3) - *buffer.get_unchecked(28);
        let x427p = *buffer.get_unchecked(4) + *buffer.get_unchecked(27);
        let x427n = *buffer.get_unchecked(4) - *buffer.get_unchecked(27);
        let x526p = *buffer.get_unchecked(5) + *buffer.get_unchecked(26);
        let x526n = *buffer.get_unchecked(5) - *buffer.get_unchecked(26);
        let x625p = *buffer.get_unchecked(6) + *buffer.get_unchecked(25);
        let x625n = *buffer.get_unchecked(6) - *buffer.get_unchecked(25);
        let x724p = *buffer.get_unchecked(7) + *buffer.get_unchecked(24);
        let x724n = *buffer.get_unchecked(7) - *buffer.get_unchecked(24);
        let x823p = *buffer.get_unchecked(8) + *buffer.get_unchecked(23);
        let x823n = *buffer.get_unchecked(8) - *buffer.get_unchecked(23);
        let x922p = *buffer.get_unchecked(9) + *buffer.get_unchecked(22);
        let x922n = *buffer.get_unchecked(9) - *buffer.get_unchecked(22);
        let x1021p = *buffer.get_unchecked(10) + *buffer.get_unchecked(21);
        let x1021n = *buffer.get_unchecked(10) - *buffer.get_unchecked(21);
        let x1120p = *buffer.get_unchecked(11) + *buffer.get_unchecked(20);
        let x1120n = *buffer.get_unchecked(11) - *buffer.get_unchecked(20);
        let x1219p = *buffer.get_unchecked(12) + *buffer.get_unchecked(19);
        let x1219n = *buffer.get_unchecked(12) - *buffer.get_unchecked(19);
        let x1318p = *buffer.get_unchecked(13) + *buffer.get_unchecked(18);
        let x1318n = *buffer.get_unchecked(13) - *buffer.get_unchecked(18);
        let x1417p = *buffer.get_unchecked(14) + *buffer.get_unchecked(17);
        let x1417n = *buffer.get_unchecked(14) - *buffer.get_unchecked(17);
        let x1516p = *buffer.get_unchecked(15) + *buffer.get_unchecked(16);
        let x1516n = *buffer.get_unchecked(15) - *buffer.get_unchecked(16);
        let sum = *buffer.get_unchecked(0)
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
        let b130re_a = buffer.get_unchecked(0).re
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
        let b229re_a = buffer.get_unchecked(0).re
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
        let b328re_a = buffer.get_unchecked(0).re
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
        let b427re_a = buffer.get_unchecked(0).re
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
        let b526re_a = buffer.get_unchecked(0).re
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
        let b625re_a = buffer.get_unchecked(0).re
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
        let b724re_a = buffer.get_unchecked(0).re
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
        let b823re_a = buffer.get_unchecked(0).re
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
        let b922re_a = buffer.get_unchecked(0).re
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
        let b1021re_a = buffer.get_unchecked(0).re
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
        let b1120re_a = buffer.get_unchecked(0).re
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
        let b1219re_a = buffer.get_unchecked(0).re
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
        let b1318re_a = buffer.get_unchecked(0).re
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
        let b1417re_a = buffer.get_unchecked(0).re
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
        let b1516re_a = buffer.get_unchecked(0).re
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

        let b130im_a = buffer.get_unchecked(0).im
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
        let b229im_a = buffer.get_unchecked(0).im
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
        let b328im_a = buffer.get_unchecked(0).im
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
        let b427im_a = buffer.get_unchecked(0).im
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
        let b526im_a = buffer.get_unchecked(0).im
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
        let b625im_a = buffer.get_unchecked(0).im
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
        let b724im_a = buffer.get_unchecked(0).im
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
        let b823im_a = buffer.get_unchecked(0).im
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
        let b922im_a = buffer.get_unchecked(0).im
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
        let b1021im_a = buffer.get_unchecked(0).im
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
        let b1120im_a = buffer.get_unchecked(0).im
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
        let b1219im_a = buffer.get_unchecked(0).im
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
        let b1318im_a = buffer.get_unchecked(0).im
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
        let b1417im_a = buffer.get_unchecked(0).im
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
        let b1516im_a = buffer.get_unchecked(0).im
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
        *buffer.get_unchecked_mut(0) = sum;
        *buffer.get_unchecked_mut(1) = Complex {
            re: out1re,
            im: out1im,
        };
        *buffer.get_unchecked_mut(2) = Complex {
            re: out2re,
            im: out2im,
        };
        *buffer.get_unchecked_mut(3) = Complex {
            re: out3re,
            im: out3im,
        };
        *buffer.get_unchecked_mut(4) = Complex {
            re: out4re,
            im: out4im,
        };
        *buffer.get_unchecked_mut(5) = Complex {
            re: out5re,
            im: out5im,
        };
        *buffer.get_unchecked_mut(6) = Complex {
            re: out6re,
            im: out6im,
        };
        *buffer.get_unchecked_mut(7) = Complex {
            re: out7re,
            im: out7im,
        };
        *buffer.get_unchecked_mut(8) = Complex {
            re: out8re,
            im: out8im,
        };
        *buffer.get_unchecked_mut(9) = Complex {
            re: out9re,
            im: out9im,
        };
        *buffer.get_unchecked_mut(10) = Complex {
            re: out10re,
            im: out10im,
        };
        *buffer.get_unchecked_mut(11) = Complex {
            re: out11re,
            im: out11im,
        };
        *buffer.get_unchecked_mut(12) = Complex {
            re: out12re,
            im: out12im,
        };
        *buffer.get_unchecked_mut(13) = Complex {
            re: out13re,
            im: out13im,
        };
        *buffer.get_unchecked_mut(14) = Complex {
            re: out14re,
            im: out14im,
        };
        *buffer.get_unchecked_mut(15) = Complex {
            re: out15re,
            im: out15im,
        };
        *buffer.get_unchecked_mut(16) = Complex {
            re: out16re,
            im: out16im,
        };
        *buffer.get_unchecked_mut(17) = Complex {
            re: out17re,
            im: out17im,
        };
        *buffer.get_unchecked_mut(18) = Complex {
            re: out18re,
            im: out18im,
        };
        *buffer.get_unchecked_mut(19) = Complex {
            re: out19re,
            im: out19im,
        };
        *buffer.get_unchecked_mut(20) = Complex {
            re: out20re,
            im: out20im,
        };
        *buffer.get_unchecked_mut(21) = Complex {
            re: out21re,
            im: out21im,
        };
        *buffer.get_unchecked_mut(22) = Complex {
            re: out22re,
            im: out22im,
        };
        *buffer.get_unchecked_mut(23) = Complex {
            re: out23re,
            im: out23im,
        };
        *buffer.get_unchecked_mut(24) = Complex {
            re: out24re,
            im: out24im,
        };
        *buffer.get_unchecked_mut(25) = Complex {
            re: out25re,
            im: out25im,
        };
        *buffer.get_unchecked_mut(26) = Complex {
            re: out26re,
            im: out26im,
        };
        *buffer.get_unchecked_mut(27) = Complex {
            re: out27re,
            im: out27im,
        };
        *buffer.get_unchecked_mut(28) = Complex {
            re: out28re,
            im: out28im,
        };
        *buffer.get_unchecked_mut(29) = Complex {
            re: out29re,
            im: out29im,
        };
        *buffer.get_unchecked_mut(30) = Complex {
            re: out30re,
            im: out30im,
        };
    }
}
impl<T: FFTnum> FFT<T> for Butterfly31<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly31<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        31
    }
}
impl<T> IsInverse for Butterfly31<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct Butterfly32<T> {
    butterfly16: Butterfly16<T>,
    butterfly8: Butterfly8<T>,
    twiddles: [Complex<T>; 7],
    inverse: bool,
}
impl<T: FFTnum> Butterfly32<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly32 {
            butterfly16: Butterfly16::new(inverse),
            butterfly8: Butterfly8::new(inverse),
            twiddles: [
                twiddles::single_twiddle(1, 32, inverse),
                twiddles::single_twiddle(2, 32, inverse),
                twiddles::single_twiddle(3, 32, inverse),
                twiddles::single_twiddle(4, 32, inverse),
                twiddles::single_twiddle(5, 32, inverse),
                twiddles::single_twiddle(6, 32, inverse),
                twiddles::single_twiddle(7, 32, inverse),
            ],
            inverse: inverse,
        }
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly32<T> {
    #[inline(always)]
    unsafe fn process_inplace(&self, buffer: &mut [Complex<T>]) {
        // we're going to hardcode a step of split radix
        // step 1: copy and reorder the  input into the scratch
        let mut scratch_evens = [
            *buffer.get_unchecked(0),
            *buffer.get_unchecked(2),
            *buffer.get_unchecked(4),
            *buffer.get_unchecked(6),
            *buffer.get_unchecked(8),
            *buffer.get_unchecked(10),
            *buffer.get_unchecked(12),
            *buffer.get_unchecked(14),
            *buffer.get_unchecked(16),
            *buffer.get_unchecked(18),
            *buffer.get_unchecked(20),
            *buffer.get_unchecked(22),
            *buffer.get_unchecked(24),
            *buffer.get_unchecked(26),
            *buffer.get_unchecked(28),
            *buffer.get_unchecked(30),
        ];

        let mut scratch_odds_n1 = [
            *buffer.get_unchecked(1),
            *buffer.get_unchecked(5),
            *buffer.get_unchecked(9),
            *buffer.get_unchecked(13),
            *buffer.get_unchecked(17),
            *buffer.get_unchecked(21),
            *buffer.get_unchecked(25),
            *buffer.get_unchecked(29),
        ];
        let mut scratch_odds_n3 = [
            *buffer.get_unchecked(31),
            *buffer.get_unchecked(3),
            *buffer.get_unchecked(7),
            *buffer.get_unchecked(11),
            *buffer.get_unchecked(15),
            *buffer.get_unchecked(19),
            *buffer.get_unchecked(23),
            *buffer.get_unchecked(27),
        ];

        // step 2: column FFTs
        self.butterfly16.process_inplace(&mut scratch_evens);
        self.butterfly8.process_inplace(&mut scratch_odds_n1);
        self.butterfly8.process_inplace(&mut scratch_odds_n3);

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
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[0], &mut scratch_odds_n3[0]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[1], &mut scratch_odds_n3[1]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[2], &mut scratch_odds_n3[2]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[3], &mut scratch_odds_n3[3]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[4], &mut scratch_odds_n3[4]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[5], &mut scratch_odds_n3[5]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[6], &mut scratch_odds_n3[6]);
        Butterfly2::perform_fft_direct(&mut scratch_odds_n1[7], &mut scratch_odds_n3[7]);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.inverse);
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.inverse);
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.inverse);
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.inverse);
        scratch_odds_n3[4] = twiddles::rotate_90(scratch_odds_n3[4], self.inverse);
        scratch_odds_n3[5] = twiddles::rotate_90(scratch_odds_n3[5], self.inverse);
        scratch_odds_n3[6] = twiddles::rotate_90(scratch_odds_n3[6], self.inverse);
        scratch_odds_n3[7] = twiddles::rotate_90(scratch_odds_n3[7], self.inverse);

        //step 5: copy/add/subtract data back to buffer
        *buffer.get_unchecked_mut(0) = scratch_evens[0] + scratch_odds_n1[0];
        *buffer.get_unchecked_mut(1) = scratch_evens[1] + scratch_odds_n1[1];
        *buffer.get_unchecked_mut(2) = scratch_evens[2] + scratch_odds_n1[2];
        *buffer.get_unchecked_mut(3) = scratch_evens[3] + scratch_odds_n1[3];
        *buffer.get_unchecked_mut(4) = scratch_evens[4] + scratch_odds_n1[4];
        *buffer.get_unchecked_mut(5) = scratch_evens[5] + scratch_odds_n1[5];
        *buffer.get_unchecked_mut(6) = scratch_evens[6] + scratch_odds_n1[6];
        *buffer.get_unchecked_mut(7) = scratch_evens[7] + scratch_odds_n1[7];
        *buffer.get_unchecked_mut(8) = scratch_evens[8] + scratch_odds_n3[0];
        *buffer.get_unchecked_mut(9) = scratch_evens[9] + scratch_odds_n3[1];
        *buffer.get_unchecked_mut(10) = scratch_evens[10] + scratch_odds_n3[2];
        *buffer.get_unchecked_mut(11) = scratch_evens[11] + scratch_odds_n3[3];
        *buffer.get_unchecked_mut(12) = scratch_evens[12] + scratch_odds_n3[4];
        *buffer.get_unchecked_mut(13) = scratch_evens[13] + scratch_odds_n3[5];
        *buffer.get_unchecked_mut(14) = scratch_evens[14] + scratch_odds_n3[6];
        *buffer.get_unchecked_mut(15) = scratch_evens[15] + scratch_odds_n3[7];
        *buffer.get_unchecked_mut(16) = scratch_evens[0] - scratch_odds_n1[0];
        *buffer.get_unchecked_mut(17) = scratch_evens[1] - scratch_odds_n1[1];
        *buffer.get_unchecked_mut(18) = scratch_evens[2] - scratch_odds_n1[2];
        *buffer.get_unchecked_mut(19) = scratch_evens[3] - scratch_odds_n1[3];
        *buffer.get_unchecked_mut(20) = scratch_evens[4] - scratch_odds_n1[4];
        *buffer.get_unchecked_mut(21) = scratch_evens[5] - scratch_odds_n1[5];
        *buffer.get_unchecked_mut(22) = scratch_evens[6] - scratch_odds_n1[6];
        *buffer.get_unchecked_mut(23) = scratch_evens[7] - scratch_odds_n1[7];
        *buffer.get_unchecked_mut(24) = scratch_evens[8] - scratch_odds_n3[0];
        *buffer.get_unchecked_mut(25) = scratch_evens[9] - scratch_odds_n3[1];
        *buffer.get_unchecked_mut(26) = scratch_evens[10] - scratch_odds_n3[2];
        *buffer.get_unchecked_mut(27) = scratch_evens[11] - scratch_odds_n3[3];
        *buffer.get_unchecked_mut(28) = scratch_evens[12] - scratch_odds_n3[4];
        *buffer.get_unchecked_mut(29) = scratch_evens[13] - scratch_odds_n3[5];
        *buffer.get_unchecked_mut(30) = scratch_evens[14] - scratch_odds_n3[6];
        *buffer.get_unchecked_mut(31) = scratch_evens[15] - scratch_odds_n3[7];
    }
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.process_inplace(chunk);
        }
    }
}
impl<T: FFTnum> FFT<T> for Butterfly32<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_inplace(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
}
impl<T> Length for Butterfly32<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        32
    }
}
impl<T> IsInverse for Butterfly32<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::algorithm::DFT;
    use crate::test_utils::{check_fft_algorithm, compare_vectors, random_signal};
    use num_traits::Zero;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(false);

                check_fft_algorithm(&butterfly, $size, false);
                check_butterfly(&butterfly, $size, false);

                let butterfly_inverse = $struct_name::new(true);

                check_fft_algorithm(&butterfly_inverse, $size, true);
                check_butterfly(&butterfly_inverse, $size, true);
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
    test_butterfly_func!(test_butterfly11, Butterfly11, 11);
    test_butterfly_func!(test_butterfly13, Butterfly13, 13);
    test_butterfly_func!(test_butterfly16, Butterfly16, 16);
    test_butterfly_func!(test_butterfly17, Butterfly17, 17);
    test_butterfly_func!(test_butterfly19, Butterfly19, 19);
    test_butterfly_func!(test_butterfly23, Butterfly23, 23);
    test_butterfly_func!(test_butterfly29, Butterfly29, 29);
    test_butterfly_func!(test_butterfly31, Butterfly31, 31);
    test_butterfly_func!(test_butterfly32, Butterfly32, 32);

    fn check_butterfly(butterfly: &FFTButterfly<f32>, size: usize, inverse: bool) {
        assert_eq!(
            butterfly.len(),
            size,
            "Butterfly algorithm reported wrong size"
        );
        assert_eq!(
            butterfly.is_inverse(),
            inverse,
            "Butterfly algorithm reported wrong inverse value"
        );

        let n = 5;

        //test the forward direction
        let dft = DFT::new(size, inverse);

        // set up buffers
        let mut expected_input = random_signal(size * n);
        let mut expected_output = vec![Zero::zero(); size * n];

        let mut inplace_buffer = expected_input.clone();
        let mut inplace_multi_buffer = expected_input.clone();

        // perform the test
        dft.process_multi(&mut expected_input, &mut expected_output);

        unsafe {
            butterfly.process_multi_inplace(&mut inplace_multi_buffer);
        }

        for chunk in inplace_buffer.chunks_mut(size) {
            unsafe { butterfly.process_inplace(chunk) };
        }

        assert!(
            compare_vectors(&expected_output, &inplace_buffer),
            "process_inplace() failed, length = {}, inverse = {}",
            size,
            inverse
        );
        assert!(
            compare_vectors(&expected_output, &inplace_multi_buffer),
            "process_multi_inplace() failed, length = {}, inverse = {}",
            size,
            inverse
        );
    }
}
