use num::{Complex, FromPrimitive, Zero};
use common::{FFTnum, verify_length, verify_length_divisible};

use twiddles;
use super::{FFTAlgorithm, FFTButterfly};




#[inline(always)]
unsafe fn swap_unchecked<T: Copy>(buffer: &mut [T], a: usize, b: usize) {
	let temp = *buffer.get_unchecked(a);
	*buffer.get_unchecked_mut(a) = *buffer.get_unchecked(b);
	*buffer.get_unchecked_mut(b) = temp;
}





pub struct Butterfly2;
impl Butterfly2 {
    #[inline(always)]
    unsafe fn perform_fft<T: FFTnum>(&self, buffer: &mut [Complex<T>]) {
        let temp = *buffer.get_unchecked(0) + *buffer.get_unchecked(1);
        
        *buffer.get_unchecked_mut(1) = *buffer.get_unchecked(0) - *buffer.get_unchecked(1);
        *buffer.get_unchecked_mut(0) = temp;
    }

    #[inline(always)]
    unsafe fn perform_fft_direct<T: FFTnum>(&self, left: &mut Complex<T>, right: &mut Complex<T>) {
        let temp = *left + *right;
        
        *right = *left - *right;
        *left = temp;
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly2 {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
    	for chunk in buffer.chunks_mut(2) {
    		self.perform_fft(chunk);
    	}
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for Butterfly2 {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, 2);
        output.copy_from_slice(input);

        unsafe { self.perform_fft(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, 2);
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
    #[inline(always)]
    fn len(&self) -> usize {
        2
    }
}



pub struct Butterfly3<T> {
	pub twiddle: Complex<T>,
}
impl<T: FFTnum> Butterfly3<T> {
	#[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly3 {
            twiddle: twiddles::single_twiddle(1, 3, inverse)
        }
    }

    #[inline(always)]
    pub fn inverse_of(fft: &Butterfly3<T>) -> Self {
        Butterfly3 {
            twiddle: fft.twiddle.conj()
        }
    }

	#[inline(always)]
	pub unsafe fn perform_fft(&self, buffer: &mut [Complex<T>]) {
        let butterfly2 = Butterfly2{};

    	butterfly2.perform_fft(&mut buffer[1..]);
    	let temp = *buffer.get_unchecked(0);

    	*buffer.get_unchecked_mut(0) = temp + *buffer.get_unchecked(1);

    	*buffer.get_unchecked_mut(1) = *buffer.get_unchecked(1) * self.twiddle.re + temp;
    	*buffer.get_unchecked_mut(2) = *buffer.get_unchecked(2) * Complex{re: Zero::zero(), im: self.twiddle.im};

    	butterfly2.perform_fft(&mut buffer[1..]);
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly3<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.perform_fft(chunk);
        }
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for Butterfly3<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.perform_fft(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, 3);
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
    #[inline(always)]
    fn len(&self) -> usize {
        3
    }
}




pub struct Butterfly4 {
    inverse: bool,
}
impl Butterfly4
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly4 { inverse:inverse }
    }

    #[inline(always)]
    pub unsafe fn perform_fft<T: FFTnum>(&self, buffer: &mut [Complex<T>]) {
        let butterfly2 = Butterfly2{};

        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose, which we're skipping because we're just going to perform non-contiguous FFTs

        // step 2: column FFTs
        {
            let (a, b) = buffer.split_at_mut(2);
            butterfly2.perform_fft_direct(a.get_unchecked_mut(0), b.get_unchecked_mut(0));
            butterfly2.perform_fft_direct(a.get_unchecked_mut(1), b.get_unchecked_mut(1));

            // step 3: apply twiddle factors (only one in this case, and it's either 0 + i or 0 - i)
            *b.get_unchecked_mut(1) = twiddles::rotate_90(*b.get_unchecked(1), self.inverse);

            // step 4: transpose, which we're skipping because we're the previous FFTs were non-contiguous

            // step 5: row FFTs
            butterfly2.perform_fft(a);
            butterfly2.perform_fft(b);
        }

        // step 6: transpose
        swap_unchecked(buffer, 1, 2);
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly4 {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(4) {
            self.perform_fft(chunk);
        }
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for Butterfly4 {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, 4);
        output.copy_from_slice(input);

        unsafe { self.perform_fft(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, 4);
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
    #[inline(always)]
    fn len(&self) -> usize {
        4
    }
}




pub struct Butterfly5<T> {
	inner_fft_multiply: [Complex<T>; 4],
	inverse: bool,
}
impl<T: FFTnum> Butterfly5<T> {
    pub fn new(inverse: bool) -> Self {

    	//we're going to hardcode a raders algorithm of size 5 and an inner FFT of size 4
    	let quarter: T = FromPrimitive::from_f32(0.25f32).unwrap();
    	let twiddle1: Complex<T> = twiddles::single_twiddle(1, 5, inverse) * quarter;
    	let twiddle2: Complex<T> = twiddles::single_twiddle(2, 5, inverse) * quarter;

    	//our primitive root will be 2, and our inverse will be 3. the powers of 3 mod 5 are 1.3.4.2, so we hardcode to use the twiddles in that order
    	let mut fft_data = [twiddle1, twiddle2.conj(), twiddle1.conj(), twiddle2];

    	let butterfly = Butterfly4::new(inverse);
    	unsafe { butterfly.perform_fft(&mut fft_data) };

        Butterfly5 { 
        	inner_fft_multiply: fft_data,
        	inverse: inverse,
        }
    }

    #[inline(always)]
    pub unsafe fn perform_fft(&self, buffer: &mut [Complex<T>]) {
    	//we're going to reorder the buffer directly into our scratch vec
    	//our primitive root is 2. the powers of 2 mod 5 are 1, 2,4,3 so use that ordering
	    let mut scratch = [*buffer.get_unchecked(1), *buffer.get_unchecked(2), *buffer.get_unchecked(4), *buffer.get_unchecked(3)];

    	//perform the first inner FFT
    	Butterfly4::new(self.inverse).perform_fft(&mut scratch);

    	//multiply the fft result with our precomputed data
    	for i in 0..4 {
    		scratch[i] = scratch[i] * self.inner_fft_multiply[i];
    	}

    	//perform the second inner FFT
    	Butterfly4::new(!self.inverse).perform_fft(&mut scratch);
    	//the first element of the output is the sum of the rest
    	let first_input = *buffer.get_unchecked_mut(0);
    	let mut sum = first_input;
    	for i in 1..5 {
    		sum = sum + *buffer.get_unchecked_mut(i);
    	}
    	*buffer.get_unchecked_mut(0) = sum;

    	//use the inverse root ordering to copy data back out
    	*buffer.get_unchecked_mut(1) = scratch[0] + first_input;
    	*buffer.get_unchecked_mut(3) = scratch[1] + first_input;
    	*buffer.get_unchecked_mut(4) = scratch[2] + first_input;
    	*buffer.get_unchecked_mut(2) = scratch[3] + first_input;
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly5<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.perform_fft(chunk);
        }
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for Butterfly5<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.perform_fft(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, 5);
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
    #[inline(always)]
    fn len(&self) -> usize {
        5
    }
}





pub struct Butterfly6<T> {
	butterfly3: Butterfly3<T>,
}
impl<T: FFTnum> Butterfly6<T> {

    pub fn new(inverse: bool) -> Self {
        Butterfly6 { butterfly3: Butterfly3::new(inverse) }
    }
    pub fn inverse_of(fft: &Butterfly6<T>) -> Self {
        Butterfly6 { butterfly3: Butterfly3::inverse_of(&fft.butterfly3) }
    }

	#[inline(always)]
    pub unsafe fn perform_fft(&self, buffer: &mut [Complex<T>]) {
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
    	self.butterfly3.perform_fft(&mut scratch_a);
    	self.butterfly3.perform_fft(&mut scratch_b);

    	// step 3: apply twiddle factors -- SKIPPED because good-thomas doesn't have twiddle factors :)

        // step 4: SKIPPED because the next FFTs will be non-contiguous

        // step 5: row FFTs
        let butterfly2 = Butterfly2{};
        butterfly2.perform_fft_direct(&mut scratch_a[0], &mut scratch_b[0]);
        butterfly2.perform_fft_direct(&mut scratch_a[1], &mut scratch_b[1]);
        butterfly2.perform_fft_direct(&mut scratch_a[2], &mut scratch_b[2]);

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
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly6<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(self.len()) {
            self.perform_fft(chunk);
        }
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for Butterfly6<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.perform_fft(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, 6);
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
    #[inline(always)]
    fn len(&self) -> usize {
        6
    }
}


pub struct Butterfly7<T> {
    inner_fft: Butterfly6<T>,
    inner_fft_multiply: [Complex<T>; 6]
}
impl<T: FFTnum> Butterfly7<T> {
    pub fn new(inverse: bool) -> Self {

        //we're going to hardcode a raders algorithm of size 5 and an inner FFT of size 4
        let sixth: T = FromPrimitive::from_f64(1f64/6f64).unwrap();
        let twiddle1: Complex<T> = twiddles::single_twiddle(1, 7, inverse) * sixth;
        let twiddle2: Complex<T> = twiddles::single_twiddle(2, 7, inverse) * sixth;
        let twiddle3: Complex<T> = twiddles::single_twiddle(3, 7, inverse) * sixth;

        //our primitive root will be 3, and our inverse will be 5. the powers of 5 mod 7 are 1,5,4,6,2,3, so we hardcode to use the twiddles in that order
        let mut fft_data = [twiddle1, twiddle2.conj(), twiddle3.conj(), twiddle1.conj(), twiddle2, twiddle3];

        let butterfly = Butterfly6::new(inverse);
        unsafe { butterfly.perform_fft(&mut fft_data) };

        Butterfly7 { 
            inner_fft: butterfly,
            inner_fft_multiply: fft_data,
        }
    }

    #[inline(always)]
    pub unsafe fn perform_fft(&self, buffer: &mut [Complex<T>]) {
        //we're going to reorder the buffer directly into our scratch vec
        //our primitive root is 3. use 3^n mod 7 to determine which index to copy from
        let mut scratch = [
            *buffer.get_unchecked(3),
            *buffer.get_unchecked(2),
            *buffer.get_unchecked(6),
            *buffer.get_unchecked(4),
            *buffer.get_unchecked(5),
            *buffer.get_unchecked(1),
            ];

        //perform the first inner FFT
        self.inner_fft.perform_fft(&mut scratch);

        //multiply the fft result with our precomputed data
        for i in 0..6 {
            scratch[i] = scratch[i] * self.inner_fft_multiply[i];
        }

        //perform the second inner FFT
        let inverse6 = Butterfly6::inverse_of(&self.inner_fft);
        inverse6.perform_fft(&mut scratch);

        //the first element of the output is the sum of the rest
        let first_input = *buffer.get_unchecked(0);
        let mut sum = first_input;
        for i in 1..7 {
            sum = sum + *buffer.get_unchecked_mut(i);
        }
        *buffer.get_unchecked_mut(0) = sum;

        //use the inverse root ordering to copy data back out
        *buffer.get_unchecked_mut(5) = scratch[0] + first_input;
        *buffer.get_unchecked_mut(4) = scratch[1] + first_input;
        *buffer.get_unchecked_mut(6) = scratch[2] + first_input;
        *buffer.get_unchecked_mut(2) = scratch[3] + first_input;
        *buffer.get_unchecked_mut(3) = scratch[4] + first_input;
        *buffer.get_unchecked_mut(1) = scratch[5] + first_input;
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly7<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(7) {
            self.perform_fft(chunk);
        }
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for Butterfly7<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());
        output.copy_from_slice(input);

        unsafe { self.perform_fft(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, 7);
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
    #[inline(always)]
    fn len(&self) -> usize {
        7
    }
}



pub struct Butterfly8<T> {
    twiddle: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly8<T>
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly8 {
            inverse: inverse,
            twiddle: twiddles::single_twiddle(1, 8, inverse)
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

    #[inline(always)]
    pub unsafe fn perform_fft(&self, buffer: &mut [Complex<T>]) {
        let butterfly2 = Butterfly2{};
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
        butterfly4.perform_fft(&mut scratch[..4]);
        butterfly4.perform_fft(&mut scratch[4..]);

        // step 3: apply twiddle factors
        let twiddle1 = self.twiddle;
        let twiddle3 = Complex{ re: -twiddle1.re, im: twiddle1.im };

        *scratch.get_unchecked_mut(5) = scratch.get_unchecked(5) * twiddle1;
        *scratch.get_unchecked_mut(6) = twiddles::rotate_90(*scratch.get_unchecked(6), self.inverse);
        *scratch.get_unchecked_mut(7) = scratch.get_unchecked(7) * twiddle3;

        // step 4: transpose
        Self::transpose_4x2_to_2x4(&mut scratch);

        // step 5: row FFTs
        butterfly2.perform_fft(&mut scratch[..2]);
        butterfly2.perform_fft(&mut scratch[2..4]);
        butterfly2.perform_fft(&mut scratch[4..6]);
        butterfly2.perform_fft(&mut scratch[6..]);

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
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly8<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(8) {
            self.perform_fft(chunk);
        }
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for Butterfly8<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, 8);
        output.copy_from_slice(input);

        unsafe { self.perform_fft(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, 8);
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
    #[inline(always)]
    fn len(&self) -> usize {
        8
    }
}



pub struct Butterfly16<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly16<T>
{
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Butterfly16 {
            inverse: inverse,
            twiddle1: twiddles::single_twiddle(1, 16, inverse),
            twiddle2: twiddles::single_twiddle(2, 16, inverse),
        }
    }

    #[inline(always)]
    unsafe fn transpose_square(buffer: &mut [Complex<T>]) {
        swap_unchecked(buffer, 1, 4);
        swap_unchecked(buffer, 2, 8);
        swap_unchecked(buffer, 3, 12);

        swap_unchecked(buffer, 6, 9);
        swap_unchecked(buffer, 7, 13);

        swap_unchecked(buffer, 11, 14);
    }

    #[inline(always)]
    pub unsafe fn perform_fft(&self, buffer: &mut [Complex<T>]) {
        let butterfly4 = Butterfly4::new(self.inverse);

        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose the input into the scratch
        Self::transpose_square(buffer);

        // step 2: column FFTs
        butterfly4.perform_fft(&mut buffer[..4]);
        butterfly4.perform_fft(&mut buffer[4..8]);
        butterfly4.perform_fft(&mut buffer[8..12]);
        butterfly4.perform_fft(&mut buffer[12..]);

        // step 3: apply twiddle factors
        let twiddle1 = self.twiddle1;
        let twiddle2 = self.twiddle2;
        let twiddle3 = if self.inverse {
            Complex{ re: twiddle1.im, im: twiddle1.re }
        } else {
            Complex{ re: -twiddle1.im, im: -twiddle1.re }
        };
        let twiddle6 = Complex{ re: -twiddle2.re, im: twiddle2.im };
        let twiddle9 = -twiddle1;

        *buffer.get_unchecked_mut(5) = buffer.get_unchecked(5) * twiddle1;
        *buffer.get_unchecked_mut(6) = buffer.get_unchecked(6) * twiddle2;
        *buffer.get_unchecked_mut(7) = buffer.get_unchecked(7) * twiddle3;

        *buffer.get_unchecked_mut(9) = buffer.get_unchecked(9) * twiddle2;
        *buffer.get_unchecked_mut(10) = twiddles::rotate_90(*buffer.get_unchecked(10), self.inverse);
        *buffer.get_unchecked_mut(11) = buffer.get_unchecked(11) * twiddle6;

        *buffer.get_unchecked_mut(13) = buffer.get_unchecked(13) * twiddle3;
        *buffer.get_unchecked_mut(14) = buffer.get_unchecked(14) * twiddle6;
        *buffer.get_unchecked_mut(15) = buffer.get_unchecked(15) * twiddle9;

        // step 4: transpose
        Self::transpose_square(buffer);

        // step 5: row FFTs
        butterfly4.perform_fft(&mut buffer[..4]);
        butterfly4.perform_fft(&mut buffer[4..8]);
        butterfly4.perform_fft(&mut buffer[8..12]);
        butterfly4.perform_fft(&mut buffer[12..]);

        // step 6: transpose
        Self::transpose_square(buffer);
    }
}
impl<T: FFTnum> FFTButterfly<T> for Butterfly16<T> {
    #[inline(always)]
    unsafe fn process_multi_inplace(&self, buffer: &mut [Complex<T>]) {
        for chunk in buffer.chunks_mut(16) {
            self.perform_fft(chunk);
        }
    }
}
impl<T: FFTnum> FFTAlgorithm<T> for Butterfly16<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, 16);
        output.copy_from_slice(input);

        unsafe { self.perform_fft(output) };
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, 16);
        output.copy_from_slice(input);

        unsafe { self.process_multi_inplace(output) };
    }
    #[inline(always)]
    fn len(&self) -> usize {
        16
    }
}




#[cfg(test)]
mod unit_tests {
	use super::*;
	use test_utils::{random_signal, compare_vectors};
	use algorithm::DFT;
	use num::Zero;

	#[test]
	fn test_butterfly2() {
		let n = 5;
		const SIZE: usize = 2;

		let butterfly2 = Butterfly2{};
		let dft = DFT::new(SIZE, false);
		let dft_inverse = DFT::new(SIZE, true);

		let input_data = random_signal(n * SIZE);

		for (i, input) in input_data.chunks(SIZE).enumerate() {
            //compute expected values
            let mut expected_input = input.to_vec();
            let mut expected = [Zero::zero(); SIZE];
            dft.process(&mut expected_input, &mut expected);

            let mut expected_inverse_input = input.to_vec();
            let mut expected_inverse = [Zero::zero(); SIZE];
            dft_inverse.process(&mut expected_inverse_input, &mut expected_inverse);

            //test process method
            let mut process_input = input.to_vec();
            let mut actual = [Zero::zero(); SIZE];
			butterfly2.process(&mut process_input, &mut actual);

            assert!(compare_vectors(&actual, &expected), "forward, i = {}", i);
            assert!(compare_vectors(&actual, &expected_inverse), "inverse, i = {}", i);

            //test perform_fft method
            let mut inplace = input.to_vec();
			unsafe { butterfly2.perform_fft(&mut inplace) };
			
			assert!(compare_vectors(&expected, &inplace), "forward inplace, i = {}", i);
		}
	}

	//the tests for all butterflies except 2 will be identical except for the identifiers used and size
	//so it's ideal for a macro
	//butterfly 2 is different because it's the only one that doesn't care about forwards vs inverse
	macro_rules! test_butterfly_func {
		($test_name:ident, $struct_name:ident, $size:expr) => (
			#[test]
			fn $test_name() {
				let n = 5;
				const SIZE: usize = $size;
				let input_data = random_signal(n * SIZE);

				//test the forward direction
				let dft = DFT::new(SIZE, false);
				let fft = $struct_name::new(false);
				for (i, input) in input_data.chunks(SIZE).enumerate() {
					//compute expected values
                    let mut expected_input = input.to_vec();
                    let mut expected = [Zero::zero(); SIZE];
                    dft.process(&mut expected_input, &mut expected);

                    //test process method
                    let mut process_input = input.to_vec();
                    let mut actual = [Zero::zero(); SIZE];
                    fft.process(&mut process_input, &mut actual);

                    if SIZE == 16 {
                        println!("actual:");
                        for chunk in actual.chunks(4) {
                            println!("{:?}", chunk);
                        }
                        println!("");
                        println!("expected:");
                        for chunk in expected.chunks(4) {
                            println!("{:?}", chunk);
                        }
                    }

                    assert!(compare_vectors(&expected, &actual), "forward, i = {}", i);

                    //test perform_fft method
                    let mut inplace = input.to_vec();
                    unsafe { fft.perform_fft(&mut inplace) };
                    
                    assert!(compare_vectors(&expected, &inplace), "forward inplace, i = {}", i);
				}

				//make sure the inverse works too
				let dft_inverse = DFT::new(SIZE, true);
				let fft_inverse = $struct_name::new(true);
				for (i, input) in input_data.chunks(SIZE).enumerate() {	
					//compute expected values
                    let mut expected_inverse_input = input.to_vec();
                    let mut expected_inverse = [Zero::zero(); SIZE];
                    dft_inverse.process(&mut expected_inverse_input, &mut expected_inverse);

                    //test process method
                    let mut process_input = input.to_vec();
                    let mut actual = [Zero::zero(); SIZE];
                    fft_inverse.process(&mut process_input, &mut actual);

                    assert!(compare_vectors(&expected_inverse, &actual), "inverse, i = {}", i);

                    //test perform_fft method
                    let mut inplace = input.to_vec();
                    unsafe { fft_inverse.perform_fft(&mut inplace) };
                    
                    assert!(compare_vectors(&expected_inverse, &inplace), "inverse inplace, i = {}", i);
				}
			}
		)
	}
	test_butterfly_func!(test_butterfly3, Butterfly3, 3);
	test_butterfly_func!(test_butterfly4, Butterfly4, 4);
	test_butterfly_func!(test_butterfly5, Butterfly5, 5);
	test_butterfly_func!(test_butterfly6, Butterfly6, 6);
    test_butterfly_func!(test_butterfly7, Butterfly7, 7);
    test_butterfly_func!(test_butterfly8, Butterfly8, 8);
    test_butterfly_func!(test_butterfly16, Butterfly16, 16);
}
