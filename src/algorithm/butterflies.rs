use num_complex::Complex;
use num_traits::{FromPrimitive, Zero};

use common::FFTnum;

use twiddles;
use ::{Length, IsInverse, Fft};

#[allow(unused)]
macro_rules! boilerplate_fft_butterfly {
    ($struct_name:ident, $len:expr, $inverse_fn:expr) => (
		impl<T: FFTnum> Fft<T> for $struct_name<T> {
			fn process_with_scratch(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
				assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
                
                output.copy_from_slice(input);
				unsafe { self.perform_fft_inplace(output) };
			}
			fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
				assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
				assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
        
                output.copy_from_slice(input);
				for out_chunk in output.chunks_exact_mut(self.len()) {
					unsafe { self.perform_fft_inplace(out_chunk) };
				}
			}
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());
        
                unsafe { self.perform_fft_inplace(buffer) };
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    unsafe { self.perform_fft_inplace(chunk) };
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_out_of_place_scratch_len(&self) -> usize {
                0
            }
        }
        impl<T: FFTnum> FFTButterfly<T> for $struct_name<T> {
            #[inline(always)]
            unsafe fn process_butterfly_inplace(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_inplace(buffer);
            }
            #[inline(always)]
            unsafe fn process_butterfly_multi_inplace(&self, buffer: &mut [Complex<T>]) {
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    self.perform_fft_inplace(chunk);
                }
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len
            }
        }
        impl<T> IsInverse for $struct_name<T> {
            #[inline(always)]
            fn is_inverse(&self) -> bool {
                $inverse_fn(self)
            }
        }
    )
}


pub trait FFTButterfly<T: FFTnum>: Length + IsInverse + Sync + Send {
    /// Computes the FFT in-place in the given buffer
    ///
    /// # Safety
    /// This method performs unsafe reads/writes on `buffer`. Make sure `buffer.len()` is equal to `self.len()`
    unsafe fn process_butterfly_inplace(&self, buffer: &mut [Complex<T>]);

    /// Divides the given buffer into chunks of length `self.len()` and computes an in-place FFT on each chunk
    ///
    /// # Safety
    /// This method performs unsafe reads/writes on `buffer`. Make sure `buffer.len()` is a multiple of `self.len()`
    unsafe fn process_butterfly_multi_inplace(&self, buffer: &mut [Complex<T>]);
}


#[inline(always)]
unsafe fn swap_unchecked<T: Copy>(buffer: &mut [T], a: usize, b: usize) {
	let temp = *buffer.get_unchecked(a);
	*buffer.get_unchecked_mut(a) = *buffer.get_unchecked(b);
	*buffer.get_unchecked_mut(b) = temp;
}





pub struct Butterfly2<T> {
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FFTnum> Butterfly2<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Self { inverse,  _phantom: std::marker::PhantomData }
    }
    #[inline(always)]
    unsafe fn perform_fft_direct(left: &mut Complex<T>, right: &mut Complex<T>) {
        let temp = *left + *right;
        
        *right = *left - *right;
        *left = temp;
    }
    #[inline(always)]
    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
        let temp = *buffer.get_unchecked(0) + *buffer.get_unchecked(1);
        
        *buffer.get_unchecked_mut(1) = *buffer.get_unchecked(0) - *buffer.get_unchecked(1);
        *buffer.get_unchecked_mut(0) = temp;
    }
}
boilerplate_fft_butterfly!(Butterfly2, 2, |this: &Butterfly2<_>| this.inverse);

pub struct Butterfly3<T> {
	pub twiddle: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly3<T> {
	#[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Self {
            twiddle: T::generate_twiddle_factor(1, 3, inverse),
            inverse: inverse,
        }
    }
    #[inline(always)]
    pub fn inverse_of(fft: &Butterfly3<T>) -> Self {
        Self {
            twiddle: fft.twiddle.conj(),
            inverse: !fft.inverse,
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
        let butterfly2 = Butterfly2::new(self.inverse);

        butterfly2.perform_fft_inplace(&mut buffer[1..]);
        let temp = *buffer.get_unchecked(0);

        *buffer.get_unchecked_mut(0) = temp + *buffer.get_unchecked(1);

        *buffer.get_unchecked_mut(1) = *buffer.get_unchecked(1) * self.twiddle.re + temp;
        *buffer.get_unchecked_mut(2) = *buffer.get_unchecked(2) * Complex{re: Zero::zero(), im: self.twiddle.im};

        butterfly2.perform_fft_inplace(&mut buffer[1..]);
    }
}
boilerplate_fft_butterfly!(Butterfly3, 3, |this: &Butterfly3<_>| this.inverse);

pub struct Butterfly4<T> {
    inverse: bool,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FFTnum> Butterfly4<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Self { inverse, _phantom: std::marker::PhantomData }
    }
    #[inline(always)]
    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
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
}
boilerplate_fft_butterfly!(Butterfly4, 4, |this: &Butterfly4<_>| this.inverse);

pub struct Butterfly5<T> {
	inner_fft_multiply: [Complex<T>; 4],
	inverse: bool,
}
impl<T: FFTnum> Butterfly5<T> {
    pub fn new(inverse: bool) -> Self {
    	//we're going to hardcode a raders algorithm of size 5 and an inner FFT of size 4
    	let quarter: T = FromPrimitive::from_f32(0.25f32).unwrap();
    	let twiddle1: Complex<T> = T::generate_twiddle_factor(1, 5, inverse) * quarter;
    	let twiddle2: Complex<T> = T::generate_twiddle_factor(2, 5, inverse) * quarter;

    	//our primitive root will be 2, and our inverse will be 3. the powers of 3 mod 5 are 1.3.4.2, so we hardcode to use the twiddles in that order
    	let mut fft_data = [twiddle1, twiddle2.conj(), twiddle1.conj(), twiddle2];

    	let butterfly = Butterfly4::new(inverse);
    	unsafe { butterfly.process_butterfly_inplace(&mut fft_data) };

        Self { 
        	inner_fft_multiply: fft_data,
        	inverse,
        }
    }

    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
        //we're going to reorder the buffer directly into our scratch vec
        //our primitive root is 2. the powers of 2 mod 5 are 1, 2,4,3 so use that ordering
        let mut scratch = [*buffer.get_unchecked(1), *buffer.get_unchecked(2), *buffer.get_unchecked(4), *buffer.get_unchecked(3)];

        //perform the first inner FFT
        Butterfly4::new(self.inverse).perform_fft_inplace(&mut scratch);

        //multiply the fft result with our precomputed data
        for i in 0..4 {
            scratch[i] = scratch[i] * self.inner_fft_multiply[i];
        }

        //perform the second inner FFT
        Butterfly4::new(!self.inverse).perform_fft_inplace(&mut scratch);

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
boilerplate_fft_butterfly!(Butterfly5, 5, |this: &Butterfly5<_>| this.inverse);

pub struct Butterfly6<T> {
	butterfly3: Butterfly3<T>,
}
impl<T: FFTnum> Butterfly6<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Self { butterfly3: Butterfly3::new(inverse) }
    }
    #[inline(always)]
    pub fn inverse_of(fft: &Butterfly6<T>) -> Self {
        Self { butterfly3: Butterfly3::inverse_of(&fft.butterfly3) }
    }
    #[inline(always)]
    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
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
        self.butterfly3.perform_fft_inplace(&mut scratch_a);
        self.butterfly3.perform_fft_inplace(&mut scratch_b);

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
}
boilerplate_fft_butterfly!(Butterfly6, 6, |this: &Butterfly6<_>| this.butterfly3.is_inverse());

pub struct Butterfly7<T> {
    inner_fft: Butterfly6<T>,
    inner_fft_multiply: [Complex<T>; 6]
}
impl<T: FFTnum> Butterfly7<T> {
    pub fn new(inverse: bool) -> Self {
        //we're going to hardcode a raders algorithm of size 5 and an inner FFT of size 4
        let sixth: T = FromPrimitive::from_f64(1f64/6f64).unwrap();
        let twiddle1: Complex<T> = T::generate_twiddle_factor(1, 7, inverse) * sixth;
        let twiddle2: Complex<T> = T::generate_twiddle_factor(2, 7, inverse) * sixth;
        let twiddle3: Complex<T> = T::generate_twiddle_factor(3, 7, inverse) * sixth;

        //our primitive root will be 3, and our inverse will be 5. the powers of 5 mod 7 are 1,5,4,6,2,3, so we hardcode to use the twiddles in that order
        let mut fft_data = [twiddle1, twiddle2.conj(), twiddle3.conj(), twiddle1.conj(), twiddle2, twiddle3];

        let butterfly = Butterfly6::new(inverse);
        unsafe { butterfly.perform_fft_inplace(&mut fft_data) };

        Butterfly7 { 
            inner_fft: butterfly,
            inner_fft_multiply: fft_data,
        }
    }
    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
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
        self.inner_fft.perform_fft_inplace(&mut scratch);

        //multiply the fft result with our precomputed data
        for i in 0..6 {
            scratch[i] = scratch[i] * self.inner_fft_multiply[i];
        }

        //perform the second inner FFT
        let inverse6 = Butterfly6::inverse_of(&self.inner_fft);
        inverse6.perform_fft_inplace(&mut scratch);

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
boilerplate_fft_butterfly!(Butterfly7, 7, |this: &Butterfly7<_>| this.inner_fft.is_inverse());

pub struct Butterfly8<T> {
    twiddle: Complex<T>,
    inverse: bool,
}
impl<T: FFTnum> Butterfly8<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Self {
            inverse: inverse,
            twiddle: T::generate_twiddle_factor(1, 8, inverse)
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

    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
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
        butterfly4.perform_fft_inplace(&mut scratch[..4]);
        butterfly4.perform_fft_inplace(&mut scratch[4..]);

        // step 3: apply twiddle factors
        let twiddle1 = self.twiddle;
        let twiddle3 = Complex{ re: -twiddle1.re, im: twiddle1.im };

        *scratch.get_unchecked_mut(5) = scratch.get_unchecked(5) * twiddle1;
        *scratch.get_unchecked_mut(6) = twiddles::rotate_90(*scratch.get_unchecked(6), self.inverse);
        *scratch.get_unchecked_mut(7) = scratch.get_unchecked(7) * twiddle3;

        // step 4: transpose
        Self::transpose_4x2_to_2x4(&mut scratch);

        // step 5: row FFTs
        butterfly2.perform_fft_inplace(&mut scratch[..2]);
        butterfly2.perform_fft_inplace(&mut scratch[2..4]);
        butterfly2.perform_fft_inplace(&mut scratch[4..6]);
        butterfly2.perform_fft_inplace(&mut scratch[6..]);

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
boilerplate_fft_butterfly!(Butterfly8, 8, |this: &Butterfly8<_>| this.inverse);


pub struct Butterfly16<T> {
    butterfly8: Butterfly8<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}
impl<T: FFTnum> Butterfly16<T> {
    #[inline(always)]
    pub fn new(inverse: bool) -> Self {
        Self {
            butterfly8: Butterfly8::new(inverse),
            twiddle1: T::generate_twiddle_factor(1, 16, inverse),
            twiddle2: T::generate_twiddle_factor(2, 16, inverse),
            twiddle3: T::generate_twiddle_factor(3, 16, inverse),
        }
    }

    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
        let butterfly4 = Butterfly4::new(self.is_inverse());

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
        self.butterfly8.perform_fft_inplace(&mut scratch_evens);
        butterfly4.perform_fft_inplace(&mut scratch_odds_n1);
        butterfly4.perform_fft_inplace(&mut scratch_odds_n3);

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
        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.is_inverse());
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.is_inverse());
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.is_inverse());
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.is_inverse());

        //step 5: copy/add/subtract data back to buffer
        *buffer.get_unchecked_mut(0) =  scratch_evens[0] + scratch_odds_n1[0];
        *buffer.get_unchecked_mut(1) =  scratch_evens[1] + scratch_odds_n1[1];
        *buffer.get_unchecked_mut(2) =  scratch_evens[2] + scratch_odds_n1[2];
        *buffer.get_unchecked_mut(3) =  scratch_evens[3] + scratch_odds_n1[3];
        *buffer.get_unchecked_mut(4) =  scratch_evens[4] + scratch_odds_n3[0];
        *buffer.get_unchecked_mut(5) =  scratch_evens[5] + scratch_odds_n3[1];
        *buffer.get_unchecked_mut(6) =  scratch_evens[6] + scratch_odds_n3[2];
        *buffer.get_unchecked_mut(7) =  scratch_evens[7] + scratch_odds_n3[3];
        *buffer.get_unchecked_mut(8) =  scratch_evens[0] - scratch_odds_n1[0];
        *buffer.get_unchecked_mut(9) =  scratch_evens[1] - scratch_odds_n1[1];
        *buffer.get_unchecked_mut(10) = scratch_evens[2] - scratch_odds_n1[2];
        *buffer.get_unchecked_mut(11) = scratch_evens[3] - scratch_odds_n1[3];
        *buffer.get_unchecked_mut(12) = scratch_evens[4] - scratch_odds_n3[0];
        *buffer.get_unchecked_mut(13) = scratch_evens[5] - scratch_odds_n3[1];
        *buffer.get_unchecked_mut(14) = scratch_evens[6] - scratch_odds_n3[2];
        *buffer.get_unchecked_mut(15) = scratch_evens[7] - scratch_odds_n3[3];
    }
}
boilerplate_fft_butterfly!(Butterfly16, 16, |this: &Butterfly16<_>| this.butterfly8.is_inverse());

pub struct Butterfly32<T> {
    butterfly16: Butterfly16<T>,
    butterfly8: Butterfly8<T>,
    twiddles: [Complex<T>; 7],
}
impl<T: FFTnum> Butterfly32<T> {
    pub fn new(inverse: bool) -> Self {
        Self {
            butterfly16: Butterfly16::new(inverse),
            butterfly8: Butterfly8::new(inverse),
            twiddles: [
                T::generate_twiddle_factor(1, 32, inverse),
                T::generate_twiddle_factor(2, 32, inverse),
                T::generate_twiddle_factor(3, 32, inverse),
                T::generate_twiddle_factor(4, 32, inverse),
                T::generate_twiddle_factor(5, 32, inverse),
                T::generate_twiddle_factor(6, 32, inverse),
                T::generate_twiddle_factor(7, 32, inverse),
            ],
        }
    }

    unsafe fn perform_fft_inplace(&self, buffer: &mut [Complex<T>]) {
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
        self.butterfly16.perform_fft_inplace(&mut scratch_evens);
        self.butterfly8.perform_fft_inplace(&mut scratch_odds_n1);
        self.butterfly8.perform_fft_inplace(&mut scratch_odds_n3);

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
        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.is_inverse());
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.is_inverse());
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.is_inverse());
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.is_inverse());
        scratch_odds_n3[4] = twiddles::rotate_90(scratch_odds_n3[4], self.is_inverse());
        scratch_odds_n3[5] = twiddles::rotate_90(scratch_odds_n3[5], self.is_inverse());
        scratch_odds_n3[6] = twiddles::rotate_90(scratch_odds_n3[6], self.is_inverse());
        scratch_odds_n3[7] = twiddles::rotate_90(scratch_odds_n3[7], self.is_inverse());

        //step 5: copy/add/subtract data back to buffer
        *buffer.get_unchecked_mut(0) =  scratch_evens[0] +  scratch_odds_n1[0];
        *buffer.get_unchecked_mut(1) =  scratch_evens[1] +  scratch_odds_n1[1];
        *buffer.get_unchecked_mut(2) =  scratch_evens[2] +  scratch_odds_n1[2];
        *buffer.get_unchecked_mut(3) =  scratch_evens[3] +  scratch_odds_n1[3];
        *buffer.get_unchecked_mut(4) =  scratch_evens[4] +  scratch_odds_n1[4];
        *buffer.get_unchecked_mut(5) =  scratch_evens[5] +  scratch_odds_n1[5];
        *buffer.get_unchecked_mut(6) =  scratch_evens[6] +  scratch_odds_n1[6];
        *buffer.get_unchecked_mut(7) =  scratch_evens[7] +  scratch_odds_n1[7];
        *buffer.get_unchecked_mut(8) =  scratch_evens[8] +  scratch_odds_n3[0];
        *buffer.get_unchecked_mut(9) =  scratch_evens[9] +  scratch_odds_n3[1];
        *buffer.get_unchecked_mut(10) = scratch_evens[10] + scratch_odds_n3[2];
        *buffer.get_unchecked_mut(11) = scratch_evens[11] + scratch_odds_n3[3];
        *buffer.get_unchecked_mut(12) = scratch_evens[12] + scratch_odds_n3[4];
        *buffer.get_unchecked_mut(13) = scratch_evens[13] + scratch_odds_n3[5];
        *buffer.get_unchecked_mut(14) = scratch_evens[14] + scratch_odds_n3[6];
        *buffer.get_unchecked_mut(15) = scratch_evens[15] + scratch_odds_n3[7];
        *buffer.get_unchecked_mut(16) = scratch_evens[0] -  scratch_odds_n1[0];
        *buffer.get_unchecked_mut(17) = scratch_evens[1] -  scratch_odds_n1[1];
        *buffer.get_unchecked_mut(18) = scratch_evens[2] -  scratch_odds_n1[2];
        *buffer.get_unchecked_mut(19) = scratch_evens[3] -  scratch_odds_n1[3];
        *buffer.get_unchecked_mut(20) = scratch_evens[4] -  scratch_odds_n1[4];
        *buffer.get_unchecked_mut(21) = scratch_evens[5] -  scratch_odds_n1[5];
        *buffer.get_unchecked_mut(22) = scratch_evens[6] -  scratch_odds_n1[6];
        *buffer.get_unchecked_mut(23) = scratch_evens[7] -  scratch_odds_n1[7];
        *buffer.get_unchecked_mut(24) = scratch_evens[8] -  scratch_odds_n3[0];
        *buffer.get_unchecked_mut(25) = scratch_evens[9] -  scratch_odds_n3[1];
        *buffer.get_unchecked_mut(26) = scratch_evens[10] - scratch_odds_n3[2];
        *buffer.get_unchecked_mut(27) = scratch_evens[11] - scratch_odds_n3[3];
        *buffer.get_unchecked_mut(28) = scratch_evens[12] - scratch_odds_n3[4];
        *buffer.get_unchecked_mut(29) = scratch_evens[13] - scratch_odds_n3[5];
        *buffer.get_unchecked_mut(30) = scratch_evens[14] - scratch_odds_n3[6];
        *buffer.get_unchecked_mut(31) = scratch_evens[15] - scratch_odds_n3[7];

    }
}
boilerplate_fft_butterfly!(Butterfly32, 32, |this: &Butterfly32<_>| this.butterfly8.is_inverse());



#[cfg(test)]
mod unit_tests {
	use super::*;
	use test_utils::{random_signal, compare_vectors, check_fft_algorithm};
	use algorithm::DFT;
	use num_traits::Zero;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => (
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(false);

                check_fft_algorithm(&butterfly, $size, false);
                check_butterfly(&butterfly, $size, false);

                let butterfly_inverse = $struct_name::new(true);

                check_fft_algorithm(&butterfly_inverse, $size, true);
                check_butterfly(&butterfly_inverse, $size, true);
            }
        )
    }
    test_butterfly_func!(test_butterfly2, Butterfly2, 2);
    test_butterfly_func!(test_butterfly3, Butterfly3, 3);
    test_butterfly_func!(test_butterfly4, Butterfly4, 4);
    test_butterfly_func!(test_butterfly5, Butterfly5, 5);
    test_butterfly_func!(test_butterfly6, Butterfly6, 6);
    test_butterfly_func!(test_butterfly7, Butterfly7, 7);
    test_butterfly_func!(test_butterfly8, Butterfly8, 8);
    test_butterfly_func!(test_butterfly16, Butterfly16, 16);
    test_butterfly_func!(test_butterfly32, Butterfly32, 32);
    

    fn check_butterfly(butterfly: &FFTButterfly<f32>, size: usize, inverse: bool) {
        assert_eq!(butterfly.len(), size, "Butterfly algorithm reported wrong size");
        assert_eq!(butterfly.is_inverse(), inverse, "Butterfly algorithm reported wrong inverse value");

        let n = 5;

        //test the forward direction
        let dft = DFT::new(size, inverse);

        // set up buffers
        let mut expected_input = random_signal(size * n);
        let mut expected_output = vec![Zero::zero(); size * n];

        let mut inplace_buffer = expected_input.clone();
        let mut inplace_multi_buffer = expected_input.clone();

        // perform the test
        dft.process_multi(&mut expected_input, &mut expected_output, &mut []);

        unsafe { butterfly.process_butterfly_multi_inplace(&mut inplace_multi_buffer); }

        for chunk in inplace_buffer.chunks_mut(size) {
            unsafe { butterfly.process_butterfly_inplace(chunk) };
        }

        assert!(compare_vectors(&expected_output, &inplace_buffer), "process_inplace() failed, length = {}, inverse = {}", size, inverse);
        assert!(compare_vectors(&expected_output, &inplace_multi_buffer), "process_multi_inplace() failed, length = {}, inverse = {}", size, inverse);
    }
}
