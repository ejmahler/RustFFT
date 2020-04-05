use num_complex::Complex;
use num_traits::FromPrimitive;

use common::FFTnum;

use twiddles;
use ::{Length, IsInverse, Fft};
use ::array_utils::{RawSlice, RawSliceMut};

#[allow(unused)]
macro_rules! boilerplate_fft_butterfly {
    ($struct_name:ident, $len:expr, $inverse_fn:expr) => (
        impl<T: FFTnum> $struct_name<T> {
            #[inline(always)]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: &mut [Complex<T>]) {
                self.perform_fft_contiguous(RawSlice::new(buffer), RawSliceMut::new(buffer));
            }
        }
		impl<T: FFTnum> Fft<T> for $struct_name<T> {
			fn process_with_scratch(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
				assert_eq!(input.len(), self.len(), "Input is the wrong length. Expected {}, got {}", self.len(), input.len());
                assert_eq!(output.len(), self.len(), "Output is the wrong length. Expected {}, got {}", self.len(), output.len());
                
				unsafe { self.perform_fft_contiguous(RawSlice::new(input), RawSliceMut::new(output)) };
			}
			fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
				assert!(input.len() % self.len() == 0, "Output is the wrong length. Expected multiple of {}, got {}", self.len(), input.len());
				assert_eq!(input.len(), output.len(), "Output is the wrong length. input = {} output = {}", input.len(), output.len());
        
				for (out_chunk, in_chunk) in output.chunks_exact_mut(self.len()).zip(input.chunks_exact(self.len())) {
					unsafe { self.perform_fft_contiguous(RawSlice::new(in_chunk), RawSliceMut::new(out_chunk)) };
				}
			}
            fn process_inplace_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len(), self.len(), "Buffer is the wrong length. Expected {}, got {}", self.len(), buffer.len());
        
                unsafe { self.perform_fft_butterfly(buffer) };
            }
            fn process_inplace_multi(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                assert_eq!(buffer.len() % self.len(), 0, "Buffer is the wrong length. Expected multiple of {}, got {}", self.len(), buffer.len());
        
                for chunk in buffer.chunks_exact_mut(self.len()) {
                    unsafe { self.perform_fft_butterfly(chunk) };
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
    unsafe fn perform_fft_strided(left: &mut Complex<T>, right: &mut Complex<T>) {
        let temp = *left + *right;
        
        *right = *left - *right;
        *left = temp;
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
        let value0 = input.load(0);
        let value1 = input.load(1);
        output.store(value0 + value1, 0);
        output.store(value0 - value1, 1);
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
    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
        let value0 = input.load(0);
        let mut value1 = input.load(1);
        let mut value2 = input.load(2);

        Butterfly2::perform_fft_strided(&mut value1, &mut value2);

        output.store(value0 + value1, 0);
        
        value1 = value1 * self.twiddle.re + value0;
        value2 = twiddles::rotate_90(value2, true) * self.twiddle.im;
        
        Butterfly2::perform_fft_strided(&mut value1, &mut value2);

        output.store(value1, 1);
        output.store(value2, 2);
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
    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
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
        value3 = twiddles::rotate_90(value3, self.inverse);

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
    	unsafe { butterfly.perform_fft_butterfly(&mut fft_data) };

        Self { 
        	inner_fft_multiply: fft_data,
        	inverse,
        }
    }

    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
        //we're going to reorder the buffer directly into our scratch vec
        //our primitive root is 2. the powers of 2 mod 5 are 1, 2,4,3 so use that ordering
        let mut scratch = [input.load(1), input.load(2), input.load(4), input.load(3)];

        let first_input = input.load(0);
        let scratch_sum : Complex<T> = scratch.iter().sum();
        output.store(first_input + scratch_sum, 0);

        //perform the first inner FFT
        Butterfly4::new(self.inverse).perform_fft_butterfly(&mut scratch);

        //multiply the fft result with our precomputed data
        for i in 0..4 {
            scratch[i] = scratch[i] * self.inner_fft_multiply[i];
        }

        //perform the second inner FFT
        Butterfly4::new(!self.inverse).perform_fft_butterfly(&mut scratch);

        //use the inverse root ordering to copy data back out
        output.store(scratch[0] + first_input, 1);
        output.store(scratch[1] + first_input, 3);
        output.store(scratch[2] + first_input, 4);
        output.store(scratch[3] + first_input, 2);
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
    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
        //since GCD(2,3) == 1 we're going to hardcode a step of the Good-Thomas algorithm to avoid twiddle factors

        // step 1: reorder the input directly into the scratch. normally there's a whole thing to compute this ordering
        //but thankfully we can just precompute it and hardcode it
        let mut scratch_a = [
            input.load(0),
            input.load(2),
            input.load(4),
        ];

        let mut scratch_b = [
            input.load(3),
            input.load(5),
            input.load(1),
        ];

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
        unsafe { butterfly.perform_fft_butterfly(&mut fft_data) };

        Butterfly7 { 
            inner_fft: butterfly,
            inner_fft_multiply: fft_data,
        }
    }
    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
        //we're going to reorder the buffer directly into our scratch vec
        //our primitive root is 3. use 3^n mod 7 to determine which index to copy from
        let mut scratch = [
            input.load(3),
            input.load(2),
            input.load(6),
            input.load(4),
            input.load(5),
            input.load(1),
        ];

        let first_input = input.load(0);
        let scratch_sum : Complex<T> = scratch.iter().sum();
        output.store(first_input + scratch_sum, 0);

        //perform the first inner FFT
        self.inner_fft.perform_fft_butterfly(&mut scratch);

        //multiply the fft result with our precomputed data
        for i in 0..6 {
            scratch[i] = scratch[i] * self.inner_fft_multiply[i];
        }

        //perform the second inner FFT
        let inverse6 = Butterfly6::inverse_of(&self.inner_fft);
        inverse6.perform_fft_butterfly(&mut scratch);
        
        //use the inverse root ordering to copy data back out
        output.store(scratch[0] + first_input, 5);
        output.store(scratch[1] + first_input, 4);
        output.store(scratch[2] + first_input, 6);
        output.store(scratch[3] + first_input, 2);
        output.store(scratch[4] + first_input, 3);
        output.store(scratch[5] + first_input, 1);
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

    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
        let butterfly4 = Butterfly4::new(self.inverse);

        //we're going to hardcode a step of mixed radix
        //aka we're going to do the six step algorithm

        // step 1: transpose the input into the scratch
        let mut scratch0 = [
            input.load(0),
            input.load(2),
            input.load(4),
            input.load(6),
        ];
        let mut scratch1 = [
            input.load(1),
            input.load(3),
            input.load(5),
            input.load(7),
        ];

        // step 2: column FFTs
        butterfly4.perform_fft_butterfly(&mut scratch0);
        butterfly4.perform_fft_butterfly(&mut scratch1);

        // step 3: apply twiddle factors
        let twiddle1 = self.twiddle;
        let twiddle3 = Complex{ re: -twiddle1.re, im: twiddle1.im };

        scratch1[1] = scratch1[1] * twiddle1;
        scratch1[2] = twiddles::rotate_90(scratch1[2], self.inverse);
        scratch1[3] = scratch1[3] * twiddle3;

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
            output.store(scratch1[i], i+4);
        }
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

    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
        let butterfly4 = Butterfly4::new(self.is_inverse());

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

        let mut scratch_odds_n1 = [
            input.load(1),
            input.load(5),
            input.load(9),
            input.load(13),
        ];
        let mut scratch_odds_n3 = [
            input.load(15),
            input.load(3),
            input.load(7),
            input.load(11),
        ];

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
        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.is_inverse());
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.is_inverse());
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.is_inverse());
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.is_inverse());

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

    unsafe fn perform_fft_contiguous(&self, input: RawSlice<Complex<T>>, output: RawSliceMut<Complex<T>>) {
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
        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.is_inverse());
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.is_inverse());
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.is_inverse());
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.is_inverse());
        scratch_odds_n3[4] = twiddles::rotate_90(scratch_odds_n3[4], self.is_inverse());
        scratch_odds_n3[5] = twiddles::rotate_90(scratch_odds_n3[5], self.is_inverse());
        scratch_odds_n3[6] = twiddles::rotate_90(scratch_odds_n3[6], self.is_inverse());
        scratch_odds_n3[7] = twiddles::rotate_90(scratch_odds_n3[7], self.is_inverse());

        //step 5: copy/add/subtract data back to buffer
        output.store(scratch_evens[0] +  scratch_odds_n1[0], 0);
        output.store(scratch_evens[1] +  scratch_odds_n1[1], 1);
        output.store(scratch_evens[2] +  scratch_odds_n1[2], 2);
        output.store(scratch_evens[3] +  scratch_odds_n1[3], 3);
        output.store(scratch_evens[4] +  scratch_odds_n1[4], 4);
        output.store(scratch_evens[5] +  scratch_odds_n1[5], 5);
        output.store(scratch_evens[6] +  scratch_odds_n1[6], 6);
        output.store(scratch_evens[7] +  scratch_odds_n1[7], 7);
        output.store(scratch_evens[8] +  scratch_odds_n3[0], 8);
        output.store(scratch_evens[9] +  scratch_odds_n3[1], 9);
        output.store(scratch_evens[10] + scratch_odds_n3[2], 10);
        output.store(scratch_evens[11] + scratch_odds_n3[3], 11);
        output.store(scratch_evens[12] + scratch_odds_n3[4], 12);
        output.store(scratch_evens[13] + scratch_odds_n3[5], 13);
        output.store(scratch_evens[14] + scratch_odds_n3[6], 14);
        output.store(scratch_evens[15] + scratch_odds_n3[7], 15);
        output.store(scratch_evens[0] -  scratch_odds_n1[0], 16);
        output.store(scratch_evens[1] -  scratch_odds_n1[1], 17);
        output.store(scratch_evens[2] -  scratch_odds_n1[2], 18);
        output.store(scratch_evens[3] -  scratch_odds_n1[3], 19);
        output.store(scratch_evens[4] -  scratch_odds_n1[4], 20);
        output.store(scratch_evens[5] -  scratch_odds_n1[5], 21);
        output.store(scratch_evens[6] -  scratch_odds_n1[6], 22);
        output.store(scratch_evens[7] -  scratch_odds_n1[7], 23);
        output.store(scratch_evens[8] -  scratch_odds_n3[0], 24);
        output.store(scratch_evens[9] -  scratch_odds_n3[1], 25);
        output.store(scratch_evens[10] - scratch_odds_n3[2], 26);
        output.store(scratch_evens[11] - scratch_odds_n3[3], 27);
        output.store(scratch_evens[12] - scratch_odds_n3[4], 28);
        output.store(scratch_evens[13] - scratch_odds_n3[5], 29);
        output.store(scratch_evens[14] - scratch_odds_n3[6], 30);
        output.store(scratch_evens[15] - scratch_odds_n3[7], 31);

    }
}
boilerplate_fft_butterfly!(Butterfly32, 32, |this: &Butterfly32<_>| this.butterfly8.is_inverse());



#[cfg(test)]
mod unit_tests {
	use super::*;
	use test_utils::check_fft_algorithm;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => (
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new(false);
                check_fft_algorithm(&butterfly, $size, false);

                let butterfly_inverse = $struct_name::new(true);
                check_fft_algorithm(&butterfly_inverse, $size, true);
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
}
