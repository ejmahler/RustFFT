# A simple Python script to generate the code for odd-sized optimized DFTs
# The generated code is simply printed in the terminal.
# This is only intended for prime lengths, where the usual tricks can't be used.
# The generated code is O(n^2), but for short lengths this is still faster than fancier algorithms.
# Example, make a length 5 Dft:
# > python genbutterflies.py 5
# Output:
# let x14p = *buffer.get_unchecked(1) + *buffer.get_unchecked(4);
# let x14n = *buffer.get_unchecked(1) - *buffer.get_unchecked(4);
# let x23p = *buffer.get_unchecked(2) + *buffer.get_unchecked(3);
# let x23n = *buffer.get_unchecked(2) - *buffer.get_unchecked(3);
# let sum = *buffer.get_unchecked(0) + x14p + x23p;
# let b14re_a = buffer.get_unchecked(0).re + self.twiddle1.re*x14p.re + self.twiddle2.re*x23p.re;
# let b14re_b = self.twiddle1.im*x14n.im + self.twiddle2.im*x23n.im;
# let b23re_a = buffer.get_unchecked(0).re + self.twiddle2.re*x14p.re + self.twiddle1.re*x23p.re;
# let b23re_b = self.twiddle2.im*x14n.im + -self.twiddle1.im*x23n.im;
# 
# let b14im_a = buffer.get_unchecked(0).im + self.twiddle1.re*x14p.im + self.twiddle2.re*x23p.im;
# let b14im_b = self.twiddle1.im*x14n.re + self.twiddle2.im*x23n.re;
# let b23im_a = buffer.get_unchecked(0).im + self.twiddle2.re*x14p.im + self.twiddle1.re*x23p.im;
# let b23im_b = self.twiddle2.im*x14n.re + -self.twiddle1.im*x23n.re;
# 
# let out1re = b14re_a - b14re_b;
# let out1im = b14im_a + b14im_b;
# let out2re = b23re_a - b23re_b;
# let out2im = b23im_a + b23im_b;
# let out3re = b23re_a + b23re_b;
# let out3im = b23im_a - b23im_b;
# let out4re = b14re_a + b14re_b;
# let out4im = b14im_a - b14im_b;
# *buffer.get_unchecked_mut(0) = sum;
# *buffer.get_unchecked_mut(1) = Complex{ re: out1re, im: out1im };
# *buffer.get_unchecked_mut(2) = Complex{ re: out2re, im: out2im };
# *buffer.get_unchecked_mut(3) = Complex{ re: out3re, im: out3im };
# *buffer.get_unchecked_mut(4) = Complex{ re: out4re, im: out4im };
#
#
# This required the Butterfly5 to already exist, with twiddles defined like this:
# pub struct Butterfly5<T> {
#     twiddle1: Complex<T>,
#     twiddle2: Complex<T>,
# 	direction: FftDirection,
# }
# 
# With twiddle values:
# twiddle1: Complex<T> = twiddles::single_twiddle(1, 5, direction);
# twiddle2: Complex<T> = twiddles::single_twiddle(2, 5, direction);

import sys

def make_shuffling_single_f64(len):
    inputs =  ", ".join([str(n) for n in range(len)])
    print(f"let values = read_complex_to_array!(buffer, {{{inputs}}});")
    print("")
    print("let out = self.perform_fft_direct(values);")
    print("")
    print(f"write_complex_to_array!(out, buffer, {{{inputs}}});")

def make_shuffling_single_f32(len):
    inputs =  ", ".join([str(n) for n in range(len)])
    print(f"let values = read_partial1_complex_to_array!(buffer, {{{inputs}}});")
    print("")
    print("let out = self.perform_parallel_fft_direct(values);")
    print("")
    print(f"write_partial_lo_complex_to_array!(out, buffer, {{{inputs}}});")


def make_shuffling_parallel_f32(len):
    inputs =  ", ".join([str(2*n) for n in range(len)])
    outputs =  ", ".join([str(n) for n in range(len)])
    print(f"let input_packed = read_complex_to_array!(buffer, {{{inputs}}});")
    print("")
    print("let values = [")
    for n in range(int(len/2)):
        print(f"    extract_lo_hi_f32(input_packed[{int(n)}], input_packed[{int(len/2 + n)}]),")
        print(f"    extract_hi_lo_f32(input_packed[{int(n)}], input_packed[{int(len/2 + n+1)}]),")
    print(f"    extract_lo_hi_f32(input_packed[{int(len/2)}], input_packed[{int(len-1)}]),")
    print("];")
    print("")
    print("let out = self.perform_parallel_fft_direct(values);")
    print("")
    print("let out_packed = [")
    for n in range(int(len/2)):
        print(f"    extract_lo_lo_f32(out[{int(2*n)}], out[{int(2*n+1)}]),")
    print(f"    extract_lo_hi_f32(out[{int(len-1)}], out[0]),")
    for n in range(int(len/2)):
        print(f"    extract_hi_hi_f32(out[{int(2*n+1)}], out[{int(2*n+2)}]),")
    print("];")
    print("")
    print(f"write_complex_to_array_strided!(out_packed, buffer, 2, {{{outputs}}});")


def make_butterfly(len, rotatefunc, fmaddfunc, fnmaddfunc):
    halflen = int((len+1)/2)

    print("let out00 = values[0];")
    for n in range(1, halflen):
        print(f"let [x{n}p{len-n}, x{n}m{len-n}] =  NeonVector::column_butterfly2([values[{n}], values[{len-n}]]);")
        print(f'let r{n}m{len-n} = self.rotate.{rotatefunc}(x{n}m{len-n});')
        print(f"let out00 = NeonVector::add(out00, x{n}p{len-n});")
    print("")

    for n in range(1, halflen):
        for m in range(1, halflen):
            variable_name = f"m{n}{len-n:02}a"
            mn = (m*n)%len
            if mn > len/2:
                mn = len-mn
            if m == 1:
                print(f"let {variable_name} = {fmaddfunc}(values[0], self.twiddle{mn}re, x{m}p{len-m});")
            else:
                print(f"let {variable_name} = {fmaddfunc}({variable_name}, self.twiddle{mn}re, x{m}p{len-m});")

        for m in range(1, halflen):
            variable_name = f"m{n}{len-n:02}b"
            mn = (m*n)%len
            fn = fmaddfunc
            if mn > len/2:
                mn = len-mn
                fn = fnmaddfunc
            if m == 1:
                print(f"let {variable_name} = NeonVector::mul(r{m}m{len-m}, self.twiddle{mn}im);")
            else:
                print(f"let {variable_name} = {fn}({variable_name}, self.twiddle{mn}im, r{m}m{len-m});")
        print(f"let [out{n:02}, out{len-n:02}] = NeonVector::column_butterfly2([m{n}{len-n:02}a, m{n}{len-n:02}b]);")
        print("")
    print("")

    items = []
    for n in range(0, len):
        items.append(f"out{n:02}")
    print(f'[{", ".join(items)}]')


if __name__ == "__main__":
    len = int(sys.argv[1])
    print("\n\n--------------- f32 ---------------")
    print("\n ----- perform_fft_contiguous -----")
    make_shuffling_single_f32(len)
    print("\n ----- perform_parallel_fft_contiguous -----")
    make_shuffling_parallel_f32(len)
    print("\n ----- perform_parallel_fft_direct -----")
    make_butterfly(len, "rotate_both", "fmadd_f32", "nmadd_f32")

    print("\n\n--------------- f64 ---------------")
    print("\n ----- perform_fft_contiguous -----")
    make_shuffling_single_f64(len)
    print("\n ----- perform_parallel_fft_direct -----")
    make_butterfly(len, "rotate", "fmadd_f64", "nmadd_f64")
