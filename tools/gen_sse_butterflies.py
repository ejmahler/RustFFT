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

def stackfuncs(func, items):
    newitems = list(items[1:])
    if len(items)>2:
        inner = stackfuncs(func, newitems)
    else:
        inner = f"{items[1]}"
    return f"{func}({items[0]}, {inner})"

def stackfuncs_flist(funcs, items):
    newitems = list(items[1:])
    if len(items)>2:
        inner = stackfuncs_flist(list(funcs[1:]), newitems)
    else:
        inner = f"{items[1]}"
    return f"{funcs[0]}({items[0]}, {inner})"

fftlen = int(sys.argv[1])

halflen = int((fftlen+1)/2)

for n in range(1, halflen):
    print(f"let x{n}{fftlen-n}p = _mm_add_pd(value{n}, value{fftlen-n});")
    print(f"let x{n}{fftlen-n}n = _mm_sub_pd(value{n}, value{fftlen-n});")



print("")
items = []
for m in range (1, halflen):
    for n in range(1, halflen):
        mn = (m*n)%fftlen
        if mn > fftlen/2:
            mn = fftlen-mn
        print(f"let temp_a{m}_{n} = _mm_mul_pd(self.twiddle{mn}re, x{n}{fftlen-n}p);")

print("")
items = []
for m in range (1, halflen):
    for n in range(1, halflen):
        mn = (m*n)%fftlen
        if mn > fftlen/2:
            mn = fftlen-mn
        print(f"let temp_b{m}_{n} = _mm_mul_pd(self.twiddle{mn}im, x{n}{fftlen-n}n);")
        

print("")
#let temp_a1 = _mm_add_ps(_mm_add_ps(value0, temp_a1_1), _mm_add_ps(temp_a1_2, temp_a1_3));
for m in range(1, halflen):
    items = ["value0"]
    for n in range(1, halflen):
        items.append(f"temp_a{m}_{n}")
    print(f'let temp_a{m} = {stackfuncs("_mm_add_pd", items)};')


# print("")
# for m in range(1, halflen):
#     items = [f"temp_b{m}_1"]
#     funcs = []
#     sign = 1
#     for n in range(2, halflen):
#         items.append(f"temp_b{m}_{n}")
#         mn = (m*n)%fftlen
#         if mn > fftlen/2:
#             if sign > 0:
#                 funcs.append("_mm_sub_pd")
#                 sign = -1
#             else: 
#                 funcs.append("_mm_add_pd")
#                 sign = -1
#         else:
#             if sign > 0:
#                 funcs.append("_mm_add_pd")
#                 #sign = -1
#             else: 
#                 funcs.append("_mm_sub_pd")
#                 #sign = -1
#     print(f'let temp_b{m} = {stackfuncs_flist(funcs, items)};')

print("")
for m in range(1, halflen):
    items = [f"temp_b{m}_1"]
    funcs = []
    signs = []
    for n in range(2, halflen):
        items.append(f"temp_b{m}_{n}")
        mn = (m*n)%fftlen
        if mn > fftlen/2:
            signs.append(-1)
        else:
            signs.append(1)
    for n in range(len(signs)):
        if signs[n] < 0:
            funcs.append("_mm_sub_pd")
            for k in range(n+1, len(signs)):
                signs[k] = -signs[k]
        else:
            funcs.append("_mm_add_pd")
    print(f'let temp_b{m} = {stackfuncs_flist(funcs, items)};')

# let temp_b1_rot = self.rotate.rotate(temp_b1);
# let temp_b2_rot = self.rotate.rotate(temp_b2);
# let temp_b3_rot = self.rotate.rotate(temp_b3);
# let x0 = _mm_add_pd(_mm_add_pd(value0, x16p), _mm_add_pd(x25p, x34p));
# let x1 = _mm_add_pd(temp_a1, temp_b1_rot);
# let x2 = _mm_add_pd(temp_a2, temp_b2_rot);
# let x3 = _mm_add_pd(temp_a3, temp_b3_rot);
# let x4 = _mm_sub_pd(temp_a3, temp_b3_rot);
# let x5 = _mm_sub_pd(temp_a2, temp_b2_rot);
# let x6 = _mm_sub_pd(temp_a1, temp_b1_rot);

print("")
#let temp_a1 = _mm_add_ps(_mm_add_ps(value0, temp_a1_1), _mm_add_ps(temp_a1_2, temp_a1_3));
for m in range(1, halflen):
    print(f'let temp_b{m}_rot = self.rotate.rotate(temp_b{m});')

print("")
items = ["value0"]
for n in range(1, halflen):
    items.append(f"x{n}{fftlen-n}p")
print(f'let x0 = {stackfuncs("_mm_add_pd", items)};')

for m in range(1, halflen):
    print(f"let x{m} = _mm_add_pd(temp_a{m}, temp_b{m}_rot);")

for m in range(1, halflen):
    print(f"let x{m+halflen-1} = _mm_sub_pd(temp_a{halflen-m}, temp_b{halflen-m}_rot);")

items = []
for n in range(0, fftlen):
    items.append(f"x{n}")
print(f'[{", ".join(items)}]')
