import sys
from matplotlib import pyplot as plt
from read_cachegrind_benches import  read_cachegrind, get_cycles
from scipy.optimize import minimize
import numpy as np



calibfile = "target/iai/cachegrind.out.iai_calibration"
calibration = read_cachegrind(calibfile)

radix4_length = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
radix4_cycles = []
for fftlen in radix4_length:
    fname = f"target/iai/cachegrind.out.bench_radix4_{fftlen}"
    fname_noop = f"target/iai/cachegrind.out.bench_radix4_setup_{fftlen}"
    results = read_cachegrind(fname)
    cycles = get_cycles(results, calibration)
    results_noop = read_cachegrind(fname_noop)
    cycles_noop = get_cycles(results_noop, calibration)
    estimated_cycles = cycles-cycles_noop
    #print(f"{fftlen} {estimated_cycles}")
    radix4_cycles.append(estimated_cycles)

def rms_rel_diff(array1, array2):
   return np.sqrt(np.mean(((array1 - array2)/array1) ** 2))

np_len_long = np.array(radix4_length[6:])
np_len_short = np.array(radix4_length[:6])
np_cycles_long = np.array(radix4_cycles[6:])
np_cycles_short = np.array(radix4_cycles[:6])
f_long = lambda x: rms_rel_diff( np_cycles_long, x[0] + x[1]*np_len_long**x[2])
f_short = lambda x: rms_rel_diff( np_cycles_short, x[0] + x[1]*np_len_short**x[2])
x0 = [0, 1, 1]
res_long = minimize(f_long, x0)
res_short = minimize(f_short, x0)

print(f"Radix 4 short, const: {res_short.x[0]}, slope: {res_short.x[1]}, exponent {res_short.x[2]}")
print(f"Radix 4 long, const: {res_long.x[0]}, slope: {res_long.x[1]}, exponent {res_long.x[2]}")
cycles_fit_long = res_long.x[0] + res_long.x[1]*np_len_long**res_long.x[2]
cycles_fit_short = res_short.x[0] + res_short.x[1]*np_len_short**res_short.x[2]

plt.figure(30)
plt.loglog(radix4_length, radix4_cycles, '*')
plt.loglog(np_len_long, cycles_fit_long)
plt.loglog(np_len_short, cycles_fit_short)
plt.show()

