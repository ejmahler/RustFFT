import sys
from matplotlib import pyplot as plt
from read_cachegrind_benches import  read_cachegrind, get_cycles
from scipy.optimize import minimize
import numpy as np



calibfile = "target/iai/cachegrind.out.iai_calibration"
calibration = read_cachegrind(calibfile)

radix4_length = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
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

np_len = np.array(radix4_length)
np_cycles = np.array(radix4_cycles)
f = lambda x: rms_rel_diff( np_cycles, x[0] + x[1]*np_len**x[2])
x0 = [0, 1, 1]
res = minimize(f, x0)

print(res.x)
cycles_fit = res.x[0] + res.x[1]*np_len**res.x[2]

plt.figure(30)
plt.loglog(radix4_length, radix4_cycles, '*')
plt.loglog(radix4_length, cycles_fit)
plt.show()

