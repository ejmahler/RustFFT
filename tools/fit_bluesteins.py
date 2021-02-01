import sys
from matplotlib import pyplot as plt
import numpy as np
from read_cachegrind_benches import  read_cachegrind, get_cycles
from scipy.optimize import minimize


calibfile = "target/iai/cachegrind.out.iai_calibration"
calibration = read_cachegrind(calibfile)
length = [10, 30, 50, 70, 90]

inner_length = [128, 256, 512, 1024, 2048]
bluestein_sweeplen_cycles = []
bluestein_sweepinner_cycles = []
inner_cycles = []
overhead_cycles = []
for bslen in length:
    fname = f"target/iai/cachegrind.out.bench_bluesteins_{bslen}_512"
    fname_noop = f"target/iai/cachegrind.out.bench_bluesteins_setup_{bslen}_512"
    results = read_cachegrind(fname)
    cycles = get_cycles(results, calibration)
    results_noop = read_cachegrind(fname_noop)
    cycles_noop = get_cycles(results_noop, calibration)
    fft_cycles = (cycles-cycles_noop)
    bluestein_sweeplen_cycles.append(fft_cycles)

for inlen in inner_length:
    fname = f"target/iai/cachegrind.out.bench_bluesteins_50_{inlen}"
    fname_noop = f"target/iai/cachegrind.out.bench_bluesteins_setup_50_{inlen}"
    results = read_cachegrind(fname)
    cycles = get_cycles(results, calibration)
    results_noop = read_cachegrind(fname_noop)
    cycles_noop = get_cycles(results_noop, calibration)
    fft_cycles = (cycles-cycles_noop)
    bluestein_sweepinner_cycles.append(fft_cycles)

    # Inners
    fname = f"target/iai/cachegrind.out.bench_planned_{inlen}"
    fname_noop = f"target/iai/cachegrind.out.bench_planned_setup_{inlen}"
    results = read_cachegrind(fname)
    cycles = get_cycles(results, calibration)
    results_noop = read_cachegrind(fname_noop)
    cycles_noop = get_cycles(results_noop, calibration)
    fft_cycles_inner = (cycles-cycles_noop)
    inner_cycles.append(2*fft_cycles_inner)

    #overhead_cycles.append(fft_cycles- 2*fft_cycles_inner)

# y = k*x + m
#k, m = np.polyfit(multi_nbr, estimated_cycles, 1)

diff_len = np.array(bluestein_sweeplen_cycles) - inner_cycles[2]
diff_inner = np.array(bluestein_sweepinner_cycles) - np.array(inner_cycles)

def rms_rel_diff(array1, array2, array3, array4):
   return np.sqrt(np.mean(((array1 - array2)/array1) ** 2)) + np.sqrt(np.mean(((array3 - array4)/array3) ** 2))

inner_len_array = np.array(inner_length)
len_array = np.array(length)

f = lambda x: rms_rel_diff( diff_len, x[0] + x[1]*len_array + x[2]*inner_len_array[2], diff_inner, x[0] + x[1]*len_array[2] + x[2]*inner_len_array)
x0 = [0, 1, 1]
res = minimize(f, x0)
print(res.x)
print(f"Overhead: const: {res.x[0]}, slope_len: {res.x[1]}, slope_inner_len: {res.x[2]}")

plt.figure(1)
plt.plot(length, bluestein_sweeplen_cycles, '*')
#plt.plot(length, inner_cycles, '*')
#plt.plot(plot_nbr, k*plot_nbr + m)
plt.figure(2)
plt.plot(inner_length, bluestein_sweepinner_cycles, '*')

plt.figure(3)
plt.plot(length, diff_len, '*')
#plt.plot(length, inner_cycles, '*')
#plt.plot(plot_nbr, k*plot_nbr + m)
plt.figure(4)
plt.plot(inner_length, diff_inner, '*')

plt.show()

