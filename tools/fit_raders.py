import sys
from matplotlib import pyplot as plt
import numpy as np
from read_cachegrind_benches import  read_cachegrind, get_cycles


calibfile = "target/iai/cachegrind.out.iai_calibration"
calibration = read_cachegrind(calibfile)
length = [73, 179, 283, 419, 547, 661, 811, 947, 1087, 1229]
rader_cycles = []
inner_cycles = []
overhead_cycles = []
for fftlen in length:
    fname = f"target/iai/cachegrind.out.bench_raders_{fftlen}"
    fname_noop = f"target/iai/cachegrind.out.bench_raders_setup_{fftlen}"
    results = read_cachegrind(fname)
    cycles = get_cycles(results, calibration)
    results_noop = read_cachegrind(fname_noop)
    cycles_noop = get_cycles(results_noop, calibration)
    fft_cycles = (cycles-cycles_noop)
    rader_cycles.append(fft_cycles)

    # Inners
    fname = f"target/iai/cachegrind.out.bench_planned_{fftlen-1}"
    fname_noop = f"target/iai/cachegrind.out.bench_planned_setup_{fftlen-1}"
    results = read_cachegrind(fname)
    cycles = get_cycles(results, calibration)
    results_noop = read_cachegrind(fname_noop)
    cycles_noop = get_cycles(results_noop, calibration)
    fft_cycles_inner = (cycles-cycles_noop)
    inner_cycles.append(2*fft_cycles_inner)

    overhead_cycles.append(fft_cycles- 2*fft_cycles_inner)

# y = k*x + m
k, m = np.polyfit(length, overhead_cycles, 1)
print(f"slope: {k}, const: {m}")
plt.figure(1)
plt.plot(length, rader_cycles, '*')
plt.plot(length, inner_cycles, '*')
#plt.plot(plot_nbr, k*plot_nbr + m)
plt.figure(2)
plt.plot(length, overhead_cycles, '*')
plt.show()

