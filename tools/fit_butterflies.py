import sys
from matplotlib import pyplot as plt
import numpy as np
from read_cachegrind_benches import  read_cachegrind, get_cycles

showplots = False
if len(sys.argv)>1:
    showplots = bool(sys.argv[1])
calibfile = "target/iai/cachegrind.out.iai_calibration"
calibration = read_cachegrind(calibfile)
length = [2, 3, 4, 5, 6, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32, 127, 233]
multi_nbr = np.arange(1,11)
plot_nbr = np.arange(0,11)
all_cycles = []
for fftlen in length:
    estimated_cycles = []
    for nbr in multi_nbr:
        fname = f"target/iai/cachegrind.out.bench_planned_multi_{fftlen}_{nbr}"
        fname_noop = f"target/iai/cachegrind.out.bench_planned_multi_setup_{fftlen}_{nbr}"
        results = read_cachegrind(fname)
        cycles = get_cycles(results, calibration)
        results_noop = read_cachegrind(fname_noop)
        cycles_noop = get_cycles(results_noop, calibration)
        fft_cycles = (cycles-cycles_noop)
        #print(f"{fftlen} {fft_cycles}")
        estimated_cycles.append(fft_cycles)
    # y = k*x + m
    k, m = np.polyfit(multi_nbr, estimated_cycles, 1)
    print(f'    {fftlen}: {{ "slope": {k}, "const": {m}}},')
    all_cycles.append(estimated_cycles)
    if showplots:
        plt.figure(fftlen)
        plt.plot(multi_nbr, estimated_cycles, '*')
        plt.plot(plot_nbr, k*plot_nbr + m)
if showplots:
    plt.show()

