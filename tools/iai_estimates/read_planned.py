import sys
from matplotlib import pyplot as plt
import numpy as np
from read_cachegrind_benches import  read_cachegrind, get_cycles

showplots = False
if len(sys.argv)>1:
    showplots = bool(sys.argv[1])
calibfile = "target/iai/cachegrind.out.iai_calibration"
calibration = read_cachegrind(calibfile)
length = [127, 233, 1031, 2003]
for fftlen in length:
    fname = f"target/iai/cachegrind.out.bench_planned_{fftlen}"
    fname_noop = f"target/iai/cachegrind.out.bench_planned_setup_{fftlen}"
    results = read_cachegrind(fname)
    cycles = get_cycles(results, calibration)
    results_noop = read_cachegrind(fname_noop)
    cycles_noop = get_cycles(results_noop, calibration)
    fft_cycles = (cycles-cycles_noop)
    print(f'{fftlen}: {fft_cycles}')


