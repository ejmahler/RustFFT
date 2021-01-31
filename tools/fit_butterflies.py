import sys
from matplotlib import pyplot as plt
import numpy as np

def read_cachegrind(name):
    with open(name) as f:
        events = None
        summary = None
        for line in f:
            if line.startswith("events:"):
                events = line
            elif line.startswith("summary:"):
                summary = line
            if events and summary:
                break
        #print(events)
        #print(summary)
        events = events.split()[1:]
        summary = summary.split()[1:]

        results = {}
        for (label, value) in zip(events, summary):
            results[label] = float(value)
    return results


def get_cycles(data, calib):
    instruction_reads = data["Ir"] - calib["Ir"]
    instruction_l1_misses = data["I1mr"] - calib["I1mr"]
    instruction_cache_misses = data["ILmr"] - calib["ILmr"]
    data_reads = data["Dr"] - calib["Dr"]
    data_l1_read_misses = data["D1mr"] - calib["D1mr"]
    data_cache_read_misses = data["DLmr"] - calib["DLmr"]
    data_writes = data["Dw"] - calib["Dw"]
    data_l1_write_misses = data["D1mw"] - calib["D1mw"]
    data_cache_write_misses = data["DLmw"] - calib["DLmw"]

    ram_hits = instruction_cache_misses + data_cache_read_misses + data_cache_write_misses
    l3_accesses = instruction_l1_misses + data_l1_read_misses + data_l1_write_misses
    l3_hits = l3_accesses - ram_hits
    total_memory_rw = instruction_reads + data_reads + data_writes
    l1_hits = total_memory_rw - (ram_hits + l3_hits)
    estimated_cycles = l1_hits + (5 * l3_hits) + (35 * ram_hits)
    return estimated_cycles


if __name__ == "__main__":
    calibfile = "target/iai/cachegrind.out.iai_calibration"
    calibration = read_cachegrind(calibfile)
    length = [2, 3, 4, 5, 6, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32, 127, 233]
    multi_nbr = np.arange(1,11)
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
        print(f"{fftlen}: slope: {k}, const: {m}")
        all_cycles.append(estimated_cycles)
        plt.figure(fftlen)
        plt.plot(multi_nbr, estimated_cycles, '*')
        plt.plot(multi_nbr, k*multi_nbr + m)

    plt.show()

