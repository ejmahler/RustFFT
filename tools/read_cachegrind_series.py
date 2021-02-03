import sys
from matplotlib import pyplot as plt

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

    # Use formula from iai
    # https://pythonspeed.com/articles/consistent-benchmarking-in-ci/
    estimated_cycles = l1_hits + (5 * l3_hits) + (35 * ram_hits)

    #print(f"Instructions: {instruction_reads}")
    #print(f"L1 Accesses: {l1_hits}")
    #print(f"L2 Accesses: {l3_hits}")
    #print(f"RAM Accesses: {ram_hits}")
    #print(f"Estimated Cycles: {estimated_cycles}")

    return estimated_cycles

    #  butterfly_noop____00000032_00000000
    #Instructions:              157357
    #L1 Accesses:               270920
    #L2 Accesses:                  237
    #RAM Accesses:               16232
    #Estimated Cycles:          840225

if __name__ == "__main__":
    calibfile = "target/iai/cachegrind.out.iai_calibration"
    calibration = read_cachegrind(calibfile)
    planned_length = [2, 3, 4, 5, 6, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32, 127, 233]
    planned_cycles = []
    planned_cycles_dict = {}
    for fftlen in planned_length:
        fname = f"target/iai/cachegrind.out.bench_planned_multi_{fftlen}_100"
        fname_noop = f"target/iai/cachegrind.out.bench_planned_multi_setup_{fftlen}_100"
        results = read_cachegrind(fname)
        cycles = get_cycles(results, calibration)
        results_noop = read_cachegrind(fname_noop)
        cycles_noop = get_cycles(results_noop, calibration)
        estimated_cycles = (cycles-cycles_noop)/100.0
        print(f"{fftlen} {estimated_cycles}")
        planned_cycles.append(estimated_cycles)
        planned_cycles_dict[fftlen] = estimated_cycles

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
        print(f"{fftlen} {estimated_cycles}")
        radix4_cycles.append(estimated_cycles)

    mixedradix_rx4_inner_length = [32, 64, 128, 256, 512, 1024, 2048]
    mixedradix_rx4_cycles = []
    mixedradix_rx4_length = []
    for fftlen in mixedradix_rx4_inner_length:
        fname = f"target/iai/cachegrind.out.bench_mixedradix_rx4_{fftlen}"
        fname_noop = f"target/iai/cachegrind.out.bench_mixedradix_rx4_setup_{fftlen}"
        results = read_cachegrind(fname)
        cycles = get_cycles(results, calibration)
        results_noop = read_cachegrind(fname_noop)
        cycles_noop = get_cycles(results_noop, calibration)
        estimated_cycles = cycles-cycles_noop
        print(f"{fftlen} {estimated_cycles}")
        mixedradix_rx4_cycles.append(estimated_cycles)
        mixedradix_rx4_length.append(fftlen**2)

    mixedradixes_lengths = [(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233)]
    mixedradixes_small_lengths = [(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)]
    mixedradixes_cycles = {}
    mixedradixes_len = {}
    mix_algos = ["mixedradix", "mixedradixsmall", "goodthomas", "goodthomassmall"]
    mixedradixes_inner_cycles = []
    for mix_lens in mixedradixes_lengths:
        len_a, len_b = mix_lens
        mixedradixes_inner_cycles.append(len_a*planned_cycles_dict[len_b] + len_b*planned_cycles_dict[len_a])

    for mr in mix_algos:
        mixedradixes_cycles[mr] = []
        mixedradixes_len[mr] = []
        if mr.endswith("small"):
            lens = mixedradixes_small_lengths
        else:
            lens = mixedradixes_lengths
        for fftlens in lens:
            len_a, len_b = fftlens
            fname = f"target/iai/cachegrind.out.bench_{mr}_{len_a}_{len_b}"
            fname_noop = f"target/iai/cachegrind.out.bench_{mr}_setup_{len_a}_{len_b}"
            results = read_cachegrind(fname)
            cycles = get_cycles(results, calibration)
            results_noop = read_cachegrind(fname_noop)
            cycles_noop = get_cycles(results_noop, calibration)
            estimated_cycles = cycles-cycles_noop
            print(f"{fftlen} {estimated_cycles}")
            mixedradixes_cycles[mr].append(estimated_cycles)
            mixedradixes_len[mr].append(len_a*len_b)
        plt.figure(10)
        plt.loglog(mixedradixes_len[mr], mixedradixes_cycles[mr])
    plt.loglog(mixedradixes_len["mixedradix"], mixedradixes_inner_cycles)
    plt.legend(mix_algos)
    

    plt.figure(20)
    plt.loglog(planned_length, planned_cycles, '*')

    plt.figure(30)
    plt.loglog(radix4_length, radix4_cycles, '*')
    plt.loglog(mixedradix_rx4_length, mixedradix_rx4_cycles, '*')
    plt.show()

