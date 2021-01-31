import sys
from matplotlib import pyplot as plt
import numpy as np

butterflies = {
    2: {"slope": 12.963636363636361, "const": 40.19999999999995},
    3: {"slope": 28.072727272727274, "const": 40.19999999999989},
    4: {"slope": 50.830303030303014, "const": 39.333333333333165},
    5: {"slope": 81.83030303030301, "const": 56.53333333333306},
    6: {"slope": 71.0969696969697, "const": 55.46666666666642},
    7: {"slope": 165.99999999999997, "const": 51.999999999999446},
    8: {"slope": 175.9757575757575, "const": 58.73333333333292},
    11: {"slope": 396.99999999999994, "const": 51.99999999999805},
    13: {"slope": 565.3636363636363, "const": 85.19999999999811},
    16: {"slope": 686.9999999999999, "const": 47.99999999999699},
    17: {"slope": 1010.781818181818, "const": 51.199999999997985},
    19: {"slope": 1243.7575757575758, "const": 52.53333333332956},
    23: {"slope": 1864.2424242424242, "const": 51.46666666666053},
    29: {"slope": 3804.999999999999, "const": 51.9999999999909},
    31: {"slope": 4484.072727272727, "const": 47.199999999986126},
    32: {"slope": 1811.024242424242, "const": 55.46666666666247},
    127: {"slope": 84539.64242424241, "const": -443.7333333336869},
    233: {"slope": 176362.83636363634, "const": -48.60000000059032},
}

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


    mixedradixes_lengths = [(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233)]
    mixedradixes_small_lengths = [(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)]
    mixedradixes_cycles = {}
    mixedradixes_len = {}
    mix_algos = ["mixedradix", "mixedradixsmall", "goodthomas", "goodthomassmall"]
    mixedradixes_inner_cycles = []
    for mix_lens in mixedradixes_lengths:
        len_a, len_b = mix_lens
        mixedradixes_inner_cycles.append(len_a*butterflies[len_b]["slope"] + butterflies[len_b]["const"] 
                                       + len_b*butterflies[len_a]["slope"] + butterflies[len_a]["const"]) 

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
            #print(f"{fftlen} {estimated_cycles}")
            mixedradixes_cycles[mr].append(estimated_cycles)
            mixedradixes_len[mr].append(len_a*len_b)
        plt.figure(10)
        plt.loglog(mixedradixes_len[mr], mixedradixes_cycles[mr])
        plt.figure(11)
        mr_overhead = np.array(mixedradixes_cycles[mr])-np.array(mixedradixes_inner_cycles[0:len(mixedradixes_len[mr])])
        plt.plot(mixedradixes_len[mr], mr_overhead)
        k, m = np.polyfit(mixedradixes_len[mr], mr_overhead, 1)
        print(f"{mr}: slope: {k}, const: {m}")

    mixedradix_large_len = mixedradixes_len["mixedradix"][len(mixedradixes_cycles["mixedradixsmall"]):]
    mixedradix_large = mixedradixes_cycles["mixedradix"][len(mixedradixes_cycles["mixedradixsmall"]):]
    goodthomas_large = mixedradixes_cycles["goodthomas"][len(mixedradixes_cycles["goodthomassmall"]):]
    mrh_overhead = np.array(mixedradix_large) - np.array(mixedradixes_inner_cycles[len(mixedradixes_cycles["mixedradixsmall"]):])
    gth_overhead = np.array(goodthomas_large) - np.array(mixedradixes_inner_cycles[len(mixedradixes_cycles["mixedradixsmall"]):])

    plt.figure(12)
    plt.plot(mixedradix_large_len, mrh_overhead, '*')
    plt.plot(mixedradix_large_len, gth_overhead, '*')
    k, m = np.polyfit(mixedradix_large_len, mrh_overhead, 1)
    print(f"mixedradix_hybrid: slope: {k}, const: {m}")
    plt.plot(mixedradix_large_len, np.array(mixedradix_large_len)*k + m)
    k, m = np.polyfit(mixedradix_large_len, gth_overhead, 1)
    print(f"goodthomas_hybrid: slope: {k}, const: {m}")
    plt.plot(mixedradix_large_len, np.array(mixedradix_large_len)*k + m)



    plt.figure(10)
    plt.loglog(mixedradixes_len["mixedradix"], mixedradixes_inner_cycles)
    plt.legend(mix_algos)
    plt.figure(11)
    plt.legend(mix_algos)

    

    #plt.figure(20)
    #plt.loglog(planned_length, planned_cycles, '*')
#
    #plt.figure(30)
    #plt.loglog(radix4_length, radix4_cycles, '*')
    #plt.loglog(mixedradix_rx4_length, mixedradix_rx4_cycles, '*')
    plt.show()

