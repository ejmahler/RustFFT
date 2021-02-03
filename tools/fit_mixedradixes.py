import sys
from matplotlib import pyplot as plt
import numpy as np
from read_cachegrind_benches import  read_cachegrind, get_cycles
from scipy.optimize import minimize

inners = {
    2: { "slope": 12.999999999999995, "const": 45.0},
    3: { "slope": 27.999999999999993, "const": 83.00000000000003},
    4: { "slope": 50.466666666666654, "const": 7.866666666666645},
    5: { "slope": 82.99999999999997, "const": 94.00000000000016},
    6: { "slope": 75.66666666666667, "const": 53.666666666666494},
    7: { "slope": 166.99999999999994, "const": 94.88888888888856},
    8: { "slope": 181.33333333333331, "const": 52.44444444444443},
    11: { "slope": 397.99999999999994, "const": 93.9999999999995},
    13: { "slope": 569.6333333333332, "const": 13.53333333333295},
    16: { "slope": 688.0, "const": 93.99999999999982},
    17: { "slope": 1011.8, "const": 26.755555555555645},
    19: { "slope": 1244.9999999999998, "const": 93.99999999999893},
    23: { "slope": 1864.8, "const": 26.755555555551666},
    29: { "slope": 3805.7333333333327, "const": 95.15555555555517},
    31: { "slope": 4484.466666666666, "const": 28.311111111111178},
    32: { "slope": 1811.8666666666666, "const": 95.68888888888473},
    127: { "slope": 64279.0, "const": 0.0 },
    233: { "slope": 107777.0, "const": 0.0 }, 
    1031: { "slope": 959002.0, "const": 0.0 }, 
    2003: { "slope": 1140184.0, "const": 0.0 }, 
}




calibfile = "target/iai/cachegrind.out.iai_calibration"
calibration = read_cachegrind(calibfile)

def rms_rel_diff(array1, array2):
   return np.sqrt(np.mean(((array1 - array2)/array1) ** 2))

mixedradixes_lengths = [(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31), (31, 127), (31, 233), (127, 233), (127, 1031), (1031, 2003)]
mixedradixes_large_lengths = [(31, 127), (31, 233), (127, 233), (127, 1031), (1031, 2003)]
mixedradixes_small_lengths = [(3, 4), (3, 5), (3, 7), (3,13), (3,31), (7, 31), (23, 31), (29,31)]
mixedradixes_cycles = {}
mixedradixes_len = {}
mix_algos = ["mixedradix", "mixedradixsmall", "goodthomas", "goodthomassmall"]
mixedradixes_inner_cycles = []
for mix_lens in mixedradixes_lengths:
    len_a, len_b = mix_lens
    mixedradixes_inner_cycles.append(len_a*inners[len_b]["slope"] + inners[len_b]["const"] 
                                   + len_b*inners[len_a]["slope"] + inners[len_a]["const"]) 

for mr in mix_algos:
    mixedradixes_cycles[mr] = []
    mixedradixes_len[mr] = []
    if mr.endswith("small"):
        lens = mixedradixes_small_lengths
    else:
        lens = mixedradixes_large_lengths
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
    plt.loglog(mixedradixes_len[mr], mr_overhead,'*')

    f = lambda x: rms_rel_diff( np.array(mr_overhead), x[0] + x[1]*np.array(mixedradixes_len[mr])**x[2])
    x0 = [0, 1, 1]
    res = minimize(f, x0)
    #k, m = np.polyfit(mixedradixes_len[mr], mr_overhead, 1)
    fit = res.x[1]*np.array(mixedradixes_len[mr])**res.x[2] + res.x[0]
    plt.loglog(mixedradixes_len[mr], fit)

    print(f"{mr}: slope: {res.x[1]}, const: {res.x[0]}, exponent: {res.x[2]}")

#mixedradix_large_len = mixedradixes_len["mixedradix"][len(mixedradixes_cycles["mixedradixsmall"]):]
#mixedradix_large = mixedradixes_cycles["mixedradix"][len(mixedradixes_cycles["mixedradixsmall"]):]
#goodthomas_large = mixedradixes_cycles["goodthomas"][len(mixedradixes_cycles["goodthomassmall"]):]
#mrh_overhead = np.array(mixedradix_large) - np.array(mixedradixes_inner_cycles[len(mixedradixes_cycles["mixedradixsmall"]):])
#gth_overhead = np.array(goodthomas_large) - np.array(mixedradixes_inner_cycles[len(mixedradixes_cycles["mixedradixsmall"]):])
#
#plt.figure(12)
#plt.loglog(mixedradix_large_len, mrh_overhead, '*')
#plt.loglog(mixedradix_large_len, gth_overhead, '*')
#k, m = np.polyfit(mixedradix_large_len, mrh_overhead, 1)
#print(f"mixedradix_hybrid: slope: {k}, const: {m}")
#plt.loglog(mixedradix_large_len, np.array(mixedradix_large_len)*k + m)
#k, m = np.polyfit(mixedradix_large_len, gth_overhead, 1)
#print(f"goodthomas_hybrid: slope: {k}, const: {m}")
#plt.plot(mixedradix_large_len, np.array(mixedradix_large_len)*k + m)


#plt.figure(10)
#plt.loglog(mixedradixes_len["mixedradix"], mixedradixes_inner_cycles)
#plt.legend(mix_algos)
plt.figure(11)
plt.legend(mix_algos)
plt.show()

