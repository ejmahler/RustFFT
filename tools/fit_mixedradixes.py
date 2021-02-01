import sys
from matplotlib import pyplot as plt
import numpy as np
from read_cachegrind_benches import  read_cachegrind, get_cycles

butterflies = {
    2: { "slope": 13.157575757575755, "const": -1.8666666666667207},
    3: { "slope": 27.927272727272726, "const": 72.79999999999997},
    4: { "slope": 50.75757575757575, "const": -2.4666666666668067},
    5: { "slope": 82.07272727272725, "const": 82.99999999999973},
    6: { "slope": 71.0, "const": 18.999999999999854},
    7: { "slope": 165.99999999999994, "const": 82.99999999999966},
    8: { "slope": 175.99999999999994, "const": 21.999999999999446},
    11: { "slope": 396.99999999999994, "const": 82.99999999999869},
    13: { "slope": 571.2666666666667, "const": -0.8666666666684496},
    16: { "slope": 686.9999999999999, "const": 82.99999999999766},
    17: { "slope": 1011.2181818181817, "const": 13.399999999996984},
    19: { "slope": 1243.9999999999998, "const": 82.99999999999498},
    23: { "slope": 1864.2424242424238, "const": 12.86666666666205},
    29: { "slope": 3804.8303030303023, "const": 84.33333333332524},
    31: { "slope": 4483.951515151514, "const": 14.46666666665442},
    32: { "slope": 1811.1696969696968, "const": 82.46666666666242},
    127: { "slope": 84536.67878787877, "const": -313.3333333336628},
    233: { "slope": 176355.97575757574, "const": 172.13333333283902},
}



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
plt.show()

