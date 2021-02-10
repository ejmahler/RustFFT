import sys
from matplotlib import pyplot as plt
from read_bench_series import read_benches
from scipy.optimize import minimize
import numpy as np
from normalize_data import normalize

fnames = sys.argv[1:]
data = read_benches(fnames)
data = normalize(data)

radix4_length = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
radix4_cost = []
for fftlen in radix4_length:
    benchname = f"bench_radix4_{fftlen}"
    cost = data[benchname]
    radix4_cost.append(cost)

def rms_rel_diff(array1, array2):
   return np.sqrt(np.mean(((array1 - array2)/array1) ** 2))

np_len = np.array(radix4_length)
np_cost = np.array(radix4_cost)
f = lambda x: rms_rel_diff( np_cost, x[0] + x[1]*np_len**x[2])
x0 = [0, 1, 1]
res = minimize(f, x0)

print(f"Radix 4, const: {res.x[0]}, slope: {res.x[1]}, exponent {res.x[2]}")
cost_fit = res.x[0] + res.x[1]*np_len**res.x[2]


print("--- Paste in scalar_planner_estimates.rs ---")
print("// --- Begin code generated by tools/estimates/fit_radix4.py --- \n")
print(f"const RADIX4_CONST: f32 = {res.x[0]:.5f};")
print(f"const RADIX4_SLOPE: f32 = {res.x[1]:.5f};")
print(f"const RADIX4_EXP: f32 = {res.x[2]:.5f};")
print("")
print(f"pub fn estimate_radix4_cost(len: usize, repeats: usize) -> f32 {{")
print(f"    (RADIX4_CONST + RADIX4_SLOPE * (len as f32).powf(RADIX4_EXP)) * repeats as f32")
print("}")
print("")
print("// --- End code generated by tools/estimates/fit_radix4.py --- \n")

plt.figure(100)
plt.loglog(radix4_length, radix4_cost, '*')
plt.loglog(np_len, cost_fit)
plt.title("Radix 4")
plt.xlabel("Length")
plt.ylabel("Cost")

plt.figure(101)
plt.semilogx(np_len, cost_fit/np_cost)
plt.title("Radix 4, fit/measured")
plt.xlabel("Length")
plt.ylabel("Cost")
plt.show()
