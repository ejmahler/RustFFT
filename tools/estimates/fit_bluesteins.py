import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from read_bench_series import read_benches
from normalize_data import normalize

fname = sys.argv[1]
data = read_benches(fname)
data = normalize(data)
length = [10, 30, 50, 70, 90]

inner_length = [128, 256, 512, 1024, 2048]
bluestein_sweeplen_cost = []
bluestein_sweepinner_cost = []
inner_cost = []
overhead_cost = []

for bslen in length:
    benchname = f"bench_bluesteins_{bslen}_512"
    fft_cost = data[benchname]
    bluestein_sweeplen_cost.append(fft_cost)

for inlen in inner_length:
    benchname = f"bench_bluesteins_50_{inlen}"
    fft_cost = data[benchname]
    bluestein_sweepinner_cost.append(fft_cost)

    # Inners
    benchname = f"bench_planned_{inlen}"
    fft_cost_inner = data[benchname]
    inner_cost.append(2*fft_cost_inner)

    #overhead_cost.append(fft_cost- 2*fft_cost_inner)

# y = k*x + m
#k, m = np.polyfit(multi_nbr, estimated_cost, 1)

diff_len = np.array(bluestein_sweeplen_cost) - inner_cost[2]
diff_inner = np.array(bluestein_sweepinner_cost) - np.array(inner_cost)

def rms_rel_diff(array1, array2, array3, array4):
   return np.sqrt(np.mean(((array1 - array2)/array1) ** 2)) + np.sqrt(np.mean(((array3 - array4)/array3) ** 2))

inner_len_array = np.array(inner_length)
len_array = np.array(length)

f = lambda x: rms_rel_diff( diff_len, x[0] + x[1]*len_array + x[2]*inner_len_array[2], diff_inner, x[0] + x[1]*len_array[2] + x[2]*inner_len_array)
x0 = [0, 1, 1]
res = minimize(f, x0)
print(res.x)
print(f"Overhead: const: {res.x[0]}, slope_len: {res.x[1]}, slope_inner_len: {res.x[2]}")

print("")
print("--- Paste in scalar_planner_estimates.rs ---")
print("// --- Begin code generated by tools/estimates/fit_bluesteins.py --- \n")
print(f"const BLUESTEINS_CONST: f32 = {res.x[0]:.5f};")
print(f"const BLUESTEINS_LEN_SLOPE: f32 = {res.x[1]:.5f};")
print(f"const BLUESTEINS_INNER_LEN_SLOPE: f32 = {res.x[2]:.5f};")
print("")
print(f"pub fn estimate_bluesteins_cost(len: usize, inner_fft: &Arc<Recipe>, repeats: usize) -> f32 {{")
print(f"    (BLUESTEINS_CONST")
print(f"        + BLUESTEINS_INNER_LEN_SLOPE * inner_fft.len() as f32")
print(f"        + BLUESTEINS_LEN_SLOPE * len as f32")
print(f"        + 2.0 * inner_fft.cost(1))")
print(f"        * repeats as f32")
print("}")
print("")
print("// --- End code generated by tools/estimates/fit_bluesteins.py --- \n")

plt.figure(300)
plt.plot(length, bluestein_sweeplen_cost, '*')
plt.title("Bluesteins, sweep length")
plt.xlabel("Length")
plt.ylabel("Cost")
plt.figure(301)
plt.plot(inner_length, bluestein_sweepinner_cost, '*')
plt.title("Bluesteins, sweep inner length")
plt.xlabel("Inner length")
plt.ylabel("Cost")

plt.figure(302)
plt.plot(length, diff_len, '*')
plt.title("Bluesteins overhead, sweep length")
plt.xlabel("Length")
plt.ylabel("Cost")
plt.figure(303)
plt.plot(inner_length, diff_inner, '*')
plt.title("Bluesteins overhead, sweep inner length")
plt.xlabel("Inner length")
plt.ylabel("Cost")

plt.show()
