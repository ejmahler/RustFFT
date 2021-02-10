import sys
from matplotlib import pyplot as plt
import numpy as np
from read_bench_series import read_benches
from normalize_data import normalize

fnames = sys.argv[1:]
data = read_benches(fnames)
data = normalize(data)

length = [73, 179, 283, 419, 547, 661, 811, 947, 1087, 1229]
rader_cost = []
inner_cost = []
overhead_cost = []
for fftlen in length:
    benchname = f"bench_raders_{fftlen}"
    cost_fft = data[benchname]
    rader_cost.append(cost_fft)

    # Inners
    benchname = f"bench_planned_{fftlen-1}"
    cost_inner = data[benchname]
    inner_cost.append(cost_inner)

    overhead_cost.append(cost_fft - 2*cost_inner)

# y = k*x + m
k, m = np.polyfit(length, overhead_cost, 1)
print(f"slope: {k}, const: {m}")
print("")

print("--- Paste in scalar_planner_estimates.rs ---")
print("// --- Begin code generated by tools/estimates/fit_raders.py --- \n")
print(f"const RADERS_CONST: f32 = {m:.5f};")
print(f"const RADERS_SLOPE: f32 = {k:.5f};")
print("")
print(f"pub fn estimate_raders_cost(inner_fft: &Arc<Recipe>, repeats: usize) -> f32 {{")
print(f"    (RADERS_CONST")
print(f"        + RADERS_SLOPE * (inner_fft.len() + 1) as f32")
print(f"        + 2.0 * inner_fft.cost(1))")
print(f"        * repeats as f32")
print("}")
print("")
print("// --- End code generated by tools/estimates/fit_raders.py --- \n")

plt.figure(400)
plt.plot(length, rader_cost, '*')
plt.plot(length, inner_cost, '*')
plt.title("Raders")
plt.xlabel("Length")
plt.ylabel("Cost")
#plt.plot(plot_nbr, k*plot_nbr + m)
plt.figure(401)
plt.plot(length, overhead_cost, '*')
plt.title("Raders overhead")
plt.xlabel("Length")
plt.ylabel("Cost")
plt.show()
