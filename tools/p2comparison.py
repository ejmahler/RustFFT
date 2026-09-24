import sys
import math
from matplotlib import pyplot as plt


with open(sys.argv[1]) as f:
    lines = f.readlines()

raw_results = {}
for line in lines:
    if line.startswith("test ") and not line.startswith("test result"):
        name, result = line.split("... bench:")
        name = name.split()[1]
        value = float(result.strip().split(" ")[0].replace(",", ""))
        group, algo, length = name.split("/")

        result_list = raw_results.setdefault(int(length), [])
        result_list.append((name, value))

lengths = sorted(list(raw_results.keys()))

processed_results = {}
def get_process_list(name):
    group, algo, length = name.split("/")
    _, arch, ftype = group.split("_")

    key = f"{ftype}_{arch}_{algo}"
    return processed_results.setdefault(key, [])

for l in lengths:
    for (name, value) in raw_results[l]:
        processed_list = get_process_list(name)
        processed_list.append(math.log(value, 2))
        
lengths_log2 = [math.log(l, 2) for l in lengths]

plt.figure()
plt.plot(
    lengths_log2, processed_results["f64_scalar_radixnbase"],
    lengths_log2, processed_results["f64_sse_mixedradix"],
    lengths_log2, processed_results["f64_sse_radixnbase"],
    lengths_log2, processed_results["f64_sse_radixncross"],
)
plt.title("f64, SSE")
plt.ylabel("log2(computation time)")
plt.xlabel("log2(length)")
plt.legend(["scalar_radixnbase", "sse_mixedradix", "sse_radixnbase", "sse_radixncross"])
plt.grid()

plt.figure()
plt.plot(
    lengths_log2, processed_results["f32_scalar_radixnbase"],
    lengths_log2, processed_results["f32_sse_mixedradix"],
    lengths_log2, processed_results["f32_sse_radixncross"],
)
plt.title("f32, SSE")
plt.ylabel("log2(computation time)")
plt.xlabel("log2(length)")
plt.legend(["scalar_radixnbase", "sse_mixedradix", "sse_radixncross"])
plt.grid()

plt.figure()
plt.plot(
    lengths_log2, processed_results["f64_scalar_mixedradix"],
    lengths_log2, processed_results["f64_scalar_radixnbase"],
    lengths_log2, processed_results["f64_scalar_radixncross"],
)
plt.title("f64, Scalar")
plt.ylabel("log2(computation time)")
plt.xlabel("log2(length)")
plt.legend(["scalar_mixedradix", "scalar_radixnbase", "scalar_radixncross"])
plt.grid()

plt.figure()
plt.plot(
    lengths_log2, processed_results["f32_scalar_mixedradix"],
    lengths_log2, processed_results["f32_scalar_radixnbase"],
    lengths_log2, processed_results["f32_scalar_radixncross"],
)
plt.title("f32, Scalar")
plt.ylabel("log2(computation time)")
plt.xlabel("log2(length)")
plt.legend(["scalar_mixedradix", "scalar_radixnbase", "scalar_radixncross"])
plt.grid()

plt.show()


