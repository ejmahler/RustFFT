import sys
from matplotlib import pyplot as plt
import numpy as np
from read_bench_series import read_benches
from scipy.optimize import minimize
from normalize_data import normalize

master_10k = read_benches(["D:\\Documents\\estimates_output\\master_10k.txt"])
master_100k = read_benches(["D:\\Documents\\estimates_output\\master_100k.txt"])
master_1000k = read_benches(["D:\\Documents\\estimates_output\\master_1000k.txt"])

estimated_10k = read_benches(["D:\\Documents\\estimates_output\\estimated_10k.txt"])
estimated_100k = read_benches(["D:\\Documents\\estimates_output\\estimated_100k.txt"])
estimated_1000k = read_benches(["D:\\Documents\\estimates_output\\estimated_1000k.txt"])

diffs_1000k = []
diffs_100k = []
diffs_10k = []

paired_diffs_1000k = []
paired_diffs_100k = []
paired_diffs_10k = []

avg_1000k = 0.0
avg_100k = 0.0
avg_10k = 0.0

for key in estimated_1000k:
    key_count = int(key.split('_')[-1]) + 1000000
    diff = float(estimated_1000k[key]) / float(master_1000k[key])
    diffs_1000k.append(diff)
    paired_diffs_1000k.append((diff, key_count))
    avg_1000k += diff / len(estimated_1000k)

for key in estimated_100k:
    key_count = int(key.split('_')[-1]) + 100000
    diff = float(estimated_100k[key]) / float(master_100k[key])
    diffs_100k.append(diff)
    paired_diffs_100k.append((diff, key_count))
    avg_100k += diff / len(estimated_100k)

for key in estimated_10k:
    key_count = int(key.split('_')[-1]) + 10000
    diff = float(estimated_10k[key]) / float(master_10k[key])
    diffs_10k.append(diff)
    paired_diffs_10k.append((diff, key_count))
    avg_10k += diff / len(estimated_10k)


diffs_1000k.sort()
diffs_100k.sort()
diffs_10k.sort()
paired_diffs_1000k.sort()
paired_diffs_100k.sort()
paired_diffs_10k.sort()

print(f"10k mean   = {avg_10k:.4f}")
print(f"10k best   = {diffs_10k[0]:.4f}")
print(f"10k 10%    = {diffs_10k[len(diffs_10k) // 10]:.4f}")
print(f"10k 25%    = {diffs_10k[len(diffs_10k) // 4]:.4f}")
print(f"10k median = {diffs_10k[len(diffs_10k) // 2]:.4f}")
print(f"10k 75%    = {diffs_10k[(len(diffs_10k) * 3) // 4]:.4f}")
print(f"10k 90%    = {diffs_10k[(len(diffs_10k) * 9) // 10]:.4f}")
print(f"10k worst  = {diffs_10k[-1]:.4f}")
print()
print(f"100k mean   = {avg_100k:.4f}")
print(f"100k best   = {diffs_100k[0]:.4f}")
print(f"100k 10%    = {diffs_100k[len(diffs_100k) // 10]:.4f}")
print(f"100k 25%    = {diffs_100k[len(diffs_100k) // 4]:.4f}")
print(f"100k median = {diffs_100k[len(diffs_100k) // 2]:.4f}")
print(f"100k 75%    = {diffs_100k[(len(diffs_100k) * 3) // 4]:.4f}")
print(f"100k 90%    = {diffs_100k[(len(diffs_100k) * 9) // 10]:.4f}")
print(f"100k worst  = {diffs_100k[-1]:.4f}")
print()
print(f"1000k mean   = {avg_1000k:.4f}")
print(f"1000k best   = {diffs_1000k[0]:.4f}")
print(f"1000k 10%    = {diffs_1000k[len(diffs_1000k) // 10]:.4f}")
print(f"1000k 25%    = {diffs_1000k[len(diffs_1000k) // 4]:.4f}")
print(f"1000k median = {diffs_1000k[len(diffs_1000k) // 2]:.4f}")
print(f"1000k 75%    = {diffs_1000k[(len(diffs_1000k) * 3) // 4]:.4f}")
print(f"1000k 90%    = {diffs_1000k[(len(diffs_1000k) * 9) // 10]:.4f}")
print(f"1000k worst  = {diffs_1000k[-1]:.4f}")

def factor(n):
    factors = []

    while n % 2 == 0:
        factors.append(2)
        n = n // 2
    
    divisor = 3
    while n > 1 and divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n = n // divisor
        divisor += 2

    if n > 1:
        factors.append(n)
    
    return factors

print()
print("10k biggest improvements")
for (diff, size) in paired_diffs_10k[:10]:
    print(f"{size}: {diff:.4f}, factors: {factor(size)}")
print()
print("10k biggest losses")
for (diff, size) in paired_diffs_10k[-10:]:
    print(f"{size}: {diff:.4f}, factors: {factor(size)}")
print()
print("100k biggest improvements")
for (diff, size) in paired_diffs_100k[:10]:
    print(f"{size}: {diff:.4f}, factors: {factor(size)}")
print()
print("100k biggest losses")
for (diff, size) in paired_diffs_100k[-10:]:
    print(f"{size}: {diff:.4f}, factors: {factor(size)}")
print()
print("1000k biggest improvements")
for (diff, size) in paired_diffs_1000k[:10]:
    print(f"{size}: {diff:.4f}, factors: {factor(size)}")
print()
print("1000k biggest losses")
for (diff, size) in paired_diffs_1000k[-10:]:
    print(f"{size}: {diff:.4f}, factors: {factor(size)}")
print()

print("let recipe_sizes = [")
for (diff, size) in paired_diffs_100k[-10:]:
    print(f"    {size},")
print("];")
