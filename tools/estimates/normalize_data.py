import sys
import numpy as np

# Normalize all bench results so that a 31-point butterfly costs 100.0
NAME = "bench_planned_multi_31"
REPS = [1,2,3,5,8,13,21,34,55,89]


def normalize(data):
    data = dict(data)
    values = []
    for nbr in REPS:
        benchname = f"{NAME}_{nbr}"
        val = data[benchname]
        values.append(val)
    # y = k*x + m
    slope, const = np.polyfit(REPS, values, 1)
    print(f"Fit result: time in ns = {slope} * len + {const}")
    factor = 100.0/slope
    print(f"Normalizing by {factor}")


    for key in data.keys():
        data[key] = factor*data[key]
    
    return data

