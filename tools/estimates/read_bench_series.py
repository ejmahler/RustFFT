import sys
import statistics

def read_benches(names):
    results = {}
    if not isinstance(names, (list,)):
        names = [names,]
    for name in names:    
        with open(name) as f:
            results = {}
            for line in f:
                if line.startswith("test ") and not line.startswith("test result:"):
                    lineval = line.split("... bench:")
                    name = lineval[0].split()[1]
                    value = lineval[1].split()[0].replace(",", "")
                    if name in results:
                        results[name].append(float(value))
                    else:
                        results[name] = [float(value),]
    for k in results:
        results[k] = statistics.median(results[k])
    return results




if __name__ == "__main__":
    file = sys.argv[1]
    print(read_benches(file))