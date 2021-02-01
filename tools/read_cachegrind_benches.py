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
    return estimated_cycles
