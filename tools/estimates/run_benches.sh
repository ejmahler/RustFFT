#!/bin/bash
for (( i = 1; i <= $2; ++i ));
do
    cargo bench --bench bench_for_estimates | tee "$1_${i}.txt"
done