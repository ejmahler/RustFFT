#!/bin/bash
for (( i = 1; i <= $2; ++i ));
do
    cargo bench --bench bench_like_iai | tee "$1_${i}.txt"
done