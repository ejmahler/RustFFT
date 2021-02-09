# Updating the estimates for the scalar planner

The scalar planner selects the best algorithm by planning several possible solutions, and then estimating the "cost" (proportional to execution time) for each one. The one with the lowest cost is chosen.

This relies on having good estimates for the time the different alorithms take to process a certain length. Whenever an algorithm is improved, the estimates should be updated.

The scripts normalize all measured times to the time of the 31-point butterfly. This is done to make it easier to compare bench results from different machines. 

## Preparing the computer
The bench results are affected by other processes competing for the CPU. Keep as little as possible running while the benches are running, In particular web browsers should be closed.

Also disable any turbo feature of the CPU. This will make the benches take a little longer to run, but the CPU speed will be more constant which improves the results.

## Run the benches
The bench results almost always contain some points that are quite off. To mitigate this, the benches shouldbe repeated several times. Using 5 repeats seems to give good results.

Run the benches with the `run_benches.sh` script:
```sh
tools/estimates/run_benches.sh mybench 5
```

This will repeat 5 times, and create the files "mybench_1.txt" to "mybench_5.txt".

## Fitting the data
 
The fitting scripts require numpy and scipy.

### Butterflies

Start with fitting the butterflies:
```sh
python tools/estimates/fit_butterflies.py mybench_1.txt mybench_2.txt mybench_3.txt mybench_4.txt mybench_5.txt
```

This fits the Butterflies, for when run once and multiple times (used when they are used as inners in the mixed radix algorithms).

It will plot each set of points togehter with the fit in a figure. Check each one for outliers. If there are outliers, try excluding one data file at a time, until they all show good fits. If this is not possible, the benches must be run again.

Once the fits look good, copy the generated code (printed in the terminal) into `src/scalar_planner_estimates.rs`.

### Radix 4
Next run the fitting for Radix4:
```sh
python tools/estimates/fit_radix4.py mybench_1.txt mybench_2.txt mybench_3.txt mybench_4.txt mybench_5.txt
```

Check that the fit is good, and then copy the generated code into `src/scalar_planner_estimates.rs`.

### Mixed radix
The MixedRadix, MixedRadixSmall, GoodTHomas and GoodThomasSmall are all fitted by the same script.

```sh
python tools/estimates/fit_mixed_radixes.py mybench_1.txt mybench_2.txt mybench_3.txt mybench_4.txt mybench_5.txt
```

This fits the "Small" variations up to a length of 32*32, and the normal algorithms above that.
The fit is done by subtracting the measured time for performing the inner ffts. Because of this, the results tends to be noisier than for Radix4. If it doesn't look good, try excluding files. Once it's good, copy the generated code into `src/scalar_planner_estimates.rs`.

### Bluesteins and Raders
Run the fitting script:

```sh
python tools/estimates/fit_bluesteins.py mybench_1.txt mybench_2.txt mybench_3.txt mybench_4.txt mybench_5.txt
```
and
```sh
python tools/estimates/fit_raders.py mybench_1.txt mybench_2.txt mybench_3.txt mybench_4.txt mybench_5.txt
```

The fits are done by subtracting the measured time for performing the inner fft. Because of this, the results tends to be noisier than for Radix4. If it doesn't look good, try excluding files. Once it's good, copy the generated code into `src/scalar_planner_estimates.rs`.

## Testing

Some of the tests in the planner check that the planned fft is the expected one. After updating the estimates, some of these might fail and need modifications.
