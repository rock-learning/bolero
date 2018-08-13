# Comparing numerical and analytical gradients for REPS and C-REPS

The two benchmark scripts *reps_benchmark.py* and *creps_benchmark.py* compare the performance of using analytical gradients rather than numerical gradients when minimizing the dual function.

For both REPS and C-REPS the analytical gradient implementation leads to:
 * Better runtime performance.
 * Similar stability.
 * In some occasions very slightly improved solutions.


 ## Benchmarks results:

 ```
 $ python reps_benchmark.py
Numerical gradient completed in average time of 0.69 seconds
Numerical gradient minimum found 2252.77131734028
Analytical gradient completed in average time of 0.62 seconds
Analytical gradient minimum found 2252.7695580270843
 ```

![alt text](reps_benchmark_result.png)

```
$ python creps_benchmark.py
...
Algorithm C-REPS-NUM and objective function sphere took 9.89173007011 seconds to complete
...
Algorithm C-REPS-AN and objective function sphere took 7.18385887146 seconds to complete
```

![alt text](creps_benchmark_result.png)