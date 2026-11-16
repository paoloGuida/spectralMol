# Matrix Wall-Clock and Throughput Summary

Base directory: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix

| Config | Run Dir | Mean elapsed (s) | Campaign wall-clock (s) | Evals/s (mean) | Evals/s (campaign) | Speedup mean | Speedup campaign | n_failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_cpu | compare_GuacaMol_20260506_120038 | 1541.237239 | 9440.000000 | 12.976588 | 12.711864 | 1.000000x | 1.000000x | 0 |
| dask_cpu | compare_GuacaMol_20260506_120224 | 1463.334877 | 1510.000000 | 13.667412 | 79.470199 | 1.053236x | 6.251656x | 0 |
| rapids_v100 | compare_GuacaMol_20260506_120813 | 1349.282909 | 1398.000000 | 14.822688 | 85.836910 | 1.142264x | 6.752504x | 0 |
| dask_rapids_v100 | compare_GuacaMol_20260506_121020 | 1316.112519 | 1359.000000 | 15.196269 | 88.300221 | 1.171053x | 6.946284x | 0 |

Notes:
- Evals/s (mean) = budget_effective / mean_elapsed_seconds
- Evals/s (campaign) = total_evaluations / campaign_wallclock_seconds
- total_evaluations = budget_effective * n_runs
