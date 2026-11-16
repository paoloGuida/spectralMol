# Ablation Summary

Config directory: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container
Run directory: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container/compare_GuacaMol_20260515_224644

## Overall

| Condition | Model | Mean Best Score | Delta vs Full | Mean Elapsed (s) | Speedup vs Full | Evals/s | n_failed |
|---|---|---:|---:|---:|---:|---:|---:|
| high-only | local_evolution_high_only | 0.754234 | 0.005026 | 1331.263564 | 1.155757x | 15.023321 | 0 |
| full-spectrum | local_evolution | 0.749207 | 0.000000 | 1538.617772 | 1.000000x | 12.998680 | 0 |
| low-only | local_evolution_low_only | 0.736732 | -0.012475 | 1548.090109 | 0.993881x | 12.919145 | 0 |
| random-matrix | local_evolution_random_matrix | 0.646941 | -0.102267 | 1256.052051 | 1.224963x | 15.922907 | 0 |

## Artifacts

- Overall TSV: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container/ablation_summary_overall_20260516_041214.tsv
- Task delta TSV: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container/ablation_summary_task_delta_20260516_041214.tsv

## Notes

- Full-spectrum baseline is selected by condition label full-spectrum.
- If full-spectrum is missing, deltas and speedup are computed against the first sorted row.
