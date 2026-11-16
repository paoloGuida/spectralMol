# Ablation Summary

Config directory: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix/dask_rapids_v100
Run directory: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix/dask_rapids_v100/compare_GuacaMol_20260506_121020

## Overall

| Condition | Model | Mean Best Score | Delta vs Full | Mean Elapsed (s) | Speedup vs Full | Evals/s | n_failed |
|---|---|---:|---:|---:|---:|---:|---:|
| full-spectrum | local_evolution | 0.752242 | 0.000000 | 1316.112519 | 1.000000x | 15.196269 | 0 |

## Artifacts

- Overall TSV: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix/dask_rapids_v100/ablation_summary_smoke_overall_20260515_151424.tsv
- Task delta TSV: /ibex/user/barradd/molevoDrugDiscovery/molscore/outputs/comparison_matrix/dask_rapids_v100/ablation_summary_smoke_task_delta_20260515_151424.tsv

## Notes

- Full-spectrum baseline is selected by condition label full-spectrum.
- If full-spectrum is missing, deltas and speedup are computed against the first sorted row.
