# GuacaMol Matrix Report (Round 2)

Date: 2026-05-06

## Run Scope
- Benchmark: GuacaMol comparison matrix
- Model: local_evolution
- Seeds: 7,8,9,10,11,12 (6 runs/config)
- Tasks: 20
- Budget: 20,000
- Generations: 30
- Population/Batch: 256/256

## SLURM Jobs
- 46909028: baseline_cpu (thread + pandas)
- 46909029: dask_cpu (dask + pandas)
- 46909030: rapids_v100 (thread + cuDF auto)
- 46909031: dask_rapids_v100 (dask + cuDF auto)

All jobs completed with exit code 0.

## Runtime Results

| Config | Mean elapsed seconds | Speedup vs baseline | Time reduction vs baseline |
|---|---:|---:|---:|
| baseline_cpu | 1541.2372 | 1.0000x | 0.00% |
| dask_cpu | 1463.3349 | 1.0532x | 5.05% |
| rapids_v100 | 1349.2829 | 1.1423x | 12.45% |
| dask_rapids_v100 | 1316.1125 | 1.1711x | 14.61% |

## Score/Quality Parity
Overall summary values are effectively identical across all 4 configurations:
- sum_task_best_score: 15.04483132505917 (floating-point noise only)
- mean_task_best_score: 0.7522415662529585 (floating-point noise only)
- sum_task_avg_score: 7.853590944819374 (floating-point noise only)
- mean_task_avg_score: 0.3926795472409687 (floating-point noise only)
- n_failed: 0 for all runs

Generation-level summary is also numerically aligned across configurations:
- mean_valid_rate remains 100.0 throughout sampled generations
- mean_score, mean_best_score_so_far, and mean_diversity tracks overlap (differences only at floating precision)

## Interpretation
- Dask alone gives a modest but real throughput gain (+5.05%).
- RAPIDS on V100 provides the larger single improvement (+12.45%).
- Dask+RAPIDS provides the best end-to-end runtime gain (+14.61%).
- Combined acceleration is sub-additive (expected), indicating overlapping bottlenecks where each method accelerates some of the same critical path.

## Recommendation
Adopt dask_rapids_v100 as the default matrix execution mode for production-scale compare runs, with fallback to rapids_v100 where scheduler constraints make Dask less convenient.

## Source Artifacts
- Consolidated speedup table:
  - molscore/outputs/comparison_matrix/speedup_matrix_20260506_round2.tsv
- Baseline:
  - molscore/outputs/comparison_matrix/baseline_cpu/compare_GuacaMol_20260506_120038/model_runtime_summary.tsv
  - molscore/outputs/comparison_matrix/baseline_cpu/compare_GuacaMol_20260506_120038/comparison_overall.tsv
  - molscore/outputs/comparison_matrix/baseline_cpu/compare_GuacaMol_20260506_120038/comparison_generation_model_summary.tsv
- Dask CPU:
  - molscore/outputs/comparison_matrix/dask_cpu/compare_GuacaMol_20260506_120224/model_runtime_summary.tsv
  - molscore/outputs/comparison_matrix/dask_cpu/compare_GuacaMol_20260506_120224/comparison_overall.tsv
  - molscore/outputs/comparison_matrix/dask_cpu/compare_GuacaMol_20260506_120224/comparison_generation_model_summary.tsv
- RAPIDS V100:
  - molscore/outputs/comparison_matrix/rapids_v100/compare_GuacaMol_20260506_120813/model_runtime_summary.tsv
  - molscore/outputs/comparison_matrix/rapids_v100/compare_GuacaMol_20260506_120813/comparison_overall.tsv
  - molscore/outputs/comparison_matrix/rapids_v100/compare_GuacaMol_20260506_120813/comparison_generation_model_summary.tsv
- Dask + RAPIDS V100:
  - molscore/outputs/comparison_matrix/dask_rapids_v100/compare_GuacaMol_20260506_121020/model_runtime_summary.tsv
  - molscore/outputs/comparison_matrix/dask_rapids_v100/compare_GuacaMol_20260506_121020/comparison_overall.tsv
  - molscore/outputs/comparison_matrix/dask_rapids_v100/compare_GuacaMol_20260506_121020/comparison_generation_model_summary.tsv
