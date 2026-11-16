# Ablation Execution Report (2026-05-16)

## Scope
This note documents completion of the ablation campaign and post-processing for:
- config root: `molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container`
- run: `compare_GuacaMol_20260515_224644`

## Execution Status
Ablation campaign execution completed successfully.

Evidence from run log:
- job log: `logs/cmp-ablation-dask-rapids-v100-47059911.out`
- executor: Dask local cluster with 16 workers
- submitted jobs: 24 total (4 model variants x 6 seeds)
- all model/seed combinations reported `[done]`
- no run-level failures reported in exported artifacts

## Generated Comparison Artifacts (Run Directory)
Run directory:
- `molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container/compare_GuacaMol_20260515_224644`

Core files produced:
- `comparison_raw.tsv`
- `comparison_task_summary.tsv`
- `comparison_overall.tsv`
- `comparison_generation_raw.tsv`
- `comparison_generation_summary.tsv`
- `comparison_generation_model_summary.tsv`
- `model_runtime.tsv`
- `model_runtime_summary.tsv`
- `failures.tsv`

## Dedicated Ablation Summary Step
The dedicated summarizer was executed after run completion:

Command used:
```bash
/ibex/user/barradd/molevoDrugDiscovery/env/bin/python \
  slurm_scripts/summarize_ablation_results.py \
  --config-dir molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container \
  --run-dir compare_GuacaMol_20260515_224644 \
  --output-prefix ablation_summary
```

Summary outputs produced:
- `molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container/ablation_summary_overall_20260516_041214.tsv`
- `molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container/ablation_summary_task_delta_20260516_041214.tsv`
- `molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container/ablation_summary_20260516_041214.md`

## Key Results (Overall Summary)
From `ablation_summary_overall_20260516_041214.tsv`:

1. high-only (`local_evolution_high_only`)
- mean best score: 0.754234
- delta vs full: +0.005026
- mean elapsed: 1331.263564 s
- speedup vs full: 1.155757x
- evals/s: 15.023321
- failures: 0

2. full-spectrum (`local_evolution`)
- mean best score: 0.749207
- delta vs full: 0.000000
- mean elapsed: 1538.617772 s
- speedup vs full: 1.000000x
- evals/s: 12.998680
- failures: 0

3. low-only (`local_evolution_low_only`)
- mean best score: 0.736732
- delta vs full: -0.012475
- mean elapsed: 1548.090109 s
- speedup vs full: 0.993881x
- evals/s: 12.919145
- failures: 0

4. random-matrix (`local_evolution_random_matrix`)
- mean best score: 0.646941
- delta vs full: -0.102267
- mean elapsed: 1256.052051 s
- speedup vs full: 1.224963x
- evals/s: 15.922907
- failures: 0

## Environment Pivot Note
Launcher execution was pivoted from container runtime to local conda env at `$PWD/env` for reliability during this testing phase. The compared/ablation SLURM launchers now execute directly with local Python from the environment.
