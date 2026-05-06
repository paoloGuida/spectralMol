# Release Notes: Acceleration Roadmap (Phase 0-3)

Date: 2026-05-06
Project: molevoDrugDiscovery
Scope: GuacaMol comparison pipeline acceleration and cluster-scale validation

## Executive Summary
This release completes the Phase 0-3 acceleration roadmap for GuacaMol benchmarking, including profiling instrumentation, Dask execution support, GPU diversity acceleration with threshold gating, and RAPIDS/cuDF dataframe backend integration. A full 4-way SLURM matrix run validated both runtime improvements and score parity.

## What Changed

### Phase 0: Profiling Instrumentation
1. Added stage-level profiling support to isolate runtime hotspots.
2. Verified evolution and scoring loop dominates end-to-end runtime.

### Phase 1: Dask Execution Path
1. Added executor switching for comparison runs.
2. Enabled Dask-based parallel orchestration for campaign-scale runs.

### Phase 2: GPU Diversity Acceleration
1. Added [core/gpu_utils.py](core/gpu_utils.py) for optional GPU-accelerated diversity kernels with CPU fallback.
2. Added threshold-gated activation in [core/reports.py](core/reports.py) using:
3. MOLEVO_DIVERSITY_GPU_ENABLED
4. MOLEVO_DIVERSITY_GPU_MIN_N (default 512)
5. Preserved numerical consistency while avoiding small-N GPU overhead.

### Phase 3: Dataframe Backend Switching
1. Added backend selection and auto-resolution in [benchmarks/Guacamol/benchmark_compare_models.py](benchmarks/Guacamol/benchmark_compare_models.py): pandas or cuDF.
2. Added safe fallback behavior on CPU-only nodes.
3. Added backend-aware summary table builders and aggregation normalization.

### Reliability Fixes During Rollout
1. Fixed comparison runner path bug to local evolution script in [benchmarks/Guacamol/benchmark_compare_models.py](benchmarks/Guacamol/benchmark_compare_models.py).
2. Standardized LD_LIBRARY_PATH handling in SLURM scripts for cuDF compatibility on V100 nodes.

## Cluster Validation Matrix
Configurations executed:
1. baseline_cpu: thread + pandas
2. dask_cpu: dask + pandas
3. rapids_v100: thread + auto backend (resolved to cuDF)
4. dask_rapids_v100: dask + auto backend (resolved to cuDF)

Jobs completed successfully with exit code 0:
1. 46909028
2. 46909029
3. 46909030
4. 46909031

## Performance Outcomes
Mean elapsed runtime by configuration:
1. baseline_cpu: 1541.2372 s
2. dask_cpu: 1463.3349 s (1.0532x, 5.05% reduction)
3. rapids_v100: 1349.2829 s (1.1423x, 12.45% reduction)
4. dask_rapids_v100: 1316.1125 s (1.1711x, 14.61% reduction)

Key observation:
1. Combined Dask+RAPIDS gives the best end-to-end runtime.
2. Improvement is sub-additive, indicating partially overlapping bottlenecks.

## Quality and Reproducibility
1. Aggregate score metrics matched across all configurations (floating-point noise only).
2. n_failed = 0 across all final matrix runs.
3. Unit test status after changes: 48 passed, 2 deselected.

## Default Recommendation
Use [slurm_scripts/compare_matrix_dask_rapids_v100.sbatch](slurm_scripts/compare_matrix_dask_rapids_v100.sbatch) as the default launch mode for production-scale matrix comparisons.

## Related Artifacts
1. Full matrix report: [molscore/outputs/comparison_matrix/matrix_report_20260506_round2.md](molscore/outputs/comparison_matrix/matrix_report_20260506_round2.md)
2. Consolidated speedup table: [molscore/outputs/comparison_matrix/speedup_matrix_20260506_round2.tsv](molscore/outputs/comparison_matrix/speedup_matrix_20260506_round2.tsv)
3. Root summary section: [README.md](README.md)
