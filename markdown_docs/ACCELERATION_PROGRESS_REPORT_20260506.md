# RAPIDS + Dask Acceleration Progress Report

Date: 2026-05-06
Project: molevoDrugDiscovery
Scope: GuacaMol benchmark path, Dask job fanout, RAPIDS diversity acceleration

## 1) Executive Summary

This report summarizes implementation and validation progress through:
- Phase 0: profiling and baseline measurement
- Phase 1: Dask campaign-level parallelization + per-task timing
- Phase 2: GPU-ready diversity computation path with threshold gating

Key outcome to date:
- Phase 1 is complete and validated.
- Phase 2 foundation is implemented and benchmarked.
- GPU acceleration for diversity is beneficial at larger set sizes; threshold gating is now implemented in production code.

---

## 2) Completed Work by Phase

## Phase 0: Profiling and Baseline

### Implemented
- Added profiling instrumentation and stage timing outputs.
- Added high-level timing in benchmark runner path used in practice.
- Produced `phase0_timing_report.json` outputs for smoke runs.

### Findings
- Evolution stage dominates runtime (~99.9% of measured walltime in tested smoke configuration).
- Baseline representative value from smoke run:
  - Evolution: ~11.62s
  - Total: ~11.63s

### Conclusion
- Profiling objective achieved.
- Established baseline for later speedup comparisons.

---

## Phase 1: Dask Campaign Parallelism

### Implemented
- Added `--executor {thread,dask}` to compare runner.
- Added `--dask-scheduler` support for remote scheduler connection.
- Implemented Dask execution backend with local-cluster fallback and result streaming.
- Refactored result collection to shared helper path.
- Added per-task elapsed logging in evolve runner.

### Validation
- Dask CLI options verified in `--help`.
- Dask dry-run path validated with corrected runtime flags.
- Unit tests passed after Phase 1 updates.

### Operational notes resolved
- Added required flags for smoke runs where defaults referenced unavailable local resources:
  - `--model-spec-file benchmarks/Guacamol/model_specs.json`
  - `--seed-smiles-file env/lib/python3.12/site-packages/guacamol/data/holdout_set_gcm_v1.smiles`
  - `--no-bootstrap-scoring-envs`

### Conclusion
- Phase 1 is complete and functional.

---

## Phase 2: RAPIDS Diversity Acceleration

## Environment Readiness

### GPU environment checks (V100)
Confirmed on allocated V100 node:
- cudf: 26.02.01
- cuml: 26.02.000
- rmm: 26.02.00
- torch CUDA: available

### Critical environment fix
Encountered:
- `GLIBCXX_3.4.31 not found` from system `libstdc++` when importing RAPIDS stack.

Resolution:
- Prepend env libs via:
  - `export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"`
- Applied in all SLURM scripts used for benchmark/testing.

## Code Implemented

### New module
- Added GPU utility module:
  - `core/gpu_utils.py`
- Includes:
  - Morgan bit-matrix construction from SMILES
  - Pairwise Tanimoto mean distance (CPU path and CuPy GPU path)
  - End-to-end helper from SMILES

### Production integration
- Updated diversity calculation in reports path to call shared utility.
- Added threshold-gated GPU activation in `core/reports.py`:
  - CPU for small sets
  - GPU preferred for larger sets
- Introduced environment controls:
  - `MOLEVO_DIVERSITY_GPU_ENABLED` (default enabled)
  - `MOLEVO_DIVERSITY_GPU_MIN_N` (default `512`)

## Benchmark Harness

### Added benchmark scripts
- `slurm_scripts/diversity_speed_sweep.py`
- `slurm_scripts/diversity_speed_sweep_cpu.sbatch`
- `slurm_scripts/diversity_speed_sweep_gpu_v100.sbatch`

### Benchmark improvements
- Expanded sweep sizes to: `32,64,128,256,512,1024,2048`
- Split timing by phase:
  - fingerprint build time
  - pairwise similarity time
- Reported both overall speedup and pairwise-kernel speedup.

## Measured Results (latest runs)

### CPU node results (overall speedup new_auto_vs_old)
- 32: 0.798x
- 64: 0.817x
- 128: 0.837x
- 256: 0.866x
- 512: 0.927x
- 1024: 1.010x
- 2048: 1.259x

### GPU node results (overall speedup new_auto_vs_old)
- 32: 0.737x
- 64: 0.771x
- 128: 0.838x
- 256: 0.899x
- 512: 1.034x
- 1024: 1.104x
- 2048: 1.399x

### GPU pairwise kernel speedup (new_auto_pairwise vs old_pairwise)
- 128: 1.332x
- 256: 4.436x
- 512: 14.504x
- 1024: 27.911x
- 2048: 43.050x

### Interpretation
- For small sizes, overhead dominates and old path can be faster.
- From ~512 molecules upward, GPU path becomes beneficial overall.
- Pairwise kernel itself scales strongly on GPU at larger sizes.

### Conclusion
- Phase 2 implementation is successful as a practical, threshold-gated acceleration.
- Current default threshold (`512`) is supported by measured crossover behavior.

---

## 3) Testing Status

Current status:
- Unit tests: passed (`48 passed, 2 deselected`) after recent changes.
- SLURM benchmark scripts: created, validated, and used successfully.

---

## 4) Known Risks / Constraints

- Most total runtime remains in external scoring/evolution path (MolScore-heavy workload), so Phase 2 gains primarily affect report/diversity components.
- For workloads with small generation diversity sets, GPU path should remain gated to avoid regressions.
- Cluster runtime portability depends on preserving `LD_LIBRARY_PATH` env override in job scripts.

---

## 5) Recommended Next Steps

1. Keep `MOLEVO_DIVERSITY_GPU_MIN_N=512` as default in production runs.
2. Optionally test `384` threshold on target workloads to evaluate earlier GPU activation trade-off.
3. Begin deeper profiling of MolScore-internal scoring paths (Phase 3 focus), since that is still the dominant runtime component.
4. Add optional telemetry line in report generation to log whether CPU or GPU path was used per generation for easier operations tuning.

---

## 6) Artifacts Produced

- Benchmark and execution scripts in `slurm_scripts/`
- Timing outputs in `molscore/outputs/phase2_benchmarks/`
- Log evidence in `logs/diversity-speed-*.out`

This concludes status reporting up to the current implementation point.
