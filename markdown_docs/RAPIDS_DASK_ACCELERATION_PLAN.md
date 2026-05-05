# RAPIDS + Dask Acceleration Plan

This document captures where acceleration is likely to help in this repository, where it is unlikely to help, and how to proceed safely.

## Scope

- Primary objective: reduce wall-clock runtime for large benchmark campaigns.
- Secondary objective: keep behavior and output files unchanged.
- Constraint: RDKit and docking kernels are largely CPU/external-tool bound, so GPU dataframe acceleration is useful only for selected pipeline segments.

## What We Know From Code Inspection

### Hot loops in evolutionary search

- The main generation loop is in `benchmarks/Guacamol/evolve_vs_molscore_benchmark.py`.
- Offspring creation and novelty filtering are Python-loop heavy.
- Key locations include repeated loops around offspring filling and archive updates.

### Oracle path bottlenecks

- Oracle dispatch and cache/history logic is in `core/oracles/oracle.py`.
- Repeated canonicalization, per-molecule checks, and DataFrame concatenation happen in the hot path.
- Similarity components are repeatedly evaluating Morgan fingerprints and Tanimoto similarities.

### Reporting/aggregation is tabular and parallel-friendly

- Cross-run aggregation and summaries are in `benchmarks/Guacamol/benchmark_compare_models.py`.
- These groupby-heavy sections are potential RAPIDS candidates once result tables are large enough.

### Saturn NSGA-II path is compute-loop heavy

- NSGA-II utilities in `benchmarks/Saturn/compare_scalar_vs_nsga2_saturn.py` use custom Python/Numpy sorting and dominance loops.
- Population sizes are typically modest, so gains may come more from task-level parallelism than GPU kernels.

## What We Changed Immediately

Low-risk CPU-side improvements were applied first to remove avoidable overhead:

1. Similarity components now call `BulkTanimotoSimilarity` against the full reference list in one call, instead of nested list constructions:
   - `core/oracles/similarity/tanimoto_similarity.py`
   - `core/oracles/similarity/jaccard_distance.py`
   - `core/oracles/similarity/jaccard_distance_dataset.py`

2. Replaced `np.vectorize(Chem.MolFromSmiles)` with explicit list comprehension in the oracle hot path:
   - `core/oracles/oracle.py`

These are baseline optimizations and reduce Python overhead regardless of RAPIDS/Dask adoption.

## Where RAPIDS Is Appropriate

### Good candidates

- Large result-table post-processing:
  - `benchmarks/Guacamol/benchmark_compare_models.py`
  - Potential migration: `pandas -> cudf` for read/groupby/aggregation/write in summary generation.

- Large-scale metrics aggregation where table sizes are millions of rows.

### Poor candidates

- RDKit chemistry kernels (`MolFromSmiles`, Morgan FP generation) are not directly accelerated by RAPIDS.
- Docking execution is external (QuickVina2-GPU). RAPIDS will not speed the docking binary itself.
- Small/medium tables may run slower on GPU due to transfer overhead.

## Where Dask Is Appropriate

### Strong candidates

- Seed x model x task fanout orchestration in benchmark campaigns.
- Independent scoring and report jobs across tasks.
- Distributed post-processing of many output files.

### Weak candidates

- Inner per-molecule mutation loops if each task is too fine-grained.
- Small single-node runs where scheduler overhead dominates.

## Phased Implementation Plan

## Phase 0: Instrumentation (required before deeper changes)

- Add per-stage timers and counters around:
  - scoring calls
  - offspring generation
  - oracle component evaluation
  - report generation
- Persist timing breakdown as structured TSV/JSON per run.

Deliverable: a profile report showing top 3 wall-time contributors on representative workloads.

## Phase 1: Dask for campaign-level parallelism

- Introduce optional Dask execution mode for model/seed/task jobs.
- Keep current `ThreadPoolExecutor` path as default fallback.
- Add deterministic seeding and per-task retry policy.

Expected impact: lower total campaign wall-time on multi-core or cluster environments.

## Phase 2: RAPIDS for heavy table aggregation

- Add optional `cudf` path for summary pipelines in comparison scripts.
- Fallback to pandas automatically when RAPIDS is unavailable.
- Guard with CLI flag, e.g., `--dataframe-backend {pandas,cudf,auto}`.

Expected impact: faster groupby-heavy summaries on large run sets.

## Phase 3: Hybrid optimization for chemistry loops

- Keep RDKit on CPU but improve throughput with batching and process-level parallelism where safe.
- Avoid repeated expensive transforms in hot loops (canonicalization/fingerprint rework).
- Evaluate cache key normalization strategy to reduce duplicate work.

Expected impact: incremental speedups in oracle and generation loops.

## Unknowns (Explicit)

- I do not know the exact speedup yet without runtime profiling on your real workloads.
- I do not know whether your dominant bottleneck is docking, RDKit parsing, or post-processing until we collect stage timings.
- I do not know the effective GPU utilization and transfer overhead on your cluster without test runs under your production settings.

## Recommended Next Benchmark Matrix

Run the following matrix and compare wall-clock and throughput:

1. Baseline (current code, single process)
2. Dask campaign parallelism enabled (CPU)
3. RAPIDS aggregation enabled (if large tables)
4. Dask + RAPIDS combined

Track:

- total runtime
- molecules evaluated per second
- oracle calls per second
- summary generation time
- peak memory (CPU/GPU)

## Acceptance Criteria

- Minimum 20% campaign wall-time reduction on representative workloads.
- No regression in benchmark outputs (score columns, report files, ranking consistency).
- Auto-fallback to non-RAPIDS path when GPU stack is absent.
