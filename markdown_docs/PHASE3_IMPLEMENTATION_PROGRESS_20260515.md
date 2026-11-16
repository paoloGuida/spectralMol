# Phase 3 Implementation Progress (May 15, 2026)

## Scope Requested
- Implement option 2: dedicated ablation summarizer script.
- Document all implementation work completed so far in this coding cycle.

## Completed Work Summary

### 1) Containerized Matrix Infrastructure
Implemented all-container 4-way matrix launchers and orchestration for:
- baseline CPU
- dask CPU
- RAPIDS
- dask + RAPIDS

Files added/updated:
- slurm_scripts/compare_matrix_baseline_container_cpu.sbatch (new)
- slurm_scripts/compare_matrix_dask_container_cpu.sbatch (new)
- slurm_scripts/compare_matrix_rapids_container_v100.sbatch (updated image default)
- slurm_scripts/compare_matrix_dask_rapids_container_v100.sbatch (updated image default)

Image standardization:
- docker.io/dxbarradas/spectralmol-rapids:wolfi-conda is now the default RAPIDS image in matrix scripts.

### 2) Matrix Post-Run Summary Automation
Added a generic matrix summarizer for wall-clock and throughput comparisons.

File added:
- slurm_scripts/summarize_matrix_results.py

Outputs produced by this script:
- matrix_summary_<timestamp>.tsv
- matrix_summary_<timestamp>.md

Metrics included:
- mean elapsed seconds
- campaign wall-clock seconds
- evals/sec (mean runtime basis)
- evals/sec (campaign basis)
- speedup and time-reduction vs baseline
- failure counts

### 3) Submit-and-Summarize Helper
Added helper that submits all 4 matrix jobs then submits a dependent summary job.

File added:
- slurm_scripts/submit_matrix_container_and_summarize.sh

Behavior:
- submits 4 matrix jobs
- captures job IDs
- submits dependent summary job with afterok dependency on all 4

### 4) Ablation Mode Integration (Active Runners)
Integrated ablation mode control end-to-end into the active GuacaMol comparison/evolution path.

Files updated:
- benchmarks/Guacamol/benchmark_compare_models.py
- benchmarks/Guacamol/evolve_vs_molscore_benchmark.py

New CLI flag:
- --fourier-mode with choices:
  - full-spectrum
  - low-only
  - high-only
  - random-matrix

What was wired:
- comparison runner accepts and forwards --fourier-mode to local evolution jobs
- run metadata now records requested/resolved mode fields
- evolution runner logs/records mode and applies mode-specific operator-space presets

Important note:
- In the current active runner, these are operator-space ablation proxies (SMILES mutation behavior controls), not direct latent-Fourier coefficient masking.

### 5) Ablation Variant Model Keys
Added model aliases to run all ablation conditions in one comparison campaign.

File updated:
- benchmarks/Guacamol/model_specs.json

Added model keys:
- local_evolution_low_only
- local_evolution_high_only
- local_evolution_random_matrix

### 6) Ablation Campaign Launcher
Added dedicated SLURM launcher to execute ablation variants in one containerized Dask+RAPIDS campaign.

File added:
- slurm_scripts/compare_ablation_dask_rapids_container_v100.sbatch

Default models:
- local_evolution
- local_evolution_low_only
- local_evolution_high_only
- local_evolution_random_matrix

### 7) Option 2 Implemented: Dedicated Ablation Summarizer
Added dedicated ablation summarizer script.

File added:
- slurm_scripts/summarize_ablation_results.py

What it does:
- reads one ablation compare run directory
- builds overall model-condition summary with runtime + throughput
- computes delta vs full-spectrum baseline
- exports task-level delta table vs full-spectrum
- writes markdown report and two TSV outputs

Generated files per invocation:
- <prefix>_overall_<timestamp>.tsv
- <prefix>_task_delta_<timestamp>.tsv
- <prefix>_<timestamp>.md

## Validation and Checks Performed

### Python checks
- py_compile passed for:
  - benchmarks/Guacamol/benchmark_compare_models.py
  - benchmarks/Guacamol/evolve_vs_molscore_benchmark.py
  - slurm_scripts/summarize_ablation_results.py

### Shell checks
- bash -n passed for:
  - slurm_scripts/compare_ablation_dask_rapids_container_v100.sbatch
  - slurm_scripts/submit_matrix_container_and_summarize.sh

### Tool-level diagnostics
- no editor/lint diagnostics reported for modified scripts during checks

### Runtime smoke checks
- summarize_ablation_results.py executed successfully against existing comparison data:
  - config-dir: molscore/outputs/comparison_matrix/dask_rapids_v100
  - outputs generated:
    - molscore/outputs/comparison_matrix/dask_rapids_v100/ablation_summary_smoke_overall_20260515_151424.tsv
    - molscore/outputs/comparison_matrix/dask_rapids_v100/ablation_summary_smoke_task_delta_20260515_151424.tsv
    - molscore/outputs/comparison_matrix/dask_rapids_v100/ablation_summary_smoke_20260515_151424.md

## How To Run (Current)

### Run ablation campaign
sbatch slurm_scripts/compare_ablation_dask_rapids_container_v100.sbatch

### Summarize latest ablation campaign
/ibex/user/barradd/molevoDrugDiscovery/env/bin/python slurm_scripts/summarize_ablation_results.py

### Summarize explicit run/config
/ibex/user/barradd/molevoDrugDiscovery/env/bin/python slurm_scripts/summarize_ablation_results.py \
  --config-dir molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container \
  --run-dir <compare_run_dir_name> \
  --output-prefix ablation_summary

### Run full 4-way matrix + auto-summary pipeline
slurm_scripts/submit_matrix_container_and_summarize.sh

## Known Caveats
- The current ablation behavior is implemented as operator-space proxies in the active SMILES mutation runner.
- A future step can add true latent-coefficient masking semantics if/when the active pipeline directly invokes latent Fourier genotype operators.

## Next Logical Extensions
1. Add LEOMol command entry + preflight launcher.
2. Add statistical testing script (ANOVA/Kruskal + effect sizes) on ablation outputs.
3. Add manuscript-ready table generator from ablation summary TSVs.
