# GuacaMol Frequency Ablation With Explicit Parent Lineage

Date analyzed: 2026-09-03

This run repeats the GuacaMol frequency ablation using explicit parent-lineage logging, so generated molecules are compared to their recorded parents instead of to a nearest-seed proxy.

## Run

- IBEX job ID: `50835722`
- Raw output root: `/ibex/scratch/colleoe/spectralMol/guacamol_frequency_ablation_lineage_20260825_094359`
- Conditions: `full-spectrum`, `high-only`, `low-only`, `random-matrix`
- Seeds: `0,1,2,3,4,5`
- Tasks: GuacaMol task indexes `0-19`
- Budget: `20000` oracle calls per task/seed/condition
- Population size: `256`
- Batch size: `256`
- Completed task runs: `480/480`
- Tracebacks found in Slurm logs: `0`

## Code Instrumentation

The theta evolution logic was kept theta-only. The changes add lineage/debug metadata and do not add non-theta decision variables.

- `spectralMol/core/spectral_evolution.py`
  - Added parent UID, parent SMILES, parent score, birth generation, and lineage operator fields to `SpectralIndividual`.
  - Added `propose_child_with_lineage(...)` to return the selected theta parents/operator while preserving the existing `propose_child(...)` API.

- `spectralMol/benchmarks/Guacamol/evolve_vs_molscore_benchmark.py`
  - Added lineage assignment/recording for initial molecules, reencoded molecules, and offspring.
  - Added per-run lineage outputs:
    - `generator_top_molecules_lineage.csv`
    - `molecule_lineage_by_generation.tsv`

- `spectralMol/benchmarks/Guacamol/analyze_frequency_lineage_feature_changes.py`
  - Added parent-child structural analysis using the most similar recorded parent among parent slots.

## Score Reproduction

Score summaries were aggregated from the fresh run and saved locally in:

`reproducibility_backups/unified_guacamol_saturn_20260817/verification/guacamol_frequency_ablation_lineage_scores_20260825_094359/`

| condition | task/seed runs | mean best score | seed aggregate mean +- std | mean elapsed seconds |
| --- | ---: | ---: | ---: | ---: |
| full-spectrum | 120 | 0.7053 | 14.1054 +- 0.2418 | 201.25 |
| high-only | 120 | 0.6577 | 13.1544 +- 0.2152 | 190.79 |
| low-only | 120 | 0.7232 | 14.4641 +- 0.3008 | 199.18 |
| random-matrix | 120 | 0.7247 | 14.4943 +- 0.2786 | 198.25 |

## Parent-Lineage Feature Results

Feature summaries were saved locally in:

`reproducibility_backups/unified_guacamol_saturn_20260817/verification/guacamol_frequency_lineage_feature_changes_20260825_094359/`

The analysis used the top 100 unique generated molecules per task/seed run with recorded parent SMILES.

| condition | molecules | parent Tanimoto | scaffold changed | same scaffold | same-scaffold FG changed | FG delta L1 | heavy atom delta | ring delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-spectrum | 10783 | 0.4044 | 0.5452 | 0.4548 | 0.3234 | 7.4319 | 3.0185 | 0.5568 |
| high-only | 8068 | 0.2933 | 0.7498 | 0.2502 | 0.2118 | 9.4602 | 8.0693 | 1.3593 |
| low-only | 11021 | 0.4508 | 0.5313 | 0.4687 | 0.4109 | 7.7117 | 3.2282 | 0.5646 |
| random-matrix | 11301 | 0.4919 | 0.4854 | 0.5146 | 0.4259 | 6.7202 | 3.6205 | 0.5515 |

## Interpretation

This explicit lineage analysis does not confirm the simple manuscript interpretation that high-frequency nodes mainly produce local-group changes while low-frequency nodes mainly produce scaffold/global changes.

Observed here:

- `high-only` has the lowest parent Tanimoto (`0.2933`) and the highest scaffold-change fraction (`0.7498`).
- `high-only` also has the largest heavy-atom and ring-count shifts.
- `low-only` is closer to its recorded parents than `high-only` and preserves scaffolds more often.
- `random-matrix` differs from `full-spectrum`, so the ablation is detecting condition-level structural differences.

The clean conclusion for the current merged code is therefore: frequency restriction changes the structural profile of generated molecules, but the explicit parent-lineage evidence reverses the expected high/local vs low/global direction. The manuscript claim should either be revised or supported with an alternative analysis that tests spectral perturbations under a controlled parent and operator distribution.

## Local Files

- `guacamol_frequency_lineage_feature_changes_20260825_094359/SUMMARY.md`
- `guacamol_frequency_lineage_feature_changes_20260825_094359/frequency_lineage_feature_summary.tsv`
- `guacamol_frequency_lineage_feature_changes_20260825_094359/frequency_lineage_feature_task_summary.tsv`
- `guacamol_frequency_lineage_feature_changes_20260825_094359/frequency_lineage_feature_rows.tsv`
- `guacamol_frequency_lineage_feature_changes_20260825_094359/frequency_lineage_feature_interpretation_checks.tsv`
- `guacamol_frequency_lineage_feature_changes_20260825_094359/frequency_lineage_feature_run_coverage.tsv`
- `guacamol_frequency_ablation_lineage_scores_20260825_094359/condition_summary.tsv`
- `guacamol_frequency_ablation_lineage_scores_20260825_094359/per_seed_aggregate.tsv`
- `guacamol_frequency_ablation_lineage_scores_20260825_094359/per_task_seed_scores.tsv`

