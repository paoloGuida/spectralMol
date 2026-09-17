# GuacaMol Frequency Feature-Change Analysis

Generated: 2026-08-24T19:49:16+00:00
Input root: `PORTABLE_OUTPUT_ROOT/guacamol_frequency_ablation_merged_20260824_150715`
Source: `generator-top`
Molecule set: top 100 unique, non-initial generated molecules per task/seed run.

Nearest initial-population molecule is used as a parent proxy because explicit parent lineage is not stored in the GuacaMol exports.

## Condition Summary

| condition | n_molecules | mean_nearest_seed_tanimoto | scaffold_changed_fraction | same_scaffold_fraction | same_scaffold_functional_group_changed_fraction | mean_functional_group_delta_l1 | mean_abs_heavy_atom_delta | mean_abs_ring_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-spectrum | 10730 | 0.1753 | 0.9896 | 0.0104 | 0.0104 | 16.8583 | 7.6275 | 1.3399 |
| high-only | 7931 | 0.1558 | 0.9767 | 0.0233 | 0.0229 | 14.4716 | 9.6950 | 1.5872 |
| low-only | 11176 | 0.1847 | 0.9863 | 0.0137 | 0.0137 | 17.7570 | 8.0028 | 1.3457 |
| random-matrix | 11370 | 0.2037 | 0.9867 | 0.0133 | 0.0133 | 15.9027 | 7.9182 | 1.2006 |

## Interpretation Checks

| check | passed | observed | expected |
| --- | --- | --- | --- |
| high-only keeps molecules closer to initial population than low-only | 0 | 0.15577742239998563 vs 0.18465544612181506 | high > low |
| high-only preserves scaffold more often than low-only | 1 | 0.023326188374732063 vs 0.013690050107372943 | high > low |
| low-only changes scaffold more often than high-only | 1 | 0.9863099498926271 vs 0.9766738116252679 | low > high |
| low-only has larger heavy-atom shifts than high-only | 0 | 8.002773801002148 vs 9.694994326062288 | low > high |
| high-only has more same-scaffold functional-group changes than low-only | 1 | 0.02294792586054722 vs 0.013690050107372943 | high > low |
| random-matrix is not identical to full-spectrum in structural-change profile | 1 | profile differs | non-identical |

## Files

- `frequency_feature_rows.tsv`: molecule-level structural-change rows.
- `frequency_feature_summary.tsv`: condition-level summary.
- `frequency_feature_task_summary.tsv`: task-level summary.
- `frequency_feature_interpretation_checks.tsv`: checks for the low/global vs high/local interpretation.
- `frequency_feature_run_coverage.tsv`: per task/seed export coverage.
- `frequency_feature_manifest.json`: run metadata.
