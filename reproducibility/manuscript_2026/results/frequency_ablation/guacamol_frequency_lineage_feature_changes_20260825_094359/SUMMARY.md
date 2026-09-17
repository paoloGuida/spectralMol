# GuacaMol Frequency Parent-Lineage Feature-Change Analysis

Generated: 2026-09-03T06:13:08+00:00
Input root: `PORTABLE_OUTPUT_ROOT/guacamol_frequency_ablation_lineage_20260825_094359`
Source: `top-lineage`
Molecule set: top 100 unique generated molecules per task/seed run with recorded parent smiles.

Each child is compared with the most similar recorded parent among parent, parent2, and parent3. This avoids the nearest-seed proxy used by the previous feature-change analysis.

## Condition Summary

| condition | n_molecules | mean_best_parent_tanimoto | scaffold_changed_fraction | same_scaffold_fraction | same_scaffold_functional_group_changed_fraction | mean_functional_group_delta_l1 | mean_abs_heavy_atom_delta | mean_abs_ring_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-spectrum | 10783 | 0.4044 | 0.5452 | 0.4548 | 0.3234 | 7.4319 | 3.0185 | 0.5568 |
| high-only | 8068 | 0.2933 | 0.7498 | 0.2502 | 0.2118 | 9.4602 | 8.0693 | 1.3593 |
| low-only | 11021 | 0.4508 | 0.5313 | 0.4687 | 0.4109 | 7.7117 | 3.2282 | 0.5646 |
| random-matrix | 11301 | 0.4919 | 0.4854 | 0.5146 | 0.4259 | 6.7202 | 3.6205 | 0.5515 |

## Interpretation Checks

| check | passed | observed | expected |
| --- | --- | --- | --- |
| high-only children are closer to their recorded parents than low-only | 0 | 0.29333928570990425 vs 0.45082487002510885 | high > low |
| high-only preserves parent scaffold more often than low-only | 0 | 0.2502478929102628 vs 0.4686507576444969 | high > low |
| low-only changes parent scaffold more often than high-only | 0 | 0.5313492423555032 vs 0.7497521070897373 | low > high |
| low-only has larger parent-child heavy-atom shifts than high-only | 0 | 3.2282007077397696 vs 8.069286068418442 | low > high |
| high-only has more same-scaffold functional-group changes than low-only | 0 | 0.21182449181953397 vs 0.4108520097994737 | high > low |
| random-matrix differs from full-spectrum in explicit parent-child structural profile | 1 | profile differs | non-identical |

## Files

- `frequency_lineage_feature_rows.tsv`: molecule-level parent-child structural-change rows.
- `frequency_lineage_feature_summary.tsv`: condition-level summary.
- `frequency_lineage_feature_task_summary.tsv`: task-level summary.
- `frequency_lineage_feature_interpretation_checks.tsv`: checks for the low/global vs high/local interpretation.
- `frequency_lineage_feature_run_coverage.tsv`: per task/seed lineage export coverage.
- `frequency_lineage_feature_manifest.json`: run metadata.
