# GuacaMol Frequency Parent-Lineage Feature-Change Analysis

Generated: 2026-09-03T07:07:44+00:00
Input root: `/ibex/scratch/colleoe/spectralMol/guacamol_frequency_ablation_lineage_20260825_094359`
Source: `lineage`
Molecule set: top 100 unique generated molecules per task/seed run with recorded parent smiles.

Each child is compared with the most similar recorded parent among parent, parent2, and parent3. This avoids the nearest-seed proxy used by the previous feature-change analysis.

## Condition Summary

| condition | n_molecules | mean_best_parent_tanimoto | scaffold_changed_fraction | same_scaffold_fraction | same_scaffold_functional_group_changed_fraction | mean_functional_group_delta_l1 | mean_abs_heavy_atom_delta | mean_abs_ring_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-spectrum | 12000 | 0.3717 | 0.5913 | 0.4087 | 0.2906 | 8.7622 | 3.7314 | 0.7081 |
| high-only | 12000 | 0.2228 | 0.8317 | 0.1683 | 0.1424 | 12.8475 | 11.9587 | 1.8763 |
| low-only | 12000 | 0.4214 | 0.5693 | 0.4308 | 0.3777 | 8.7490 | 3.9917 | 0.7216 |
| random-matrix | 12000 | 0.4685 | 0.5152 | 0.4848 | 0.4012 | 7.6016 | 4.4807 | 0.7137 |

## Interpretation Checks

| check | passed | observed | expected |
| --- | --- | --- | --- |
| high-only children are closer to their recorded parents than low-only | 0 | 0.222802557644755 vs 0.421358621470323 | high > low |
| high-only preserves parent scaffold more often than low-only | 0 | 0.16825 vs 0.43075 | high > low |
| low-only changes parent scaffold more often than high-only | 0 | 0.56925 vs 0.83175 | low > high |
| low-only has larger parent-child heavy-atom shifts than high-only | 0 | 3.9916666666666667 vs 11.958666666666666 | low > high |
| high-only has more same-scaffold functional-group changes than low-only | 0 | 0.14241666666666666 vs 0.37766666666666665 | high > low |
| random-matrix differs from full-spectrum in explicit parent-child structural profile | 1 | profile differs | non-identical |

## Files

- `frequency_lineage_feature_rows.tsv`: molecule-level parent-child structural-change rows.
- `frequency_lineage_feature_summary.tsv`: condition-level summary.
- `frequency_lineage_feature_task_summary.tsv`: task-level summary.
- `frequency_lineage_feature_interpretation_checks.tsv`: checks for the low/global vs high/local interpretation.
- `frequency_lineage_feature_run_coverage.tsv`: per task/seed lineage export coverage.
- `frequency_lineage_feature_manifest.json`: run metadata.
