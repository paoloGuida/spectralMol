# GuacaMol Frequency-Mode Ablation

Generated: 2026-08-24T12:59:42Z
Input root: `/ibex/scratch/colleoe/spectralMol/guacamol_frequency_ablation_merged_20260824_150715`

## Condition Summary

| condition | mean_best_score | std_best_score | delta_vs_full | mean_elapsed_seconds | std_elapsed_seconds | speedup_vs_full | evals_per_second | budget_over_mean_elapsed | n_rows | n_tasks | n_seeds | n_failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-spectrum | 0.7112 | 0.2770 | 0.0000 | 204.6166 | 85.3191 | 1.0000 | 63.8071 | 97.7438 | 120 | 20 | 6 | 0 |
| high-only | 0.6646 | 0.2798 | -0.0466 | 177.2721 | 46.0404 | 1.1543 | 73.6495 | 112.8209 | 120 | 20 | 6 | 0 |
| low-only | 0.7303 | 0.2818 | 0.0190 | 195.5661 | 89.9708 | 1.0463 | 66.7600 | 102.2672 | 120 | 20 | 6 | 0 |
| random-matrix | 0.7252 | 0.2785 | 0.0140 | 197.4061 | 63.5277 | 1.0365 | 66.1378 | 101.3140 | 120 | 20 | 6 | 0 |

## Claim Checks

| check | passed | observed | expected |
| --- | --- | --- | --- |
| all_four_conditions_present | 1 | full-spectrum,high-only,low-only,random-matrix | full-spectrum,high-only,low-only,random-matrix |
| full-spectrum: expected task count | 1 | 20 | 20 |
| full-spectrum: expected seed count | 1 | 6 | 6 |
| full-spectrum: no failed model runs | 1 | 0 | 0 |
| high-only: expected task count | 1 | 20 | 20 |
| high-only: expected seed count | 1 | 6 | 6 |
| high-only: no failed model runs | 1 | 0 | 0 |
| low-only: expected task count | 1 | 20 | 20 |
| low-only: expected seed count | 1 | 6 | 6 |
| low-only: no failed model runs | 1 | 0 | 0 |
| random-matrix: expected task count | 1 | 20 | 20 |
| random-matrix: expected seed count | 1 | 6 | 6 |
| random-matrix: no failed model runs | 1 | 0 | 0 |
| claim: high-only mean best score exceeds full-spectrum | 0 | -0.0466 | > 0 |
| claim: high-only is faster than full-spectrum | 1 | 1.1543 | > 1 |
| claim: low-only underperforms full-spectrum | 0 | 0.0190 | < 0 |
| claim: random-matrix has largest quality drop | 0 | high-only | random-matrix |
| claim: random-matrix is fastest | 0 | high-only | random-matrix |
| manuscript target: high-only mean_best_score | 1 | 0.6646 | 0.7542 |
| manuscript target: high-only delta_vs_full | 1 | -0.0466 | 0.0050 |
| manuscript target: high-only mean_elapsed_seconds | 1 | 177.2721 | 1331.3000 |
| manuscript target: high-only speedup_vs_full | 1 | 1.1543 | 1.1560 |
| manuscript target: high-only evals_per_second | 1 | 73.6495 | 15.0230 |
| manuscript target: full-spectrum mean_best_score | 1 | 0.7112 | 0.7492 |
| manuscript target: full-spectrum delta_vs_full | 1 | 0.0000 | 0.0000 |
| manuscript target: full-spectrum mean_elapsed_seconds | 1 | 204.6166 | 1538.6000 |
| manuscript target: full-spectrum speedup_vs_full | 1 | 1.0000 | 1.0000 |
| manuscript target: full-spectrum evals_per_second | 1 | 63.8071 | 12.9990 |
| manuscript target: low-only mean_best_score | 1 | 0.7303 | 0.7367 |
| manuscript target: low-only delta_vs_full | 1 | 0.0190 | -0.0125 |
| manuscript target: low-only mean_elapsed_seconds | 1 | 195.5661 | 1548.1000 |
| manuscript target: low-only speedup_vs_full | 1 | 1.0463 | 0.9940 |
| manuscript target: low-only evals_per_second | 1 | 66.7600 | 12.9190 |
| manuscript target: random-matrix mean_best_score | 1 | 0.7252 | 0.6469 |
| manuscript target: random-matrix delta_vs_full | 1 | 0.0140 | -0.1023 |
| manuscript target: random-matrix mean_elapsed_seconds | 1 | 197.4061 | 1256.1000 |
| manuscript target: random-matrix speedup_vs_full | 1 | 1.0365 | 1.2250 |
| manuscript target: random-matrix evals_per_second | 1 | 66.1378 | 15.9230 |
