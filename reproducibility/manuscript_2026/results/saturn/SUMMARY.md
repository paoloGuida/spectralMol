# SATURN Theta NSGA-II Analysis

Input roots: /ibex/user/colleoe/spectralMol/merged_fresh_repro_20260818_083101/saturn_table8_v119/20260819_104640/table8_v118_strict_mode_cap_v119
Strategy: `theta_nsga2_multiobjective`
Expected seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

## Table 8 Replacement Rows

| Threshold | Method | Replicates | Modes | Yield | QED | SA | MolWt |
|---:|---|---:|---:|---:|---:|---:|---:|
| -9 | Saturn* | 10 | 38.0+/-13.0 | 83.0+/-33.0 | 0.80+/-0.05 | 2.10+/-0.10 | 343.3+/-6.2 |
| -9 | SpectralMol | 10 | 94.9+/-11.5 | 316.2+/-38.1 | 0.85+/-0.01 | 2.66+/-0.04 | 312.4+/-5.2 |
| -10 | Saturn* | 9 | 3.0+/-1.0 | 4.0+/-2.0 | 0.72+/-0.16 | 2.26+/-0.16 | 379.9+/-18.8 |
| -10 | SpectralMol | 10 | 8.1+/-3.0 | 12.9+/-6.9 | 0.84+/-0.02 | 2.62+/-0.12 | 309.6+/-7.1 |

## Generated Files

- `table8_manuscript_replacement.tsv`: Saturn reference rows plus updated SpectralMol rows.
- `table8_spectralmol.tsv`: SpectralMol-only Table 8 statistics.
- `table8_per_seed.tsv`: per-seed hit counts and scaffold modes.
- `all_molecules_standardized.tsv`: normalized molecule-level archive.
- `figure6_qed_sa_docking_hits.tsv`: QED-SA scatter inputs for docking-hit thresholds.
- `figure7_initial_final_population.tsv`: initial/final population projections when snapshots are available.
- `figure8_population_with_f1_flag.tsv`: final population with F1 membership flag.
- `figure8_f1_pareto_front.tsv`: first non-dominated front.
- `figure9_f1_crowding_distances.tsv`: crowding-distance data for F1.
