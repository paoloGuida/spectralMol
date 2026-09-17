# Unified GuacaMol + SATURN Reproducibility Layer

This folder defines one portable SpectralMol software tree for both final test
cases:

- GuacaMol 10-seed theta/MPO v3 benchmark.
- SATURN Table 8 theta-only NSGA-II v119 benchmark.

The profile layer keeps benchmark settings, exact repository-local inputs,
expected values, local execution commands, and optional Slurm submission
commands together. No site-specific filesystem layout is assumed.

## Theta-Only Constraint

Both profile sets preserve theta-only optimization:

- GuacaMol disables phenotype proposals, BRICS crossover, and BRICS fragment
  replacement.
- SATURN runs through `auto_saturn_theta_tune.sh` with profile
  `table8_v118_strict_mode_cap_v119`, which uses `--nsga2-genotype theta`.

## Files

- `settings/guacamol_mpo_v3.env`: final GuacaMol settings.
- `settings/saturn_table8_v119.env`: final SATURN v119 settings.
- `settings/final_benchmark_profiles.env`: combined manifest pointing to both
  final profile settings, commands, expected values, and output roots.
- `commands/run_guacamol_mpo_v3.sh`: run a fresh GuacaMol 10-seed benchmark locally or through Slurm.
- `commands/aggregate_guacamol_mpo_v3.sh`: print bundled GuacaMol manuscript tables.
- `commands/run_saturn_table8_v119.sh`: run a fresh SATURN v119 benchmark locally or through Slurm.
- `commands/analyze_saturn_table8_v119.sh`: print bundled SATURN results or analyze `SATURN_RAW_OUTPUT`.
- `commands/verify_bundled_results.sh`: validate checksums and print both final result summaries.
- `verification/`: historical v119 verification outputs retained for provenance.
  The final fresh manuscript rerun and publication tables are under
  `reproducibility/manuscript_2026/results/`.

## User-Friendly Entry Points

Run custom cases locally with editable environment files:

```bash
cd /path/to/SpectralMol
CONFIG_FILE=examples/custom_guacamol.env bash run_custom_spectralmol.sh
CONFIG_FILE=examples/custom_saturn.env bash run_custom_spectralmol.sh
```

Set `EXECUTION_BACKEND=slurm SUBMIT=1` when a Slurm scheduler is available.
The pinned manuscript profiles use the same local-by-default convention:

```bash
bash run_reproducibility_profile.sh guacamol_mpo_v3 run
bash run_reproducibility_profile.sh saturn_table8_v119 run
bash run_reproducibility_profile.sh all verify
```

The `verify` action uses bundled data and does not launch a benchmark.

## Expected GuacaMol Result

The final GuacaMol 10-seed comparison should reproduce the stored result:

```text
SpectralMol  n=10 mean=15.230485 std=0.176493
GraphGA      n=10 mean=14.574242 std=0.256197
Delta        n=10 mean=+0.656244 std=0.295991
SpectralMol wins: 10 / 10 seeds
```

Bundled tables are in `reproducibility/manuscript_2026/results/guacamol`.

## Expected SATURN Result

The final SATURN Table 8 v119 result should reproduce:

| Threshold | Successful replicates | Modes | Yield | QED | SA | MolWt |
|---:|---:|---:|---:|---:|---:|---:|
| Docking < -9 kcal/mol | 10/10 | 94.9 +/- 11.5 | 316.2 +/- 38.1 | 0.85 +/- 0.01 | 2.66 +/- 0.04 | 312.4 +/- 5.2 |
| Docking < -10 kcal/mol | 10/10 | 8.1 +/- 3.0 | 12.9 +/- 6.9 | 0.84 +/- 0.02 | 2.62 +/- 0.12 | 309.6 +/- 7.1 |

Bundled tables are in `reproducibility/manuscript_2026/results/saturn`.

## Verification

Run `bash run_reproducibility_profile.sh all verify` to check required files,
Python and shell syntax, compressed archives, theta-only constraints, and all
recorded SHA256 checksums. Historical verification files remain under
`verification/`; the publication tables are under
`reproducibility/manuscript_2026/results/`.
