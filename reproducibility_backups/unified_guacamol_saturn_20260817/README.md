# Unified GuacaMol + SATURN Reproducibility Layer

This folder records the first IBEX-side merge step toward a single SpectralMol
software tree that can reproduce both final test cases:

- GuacaMol 10-seed theta/MPO v3 benchmark.
- SATURN Table 8 theta-only NSGA-II v119 benchmark.

## Merge Status

The active IBEX working tree already matches the SATURN v119 source snapshot.
The final GuacaMol launch scripts in the active tree match the GuacaMol backup
scripts. Therefore, the safe merge is not a rollback of shared core files. The
safe merge is a unified benchmark profile layer that keeps the current newer
source code and exposes explicit, reproducible commands/settings for both final
test cases.

The GuacaMol backup shared-core files are older than the current SATURN v119
source. The current source keeps later improvements such as configurable
allowed elements, dynamic macro cache fixes, static target smiles, macro
expansion before token edits, token site scan mutation, and rank-biased target
theta sampling. Rolling these shared files back would risk breaking SATURN v119.

## Theta-Only Constraint

Both profile sets preserve theta-only optimization:

- GuacaMol disables phenotype proposals, BRICS crossover, and BRICS fragment
  replacement.
- SATURN runs through `auto_saturn_theta_tune_ibex.sh` with profile
  `table8_v118_strict_mode_cap_v119`, which uses `--nsga2-genotype theta`.

## Files

- `settings/guacamol_mpo_v3.env`: final GuacaMol settings.
- `settings/saturn_table8_v119.env`: final SATURN v119 settings.
- `settings/final_benchmark_profiles.env`: combined manifest pointing to both
  final profile settings, commands, expected values, and output roots.
- `commands/run_guacamol_mpo_v3_ibex.sh`: submit a fresh GuacaMol 10-seed run.
- `commands/aggregate_guacamol_mpo_v3_existing_ibex.sh`: analyze the completed
  GuacaMol final output root.
- `commands/run_saturn_table8_v119_ibex.sh`: submit a fresh SATURN v119 run.
- `commands/analyze_saturn_table8_v119_existing_ibex.sh`: analyze the completed
  SATURN final output root.
- `commands/verify_existing_results_ibex.sh`: verify both final existing result
  roots without submitting new jobs.
- `verification/`: historical v119 verification outputs retained for provenance.
  The final fresh manuscript rerun and publication tables are under
  `reproducibility/manuscript_2026/results/`.

## User-Friendly Entry Points

The repository root now includes two small wrappers for open-source use:

- `run_custom_spectralmol_ibex.sh`: dry-run or submit custom GuacaMol/SATURN
  jobs from editable env files.
- `run_reproducibility_profile_ibex.sh`: run or analyze the pinned final
  GuacaMol and SATURN profiles.

Editable custom-run examples are saved in:

```text
examples/custom_guacamol.env
examples/custom_saturn.env
```

Dry-run examples:

```bash
cd /home/colleoe/molevoDrugDiscovery_2/SpectralMol
CONFIG_FILE=examples/custom_guacamol.env bash run_custom_spectralmol_ibex.sh
CONFIG_FILE=examples/custom_saturn.env bash run_custom_spectralmol_ibex.sh
```

Submit after inspecting the printed command:

```bash
CONFIG_FILE=examples/custom_guacamol.env SUBMIT=1 bash run_custom_spectralmol_ibex.sh
```

## Fresh Reproduction Commands

Run GuacaMol:

```bash
cd /home/colleoe/molevoDrugDiscovery_2/SpectralMol
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/run_guacamol_mpo_v3_ibex.sh
```

Run SATURN:

```bash
cd /home/colleoe/molevoDrugDiscovery_2/SpectralMol
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/run_saturn_table8_v119_ibex.sh
```

## Analysis-Only Verification Commands

These commands use the already completed final output roots and do not submit
new Slurm jobs.

Verify both final result sets:

```bash
cd /home/colleoe/molevoDrugDiscovery_2/SpectralMol
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/verify_existing_results_ibex.sh
```

Verify GuacaMol only:

```bash
cd /home/colleoe/molevoDrugDiscovery_2/SpectralMol
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/aggregate_guacamol_mpo_v3_existing_ibex.sh
```

Verify SATURN only:

```bash
cd /home/colleoe/molevoDrugDiscovery_2/SpectralMol
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/analyze_saturn_table8_v119_existing_ibex.sh
```

## Expected GuacaMol Result

The final GuacaMol 10-seed comparison should reproduce the stored result:

```text
SpectralMol  n=10 mean=15.230485 std=0.176493
GraphGA      n=10 mean=14.574242 std=0.256197
Delta        n=10 mean=+0.656244 std=0.295991
SpectralMol wins: 10 / 10 seeds
```

Existing final output root:

```text
/ibex/scratch/colleoe/spectralMol/merged_fresh_repro_20260818_083101/guacamol_mpo_v3
```

## Expected SATURN Result

The final SATURN Table 8 v119 result should reproduce:

| Threshold | Successful replicates | Modes | Yield | QED | SA | MolWt |
|---:|---:|---:|---:|---:|---:|---:|
| Docking < -9 kcal/mol | 10/10 | 94.9 +/- 11.5 | 316.2 +/- 38.1 | 0.85 +/- 0.01 | 2.66 +/- 0.04 | 312.4 +/- 5.2 |
| Docking < -10 kcal/mol | 10/10 | 8.1 +/- 3.0 | 12.9 +/- 6.9 | 0.84 +/- 0.02 | 2.62 +/- 0.12 | 309.6 +/- 7.1 |

Existing final raw output root:

```text
/ibex/scratch/colleoe/spectralMol/merged_fresh_repro_20260818_083101/saturn_table8_v119/20260818_083655/table8_v118_strict_mode_cap_v119
```

## Next Verification Step

This unified layer was verified on IBEX after creation:

- `bash -n` passed for all unified commands and the benchmark launchers.
- Python `py_compile` passed for the shared core, GuacaMol, and SATURN modules.
- `verify_existing_results_ibex.sh` reproduced the expected GuacaMol and
  SATURN summaries from the existing final outputs.

Verification outputs are saved under:

```text
reproducibility_backups/unified_guacamol_saturn_20260817/verification
```

A short smoke run can still be launched if desired before a full fresh rerun.
