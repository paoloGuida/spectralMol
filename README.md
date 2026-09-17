# SpectralMol

SpectralMol evolves a Fourier theta representation and decodes it into
molecules for molecular optimization benchmarks. This repository includes the
open-source code path used for the final GuacaMol and SATURN reproductions, plus
small IBEX launch helpers for custom test cases.

## What Evolves

For the benchmark reproduction scripts, the evolved genotype is theta. The
provided launchers disable phenotype proposals and BRICS-based genotype changes
so that the optimization path remains theta-only.

## Quick Start On IBEX

Dry-run a custom GuacaMol-style optimization:

```bash
cd /home/$USER/molevoDrugDiscovery_2/SpectralMol
CONFIG_FILE=examples/custom_guacamol.env bash run_custom_spectralmol_ibex.sh
```

Dry-run a custom SATURN-style optimization:

```bash
cd /home/$USER/molevoDrugDiscovery_2/SpectralMol
CONFIG_FILE=examples/custom_saturn.env bash run_custom_spectralmol_ibex.sh
```

The wrapper prints the exact `sbatch` command by default. Submit the job by
adding `SUBMIT=1`:

```bash
CONFIG_FILE=examples/custom_guacamol.env SUBMIT=1 bash run_custom_spectralmol_ibex.sh
```

## Reproduce The Final Benchmarks

The publication branch is self-contained for the SpectralMol code, benchmark
configuration, initial populations, docking structures, compact result tables,
and manuscript figure-generation scripts. Start with:

```text
reproducibility/manuscript_2026/README.md
```

QuickVina2-GPU, OpenBabel, the SATURN checkout, and MolScore GraphGA examples
remain external dependencies; their exact revisions and binary checksums are
recorded in that guide.
The final GuacaMol and SATURN settings are pinned in:

```text
reproducibility_backups/unified_guacamol_saturn_20260817/settings/final_benchmark_profiles.env
```

Verify the existing completed result roots:

```bash
bash run_reproducibility_profile_ibex.sh all verify
```

Submit fresh reproduction jobs:

```bash
bash run_reproducibility_profile_ibex.sh guacamol_mpo_v3 run
bash run_reproducibility_profile_ibex.sh saturn_table8_v119 run
```

Analyze existing final outputs without submitting jobs:

```bash
bash run_reproducibility_profile_ibex.sh guacamol_mpo_v3 analyze
bash run_reproducibility_profile_ibex.sh saturn_table8_v119 analyze
```

## Custom GuacaMol Cases

Use `examples/custom_guacamol.env` as a starting point. Common edits are:

- `MODELS=spectralmol` for SpectralMol only, or `spectralmol,graphga` for a
  comparison.
- `BENCHMARK=GuacaMol` for the standard MolScore preset.
- `CUSTOM_BENCHMARK=/path/to/task_json_directory` for a custom MolScore task
  directory.
- `TASK_INDEXES_CSV=10` or `INCLUDE_CSV=TaskName` to restrict tasks.
- `SEED_SMILES_FILE=/path/to/seeds.smi` to set the initial population source.

The standard 20-task GuacaMol array launcher is still available for full
benchmark runs. Set `GUACAMOL_USE_TASK_ARRAY=1` in the custom env only when the
standard task index array mapping is what you want.

## Custom SATURN Cases

Use `examples/custom_saturn.env` as a starting point. Common edits are:

- `SATURN_ORACLE_TEMPLATE=/path/to/oracle_template.json`.
- `SATURN_REPO_ROOT=/path/to/saturn_repo`.
- `QUICKVINA_BINARY`, `RECEPTOR_FILE`, and `REFERENCE_LIGAND_FILE` for docking.
- `SEED_SMILES_FILE=/path/to/seeds.smi` for one shared seed pool, together with
  `USE_PER_SEED_SEED_SMILES=0`.
- `PER_SEED_SEED_SMILES_DIR=/path/to/seed_files` for files named
  `seed_0.smi`, `seed_1.smi`, and so on.

SATURN runs require a GPU node, OpenBabel, QuickVina2-GPU, and a working NVIDIA
OpenCL stack. The IBEX defaults in `sbatch_saturn_theta_nsga2_ibex.sh` match the
environment used for the final Table 8 reproduction.

## Main Files

- `run_custom_spectralmol_ibex.sh`: dry-run or submit custom GuacaMol/SATURN
  jobs on IBEX.
- `run_reproducibility_profile_ibex.sh`: run or analyze the two final
  reproducibility profiles.
- `examples/custom_guacamol.env`: editable GuacaMol custom-run settings.
- `examples/custom_saturn.env`: editable SATURN custom-run settings.
- `reproducibility/manuscript_2026/`: exact inputs, compact outputs, provenance,
  checksums, and scripts used to update the manuscript.
- `reproducibility_backups/unified_guacamol_saturn_20260817/`: final benchmark
  settings, commands, verification data, and result summaries.

