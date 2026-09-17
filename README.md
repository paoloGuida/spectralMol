# SpectralMol

SpectralMol evolves a Fourier theta representation and decodes it into
molecules for molecular optimization benchmarks. This repository includes the
open-source code path used for the final GuacaMol and SATURN reproductions, plus
portable local and optional Slurm launchers for custom test cases.

## What Evolves

For the benchmark reproduction scripts, the evolved genotype is theta. The
provided launchers disable phenotype proposals and BRICS-based genotype changes
so that the optimization path remains theta-only.

## Quick Start

Create the environment and activate it:

```bash
conda env create -f spectralMol/environment.yml
conda activate spectralmol
```

Run a custom GuacaMol-style optimization locally:

```bash
CONFIG_FILE=examples/custom_guacamol.env bash run_custom_spectralmol.sh
```

Run a custom SATURN-style optimization locally after installing the external
docking dependencies described below:

```bash
CONFIG_FILE=examples/custom_saturn.env bash run_custom_spectralmol.sh
```

Local execution is the default. On a system with Slurm, set
`EXECUTION_BACKEND=slurm SUBMIT=1` to submit the same configuration.

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

Validate the bundled code, inputs, checksums, and manuscript results:

```bash
bash run_reproducibility_profile.sh all verify
```

Start fresh reproduction runs locally:

```bash
bash run_reproducibility_profile.sh guacamol_mpo_v3 run
bash run_reproducibility_profile.sh saturn_table8_v119 run
```

Print the bundled final result summaries without running jobs:

```bash
bash run_reproducibility_profile.sh guacamol_mpo_v3 analyze
bash run_reproducibility_profile.sh saturn_table8_v119 analyze
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

SATURN runs require an NVIDIA GPU, OpenBabel, QuickVina2-GPU, and a working
OpenCL stack. Configure their locations in `examples/custom_saturn.env`; no
site-specific filesystem layout or scheduler is required.

## Main Files

- `run_custom_spectralmol.sh`: run custom GuacaMol/SATURN jobs locally or
  submit them to Slurm.
- `run_reproducibility_profile.sh`: run or analyze the two final
  reproducibility profiles.
- `run_frequency_ablation.sh`: run the manuscript frequency ablation locally
  or submit it to Slurm.
- `examples/custom_guacamol.env`: editable GuacaMol custom-run settings.
- `examples/custom_saturn.env`: editable SATURN custom-run settings.
- `reproducibility/manuscript_2026/`: exact inputs, compact outputs, provenance,
  checksums, and scripts used to update the manuscript.
- `reproducibility_backups/unified_guacamol_saturn_20260817/`: final benchmark
  settings, commands, verification data, and result summaries.

