# SpectralMol manuscript reproducibility package

This directory ties the publication code to the exact inputs, pinned profiles,
compact outputs, and analysis scripts used for the August-September 2026
GuacaMol, SATURN, and frequency-mode ablation results.

## Invariants

- The evolved genotype is theta.
- GuacaMol phenotype proposals and BRICS genotype operations are disabled.
- SATURN uses `--nsga2-genotype theta` and jointly evaluates docking, QED, and
  synthetic accessibility.
- Parent-lineage fields are observability metadata; they do not add an evolved
  variable or alter theta proposal behavior.

## Reported results

### GuacaMol, 10 matched seeds (7-16)

| Method | Aggregate mean +/- SD | Seed wins |
|---|---:|---:|
| SpectralMol | 15.230485 +/- 0.176493 | 10/10 |
| GraphGA | 14.574242 +/- 0.256197 | 0/10 |
| Paired difference | +0.656244 +/- 0.295991 | 10/10 positive |

The paired sign-test value is `p=9.8e-4`.

### SATURN Table 8, 1000 oracle calls, 10 seeds

| Docking threshold | Successful replicates | Modes | Yield | QED | SA | MolWt |
|---|---:|---:|---:|---:|---:|---:|
| < -9 kcal/mol | 10/10 | 94.9 +/- 11.5 | 316.2 +/- 38.1 | 0.85 +/- 0.01 | 2.66 +/- 0.04 | 312.4 +/- 5.2 |
| < -10 kcal/mol | 10/10 | 8.1 +/- 3.0 | 12.9 +/- 6.9 | 0.84 +/- 0.02 | 2.62 +/- 0.12 | 309.6 +/- 7.1 |

### Frequency-mode ablation

This separate mechanistic study uses 20 tasks, 6 seeds, and 20,000 evaluations
per task. It is not the 10-seed GraphGA comparison.

| Condition | Mean best score | Delta vs full | Speedup |
|---|---:|---:|---:|
| low-only | 0.730258 | +0.019037 | 1.046x |
| random-matrix | 0.725250 | +0.014029 | 1.036x |
| full-spectrum | 0.711221 | 0 | 1.000x |
| high-only | 0.664649 | -0.046572 | 1.154x |

The explicit parent-lineage outputs support a cautious interpretation:
frequency restrictions bias parent-child molecular changes, but the mapping
between a frequency band and one chemical operation is not universal.

## Fresh runs on IBEX

Create the conda environment from `spectralMol/environment_exported.yml` or
`spectralMol/environment.yml`. Install the external dependencies listed below,
then run from the repository root:

```bash
bash run_reproducibility_profile_ibex.sh guacamol_mpo_v3 run
bash run_reproducibility_profile_ibex.sh saturn_table8_v119 run
```

Pinned settings are under
`reproducibility_backups/unified_guacamol_saturn_20260817/settings`.

For the ablation, decompress the included ChEMBL input and override the path:

```bash
gzip -dk reproducibility/manuscript_2026/inputs/guacamol/chembl.filtered.smi.gz
SEED_SMILES_FILE="$PWD/reproducibility/manuscript_2026/inputs/guacamol/chembl.filtered.smi" \
  bash sbatch_guacamol_frequency_ablation_ibex.sh
```

## External dependencies

- SATURN: `https://github.com/schwallergroup/saturn.git`, commit
  `3aea130158c488050426a91716b8c6ff34f05473`.
- MolScore examples: `https://github.com/MorganCThomas/MolScore_examples.git`,
  commit `92bb08abb1584cd04f6ee21d894d17de9be56d7b`.
- QuickVina2-GPU SHA256:
  `d1d0b0f42025ab9c6903c88dbbdfabf6337f91da48790d2c2416ea24745b0411`.
- OpenBabel 3.1.0 executable SHA256:
  `763574be7ac40f5ece8ca4b186aaf70246915614447eca518b3be87b901d724e`.
- NVIDIA OpenCL runtime; verified runs used CUDA 11.8 and
  `/lib64/libOpenCL.so.1`.

The repository includes the SATURN seed sets, receptor, and reference ligand.
Set `SATURN_REPO_ROOT`, `QUICKVINA_BINARY`, or
`MOLSCORE_SATURN_OBABEL_BINARY` when installed elsewhere.

## Included data

- `inputs/guacamol`: exact shared GuacaMol population and compressed ChEMBL
  ablation source.
- `inputs/saturn`: ten initial populations plus docking structures.
- `results/guacamol`: aggregate, per-seed, per-task, trajectory, and top-200
  tables.
- `results/saturn`: Table 8, molecule-level, Pareto, crowding, and Figure 10/11
  source tables.
- `results/frequency_ablation`: score, runtime, feature-change, and explicit
  parent-lineage analyses; large row tables are gzip-compressed.
- `scripts`: manuscript table and Figure 3-11 generators.
- `provenance`: jobs, versions, and checksums.

## Validation

```bash
bash reproducibility/manuscript_2026/validate_release.sh
```

This checks shell and Python syntax, required files, theta-only settings, and
SHA256 hashes. Full numerical reproduction requires the HPC/GPU dependencies.

Original completed roots:

- GuacaMol: `/ibex/scratch/colleoe/spectralMol/merged_fresh_repro_20260818_083101/guacamol_mpo_v3`
- SATURN: `/ibex/scratch/colleoe/spectralMol/merged_fresh_repro_20260818_083101/saturn_table8_v119/20260818_083655/table8_v118_strict_mode_cap_v119`

Those private paths are provenance only; compact manuscript data is included.
