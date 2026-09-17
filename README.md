# SpectralMol

SpectralMol optimizes molecules by evolving a Fourier-parameterized `theta`
genotype and decoding it into SELFIES/SMILES. This repository contains the
code, pinned configurations, initial populations, compact outputs, and analysis
scripts used for the reported GuacaMol, SATURN, and frequency-mode experiments.

The public workflow is driven by one Python entry point and does not require a
particular scheduler, filesystem layout, username, or compute site.

## Scientific Scope

- **GuacaMol:** 20 goal-directed tasks, 10 matched seeds, SpectralMol versus
  GraphGA.
- **SATURN:** theta-only NSGA-II optimization of docking, QED, and synthetic
  accessibility at 1,000 oracle calls.
- **Frequency ablation:** full-spectrum, high-only, low-only, and random-matrix
  conditions with parent-lineage analyses.

The portability refactor does not change objectives, scoring, evolution,
docking parameters, filtering, seeds, or numerical calculations. Pinned
environment snapshots in `configs/` replace the former launch-layer defaults.

## Installation

### Python virtual environment

```bash
git clone https://github.com/paoloGuida/spectralMol.git
cd spectralMol
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[benchmarks,test]
```

Python 3.9 or newer is supported. Python 3.11 is recommended. For SATURN,
OpenBabel is most reliably installed with Conda:

```bash
conda env create -f environment.yml
conda activate spectralmol
```

## Quick Test

```bash
python run.py --help
python run.py --config configs/test.toml
```

The smoke test runs on CPU, constructs a small theta population, mutates theta,
validates the resulting arrays and molecules, and writes
`results/smoke/smoke_test.json`.

Run the release and checksum audit with:

```bash
python run.py --config configs/verify.toml
```

## Reproducing the Main Results

### GuacaMol

Install the benchmark extras and clone the public MolScore examples repository
next to this checkout, or update `external.molscore_examples` in the TOML file.

```bash
pip install -e .[benchmarks]
git clone https://github.com/MorganCThomas/MolScore_examples.git ../MolScore_examples
python run.py --config configs/guacamol_manuscript.toml
```

The pinned configuration runs seeds 7-16 and all 20 tasks. Each task receives
the exact task-specific theta profile captured in
`configs/guacamol_task_profiles.json`. To inspect one task before a full run:

```bash
python run.py --config configs/guacamol_manuscript.toml --seed 7 --task 10 --dry-run
```

### SATURN Table 8

SATURN reproduction requires a supported NVIDIA GPU/OpenCL runtime,
QuickVina2-GPU, OpenBabel, and the public SATURN repository. Configure the three
external paths in `configs/saturn_table8.toml`, then run:

```bash
pip install -e .[saturn]
git clone https://github.com/schwallergroup/saturn.git ../saturn
python run.py --config configs/saturn_table8.toml
```

The exact v119 environment is stored in
`configs/saturn_table8_v119_environment.json`. The runner enforces
`--nsga2-genotype theta`; phenotype proposals and BRICS genotype operations are
disabled. Use `--seed 0 --dry-run` to validate paths and inspect the command
without consuming oracle calls.

### Frequency-Mode Ablation

```bash
python run.py --config configs/frequency_ablation.toml
```

The bundled compressed ChEMBL input is decompressed into the selected output
directory when needed. A single-condition diagnostic can be made by copying the
TOML file and reducing its `conditions`, `seeds`, and `tasks` arrays.

### Bundled Results

Print the principal manuscript tables without launching experiments:

```bash
python run.py --config configs/results.toml
```

Expected headline results are:

| Benchmark | SpectralMol result |
|---|---:|
| GuacaMol aggregate, 10 seeds | 15.230486 +/- 0.176493 |
| GuacaMol paired delta vs GraphGA | +0.656244 +/- 0.295991 |
| SATURN docking < -9: modes (yield) | 94.9 +/- 11.5 (316.2 +/- 38.1) |
| SATURN docking < -10: modes (yield) | 8.1 +/- 3.0 (12.9 +/- 6.9) |

## Configuration

All user-editable settings are TOML values. Paths may be absolute or relative
to the repository root. The primary profiles are:

- `configs/default.toml`: lightweight smoke test.
- `configs/guacamol_manuscript.toml`: final 10-seed GuacaMol comparison.
- `configs/saturn_table8.toml`: final 10-seed SATURN profile.
- `configs/frequency_ablation.toml`: manuscript frequency ablation.
- `examples/custom_guacamol.toml`: small editable GuacaMol example.
- `examples/custom_saturn.toml`: editable SATURN example.

Common command-line overrides are `--output-dir`, `--seed`, `--task`,
`--device`, and `--dry-run`. Output directories are created automatically.
Run metadata records the resolved configuration, Python version, timestamp,
and exact commands in `run_metadata.json`.

## Expected Outputs

- GuacaMol: one directory per seed/task containing model outputs and comparison
  summaries, plus top-level run metadata.
- SATURN: one run directory per seed and an `analysis/` directory containing
  Table 8, Pareto-front, and molecule-level tables.
- Frequency ablation: one directory per frequency condition, then seed/task.
- Smoke test: `smoke_test.json`.

Reference outputs used by the manuscript are versioned under
`reproducibility/manuscript_2026/results/`.

## Repository Structure

```text
run.py                         Portable experiment CLI
configs/                       Pinned TOML and environment snapshots
examples/                      Small editable configurations
spectralMol/core/              Fourier theta, decoding, and oracle code
spectralMol/benchmarks/        GuacaMol and SATURN Python entry points
spectralMol/tests/             Unit and lightweight integration tests
reproducibility/manuscript_2026/
                               Inputs, results, provenance, figure scripts
scripts/validate_release.py    Portability/checksum validator
```

## Hardware and Runtime

- The smoke test and GuacaMol workflow support CPU execution.
- GPU use in optional diversity calculations is auto-detected; CPU fallback is
  used when available.
- The publication SATURN docking workflow requires a supported GPU and OpenCL.
- Full benchmark reproductions are expensive. The test configuration is meant
  only to verify installation and is not a scientific benchmark.

## External Software and Data

- SATURN repository, revision `3aea130158c488050426a91716b8c6ff34f05473`.
- MolScore examples, revision `92bb08abb1584cd04f6ee21d894d17de9be56d7b`.
- QuickVina2-GPU for SATURN docking (not redistributed here).
- OpenBabel for ligand conversion.
- An NVIDIA OpenCL runtime for the SATURN publication profile.

The exact seed sets, receptor, reference ligand, shared GuacaMol population,
and compressed ablation input are included. External executables are never
located through hard-coded installation paths; configure them in TOML or put
them on `PATH`. Missing requirements produce an explicit preflight error.

### Bundled data

- `chembl.filtered.smi.gz` contains 524,186 SMILES used by the frequency
  ablation; the runner expands it inside the selected output directory.
- The GuacaMol shared population contains 256 molecules.
- The SATURN input directory contains ten disjoint 256-molecule seed sets;
  selection thresholds and the deterministic shuffle seed are recorded in
  `inputs/saturn/seed_sets/metadata.json`.
- The prepared receptor and reference ligand needed by the reported SATURN
  case are bundled. The original ranked ZINC source table is not redistributed
  because it is not needed once the selected seed sets have been recorded.

Users should review the upstream ChEMBL, ZINC, MolScore, SATURN, OpenBabel, and
QuickVina terms for their intended use. The repository license covers this
software and does not replace third-party licenses.

### Legacy launcher migration

The former custom-run, reproducibility, aggregation, frequency-ablation, and
scheduler wrappers are replaced by the `run.py` workflows and TOML profiles.
The development-time automatic profile search was retired after selecting the
published v119 profile; its final values are preserved exactly in
`configs/saturn_table8_v119_environment.json`.

## Reproducibility Notes

Random seeds and experiment settings are pinned in `configs/`. External
docking software, GPU drivers, thread scheduling, and floating-point libraries
may introduce small platform-dependent differences. Reference package versions
and executable checksums are recorded in
`reproducibility/manuscript_2026/provenance/`.

## Development and Testing

```bash
pip install -e .[test]
pytest -q -m unit
python scripts/validate_release.py
```

CI intentionally runs only lightweight checks and never launches full docking
or goal-directed benchmark campaigns.

## License

The source is provided under the MIT License. Before tagging a formal release,
replace the clearly marked copyright-holder placeholder in `LICENSE`.

## Citation

See `CITATION.cff`. Publication metadata is intentionally marked for completion
when the associated manuscript receives its final bibliographic record.
