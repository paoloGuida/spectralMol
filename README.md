# molevoDrugDiscovery

An evolutionary drug discovery framework that optimises molecules against multi-parameter oracles using SELFIES-based mutation, BRICS fragmentation, and optional docking-guided selection. benchmarks are provided for the GuacaMol preset suite (via MolScore) and the SATURN docking-guided benchmark.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Environment Setup](#environment-setup)
3. [Core Package Overview](#core-package-overview)
4. [GuacaMol Benchmark](#guacamol-benchmark)
   - [Single-model evolution run](#1-single-model-evolution-run)
   - [Multi-model comparison run](#2-multi-model-comparison-run)
   - [Compare existing runs post-hoc](#3-compare-existing-runs-post-hoc)
   - [External model wrappers](#4-external-model-wrappers)
   - [Bootstrapping scoring environments](#5-bootstrapping-scoring-environments)
5. [Saturn Benchmark](#saturn-benchmark)
   - [Scalar vs NSGA-II comparison](#scalar-vs-nsga-ii-comparison)
6. [Configuration Reference](#configuration-reference)
7. [Output Files](#output-files)
8. [Matrix Results (May 2026)](#matrix-results-may-2026)
9. [Extending the Framework](#extending-the-framework)

---

## Repository Layout

```
molevoDrugDiscovery/
├── core/                          # Shared Python package (importable as `core`)
│   ├── config.py                  # Central defaults (all CLI flags mirror these)
│   ├── vocab.py                   # SELFIES vocabulary & motif macro definitions
│   ├── decoder.py                 # Molecule decoder from Theta parameter space
│   ├── embedding.py               # Structured embedding matrix builder
│   ├── fourier_theta.py           # Fourier parameter space + mutation operators
│   ├── smiles_theta_encoder.py    # SMILES → SELFIES token encoder utilities
│   ├── reports.py                 # GuacaMol-style report generation
│   ├── oracles/                   # Modular oracle scoring components
│   │   ├── oracle.py              # Oracle class (multi-component aggregation)
│   │   ├── oracle_component.py    # Abstract base oracle component
│   │   ├── dataclass.py           # Oracle configuration dataclasses
│   │   ├── utils.py               # Oracle component factory
│   │   ├── docking/               # QuickVina2, QuickVina2-GPU, DockStream, GEAM
│   │   ├── physchem/              # MW, QED, HBA, HBD, TPSA, rings, rot. bonds...
│   │   ├── similarity/            # Tanimoto, Jaccard distance
│   │   ├── structural/            # SMARTS alerts, substructure matching
│   │   ├── synthesizability/      # SA score (fpscores.pkl.gz included)
│   │   ├── xtb/                   # xTB quantum-chemistry descriptors
│   │   └── reward_aggregator/     # Reward aggregation strategies
│   ├── diversity_filter/          # Scaffold-based diversity filter
│   └── utils/
│       └── chemistry_utils.py     # Canonicalization, Bemis-Murcko scaffolds
│
├── benchmarks/
│   ├── Guacamol/                  # GuacaMol / MolScore benchmark runners
│   │   ├── evolve_vs_molscore_benchmark.py   # Single-model evolution + baseline
│   │   ├── benchmark_compare_models.py       # Multi-model parallel comparison
│   │   ├── compare_local_graphga_runs.py     # Analyse existing run CSVs
│   │   ├── run_graphga_example_wrapper.py    # Wrapper for external GraphGA
│   │   ├── run_smiles_rnn_example.py         # Wrapper for external SMILES-RNN
│   │   ├── run_crem_example.py               # Wrapper for external CReM
│   │   ├── bootstrap_scoring_envs.py         # Pre-create MolScore scoring envs
│   │   ├── model_specs.json                  # Model registry for comparison runner
│   │   └── benchmark_descriptions.txt        # Human-readable task descriptions
│   │
│   └── Saturn/                    # SATURN docking-guided benchmark runners
│       ├── config.py              # Saturn-specific configuration overrides
│       ├── evolve_vs_molscore_benchmark.py   # Core evolution engine (Saturn flavour)
│       ├── compare_scalar_vs_nsga2_saturn.py # Scalar vs NSGA-II comparison
│       ├── guacamol_reports.py               # Report writer for Saturn runs
│       └── table2_r_sa_qed_oracle_template.json  # Default oracle config (7UVU target)
│
└── Saturn_TestCase/               # Reference run artefacts for run 46364391
```

---

## Environment Setup

### 1. Create the conda environment

```bash
conda create -n molevoDrugDiscovery python=3.10
conda activate molevoDrugDiscovery
```

### 2. Install compiled dependencies via conda-forge

RDKit and OpenBabel must be installed through conda to get correct native binaries:

```bash
conda install -c conda-forge numpy pandas scipy joblib tqdm rdkit openbabel
```

### 3. Install Python packages via pip

```bash
pip install selfies guacamol morfeus-ml molscore
```

> **Note on `molbloom`:** `molscore` optionally depends on `molbloom`, which requires compiling a C extension.
> Make sure a C compiler is available (GCC on Linux, Visual Studio Build Tools on Windows, Xcode Command Line Tools on macOS), then:
> ```bash
> pip install molbloom
> ```

### 4. (Optional) Install additional packages

```bash
pip install torch dask distributed dask-jobqueue flask streamlit streamlit-plotly-events plotly seaborn
```

### 5. (Optional) Docking dependencies

For Saturn oracle scoring with QuickVina2-GPU you will need:

- A compiled **QuickVina2-GPU-2.1** binary (see the [Vina-GPU GitHub repository](https://github.com/DeltaGroupNJUPT/Vina-GPU))
- The **receptor PDBQT** and **reference ligand PDB** for your target (7UVU example files are referenced in `Saturn_TestCase/`)

### 6. Make the `Core` package importable

The benchmark scripts automatically insert the repository root into `sys.path`, so no installation step is required. If you import `core` from a custom script, either:

- Run from the repository root, **or**
- Export `PYTHONPATH`:

  **Linux / macOS (bash/zsh):**
  ```bash
  export PYTHONPATH="/path/to/molevoDrugDiscovery:$PYTHONPATH"
  ```

  **Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH = "C:\path\to\molevoDrugDiscovery;$env:PYTHONPATH"
  ```

  **Windows (Command Prompt):**
  ```cmd
  set PYTHONPATH=C:\path\to\molevoDrugDiscovery;%PYTHONPATH%
  ```

---

## Core Package Overview

| Module | Purpose |
|---|---|
| `core.config` | All runtime defaults. Every CLI flag has a matching constant here. |
| `core.vocab` | SELFIES token vocabulary and motif macro expansions (e.g. `[PHENYL]`, `[PIPER]`). |
| `core.fourier_theta` | Fourier parameter-space basis; `mutate_Theta`, `make_initial_population`. |
| `core.embedding` | Builds the structured embedding matrix `E` that maps Theta to molecule token probabilities. |
| `core.decoder` | Decodes a `Theta` matrix into a SMILES string via the embedding. |
| `core.smiles_theta_encoder` | Encodes a SMILES into SELFIES tokens; applies macro compression. |
| `core.reports` | Writes GuacaMol-style benchmark HTML/CSV reports from run directories. |
| `core.oracles.oracle` | `Oracle` class — composes multiple `OracleComponent` instances and applies reward aggregation and diversity filter. |
| `core.utils.chemistry_utils` | `canonicalize_smiles`, `canonicalize_smiles_batch`, `get_bemis_murcko_scaffold`, SMILES randomization. |

---

## GuacaMol Benchmark

All commands below are run from the **repository root** with the `molevoDrugDiscovery` environment active.

### 1. Single-model evolution run

Runs the built-in local-evolution strategy against every task in the chosen MolScore benchmark preset, then optionally compares it to a random-mutation baseline.

```bash
python benchmarks/Guacamol/evolve_vs_molscore_benchmark.py \
  --benchmark GuacaMol \
  --budget 10000 \
  --population-size 256 \
  --batch-size 64 \
  --max-generations 50 \
  --tournament-k 8 \
  --seed 7 \
  --seed-smiles-file /path/to/chembl24_canon_train.smiles \
  --seed-pool-size 2000 \
  --output-dir outputs/runs
```

#### Full flag reference

| Flag | Default | Description |
|---|---|---|
| `--benchmark` | `GuacaMol` | MolScore preset name (`GuacaMol`, `MolOpt`, …) |
| `--custom-benchmark` | *(none)* | Path to a directory of task JSON files; overrides `--benchmark` |
| `--include` | *(all)* | Comma-separated task names to run |
| `--exclude` | *(none)* | Comma-separated task names to skip |
| `--budget` | `100000000` | Max molecule evaluations per task |
| `--population-size` | `256` | Population size |
| `--batch-size` | `256` | Evaluations per generation |
| `--max-generations` | `50` | Hard generation cap (0 = unlimited) |
| `--tournament-k` | `8` | Tournament selection size |
| `--elite-fraction` | `0.15` | Fraction of population preserved each generation |
| `--immigrant-fraction` | `0.10` | Fraction injected from the seed pool each generation |
| `--parent-pool-fraction` | `0.50` | Fraction of population eligible as parents |
| `--stagnation-patience` | `12` | Generations without improvement before increasing exploration |
| `--stagnation-mutation-boost` | `2` | Extra mutation depth during stagnation |
| `--seed` | `7` | Global random seed |
| `--seed-smiles-file` | *(repo default)* | SMILES file for initial population seeds |
| `--seed-pool-size` | `2000` | Max seeds loaded from file |
| `--skip-random-baseline` | `False` | Skip the random-mutation comparison run |
| `--output-dir` | `outputs/runs` | Root output directory |

#### Using a SATURN oracle as the scoring backend

The same script can score molecules against a SATURN oracle instead of a MolScore preset:

```bash
python benchmarks/Guacamol/evolve_vs_molscore_benchmark.py \
  --objective-backend saturn \
  --saturn-repo-root Core \
  --saturn-oracle-config benchmarks/Saturn/table2_r_sa_qed_oracle_template.json \
  --budget 1000
```

| Flag | Description |
|---|---|
| `--objective-backend` | `molscore` (default) or `saturn` |
| `--saturn-repo-root` | Path to the oracle root directory (defaults to `Core/`) |
| `--saturn-oracle-config` | Oracle JSON config path |
| `--saturn-oracle-config-key` | Key containing the oracle payload (default `"oracle"`) |
| `--saturn-task-name` | Label in output files (default `"saturn_objective"`) |
| `--saturn-apply-diversity-penalty` | Enable diversity filter penalty across calls |
| `--saturn-diversity-bucket-size` | Bucket size for diversity filter (default `10`) |
| `--saturn-invalid-score` | Score for invalid molecules (default `0.0`) |
| `--saturn-allow-oracle-repeats` | Override config to allow repeat calls on identical molecules |
| `--saturn-disallow-oracle-repeats` | Override config to disallow repeat calls |

---

### 2. Multi-model comparison run

Runs multiple generators (local evolution, GraphGA, SMILES-RNN, CReM) under identical benchmark settings and produces side-by-side comparison tables.

```bash
python benchmarks/Guacamol/benchmark_compare_models.py \
  --benchmark GuacaMol \
  --budget 10000 \
  --generations 50 \
  --population-size 256 \
  --batch-size 64 \
  --seeds 7,8,9 \
  --models local_evolution,graphga_example \
  --model-spec-file benchmarks/Guacamol/model_specs.json \
  --examples-root /path/to/MolScore_examples \
  --output-dir outputs/comparisons
```

#### Full flag reference

| Flag | Default | Description |
|---|---|---|
| `--benchmark` | `GuacaMol` | MolScore preset name |
| `--custom-benchmark` | *(none)* | Custom benchmark config directory; overrides `--benchmark` |
| `--include` / `--exclude` | *(all)* / *(none)* | Task filter lists |
| `--budget` | `100000000` | Evaluations per task |
| `--generations` | `50` | Generation target for each model |
| `--population-size` | `256` | Population size for local_evolution |
| `--batch-size` | `256` | Batch size for local_evolution |
| `--local-evo-tournament-k` | `8` | Tournament size |
| `--local-evo-elite-fraction` | `0.15` | Elite fraction |
| `--local-evo-immigrant-fraction` | `0.10` | Immigrant fraction |
| `--local-evo-parent-pool-fraction` | `0.50` | Parent pool fraction |
| `--local-evo-stagnation-patience` | `12` | Stagnation patience |
| `--local-evo-stagnation-mutation-boost` | `2` | Extra mutation depth on stagnation |
| `--seed-smiles-file` | *(repo default)* | Seed SMILES file for local_evolution |
| `--seed-pool-size` | `2000` | Seed pool size |
| `--seeds` | `7,8,9` | Comma-separated seeds for repeated runs |
| `--models` | `local_evolution` | Comma-separated model keys from `model_specs.json` |
| `--model-spec-file` | *(repo default)* | JSON registry describing each runnable model |
| `--examples-root` | *(none)* | Path to local `MolScore_examples` clone |
| `--python-bin` | *(active python)* | Python interpreter override |
| `--equal-initial-population` | `True` | Share a per-seed init population file across all models |
| `--continue-on-error` | `False` | Continue with other models/seeds if one fails |
| `--dry-run` | `False` | Print commands without executing |
| `--graphga-n-jobs` | `1` | Parallel workers for GraphGA |
| `--crem-ncpu` | `1` | CPU workers for CReM |
| `--crem-replacements` | `1000` | Max replacements per CReM step |
| `--smiles-rnn-device` | `auto` | Device for SMILES-RNN (`cpu`, `cuda`, `auto`) |
| `--bootstrap-scoring-envs` | `False` | Pre-create MolScore scoring conda envs before running |
| `--output-dir` | `outputs/comparisons` | Root output directory |

---

### 3. Compare existing runs post-hoc

If you have already completed runs and want to analyse the CSVs without re-running:

```bash
python benchmarks/Guacamol/compare_local_graphga_runs.py \
  --local-dir outputs/runs/local_evolution_seed7 \
  --graphga-dir outputs/runs/graphga_seed7 \
  --output-dir outputs/analysis
```

---

### 4. External model wrappers

These wrappers handle compatibility shims, environment variable injection, and script discovery for third-party generators. They are invoked automatically by `benchmark_compare_models.py` but can also be run standalone.

#### GraphGA

```bash
python benchmarks/Guacamol/run_graphga_example_wrapper.py \
  --graphga-script /path/to/MolScore_examples/GraphGA/graph_ga.py \
  --benchmark GuacaMol \
  --budget 10000 \
  --output-dir outputs/graphga_run
```

#### SMILES-RNN

```bash
python benchmarks/Guacamol/run_smiles_rnn_example.py \
  --smiles-rnn-root /path/to/MolScore_examples/SMILES-RNN \
  --benchmark GuacaMol \
  --budget 10000 \
  --device cpu
```

#### CReM

CReM requires a fragment database built from your SMILES file. The wrapper handles building the database automatically:

```bash
python benchmarks/Guacamol/run_crem_example.py \
  --crem-root /path/to/MolScore_examples/CReM \
  --smiles-file /path/to/chembl24_canon_train.smiles \
  --db-path outputs/crem_fragments.db \
  --benchmark GuacaMol \
  --budget 10000
```

---

### 5. Bootstrapping scoring environments

Some MolScore tasks (e.g. PIDGIN, MolOpt) require their own conda environments. Bootstrap them before a long benchmark run to avoid failures mid-task:

```bash
python benchmarks/Guacamol/bootstrap_scoring_envs.py \
  --targets pidgin,ms_molopt \
  --clone-from molevoDrugDiscovery
```

| Flag | Description |
|---|---|
| `--targets` | Comma-separated targets: `pidgin`, `ms_molopt` |
| `--clone-from` | Source env to clone from if YAML creation fails |
| `--strict` | Fail immediately instead of attempting clone fallback |
| `--dry-run` | Print actions without executing |

---

## Saturn Benchmark

The Saturn benchmark focuses on docking-guided multi-parameter optimisation against a protein target (default: 7UVU). It compares a scalar aggregated reward strategy against an NSGA-II multi-objective variant.

### Prerequisites

| Requirement | Description |
|---|---|
| QuickVina2-GPU binary | Pass path via `--quickvina-binary`. Build from [Vina-GPU repo](https://github.com/DeltaGroupNJUPT/Vina-GPU) |
| Receptor PDBQT | Pass via `--receptor-file`. Example: `7uvu-2-monomers-pdbfixer.pdbqt` |
| Reference ligand PDB | Pass via `--reference-ligand-file`. Example: `7uvu-reference.pdb` |
| Seed SMILES file | e.g. `chembl24_canon_train.smiles` (ChEMBL 24 training set) |
| ZINC-250k SMILES file | Required when `--init-population-mode graphga_zinc250k` |

### Scalar vs NSGA-II comparison

```bash
cd benchmarks/Saturn

python compare_scalar_vs_nsga2_saturn.py \
  --seeds 0,1,2 \
  --budgets 1000 \
  --population-size 256 \
  --batch-size 32 \
  --seed-smiles-file /path/to/chembl24_canon_train.smiles \
  --seed-pool-size 256 \
  --init-population-mode graphga_zinc250k \
  --graphga-zinc250k-seed-smiles-file /path/to/zinc250k_ranked_qed_sa.smi \
  --oracle-template table2_r_sa_qed_oracle_template.json \
  --saturn-repo-root /path/to/Core \
  --quickvina-binary /path/to/QuickVina2-GPU-2-1 \
  --receptor-file /path/to/7uvu-2-monomers-pdbfixer.pdbqt \
  --reference-ligand-file /path/to/7uvu-reference.pdb \
  --output-dir outputs/saturn_comparison
```

#### Full flag reference

| Flag | Default | Description |
|---|---|---|
| `--seeds` | `0` | Comma-separated integer seeds |
| `--budgets` | `1000` | Comma-separated oracle budgets per seed |
| `--population-size` | `256` | Evolution population size |
| `--batch-size` | `32` | Oracle calls per generation |
| `--max-generations` | `0` | Hard generation cap (0 = run until budget is exhausted) |
| `--top-k` | `100` | Top-K molecules used in summary metrics |
| `--seed-smiles-file` | *(repo default)* | SMILES file for seeding the initial population |
| `--seed-pool-size` | `2000` | Max seeds loaded from file |
| `--init-population-mode` | `current` | `current` (seed from SMILES file) or `graphga_zinc250k` |
| `--graphga-zinc250k-seed-smiles-file` | *(none)* | ZINC-250k SMILES for `graphga_zinc250k` init mode |
| `--oracle-template` | *(repo default)* | Oracle configuration JSON |
| `--oracle-config-key` | `oracle` | Top-level JSON key containing the oracle payload |
| `--saturn-repo-root` | `Core/` | Root from which oracle modules are imported |
| `--quickvina-binary` | *(auto from repo root)* | Path to compiled QuickVina2-GPU binary |
| `--receptor-file` | *(auto from repo root)* | Receptor PDBQT file |
| `--reference-ligand-file` | *(auto from repo root)* | Reference ligand PDB file |
| `--docking-component` | *(auto-detected)* | Override the docking oracle component name |
| `--tournament-k` | `8` | Tournament selection size |
| `--elite-fraction` | `0.15` | Elite fraction |
| `--immigrant-fraction` | `0.10` | Immigrant fraction |
| `--parent-pool-fraction` | `0.50` | Parent pool fraction |
| `--stagnation-patience` | `12` | Generations before exploration boost |
| `--stagnation-mutation-boost` | `2` | Extra mutation depth on stagnation |
| `--skip-scalar` | `False` | Skip the scalar aggregated baseline |
| `--skip-nsga2` | `False` | Skip the NSGA-II run |
| `--output-dir` | `outputs/compare_scalar_vs_nsga2` | Output root |
| `--run-id` | *(auto timestamp)* | Run identifier appended to output paths |

#### Replicating run 46364391

The exact settings for the reference IBEX cluster run are recorded in `Saturn_TestCase/software_saturn46364391/`. To reproduce locally:

```bash
cd benchmarks/Saturn

python compare_scalar_vs_nsga2_saturn.py \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --budgets 1000 \
  --population-size 256 \
  --batch-size 32 \
  --seed-pool-size 256 \
  --init-population-mode graphga_zinc250k \
  --graphga-zinc250k-seed-smiles-file /path/to/zinc250k_ranked_qed_sa.smi \
  --oracle-template table2_r_sa_qed_oracle_template.json \
  --quickvina-binary /path/to/QuickVina2-GPU-2-1 \
  --receptor-file /path/to/7uvu-2-monomers-pdbfixer.pdbqt \
  --reference-ligand-file /path/to/7uvu-reference.pdb
```

---

## Configuration Reference

All defaults live in `core/config.py`. They can be overridden at three levels, in increasing priority:

1. **Hardcoded defaults** — constants in `core/config.py`
2. **Environment variables** — matching the constant name in uppercase
3. **CLI flags** — any `--flag` passed directly to a script

### Commonly tuned defaults

| Constant | Default | Description |
|---|---|---|
| `BENCHMARK_DEFAULT` | `GuacaMol` | MolScore preset |
| `BUDGET_DEFAULT` | `100000000` | Evaluations per task |
| `POPULATION_SIZE_DEFAULT` | `256` | Population size |
| `BATCH_SIZE_DEFAULT` | `256` | Batch size |
| `MAX_GENERATIONS_DEFAULT` | `50` | Max generations |
| `TOURNAMENT_K_DEFAULT` | `8` | Tournament size |
| `SEED_DEFAULT` | `7` | Random seed |
| `SEED_POOL_SIZE_DEFAULT` | `2000` | Seed pool size |
| `SEEDS_CSV_DEFAULT` | `7,8,9` | Seeds for multi-run comparison |
| `LOCAL_EVO_ELITE_FRACTION` | `0.15` | Elite fraction |
| `LOCAL_EVO_IMMIGRANT_FRACTION` | `0.10` | Immigrant fraction |
| `LOCAL_EVO_PARENT_POOL_FRACTION` | `0.50` | Parent pool fraction |
| `LOCAL_EVO_STAGNATION_PATIENCE` | `12` | Stagnation patience |
| `LOCAL_EVO_STAGNATION_MUTATION_BOOST` | `2` | Mutation boost on stagnation |
| `ALLOWED_ATOMIC_NUMBERS` | `C,N,O,F,P,S,Cl,Br` | Elements allowed in generated molecules |

---

## Output Files

Each run creates a timestamped directory under `--output-dir`.

### Single-model evolution run

```
outputs/runs/
└── GuacaMol_local_evo_seed7_20260424_120000/
    ├── task_AlbuterolSimilarity/
    │   ├── step_*.csv          # Per-step scoring results (MolScore format)
    │   └── task_config.json    # MolScore task configuration snapshot
    ├── task_Fexofenadine/
    │   └── ...
    ├── strategy_results.csv    # Per-task summary: best score, top-10 mean, budget used
    └── guacamol_report.json    # GuacaMol-style aggregated benchmark report
```

### Multi-model comparison run

```
outputs/comparisons/
└── GuacaMol_comparison_20260424_120000/
    ├── seed7/
    │   ├── local_evolution/    # Per-model run outputs (same structure as above)
    │   └── graphga_example/
    ├── seed8/
    │   └── ...
    ├── comparison_summary.csv  # All models x seeds x tasks
    └── guacamol_comparison_report.json
```

### Saturn comparison run

```
outputs/saturn_comparison/
└── compare_scalar_vs_nsga2_20260424_120000/
    ├── seed0_budget1000/
    │   ├── scalar/             # Scalar aggregation strategy outputs
    │   └── nsga2/              # NSGA-II strategy outputs
    ├── seed1_budget1000/
    │   └── ...
    └── results_summary.csv     # Strategy x seed x budget: docking %, top-K metrics
```

---

  ## Matrix Results (May 2026)

  Round-2 cluster matrix runs completed successfully for all four configurations:

  - baseline_cpu: thread executor + pandas backend
  - dask_cpu: dask executor + pandas backend
  - rapids_v100: thread executor + dataframe backend auto (resolved to cuDF on V100)
  - dask_rapids_v100: dask executor + dataframe backend auto (resolved to cuDF on V100)

  ### Runtime Summary

  | Config | Mean elapsed seconds | Speedup vs baseline | Time reduction vs baseline |
  |---|---:|---:|---:|
  | baseline_cpu | 1541.2372 | 1.0000x | 0.00% |
  | dask_cpu | 1463.3349 | 1.0532x | 5.05% |
  | rapids_v100 | 1349.2829 | 1.1423x | 12.45% |
  | dask_rapids_v100 | 1316.1125 | 1.1711x | 14.61% |

  ### Quality Check

  - Aggregate score metrics are matched across configurations (differences only at floating-point precision).
  - n_failed is 0 for all matrix runs.

  ### Recommended Default Launch Mode

  - Use dask_rapids_v100 for production-scale matrix runs.
  - Launch script: slurm_scripts/compare_matrix_dask_rapids_v100.sbatch

  ### Report and Data Artifacts

  - One-page report: molscore/outputs/comparison_matrix/matrix_report_20260506_round2.md
  - Speedup table: molscore/outputs/comparison_matrix/speedup_matrix_20260506_round2.tsv
  - Baseline runtime summary: molscore/outputs/comparison_matrix/baseline_cpu/compare_GuacaMol_20260506_120038/model_runtime_summary.tsv
  - Dask CPU runtime summary: molscore/outputs/comparison_matrix/dask_cpu/compare_GuacaMol_20260506_120224/model_runtime_summary.tsv
  - RAPIDS V100 runtime summary: molscore/outputs/comparison_matrix/rapids_v100/compare_GuacaMol_20260506_120813/model_runtime_summary.tsv
  - Dask + RAPIDS V100 runtime summary: molscore/outputs/comparison_matrix/dask_rapids_v100/compare_GuacaMol_20260506_121020/model_runtime_summary.tsv

  ---

## Extending the Framework

### Adding a new oracle component

1. Create `core/oracles/<category>/my_oracle.py` inheriting from `OracleComponent`.
2. Implement `__call__(self, mol: Mol) -> float`.
3. Register it in `core/oracles/utils.py` inside the `construct_oracle_component` factory.

### Adding a new benchmark model

1. Add an entry to `benchmarks/Guacamol/model_specs.json`:
   ```json
   {
     "name": "my_model",
     "type": "command",
     "description": "My custom generator",
     "enabled": true,
     "command": "python {examples_root}/my_model/run.py --molscore {molscore_config} --output-dir {output_dir} --seed {seed}"
   }
   ```
2. Pass `--models my_model` to `benchmark_compare_models.py`.

### Adding a new xTB descriptor

Add a new file under `core/oracles/xtb/` following the pattern of e.g. `homo.py`: import `GeometryOptimizer`, call `morfeus.XTB`, extract the descriptor, and return a normalised float. Then register the new component in `core/oracles/utils.py`.
