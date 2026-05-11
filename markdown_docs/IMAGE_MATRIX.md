# Image Matrix

## Goal
Use split images to keep Saturn docking dependencies independent from RAPIDS acceleration dependencies.

## Images

| Image | Build spec | Primary purpose | Includes | Excludes |
|---|---|---|---|---|
| `molevo-saturn:cuda12.2-quickvina2` | `Dockerfile.saturn` | SATURN docking workflows | QuickVina2-GPU, RDKit, OpenBabel, MolScore stack | RAPIDS/cuDF stack |
| `molevo-rapids:cuda12.2` | `Dockerfile.rapids` | GuacaMol acceleration workflows | cuDF/dask-cudf/cuML, RDKit, OpenBabel, Dask, MolScore stack | QuickVina2-GPU target bundle |

## Launcher Mapping

### Current launchers (conda-based)

| Launcher | Runtime today | Target image (proposed) |
|---|---|---|
| `slurm_scripts/compare_matrix_baseline_cpu.sbatch` | local conda env | none (keep conda CPU baseline) |
| `slurm_scripts/compare_matrix_dask_cpu.sbatch` | local conda env | none (keep conda CPU baseline) |
| `slurm_scripts/compare_matrix_rapids_v100.sbatch` | local conda env | `molevo-rapids:cuda12.2` |
| `slurm_scripts/compare_matrix_dask_rapids_v100.sbatch` | local conda env | `molevo-rapids:cuda12.2` |
| `slurm_scripts/gpu_evolve_smoke.sbatch` | local conda env | optional `molevo-rapids:cuda12.2` |

### Current container launchers

| Launcher | Runtime today | Image |
|---|---|---|
| `slurm_scripts/validate_saturn_container.sh` | podman or singularity/apptainer | `molevo-saturn:cuda12.2-quickvina2` |
| `slurm_scripts/run_saturn_container_smoke_interactive.sh` | singularity/apptainer | `molevo-saturn:cuda12.2-quickvina2` |
| `slurm_scripts/pull_saturn_container_podman.sh` | podman | pulls SATURN image from registry |

## Recommendation
1. Keep split-image strategy for production: SATURN image and RAPIDS image.
2. Keep CPU baselines on conda launchers for fair comparisons and minimal complexity.
3. Containerize only GPU-accelerated paths first (`rapids_v100`, `dask_rapids_v100`, SATURN).
4. Avoid all-in-one image unless you need a debugging convenience image.

## Next integration step
Create containerized variants of:
1. `compare_matrix_rapids_v100.sbatch`
2. `compare_matrix_dask_rapids_v100.sbatch`

These should execute `benchmark_compare_models.py` inside `molevo-rapids:cuda12.2` via Singularity/Apptainer on IBEX.
