#!/usr/bin/env python3
"""Portable command-line entry point for SpectralMol experiments."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shlex
import shutil
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
TASK_PROFILES = PROJECT_ROOT / "configs" / "guacamol_task_profiles.json"
SATURN_ENVIRONMENT = PROJECT_ROOT / "configs" / "saturn_table8_v119_environment.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run portable SpectralMol smoke tests, benchmarks, analyses, or release checks.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.toml",
        help="TOML configuration file (default: configs/default.toml).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without executing them.")
    parser.add_argument("--output-dir", type=Path, help="Override run.output_dir.")
    parser.add_argument("--seed", type=int, help="Run one seed instead of the configured seed list.")
    parser.add_argument("--task", type=int, help="Run one GuacaMol task index instead of the configured task list.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override run.device. SATURN docking requires a supported GPU/OpenCL runtime.",
    )
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if not isinstance(config.get("run"), dict) or not config["run"].get("workflow"):
        raise ValueError(f"{path} must define [run] workflow = ...")
    config["_config_path"] = str(path)
    return config


def _path(value: str | Path | None, *, required: bool = False) -> Path | None:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError("A required path is empty in the configuration.")
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _require_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path or '<not configured>'}")
    return path


def _require_dir(path: Path | None, label: str) -> Path:
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path or '<not configured>'}")
    return path


def _executable(value: str | Path | None, label: str) -> Path:
    if value:
        raw = str(value)
        resolved = shutil.which(raw) if not Path(raw).parent.name else None
        if resolved:
            return Path(resolved).resolve()
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(f"{label} executable not found. Configure its path explicitly or add it to PATH.")


def _output_dir(config: dict[str, Any], override: Path | None) -> Path:
    value = override or config["run"].get("output_dir", "results")
    path = _path(value, required=True)
    assert path is not None
    path.mkdir(parents=True, exist_ok=True)
    return path


def _environment(extra: dict[str, Any] | None = None, *, device: str = "auto") -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(PACKAGE_ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(key, "1")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def _run(command: list[str | Path], *, env: dict[str, str], dry_run: bool, cwd: Path = PROJECT_ROOT) -> None:
    printable = [str(part) for part in command]
    print("[spectralmol]", shlex.join(printable), flush=True)
    if not dry_run:
        subprocess.run(printable, check=True, cwd=cwd, env=env)


def _write_metadata(output: Path, config: dict[str, Any], commands: list[list[str]]) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "python": sys.version,
        "commands": commands,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


def _materialize_gzip_input(path: Path, cache_dir: Path) -> Path:
    if path.is_file():
        return path
    archive = Path(f"{path}.gz")
    if not archive.is_file():
        raise FileNotFoundError(f"Input file not found: {path} (also checked {archive})")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / path.name
    if not target.is_file():
        with gzip.open(archive, "rb") as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    return target


def _smoke(config: dict[str, Any], output: Path, dry_run: bool, device: str) -> None:
    if dry_run:
        print("[smoke] imports and theta construction would be tested")
        return
    sys.path.insert(0, str(PACKAGE_ROOT))
    import numpy as np
    from rdkit import Chem
    from core.spectral_evolution import SpectralGenerator, SpectralSettings

    seed = int(config.get("smoke", {}).get("seed", 7))
    settings = SpectralSettings(L=12, K=3, D=12, decode_attempts=2)
    generator = SpectralGenerator(settings=settings, seed=seed)
    population = generator.build_initial_population(["CCO", "CCN", "c1ccccc1"], pop_size=2)
    mutated = generator.mutate_theta(population[0].theta, gen=1, generations=4)
    passed = bool(
        Chem.MolFromSmiles(population[0].smiles) is not None
        and mutated.shape == population[0].theta.shape
        and np.isfinite(mutated).all()
        and not np.allclose(mutated, population[0].theta)
    )
    result = {"passed": passed, "seed": seed, "population": [item.smiles for item in population]}
    (output / "smoke_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("SpectralMol smoke test failed")
    print(f"[smoke] passed; output={output / 'smoke_test.json'}")


def _model_spec(output: Path, graphga_patience: int) -> Path:
    spec = {
        "spectralmol": {
            "type": "builtin_local_evolution",
            "enabled": True,
            "description": "SpectralMol Fourier-theta local evolution runner.",
        },
        "graphga": {
            "type": "command",
            "enabled": True,
            "description": "MolScore GraphGA example runner.",
            "command": (
                "{python} {repo_root}/benchmarks/Guacamol/run_graphga_example_wrapper.py "
                "--graphga-script {examples_root}/GraphGA/molscore_GB_GA.py --molscore {molscore_target} "
                "--smiles_file {shared_init_file} --starting_population_file {shared_init_file} "
                "--seed {seed} --budget {budget} --output_dir {model_output_dir} "
                "--population_size {population_size} --offspring_size {population_size} "
                f"--generations {{generations}} --n_jobs {{graphga_n_jobs}} --patience {graphga_patience} "
                "{include_args} {exclude_args} {task_indexes_args}"
            ),
        },
    }
    path = output / "model_specs.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


def _guacamol(config: dict[str, Any], output: Path, dry_run: bool, device: str, seed_override: int | None, task_override: int | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    settings = config.get("guacamol", {})
    external = config.get("external", {})
    profiles = json.loads(TASK_PROFILES.read_text(encoding="utf-8"))
    seeds = [seed_override] if seed_override is not None else [int(v) for v in settings.get("seeds", [7])]
    tasks = [task_override] if task_override is not None else [int(v) for v in settings.get("tasks", range(20))]
    models = [str(v) for v in settings.get("models", ["spectralmol"])]
    unknown = [task for task in tasks if str(task) not in profiles]
    if unknown:
        raise ValueError(f"Unknown GuacaMol task indexes: {unknown}")
    seed_file = _require_file(_path(settings.get("seed_smiles_file")), "Seed SMILES file")
    examples_root = _path(external.get("molscore_examples"))
    if "graphga" in models:
        examples_root = _require_dir(examples_root, "MolScore examples checkout")
        _require_file(examples_root / "GraphGA" / "molscore_GB_GA.py", "GraphGA runner")
    profile_enabled = bool(settings.get("use_task_profiles", True))
    generations = int(settings.get("generations", 500))
    population = int(settings.get("population_size", 256))
    batch = int(settings.get("batch_size", 256))
    base_budget = int(settings.get("budget", 128256))
    graphga_patience = int(settings.get("graphga_patience", 500))
    model_spec = _model_spec(output, graphga_patience)
    commands: list[list[str]] = []
    for seed in seeds:
        for task in tasks:
            profile = profiles[str(task)]
            profile_env = profile["environment"] if profile_enabled else {}
            candidates = max(1, int(profile_env.get("MOLSCORE_SPECTRAL_TASK_AWARE_DECODE_CANDIDATES", "1")))
            budget = max(base_budget, population + batch * generations * candidates)
            task_output = output / f"seed_{seed}" / f"task_{task}_{profile['name']}"
            task_output.mkdir(parents=True, exist_ok=True)
            env_values: dict[str, Any] = {
                **profile_env,
                **config.get("environment", {}),
                "SPECTRAL_TASK_PROFILE_MODE": "manual" if profile_enabled else "off",
                "SPECTRAL_TASK_PROFILE": profile_env.get("PROFILE", "off"),
                "MOLSCORE_LOCAL_EVO_GENERATOR": "spectral",
                "MOLSCORE_FREQUENCY_MODE": settings.get("frequency_mode", "full-spectrum"),
                "MOLSCORE_SPECTRAL_L": settings.get("spectral_l", 32),
                "MOLSCORE_SPECTRAL_K": settings.get("spectral_k", 16),
                "MOLSCORE_SPECTRAL_D": settings.get("spectral_d", 32),
                "MOLSCORE_DECODE_ATTEMPTS": profile_env.get(
                    "SPECTRAL_DECODE_ATTEMPTS", settings.get("spectral_decode_attempts", 4)
                ),
                "GRAPHGA_PATIENCE": graphga_patience,
            }
            command = [
                sys.executable,
                PACKAGE_ROOT / "benchmarks" / "Guacamol" / "benchmark_compare_models.py",
                "--benchmark", str(settings.get("benchmark", "GuacaMol")),
                "--budget", str(budget),
                "--generations", str(generations),
                "--population-size", str(population),
                "--batch-size", str(batch),
                "--seeds", str(seed),
                "--models", ",".join(models),
                "--model-spec-file", model_spec,
                "--examples-root", examples_root or PROJECT_ROOT,
                "--python-bin", sys.executable,
                "--seed-smiles-file", seed_file,
                "--seed-pool-size", str(settings.get("seed_pool_size", 256)),
                "--local-evo-generator", "spectral",
                "--local-evo-frequency-mode", str(settings.get("frequency_mode", "full-spectrum")),
                "--equal-initial-population",
                "--graphga-n-jobs", str(settings.get("graphga_n_jobs", 1)),
                "--executor", str(settings.get("executor", "thread")),
                "--dataframe-backend", "pandas",
                "--no-bootstrap-scoring-envs",
                "--task-indexes", str(profile["molscore_task_index"]),
                "--output-dir", task_output,
            ]
            commands.append([str(item) for item in command])
            _run(command, env=_environment(env_values, device=device), dry_run=dry_run, cwd=PACKAGE_ROOT)
    _write_metadata(output, config, commands)


def _saturn(config: dict[str, Any], output: Path, dry_run: bool, device: str, seed_override: int | None) -> None:
    if device == "cpu":
        raise ValueError("The publication SATURN profile uses QuickVina2-GPU and cannot run with --device cpu.")
    settings = config.get("saturn", {})
    external = config.get("external", {})
    saturn_root = _require_dir(_path(external.get("saturn_repository")), "SATURN checkout")
    quickvina = _executable(external.get("quickvina_binary"), "QuickVina2-GPU")
    obabel = _executable(external.get("openbabel_binary", "obabel"), "OpenBabel")
    oracle = _require_file(_path(settings.get("oracle_template")), "SATURN oracle template")
    receptor = _require_file(_path(settings.get("receptor_file")), "Docking receptor")
    reference = _require_file(_path(settings.get("reference_ligand_file")), "Docking reference ligand")
    seed_dir = _require_dir(_path(settings.get("seed_smiles_dir")), "Per-seed SMILES directory")
    seeds = [seed_override] if seed_override is not None else [int(v) for v in settings.get("seeds", range(10))]
    profile_env = json.loads(SATURN_ENVIRONMENT.read_text(encoding="utf-8"))
    profile_env.update(config.get("environment", {}))
    profile_env["MOLSCORE_SATURN_OBABEL_BINARY"] = str(obabel)
    opencl_dir = external.get("opencl_library_dir", "")
    if opencl_dir:
        opencl_path = _require_dir(_path(opencl_dir), "OpenCL library directory")
        current = os.environ.get("LD_LIBRARY_PATH", "")
        profile_env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(opencl_path), current)))
    commands: list[list[str]] = []
    for seed in seeds:
        seed_file = _require_file(seed_dir / f"seed_{seed}.smi", f"Seed {seed} SMILES")
        command = [
            sys.executable,
            PACKAGE_ROOT / "benchmarks" / "Saturn" / "compare_scalar_vs_nsga2_saturn.py",
            "--seeds", str(seed),
            "--budgets", str(settings.get("budget", 1000)),
            "--population-size", str(settings.get("population_size", 256)),
            "--batch-size", str(settings.get("batch_size", 16)),
            "--max-generations", str(settings.get("max_generations", 0)),
            "--top-k", str(settings.get("top_k", 100)),
            "--seed-smiles-file", seed_file,
            "--per-seed-seed-smiles-dir", seed_dir,
            "--seed-pool-size", str(settings.get("seed_pool_size", 256)),
            "--oracle-template", oracle,
            "--oracle-config-key", str(settings.get("oracle_config_key", "oracle")),
            "--saturn-repo-root", saturn_root,
            "--quickvina-binary", quickvina,
            "--receptor-file", receptor,
            "--reference-ligand-file", reference,
            "--tournament-k", str(settings.get("tournament_k", 8)),
            "--immigrant-fraction", str(settings.get("immigrant_fraction", 0.02)),
            "--parent-pool-fraction", str(settings.get("parent_pool_fraction", 0.5)),
            "--stagnation-patience", str(settings.get("stagnation_patience", 12)),
            "--stagnation-mutation-boost", str(settings.get("stagnation_mutation_boost", 2)),
            "--skip-scalar",
            "--nsga2-genotype", "theta",
            "--frequency-mode", str(settings.get("frequency_mode", "full-spectrum")),
            "--spectral-l", str(settings.get("spectral_l", 48)),
            "--spectral-k", str(settings.get("spectral_k", 24)),
            "--spectral-d", str(settings.get("spectral_d", 32)),
            "--spectral-decode-attempts", str(settings.get("spectral_decode_attempts", 16)),
            "--output-dir", output,
            "--run-id", f"saturn_theta_nsga2_budget{settings.get('budget', 1000)}_seed{seed}",
        ]
        commands.append([str(item) for item in command])
        _run(command, env=_environment(profile_env, device=device), dry_run=dry_run, cwd=PACKAGE_ROOT)
    if not dry_run and bool(settings.get("analyze_after_run", True)):
        analysis = output / "analysis"
        command = [
            sys.executable,
            PACKAGE_ROOT / "benchmarks" / "Saturn" / "analyze_theta_nsga2_saturn.py",
            "--input-root", output,
            "--output-dir", analysis,
            "--expected-seeds", ",".join(map(str, seeds)),
            "--budget", str(settings.get("budget", 1000)),
        ]
        commands.append([str(item) for item in command])
        _run(command, env=_environment(device=device), dry_run=False, cwd=PROJECT_ROOT)
    _write_metadata(output, config, commands)


def _frequency_ablation(config: dict[str, Any], output: Path, dry_run: bool, device: str, seed_override: int | None, task_override: int | None) -> None:
    settings = config.get("frequency_ablation", {})
    seed_path = _path(settings.get("seed_smiles_file"), required=True)
    assert seed_path is not None
    seed_file = _materialize_gzip_input(seed_path, output / "prepared_inputs")
    guacamol = {
        "run": {"workflow": "guacamol", "output_dir": str(output)},
        "guacamol": {
            "seeds": [seed_override] if seed_override is not None else settings.get("seeds", list(range(6))),
            "tasks": [task_override] if task_override is not None else settings.get("tasks", list(range(20))),
            "models": ["spectralmol"],
            "benchmark": "GuacaMol",
            "generations": int(settings.get("generations", 0)),
            "budget": int(settings.get("budget", 20000)),
            "population_size": int(settings.get("population_size", 256)),
            "batch_size": int(settings.get("batch_size", 256)),
            "seed_pool_size": int(settings.get("seed_pool_size", 2000)),
            "seed_smiles_file": str(seed_file),
            "use_task_profiles": True,
        },
        "external": config.get("external", {}),
        "environment": config.get("environment", {}),
        "_config_path": config["_config_path"],
    }
    for condition in settings.get("conditions", ["full-spectrum", "high-only", "low-only", "random-matrix"]):
        guacamol["guacamol"]["frequency_mode"] = str(condition)
        _guacamol(guacamol, output / str(condition), dry_run, device, None, None)


def _results(config: dict[str, Any]) -> None:
    section = config.get("results", {})
    files = section.get("files", [])
    if not files:
        raise ValueError("[results] files must contain at least one repository-relative path")
    for value in files:
        path = _require_file(_path(value), "Result table")
        print(f"===== {path.relative_to(PROJECT_ROOT)}")
        print(path.read_text(encoding="utf-8").rstrip())


def _verify(dry_run: bool) -> None:
    command = [sys.executable, PROJECT_ROOT / "scripts" / "validate_release.py"]
    _run(command, env=_environment(), dry_run=dry_run)


def main() -> int:
    args = _parser().parse_args()
    try:
        config = _load_config(args.config)
        run = config["run"]
        workflow = str(run["workflow"]).lower().replace("_", "-")
        device = args.device or str(run.get("device", "auto"))
        output = _output_dir(config, args.output_dir) if workflow not in {"results", "verify"} else PROJECT_ROOT
        if workflow == "smoke":
            _smoke(config, output, args.dry_run, device)
        elif workflow == "guacamol":
            _guacamol(config, output, args.dry_run, device, args.seed, args.task)
        elif workflow == "saturn":
            _saturn(config, output, args.dry_run, device, args.seed)
        elif workflow == "frequency-ablation":
            _frequency_ablation(config, output, args.dry_run, device, args.seed, args.task)
        elif workflow == "results":
            _results(config)
        elif workflow == "verify":
            _verify(args.dry_run)
        else:
            raise ValueError(f"Unsupported workflow {workflow!r}")
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
