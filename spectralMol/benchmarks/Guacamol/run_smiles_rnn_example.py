#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from molscore import MolScoreBenchmark


def _run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    smiles_root = env.get("MOLSCORE_SMILES_RNN_ROOT", "")
    if smiles_root:
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{smiles_root}:{py_path}" if py_path else smiles_root
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def _find_script(smiles_root: Path) -> Path:
    direct_candidates = [
        smiles_root / "scripts" / "reinforcement_learning.py",
        smiles_root / "reinforcement_learning.py",
        smiles_root / "scripts" / "molscore_smiles_rnn.py",
        smiles_root / "scripts" / "molscore_smilesrnn.py",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    discovered: list[Path] = []
    for py_file in smiles_root.rglob("*.py"):
        name = py_file.name.lower()
        if "reinforcement_learning" in name or ("molscore" in name and ("smiles" in name or "rnn" in name)):
            discovered.append(py_file.resolve())
    if discovered:
        return sorted(discovered)[0]

    raise FileNotFoundError(
        "Could not find a SMILES-RNN MolScore runner script under "
        f"{smiles_root}. If this folder is empty, initialize the submodule:\n"
        "git -C MolScore_examples submodule update --init --recursive -- SMILES-RNN"
    )


def _supports(help_text: str, flag: str) -> bool:
    return flag in help_text


def _resolve_device(requested: str) -> str:
    value = (requested or "cpu").strip().lower()
    if value not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported --device {requested!r}; expected cpu, cuda, or auto.")
    if value in {"cpu", "cuda"}:
        return value
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Runner wrapper for MolScore_examples SMILES-RNN model.")
    p.add_argument("--examples-root", required=True)
    p.add_argument("--molscore", required=True, help="MolScore benchmark preset name or custom benchmark path.")
    p.add_argument("--smiles-file", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--generations", type=int, required=True)
    p.add_argument("--population-size", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--include", default="")
    p.add_argument("--exclude", default="")
    p.add_argument("--model", default="RNN", choices=["RNN", "Transformer", "GTr"])
    p.add_argument("--prior", default="", help="Optional prior checkpoint; defaults to SMILES-RNN/priors/ChEMBL28pur.ckpt")
    p.add_argument("--rl-strategy", default="HC", help="RL strategy key for reinforcement_learning.py (default: HC)")
    p.add_argument("--device", default="auto", help="Training device: cpu, cuda, or auto.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    python_bin = sys.executable
    examples_root = Path(args.examples_root).resolve()
    smiles_root = examples_root / "SMILES-RNN"
    if not smiles_root.exists():
        raise FileNotFoundError(f"SMILES-RNN directory not found: {smiles_root}")
    script = _find_script(smiles_root)
    os.environ["MOLSCORE_SMILES_RNN_ROOT"] = str(smiles_root)

    prior_path = Path(args.prior).resolve() if args.prior.strip() else (smiles_root / "priors" / "ChEMBL28pur.ckpt")
    if not prior_path.exists():
        raise FileNotFoundError(
            f"SMILES-RNN prior not found: {prior_path}. Provide --prior or ensure priors are available."
        )

    include = [x.strip() for x in (args.include or "").split(",") if x.strip()]
    exclude = [x.strip() for x in (args.exclude or "").split(",") if x.strip()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preset_keys = set(getattr(MolScoreBenchmark, "presets", {}).keys())
    if args.molscore in preset_keys:
        bench = MolScoreBenchmark(
            model_name="SMILES-RNN",
            output_dir=str(output_dir),
            budget=int(args.budget),
            benchmark=args.molscore,
            include=include,
            exclude=exclude,
        )
    elif Path(args.molscore).is_dir():
        bench = MolScoreBenchmark(
            model_name="SMILES-RNN",
            output_dir=str(output_dir),
            budget=int(args.budget),
            custom_benchmark=args.molscore,
            include=include,
            exclude=exclude,
        )
    else:
        bench = MolScoreBenchmark(
            model_name="SMILES-RNN",
            output_dir=str(output_dir),
            budget=int(args.budget),
            benchmark=args.molscore,
            include=include,
            exclude=exclude,
        )

    n_steps = max(1, int(args.generations))
    save_freq = max(1, n_steps)
    batch_size = max(1, int(args.batch_size))
    device = _resolve_device(args.device)

    for i, task in enumerate(bench, start=1):
        cfg = dict(getattr(task, "cfg", {}))
        if not cfg:
            raise RuntimeError(f"Missing task config for SMILES-RNN benchmark task #{i}")
        task_name = str(cfg.get("task", f"task_{i}"))
        cfg["output_dir"] = str(output_dir)
        cfg_path = output_dir / f"{i:03d}_{task_name}_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        cmd = [
            python_bin,
            str(script),
            "--prior",
            str(prior_path),
            "--molscore_config",
            str(cfg_path),
            "--model",
            args.model,
            "--device",
            device,
            "--save_freq",
            str(save_freq),
            args.rl_strategy,
            "--n_steps",
            str(n_steps),
            "--batch_size",
            str(batch_size),
        ]
        proc = _run(cmd, cwd=smiles_root, capture=False)
        if proc.returncode != 0:
            raise RuntimeError(f"SMILES-RNN command failed ({proc.returncode}) on task {task_name}: {' '.join(cmd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
