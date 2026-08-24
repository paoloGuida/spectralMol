#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


TARGET_ENV_YAMLS = {
    "pidgin": Path("PIDGINv5") / "environment.yml",
    "ms_molopt": Path("molopt") / "ms_molopt.yml",
}


def run_cmd(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def conda_binary() -> str:
    path = shutil.which("conda")
    if not path:
        raise RuntimeError("conda command not found in PATH")
    return path


def mamba_binary() -> str | None:
    return shutil.which("mamba")


def conda_envs() -> tuple[set[str], set[str]]:
    cp = run_cmd([conda_binary(), "env", "list", "--json"], check=True)
    payload = json.loads(cp.stdout or "{}")
    prefixes = {str(p) for p in payload.get("envs", []) if p}
    names = {Path(p).name for p in prefixes}
    return names, prefixes


def yaml_env_name(yaml_path: Path, fallback: str) -> str:
    try:
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean.startswith("name:"):
                name = clean.split(":", 1)[1].strip()
                if name:
                    return name
    except Exception:
        pass
    return fallback


def create_from_yaml(yaml_path: Path) -> tuple[bool, str]:
    conda = conda_binary()
    mamba = mamba_binary()
    attempts: list[list[str]] = []
    if mamba:
        attempts.append([mamba, "env", "create", "-f", str(yaml_path), "-y"])
    attempts.append([conda, "env", "create", "-f", str(yaml_path), "-y"])

    logs: list[str] = []
    for cmd in attempts:
        cp = run_cmd(cmd, check=False)
        logs.append(
            f"$ {' '.join(cmd)}\n"
            f"[rc={cp.returncode}]\n"
            f"{cp.stdout}\n{cp.stderr}\n"
        )
        if cp.returncode == 0:
            return True, "\n".join(logs)
    return False, "\n".join(logs)


def clone_env(target_name: str, clone_from: str) -> tuple[bool, str]:
    cmd = [conda_binary(), "create", "-y", "-n", target_name, "--clone", clone_from]
    cp = run_cmd(cmd, check=False)
    log = (
        f"$ {' '.join(cmd)}\n"
        f"[rc={cp.returncode}]\n"
        f"{cp.stdout}\n{cp.stderr}\n"
    )
    return (cp.returncode == 0), log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-create MolScore scoring environments so benchmark runs do not fail mid-task."
    )
    p.add_argument(
        "--targets",
        default="pidgin,ms_molopt",
        help="Comma-separated targets from: pidgin, ms_molopt",
    )
    p.add_argument(
        "--clone-from",
        default=os.environ.get("MOLSCORE_SCORING_ENV_CLONE_FROM", os.environ.get("CONDA_DEFAULT_ENV", "")),
        help="Fallback source environment name for `conda create --clone` if YAML creation fails.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Do not use clone fallback; fail when YAML creation fails.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_models_dir() -> Path:
    import molscore  # Imported lazily so this script can show clear import failures.

    return Path(molscore.__file__).resolve().parent / "data" / "models"


def bool_exists(target_name: str, target_prefix: str | None, names: set[str], prefixes: set[str]) -> bool:
    if target_name in names:
        return True
    if target_prefix and target_prefix in prefixes:
        return True
    return False


def main() -> int:
    args = parse_args()
    targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    unknown = [t for t in targets if t not in TARGET_ENV_YAMLS]
    if unknown:
        raise SystemExit(f"Unknown --targets entries: {', '.join(unknown)}")

    models_dir = resolve_models_dir()
    names, prefixes = conda_envs()
    clone_from = str(args.clone_from).strip()
    all_ok = True

    for target in targets:
        yaml_path = models_dir / TARGET_ENV_YAMLS[target]
        env_name = target
        if not yaml_path.exists():
            print(f"[bootstrap] missing yaml for {target}: {yaml_path}", flush=True)
            if args.strict:
                all_ok = False
                continue
            if not clone_from:
                print(
                    f"[bootstrap] {target}: cannot clone fallback (no --clone-from / CONDA_DEFAULT_ENV)",
                    flush=True,
                )
                all_ok = False
                continue
            if args.dry_run:
                print(f"[bootstrap] {target}: dry-run would clone env '{env_name}' from '{clone_from}'", flush=True)
                continue
            print(f"[bootstrap] {target}: trying clone fallback from '{clone_from}'", flush=True)
            cloned, clone_log = clone_env(target_name=env_name, clone_from=clone_from)
            print(clone_log, flush=True)
            if not cloned:
                all_ok = False
                continue
            names, prefixes = conda_envs()
            if bool_exists(env_name, None, names, prefixes):
                print(f"[bootstrap] {target}: clone fallback succeeded", flush=True)
            else:
                print(f"[bootstrap] {target}: clone command reported success but env not found", flush=True)
                all_ok = False
            continue

        env_name = yaml_env_name(yaml_path, fallback=target)
        exists = bool_exists(env_name, None, names, prefixes)
        if exists:
            print(f"[bootstrap] {target}: env '{env_name}' already exists", flush=True)
            continue

        print(f"[bootstrap] {target}: creating env '{env_name}' from {yaml_path}", flush=True)
        if args.dry_run:
            continue

        created, create_log = create_from_yaml(yaml_path)
        if created:
            print(f"[bootstrap] {target}: created from yaml", flush=True)
            names, prefixes = conda_envs()
            continue

        print(f"[bootstrap] {target}: yaml creation failed", flush=True)
        print(create_log, flush=True)
        if args.strict:
            all_ok = False
            continue

        if not clone_from:
            print(
                f"[bootstrap] {target}: clone fallback disabled (no --clone-from / CONDA_DEFAULT_ENV)",
                flush=True,
            )
            all_ok = False
            continue

        print(f"[bootstrap] {target}: trying clone fallback from '{clone_from}'", flush=True)
        cloned, clone_log = clone_env(target_name=env_name, clone_from=clone_from)
        print(clone_log, flush=True)
        if not cloned:
            all_ok = False
            continue
        names, prefixes = conda_envs()
        if bool_exists(env_name, None, names, prefixes):
            print(f"[bootstrap] {target}: clone fallback succeeded", flush=True)
        else:
            print(f"[bootstrap] {target}: clone command reported success but env not found", flush=True)
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
