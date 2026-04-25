#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _find_crem_root(examples_root: Path) -> Path:
    candidates = [
        examples_root / "crem",
        examples_root / "CReM",
    ]
    for path in candidates:
        if (path / "molscore_crem.py").exists():
            return path
    raise FileNotFoundError(
        "Could not find CReM example under MolScore_examples. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def _write_counted_lines(src_txt: Path, dst_txt: Path) -> None:
    counter: Counter[str] = Counter()
    with src_txt.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if row:
                counter[row] += 1
    if not counter:
        raise RuntimeError(f"No fragment context lines were produced from {src_txt}")

    with dst_txt.open("w", encoding="utf-8") as f:
        for row, count in counter.items():
            f.write(f"{count} {row}\n")


def _build_crem_db(*, python_bin: str, crem_root: Path, smiles_file: Path, db_path: Path, ncpu: int, radius: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="crem_db_build_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_db = work_dir / "crem_fragments.db"

    frags_txt = work_dir / "frags.txt"
    env_txt = work_dir / f"r{radius}.txt"
    counted_txt = work_dir / f"r{radius}_c.txt"

    _run(
        [
            python_bin,
            str(crem_root / "fragmentation.py"),
            "-i",
            str(smiles_file),
            "-o",
            str(frags_txt),
            "-c",
            str(max(1, int(ncpu))),
        ],
        cwd=crem_root,
    )
    _run(
        [
            python_bin,
            str(crem_root / "frag_to_env_mp.py"),
            "-i",
            str(frags_txt),
            "-o",
            str(env_txt),
            "-r",
            str(int(radius)),
            "-c",
            str(max(1, int(ncpu))),
        ],
        cwd=crem_root,
    )
    _write_counted_lines(env_txt, counted_txt)
    _run(
        [
            python_bin,
            str(crem_root / "import_env_to_db.py"),
            "-i",
            str(counted_txt),
            "-o",
            str(tmp_db),
            "-r",
            str(int(radius)),
            "-c",
            "-n",
            str(max(1, int(ncpu))),
        ],
        cwd=crem_root,
    )
    shutil.copy2(tmp_db, db_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Runner wrapper for MolScore_examples CReM model.")
    p.add_argument("--examples-root", required=True)
    p.add_argument("--molscore", required=True, help="MolScore benchmark preset name or custom benchmark path.")
    p.add_argument("--smiles-file", required=True, help="Initial SMILES pool file.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--generations", type=int, required=True)
    p.add_argument("--population-size", type=int, required=True)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--include", default="")
    p.add_argument("--exclude", default="")
    p.add_argument("--ncpu", type=int, default=1)
    p.add_argument("--replacements", type=int, default=1000)
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--db-fname", default="", help="Optional prebuilt CReM database file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    python_bin = sys.executable

    examples_root = Path(args.examples_root).resolve()
    smiles_file = Path(args.smiles_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not smiles_file.exists():
        raise FileNotFoundError(f"--smiles-file does not exist: {smiles_file}")

    crem_root = _find_crem_root(examples_root)
    if args.db_fname:
        db_path = Path(args.db_fname).resolve()
    else:
        cache_dir = Path(tempfile.gettempdir()) / "molevo_crem_db_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(smiles_file.read_bytes()).hexdigest()[:12]
        db_path = cache_dir / f"crem_r{int(args.radius)}_{digest}.db"
    if not db_path.exists():
        _build_crem_db(
            python_bin=python_bin,
            crem_root=crem_root,
            smiles_file=smiles_file,
            db_path=db_path,
            ncpu=args.ncpu,
            radius=args.radius,
        )

    cmd = [
        python_bin,
        str(crem_root / "molscore_crem.py"),
        "--molscore",
        args.molscore,
        "--smiles_file",
        str(smiles_file),
        "--db_fname",
        str(db_path),
        "--selection_size",
        str(args.population_size),
        "--generations",
        str(args.generations),
        "--seed",
        str(args.seed),
        "--ncpu",
        str(max(1, int(args.ncpu))),
        "--replacements",
        str(max(1, int(args.replacements))),
        "--budget",
        str(int(args.budget)),
        "--output_dir",
        str(output_dir),
    ]
    if args.include.strip():
        cmd += ["--include", args.include.strip()]
    if args.exclude.strip():
        cmd += ["--exclude", args.exclude.strip()]

    _run(cmd, cwd=crem_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
