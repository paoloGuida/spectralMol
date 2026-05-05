#!/usr/bin/env python3
"""
Phase 0 smoke test: Minimal GuacaMol run with profiling instrumentation.

This script runs a tiny GuacaMol benchmark to collect wall-time profiling data.
Parameters are tuned for a fast run (~5-10 minutes):
  - 1 seed
  - 1 task (first task only)
  - 1000 molecule budget
  - Small population (32)
  - 5 generations maximum
  - Small batch size (16)

Profiling timings are saved to profiling_runs/smoke_test_<timestamp>.json
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Make the repo root importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.profiling import get_profiler, enable_profiling


def run_smoke_test(
    output_dir: str = "smoke_test_runs",
    budget: int = 1000,
    population_size: int = 32,
    batch_size: int = 16,
    max_generations: int = 5,
    benchmark: str = "GuacaMol",
    include_tasks: str = "",
    profiling_output: str = None,
):
    """
    Run a minimal GuacaMol benchmark with profiling enabled.
    
    Args:
        output_dir: Directory for benchmark results
        budget: Molecule budget per task
        population_size: Population size
        batch_size: Batch size per generation
        max_generations: Maximum generations
        benchmark: Benchmark preset
        include_tasks: Comma-separated task names (empty = first task only)
        profiling_output: Path to save profiling JSON (auto-generated if None)
    """
    
    # Enable profiling globally
    enable_profiling()
    
    # Generate output filename
    if profiling_output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profiling_dir = _REPO_ROOT / "profiling_runs"
        profiling_dir.mkdir(parents=True, exist_ok=True)
        profiling_output = str(profiling_dir / f"smoke_test_{timestamp}.json")
    
    # Build command
    script = _REPO_ROOT / "benchmarks" / "Guacamol" / "evolve_vs_molscore_benchmark.py"
    cmd = [
        sys.executable,
        str(script),
        "--benchmark", benchmark,
        "--output-dir", output_dir,
        "--budget", str(budget),
        "--population-size", str(population_size),
        "--batch-size", str(batch_size),
        "--max-generations", str(max_generations),
        "--seed", "42",
        "--skip-random-baseline",  # Skip baseline for speed
    ]
    
    if include_tasks:
        cmd.extend(["--include", include_tasks])
    
    print(f"[Phase 0 Smoke Test] Running GuacaMol with profiling enabled...")
    print(f"  Budget: {budget}")
    print(f"  Population: {population_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max generations: {max_generations}")
    print(f"  Output: {profiling_output}")
    print()
    
    # Run the benchmark
    try:
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        exit_code = result.returncode
    except Exception as e:
        print(f"[Error] Benchmark execution failed: {e}")
        exit_code = 1
    
    # Save profiling results
    profiler = get_profiler()
    if profiler.timings:
        profiler.save_to_json(Path(profiling_output))
        profiler.print_summary()
        print(f"[Phase 0 Smoke Test] Profiling data saved to: {profiling_output}")
    else:
        print("[Warning] No profiling data collected.")
    
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 0 smoke test with profiling instrumentation"
    )
    parser.add_argument(
        "--output-dir",
        default="smoke_test_runs",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=1000,
        help="Molecule budget per task"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Population size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per generation"
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=5,
        help="Maximum generations"
    )
    parser.add_argument(
        "--benchmark",
        default="GuacaMol",
        help="Benchmark preset"
    )
    parser.add_argument(
        "--include-tasks",
        default="",
        help="Comma-separated task names to include (empty = use all)"
    )
    parser.add_argument(
        "--profiling-output",
        default=None,
        help="Path to save profiling JSON (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    exit_code = run_smoke_test(
        output_dir=args.output_dir,
        budget=args.budget,
        population_size=args.population_size,
        batch_size=args.batch_size,
        max_generations=args.max_generations,
        benchmark=args.benchmark,
        include_tasks=args.include_tasks,
        profiling_output=args.profiling_output,
    )
    sys.exit(exit_code)
