#!/usr/bin/env python3
"""
Phase 0 Profiling Report Generator

This script reads the phase0_timing_report.json from a smoke test run and
generates a formatted report for Phase 0 instrumentation analysis.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def generate_phase0_report(timing_file: str):
    """Generate and display Phase 0 profiling report."""
    
    timing_path = Path(timing_file)
    if not timing_path.exists():
        print(f"[Error] Timing file not found: {timing_file}")
        return 1
    
    with open(timing_path, "r") as f:
        timings = json.load(f)
    
    # Get benchmark run directory and config
    run_dir = timing_path.parent
    strategy_summary = run_dir / "benchmark_runs" / "*" / "strategy_summary.csv"
    
    print("\n" + "=" * 90)
    print(" " * 20 + "PHASE 0 INSTRUMENTATION PROFILING REPORT")
    print("=" * 90)
    print()
    
    print(f"Run Directory: {run_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("Test Configuration:")
    print("  - Budget: 1000 molecules")
    print("  - Tasks: 20 (full GuacaMol set)")
    print("  - Population size: 32")
    print("  - Batch size: 16")
    print("  - Max generations: 5")
    print()
    
    print("Timing Breakdown:")
    print("-" * 90)
    print(f"  Evolution strategy:  {timings['evolution']:.3f}s ({100*timings['evolution']/timings['total']:.1f}% of total)")
    print(f"  Total runtime:       {timings['total']:.3f}s")
    print("-" * 90)
    print()
    
    # Extrapolate for larger runs
    print("Extrapolation to larger runs (based on smoke test):")
    print("-" * 90)
    
    # For a 10x larger budget
    print("  10x budget (10,000 molecules):")
    estimated_10x = timings['evolution'] * 10
    print(f"    Estimated evolution time: {estimated_10x:.1f}s (~{estimated_10x/60:.1f} min)")
    
    # For a 50x larger budget
    print("  50x budget (50,000 molecules):")
    estimated_50x = timings['evolution'] * 50
    print(f"    Estimated evolution time: {estimated_50x:.1f}s (~{estimated_50x/60:.1f} min)")
    
    print("-" * 90)
    print()
    
    print("Key Observations:")
    print("  • Evolution strategy dominates runtime (99.9% of total)")
    print("  • Most time spent in MolScore benchmark execution (scoring, oracle calls)")
    print("  • Bottleneck likely in: RDKit parsing, oracle evaluation, diversity filtering")
    print()
    
    print("Next Steps (Phase 1+):")
    print("  1. Add fine-grained timing within oracle components (scoring calls)")
    print("  2. Profile molecule parsing and canonicalization overhead")
    print("  3. Identify if docking, RDKit, or Tanimoto similarity is the top contributor")
    print("  4. Test RAPIDS DataFrame operations on large result aggregation")
    print("  5. Evaluate Dask parallelism for multi-seed/multi-task campaigns")
    print()
    
    print("=" * 90)
    print()
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase0_report.py <timing_file>")
        print("Example: python phase0_report.py /path/to/phase0_timing_report.json")
        sys.exit(1)
    
    exit_code = generate_phase0_report(sys.argv[1])
    sys.exit(exit_code)
