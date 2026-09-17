#!/usr/bin/env python3
"""
Utility to read and display Phase 0 profiling results.
"""

import json
import sys
from pathlib import Path

def read_profiling_report(run_dir: str):
    """Read and display profiling summary JSON."""
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"[Error] Run directory not found: {run_dir}")
        return 1
    
    # Find profiling_summary.json
    profiling_file = None
    for f in run_path.rglob("profiling_summary.json"):
        profiling_file = f
        break
    
    if not profiling_file:
        print(f"[Error] No profiling_summary.json found in {run_dir}")
        print(f"[Info] Directory contents:")
        for item in sorted(run_path.rglob("*")):
            if item.is_file():
                print(f"  {item.relative_to(run_path)}")
        return 1
    
    print(f"[Info] Found profiling file: {profiling_file}")
    print()
    
    with open(profiling_file, "r") as f:
        data = json.load(f)
    
    # Display report
    print("=" * 100)
    print("Phase 0 Instrumentation Report - Wall-Time Profiling Summary")
    print("=" * 100)
    print()
    
    # Sort by total time descending
    sorted_stages = sorted(
        data.items(),
        key=lambda x: x[1].get("total_time_sec", 0),
        reverse=True
    )
    
    total_time = sum(s[1].get("total_time_sec", 0) for s in sorted_stages)
    
    print(f"{'Stage':<50} {'Total (s)':<12} {'Calls':<8} {'Avg (s)':<12} {'%':<6}")
    print("-" * 100)
    
    for stage, stats in sorted_stages:
        total_s = stats.get("total_time_sec", 0)
        calls = stats.get("call_count", 0)
        avg_s = stats.get("avg_time_sec", 0)
        pct = 100 * total_s / total_time if total_time > 0 else 0
        
        print(
            f"{stage:<50} {total_s:<12.6f} {calls:<8} {avg_s:<12.6f} {pct:<6.1f}%"
        )
    
    print("-" * 100)
    print(f"{'TOTAL':<50} {total_time:<12.6f}")
    print()
    
    # Top 3 contributors
    print("Top 3 Bottlenecks:")
    for i, (stage, stats) in enumerate(sorted_stages[:3], 1):
        pct = 100 * stats.get("total_time_sec", 0) / total_time if total_time > 0 else 0
        print(f"  {i}. {stage}: {stats.get('total_time_sec', 0):.3f}s ({pct:.1f}%)")
    
    print()
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_profiling_report.py <run_dir>")
        print("Example: python read_profiling_report.py /path/to/smoke_test_run_002")
        sys.exit(1)
    
    exit_code = read_profiling_report(sys.argv[1])
    sys.exit(exit_code)
