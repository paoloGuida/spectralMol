"""
Profiling utilities for instrumentation of hot paths.

Provides a global profiler for collecting wall-time and call counts
across scoring, offspring generation, and report generation stages.
"""

import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict


class Profiler:
    """Collects timing and call-count statistics for named stages."""
    
    def __init__(self):
        self.timings: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_time": 0.0,
            "call_count": 0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "start_time": None,
        })
        self.enabled = True
    
    @contextmanager
    def timer(self, stage_name: str):
        """Context manager to time a code block."""
        if not self.enabled:
            yield
            return
        
        stats = self.timings[stage_name]
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stats["total_time"] += elapsed
            stats["call_count"] += 1
            stats["min_time"] = min(stats["min_time"], elapsed)
            stats["max_time"] = max(stats["max_time"], elapsed)
    
    def reset(self):
        """Clear all timing data."""
        self.timings.clear()
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return summary stats for all stages."""
        summary = {}
        for stage, stats in self.timings.items():
            if stats["call_count"] > 0:
                summary[stage] = {
                    "total_time_sec": round(stats["total_time"], 6),
                    "call_count": stats["call_count"],
                    "avg_time_sec": round(stats["total_time"] / stats["call_count"], 6),
                    "min_time_sec": round(stats["min_time"], 6),
                    "max_time_sec": round(stats["max_time"], 6),
                }
        return summary
    
    def save_to_json(self, filepath: Path):
        """Persist timing summary to JSON."""
        summary = self.get_summary()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Profiler] Saved timings to {filepath}")
    
    def print_summary(self):
        """Print human-readable summary to stdout."""
        summary = self.get_summary()
        if not summary:
            print("[Profiler] No timing data collected.")
            return
        
        print("\n=== Phase 0 Instrumentation Summary ===")
        print(f"{'Stage':<40} {'Total (s)':<12} {'Calls':<8} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        print("-" * 96)
        
        # Sort by total time descending
        sorted_stages = sorted(summary.items(), key=lambda x: x[1]["total_time_sec"], reverse=True)
        total_measured = sum(s[1]["total_time_sec"] for s in sorted_stages)
        
        for stage, stats in sorted_stages:
            pct = 100 * stats["total_time_sec"] / total_measured if total_measured > 0 else 0
            print(
                f"{stage:<40} {stats['total_time_sec']:<12.6f} "
                f"{stats['call_count']:<8} {stats['avg_time_sec']:<12.6f} "
                f"{stats['min_time_sec']:<12.6f} {stats['max_time_sec']:<12.6f} ({pct:.1f}%)"
            )
        print("-" * 96)
        print(f"{'TOTAL':<40} {total_measured:<12.6f}")
        print()


# Global profiler instance
_global_profiler = Profiler()


def get_profiler() -> Profiler:
    """Return the global profiler instance."""
    return _global_profiler


def enable_profiling():
    """Enable global profiler."""
    _global_profiler.enabled = True


def disable_profiling():
    """Disable global profiler."""
    _global_profiler.enabled = False
