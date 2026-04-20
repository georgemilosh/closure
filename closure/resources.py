"""Runtime resource helpers for logging RAM and GPU usage."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import psutil
import torch


def _read_int_file(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _self_cgroup_path() -> str | None:
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                if parts[0] == "0":
                    return parts[2]
    except OSError:
        return None
    return None


def cgroup_memory_usage_bytes() -> int | None:
    """Return memory usage from cgroup accounting for current process.

    Tries cgroup v2 first (memory.current), then cgroup v1
    (memory.usage_in_bytes), and falls back to None if unavailable.
    """
    cg_rel = _self_cgroup_path()
    if cg_rel:
        cg_rel = cg_rel.strip()
        if not cg_rel.startswith("/"):
            cg_rel = f"/{cg_rel}"

        # cgroup v2
        v2_path = Path(f"/sys/fs/cgroup{cg_rel}/memory.current")
        value = _read_int_file(v2_path)
        if value is not None:
            return value

        # cgroup v1
        v1_path = Path(f"/sys/fs/cgroup/memory{cg_rel}/memory.usage_in_bytes")
        value = _read_int_file(v1_path)
        if value is not None:
            return value

    # Fallback probe for uncommon setups where process cgroup path lookup fails.
    value = _read_int_file(Path("/sys/fs/cgroup/memory.current"))
    if value is not None:
        return value

    value = _read_int_file(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    if value is not None:
        return value

    return None


def process_tree_ram_bytes() -> int:
    """Return RSS bytes for current process plus children."""
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def process_tree_ram_gb() -> float:
    """Return RAM usage in GiB, preferring cgroup accounting.

    When running under Slurm cgroups, this reflects memory charged to the job
    step and avoids large over-counting from summing process RSS values.
    Falls back to process-tree RSS when cgroup files are unavailable.
    """
    cgroup_bytes = cgroup_memory_usage_bytes()
    if cgroup_bytes is not None:
        return cgroup_bytes / (1024.0 ** 3)
    return process_tree_ram_bytes() / (1024.0 ** 3)


def gpu_stats() -> list[dict[str, Any]]:
    """Return per-GPU utilization/memory data for visible devices.

    Preferred source is ``nvidia-smi`` because it exposes both utilization and
    memory. Falls back to torch memory counters when unavailable.
    """
    if not torch.cuda.is_available():
        return []

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "utilization_pct": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                }
            )
        if rows:
            return rows
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    rows = []
    for dev in range(torch.cuda.device_count()):
        used_mb = torch.cuda.memory_allocated(dev) / (1024.0 ** 2)
        total_mb = torch.cuda.get_device_properties(dev).total_memory / (1024.0 ** 2)
        rows.append(
            {
                "index": dev,
                "utilization_pct": None,
                "memory_used_mb": used_mb,
                "memory_total_mb": total_mb,
            }
        )
    return rows


def aggregate_gpu_stats(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Return average utilization and memory usage across provided GPU rows."""
    if not rows:
        return {
            "avg_gpu_utilization_pct": None,
            "avg_gpu_memory_used_mb": None,
            "avg_gpu_memory_total_mb": None,
        }

    util_values = [r["utilization_pct"] for r in rows if r.get("utilization_pct") is not None]
    mem_used_values = [r["memory_used_mb"] for r in rows if r.get("memory_used_mb") is not None]
    mem_total_values = [r["memory_total_mb"] for r in rows if r.get("memory_total_mb") is not None]

    return {
        "avg_gpu_utilization_pct": (sum(util_values) / len(util_values)) if util_values else None,
        "avg_gpu_memory_used_mb": (sum(mem_used_values) / len(mem_used_values)) if mem_used_values else None,
        "avg_gpu_memory_total_mb": (sum(mem_total_values) / len(mem_total_values)) if mem_total_values else None,
    }