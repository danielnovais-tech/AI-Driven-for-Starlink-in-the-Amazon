#!/usr/bin/env python3
"""
Generate benchmark comparison figures and tables for scientific publication.

Runs ``simulate_long_duration.run_simulation()`` for multiple agent policies
and produces:

1. A Markdown comparison table (stdout and ``--output-md``).
2. A CSV data file (``--output-csv``) for use with Matplotlib / R / LaTeX.
3. An optional JSON dump of all raw simulation results (``--output-json``).

Policies evaluated:
    - ``random``   – uniform random satellite selection (lower bound baseline)
    - ``max_snr``  – greedy SNR-maximising heuristic (deterministic baseline)

The simulation uses a 1-day run with 100 satellites as the default for a
quick turnaround.  For publication quality, use ``--n-days 7``.

Usage::

    python scripts/generate_paper_figures.py \\
        --n-days 1 \\
        --n-satellites 100 \\
        --max-satellites 5 \\
        --seeds 42 43 44 \\
        --output-md /tmp/paper_table.md \\
        --output-csv /tmp/paper_data.csv \\
        --output-json /tmp/paper_data.json

Output (stdout excerpt)::

    | Policy    | Throughput (Mbps) | Outage Rate | Handover /min | P95 Latency (ms) |
    |-----------|-------------------|-------------|---------------|------------------|
    | random    |              0.00 |      1.0000 |        94.88  |            31.73 |
    | max_snr   |             12.45 |      0.2300 |        12.44  |            28.91 |
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from simulate_long_duration import run_simulation  # type: ignore[import]


# ---------------------------------------------------------------------------
# Policies to evaluate
# ---------------------------------------------------------------------------

_POLICIES = ["random", "max_snr"]

# KPI extraction helpers
_KPI_COLUMNS = [
    ("mean_throughput_mbps", "Throughput (Mbps)", ".2f"),
    ("outage_rate", "Outage Rate", ".4f"),
    ("handover_rate_per_min", "Handover /min", ".2f"),
    ("p95_latency_ms", "P95 Latency (ms)", ".2f"),
    ("mean_queue_delay_ms", "Queue Delay (ms)", ".2f"),
    ("mean_packet_drop_rate", "Drop Rate", ".4f"),
    ("total_compliance_violations", "Compliance Violations", "d"),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all(
    n_days: float,
    n_satellites: int,
    max_satellites: int,
    seeds: List[int],
    verbose: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run each policy for each seed, return nested dict {policy: [results]}."""
    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in _POLICIES}

    for policy in _POLICIES:
        for seed in seeds:
            if verbose:
                print(
                    f"  Running: policy={policy}  seed={seed}  "
                    f"n_days={n_days}  n_satellites={n_satellites} …",
                    flush=True,
                )
            result = run_simulation(
                n_days=n_days,
                n_satellites=n_satellites,
                max_satellites=max_satellites,
                seed=seed,
                policy=policy,
                verbose=False,
            )
            all_results[policy].append(result)
    return all_results


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average KPIs across multiple seeds."""
    import numpy as _np
    agg: Dict[str, float] = {}
    for key, _, _ in _KPI_COLUMNS:
        values = [r["kpis"][key] for r in results]
        agg[key] = float(_np.mean(values))
        agg[f"{key}_std"] = float(_np.std(values))
    return agg


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _build_table_rows(agg: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    rows = []
    for policy, stats in agg.items():
        row: Dict[str, Any] = {"policy": policy}
        for key, _, fmt in _KPI_COLUMNS:
            row[key] = stats[key]
            row[f"{key}_std"] = stats[f"{key}_std"]
        rows.append(row)
    return rows


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    header = ["Policy"] + [col for _, col, _ in _KPI_COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        cells = [row["policy"]]
        for key, _, fmt in _KPI_COLUMNS:
            if fmt == "d":
                cells.append(str(int(row[key])))
            else:
                cells.append(f"{row[key]:{fmt}}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _to_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = ["policy"] + [k for k, _, _ in _KPI_COLUMNS] + \
                 [f"{k}_std" for k, _, _ in _KPI_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate benchmark comparison figures for publication."
    )
    p.add_argument("--n-days", type=float, default=1.0,
                   help="Simulation duration in days per run (default 1).")
    p.add_argument("--n-satellites", type=int, default=100,
                   help="Constellation size (default 100).")
    p.add_argument("--max-satellites", type=int, default=5,
                   help="Max simultaneous visible satellites (default 5).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="Random seeds to average over (default: 42 43 44).")
    p.add_argument("--output-md", default=None,
                   help="Optional path to save Markdown table.")
    p.add_argument("--output-csv", default=None,
                   help="Optional path to save CSV data file.")
    p.add_argument("--output-json", default=None,
                   help="Optional path to save raw JSON results.")
    p.add_argument("--verbose", action="store_true",
                   help="Print progress during simulation runs.")
    args = p.parse_args(argv)

    print(f"Running {len(_POLICIES)} policies × {len(args.seeds)} seeds …")
    all_results = _run_all(
        n_days=args.n_days,
        n_satellites=args.n_satellites,
        max_satellites=args.max_satellites,
        seeds=args.seeds,
        verbose=args.verbose,
    )

    # Aggregate across seeds
    agg: Dict[str, Dict[str, float]] = {
        policy: _aggregate(results)
        for policy, results in all_results.items()
    }
    rows = _build_table_rows(agg)

    # Markdown output
    md = _to_markdown(rows)
    print()
    print("## Benchmark Results (averaged over seeds: {})".format(args.seeds))
    print()
    print(md)
    print()

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write("## Benchmark Results\n\n")
            f.write(f"**Seeds**: {args.seeds}  ")
            f.write(f"**n_days**: {args.n_days}  ")
            f.write(f"**n_satellites**: {args.n_satellites}\n\n")
            f.write(md + "\n")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        _to_csv(rows, args.output_csv)
        print(f"CSV saved to {args.output_csv}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        # Serialise raw results (replace non-serialisable numpy floats)
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            try:
                return float(obj)
            except (TypeError, ValueError):
                return obj

        output_doc = {
            "aggregated": agg,
            "raw": {p: [_clean(r) for r in rs] for p, rs in all_results.items()},
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(_clean(output_doc), f, indent=2)
        print(f"JSON saved to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
