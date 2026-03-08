#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse MIP scaling benchmark logs into CSV.

Reads log output (from stdin or file arguments) produced by the cuOpt MIP
solver with the MIP_SCALING_METRICS / MIP_SCALING_SUMMARY / MIP_OBJ_SCALING /
MIP_GAP_METRICS log lines and the standard solver output.

Usage:
    python parse_scaling_logs.py < benchmark_output.log > results.csv
    python parse_scaling_logs.py log1.log log2.log > results.csv
"""

import csv
import re
import sys
from collections import defaultdict


def _float_or_na(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return ""


def _int_or_na(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return ""


INSTANCE_RE = re.compile(
    r"(?:Reading|Solving|instance[=:]\s*)(\S+?)(?:\.mps)?(?:\s|$)", re.I
)
FEASIBLE_RE = re.compile(
    r"Found new best solution.*objective[=:\s]+([\d.eE+\-]+)", re.I
)
OPTIMAL_RE = re.compile(r"Optimal solution found", re.I)
OBJ_RE = re.compile(r"Objective\s+([\d.eE+\-]+)", re.I)
REL_GAP_RE = re.compile(r"relative_mip_gap\s+([\d.eE+\-]+)", re.I)
SIMPLEX_RE = re.compile(r"simplex_iterations\s+([\d,]+)", re.I)
NODES_RE = re.compile(r"Explored\s+(\d+)\s+nodes\s+in\s+([\d.]+)s", re.I)
INFEASIBLE_RE = re.compile(r"Integer infeasible|Infeasible", re.I)

SCALING_METRICS_RE = re.compile(
    r"MIP_SCALING_METRICS\s+iteration=(\d+)\s+log2_spread=([\d.eE+\-]+)\s+"
    r"target_norm=([\d.eE+\-]+)\s+scaled_rows=(\d+)\s+valid_rows=(\d+)"
)
SCALING_SUMMARY_RE = re.compile(
    r"MIP_SCALING_SUMMARY\s+rows=(\d+)\s+bigm_rows=(\d+)\s+final_spread=([\d.eE+\-inf]+)"
)
OBJ_SCALING_RE = re.compile(
    r"MIP_OBJ_SCALING\s+(applied|skipped).*?(?:scale=([\d.eE+\-]+))?"
    r".*?(?:new_scaling_factor=([\d.eE+\-]+))?"
)
GAP_METRICS_RE = re.compile(
    r"MIP_GAP_METRICS\s+abs_gap_user=([\d.eE+\-]+)\s+rel_gap=([\d.eE+\-]+)\s+"
    r"obj_user=([\d.eE+\-]+)\s+bound_user=([\d.eE+\-]+)\s+obj_scale=([\d.eE+\-]+)"
)
SCALING_SKIPPED_RE = re.compile(r"MIP row scaling skipped", re.I)
SOL_FOUND_RE = re.compile(r"sol_found=(\d+).*?obj_val=([\d.eE+\-inf]+)", re.I)
RUN_MPS_RE = re.compile(r"run_mps\s+(\S+)", re.I)


def parse_logs(lines):
    records = []
    cur = defaultdict(lambda: "")
    instance_name = ""

    def flush():
        nonlocal instance_name
        if instance_name:
            cur["instance"] = instance_name
            records.append(dict(cur))
        cur.clear()
        instance_name = ""

    for line in lines:
        line = line.rstrip("\n")

        m = RUN_MPS_RE.search(line)
        if m:
            flush()
            instance_name = m.group(1).replace(".mps", "")
            continue

        m = INSTANCE_RE.search(line)
        if m and not instance_name:
            instance_name = m.group(1)

        m = SCALING_SKIPPED_RE.search(line)
        if m:
            cur["scaling_applied"] = "no"

        m = SCALING_METRICS_RE.search(line)
        if m:
            cur["scaling_applied"] = "yes"
            cur["scaling_last_iteration"] = m.group(1)
            cur["scaling_last_spread"] = m.group(2)
            cur["scaling_target_norm"] = m.group(3)

        m = SCALING_SUMMARY_RE.search(line)
        if m:
            cur["rows"] = m.group(1)
            cur["bigm_rows"] = m.group(2)
            cur["final_spread"] = m.group(3)

        m = OBJ_SCALING_RE.search(line)
        if m:
            cur["obj_scaling_status"] = m.group(1)
            if m.group(2):
                cur["obj_scaling_factor"] = m.group(2)
            if m.group(3):
                cur["obj_new_scaling_factor"] = m.group(3)

        m = FEASIBLE_RE.search(line)
        if m:
            cur["feasible"] = 1
            cur["objective"] = m.group(1)

        m = OPTIMAL_RE.search(line)
        if m:
            cur["optimal"] = 1

        m = NODES_RE.search(line)
        if m:
            cur["nodes_explored"] = m.group(1)
            cur["solve_time_s"] = m.group(2)

        m = SIMPLEX_RE.search(line)
        if m:
            cur["simplex_iters"] = m.group(1).replace(",", "")

        m = GAP_METRICS_RE.search(line)
        if m:
            cur["abs_gap_user"] = m.group(1)
            cur["rel_gap"] = m.group(2)
            cur["obj_user"] = m.group(3)
            cur["bound_user"] = m.group(4)
            cur["obj_scale"] = m.group(5)

        m = SOL_FOUND_RE.search(line)
        if m:
            cur["feasible"] = int(m.group(1))
            cur["objective"] = m.group(2)

        m = INFEASIBLE_RE.search(line)
        if m:
            cur["feasible"] = 0

    flush()
    return records


COLUMNS = [
    "instance",
    "feasible",
    "optimal",
    "objective",
    "rel_gap",
    "abs_gap_user",
    "obj_user",
    "bound_user",
    "solve_time_s",
    "simplex_iters",
    "nodes_explored",
    "scaling_applied",
    "bigm_rows",
    "rows",
    "final_spread",
    "scaling_last_iteration",
    "scaling_target_norm",
    "obj_scaling_status",
    "obj_scaling_factor",
    "obj_new_scaling_factor",
    "obj_scale",
]


def main():
    if len(sys.argv) > 1:
        lines = []
        for path in sys.argv[1:]:
            with open(path) as f:
                lines.extend(f.readlines())
    else:
        lines = sys.stdin.readlines()

    records = parse_logs(lines)

    writer = csv.DictWriter(
        sys.stdout, fieldnames=COLUMNS, extrasaction="ignore"
    )
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)

    n_feasible = sum(1 for r in records if r.get("feasible") == 1)
    n_optimal = sum(1 for r in records if r.get("optimal") == 1)
    n_total = len(records)
    print(
        f"# Summary: {n_total} instances, {n_feasible} feasible, {n_optimal} optimal",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
