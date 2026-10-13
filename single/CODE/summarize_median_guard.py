#!/usr/bin/env python3
"""Isolate the single added high-median rule from the waveform-guard audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from strict_single_node_scdt import crossing, plot_clean_survival, write_csv


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def isolate(rows: list[dict], coordinate: str) -> list[dict]:
    output = []
    for row in rows:
        survivors = int(row["old_survivors"]) - int(row["new_high_median_only_departures"])
        output.append({
            coordinate: float(row[coordinate]),
            "old_survivors": int(row["old_survivors"]),
            "new_high_median_only_departures": int(row["new_high_median_only_departures"]),
            "survivors": survivors,
            "ensemble": int(row["ensemble"]),
            "survival_rate": survivors / int(row["ensemble"]),
            **({"sigma_coordinate": float(row["sigma_coordinate"])} if coordinate == "x0" else {}),
        })
    return output


def main() -> None:
    rc = isolate(read_csv(ROOT / "data/scdt_rate_sweep_waveform_guard.csv"), "sigma_input")
    gt = isolate(read_csv(ROOT / "data/gt_boundary_waveform_guard.csv"), "x0")
    write_csv(ROOT / "data/scdt_rate_sweep_median_guard.csv", rc)
    write_csv(ROOT / "data/gt_boundary_median_guard.csv", gt)
    plot_clean_survival(
        ROOT / "figures/11_rate_with_median_guard.png", rc, gt,
        title="Single-node survival with median guard (exploratory)",
    )
    with (ROOT / "results/waveform_guard.json").open() as handle:
        combined = json.load(handle)
    with (ROOT / "results/waveform_guard_fresh_normal.json").open() as handle:
        fresh = json.load(handle)
    result = {
        "status": "exploratory post-hoc criterion on a locked model; BO not rerun",
        "event_rule": "old trough OR sustained low std OR 10-s rolling median above training-only ceiling for five consecutive 1-s steps",
        "median_reference": combined["training_only_median_reference"],
        "original_rc_sigma_50": combined["old_rc_sigma_50"],
        "median_guard_rc_sigma_50": crossing(
            np.asarray([r["sigma_input"] for r in rc]),
            np.asarray([r["survival_rate"] for r in rc]),
        ),
        "original_gt_x0_50": combined["old_gt_x0_50"],
        "median_guard_gt_x0_50": crossing(
            np.asarray([r["x0"] for r in gt]),
            np.asarray([r["survival_rate"] for r in gt]),
        ),
        "original_rebound_survivors": rc[31]["old_survivors"],
        "median_guard_rebound_survivors": rc[31]["survivors"],
        "fresh_normal_gt_old_departures": fresh["gt_old_departures"],
        "fresh_normal_gt_new_departures": fresh["gt_new_departures"],
        "fresh_normal_gt_total": fresh["gt_total"],
        "fresh_normal_rc_old_departures": fresh["rc_old_departures"],
        "fresh_normal_rc_median_guard_departures": fresh["rc_median_only_departures"],
        "fresh_normal_rc_total": fresh["rc_total"],
        "fresh_rc_sweep_checks": [
            {key: value for key, value in row.items() if key != "new_survivors"}
            for row in fresh["fresh_rc_sweep_checks"]
        ],
    }
    with (ROOT / "results/median_guard.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
