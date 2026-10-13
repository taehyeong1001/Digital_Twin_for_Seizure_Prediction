"""Arm B for the single node: member-0 reservoir x K innovation rollouts (docs/04).

Uses the phase-2 cohorts (coarse, fine) and their cache, so arm B is paired
with the ensemble trial for trial. Output: outputs/single_node/phase2/armB/.

Run:  .venv/bin/python survival_dist/run_arm_b_single_node.py [--rollouts 20] [--workers 6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ensemble import run_arm_b  # noqa: E402
from run_phase2_single_node import OUT, cohort_specs  # noqa: E402
from single_node_setup import load_setup  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rc_specs, _ = cohort_specs(load_setup())
    run_arm_b(a.rollouts, rc_specs, OUT / "armB", OUT / "cohorts", workers=a.workers)


if __name__ == "__main__":
    main()
