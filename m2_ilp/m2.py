"""
m2_ilp/m2.py  --  the grader-facing interface of Module 2
================================================================================
Two output files, always written to  m2_ilp/  (relative to the working directory):

    m2_ilp/schedule.csv   columns EXACTLY  invigilator_id, session_id, assigned
                          one row per (invigilator, session) = |I| x |J| rows,
                          assigned in {0, 1};  session_id = the "MS Ca thi" of the input file
    m2_ilp/metrics.json   keys EXACTLY  max_load (int), baseline_max_load (int),
                          beats_baseline (bool)  with  beats_baseline = max_load < baseline_max_load

Notes
  * Internally the model works at (shift, campus) level (location preference needs the campus).
    The exported schedule is aggregated back to the SHIFT id: assigned(i, shift) = 1 iff i works
    the shift on either campus.  Rule R1 (no double-booking) guarantees at most one campus.
  * `availabilities` = the (invigilator_id, session_id) pairs that are NOT available (busy);
    the schedule never assigns any of them.
  * `soft_weights` = [w1, w2, w3] = [fairness, fatigue, location].

Functions
  export_schedule_metrics   write the two files from a solved assignment
  solve_m2                  input file + busy pairs + weights  ->  solve  ->  the two files
  simulate_m2_ilp           same signature as the unit test; thin wrapper of solve_m2
                            (instructions: delete it once the test passes; run_all.py -> run_m2.py
                             writes the same two files with seed-derived availabilities and weights)
================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from m2_ilp.data import IAPInstance, load_instance
from m2_ilp.model import loads_of
from m2_ilp.tuning import solve_with_tuning

SCHEDULE_COLUMNS = ["invigilator_id", "session_id", "assigned"]
DEFAULT_EXPORT_DIR = Path("m2_ilp")


def export_schedule_metrics(inst: IAPInstance, assignment, out_dir: Path = DEFAULT_EXPORT_DIR) -> Dict:
    """Write schedule.csv and metrics.json; returns the metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_start = {}
    for s in inst.sessions.values():
        first_start[s.shift_id] = min(first_start.get(s.shift_id, s.start), s.start)
    shifts = sorted(first_start, key=lambda k: (first_start[k], k))

    works = {(i, inst.sessions[j].shift_id) for (i, j) in assignment}
    rows = [{"invigilator_id": i, "session_id": sid, "assigned": int((i, sid) in works)}
            for i in inst.invigilators for sid in shifts]
    df = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    df.to_csv(out_dir / "schedule.csv", index=False)

    max_load = int(df.groupby("invigilator_id")["assigned"].sum().max())
    baseline_max = int(max(loads_of(inst, inst.baseline).values()))
    metrics = {"max_load": max_load, "baseline_max_load": baseline_max,
               "beats_baseline": bool(max_load < baseline_max)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def solve_m2(input_file, availabilities: Sequence[Tuple[str, str]], soft_weights: Sequence[float],
             seed: Optional[int] = None, out_dir: Path = DEFAULT_EXPORT_DIR,
             time_limit: float = 60, verbose: bool = False) -> Dict:
    if len(soft_weights) != 3:
        raise ValueError("soft_weights must be [w1_fairness, w2_fatigue, w3_location]")
    inst = load_instance(Path(input_file), seed=seed, weights=tuple(float(w) for w in soft_weights),
                         busy_pairs=availabilities, verbose=verbose)
    res, _used, _log = solve_with_tuning(inst, "l1", True, time_limit, verbose=verbose)
    if not res.has_solution:
        raise RuntimeError("no feasible schedule for the given availabilities")
    return export_schedule_metrics(inst, res.assignment, out_dir)


def simulate_m2_ilp(input_file, availabilities, soft_weights) -> Dict:
    """Stand-in used by the unit test (same keyword arguments)."""
    return solve_m2(input_file, availabilities, soft_weights)
