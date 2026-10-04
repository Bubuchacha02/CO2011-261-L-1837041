"""
m2_ilp/run_m2.py  --  Module 2 entry point: reproduces EVERY Module-2 number.

    python m2_ilp/run_m2.py --seed $(cat data/seed.txt)
    python m2_ilp/run_m2.py --seed 12345 --quick          # skip the slow studies
    python -m m2_ilp.run_m2 --seed 12345                  # same thing

Pipeline (requirement -> function)
    2.1 2.6  data.load_instance          parameters + seeded simulated preferences / weights
    2.2-2.5  model.build_model           variables, objective, hard (via M1) + soft constraints
    2.7 2.8  tuning.solve_with_tuning    solve, relax if infeasible, re-weight if not better
    2.9      analysis.*                  baseline comparison, fairness modes, performance
    2.10     counterexample.*            two wrong models + concrete invalid schedules

Also usable from run_all.py:   from m2_ilp.run_m2 import run_pipeline
"""


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from m2_ilp.analysis import (compare_to_baseline, fairness_comparison, performance_study,
                             worked_example)
from m2_ilp.counterexample import run_counterexamples
from m2_ilp.data import DEFAULT_DATA_PATH, dump_parameters, load_instance
from m2_ilp.model import evaluate, loads_of, validate_schedule
from m2_ilp.tuning import infeasibility_demo, solve_with_tuning


def write_schedule(inst, assign, out_dir: Path, name: str) -> None:
    rows = []
    for (i, j) in sorted(assign, key=lambda t: (inst.sessions[t[1]].start, t[1], t[0])):
        s = inst.sessions[j]
        rows.append({"invigilator": i, "session": j, "shift_id": s.shift_id,
                     "date": s.date.isoformat(), "slot": s.slot, "period": s.period,
                     "campus": s.campus, "in_baseline": (i, j) in inst.baseline})
    pd.DataFrame(rows).to_csv(out_dir / name, index=False)


def run_pipeline(data: Path = DEFAULT_DATA_PATH, seed: Optional[int] = None,
                 team_id: Optional[str] = None,
                 weights: Optional[Tuple[float, float, float]] = None,
                 fairness: str = "l1", availability: str = "day", time_limit: float = 120,
                 tighten: bool = True, out_dir: Optional[Path] = None,
                 quick: bool = False) -> dict:
    out_dir = Path(out_dir or ROOT / "m2_ilp" / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    # ---- 2.1 / 2.6 -----------------------------------------------------------------
    inst = load_instance(data, seed, team_id, weights, availability)
    dump_parameters(inst, out_dir)
    R["instance"] = {"invigilators": len(inst.invigilators), "sessions": len(inst.sessions),
                     "assignments": inst.total_demand, "mean_load": inst.mean_load,
                     "overlap_pairs": len(inst.overlaps), "busy_pairs": len(inst.busy),
                     "availability": inst.availability_mode, "seed": inst.seed,
                     "weights": inst.weights, "weights_source": inst.notes["weights_source"],
                     "weights_mapping": "w1->fairness, w2->fatigue, w3->location"}

    R["worked_example"] = worked_example()

    # ---- 2.7 / 2.8 -------------------------------------------------------------------
    res, solved_inst, log = solve_with_tuning(inst, fairness, tighten, time_limit)
    R["tuning_log"] = log
    if not res.has_solution:
        print("!! no feasible schedule found")
        R["ok"] = False
        _dump(R, out_dir)
        return R

    R["solve"] = {"status": res.status, "objective": res.objective, "seconds": round(res.runtime, 3),
                  "solver": res.solver, "fairness": fairness, "model_size": res.model.stats,
                  "objective_terms": res.terms, "availability_used": solved_inst.availability_mode}
    print("-" * 78)
    print(f"M2 / 2.7  SOLVE   status={res.status}  Z={res.objective:.4f}  "
          f"{res.runtime:.2f}s  solver={res.solver}")
    print(f"          terms: fairness F={res.terms['fair']:.3f}  location L={res.terms['loc']:.0f}  "
          f"fatigue T={res.terms['fat']:.0f}")
    print(f"          size : {res.model.stats}")

    viol = validate_schedule(solved_inst, res.assignment)
    R["hard_rule_violations"] = viol
    print(f"          independent hard-rule check (R1,R2,R3): "
          f"{'PASS' if not viol else 'FAIL ' + str(viol[:3])}")
    # model objective must equal the independent evaluation
    ev = evaluate(solved_inst, res.assignment, fairness, res.model.weights)
    R["objective_matches_independent_evaluation"] = abs(ev["objective"] - res.objective) < 1e-4
    print(f"          model objective == independent evaluation: "
          f"{R['objective_matches_independent_evaluation']}")

    write_schedule(inst, res.assignment, out_dir, "schedule_optimized.csv")
    write_schedule(inst, inst.baseline, out_dir, "schedule_baseline.csv")
    res.model.prob.writeLP(str(out_dir / "model.lp"))
    b_l, n_l = loads_of(inst, inst.baseline), loads_of(inst, res.assignment)
    pd.DataFrame([{"invigilator": i, "baseline_load": b_l[i], "optimised_load": n_l[i],
                   "preference": inst.pref[i]} for i in inst.invigilators]
                 ).to_csv(out_dir / "loads_baseline_vs_optimised.csv", index=False)

    # ---- 2.9 -----------------------------------------------------------------------------
    R["baseline_comparison"] = compare_to_baseline(inst, res.assignment, fairness)
    if not quick:
        R["fairness_modes"] = fairness_comparison(solved_inst, min(time_limit, 60))

    # ---- 2.10 ------------------------------------------------------------------------------
    z = res.objective if fairness == "l1" else None
    if z is None:                                   # counter-examples are stated for the L1 model
        from m2_ilp.model import build_model
        from m2_ilp.tuning import solve
        z = solve(build_model(solved_inst, "l1", None, True), time_limit).objective
    R["counterexamples"] = run_counterexamples(solved_inst, z, min(time_limit, 60))

    # ---- 2.9 performance + 2.8 relax demo ------------------------------------------------------
    if not quick:
        R["performance"] = performance_study(solved_inst, z, min(time_limit, 60), no_cut_time_limit=60)
        print("-" * 78)
        print("M2 / 2.8  RELAX PATH DEMO (declared, seeded sick-leave shock)")
        print("-" * 78)
        R["infeasibility_demo"] = infeasibility_demo(inst, min(time_limit, 60))

    R["ok"] = bool(not viol and R["baseline_comparison"]["beats_baseline_strictly"]
                   and R["counterexamples"]["A_drop_overlap"]["conclusion_holds"]
                   and R["counterexamples"]["B_lp_relaxation"]["conclusion_holds"])
    print("=" * 78)
    print(f"M2 OVERALL: {'ALL CHECKS PASS' if R['ok'] else 'SOME CHECK FAILED - read the log above'}")
    print(f"outputs in {out_dir}")
    _dump(R, out_dir)
    return R


def _dump(R: dict, out_dir: Path) -> None:
    def conv(o):
        if isinstance(o, (set, frozenset)):
            return sorted(map(str, o))
        return str(o)
    (out_dir / "results_m2.json").write_text(json.dumps(R, indent=2, default=conv), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 2 - ILP for the Invigilator Assignment Problem")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--seed", type=int, default=None, help="team seed (default: data/seed.txt)")
    ap.add_argument("--team-id", type=str, default=None,
                    help="optional: only to CHECK that it hashes to the seed in use")
    ap.add_argument("--weights", type=float, nargs=3, metavar=("W1_FAIR", "W2_FAT", "W3_LOC"),
                    default=None, help="override the 3 weights, in seed order w1 w2 w3 = fairness fatigue location")
    ap.add_argument("--fairness", choices=["l1", "minmax", "spread"], default="l1")
    ap.add_argument("--availability", choices=["day", "week", "all"], default="day")
    ap.add_argument("--time-limit", type=float, default=120)
    ap.add_argument("--no-tighten", action="store_true", help="disable the chord cut")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="skip performance / fairness-mode studies")
    a = ap.parse_args()
    R = run_pipeline(a.data, a.seed, a.team_id, tuple(a.weights) if a.weights else None,
                     a.fairness, a.availability, a.time_limit, not a.no_tighten, a.out_dir, a.quick)
    sys.exit(0 if R.get("ok") else 1)


if __name__ == "__main__":
    main()