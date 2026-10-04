#!/usr/bin/env python3
"""
run_all.py  -  THE single entry point for the CO2011 SEM261 assignment.

Grading runs exactly:  python run_all.py --seed $(cat data/seed.txt)
It must reproduce EVERY number in your report from a clean clone, with no manual steps.
Fill in each stage to call your module code; keep the CLI and the stage order stable.

This skeleton ships in the course template repository ("Use this template", not a fork).
"""
import argparse, sys
from pathlib import Path
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="from data/seed.txt")
    ap.add_argument("--stage", default="all",
                    choices=["all", "m1", "m2", "m3", "m4", "m5"])
    a = ap.parse_args()

    if a.stage in ("all", "m1"):
        pass  # TODO: m1_logic  -> SAT feasibility / unsat core, logic->LP table
        from m1_logic.m1 import run_pipeline          # <-- m1_logic.cnf -> m1_logic.m1
        run_pipeline(
            mode="all",
            data=Path("data/Dataset_Anonymized_Invigilator_Assignment_Problem.xlsx"),
            seed=a.seed,
            json_path=Path("m1_logic/m1_report.json"),
        )
    if a.stage in ("all", "m2"):
         # m2_ilp    -> build & solve the seeded ILP, report fairness vs baseline
        # (weights, preferences, everything random come from --seed; outputs -> m2_ilp/outputs/)
        from m2_ilp.run_m2 import run_pipeline as run_m2
        r = run_m2(data=data, seed=a.seed, out_dir=Path("m2_ilp/outputs"))
        if not r.get("ok"):
            sys.exit("[run_all] M2 checks FAILED - see the log above")
    if a.stage in ("all", "m3"):
        pass  # TODO: m3_automata-> load DFAs, product/minimization, regular->ILP, pumping
    if a.stage in ("all", "m4"):
        pass  # TODO: m4_dynamics-> recurrence, equilibrium, stability, plots
    if a.stage in ("all", "m5"):
        pass  # TODO: (optional) precompute artifacts the Streamlit app loads
    print(f"[run_all] seed={a.seed} stage={a.stage}: fill in each stage.")

if __name__ == "__main__":
    main()
