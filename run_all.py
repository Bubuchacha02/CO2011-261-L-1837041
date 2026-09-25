#!/usr/bin/env python3
"""
run_all.py  -  THE single entry point for the CO2011 SEM261 assignment.

Grading runs exactly:  python run_all.py --seed $(cat data/seed.txt)
It must reproduce EVERY number in your report from a clean clone, with no manual steps.
Fill in each stage to call your module code; keep the CLI and the stage order stable.

This skeleton ships in the course template repository ("Use this template", not a fork).
"""
import argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="from data/seed.txt")
    ap.add_argument("--stage", default="all",
                    choices=["all", "m1", "m2", "m3", "m4", "m5"])
    a = ap.parse_args()

    if a.stage in ("all", "m1"):
        pass  # TODO: m1_logic  -> SAT feasibility / unsat core, logic->LP table
    if a.stage in ("all", "m2"):
        pass  # TODO: m2_ilp    -> build & solve the seeded ILP, report fairness vs baseline
    if a.stage in ("all", "m3"):
        pass  # TODO: m3_automata-> load DFAs, product/minimization, regular->ILP, pumping
    if a.stage in ("all", "m4"):
        pass  # TODO: m4_dynamics-> recurrence, equilibrium, stability, plots
    if a.stage in ("all", "m5"):
        pass  # TODO: (optional) precompute artifacts the Streamlit app loads
    print(f"[run_all] seed={a.seed} stage={a.stage}: fill in each stage.")

if __name__ == "__main__":
    main()
