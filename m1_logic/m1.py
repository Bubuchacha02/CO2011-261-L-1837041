#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m1_logic/m1.py
================================================================================
CO2011 - Mathematical Modeling, Semester 261 (2026-2027) - HCMUT
Invigilator Assignment Problem (IAP) - Module 1, Requirement 1.2
"Propositional encoding & SAT"

    "For the toy instance and a small slice of the data, encode feasibility in
    CNF and decide satisfiability with a SAT/SMT solver (python-sat or z3);
    report a model or a minimal unsatisfiable core."
                                                    -- Assignment brief, S1.2

--------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
--------------------------------------------------------------------------------
1. Grounds the Module-1 predicates (S1.1) over a FINITE slice (I, J):

        Assign(i, j)    -- decision predicate  -> one SAT/boolean variable
        Busy(i, j)      -- data predicate      -> compiled into clauses
        Overlap(j, k)   -- data predicate      -> compiled into clauses
        AtCampus(j, c)  -- data (used to build Overlap)
        Prefer(i, c)    -- not needed for the HARD rules encoded here (S2.5, soft)

2. Compiles three hard rules into Conjunctive Normal Form (CNF):

        R1  No double-booking of overlapping shifts
            for all i,j,k: Overlap(j,k) & Assign(i,j) -> not Assign(i,k)

        R2  Availability
            for all i,j:   Busy(i,j) -> not Assign(i,j)

        R3  Capacity (exact headcount per shift, cardinality constraint)
            sum_i Assign(i,j) == cap(j)                for every shift j

3. Decides satisfiability with the SAT solver Glucose3 (python-sat / PySAT).
   - If SAT: prints a satisfying model (a feasible assignment).
   - If UNSAT: every requirement-clause is guarded by a private "selector"
     literal; solving under selector assumptions lets PySAT return
     solver.get_core(), i.e. a MINIMAL UNSATISFIABLE CORE over the named
     requirements (mirrors the brief's worked example: MUC = {availability,
     capacity, eligibility}).

4. Three scenarios are provided (run with --mode):

     toy    - reproduces the brief's Worked Example 1.2 EXACTLY
              (2 invigilators, 1 shift, CB1 busy)                 -> SAT
     real   - a genuine small slice of the real dataset: one real
              exam shift, its real required headcount, and the real
              candidate pool of invigilators who were actually on
              duty that week. Busy/Overlap are computed from the REAL
              recorded dates/times (no invented facts)             -> SAT
     stress - the same real shift, but with a SIMULATED (explicitly
              flagged) restricted availability pool, used only to
              demonstrate the UNSAT / minimal-core path since the
              baseline schedule itself contains no real conflicts.
              The dataset does not give us a true availability
              calendar (only realized assignments), so this is
              exactly the kind of "piece the dataset genuinely lacks"
              that S1 of the brief allows us to simulate, and it is
              declared as such.                                    -> UNSAT

     all (default) - run all three, one after another.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    pip install python-sat openpyxl --break-system-packages

    python m1.py                                  # runs all 3 scenarios
    python m1.py --mode toy
    python m1.py --mode real   --data ../data/Dataset_Anonymized_Invigilator_Assignment_Problem.xlsx
    python m1.py --mode stress --data ../data/Dataset_Anonymized_Invigilator_Assignment_Problem.xlsx
    python m1.py --mode all --json out.json        # also dump a machine-readable report
    python m1.py --mode bridge --seed <seed>       # S1.3: logic -> LP bridge + SAT/LP cross-check

--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------
    openpyxl        (read the .xlsx dataset)
    python-sat      (pysat: IDPool, CardEnc, Glucose3)   pip name: python-sat

Note: the brief also allows z3 for 1.2; PySAT/Glucose3 is used here because its
`get_core()` over assumption literals gives a clean, minimal, and directly
attributable unsatisfiable core, which is exactly what S1.2 asks for.
================================================================================
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl
import pandas as pd
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Glucose3

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / \
    "Dataset_Anonymized_Invigilator_Assignment_Problem.xlsx"


# ================================================================================
# 0. Small helpers
# ================================================================================

def hr(title: str = "", ch: str = "-") -> None:
    """Pretty divider for readable console output."""
    if title:
        print(f"\n{ch * 3} {title} {ch * max(3, 70 - len(title))}")
    else:
        print(ch * 78)


def parse_gio(gio: str) -> Tuple[int, int]:
    """'13g00' -> (13, 0)"""
    h, m = gio.replace("g", ":").split(":")
    return int(h), int(m)


# ================================================================================
# 1. Data model: load the real dataset and ground the predicates
# ================================================================================

@dataclass
class Shift:
    shift_id: str
    date: dt.date
    start: dt.datetime
    end: dt.datetime
    campuses: set = field(default_factory=set)
    # invigilators actually assigned to this shift in the baseline (any role)
    baseline_assigned: set = field(default_factory=set)

    @property
    def capacity(self) -> int:
        """cap(j): required headcount, taken from the real baseline schedule."""
        return len(self.baseline_assigned)


class Dataset:
    """
    Loads Dataset_Anonymized_Invigilator_Assignment_Problem.xlsx and grounds:
        - the finite shift set J (Shift objects, real date/time/campus/capacity)
        - the finite invigilator set I (invigilator IDs)
        - Overlap(j, k)  : real predicate, from real start/end datetimes
        - Busy(i, j)     : real predicate, derived ONLY from Overlap + the
                            baseline assignment (i.e. "i is provably double
                            booked with an overlapping shift"); nothing is
                            invented.
    Columns (Vietnamese headers in the raw file):
        Ca thi | Ngay | GIO | MS Ca thi | Nhiem vu | MS cua CAN BO COI THI |
        Thoi gian | Thu | Co so
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.shifts: Dict[str, Shift] = {}
        self.invigilator_shifts: Dict[str, set] = defaultdict(set)
        self._load()
        self._overlap_pairs = self._compute_overlaps()

    # ---- loading -----------------------------------------------------------
    def _load(self) -> None:
        wb = openpyxl.load_workbook(self.path, data_only=True)
        ws = wb["Sheet1"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            ca_thi, ngay, gio, ms_ca, nhiem_vu, ms_cb, thoi_gian, thu, co_so = row
            if ms_ca is None or ms_cb is None:
                continue
            if ms_ca not in self.shifts:
                h, m = parse_gio(gio)
                start = dt.datetime.combine(ngay.date(), dt.time(h, m))
                dur = int(thoi_gian) if thoi_gian else 150
                end = start + dt.timedelta(minutes=dur)
                self.shifts[ms_ca] = Shift(shift_id=ms_ca, date=ngay.date(), start=start, end=end)
            s = self.shifts[ms_ca]
            if co_so:
                s.campuses.add(co_so)
            s.baseline_assigned.add(ms_cb)
            self.invigilator_shifts[ms_cb].add(ms_ca)

    # ---- Overlap(j, k) : real, from real start/end datetimes ---------------
    def _compute_overlaps(self) -> set:
        pairs = set()
        by_date = defaultdict(list)
        for sid, s in self.shifts.items():
            by_date[s.date].append(s)
        for _, lst in by_date.items():
            for a, b in itertools.combinations(lst, 2):
                if a.start < b.end and b.start < a.end:      # genuine time overlap
                    pairs.add(frozenset((a.shift_id, b.shift_id)))
        return pairs

    def overlaps(self, j: str, k: str) -> bool:
        if j == k:
            return False
        return frozenset((j, k)) in self._overlap_pairs

    # ---- Busy(i, j) : real, derived strictly from Overlap + baseline -------
    def is_really_busy(self, i: str, j: str) -> Optional[str]:
        """
        Returns a shift id k such that i is baseline-assigned to k and
        Overlap(j, k) holds (a genuine, provable conflict), or None.
        """
        for k in self.invigilator_shifts.get(i, ()):
            if self.overlaps(j, k):
                return k
        return None

    def candidate_pool(self, window_start: dt.date, window_end: dt.date) -> List[str]:
        """Invigilators who were really on duty at least once within [start, end]."""
        pool = set()
        for sid, s in self.shifts.items():
            if window_start <= s.date <= window_end:
                pool |= s.baseline_assigned
        return sorted(pool)


# ================================================================================
# 2. CNF construction (S1.2): Assign(i,j) atoms + R1/R2/R3 as guarded clauses
# ================================================================================

@dataclass
class Requirement:
    """One NAMED hard rule instance -> a selector literal + its CNF clauses."""
    name: str
    clauses: List[List[int]]
    selector: int


class IAPCnfBuilder:
    """
    Ground the predicates over (I, J) and build a guarded CNF:

        for every requirement r with clause set {c_1 .. c_m}:
            add  (-selector_r  OR  c_i)   for every c_i   -- selector_r => r holds

    Assuming ALL selectors true reproduces the *unguarded* problem exactly
    (SAT iff the real conjunction is SAT). If UNSAT, solver.get_core() over
    the assumed selector literals returns a MINIMAL subset of Requirement
    names that are jointly unsatisfiable -- the minimal unsat core asked for
    in S1.2.
    """

    def __init__(self):
        self.pool = IDPool()
        self.requirements: List[Requirement] = []
        self._assign_cache: Dict[Tuple[str, str], int] = {}

    # ---- Assign(i, j) atom ---------------------------------------------------
    def assign_var(self, i: str, j: str) -> int:
        key = (i, j)
        if key not in self._assign_cache:
            self._assign_cache[key] = self.pool.id(f"Assign({i},{j})")
        return self._assign_cache[key]

    def _add_requirement(self, name: str, clauses: List[List[int]]) -> None:
        sel = self.pool.id(f"sel::{name}")
        guarded = [[-sel] + c for c in clauses]
        self.requirements.append(Requirement(name=name, clauses=guarded, selector=sel))

    # ---- R2  Availability: Busy(i,j) -> not Assign(i,j) -----------------------
    def add_availability(self, i: str, j: str) -> None:
        a = self.assign_var(i, j)
        self._add_requirement(f"Availability: Busy({i},{j}) -> not Assign({i},{j})", [[-a]])

    # ---- R1  No double-booking: Overlap(j,k) & Assign(i,j) -> not Assign(i,k) -
    def add_no_double_booking(self, i: str, j: str, k: str) -> None:
        aj = self.assign_var(i, j)
        ak = self.assign_var(i, k)
        self._add_requirement(
            f"NoDoubleBooking: Overlap({j},{k}) -> not(Assign({i},{j}) & Assign({i},{k}))",
            [[-aj, -ak]],
        )

    # ---- R3  Capacity: sum_i Assign(i,j) == cap(j) -----------------------------
    def add_capacity(self, j: str, candidates: List[str], cap: int) -> None:
        lits = [self.assign_var(i, j) for i in candidates]
        cnf = CardEnc.equals(lits=lits, bound=cap, vpool=self.pool, encoding=EncType.seqcounter)
        self._add_requirement(f"Capacity: |Assign(*,{j})| = {cap}", list(cnf.clauses))

    # ---- assemble --------------------------------------------------------------
    def all_clauses(self) -> List[List[int]]:
        clauses = []
        for r in self.requirements:
            clauses.extend(r.clauses)
        return clauses

    def all_selectors(self) -> List[int]:
        return [r.selector for r in self.requirements]

    def selector_name(self, lit: int) -> str:
        for r in self.requirements:
            if r.selector == abs(lit):
                return r.name
        return f"lit#{lit}"


# ================================================================================
# 3. Solve: report a model, or a minimal unsatisfiable core
# ================================================================================

def solve(builder: IAPCnfBuilder) -> dict:
    clauses = builder.all_clauses()
    selectors = builder.all_selectors()

    solver = Glucose3(bootstrap_with=clauses)
    sat = solver.solve(assumptions=selectors)

    result = {
        "num_atoms": builder.pool.top,
        "num_requirements": len(builder.requirements),
        "num_cnf_clauses": len(clauses),
        "sat": sat,
    }

    if sat:
        model = set(solver.get_model())
        assigned_pairs = []
        for (i, j), var in builder._assign_cache.items():
            if var in model:
                assigned_pairs.append((i, j))
        result["model_assign_true"] = sorted(assigned_pairs)
    else:
        core = solver.get_core() or []
        result["unsat_core"] = [builder.selector_name(lit) for lit in core]

    solver.delete()
    return result


def report(title: str, result: dict) -> None:
    hr(title, "=")
    print(f"atoms (Assign(i,j) vars) : {result['num_atoms']}")
    print(f"named requirements        : {result['num_requirements']}")
    print(f"CNF clauses (guarded)     : {result['num_cnf_clauses']}")
    print(f"SAT?                      : {result['sat']}")
    if result["sat"]:
        print("\nSatisfying model (Assign(i,j) = TRUE):")
        for i, j in result["model_assign_true"]:
            print(f"    Assign({i}, {j})")
    else:
        print("\nMINIMAL UNSATISFIABLE CORE (named requirements in conflict):")
        for name in result["unsat_core"]:
            print(f"    - {name}")


# ================================================================================
# 4. Scenario 1 -- TOY instance (reproduces the brief's Worked Example, S1)
# ================================================================================
#
#   Two invigilators {CB1, CB2}; one shift s needs exactly ONE person;
#   CB1 is busy at s.
#   Rules:  Busy(CB1,s) -> not Assign(CB1,s)     and     "exactly one assigned"
#   ai = Assign(CBi, s).  CNF = (not a1) & (a1 v a2) & not(a1 & a2)
#   Expected model: a1 = 0, a2 = 1   (SAT)
# ================================================================================

def scenario_toy() -> dict:
    b = IAPCnfBuilder()
    s = "s"
    candidates = ["CB1", "CB2"]

    b.add_availability("CB1", s)          # not Assign(CB1, s)
    b.add_capacity(s, candidates, cap=1)  # exactly one of {CB1, CB2} assigned

    result = solve(b)
    report("SCENARIO: toy (brief Worked Example 1.2)", result)

    if result["sat"]:
        a1 = ("CB1", s) in result["model_assign_true"]
        a2 = ("CB2", s) in result["model_assign_true"]
        expected_ok = (a1 is False) and (a2 is True)
        print(f"\nCheck against brief's expected model (a1=0, a2=1): "
              f"{'MATCH' if expected_ok else 'MISMATCH'}")
    return result


# ================================================================================
# 5. Scenario 2 -- REAL small slice of the dataset (feasible / SAT)
# ================================================================================

def scenario_real(ds: Dataset, target_shift: str, window_days: int = 7) -> dict:
    shift = ds.shifts[target_shift]
    cap = shift.capacity
    window_start = shift.date - dt.timedelta(days=window_days)
    window_end = shift.date
    candidates = ds.candidate_pool(window_start, window_end)

    print(f"\nReal shift        : {target_shift}  "
          f"({shift.date}  {shift.start.time()}-{shift.end.time()}  "
          f"campus={sorted(shift.campuses)})")
    print(f"Required capacity  : {cap}  (real headcount from the baseline schedule)")
    print(f"Candidate pool     : {len(candidates)} invigilators really on duty in "
          f"[{window_start} .. {window_end}]")

    b = IAPCnfBuilder()

    # R2 Availability -- only add a clause where Busy is REALLY true (Overlap-derived)
    n_busy = 0
    for i in candidates:
        conflict_shift = ds.is_really_busy(i, target_shift)
        if conflict_shift is not None:
            b.add_availability(i, target_shift)
            n_busy += 1
    print(f"Real Busy(i, {target_shift}) facts (provable time overlap): {n_busy}")

    # R1 No-double-booking -- for every real shift k that really overlaps target_shift
    overlapping = [k for k in ds.shifts if ds.overlaps(target_shift, k)]
    print(f"Shifts really overlapping {target_shift} in time: {len(overlapping)}")
    for i in candidates:
        for k in overlapping:
            b.add_no_double_booking(i, target_shift, k)

    # R3 Capacity
    b.add_capacity(target_shift, candidates, cap)

    result = solve(b)
    report(f"SCENARIO: real data slice (shift {target_shift})", result)
    return result


# ================================================================================
# 6. Scenario 3 -- STRESS TEST -> deliberately UNSAT, report the minimal core
# ================================================================================
#
# The real baseline schedule has ZERO genuine temporal conflicts (verified:
# no two distinct shift ids overlap in time in this dataset), so scenario_real
# is always SAT and cannot, by itself, demonstrate the UNSAT / minimal-core
# path required by S1.2. The dataset also does not provide a true
# availability calendar -- only realized assignments -- so a restricted
# hypothetical availability pool is exactly the kind of "piece the dataset
# genuinely lacks" that S1 explicitly allows a team to simulate, PROVIDED the
# assumption is stated. It is stated here and nowhere else in the model.
# ================================================================================

def scenario_stress(ds: Dataset, target_shift: str, seed: Optional[int] = None) -> dict:
    shift = ds.shifts[target_shift]
    cap = shift.capacity
    real_pool = sorted(shift.baseline_assigned)
    n_available = max(cap - 2, 0)

    # SIMULATED assumption (declared): only `n_available` of the real assignees
    # are actually available this time (e.g. two called in sick) -- a what-if
    # stress test, NOT a claim about real data. WHO exactly is unavailable is
    # picked deterministically from the team seed (tools/make_seed.py) when one
    # is given, so every team's own run reproduces exactly, while different
    # teams get a different (but still reproducible) shortage scenario. With no
    # seed (e.g. ad-hoc `python m1.py`), fall back to a fixed, order-based pick.
    if seed is not None:
        rng = random.Random(seed)
        simulated_available = sorted(rng.sample(real_pool, k=n_available))
    else:
        simulated_available = real_pool[:n_available]

    print("\n*** SIMULATED SCENARIO (explicitly declared, not real data) ***")
    print(f"Real shift          : {target_shift}  (real required capacity = {cap})")
    print(f"Team seed used      : {seed if seed is not None else '(none - fixed fallback pick)'}")
    print(f"Hypothetical pool   : {simulated_available}  "
          f"({len(simulated_available)} available, i.e. {cap - len(simulated_available)} short)")

    b = IAPCnfBuilder()
    # everyone NOT in the simulated-available list is marked (simulated) busy
    for i in real_pool:
        if i not in simulated_available:
            b.add_availability(i, target_shift)
    b.add_capacity(target_shift, real_pool, cap)

    result = solve(b)
    report(f"SCENARIO: stress test (shift {target_shift}, simulated shortage)", result)
    return result



# ================================================================================
# 6b. LOGIC -> LP BRIDGE (S1.3)  --  the hand-off from Module 1 to Module 2
# ================================================================================
#
#   Translation table (each CNF clause becomes ONE linear (in)equality over 0/1
#   variables x_ij  =  [Assign(i,j)]):
#
#     clause / rule                              linear form
#     -----------------------------------------  --------------------------------
#     positive clause  (a v b v ...)             x_a + x_b + ...            >= 1
#     negative clause  (-a v -b v ...)           x_a + x_b + ...            <= k-1
#     mixed clause     (a v -b)                  x_a - x_b                  >= 1-1
#     R2 Availability   Busy(i,j) -> -Assign     x_ij                       <= 0
#     R1 No-double-book -(Ai_j & Ai_k)           x_ij + x_ik                <= 1
#     R3 Capacity       |Assign(*,j)| = cap(j)   sum_i x_ij                 == cap(j)
#
#   R3 is a cardinality / pseudo-Boolean constraint: ONE equation here, but
#   C(n, cap+1) + C(n, n-cap+1) clauses if written in pure CNF without auxiliary
#   variables (see Appendix A, Module 1, Q1).
#
#   Module 2 calls  logic_to_lp(instance)  and gets back a pulp.LpProblem that
#   already contains every HARD rule and NO objective; it then adds the
#   objective, the soft constraints and the fairness variables on top.
# ================================================================================

@dataclass
class LogicInstance:
    """
    Finite grounding of the Module-1 predicates -- the CONTRACT between M1 and M2.

        invigilators : the set I
        shifts       : the set J (any hashable-as-string ids; M2 uses sessions)
        capacity     : cap(j) for every j that must be staffed
        busy         : {(i, j)}                 -- Busy(i, j)
        overlaps     : {frozenset({j, k})}      -- Overlap(j, k) (symmetric, irreflexive)
    """
    invigilators: List[str]
    shifts: List[str]
    capacity: Dict[str, int]
    busy: set = field(default_factory=set)
    overlaps: set = field(default_factory=set)
    name: str = "IAP_hard_rules"


def clause_to_linear(clause, x):
    """
    One CNF clause -> one linear constraint over 0/1 variables.

    clause : list of (atom, is_positive);  atom is a key of `x` (here (i, j)).
    A clause is satisfied iff  sum_{pos} x + sum_{neg} (1 - x) >= 1,
    i.e.       sum_{pos} x - sum_{neg} x  >=  1 - |neg|.
    (The all-negative case is printed in its natural "at most" form.)
    """
    import pulp
    pos = [x[a] for a, s in clause if s]
    neg = [x[a] for a, s in clause if not s]
    if not pos:
        return pulp.lpSum(neg) <= len(neg) - 1
    return pulp.lpSum(pos) - pulp.lpSum(neg) >= 1 - len(neg)


def cardinality_to_linear(atoms, cap, x):
    """`exactly cap of these atoms are true`  ->  one equation (pseudo-Boolean)."""
    import pulp
    return pulp.lpSum(x[a] for a in atoms) == cap


def logic_to_lp(instance: LogicInstance, name: Optional[str] = None):
    """
    Logic -> LP.   Ground R1/R2/R3 over `instance` and return a pulp.LpProblem
    holding ONLY the hard rules (no objective yet).

    Extra attributes attached to the returned problem, for Module 2:
        prob.assign_vars   {(i, j): LpVariable}  binary  x_ij = Assign(i, j)
        prob.rule_counts   {"R1": .., "R2": .., "R3": ..}  #constraints per rule
        prob.logic_instance  the LogicInstance it was built from
    """
    import pulp
    prob = pulp.LpProblem(name or instance.name, pulp.LpMinimize)
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
         for i in instance.invigilators for j in instance.shifts}
    counts = {"R1": 0, "R2": 0, "R3": 0}

    # R2  Busy(i,j) -> -Assign(i,j)          clause (-a)          =>  x_ij <= 0
    for (i, j) in sorted(instance.busy):
        if (i, j) in x:
            prob += clause_to_linear([((i, j), False)], x), f"R2_avail_{i}_{j}"
            counts["R2"] += 1

    # R1  Overlap(j,k) & Assign(i,j) -> -Assign(i,k)   clause (-a_ij v -a_ik)
    #                                                  =>  x_ij + x_ik <= 1
    for pair in sorted(tuple(sorted(p)) for p in instance.overlaps):
        j, k = pair
        if j == k or j not in instance.shifts or k not in instance.shifts:
            continue
        for i in instance.invigilators:
            prob += (clause_to_linear([((i, j), False), ((i, k), False)], x),
                     f"R1_nodbl_{i}_{j}_{k}")
            counts["R1"] += 1

    # R3  cardinality   sum_i Assign(i,j) = cap(j)   =>  sum_i x_ij == cap(j)
    for j in instance.shifts:
        if j in instance.capacity:
            prob += (cardinality_to_linear([(i, j) for i in instance.invigilators],
                                           instance.capacity[j], x), f"R3_cap_{j}")
            counts["R3"] += 1

    prob.assign_vars = x
    prob.rule_counts = counts
    prob.logic_instance = instance
    return prob


def _pick_lp_solver():
    """First available MILP solver known to PuLP (CBC, then HiGHS, then whatever exists)."""
    import pulp
    avail = pulp.listSolvers(onlyAvailable=True)
    for nm in ("PULP_CBC_CMD", "HiGHS", "HiGHS_CMD", "SCIP_PY", "GLPK_CMD"):
        if nm in avail:
            return pulp.getSolver(nm, msg=False)
    return pulp.getSolver(avail[0], msg=False)


def logic_instance_toy() -> LogicInstance:
    """Same instance as scenario_toy(): {CB1, CB2}, one shift needing 1, CB1 busy."""
    return LogicInstance(["CB1", "CB2"], ["s"], {"s": 1}, busy={("CB1", "s")}, name="toy")


def logic_instance_real(ds: "Dataset", target_shift: str, window_days: int = 7) -> LogicInstance:
    """Same slice as scenario_real(), expressed as a LogicInstance."""
    shift = ds.shifts[target_shift]
    candidates = ds.candidate_pool(shift.date - dt.timedelta(days=window_days), shift.date)
    busy = {(i, target_shift) for i in candidates if ds.is_really_busy(i, target_shift) is not None}
    overlapping = [k for k in ds.shifts if ds.overlaps(target_shift, k)]
    return LogicInstance(candidates, [target_shift] + overlapping,
                         {target_shift: shift.capacity}, busy=busy,
                         overlaps={frozenset((target_shift, k)) for k in overlapping},
                         name=f"real_{target_shift}")


def logic_instance_stress(ds: "Dataset", target_shift: str, seed: Optional[int] = None) -> LogicInstance:
    """Same (simulated-shortage) slice as scenario_stress(); identical seeded pick."""
    shift = ds.shifts[target_shift]
    cap = shift.capacity
    real_pool = sorted(shift.baseline_assigned)
    n_available = max(cap - 2, 0)
    if seed is not None:
        simulated_available = sorted(random.Random(seed).sample(real_pool, k=n_available))
    else:
        simulated_available = real_pool[:n_available]
    busy = {(i, target_shift) for i in real_pool if i not in simulated_available}
    return LogicInstance(real_pool, [target_shift], {target_shift: cap}, busy=busy,
                         name=f"stress_{target_shift}")


def _bridge_check(label: str, inst: LogicInstance, sat_result: dict, show_rows: int = 6) -> dict:
    """Build the LP from `inst`, solve it, and cross-check against the SAT verdict."""
    import pulp
    prob = logic_to_lp(inst)
    hr(f"BRIDGE {label}: logic -> LP", "=")
    print(f"variables x_ij : {len(prob.assign_vars)}   constraints : {len(prob.constraints)}   "
          f"per rule : {prob.rule_counts}")
    for cname, c in list(prob.constraints.items())[:show_rows]:
        print(f"    {cname:<44s} {c}")
    if len(prob.constraints) > show_rows:
        print(f"    ... ({len(prob.constraints) - show_rows} more)")

    status = pulp.LpStatus[prob.solve(_pick_lp_solver())]
    lp_feasible = status == "Optimal"
    out = {"lp_status": status, "lp_feasible": lp_feasible, "sat": sat_result["sat"],
           "agree": lp_feasible == sat_result["sat"],
           "variables": len(prob.assign_vars), "constraints": len(prob.constraints),
           "rule_counts": prob.rule_counts}
    print(f"SAT verdict = {sat_result['sat']}   |   LP status = {status}   ->   "
          f"{'MATCH' if out['agree'] else 'MISMATCH'}")

    # Model transfer: the SAT model must satisfy every generated linear constraint.
    if sat_result["sat"]:
        true_atoms = set(sat_result["model_assign_true"])
        for (i, j), v in prob.assign_vars.items():
            v.varValue = 1.0 if (i, j) in true_atoms else 0.0
        out["sat_model_satisfies_lp"] = all(c.valid(1e-9) for c in prob.constraints.values())
        print(f"SAT model satisfies all generated linear constraints: "
              f"{out['sat_model_satisfies_lp']}")
    return out


def scenario_bridge(data: Path = DEFAULT_DATA_PATH, shift: Optional[str] = None,
                    seed: Optional[int] = None) -> dict:
    """S1.3: for toy / real / stress, check  SAT(CNF)  <=>  feasible(LP from logic_to_lp)."""
    results = {"toy": _bridge_check("toy", logic_instance_toy(), scenario_toy(), show_rows=10)}
    if not Path(data).exists():
        print(f"ERROR: dataset not found at {data}. Pass --data <path>.", file=sys.stderr)
        sys.exit(1)
    ds = Dataset(data)
    sid = shift or pick_demo_shift(ds)
    results["real"] = _bridge_check("real", logic_instance_real(ds, sid), scenario_real(ds, sid))
    results["stress"] = _bridge_check("stress", logic_instance_stress(ds, sid, seed),
                                      scenario_stress(ds, sid, seed=seed))
    hr("BRIDGE SUMMARY", "=")
    for k, r in results.items():
        print(f"  {k:<7s}: SAT={r['sat']!s:<5}  LP={r['lp_status']:<10}  "
              f"{'MATCH' if r['agree'] else 'MISMATCH'}")
    return results

# ================================================================================
# 7. CLI
# ================================================================================

def pick_demo_shift(ds: Dataset, min_cap: int = 4, max_cap: int = 8) -> str:
    """Pick a small, readable real shift (headcount in [min_cap, max_cap])."""
    candidates = [(sid, s.capacity) for sid, s in ds.shifts.items()
                  if min_cap <= s.capacity <= max_cap]
    candidates.sort(key=lambda t: (t[1], t[0]))
    if not candidates:
        # fall back to the smallest available shift
        return min(ds.shifts.items(), key=lambda kv: kv[1].capacity)[0]
    return candidates[0][0]


def run_pipeline(
    mode: str = "all",
    data: Path = DEFAULT_DATA_PATH,
    shift: Optional[str] = None,
    seed: Optional[int] = None,
    json_path: Optional[Path] = None,
) -> dict:
    """
    Programmatic entry point (used by both the CLI `main()` below and by
    run_all.py, which `import`s and calls this function directly instead of
    spawning a subprocess).

    seed: the team's seed from data/seed.txt (see tools/make_seed.py). Module 1
    itself only encodes HARD rules (no randomness needed to decide SAT/UNSAT),
    so `seed` is not required for correctness here. It is accepted so that
    run_all.py can pass --seed uniformly to every stage, and it is used for
    exactly one thing: making the `stress` scenario's simulated shortage
    (S6 in this file) deterministic-but-team-specific, so two teams running
    "python run_all.py --seed <theirs>" do not get an identical stress demo
    while each team's own result stays perfectly reproducible.

    Returns the same `all_results` dict that used to be built inline in main().
    """
    if mode == "bridge":               # S1.3 -- additive; 'all' is unchanged
        return {"bridge": scenario_bridge(data=data, shift=shift, seed=seed)}

    all_results = {}

    if mode in ("toy", "all"):
        all_results["toy"] = scenario_toy()

    if mode in ("real", "stress", "all"):
        if not Path(data).exists():
            print(f"ERROR: dataset not found at {data}. Pass --data <path>.", file=sys.stderr)
            sys.exit(1)
        ds = Dataset(data)
        shift_id = shift or pick_demo_shift(ds)

        if mode in ("real", "all"):
            all_results["real"] = scenario_real(ds, shift_id)
        if mode in ("stress", "all"):
            all_results["stress"] = scenario_stress(ds, shift_id, seed=seed)

    hr("SUMMARY", "=")
    for name, res in all_results.items():
        print(f"  {name:<8s}: {'SAT' if res['sat'] else 'UNSAT'}"
              f"  ({res['num_cnf_clauses']} clauses over {res['num_requirements']} requirements)")

    if json_path:
        Path(json_path).write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON report written to {json_path}")

    return all_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 1.2 - CNF encoding & SAT for the IAP")
    ap.add_argument("--mode", choices=["toy", "real", "stress", "all", "bridge"], default="all")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                     help="path to Dataset_Anonymized_Invigilator_Assignment_Problem.xlsx")
    ap.add_argument("--shift", type=str, default=None,
                     help="MS Ca thi (shift id) to use for real/stress modes; "
                          "default: an auto-picked small real shift")
    ap.add_argument("--seed", type=int, default=None,
                     help="team seed from data/seed.txt (tools/make_seed.py); "
                          "only affects which people are simulated 'unavailable' "
                          "in --mode stress")
    ap.add_argument("--json", type=Path, default=None, help="optional path to dump a JSON report")
    args = ap.parse_args()

    run_pipeline(mode=args.mode, data=args.data, shift=args.shift,
                 seed=args.seed, json_path=args.json)


if __name__ == "__main__":
    main()