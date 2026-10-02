from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pulp

from m2_ilp.data import IAPInstance, import_m1

FAIRNESS_MODES = ("l1", "minmax", "spread")
EPS = 1e-3            # tie-break weight for minmax / spread


@dataclass
class Model:
    prob: "pulp.LpProblem"
    x: Dict[Tuple[str, str], "pulp.LpVariable"]
    inst: IAPInstance
    fairness: str
    weights: Dict[str, float]
    terms: Dict[str, "pulp.LpAffineExpression"] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
def build_model(inst: IAPInstance, fairness: str = "l1",
                weights: Optional[Dict[str, float]] = None, tighten: bool = True,
                name: str = "IAP_M2", relax_binary: bool = False) -> Model:
    """
    HARD rules come from Module 1 (`logic_to_lp`), then this function adds the
    fairness variables, the soft constraints and the objective.

    relax_binary=True builds the LP relaxation (x in [0,1]); it is used ONLY by the
    counter-example study (2.10) and for the root-gap report (2.9).
    """
    if fairness not in FAIRNESS_MODES:
        raise ValueError(f"fairness must be one of {FAIRNESS_MODES}")
    w = dict(weights or inst.weights)
    _, logic_to_lp = import_m1()

    prob = logic_to_lp(inst.logic_instance(), name=name)          # <- Module 1 -> Module 2
    x = prob.assign_vars
    if relax_binary:
        for v in x.values():
            v.cat = pulp.LpContinuous
            v.lowBound, v.upBound = 0, 1

    I, J = inst.invigilators, inst.J
    ses = inst.sessions
    ok = lambda i, j: (i, j) not in inst.busy         # (i,j) usable

    load = {i: pulp.lpSum(x[i, j] for j in J) for i in I}
    wbar = inst.mean_load

    # ---- fairness -------------------------------------------------------------------
    d = {i: pulp.LpVariable(f"d_{i}", lowBound=0) for i in I}
    for i in I:
        prob += d[i] >= load[i] - wbar, f"F_dev_pos_{i}"
        prob += d[i] >= wbar - load[i], f"F_dev_neg_{i}"
    n_cut = 0
    if tighten:
        a = math.floor(wbar)
        f = wbar - a
        if f > 1e-9:
            for i in I:
                prob += d[i] >= f + (load[i] - a) * (1 - 2 * f), f"F_chord_{i}"
                n_cut += 1
    L1 = pulp.lpSum(d.values())
    if fairness == "l1":
        F = L1
    else:
        tmax = pulp.LpVariable("t_max", lowBound=0)
        for i in I:
            prob += tmax >= load[i], f"F_max_{i}"
        if fairness == "minmax":
            F = tmax + EPS * L1
        else:
            tmin = pulp.LpVariable("t_min", lowBound=0)
            for i in I:
                prob += tmin <= load[i], f"F_min_{i}"
            F = tmax - tmin + EPS * L1

    # ---- soft: location ---------------------------------------------------------------
    L = pulp.lpSum(x[i, j] for i in I for j in J
                   if ok(i, j) and inst.preferred_campus(i) not in (None, ses[j].campus))

    # ---- soft: fatigue (daily and weekly excess) -----------------------------------------
    e_terms: List = []
    n_e = 0
    for i in I:
        for day in inst.days():
            js = [j for j in inst.sessions_on(day) if ok(i, j)]
            if len(js) > inst.fatigue_day:
                e = pulp.LpVariable(f"eday_{i}_{day:%Y%m%d}", lowBound=0)
                prob += e >= pulp.lpSum(x[i, j] for j in js) - inst.fatigue_day, \
                    f"S_fat_day_{i}_{day:%Y%m%d}"
                e_terms.append(e)
                n_e += 1
        for wk in inst.weeks():
            js = [j for j in J if ses[j].week == wk and ok(i, j)]
            if len(js) > inst.fatigue_week:
                e = pulp.LpVariable(f"ewk_{i}_{wk}", lowBound=0)
                prob += e >= pulp.lpSum(x[i, j] for j in js) - inst.fatigue_week, \
                    f"S_fat_wk_{i}_{wk}"
                e_terms.append(e)
                n_e += 1
    T = pulp.lpSum(e_terms)

    # ---- objective --------------------------------------------------------------------
    Z = w["w_fair"] * F + w["w_loc"] * L + w["w_fat"] * T
    prob.setObjective(Z)

    m = Model(prob=prob, x=x, inst=inst, fairness=fairness, weights=w,
              terms={"fair": F, "loc": L, "fat": T, "L1": L1})
    m.stats = {"binary_x": sum(1 for v in x.values() if v.cat == pulp.LpInteger and v.lowBound == 0 and v.upBound == 1),
               "continuous_aux": len(prob.variables()) - len(x),
               "constraints": len(prob.constraints),
               "hard_R1": prob.rule_counts["R1"], "hard_R2": prob.rule_counts["R2"],
               "hard_R3": prob.rule_counts["R3"], "chord_cuts": n_cut,
               "fatigue_vars": n_e}
    return m


def extract_assignment(m: Model, thr: float = 0.5) -> Set[Tuple[str, str]]:
    return {(i, j) for (i, j), v in m.x.items() if (v.varValue or 0.0) > thr}


# ------------------------------------------------------------------------------
# Independent evaluation (pure Python: no solver) -- used for baseline and optimum
# ------------------------------------------------------------------------------
def loads_of(inst: IAPInstance, assign: Iterable[Tuple[str, str]]) -> Dict[str, int]:
    ld = {i: 0 for i in inst.invigilators}
    for (i, _j) in assign:
        ld[i] += 1
    return ld


def evaluate(inst: IAPInstance, assign: Iterable[Tuple[str, str]], fairness: str = "l1",
             weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """All fairness metrics + soft penalties + the SAME objective value the model minimises."""
    assign = set(assign)
    w = weights or inst.weights
    ld = loads_of(inst, assign)
    vals = list(ld.values())
    n = len(vals)
    mean = sum(vals) / n
    l1 = sum(abs(v - inst.mean_load) for v in vals)
    var = sum((v - mean) ** 2 for v in vals) / n
    srt = sorted(vals)
    gini = sum((2 * (k + 1) - n - 1) * v for k, v in enumerate(srt)) / (n * sum(srt))
    ses = inst.sessions

    loc = sum(1 for (i, j) in assign
              if inst.preferred_campus(i) not in (None, ses[j].campus))
    per_day: Dict[Tuple[str, object], int] = {}
    per_wk: Dict[Tuple[str, int], int] = {}
    for (i, j) in assign:
        per_day[(i, ses[j].date)] = per_day.get((i, ses[j].date), 0) + 1
        per_wk[(i, ses[j].week)] = per_wk.get((i, ses[j].week), 0) + 1
    fat_day = sum(max(0, c - inst.fatigue_day) for c in per_day.values())
    fat_wk = sum(max(0, c - inst.fatigue_week) for c in per_wk.values())
    fat = fat_day + fat_wk

    mx, mn = max(vals), min(vals)
    F = {"l1": l1, "minmax": mx + EPS * l1, "spread": (mx - mn) + EPS * l1}[fairness]
    Z = w["w_fair"] * F + w["w_loc"] * loc + w["w_fat"] * fat
    return {"max_load": mx, "min_load": mn, "spread": mx - mn, "L1_deviation": l1,
            "variance": var, "std": math.sqrt(var), "gini": gini,
            "loc_violations": loc, "fatigue_day_excess": fat_day, "fatigue_week_excess": fat_wk,
            "fatigue_total": fat, "F": F, "objective": Z}


def validate_schedule(inst: IAPInstance, assign: Iterable[Tuple[str, str]]) -> List[str]:
    """Check the three HARD rules directly on the data (does not trust the solver)."""
    assign = set(assign)
    bad: List[str] = []
    cnt: Dict[str, int] = {j: 0 for j in inst.sessions}
    by_i: Dict[str, List[str]] = {}
    for (i, j) in assign:
        if i not in inst.invigilators or j not in inst.sessions:
            bad.append(f"unknown pair {(i, j)}")
            continue
        cnt[j] += 1
        by_i.setdefault(i, []).append(j)
        if (i, j) in inst.busy:
            bad.append(f"R2 availability: {i} busy at {j}")
    for j, c in cnt.items():
        if c != inst.sessions[j].capacity:
            bad.append(f"R3 capacity: {j} has {c}, needs {inst.sessions[j].capacity}")
    for i, js in by_i.items():
        js = sorted(js)
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                if frozenset((js[a], js[b])) in inst.overlaps:
                    bad.append(f"R1 double-booking: {i} on {js[a]} and {js[b]}")
    return bad