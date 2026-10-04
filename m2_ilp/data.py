"""
m2_ilp/data.py  --  Module 2, requirements 2.1 (parameters) and 2.6 (preprocessing)
================================================================================
Reads Dataset_Anonymized_Invigilator_Assignment_Problem.xlsx and produces ONE
object, `IAPInstance`, holding every set and parameter of the ILP.

SETS
    I   invigilators           73 ids  CB001..CB073   (everyone in the file)
    J   SESSIONS               a session = (shift id, campus)  e.g. 20260601_2_CS2
        The file gives, for the same time slot, different headcounts on the two
        campuses (Co so 1 = LTK, Co so 2 = Di An), so each (slot, campus) is its
        own room-group that must be staffed.  Sessions of the same slot on
        different campuses overlap in time  ->  nobody can be in both.

PARAMETERS
    cap_j          required headcount = #rows of that (shift, campus) in the file
                   (all roles CBCT / Thu ky / Truong HD are counted together,
                   exactly as Module 1 does)
    start_j,end_j  real start time + duration (150 min)
    Overlap(j,k)   real predicate from the intervals
    Busy(i,j)      DATA-DERIVED availability  (see `availability` below)
    baseline       the schedule in the file  (the thing to beat)

WHAT IS REAL AND WHAT IS SIMULATED   (declared assumptions, see ASSUMPTIONS)
    real       : sets, capacities, times, campuses, baseline, presence per day
    imputed    : 28 rows have an empty 'Co so' -> campus taken from the role
                 prefix (LTK_* = Co so 1, DiAn_* = Co so 2); the prefix is 100 %
                 consistent with the campus column on the other 741 rows
    simulated  : location preference of each invigilator (3 categories),
                 soft-constraint weights (3 values in [0.5, 2.0])  -- both from
                 the team seed, as required by the brief.

AVAILABILITY  (the dataset has no calendar, only realised assignments)
    mode "day"  (default)  Busy(i,j) <=> i has NO baseline row on the date of j.
                           Reading: someone who appears on a date is at the
                           Faculty that day; someone who never appears that day
                           is treated as busy / on leave.  The baseline is then
                           feasible by construction, so the model is never
                           infeasible because of data.
    mode "week"            present on some day of the same ISO week  (relaxation)
    mode "all"             nobody busy                                (relaxation)
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "Dataset_Anonymized_Invigilator_Assignment_Problem.xlsx"
SEED_FILE = ROOT / "data" / "seed.txt"

CATEGORIES = ("near_c1", "near_c2", "balanced")
PREFERRED_CAMPUS = {"near_c1": "CS1", "near_c2": "CS2", "balanced": None}
ROLE_PREFIX_TO_CAMPUS = {"LTK": "CS1", "DiAn": "CS2"}

FATIGUE_DAY_LIMIT = 2        # soft: > 2 sessions in one day is penalised
FATIGUE_WEEK_LIMIT = 8       # soft: > 8 sessions in one ISO week is penalised


# ------------------------------------------------------------------------------
# M1 hand-off (M1 lives in m1_logic/m1.py, the file run_all.py imports)
# ------------------------------------------------------------------------------
def import_m1():
    """Return (LogicInstance, logic_to_lp) from Module 1, wherever the file lives."""
    for p in (str(ROOT), str(ROOT / "m1_logic")):
        if p not in sys.path:
            sys.path.insert(0, p)
    for mod in ("m1_logic.m1", "m1_logic.cnf", "m1", "cnf"):
        try:
            m = __import__(mod, fromlist=["LogicInstance", "logic_to_lp"])
            return m.LogicInstance, m.logic_to_lp
        except (ImportError, AttributeError):
            continue
    raise ImportError("Module 1 not found: expected m1_logic/m1.py (with logic_to_lp)")


# ------------------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Session:
    sid: str                 # e.g. "20260601_2_CS2"  (safe inside LP variable names)
    shift_id: str            # "20260601_2"
    date: dt.date
    slot: int                # 1..5
    period: str              # M (morning) / A (afternoon) / N (night)   -> Module 3 alphabet
    campus: str              # CS1 / CS2
    start: dt.datetime
    end: dt.datetime
    week: int                # ISO week number
    capacity: int


@dataclass
class IAPInstance:
    invigilators: List[str]
    sessions: Dict[str, Session]
    overlaps: Set[frozenset]
    present_days: Dict[str, Set[dt.date]]
    baseline: Set[Tuple[str, str]]
    pref: Dict[str, str]                       # invigilator -> category
    weights: Dict[str, float]                  # w_fair (=w1), w_fat (=w2), w_loc (=w3)
    seed: Optional[int]
    availability_mode: str = "day"
    busy: Set[Tuple[str, str]] = field(default_factory=set)
    fatigue_day: int = FATIGUE_DAY_LIMIT
    fatigue_week: int = FATIGUE_WEEK_LIMIT
    notes: Dict[str, object] = field(default_factory=dict)

    # ---- derived helpers ------------------------------------------------------
    @property
    def J(self) -> List[str]:
        return list(self.sessions)

    @property
    def total_demand(self) -> int:
        return sum(s.capacity for s in self.sessions.values())

    @property
    def mean_load(self) -> float:
        return self.total_demand / len(self.invigilators)

    def preferred_campus(self, i: str) -> Optional[str]:
        return PREFERRED_CAMPUS[self.pref[i]]

    def sessions_on(self, day: dt.date) -> List[str]:
        return [j for j, s in self.sessions.items() if s.date == day]

    def days(self) -> List[dt.date]:
        return sorted({s.date for s in self.sessions.values()})

    def weeks(self) -> List[int]:
        return sorted({s.week for s in self.sessions.values()})

    # ---- availability (re-computable: used by the relaxation ladder) -----------
    def with_availability(self, mode: str) -> "IAPInstance":
        new = IAPInstance(**{**self.__dict__})
        new.availability_mode = mode
        new.busy = compute_busy(self, mode)
        return new

    def restrict_days(self, keep: Set[dt.date]) -> "IAPInstance":
        """Sub-instance made of the sessions on `keep` days (scalability study)."""
        sess = {j: s for j, s in self.sessions.items() if s.date in keep}
        ov = {p for p in self.overlaps if all(j in sess for j in p)}
        new = IAPInstance(invigilators=self.invigilators, sessions=sess, overlaps=ov,
                          present_days=self.present_days,
                          baseline={(i, j) for (i, j) in self.baseline if j in sess},
                          pref=self.pref, weights=self.weights, seed=self.seed,
                          availability_mode=self.availability_mode,
                          fatigue_day=self.fatigue_day, fatigue_week=self.fatigue_week,
                          notes=self.notes)
        new.busy = compute_busy(new, new.availability_mode)
        return new

    # ---- M1 hand-off -------------------------------------------------------------
    def logic_instance(self):
        LogicInstance, _ = import_m1()
        return LogicInstance(invigilators=list(self.invigilators), shifts=self.J,
                             capacity={j: s.capacity for j, s in self.sessions.items()},
                             busy=set(self.busy), overlaps=set(self.overlaps),
                             name="IAP_M2")


def compute_busy(inst: IAPInstance, mode: str) -> Set[Tuple[str, str]]:
    if mode == "all":
        return set()
    busy = set()
    weeks_present: Dict[str, Set[int]] = {
        i: {d.isocalendar()[1] for d in days} for i, days in inst.present_days.items()}
    for i in inst.invigilators:
        for j, s in inst.sessions.items():
            if mode == "day":
                ok = s.date in inst.present_days.get(i, set())
            elif mode == "week":
                ok = s.week in weeks_present.get(i, set())
            else:
                raise ValueError(f"unknown availability mode {mode!r}")
            if not ok:
                busy.add((i, j))
    return busy


# ------------------------------------------------------------------------------
# Seed, weights, simulated preferences
# ------------------------------------------------------------------------------
def read_seed(seed: Optional[int] = None) -> Optional[int]:
    if seed is not None:
        return int(seed)
    if SEED_FILE.exists():
        return int(SEED_FILE.read_text().strip())
    return None


def seed_for(team_id: str) -> int:
    """Identical to tools/make_seed.py::seed_for (used only to cross-check --team-id)."""
    import hashlib
    h = hashlib.blake2b(team_id.strip().encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % (2**31 - 1)


def resolve_weights(seed: Optional[int], team_id: Optional[str] = None,
                    override: Optional[Tuple[float, float, float]] = None) -> Tuple[Dict[str, float], str]:
    """
    The 3 soft-constraint weights, MAPPING (team decision, record it in DECISIONS.md):
        w1 -> fairness penalty   (w_fair)
        w2 -> fatigue penalty    (w_fat)
        w3 -> location penalty   (w_loc)
    All three appear in the objective  Z = w1*F + w2*T + w3*L  (see model.py).

    OFFICIAL values: tools/make_seed.py defines them as
        rng = random.Random(seed);  [round(rng.uniform(0.5, 2.0), 2) for _ in range(3)]
    i.e. they depend ONLY on the integer seed, so `python run_all.py --seed <seed>`
    is enough -- no team id needed.  We reproduce that formula exactly.
    If --team-id is given we also check that it hashes to the seed in use.
    An explicit override (--weights a b c) wins and is labelled as such.
    """
    if override is not None:
        w = tuple(float(v) for v in override)
        src = "command-line override (NOT the official weights)"
    else:
        if seed is None:
            raise ValueError("a seed is required (pass --seed or create data/seed.txt)")
        rng = random.Random(seed)
        w = tuple(round(rng.uniform(0.5, 2.0), 2) for _ in range(3))
        src = "official formula of tools/make_seed.py (function of the seed)"
    if team_id is not None and seed is not None and seed_for(team_id) != seed:
        print(f"WARNING: --team-id {team_id!r} hashes to {seed_for(team_id)}, not to seed {seed}. "
              f"Check data/seed.txt and the team id (must match byte-for-byte).")
    return {"w_fair": w[0], "w_fat": w[1], "w_loc": w[2]}, src


def simulate_preferences(invigilators: List[str], seed: Optional[int]) -> Dict[str, str]:
    """Location preference in 3 categories, uniformly at random, reproducible from the seed."""
    rng = random.Random(f"prefs-{seed}")
    return {i: rng.choice(CATEGORIES) for i in sorted(invigilators)}


# ------------------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------------------
def _parse_gio(gio: str) -> Tuple[int, int]:
    h, m = str(gio).replace("g", ":").split(":")
    return int(h), int(m)


def load_instance(path: Path = DEFAULT_DATA_PATH, seed: Optional[int] = None,
                  team_id: Optional[str] = None,
                  weights: Optional[Tuple[float, float, float]] = None,
                  availability: str = "day", verbose: bool = True) -> IAPInstance:
    seed = read_seed(seed)
    df = pd.read_excel(path)                 # positional columns: robust to Vietnamese headers
    df = df.iloc[:, :9].copy()
    df.columns = ["ca", "ngay", "gio", "shift", "role", "cb", "dur", "thu", "campus"]
    df = df.dropna(subset=["shift", "cb"])

    # ---- campus: from the column, else from the role prefix ---------------------
    df["prefix"] = df["role"].astype(str).str.split("_").str[0]
    from_role = df["prefix"].map(ROLE_PREFIX_TO_CAMPUS)
    from_col = df["campus"].map(lambda v: None if pd.isna(v) else f"CS{str(v).strip()[-1]}")
    known = from_col.notna()
    consistency = float((from_col[known] == from_role[known]).mean())
    n_imputed = int((~known).sum())
    df["cs"] = from_col.where(known, from_role)
    if df["cs"].isna().any():
        raise ValueError("cannot determine campus for some rows")

    # ---- sessions -----------------------------------------------------------------
    sessions: Dict[str, Session] = {}
    counts = df.groupby(["shift", "cs"]).size()
    for (shift_id, cs), n in counts.items():
        row = df[(df["shift"] == shift_id) & (df["cs"] == cs)].iloc[0]
        h, m = _parse_gio(row["gio"])
        day = row["ngay"].date()
        start = dt.datetime.combine(day, dt.time(h, m))
        end = start + dt.timedelta(minutes=int(row["dur"]) if pd.notna(row["dur"]) else 150)
        period = "M" if h < 12 else ("A" if h < 18 else "N")
        sid = f"{shift_id}_{cs}"
        sessions[sid] = Session(sid, shift_id, day, int(str(shift_id).split("_")[1]), period,
                                cs, start, end, day.isocalendar()[1], int(n))
    sessions = dict(sorted(sessions.items(), key=lambda kv: (kv[1].start, kv[1].campus)))

    # ---- Overlap(j,k) -------------------------------------------------------------
    overlaps: Set[frozenset] = set()
    by_day = defaultdict(list)
    for s in sessions.values():
        by_day[s.date].append(s)
    for lst in by_day.values():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                if lst[a].start < lst[b].end and lst[b].start < lst[a].end:
                    overlaps.add(frozenset((lst[a].sid, lst[b].sid)))

    # ---- baseline + presence --------------------------------------------------------
    baseline = {(r.cb, f"{r.shift}_{r.cs}") for r in df.itertuples()}
    present: Dict[str, Set[dt.date]] = defaultdict(set)
    for r in df.itertuples():
        present[r.cb].add(r.ngay.date())
    invigilators = sorted(df["cb"].unique())

    w, wsrc = resolve_weights(seed, team_id, weights)
    inst = IAPInstance(invigilators=invigilators, sessions=sessions, overlaps=overlaps,
                       present_days=dict(present), baseline=baseline,
                       pref=simulate_preferences(invigilators, seed), weights=w, seed=seed,
                       availability_mode=availability)
    inst.busy = compute_busy(inst, availability)
    inst.notes = {"campus_imputed_rows": n_imputed,
                  "role_prefix_vs_campus_consistency": round(consistency, 4),
                  "weights_source": wsrc, "rows": int(len(df))}

    # sanity: baseline must satisfy the hard rules it is going to be compared against
    for j, s in sessions.items():
        got = sum(1 for (i, jj) in baseline if jj == j)
        assert got == s.capacity, f"capacity mismatch on {j}"
    if verbose:
        print_summary(inst)
    return inst


def print_summary(inst: IAPInstance) -> None:
    ss = inst.sessions.values()
    print("=" * 78)
    print("M2 / 2.1 + 2.6  PARAMETERS EXTRACTED FROM THE DATASET")
    print("=" * 78)
    print(f"|I| invigilators         : {len(inst.invigilators)}")
    print(f"|J| sessions             : {len(inst.sessions)}  "
          f"(shift ids: {len({s.shift_id for s in ss})}, "
          f"CS1: {sum(s.campus == 'CS1' for s in ss)}, CS2: {sum(s.campus == 'CS2' for s in ss)})")
    print(f"sum cap_j (assignments)  : {inst.total_demand}   mean load = {inst.mean_load:.4f}")
    print(f"horizon                  : {inst.days()[0]} .. {inst.days()[-1]}  "
          f"({len(inst.days())} days, {len(inst.weeks())} ISO weeks)")
    print(f"Overlap pairs            : {len(inst.overlaps)}")
    print(f"availability mode        : {inst.availability_mode}   |Busy| = {len(inst.busy)} "
          f"of {len(inst.invigilators) * len(inst.sessions)} (i,j) pairs")
    cats = {c: sum(1 for v in inst.pref.values() if v == c) for c in CATEGORIES}
    print(f"SIMULATED preferences    : {cats}   (seed = {inst.seed})")
    print(f"SIMULATED weights        : {inst.weights}   [{inst.notes['weights_source']}]")
    print(f"campus imputed from role : {inst.notes['campus_imputed_rows']} rows "
          f"(prefix/campus consistency on the rest = "
          f"{inst.notes['role_prefix_vs_campus_consistency']:.0%})")
    print("=" * 78)


def dump_parameters(inst: IAPInstance, out_dir: Path) -> None:
    """Write the extracted + simulated data next to the results (reproducible from the seed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"session": j, "shift_id": s.shift_id, "date": s.date.isoformat(), "slot": s.slot,
             "period": s.period, "campus": s.campus, "week": s.week, "capacity": s.capacity,
             "start": s.start.isoformat(timespec="minutes"), "end": s.end.isoformat(timespec="minutes")}
            for j, s in inst.sessions.items()]
    pd.DataFrame(rows).to_csv(out_dir / "sessions.csv", index=False)
    load = defaultdict(int)
    for (i, _j) in inst.baseline:
        load[i] += 1
    pd.DataFrame([{"invigilator": i, "preference": inst.pref[i],
                   "preferred_campus": inst.preferred_campus(i) or "-",
                   "baseline_load": load[i],
                   "days_present": len(inst.present_days.get(i, ()))}
                  for i in inst.invigilators]).to_csv(out_dir / "invigilators_simulated_prefs.csv",
                                                      index=False)
    (out_dir / "weights.json").write_text(json.dumps(
        {"seed": inst.seed, **inst.weights, "source": inst.notes["weights_source"],
         "mapping": {"w1": "w_fair (fairness)", "w2": "w_fat (fatigue)", "w3": "w_loc (location)"}}, indent=2), encoding="utf-8")