#!/usr/bin/env python3
"""
make_seed.py  -  Compute your team's seed AND soft-constraint weights deterministically
from your team ID. Nothing is handed out: you generate these yourselves and the
instructors recompute the identical values at grading. This matches
grading/make_team_seeds.py byte-for-byte.

Your team ID MUST be the canonical string, exactly:
    CO2011-261-<group>-<smallest student ID>
    group in {L, CC, A01, TN01}   (L = L01/L02/L03, CC = CC01..CC05)
e.g. CO2011-261-L-2252107     (NOT L01, NOT a group number)
The seed is a hash of this exact string, so any difference in case, hyphens, or the
group/section token yields a DIFFERENT seed. Match it exactly.

Usage:
    python tools/make_seed.py CO2011-261-L-2252107            # prints the seed
    python tools/make_seed.py CO2011-261-L-2252107 --weights  # prints seed + 3 weights
    python tools/make_seed.py CO2011-261-L-2252107 > data/seed.txt

Then use it everywhere:
    python run_all.py --seed $(cat data/seed.txt)
"""
import hashlib, random, sys

def seed_for(team_id: str) -> int:
    h = hashlib.blake2b(team_id.strip().encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % (2**31 - 1)

def soft_weights(team_id: str):
    # Exactly three soft-constraint weights in [0.5, 2.0], drawn IN ORDER from a
    # Python random.Random seeded with the integer seed, rounded to 2 decimals.
    rng = random.Random(seed_for(team_id))
    return [round(rng.uniform(0.5, 2.0), 2) for _ in range(3)]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python tools/make_seed.py <TEAM_ID> [--weights]")
    tid = sys.argv[1]
    print(seed_for(tid))
    if "--weights" in sys.argv[2:]:
        print("soft_weights =", soft_weights(tid))
