"""Build the per-family generalization probe set (PROBE ONLY — no corpus generation).

From train.csv + problems.jsonl, classify every problem into its family
(cryptarithm_* and equation_numeric_* sub-categories merged into one family each),
then sample ~N_PER_FAMILY problems per family, stratified by status
(rule_found / hypothesis_formed / rule_unknown) where multiple statuses exist:
equal per-status targets, shortfall redistributed to the remaining statuses
proportionally to availability.

Outputs (to OUT_DIR):
  probe.jsonl          {id, family, sub_category, status, prompt, gold}
  probe_counts.json    per-family x status counts (sampled vs available)

Usage: python build_probe_set.py
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------- paths (Kaggle-parameterized) ----------------
INPUT_DIR = Path(os.environ.get("NEMOTRON_INPUT", "/kaggle/input/nemotron-master"))
OUT_DIR = Path(os.environ.get("PROBE_OUT", "/kaggle/working/family_probe_out"))
TRAIN_CSV = INPUT_DIR / "train.csv"
PROBLEMS_JSONL = INPUT_DIR / "problems.jsonl"

N_PER_FAMILY = 80
SEED = 20260611

FAMILY_OF = {
    "bit_manipulation": "bit_manipulation",
    "cipher": "cipher",
    "unit_conversion": "unit_conversion",
    "gravity": "gravity",
    "numeral": "numeral",
    "equation_numeric_deduce": "equation_numeric",
    "equation_numeric_guess": "equation_numeric",
    "cryptarithm_deduce": "cryptarithm",
    "cryptarithm_guess": "cryptarithm",
}
STATUS_ORDER = ["rule_found", "hypothesis_formed", "rule_unknown"]


def main() -> None:
    csv.field_size_limit(10**9)
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = {r["id"]: r for r in csv.DictReader(open(TRAIN_CSV, newline=""))}
    meta = [json.loads(l) for l in open(PROBLEMS_JSONL)]
    assert all(m["id"] in rows for m in meta), "problems.jsonl id missing from train.csv"

    # family -> status -> [problem meta]
    pool: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for m in meta:
        fam = FAMILY_OF[m["category"]]
        pool[fam][m["status"]].append(m)

    probe: list[dict] = []
    counts: dict[str, dict] = {}
    for fam in sorted(pool):
        by_status = pool[fam]
        statuses = [s for s in STATUS_ORDER if by_status.get(s)]
        avail = {s: len(by_status[s]) for s in statuses}

        # equal targets per status, redistribute shortfall proportionally
        target = {s: N_PER_FAMILY // len(statuses) for s in statuses}
        for s in statuses[: N_PER_FAMILY % len(statuses)]:
            target[s] += 1
        # cap at availability, redistribute remainder where capacity remains
        for _ in range(len(statuses)):
            shortfall = sum(max(0, target[s] - avail[s]) for s in statuses)
            if not shortfall:
                break
            for s in statuses:
                target[s] = min(target[s], avail[s])
            spare = {s: avail[s] - target[s] for s in statuses}
            total_spare = sum(spare.values())
            if total_spare == 0:
                break
            for s in statuses:
                add = min(spare[s], round(shortfall * spare[s] / total_spare))
                target[s] += add
                shortfall -= add
                if shortfall <= 0:
                    break

        sampled: dict[str, int] = {}
        for s in statuses:
            picks = rng.sample(by_status[s], min(target[s], avail[s]))
            sampled[s] = len(picks)
            for m in picks:
                probe.append(
                    {
                        "id": m["id"],
                        "family": fam,
                        "sub_category": m["category"],
                        "status": m["status"],
                        "prompt": rows[m["id"]]["prompt"],
                        "gold": rows[m["id"]]["answer"],
                    }
                )
        counts[fam] = {
            "sampled_total": sum(sampled.values()),
            "sampled_by_status": sampled,
            "available_by_status": avail,
        }

    ids = [p["id"] for p in probe]
    assert len(ids) == len(set(ids)), "duplicate ids in probe"

    rng.shuffle(probe)  # avoid family-ordered batches at inference
    with open(OUT_DIR / "probe.jsonl", "w") as f:
        for p in probe:
            f.write(json.dumps(p) + "\n")
    with open(OUT_DIR / "probe_counts.json", "w") as f:
        json.dump(counts, f, indent=2)

    print(f"Wrote {len(probe)} probe problems -> {OUT_DIR / 'probe.jsonl'}")
    for fam, c in counts.items():
        print(f"  {fam:<18} n={c['sampled_total']:<3} by_status={c['sampled_by_status']}")
    fam_total = Counter(p["family"] for p in probe)
    assert all(v <= N_PER_FAMILY for v in fam_total.values())


if __name__ == "__main__":
    main()
