"""Generate bit_manipulation problems with solver-verified, uniquely-determined rules.

Per problem:
  1. Sample a true rule: 8 per-bit functions from the menu. Train cells use ONLY
     basic ops {C0,C1,ID,NOT,AND,OR,XOR}; the held-out cell forces >=1 negated
     binary op (AND-NOT / OR-NOT / XOR-NOT) — an entire rule STRUCTURE absent
     from training. Operands are biased toward neighbor bits (matches the
     offsets observed in solver-unique real train.csv rules).
  2. Start with 7 random distinct example inputs; run the solver over the FULL
     menu; while any output bit is ambiguous, append a disambiguating input
     (one where surviving truth tables disagree), up to MAX_EXAMPLES; else
     resample. Final problems are UNIQUELY determined — no ambiguity.
  3. VERIFY: solver rule reproduces every example (asserted inside solve());
     gold = apply(rule, query), exactly 8 binary chars; solver's rule and the
     true rule agree on the query (and on all 256 inputs).
Prompt = byte-exact original template (asserted against train.csv in common.py).

Output: OUT_DIR/problems_all.jsonl
Usage:  python generate_problems.py   (gate enforced)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter

from common import (
    BASIC_OPS,
    NEGATED_OPS,
    OUT_DIR,
    SEED,
    apply_rule,
    basic_tables,
    bits_to_str,
    build_problem_prompt,
    menu,
    value_bits,
)
from solve_rules import disambiguating_input, solve

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")

# ---------------- cells (EDIT HERE) ----------------
CELLS = [
    dict(cell="basic_simple", n=800, ops="unary"),     # ID/NOT/constants only
    dict(cell="basic_mixed", n=2000, ops="basic"),     # + AND/OR/XOR
    dict(cell="negated_ops", n=300, ops="negated", heldout=True),  # HELD-OUT structure
]
START_EXAMPLES = 7
MAX_EXAMPLES = 12   # original problems use 7-10; we allow up to 12 to reach uniqueness
                    # (uniqueness is a hard requirement; example count distribution
                    #  is reported in the manifest) — FLAGGED in README

UNARY_W = [("ID", 0.30), ("NOT", 0.46), ("C0", 0.12), ("C1", 0.12)]
BASIC_W = [("ID", 0.13), ("NOT", 0.21), ("C0", 0.04), ("C1", 0.04),
           ("AND", 0.20), ("OR", 0.20), ("XOR", 0.18)]
NEG_W = [("ANDN", 0.40), ("ORN", 0.35), ("XORN", 0.25)]


def _pick(rng: random.Random, weights: list[tuple[str, float]]) -> str:
    r = rng.random()
    acc = 0.0
    for op, w in weights:
        acc += w
        if r <= acc:
            return op
    return weights[-1][0]


def _operands(rng: random.Random, j: int, k: int) -> list[int]:
    """k distinct operand bits; 70% neighbor-structured around output bit j."""
    if rng.random() < 0.7:
        pool = [(j - 1) % 8, (j + 1) % 8, j, (j - 2) % 8, (j + 2) % 8]
    else:
        pool = list(range(8))
    rng.shuffle(pool)
    out = []
    for x in pool:
        if x not in out:
            out.append(x)
        if len(out) == k:
            return out
    raise AssertionError


_BY_KEY = {}
for c in menu():
    _BY_KEY.setdefault((c.op, c.a, c.b), c)
    if c.op in ("AND", "OR", "XOR", "XORN") and c.a is not None:
        _BY_KEY.setdefault((c.op, c.b, c.a), c)  # symmetric aliases


def sample_rule(rng: random.Random, ops_kind: str) -> list:
    while True:
        rule = []
        for j in range(8):
            if ops_kind == "unary":
                op = _pick(rng, UNARY_W)
            elif ops_kind == "basic":
                op = _pick(rng, BASIC_W)
            else:  # negated cell: mix basic + negated, force >=1 negated below
                op = _pick(rng, NEG_W) if rng.random() < 0.35 else _pick(rng, BASIC_W)
            if op in ("C0", "C1"):
                cand = _BY_KEY[(op, None, None)]
            elif op in ("ID", "NOT"):
                (a,) = _operands(rng, j, 1)
                cand = _BY_KEY[(op, a, None)]
            else:
                a, b = _operands(rng, j, 2)
                cand = _BY_KEY[(op, a, b)]
            rule.append(cand)
        kinds = {c.op for c in rule}
        if ops_kind == "negated" and not (kinds & NEGATED_OPS):
            continue
        if ops_kind != "negated" and (kinds & NEGATED_OPS):
            continue
        # structural invariants via truth tables (the airtight leakage criterion)
        in_basic = [c.table in basic_tables() for c in rule]
        if ops_kind == "negated" and all(in_basic):
            continue  # negated op degenerated into a basic-expressible function
        if ops_kind != "negated":
            assert all(in_basic)
        # avoid fully-constant outputs (degenerate problems)
        if all(c.op in ("C0", "C1") for c in rule):
            continue
        return rule


def make_problem(rng: random.Random | None, spec: dict, seq: int) -> dict:
    # Per-problem deterministic RNG: reproducible regardless of chunking order.
    if rng is None:
        rng = random.Random(f"{SEED}:{spec['cell']}:{seq}")
    for _attempt in range(60):
        rule = sample_rule(rng, spec["ops"])
        used = set(rng.sample(range(256), START_EXAMPLES))
        while True:
            examples = [
                (bits_to_str(value_bits(x)), apply_rule(rule, bits_to_str(value_bits(x))))
                for x in sorted(used, key=lambda v: rng.random())
            ]
            res = solve(examples)
            if res.unsat_bits:
                raise AssertionError("true rule unsat — solver/menu bug")
            if res.solved:
                break
            if len(used) >= MAX_EXAMPLES:
                res = None
                break
            # add a disambiguating input for the first still-ambiguous bit
            added = False
            for j in res.ambiguous_bits:
                x = disambiguating_input(res.consistent_tables[j], used)
                if x is not None:
                    used.add(x)
                    added = True
                    break
            if not added:
                res = None
                break
        if res is None:
            continue
        # query: unused input
        query_pool = [x for x in range(256) if x not in used]
        query = bits_to_str(value_bits(rng.choice(query_pool)))
        gold = apply_rule(res.rule, query)
        # solver rule and true rule must agree EVERYWHERE (not just on examples)
        for x in range(256):
            s = bits_to_str(value_bits(x))
            assert apply_rule(res.rule, s) == apply_rule(rule, s), \
                "unique-fit rule differs from true rule"
        assert re.fullmatch(r"[01]{8}", gold)
        pid = hashlib.sha256(f"bit_v2:{spec['cell']}:{seq}".encode()).hexdigest()[:8]
        return dict(
            id=pid,
            _seq=seq,
            cell=spec["cell"],
            heldout=bool(spec.get("heldout")),
            rule=[{"op": c.op, "a": c.a, "b": c.b, "name": c.name()} for c in res.rule],
            rule_ops=sorted({c.op for c in res.rule}),
            examples=examples,
            n_examples=len(examples),
            query=query,
            answer=gold,
            prompt=build_problem_prompt(examples, query),
        )
    raise RuntimeError(f"failed to build unique problem for {spec['cell']} seq={seq}")


def _emit_chunk(spec: dict, start: int, count: int) -> None:
    """Generate seqs [start, start+count) for one cell into a part file.

    Per-problem seeding makes output identical however the work is chunked
    (needed for environments with per-call time limits)."""
    parts_dir = OUT_DIR / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    end = min(start + count, spec["n"])
    path = parts_dir / f"{spec['cell']}_{start:05d}.jsonl"
    with open(path, "w") as f:
        for seq in range(start, end):
            f.write(json.dumps(make_problem(None, spec, seq)) + "\n")
    print(f"  part {path.name}: seqs {start}..{end - 1} done")


def _merge_parts() -> None:
    problems = []
    seen = set()
    for spec in CELLS:
        cell_probs: dict[int, dict] = {}
        for path in sorted((OUT_DIR / "parts").glob(f"{spec['cell']}_*.jsonl")):
            for line in open(path):
                p = json.loads(line)
                cell_probs[p["_seq"]] = p
        assert sorted(cell_probs) == list(range(spec["n"])), (
            f"{spec['cell']}: missing seqs "
            f"{sorted(set(range(spec['n'])) - set(cell_probs))[:5]}..."
        )
        ex_counts = Counter(p["n_examples"] for p in cell_probs.values())
        ho = " [HELD-OUT]" if spec.get("heldout") else ""
        print(f"  {spec['cell']:<14} n={spec['n']}{ho} "
              f"example-count dist={dict(sorted(ex_counts.items()))}")
        for seq in range(spec["n"]):
            p = cell_probs[seq]
            assert p["id"] not in seen
            seen.add(p["id"])
            problems.append(p)
    with open(OUT_DIR / "problems_all.jsonl", "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    n_ho = sum(p["heldout"] for p in problems)
    print(f"\nWrote {len(problems)} problems ({len(problems) - n_ho} train, {n_ho} held-out)")
    print("All rules uniquely determined; all golds solver-verified against every example.")


def main() -> None:
    import os

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_env = os.environ.get("BIT_CELL")
    if os.environ.get("BIT_MERGE") == "1":
        _merge_parts()
        return
    if cell_env:  # chunked mode (sandbox/CI with per-call time limits)
        spec = next(c for c in CELLS if c["cell"] == cell_env)
        _emit_chunk(spec, int(os.environ.get("BIT_START", "0")),
                    int(os.environ.get("BIT_COUNT", "100000")))
        return
    # default: full sequential run (Kaggle)
    for spec in CELLS:
        _emit_chunk(spec, 0, spec["n"])
    _merge_parts()


if __name__ == "__main__":
    main()
