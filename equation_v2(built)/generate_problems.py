"""Generate equation_numeric_deduce problems with solver-verified, uniquely-determined
rules.

Per problem:
  1. Sample a TARGET rule = (operation from the cell's pool, operand_reverse,
     result_reverse) and a query operator symbol.
  2. Build the QUERY-operator example group: start with a few random operand pairs
     (rendered by the target rule, skipping operands where the op is undefined), then
     while the consistent set (over the FULL 128-rule menu) is NOT functionally collapsed,
     append a disambiguating operand pair (one where surviving rules disagree), up to
     MAX_GROUP; else resample. A collapsed consistent set => the gold is invariant to the
     representative for ANY query (the operation is uniquely determined up to functional
     equivalence).
  3. Add 0-2 DISTRACTOR operators (different symbols, their own rules, 1-2 examples each)
     to match the real multi-operator format. They never touch the query-operator group.
  4. Query operand pair (unused); gold = render(target, query). VERIFY: the solver over
     the query-operator group reproduces the gold and is unique; the representative rule
     reproduces every group example exactly (asserted inside solve()).

Held-out cell uses ONLY the determinant operations (entire op types absent from train).
Prompt = byte-exact original template (asserted against train.csv in verify_tokenizer).

Output: OUT_DIR/problems_all.jsonl
Usage:  python generate_problems.py   (gate enforced)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter

from common import (
    OPERATOR_ALPHABET,
    OUT_DIR,
    SEED,
    Rule,
    build_problem_prompt,
    render,
)
from solve_rules import (
    collapsed,
    consistent_rules,
    next_example_for_target,
    solve,
)

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")

# ---------------- operation pools ----------------
POOLS = {
    "arithmetic": [
        "addition", "subtraction", "reverse subtraction",
        "absolute difference", "negated absolute difference", "multiplication",
    ],
    "concat": ["concatenation", "reverse concatenation"],
    "modular": [
        "max mod min", "integer division", "modulo",
        "reverse division", "reverse modulo",
    ],
    "offsets": ["multiply+1", "multiply-1", "add+1", "add-1", "sub+1", "sub-1"],
    "digitwise": [
        "digit absolute diff", "digit add mod10", "digit sub mod10",
        "digit multiply", "digit multiply rev",
    ],
    "digit_agg": [
        "cross multiply", "cross multiply rev", "digit sum diff",
        "digit sum sum", "digit product diff", "digit product sum",
    ],
    "determinant": ["determinant", "abs determinant"],  # HELD-OUT
}
# every train op pool (for sampling distractor rules -- never includes held-out ops)
TRAIN_OPS = [op for cell, ops in POOLS.items() if cell != "determinant" for op in ops]

# ---------------- cells (EDIT HERE) ----------------
CELLS = [
    dict(cell="arithmetic", n=900, pool="arithmetic"),
    dict(cell="concat", n=300, pool="concat"),
    dict(cell="modular", n=500, pool="modular"),
    dict(cell="offsets", n=500, pool="offsets"),
    dict(cell="digitwise", n=600, pool="digitwise"),
    dict(cell="digit_agg", n=600, pool="digit_agg"),
    dict(cell="determinant", n=250, pool="determinant", heldout=True),
]

START_GROUP = 2          # query-operator examples to start with
MAX_GROUP = 6            # cap on query-operator example count (uniqueness is required)
MAX_DISTRACTORS = 2      # extra operator symbols acting as distractors


def _rand_pair(rng: random.Random) -> tuple[str, str]:
    return f"{rng.randint(0, 99):02d}", f"{rng.randint(0, 99):02d}"


def _make_group(rng: random.Random, target: Rule, sym: str):
    """Return a query-operator group [(A,B,R)...] whose consistent set is fully collapsed
    (every consistent rule is functionally identical over ALL operands -> gold invariant
    for ANY query), or None if that can't be reached within MAX_GROUP examples."""
    used: set[tuple[str, str]] = set()
    group: list[tuple[str, str, str]] = []
    tries = 0
    while len(group) < START_GROUP and tries < 400:
        tries += 1
        a, b = _rand_pair(rng)
        if (a, b) in used:
            continue
        r = render(target, a, b, sym)
        if r is None:
            continue
        used.add((a, b))
        group.append((a, b, r))
    if len(group) < START_GROUP:
        return None
    while True:
        cons = consistent_rules(group, sym)
        dp = next_example_for_target(cons, target, sym, used)
        if dp is None:
            # collapsed (gold invariant) or remaining disagreements only where target
            # is undefined -- accept only if truly collapsed
            return group if collapsed(cons) else None
        if len(group) >= MAX_GROUP:
            return None
        used.add((dp[0], dp[1]))
        group.append((dp[0], dp[1], dp[2]))


def _make_distractors(rng: random.Random, exclude_syms: set[str], used_global: set):
    """0..MAX_DISTRACTORS extra operators, each with its own rule + 1-2 examples."""
    k = rng.choices([0, 1, 2], weights=[0.15, 0.5, 0.35])[0]
    out = []
    syms = [s for s in OPERATOR_ALPHABET if s not in exclude_syms]
    rng.shuffle(syms)
    for i in range(k):
        if i >= len(syms):
            break
        sym = syms[i]
        rule = Rule(rng.choice(TRAIN_OPS), rng.random() < 0.5, rng.random() < 0.5)
        m = rng.choice([1, 2])
        ex = []
        tries = 0
        seen: set[tuple[str, str]] = set()
        while len(ex) < m and tries < 200:
            tries += 1
            a, b = _rand_pair(rng)
            if (a, b) in seen:
                continue
            r = render(rule, a, b, sym)
            if r is None:
                continue
            seen.add((a, b))
            ex.append((a, sym, b, r))
        out.extend(ex)
    return out


def make_problem(rng: random.Random | None, spec: dict, seq: int) -> dict:
    if rng is None:
        rng = random.Random(f"{SEED}:{spec['cell']}:{seq}")
    pool = POOLS[spec["pool"]]
    for _attempt in range(120):
        op = rng.choice(pool)
        target = Rule(op, rng.random() < 0.5, rng.random() < 0.5)
        sym = rng.choice(OPERATOR_ALPHABET)
        group = _make_group(rng, target, sym)
        if group is None:
            continue
        used = {(a, b) for (a, b, _) in group}
        qpair = None
        for _ in range(300):
            a, b = _rand_pair(rng)
            if (a, b) in used:
                continue
            if render(target, a, b, sym) is not None:
                qpair = (a, b)
                break
        if qpair is None:
            continue
        gold = render(target, qpair[0], qpair[1], sym)
        res = solve(group, sym, qpair)
        if not res.unique or res.gold != gold:
            continue
        rep = res.rep
        q_examples = [(a, sym, b, r) for (a, b, r) in group]
        distractors = _make_distractors(rng, {sym}, used)
        all_examples = q_examples + distractors
        rng.shuffle(all_examples)
        prompt = build_problem_prompt(all_examples, (qpair[0], sym, qpair[1]))
        assert any(o == sym for (_, o, _, _) in all_examples)
        pid = hashlib.sha256(f"eq_v2:{spec['cell']}:{seq}".encode()).hexdigest()[:8]
        return dict(
            id=pid,
            _seq=seq,
            cell=spec["cell"],
            heldout=bool(spec.get("heldout")),
            target_op=target.op,
            rep_rule={"op": rep.op, "opnd_rev": rep.opnd_rev, "res_rev": rep.res_rev},
            op_symbol=sym,
            n_group=len(group),
            n_distractor=len(distractors),
            n_examples=len(all_examples),
            n_operators=len({o for (_, o, _, _) in all_examples}),
            examples=all_examples,
            group=[list(g) for g in group],
            query=[qpair[0], sym, qpair[1]],
            answer=gold,
            prompt=prompt,
        )
    raise RuntimeError(f"failed to build unique problem for {spec['cell']} seq={seq}")


def _emit_chunk(spec: dict, start: int, count: int) -> None:
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
        grp_counts = Counter(p["n_group"] for p in cell_probs.values())
        ho = " [HELD-OUT]" if spec.get("heldout") else ""
        print(f"  {spec['cell']:<12} n={spec['n']}{ho} "
              f"examples={dict(sorted(ex_counts.items()))} group={dict(sorted(grp_counts.items()))}")
        for seq in range(spec["n"]):
            p = cell_probs[seq]
            assert p["id"] not in seen, f"id collision {p['id']}"
            seen.add(p["id"])
            problems.append(p)
    with open(OUT_DIR / "problems_all.jsonl", "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    n_ho = sum(p["heldout"] for p in problems)
    print(f"\nWrote {len(problems)} problems ({len(problems) - n_ho} train, {n_ho} held-out)")
    print("All query-operator groups uniquely determine the gold; "
          "all golds solver-verified against every group example.")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_env = os.environ.get("EQ_CELL")
    if os.environ.get("EQ_MERGE") == "1":
        _merge_parts()
        return
    if cell_env:
        spec = next(c for c in CELLS if c["cell"] == cell_env)
        _emit_chunk(spec, int(os.environ.get("EQ_START", "0")),
                    int(os.environ.get("EQ_COUNT", "100000")))
        return
    for spec in CELLS:
        _emit_chunk(spec, 0, spec["n"])
    _merge_parts()


if __name__ == "__main__":
    main()
