"""Rule solver for equation_numeric_deduce.

Given the example pairs that share the QUERY operator, find every rule
(operation x operand_reverse x result_reverse) from the 128-candidate menu that
reproduces ALL of those examples exactly (string-equal, negative encoding included).

The group SOLVES the gold iff the consistent set is non-empty and all consistent rules
yield the SAME, defined result on the query (gold invariant to representative choice --
the same criterion as bit_v2). Distinct operation NAMES may remain in the consistent set
because some operations are functionally identical (e.g. concatenation == reverse
concatenation with reversed operands AND result); the gold is still uniquely determined.

Speed: every rule's full behaviour over all 100x100 operand pairs is precomputed once into
a signature; functional-equivalence classes follow, so "consistent set collapsed to one
gold" is an O(1) set-of-class-ids check. The negative-encoding operator symbol only
substitutes the '-' character uniformly, so equivalence classes are symbol-independent
(signatures use a fixed symbol).

Self-test: `python solve_rules.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from common import Rule, all_rules, render

_SIG_SYM = "|"  # fixed symbol for equivalence signatures (classes are symbol-independent)
_PROBE = [(f"{a:02d}", f"{b:02d}") for a in range(100) for b in range(100)]

_ALL: list[Rule] | None = None
_SIG: dict[Rule, tuple] | None = None
_CLASS: dict[Rule, int] | None = None


def rules() -> list[Rule]:
    global _ALL
    if _ALL is None:
        _ALL = all_rules()
    return _ALL


def _ensure_sigs() -> None:
    global _SIG, _CLASS
    if _SIG is not None:
        return
    _SIG = {}
    sig_to_class: dict[tuple, int] = {}
    _CLASS = {}
    for r in rules():
        sig = tuple(render(r, a, b, _SIG_SYM) for (a, b) in _PROBE)
        _SIG[r] = sig
        cid = sig_to_class.setdefault(sig, len(sig_to_class))
        _CLASS[r] = cid


def class_id(r: Rule) -> int:
    _ensure_sigs()
    assert _CLASS is not None
    return _CLASS[r]


@dataclass
class SolveResult:
    consistent: list[Rule]          # all rules reproducing every group example
    gold: str | None                # query render, if invariant across consistent set
    unique: bool                    # gold well-defined + invariant across consistent set
    rep: Rule | None                # representative (canonical priority) if unique
    ops: set[str]                   # distinct operation names in the consistent set
    classes: set[int]               # distinct functional-equivalence classes


def consistent_rules(group: list[tuple[str, str, str]], op_symbol: str) -> list[Rule]:
    """All candidate rules reproducing every (A,B,R) in the group (string-exact)."""
    out = []
    for r in rules():
        ok = True
        for a, b, res in group:
            got = render(r, a, b, op_symbol)
            if got is None or got != res:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def solve(group: list[tuple[str, str, str]], op_symbol: str,
          query: tuple[str, str]) -> SolveResult:
    cons = consistent_rules(group, op_symbol)
    qa, qb = query
    golds = {render(r, qa, qb, op_symbol) for r in cons}
    ops = {r.op for r in cons}
    classes = {class_id(r) for r in cons}
    gold = next(iter(golds)) if len(golds) == 1 else None
    unique = bool(cons) and gold is not None and None not in golds
    rep = cons[0] if unique else None
    if unique:
        for a, b, res in group:
            assert render(rep, a, b, op_symbol) == res, (
                f"rep {rep} fails example {a}{op_symbol}{b}={res}"
            )
        assert render(rep, qa, qb, op_symbol) == gold
    return SolveResult(consistent=cons, gold=gold, unique=unique, rep=rep,
                       ops=ops, classes=classes)


def collapsed(cons: list[Rule]) -> bool:
    """True iff all consistent rules are functionally identical (one equivalence class)."""
    return len({class_id(r) for r in cons}) <= 1


def disambiguating_pair(cons: list[Rule], op_symbol: str,
                        used: set[tuple[str, str]]) -> tuple[str, str] | None:
    """An unused operand pair where consistent rules from >=2 classes disagree, or None
    if the consistent set is functionally collapsed. Compares class representatives only."""
    if collapsed(cons):
        return None
    _ensure_sigs()
    assert _SIG is not None
    reps: dict[int, Rule] = {}
    for r in cons:
        reps.setdefault(class_id(r), r)
    rep_list = list(reps.values())
    for idx, (a, b) in enumerate(_PROBE):
        if (a, b) in used:
            continue
        vals = {_SIG[r][idx] for r in rep_list}
        if len(vals) > 1:
            return (a, b)
    return None


def next_example_for_target(cons: list[Rule], target: Rule, op_symbol: str,
                            used: set[tuple[str, str]]) -> tuple[str, str, str] | None:
    """An unused operand pair, DEFINED under `target`, where the consistent rules
    disagree -- returned as (A, B, result-under-target). Signature-backed (fast)."""
    if collapsed(cons):
        return None
    _ensure_sigs()
    assert _SIG is not None
    reps: dict[int, Rule] = {}
    for r in cons:
        reps.setdefault(class_id(r), r)
    rep_list = list(reps.values())
    tsig = _SIG[target]
    for idx, (a, b) in enumerate(_PROBE):
        if (a, b) in used or tsig[idx] is None:
            continue
        if len({_SIG[r][idx] for r in rep_list}) > 1:
            return (a, b, render(target, a, b, op_symbol))
    return None


# ---------------- self-test ----------------
def _self_test() -> None:
    import csv
    import json
    import random
    import re

    from common import OPERATOR_ALPHABET, TRAIN_CSV

    _ensure_sigs()
    assert _CLASS is not None
    print(f"precomputed {len(rules())} rules -> {len(set(_CLASS.values()))} "
          f"functional-equivalence classes")

    rng = random.Random(0)
    all_r = rules()
    ok1 = 0
    for _ in range(300):
        r = rng.choice(all_r)
        sym = rng.choice(OPERATOR_ALPHABET)
        group = []
        bad = False
        for _ in range(6):
            a, b = f"{rng.randint(0,99):02d}", f"{rng.randint(0,99):02d}"
            res = render(r, a, b, sym)
            if res is None:
                bad = True
                break
            group.append((a, b, res))
        if bad:
            continue
        qa, qb = f"{rng.randint(0,99):02d}", f"{rng.randint(0,99):02d}"
        gold_true = render(r, qa, qb, sym)
        if gold_true is None:
            continue
        cons = consistent_rules(group, sym)
        assert r in cons
        assert any(render(c, qa, qb, sym) == gold_true for c in cons)
        ok1 += 1
    print(f"self-test 1 (random known rules): OK ({ok1} trials)")

    csv.field_size_limit(10**9)
    cat = {}
    for line in open(TRAIN_CSV.parent / "corpus.jsonl"):
        d = json.loads(line)
        cat.setdefault(d["category"], []).append(d["problem_id"])
    prompts = {r["id"]: r["prompt"] for r in csv.DictReader(open(TRAIN_CSV, newline=""))}
    exline = re.compile(r"^(\d{2})(\S)(\d{2}) = (\S+)$")
    qline = re.compile(r"Now, determine the result for:\s*(\d{2})(\S)(\d{2})")
    n = found = 0
    for pid in cat["equation_numeric_deduce"]:
        pr = prompts[pid]
        exs = []
        for ln in pr.splitlines():
            m = exline.match(ln.strip())
            if m:
                exs.append((m.group(1), m.group(2), m.group(3), m.group(4)))
        qm = qline.search(pr)
        if not qm:
            continue
        qop = qm.group(2)
        group = [(a, b, res) for (a, o, b, res) in exs if o == qop]
        if not group:
            continue
        n += 1
        if consistent_rules(group, qop):
            found += 1
    print(f"self-test 2 (real deduce groups w/ a consistent menu rule): {found}/{n} "
          f"({100*found//n}%) -- remainder are '-'-operator sign collisions / single-"
          f"example rules outside the menu (informational; see README)")


if __name__ == "__main__":
    _self_test()
