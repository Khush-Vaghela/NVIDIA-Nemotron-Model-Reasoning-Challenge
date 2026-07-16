"""Rule solver for bit_manipulation: fit each output bit independently.

For output bit j, the consistent set = all menu candidates whose truth table
agrees with every example pair at bit j. The bit is SOLVED iff all consistent
candidates share one truth table (unique function — gold invariant to
representative). solve() returns the per-bit representative (simplest by rank)
or reports which bits are ambiguous / unsatisfiable.

Self-test: `python solve_rules.py` runs the solver against random known rules
and against real train.csv bit problems, asserting reproduction of all examples.
"""

from __future__ import annotations

from dataclasses import dataclass

from common import Candidate, apply_rule, menu, str_to_bits


@dataclass
class SolveResult:
    solved: bool
    rule: list[Candidate] | None          # representative per bit (if solved)
    consistent_tables: list[set[int]]      # distinct truth tables per bit
    ambiguous_bits: list[int]
    unsat_bits: list[int]


def solve(examples: list[tuple[str, str]]) -> SolveResult:
    pairs = [(int(i, 2), str_to_bits(o)) for i, o in examples]
    cands = menu()
    rule: list[Candidate | None] = [None] * 8
    tables: list[set[int]] = []
    ambiguous: list[int] = []
    unsat: list[int] = []
    for j in range(8):
        consistent = [
            c for c in cands
            if all(((c.table >> x) & 1) == ob[j] for x, ob in pairs)
        ]
        distinct = {c.table for c in consistent}
        tables.append(distinct)
        if not consistent:
            unsat.append(j)
        elif len(distinct) > 1:
            ambiguous.append(j)
        else:
            rule[j] = min(consistent, key=lambda c: (c.rank, c.a or 0, c.b or 0))
    solved = not ambiguous and not unsat
    result = SolveResult(
        solved=solved,
        rule=rule if solved else None,  # type: ignore[arg-type]
        consistent_tables=tables,
        ambiguous_bits=ambiguous,
        unsat_bits=unsat,
    )
    if solved:
        # VERIFY: the solved rule must reproduce every example exactly
        for i, o in examples:
            got = apply_rule(result.rule, i)
            assert got == o, f"solved rule fails example {i} -> {o} (got {got})"
    return result


def disambiguating_input(tables_j: set[int], used: set[int]) -> int | None:
    """An unused input value where the surviving truth tables disagree."""
    ts = list(tables_j)
    for x in range(256):
        if x in used:
            continue
        vals = {(t >> x) & 1 for t in ts}
        if len(vals) > 1:
            return x
    return None


# ---------------- self-test ----------------
def _self_test() -> None:
    import csv
    import json
    import random

    from common import (
        PROMPT_INTRO,
        PROMPT_QUERY,
        TRAIN_CSV,
        bits_to_str,
        value_bits,
    )

    rng = random.Random(7)
    cands = menu()
    # 1) random known rules -> solver must recover an equivalent rule
    for trial in range(50):
        true_rule = [rng.choice(cands) for _ in range(8)]
        inputs = rng.sample(range(256), 10)
        examples = [
            (bits_to_str(value_bits(x)), apply_rule(true_rule, bits_to_str(value_bits(x))))
            for x in inputs
        ]
        res = solve(examples)
        if res.solved:
            for x in range(256):
                s = bits_to_str(value_bits(x))
                assert apply_rule(res.rule, s) == apply_rule(true_rule, s), \
                    "solved rule not equivalent to true rule"
    print("self-test 1 (random known rules): OK")

    # 2) real train.csv bit problems: solver output must reproduce all examples
    import re as _re
    csv.field_size_limit(10**9)
    rows = list(csv.DictReader(open(TRAIN_CSV, newline="")))
    bit_rows = [
        r for r in rows
        if r["prompt"].startswith(PROMPT_INTRO)
    ][:200]
    n_solved = 0
    for r in bit_rows:
        body = r["prompt"][len(PROMPT_INTRO):]
        m = _re.fullmatch(
            r"((?:[01]{8} -> [01]{8}\n)+)" + _re.escape(PROMPT_QUERY) + r"([01]{8})",
            body,
        )
        examples = _re.findall(r"([01]{8}) -> ([01]{8})", m.group(1))
        res = solve(examples)
        if res.solved:
            n_solved += 1
            # verification already asserted inside solve()
    print(f"self-test 2 (real train problems): {n_solved}/{len(bit_rows)} fully solved "
          f"by the menu (rest use rules outside the 1-2-bit menu or are ambiguous; "
          f"informational only)")


if __name__ == "__main__":
    _self_test()
