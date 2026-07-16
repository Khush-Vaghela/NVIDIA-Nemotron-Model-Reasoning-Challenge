"""Concise, induction-explicit traces for equation_v2 -- conciseness is the objective.

Structure per trace:
  (a) approach in 2 lines (per-operator fit; operator marks the sign of negatives)
  (b) re-list the example pairs once (the model needs them transcribed)
  (c) identify the query operator and the examples that share it
  (d) a FEW rejected candidate operations (one compact line each, computed on the first
      group example) -- NOT the full 32-op enumeration that bloated the original to
      ~5,600 tokens
  (e) the winning rule, verified against EVERY group example on one compact line each
  (f) apply the rule to the query, one compact line, and box -- terminating exactly like
      the original corpus.

Every number shown is recomputed and asserted true at build time; the trace's derived
output is asserted equal to the solver-verified gold. Conciseness caps (median<3000,
p90<3000, max<4500 completion tokens) are enforced downstream in tokenize_corpus.py.

Output: OUT_DIR/traces_all.jsonl {id, reasoning}
Usage:  python make_traces.py
"""

from __future__ import annotations

import json
import random
import sys

from common import OUT_DIR, SEED, Rule, render

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")

# decoy candidates shown as "rejected" (never includes the held-out determinant ops, so
# that name never leaks into training traces)
DECOY_POOL = [
    "addition", "subtraction", "reverse subtraction", "absolute difference",
    "multiplication", "concatenation", "reverse concatenation", "digit add mod10",
    "modulo", "digit multiply",
]


def _d(s: str) -> tuple[int, int]:
    return int(s[0]), int(s[1])


def op_formula(op: str, sa: str, sb: str) -> tuple[str, object]:
    """Return (human formula string, raw value) for an operation on operand strings
    sa, sb (already operand-reversed if the action says so). Value is int or str; None
    if undefined."""
    ia, ib = int(sa), int(sb)
    a1, a2 = _d(sa)
    b1, b2 = _d(sb)
    F = {
        "concatenation": (f"{sa}||{sb}", sa + sb),
        "reverse concatenation": (f"{sb}||{sa}", sb + sa),
        "addition": (f"{ia}+{ib}", ia + ib),
        "absolute difference": (f"|{ia}-{ib}|", abs(ia - ib)),
        "negated absolute difference": (f"-|{ia}-{ib}|", -abs(ia - ib)),
        "subtraction": (f"{ia}-{ib}", ia - ib),
        "reverse subtraction": (f"{ib}-{ia}", ib - ia),
        "multiplication": (f"{ia}*{ib}", ia * ib),
        "multiply+1": (f"{ia}*{ib}+1", ia * ib + 1),
        "multiply-1": (f"{ia}*{ib}-1", ia * ib - 1),
        "add+1": (f"{ia}+{ib}+1", ia + ib + 1),
        "add-1": (f"{ia}+{ib}-1", ia + ib - 1),
        "sub+1": (f"{ia}-{ib}+1", ia - ib + 1),
        "sub-1": (f"{ia}-{ib}-1", ia - ib - 1),
        "max mod min": (
            f"max({ia},{ib}) mod min({ia},{ib})",
            (max(ia, ib) % min(ia, ib)) if min(ia, ib) != 0 else None,
        ),
        "integer division": (f"{ia}//{ib}", (ia // ib) if ib != 0 else None),
        "modulo": (f"{ia} mod {ib}", (ia % ib) if ib != 0 else None),
        "reverse division": (f"{ib}//{ia}", (ib // ia) if ia != 0 else None),
        "reverse modulo": (f"{ib} mod {ia}", (ib % ia) if ia != 0 else None),
        "digit absolute diff": (f"|{a1}-{b1}|,|{a2}-{b2}|",
                                str(abs(a1 - b1)) + str(abs(a2 - b2))),
        "digit add mod10": (f"({a1}+{b1})%10,({a2}+{b2})%10",
                            str((a1 + b1) % 10) + str((a2 + b2) % 10)),
        "digit sub mod10": (f"({a1}-{b1})%10,({a2}-{b2})%10",
                            str((a1 - b1) % 10) + str((a2 - b2) % 10)),
        "cross multiply": (f"{a1}*{b1}+{a2}*{b2}", a1 * b1 + a2 * b2),
        "cross multiply rev": (f"{a1}*{b2}+{a2}*{b1}", a1 * b2 + a2 * b1),
        "digit multiply": (f"{a1}*{b1},{a2}*{b2}", str(a1 * b1) + str(a2 * b2)),
        "digit multiply rev": (f"{a1}*{b2},{a2}*{b1}", str(a1 * b2) + str(a2 * b1)),
        "digit sum diff": (f"({a1}+{a2})-({b1}+{b2})", (a1 + a2) - (b1 + b2)),
        "digit sum sum": (f"({a1}+{a2})+({b1}+{b2})", (a1 + a2) + (b1 + b2)),
        "digit product diff": (f"{a1}*{a2}-{b1}*{b2}", a1 * a2 - b1 * b2),
        "digit product sum": (f"{a1}*{a2}+{b1}*{b2}", a1 * a2 + b1 * b2),
        "determinant": (f"{a1}*{b2}-{a2}*{b1}", a1 * b2 - a2 * b1),
        "abs determinant": (f"|{a1}*{b2}-{a2}*{b1}|", abs(a1 * b2 - a2 * b1)),
    }
    return F[op]


def apply_oneline(rule: Rule, A: str, B: str, sym: str) -> tuple[str, str]:
    """One compact line computing render(rule, A, B). Asserts equality with render()."""
    sa = A[::-1] if rule.opnd_rev else A
    sb = B[::-1] if rule.opnd_rev else B
    formula, raw = op_formula(rule.op, sa, sb)
    assert raw is not None
    seg = []
    if rule.opnd_rev:
        seg.append(f"reverse operands -> {sa},{sb}")
    s = str(raw)
    seg.append(f"{formula} = {raw}")
    cur = s
    if rule.res_rev:
        cur = s[::-1]
        seg.append(f"reverse result -> {cur}")
    if "-" in cur:
        enc = cur.replace("-", sym)
        seg.append(f"negative, write '-' as '{sym}' -> {enc}")
        cur = enc
    gold = render(rule, A, B, sym)
    assert cur == gold, f"oneline {cur} != render {gold}"
    return "; ".join(seg), gold


def make_trace(p: dict) -> str:
    sym = p["op_symbol"]
    rep = Rule(p["rep_rule"]["op"], p["rep_rule"]["opnd_rev"], p["rep_rule"]["res_rev"])
    examples = p["examples"]  # (a, op, b, r)
    group = [(g[0], g[1], g[2]) for g in p["group"]]  # (a, b, r)
    rng = random.Random(f"{SEED}:trace:{p['id']}")

    lines = [
        "I need to find the transformation rule for each operator from the examples, then "
        "apply the rule for the query's operator. Operands are two digits; the operator "
        "symbol is decorative, except a negative result is written with the operator "
        "symbol in place of the minus sign.",
        "I will put my final answer inside \\boxed{}.",
        "",
        "Examples:",
    ]
    for (a, o, b, r) in examples:
        lines.append(f"  {a}{o}{b} = {r}")

    lines += [
        "",
        f"The query is {p['query'][0]}{sym}{p['query'][2]}, operator '{sym}'.",
        f"Examples using '{sym}': " + ", ".join(f"{a}{sym}{b} = {r}" for (a, b, r) in group)
        + ".",
        "",
        f"Fitting the rule for '{sym}'. Trying candidate operations on {group[0][0]}{sym}"
        f"{group[0][1]} (expected {group[0][2]}):",
    ]

    # rejected decoys: 2 ops (identity action) that differ from expected on group[0]
    g0a, g0b, g0r = group[0]
    shown = 0
    for op in rng.sample(DECOY_POOL, len(DECOY_POOL)):
        if op == rep.op:
            continue
        formula, raw = op_formula(op, g0a, g0b)
        if raw is None:
            continue
        if str(raw) == g0r:
            continue  # accidentally matches -> not a clean rejection, skip
        lines.append(f"  {op}: {formula} = {raw} != {g0r}, reject.")
        shown += 1
        if shown == 2:
            break

    # winning rule verified against EVERY group example
    lines.append(f"  Try {rep.label()}:")
    for (a, b, r) in group:
        line, got = apply_oneline(rep, a, b, sym)
        assert got == r, f"{p['id']}: winner fails example {a}{sym}{b}={r} (got {got})"
        lines.append(f"    {a}{sym}{b}: {line} = {got} (matches {r})")
    lines.append(f"  All '{sym}' examples match, so the rule is {rep.label()}.")

    # apply to query
    qa, _, qb = p["query"]
    qline, gold = apply_oneline(rep, qa, qb, sym)
    assert gold == p["answer"], f"{p['id']}: derived {gold} != gold {p['answer']}"
    lines += [
        "",
        f"Apply to {qa}{sym}{qb}: {qline} = {gold}.",
        "",
        "I will now return the answer in \\boxed{}",
        f"The answer in \\boxed{{–}} is \\boxed{{{gold}}}",
    ]
    return "\n".join(lines)


def main() -> None:
    problems = [json.loads(l) for l in open(OUT_DIR / "problems_all.jsonl")]
    n = 0
    with open(OUT_DIR / "traces_all.jsonl", "w") as f:
        for p in problems:
            t = make_trace(p)
            assert t.endswith(f"\\boxed{{{p['answer']}}}"), p["id"]
            assert "</think>" not in t and "<|im_end|>" not in t, p["id"]
            f.write(json.dumps({"id": p["id"], "reasoning": t}) + "\n")
            n += 1
    print(f"Wrote {n} traces (token distribution measured/enforced in tokenize_corpus.py)")


if __name__ == "__main__":
    main()
