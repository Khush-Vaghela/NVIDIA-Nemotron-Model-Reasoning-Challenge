"""Concise joint-induction traces for cryptarithm_v2 -- conciseness is the objective
(truncation is the family's failure mode: base model ~7,610 tokens, 98.8% truncated).

POSITIONAL traces: the symbols are opaque tokens; identify the query operator's example
group, test the 4 symbol-arrangements {A.B, B.A, rev(A).rev(B), rev(B).rev(A)} on the first
example, confirm the winner on the rest, apply to the query, box.

ARITHMETIC traces: each symbol is a digit and the operator hides an arithmetic op. State the
deduced symbol->digit cipher and operation, VERIFY it on every example
(decode -> compute -> re-encode), apply to the query, box.

Every value shown is recomputed and asserted true; the derived output is asserted equal to
the code-verified gold. Caps (median<3500, max<4500 completion tokens) enforced in
tokenize_corpus.py. Terminates exactly: ...\boxed{ANS}\n</think>\n\boxed{ANS}<|im_end|>.

Output: OUT_DIR/traces_all.jsonl {id, reasoning}
"""

from __future__ import annotations

import json
import sys

from common import OUT_DIR, PosRule, pos_render

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")

POS4 = {
    "AB": PosRule("concat", False, False),
    "BA": PosRule("rconcat", False, False),
    "rArB": PosRule("concat", True, False),
    "rBrA": PosRule("rconcat", True, False),
}
POS_LABEL = {
    "AB": "left then right (A|B)",
    "BA": "right then left (B|A)",
    "rArB": "reverse each operand, left then right (revA|revB)",
    "rBrA": "reverse each operand, right then left (revB|revA)",
}

ARITH_OPS = {
    "addition": lambda a, b: a + b,
    "subtraction": lambda a, b: a - b,
    "reverse subtraction": lambda a, b: b - a,
    "absolute difference": lambda a, b: abs(a - b),
    "multiplication": lambda a, b: a * b,
    "modulo": lambda a, b: (a % b) if b else None,
    "reverse modulo": lambda a, b: (b % a) if a else None,
    "integer division": lambda a, b: (a // b) if b else None,
}
ARITH_SYM = {"addition": "+", "subtraction": "-", "reverse subtraction": "-(swapped)",
             "absolute difference": "abs diff", "multiplication": "*",
             "modulo": "mod", "reverse modulo": "mod(swapped)",
             "integer division": "//"}


def _arr(cls, A, B):
    return pos_render(POS4[cls], A, B)


def make_pos_trace(p):
    sym = p["op_symbol"]
    cls = p["pos_class"]
    group = [(g[0], g[1], g[2]) for g in p["group"]]
    qa, _, qb = p["query"]
    lines = [
        "The symbols are opaque tokens. For the query operator I must find how the two "
        "operand groups are rearranged into the result; then apply that to the query.",
        "I will put my final answer inside \\boxed{}.",
        "",
        "Examples:",
    ]
    for (a, o, b, r) in p["examples"]:
        lines.append(f"  {a}{o}{b} = {r}")
    g0a, g0b, g0r = group[0]
    lines += [
        "",
        f"Query operator is '{sym}'. Examples with '{sym}': "
        + ", ".join(f"{a}{sym}{b} = {r}" for (a, b, r) in group) + ".",
        "",
        f"Test the arrangements on {g0a}{sym}{g0b} (target {g0r}):",
    ]
    for c in ("AB", "BA", "rArB", "rBrA"):
        val = _arr(c, g0a, g0b)
        mark = "match" if val == g0r else "no"
        lines.append(f"  {POS_LABEL[c]} = {val} -> {mark}")
    # winner verified on all group examples
    for (a, b, r) in group:
        assert _arr(cls, a, b) == r, f"{p['id']}: winner fails {a}{sym}{b}={r}"
    lines.append(f"Rule: {POS_LABEL[cls]}; confirm on all '{sym}' examples: "
                 + ", ".join(f"{a}{sym}{b}={_arr(cls,a,b)}" for (a, b, r) in group) + " - all match.")
    gold = _arr(cls, qa, qb)
    assert gold == p["answer"], f"{p['id']}: derived {gold} != gold {p['answer']}"
    lines += [
        "",
        f"Apply to {qa}{sym}{qb}: {POS_LABEL[cls]} = {gold}.",
        "",
        "I will now return the answer in \\boxed{}",
        f"The answer in \\boxed{{–}} is \\boxed{{{gold}}}",
    ]
    return "\n".join(lines)


def make_arith_trace(p):
    sym = p["op_symbol"]
    op = p["op"]
    orv, rrv = p["orv"], p["rrv"]
    s2d = {k: int(v) for k, v in p["cipher"].items()}
    d2s = {d: s for s, d in s2d.items()}
    group = [(g[0], g[1], g[2]) for g in p["group"]]
    qa, _, qb = p["query"]
    f = ARITH_OPS[op]

    def dec(two):
        t = two[::-1] if orv else two
        return s2d[t[0]] * 10 + s2d[t[1]]

    act = []
    if orv:
        act.append("reverse the two operand symbols before reading")
    if rrv:
        act.append("reverse the result digits")
    act_s = ("; " + ", ".join(act)) if act else ""

    cipher_str = ", ".join(f"{k}={s2d[k]}" for k in sorted(s2d, key=lambda c: s2d[c]))
    lines = [
        "Each symbol is a digit and the operator hides an arithmetic operation. I deduce the "
        "symbol->digit code and the operation from the examples, then apply them to the query.",
        "I will put my final answer inside \\boxed{}.",
        "",
        "Examples:",
    ]
    for (a, o, b, r) in p["examples"]:
        lines.append(f"  {a}{o}{b} = {r}")
    lines += [
        "",
        f"Query operator is '{sym}'. The consistent code is: {cipher_str}.",
        f"The operation is {op}{act_s}.",
        "",
        "Verify on each example (decode, compute, encode):",
    ]
    for (a, b, r) in group:
        ia, ib = dec(a), dec(b)
        v = f(ia, ib)
        s = str(v)
        if rrv:
            s = s[::-1]
        enc = "".join(d2s[int(c)] for c in s)
        assert enc == r, f"{p['id']}: verify fail {a}{sym}{b}={r} got {enc}"
        lines.append(f"  {a}{sym}{b}: {ia} {ARITH_SYM[op]} {ib} = {v} -> {enc} (matches {r})")
    iqa, iqb = dec(qa), dec(qb)
    v = f(iqa, iqb)
    s = str(v)
    if rrv:
        s = s[::-1]
    gold = "".join(d2s[int(c)] for c in s)
    assert gold == p["answer"], f"{p['id']}: derived {gold} != gold {p['answer']}"
    lines += [
        "",
        f"Apply to {qa}{sym}{qb}: {iqa} {ARITH_SYM[op]} {iqb} = {v} -> {gold}.",
        "",
        "I will now return the answer in \\boxed{}",
        f"The answer in \\boxed{{–}} is \\boxed{{{gold}}}",
    ]
    return "\n".join(lines)


def main():
    problems = [json.loads(l) for l in open(OUT_DIR / "problems_all.jsonl")]
    n = 0
    with open(OUT_DIR / "traces_all.jsonl", "w") as f:
        for p in problems:
            t = make_pos_trace(p) if p["regime"] == "pos" else make_arith_trace(p)
            assert t.endswith(f"\\boxed{{{p['answer']}}}"), p["id"]
            assert "</think>" not in t and "<|im_end|>" not in t, p["id"]
            f.write(json.dumps({"id": p["id"], "reasoning": t}) + "\n")
            n += 1
    print(f"Wrote {n} traces (token distribution enforced in tokenize_corpus.py)")


if __name__ == "__main__":
    main()
