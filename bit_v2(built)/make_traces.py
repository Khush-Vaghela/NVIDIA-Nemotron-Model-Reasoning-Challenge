"""Concise, induction-explicit traces for bit_v2 — conciseness is the objective.

Structure per trace:
  (a) approach in 2 lines (per-bit fit)
  (b) re-list example pairs once (indexed; the model needs them transcribed)
  (c) ONE compact line per output bit naming the fitted rule, verified against
      ALL examples ("out0 = OR(in1, NOT(in7)): 0,0->1 1,0->1 ... all match") —
      NO menu enumeration (that is what bloated the original to ~6,700 tokens)
  (d) apply the 8-bit rule to the query, one line per bit
  (e) box, terminating exactly like the original corpus.

Hard caps enforced downstream on COMPLETION tokens: median<3500, p90<4000, max<4500.
Internal consistency assert: the trace's derived output == the solver-verified gold.

Output: OUT_DIR/traces_all.jsonl {id, reasoning}
Usage:  python make_traces.py
"""

from __future__ import annotations

import json
import sys

from common import OUT_DIR, str_to_bits

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")


def _eval(op: str, a, b, bits) -> int:
    if op == "C0":
        return 0
    if op == "C1":
        return 1
    if op == "ID":
        return bits[a]
    if op == "NOT":
        return 1 - bits[a]
    x, y = bits[a], bits[b]
    return {"AND": x & y, "OR": x | y, "XOR": x ^ y,
            "ANDN": x & (1 - y), "ORN": x | (1 - y), "XORN": x ^ (1 - y)}[op]


def fit_line(j: int, r: dict, examples: list) -> str:
    op, a, b, name = r["op"], r["a"], r["b"], r["name"]
    if op in ("C0", "C1"):
        vals = " ".join(o[j] for _, o in examples)
        return f"out{j} = {name}: outputs {vals} - all match"
    if op in ("ID", "NOT"):
        ev = " ".join(f"{i[a]}->{o[j]}" for i, o in examples)
        return f"out{j} = {name}: {ev} - all match"
    ev = " ".join(f"{i[a]},{i[b]}->{o[j]}" for i, o in examples)
    return f"out{j} = {name}: {ev} - all match"


def apply_line(j: int, r: dict, bits: list[int]) -> tuple[str, int]:
    op, a, b, name = r["op"], r["a"], r["b"], r["name"]
    v = _eval(op, a, b, bits)
    if op in ("C0", "C1"):
        return f"out{j} = {name} = {v}", v
    if op == "ID":
        return f"out{j} = in{a} = {v}", v
    if op == "NOT":
        return f"out{j} = NOT(in{a}={bits[a]}) = {v}", v
    if op in ("AND", "OR", "XOR"):
        return f"out{j} = {op}(in{a}={bits[a]}, in{b}={bits[b]}) = {v}", v
    base = {"ANDN": "AND", "ORN": "OR", "XORN": "XOR"}[op]
    return f"out{j} = {base}(in{a}={bits[a]}, NOT(in{b}={bits[b]})) = {v}", v


def make_trace(p: dict) -> str:
    examples = p["examples"]
    lines = [
        "We need to deduce the transformation rule. I will fit each output bit as "
        "a function of the input bits using the examples, then apply the fitted "
        "rule to the query. Bits are indexed 0-7 from the left.",
        "I will put my final answer inside \\boxed{}.",
        "",
        "Example pairs:",
    ]
    for k, (i, o) in enumerate(examples):
        lines.append(f"{k}: {i} -> {o}")
    lines += ["", "Fitting each output bit against all examples:"]
    for j, r in enumerate(p["rule"]):
        # verify the cited evidence is true (no unverified claims in traces)
        for i, o in examples:
            assert _eval(r["op"], r["a"], r["b"], str_to_bits(i)) == int(o[j]), \
                f"{p['id']}: evidence false for out{j}"
        lines.append(fit_line(j, r, examples))
    lines += ["", "All 8 output bits fitted. Applying the rule to the query."]
    qbits = str_to_bits(p["query"])
    lines.append(f"Query {p['query']}: " + " ".join(f"in{i}={b}" for i, b in enumerate(qbits)))
    out = []
    for j, r in enumerate(p["rule"]):
        line, v = apply_line(j, r, qbits)
        lines.append(line)
        out.append(str(v))
    derived = "".join(out)
    assert derived == p["answer"], f"{p['id']}: derived {derived} != gold {p['answer']}"
    lines += [
        f"Output bits: {' '.join(out)} -> {derived}",
        "",
        "I will now return the answer in \\boxed{}",
        f"The answer in \\boxed{{–}} is \\boxed{{{derived}}}",
    ]
    return "\n".join(lines)


def main() -> None:
    problems = [json.loads(l) for l in open(OUT_DIR / "problems_all.jsonl")]
    n = 0
    with open(OUT_DIR / "traces_all.jsonl", "w") as f:
        for p in problems:
            t = make_trace(p)
            assert t.endswith(f"\\boxed{{{p['answer']}}}")
            f.write(json.dumps({"id": p["id"], "reasoning": t}) + "\n")
            n += 1
    print(f"Wrote {n} traces (token distribution measured/enforced in tokenize_corpus.py)")


if __name__ == "__main__":
    main()
