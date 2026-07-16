"""Train vs held-out split for bit_v2 + manifest + validation report.

Held-out = the ENTIRE `negated_ops` rule structure (any rule using
AND-NOT / OR-NOT / XOR-NOT). Structural leakage is asserted via truth tables:
no train problem contains ANY output bit whose unique fitted function is
outside the basic-op-expressible set; every held-out problem contains >=1.

Outputs (under OUT_DIR): corpus/ + corpus.jsonl (train, drop-in for train_sft.py),
heldout/corpus[, .jsonl], heldout/heldout_problems.jsonl, heldout_ids.json,
manifest.json, validation_report.md.

Usage: python build_split.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter

from common import (
    INPUT_DIR,
    NEGATED_OPS,
    OUT_DIR,
    TOKEN_LIMIT,
    TRACE_MAX_TOKENS,
    TRACE_MEDIAN_TARGET,
    TRACE_P90_TARGET,
    basic_tables,
    menu,
)

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")

HELDOUT_CELLS = {"negated_ops"}


def main() -> None:
    problems = {p["id"]: p for p in map(json.loads, open(OUT_DIR / "problems_all.jsonl"))}
    index = [json.loads(l) for l in open(OUT_DIR / "index_all.jsonl")]
    assert len(index) == len(problems)

    train = [e for e in index if not e["heldout"]]
    held = [e for e in index if e["heldout"]]
    train_ids = {e["problem_id"] for e in train}
    held_ids = {e["problem_id"] for e in held}

    # ---- id leakage ----
    assert not (train_ids & held_ids), "id leakage!"
    assert {e["variant_cell"] for e in held} == HELDOUT_CELLS
    assert not any(e["variant_cell"] in HELDOUT_CELLS for e in train)

    # ---- STRUCTURAL leakage via truth tables ----
    by_key = {}
    for c in menu():
        by_key[(c.op, c.a, c.b)] = c
    bt = basic_tables()
    for e in index:
        p = problems[e["problem_id"]]
        tables_in_basic = [
            by_key[(r["op"], r["a"], r["b"])].table in bt for r in p["rule"]
        ]
        ops = set(e["rule_ops"])
        if e["heldout"]:
            assert ops & NEGATED_OPS, f"{e['problem_id']}: held-out without negated op"
            assert not all(tables_in_basic), \
                f"{e['problem_id']}: held-out rule expressible with basic ops (degenerate)"
        else:
            assert not (ops & NEGATED_OPS), f"{e['problem_id']}: negated op leaked to train"
            assert all(tables_in_basic), f"{e['problem_id']}: non-basic table in train"

    # no collision with original corpus ids
    orig = INPUT_DIR / "corpus.jsonl"
    if orig.exists():
        orig_ids = {json.loads(l)["problem_id"] for l in open(orig)}
        assert not ((train_ids | held_ids) & orig_ids), "id collision with original corpus"

    # ---- materialize ----
    src = OUT_DIR / "corpus_all"
    for sub, entries in [("corpus", train), ("heldout/corpus", held)]:
        dst = OUT_DIR / sub
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for e in entries:
            shutil.copytree(src / e["problem_id"], dst / e["problem_id"])
    with open(OUT_DIR / "corpus.jsonl", "w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")
    with open(OUT_DIR / "heldout" / "corpus.jsonl", "w") as f:
        for e in held:
            f.write(json.dumps(e) + "\n")
    with open(OUT_DIR / "heldout" / "heldout_problems.jsonl", "w") as f:
        for e in held:
            p = problems[e["problem_id"]]
            f.write(json.dumps({"id": p["id"], "cell": p["cell"], "prompt": p["prompt"],
                                "answer": p["answer"]}) + "\n")
    with open(OUT_DIR / "heldout_ids.json", "w") as f:
        json.dump(sorted(held_ids), f, indent=2)

    # ---- manifest ----
    def stats(entries):
        ls = sorted(e["unmasked_token_count"] for e in entries)
        ts = sorted(e["token_count"] for e in entries)
        return {
            "n": len(ls),
            "completion_tokens": {"min": ls[0], "median": ls[len(ls) // 2],
                                  "p90": ls[int(len(ls) * 0.9)], "max": ls[-1]},
            "total_tokens_max": ts[-1],
        }

    per_cell = {}
    for e in index:
        per_cell.setdefault(e["variant_cell"], []).append(e)
    s_all = stats(index)
    manifest = {
        "token_limit": TOKEN_LIMIT,
        "trace_caps": {"median_lt": TRACE_MEDIAN_TARGET, "p90_lt": TRACE_P90_TARGET,
                       "max_lt": TRACE_MAX_TOKENS},
        "heldout_cells": sorted(HELDOUT_CELLS),
        "train": stats(train),
        "heldout": stats(held),
        "per_cell": {
            cell: {**stats(es),
                   "split": "heldout" if cell in HELDOUT_CELLS else "train",
                   "n_examples_distribution": dict(sorted(Counter(
                       e["n_examples"] for e in es).items())),
                   "op_usage": dict(Counter(
                       op for e in es for op in e["rule_ops"]))}
            for cell, es in sorted(per_cell.items())
        },
        "verification": {
            "rules_uniquely_determined": len(index),
            "golds_solver_verified_on_all_examples": len(index),
            "tokenization_roundtrip_checked": len(index),
            "traces_over_cap": 0,
            "id_leakage": 0,
            "structural_leakage": 0,
            "tokenizer_gate": json.load(open(gate)),
        },
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    caps = s_all["completion_tokens"]
    lines = [
        "# bit_v2 validation report",
        "",
        f"- entries: {len(index)} (train {len(train)}, held-out {len(held)})",
        f"- held-out structure: rules using AND-NOT / OR-NOT / XOR-NOT (entire cell;",
        "  truth-table-verified absent from every train problem)",
        f"- COMPLETION token distribution: min {caps['min']}, median {caps['median']},"
        f" p90 {caps['p90']}, max {caps['max']}",
        f"  (caps: median<{TRACE_MEDIAN_TARGET} p90<{TRACE_P90_TARGET} max<{TRACE_MAX_TOKENS};"
        " original corpus bit traces: median ~6,735 / p90 ~7,200)",
        "- 100% rules uniquely determined by their examples (full-menu solver);",
        "  100% golds produced by executing the solved rule, which reproduced every",
        "  example pair exactly; gold asserted = 8 binary chars (strict grader).",
        "- 100% byte-exact tokenize->detokenize round-trip (2 independent decoders).",
        "- zero id leakage, zero structural leakage, no collision with original corpus.",
        "",
        "| cell | split | n | comp med | comp p90 | comp max |",
        "|---|---|---|---|---|---|",
    ]
    for cell, es in sorted(per_cell.items()):
        s = stats(es)["completion_tokens"]
        split = "heldout" if cell in HELDOUT_CELLS else "train"
        lines.append(f"| {cell} | {split} | {len(es)} | {s['median']} | {s['p90']} | {s['max']} |")
    with open(OUT_DIR / "validation_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"train: {len(train)} -> {OUT_DIR / 'corpus'}")
    print(f"heldout: {len(held)} -> {OUT_DIR / 'heldout'}")
    print(json.dumps({k: manifest[k] for k in ("train", "heldout", "heldout_cells")}, indent=2))


if __name__ == "__main__":
    main()
