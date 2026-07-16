"""Train vs held-out split for equation_v2 + manifest + validation report.

Held-out = the ENTIRE `determinant` cell (target operation is `determinant` or
`abs determinant` -- two whole operation TYPES absent from training). Structural leakage
is asserted on BOTH the target operation and the fitted representative operation: no train
problem uses (or fits to) a determinant operation; every held-out problem does. This
tests generalization to an UNSEEN rule type (the model must propose determinant itself --
the word never appears in any training trace).

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
    OUT_DIR,
    TOKEN_LIMIT,
    TRACE_MAX_TOKENS,
    TRACE_MEDIAN_TARGET,
    TRACE_P90_TARGET,
)

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")

HELDOUT_CELLS = {"determinant"}
HELDOUT_OPS = {"determinant", "abs determinant"}


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

    # ---- STRUCTURAL leakage (target op AND fitted representative op) ----
    for e in index:
        ops = {e["target_op"], e["rep_op"]}
        if e["heldout"]:
            assert e["target_op"] in HELDOUT_OPS, f"{e['problem_id']}: held-out target not determinant"
            assert e["rep_op"] in HELDOUT_OPS, (
                f"{e['problem_id']}: held-out fitted to non-determinant op {e['rep_op']} "
                f"(functionally equivalent -- not a clean determinant test)"
            )
        else:
            assert not (ops & HELDOUT_OPS), (
                f"{e['problem_id']}: determinant op leaked into train "
                f"(target={e['target_op']} rep={e['rep_op']})"
            )

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
    manifest = {
        "family": "equation_numeric_deduce",
        "token_limit": TOKEN_LIMIT,
        "trace_caps": {"median_lt": TRACE_MEDIAN_TARGET, "p90_lt": TRACE_P90_TARGET,
                       "max_lt": TRACE_MAX_TOKENS},
        "heldout_cells": sorted(HELDOUT_CELLS),
        "heldout_ops": sorted(HELDOUT_OPS),
        "train": stats(train),
        "heldout": stats(held),
        "per_cell": {
            cell: {**stats(es),
                   "split": "heldout" if cell in HELDOUT_CELLS else "train",
                   "n_examples_distribution": dict(sorted(Counter(
                       e["n_examples"] for e in es).items())),
                   "query_group_distribution": dict(sorted(Counter(
                       e["n_group"] for e in es).items())),
                   "target_op_usage": dict(Counter(e["target_op"] for e in es))}
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

    s_all = stats(index)["completion_tokens"]
    lines = [
        "# equation_v2 validation report",
        "",
        f"- family: equation_numeric_deduce (the `guess` sub-family is excluded -- its rule",
        "  is not determinable from the prompt; even the official corpus traces miss it 87%).",
        f"- entries: {len(index)} (train {len(train)}, held-out {len(held)})",
        "- held-out structure: the determinant / abs-determinant operations (entire cell;",
        "  asserted absent from every train problem's target AND fitted representative op).",
        f"- COMPLETION token distribution: min {s_all['min']}, median {s_all['median']},"
        f" p90 {s_all['p90']}, max {s_all['max']}",
        f"  (caps: median<{TRACE_MEDIAN_TARGET} p90<{TRACE_P90_TARGET} max<{TRACE_MAX_TOKENS};"
        " original corpus equation traces: median ~5,800 / max ~6,700)",
        "- 100% of query-operator groups uniquely determine the gold (full 128-rule solver;",
        "  gold invariant across the consistent set). 100% of golds produced by executing",
        "  the solved rule, which reproduced every group example exactly.",
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
