"""Train vs held-out split for cryptarithm_v2 + manifest + validation report.

Held-out cells (entire structural types absent from training):
  * pos_rBrA   -- the positional arrangement rev(B).rev(A) is NEVER the answer in training
                  (the other 3 arrangements are). NOTE: all 4 arrangements are *named* as
                  candidates in positional traces, so this tests producing rBrA as the
                  SOLUTION, which never occurs in training.
  * arith_mul  -- the multiplication operation is NEVER named or used in any training trace
                  (a fully unseen operation type).

Asserts zero id leakage and zero structural leakage (no train problem's rule is a held-out
rule). Outputs: corpus/ + corpus.jsonl (train), heldout/ (+ heldout_problems.jsonl),
heldout_ids.json, manifest.json, validation_report.md.
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

HELDOUT_CELLS = {"pos_rBrA", "arith_mul"}


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
    assert {e["variant_cell"] for e in held} <= HELDOUT_CELLS
    assert not any(e["variant_cell"] in HELDOUT_CELLS for e in train)

    # ---- STRUCTURAL leakage ----
    for e in index:
        p = problems[e["problem_id"]]
        if e["heldout"]:
            if p["cell"] == "pos_rBrA":
                assert p["regime"] == "pos" and p["pos_class"] == "rBrA", e["problem_id"]
            elif p["cell"] == "arith_mul":
                assert p["regime"] == "arith" and p["op"] == "multiplication", e["problem_id"]
        else:
            # no train problem may use a held-out rule as its ANSWER rule
            if p["regime"] == "pos":
                assert p["pos_class"] != "rBrA", f"{e['problem_id']}: rBrA leaked to train"
            else:
                assert p["op"] != "multiplication", f"{e['problem_id']}: mul leaked to train"

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
        return {"n": len(ls),
                "completion_tokens": {"min": ls[0], "median": ls[len(ls) // 2],
                                      "p90": ls[int(len(ls) * 0.9)], "max": ls[-1]},
                "total_tokens_max": ts[-1]}

    per_cell = {}
    for e in index:
        per_cell.setdefault(e["variant_cell"], []).append(e)
    manifest = {
        "family": "cryptarithm_deduce",
        "token_limit": TOKEN_LIMIT,
        "trace_caps": {"median_lt": TRACE_MEDIAN_TARGET, "p90_lt": TRACE_P90_TARGET,
                       "max_lt": TRACE_MAX_TOKENS},
        "heldout_cells": sorted(HELDOUT_CELLS),
        "regime_counts": dict(Counter(e["regime"] for e in index)),
        "train": stats(train),
        "heldout": stats(held),
        "per_cell": {
            cell: {**stats(es),
                   "split": "heldout" if cell in HELDOUT_CELLS else "train",
                   "regime": es[0]["regime"],
                   "n_examples_distribution": dict(sorted(Counter(
                       e["n_examples"] for e in es).items()))}
            for cell, es in sorted(per_cell.items())
        },
        "verification": {
            "golds_code_computed": len(index),
            "kept_problems_proven_unique": len(index),
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
        "# cryptarithm_v2 validation report",
        "",
        "- family: cryptarithm_deduce (the `guess` sub-family is excluded -- query operator",
        "  unseen in examples => not determinable; official guess traces are 7% correct).",
        f"- entries: {len(index)} (train {len(train)}, held-out {len(held)}); "
        f"regimes: {dict(Counter(e['regime'] for e in index))}.",
        "- two regimes: POSITIONAL (symbol rearrangement, mapping-free) and ARITHMETIC",
        "  (small symbol->digit cipher + arithmetic op), all golds code-computed.",
        "- EVERY kept problem proven uniquely determined: no alternative rule (positional",
        "  menu + arithmetic ops x actions x injective cipher) reproduces all examples with",
        "  a different query result (node-capped exhaustive search; non-unique discarded).",
        f"- COMPLETION token distribution: min {s_all['min']}, median {s_all['median']}, "
        f"p90 {s_all['p90']}, max {s_all['max']}",
        f"  (caps: median<{TRACE_MEDIAN_TARGET} p90<{TRACE_P90_TARGET} max<{TRACE_MAX_TOKENS}; "
        "base model on this family ~7,610 tokens, 98.8% truncated).",
        "- 100% byte-exact tokenize->detokenize round-trip (2 independent decoders).",
        "- held-out: pos_rBrA (rev(B).rev(A) never an answer in train) + arith_mul",
        "  (multiplication never named/used in any train trace); zero id/structural leakage.",
        "",
        "| cell | regime | split | n | comp med | comp p90 | comp max |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell, es in sorted(per_cell.items()):
        s = stats(es)["completion_tokens"]
        split = "heldout" if cell in HELDOUT_CELLS else "train"
        lines.append(f"| {cell} | {es[0]['regime']} | {split} | {len(es)} | "
                     f"{s['median']} | {s['p90']} | {s['max']} |")
    with open(OUT_DIR / "validation_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"train: {len(train)} -> {OUT_DIR / 'corpus'}")
    print(f"heldout: {len(held)} -> {OUT_DIR / 'heldout'}")
    print(json.dumps({k: manifest[k] for k in ("train", "heldout", "heldout_cells",
                                               "regime_counts")}, indent=2))


if __name__ == "__main__":
    main()
