"""Tokenize equation_v2 into the exact corpus segment format, enforcing conciseness caps.

Identical mechanics to bit_v2: masked prompt (chat template) + trained completion
`{reasoning}\n</think>\n\boxed{answer}<|im_end|>`, interleaved segments, byte-exact
round-trip with two independent decoders per entry.

HARD conciseness enforcement on COMPLETION (trained) tokens:
  every trace <= 4500; distribution must satisfy median < 3000 and p90 < 3000
  (asserted at the end; the build FAILS otherwise).

Output: OUT_DIR/corpus_all/<id>/synthetic.jsonl + OUT_DIR/index_all.jsonl
Usage:  python tokenize_corpus.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys

from common import (
    CHAT_POST,
    CHAT_PRE,
    OUT_DIR,
    PROMPT_SUFFIX,
    TOKEN_LIMIT,
    TRACE_MAX_TOKENS,
    TRACE_MEDIAN_TARGET,
    TRACE_P90_TARGET,
    build_segments,
    decode,
    encode,
    vocab_decode,
)

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")

CATEGORY = "equation_numeric_deduce"


def main() -> None:
    problems = {p["id"]: p for p in map(json.loads, open(OUT_DIR / "problems_all.jsonl"))}
    traces = {t["id"]: t["reasoning"] for t in map(json.loads, open(OUT_DIR / "traces_all.jsonl"))}
    assert set(problems) == set(traces)

    corpus_dir = OUT_DIR / "corpus_all"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True)

    index, comp_lengths = [], []
    for pid, p in problems.items():
        prompt_text = CHAT_PRE + p["prompt"] + PROMPT_SUFFIX + CHAT_POST
        completion_text = f"{traces[pid]}\n</think>\n\\boxed{{{p['answer']}}}<|im_end|>"
        prompt_ids = encode(prompt_text)
        completion_ids = encode(completion_text)

        assert decode(prompt_ids) == prompt_text, f"{pid}: prompt round-trip"
        assert decode(completion_ids) == completion_text, f"{pid}: completion round-trip"
        assert vocab_decode(prompt_ids) == prompt_text, f"{pid}: vocab decoder prompt"
        assert vocab_decode(completion_ids) == completion_text, f"{pid}: vocab decoder completion"

        boxed = [b for b in re.findall(r"\\boxed\{([^}]*)\}", completion_text) if b.strip()]
        assert boxed[-1] == p["answer"], f"{pid}: boxed {boxed[-1]!r} != answer {p['answer']!r}"

        # HARD conciseness cap per trace
        assert len(completion_ids) <= TRACE_MAX_TOKENS, (
            f"{pid}: completion {len(completion_ids)} tokens > {TRACE_MAX_TOKENS} -- "
            f"trace too long, conciseness is a hard constraint"
        )
        tokens = prompt_ids + completion_ids
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        assert len(tokens) < TOKEN_LIMIT, pid

        segments = build_segments(tokens, mask)
        assert [s["type"] for s in segments] == ["masked", "unmasked"], pid
        assert segments[1]["pos"] == len(prompt_ids), pid
        assert decode(segments[0]["tokens"]).endswith("<think>\n"), pid

        pdir = corpus_dir / pid
        pdir.mkdir()
        with open(pdir / "synthetic.jsonl", "w") as f:
            for seg in segments:
                f.write(json.dumps(seg) + "\n")

        comp_lengths.append(len(completion_ids))
        index.append(
            {
                "problem_id": pid,
                "segment": "synthetic.jsonl",
                "category": CATEGORY,
                "masked_token_count": len(prompt_ids),
                "unmasked_token_count": len(completion_ids),
                "token_count": len(tokens),
                "answer": p["answer"],
                "included": True,
                "variant_cell": p["cell"],
                "heldout": p["heldout"],
                "target_op": p["target_op"],
                "rep_op": p["rep_rule"]["op"],
                "op_symbol": p["op_symbol"],
                "n_examples": p["n_examples"],
                "n_group": p["n_group"],
            }
        )

    index.sort(key=lambda e: e["problem_id"])
    with open(OUT_DIR / "index_all.jsonl", "w") as f:
        for e in index:
            f.write(json.dumps(e) + "\n")

    comp_lengths.sort()
    median = comp_lengths[len(comp_lengths) // 2]
    p90 = comp_lengths[int(len(comp_lengths) * 0.9)]
    mx = comp_lengths[-1]
    print(f"Tokenized {len(index)} entries -> {corpus_dir}")
    print(f"COMPLETION tokens: min={comp_lengths[0]} median={median} p90={p90} max={mx}")
    print("(original corpus equation traces: median ~5,800, max ~6,700 -- the thing we fix)")
    assert median < TRACE_MEDIAN_TARGET, f"median {median} >= {TRACE_MEDIAN_TARGET}"
    assert p90 < TRACE_P90_TARGET, f"p90 {p90} >= {TRACE_P90_TARGET}"
    assert mx < TRACE_MAX_TOKENS, f"max {mx} >= {TRACE_MAX_TOKENS}"
    print(f"Conciseness caps SATISFIED: median<{TRACE_MEDIAN_TARGET}, "
          f"p90<{TRACE_P90_TARGET}, max<{TRACE_MAX_TOKENS}")
    print("All entries passed byte-exact round-trip + boxed-format checks.")


if __name__ == "__main__":
    main()
