"""MANDATORY GATE for bit_v2: prove tokenizer fidelity before any generation.

Same 4 checks as numeral_v2, referenced against 3 EXISTING bit_manipulation
entries from the attached corpus (00066667, 0031df9c, 004ef7c7):
  1. round-trip (two independent decoders)
  2. byte-exact id match: rebuilt prompt ids == reference masked ids AND
     re-encoded completion text == reference unmasked ids
  3. special-token boundary ids match; <think>\n masked, </think>/\boxed trained
  4. masked->unmasked transition at end of `<think>\n`

Writes OUT_DIR/tokenizer_gate.json; generation scripts refuse to run without it.
Usage: python verify_tokenizer.py
"""

from __future__ import annotations

import csv
import json
import sys

from common import (
    CHAT_POST,
    CHAT_PRE,
    OUT_DIR,
    PROMPT_SUFFIX,
    REF_BIT_IDS,
    REF_CORPUS_DIR,
    TRAIN_CSV,
    decode,
    encode,
    vocab_decode,
)

csv.field_size_limit(10**9)
results: dict[str, bool] = {}


def report(name: str, ok: bool, detail: str = "") -> None:
    results[name] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# ---------- Check 1: round-trip ----------
samples = [
    "Here are some examples of input -> output:\n01010001 -> 11011101\n",
    "out0 = OR(in1, NOT(in7)): 0,0->1 1,0->1 all match",
    "The answer in \\boxed{–} is \\boxed{10010111}\n</think>\n\\boxed{10010111}<|im_end|>",
    CHAT_PRE + "x" + PROMPT_SUFFIX + CHAT_POST,
    "Query 00110100: in0=0 in1=0 in2=1 in3=1 in4=0 in5=1 in6=0 in7=0",
]
rt_ok = True
for i, s in enumerate(samples):
    ids = encode(s)
    if decode(ids) != s or vocab_decode(ids) != s:
        rt_ok = False
        print(f"  sample {i} round-trip mismatch")
report("1. round-trip (2 independent decoders)", rt_ok, f"{len(samples)} samples")

# ---------- Check 2: reference-match on existing bit entries ----------
prompts = {r["id"]: r["prompt"] for r in csv.DictReader(open(TRAIN_CSV, newline=""))}
ref_ok = True
boundary_info = []
for pid in REF_BIT_IDS:
    segs = [json.loads(l) for l in open(REF_CORPUS_DIR / pid / "synthetic.jsonl")]
    assert [s["type"] for s in segs] == ["masked", "unmasked"], pid
    masked_ids, train_ids = segs[0]["tokens"], segs[1]["tokens"]
    masked_text, train_text = decode(masked_ids), decode(train_ids)
    my_prompt_ids = encode(CHAT_PRE + prompts[pid] + PROMPT_SUFFIX + CHAT_POST)
    my_train_ids = encode(train_text)
    if my_prompt_ids != masked_ids or my_train_ids != train_ids:
        ref_ok = False
        print(f"  {pid}: prompt match={my_prompt_ids == masked_ids} "
              f"completion match={my_train_ids == train_ids}")
    boundary_info.append((pid, masked_ids, train_ids, masked_text, train_text))
report("2. reference-match (ids == original, prompt+completion)", ref_ok,
       f"{len(REF_BIT_IDS)} bit_manipulation entries")

# ---------- Check 3: special-token layout ----------
pre_ids, post_ids, end_ids = encode(CHAT_PRE), encode(CHAT_POST), encode("<|im_end|>")
print(f"  CHAT_PRE ids  {pre_ids}")
print(f"  CHAT_POST ids {post_ids}")
layout_ok = True
for pid, masked_ids, train_ids, masked_text, train_text in boundary_info:
    checks = [
        masked_ids[: len(pre_ids)] == pre_ids,
        masked_ids[-len(post_ids):] == post_ids,
        masked_text.endswith("<think>\n"),
        "</think>" in train_text,
        train_text.endswith("<|im_end|>"),
        train_ids[-len(end_ids):] == end_ids,
        "\\boxed{" in train_text.split("</think>")[-1],
    ]
    if not all(checks):
        layout_ok = False
        print(f"  {pid}: {checks}")
    else:
        print(f"  {pid}: template ids + '<think>\\n' masked + '</think>'/boxed trained: OK")
report("3. special-token layout matches reference", layout_ok)

# ---------- Check 4: mask boundary ----------
mask_ok = True
for pid, masked_ids, train_ids, masked_text, train_text in boundary_info:
    my_prompt_ids = encode(CHAT_PRE + prompts[pid] + PROMPT_SUFFIX + CHAT_POST)
    if len(my_prompt_ids) != len(masked_ids) or not masked_text.endswith("<think>\n"):
        mask_ok = False
        print(f"  {pid}: transition {len(my_prompt_ids)} vs ref {len(masked_ids)}")
    else:
        print(f"  {pid}: transition at token {len(masked_ids)} (end of '<think>\\n') OK")
report("4. mask boundary position matches reference", mask_ok)

passed = all(results.values())
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "tokenizer_gate.json", "w") as f:
    json.dump({"passed": passed, "checks": results}, f, indent=2)
print("\nGATE " + ("PASSED" if passed else "FAILED"))
if not passed:
    sys.exit(1)
