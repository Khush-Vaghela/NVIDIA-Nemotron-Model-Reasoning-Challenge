# Phase 1 — Corpus Analysis Findings

Analysis of the public 0.85-scoring Nemotron corpus (this folder). Evidence cited as file paths and counts. Several assumptions in the task brief are contradicted by the data; corrections are flagged inline and collected at the end.

## 1. Directory structure and file formats

```
train.csv            9,500 rows: id, prompt, answer (official train set — not 6,355)
problems.jsonl       9,500 rows: {id, category, status, submission}
generation.jsonl     9,500 rows: base-model probing runs (extracted answer, correctness, logprobs)
corpus.jsonl         17,963 rows: index {problem_id, segment, category, masked_token_count,
                     unmasked_token_count, token_count, answer, included}
corpus/<id>/synthetic.jsonl   14,818 dirs — the actual tokenized training data
augmentations/<id>.txt        8,463 files: [category]/[prompt]/[completion] plain text
investigations/<id>.txt       240 files: free-form hypothesis notes
augmenters/                   5 generators (spelling, concatenation, splitting, matching, lstrip)
vocab.jsonl          131,072 rows {token_id, token} — enables offline detokenization
tokenizer.json, corpus.py, reasoning.py, augmentation.py, train_sft.py, train_common.py
```

**`synthetic.jsonl` is NOT one record with `tokens`+`mask` fields.** It is multiple lines, each a *segment*: `{"type": "masked"|"unmasked", "pos": int, "tokens": [ids]}`. The training loader (`train_common.py:load_tokens`) concatenates segments and derives the mask (masked=0, unmasked=1). Loss weight for predicting token i+1 is `mask[i+1]` (`train_common.py:build_datum`).

**Token layout** (`corpus.py`): prompt = chat template (`<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}{suffix}<|im_end|>\n<|im_start|>assistant\n<think>\n`) — all masked. Completion = `{reasoning}\n</think>\n\boxed{answer}<|im_end|>` — all trained. Augmented entries instead end `{completion}\n</think><|im_end|>` with **no `\boxed{}`** and no prompt suffix. Hard truncation at 8,192 tokens (no on-disk entry hits it; max is 7,702).

**Missing from the archive** (gitignored): `reasoning/*.txt` (plain-text traces), `raw/` (parsed problem JSON), `reasoners/` (the per-family trace generators), `problems/<id>.jsonl` details. Traces are recoverable only by detokenizing `corpus/` (decoder built at `analysis/decode.py` using `vocab.jsonl`). **The trace-generation source code is not reproducible from this archive** — we must write our own generators.

**Index/disk mismatch:** `corpus.jsonl` lists 17,963 entries (all 9,500 train + 8,463 augmented, all `included: true`), but only 14,818 have token files on disk: 6,355 train-derived + 8,463 augmented. The 3,145 index-only entries (numeral 967, gravity 650, unit_conversion 629, cryptarithm_deduce 605, bit_manipulation 238, equation_numeric_deduce 56) have no data. The index is from a later pipeline run than the corpus snapshot. The effective training set is the 14,818 on disk — matching the "~14,800" figure.

## 2. Family taxonomy and counts

Labels live in `problems.jsonl` (train) and `corpus.jsonl` (all). Nine train categories = six families, with cryptarithm and equation_numeric split by solve strategy (`_deduce` = rule solved, `_guess` = heuristic fallback):

| category | train.csv | on-disk corpus | status: found / hyp / unknown (train) |
|---|---|---|---|
| bit_manipulation | 1,602 | 1,364 | 1364 / 121 / 117 |
| cipher | 1,576 | 1,576 | 1576 / 0 / 0 |
| gravity | 1,597 | 947 | 1597 / 0 / 0 |
| unit_conversion | 1,594 | 965 | 1594 / 0 / 0 |
| numeral | 1,576 | 609 | 1576 / 0 / 0 |
| equation_numeric_deduce | 596 | 540 | 540 / 22 / 34 |
| equation_numeric_guess | 136 | 136 | 21 / 35 / 80 |
| cryptarithm_deduce | 659 | 54 | 54 / 46 / 559 |
| cryptarithm_guess | 164 | 164 | 11 / 25 / 128 |

Plus 5 **augmented skill-drill categories** (8,463 entries): matching 4,515, concatenation 1,500, splitting 1,500, spelling 648, lstrip 300.

Why gravity/unit/numeral have fewer on-disk entries than rule_found count is not explained anywhere in the repo (947/1,597, 965/1,594, 609/1,576) — the subset criterion is unknown. Flagging rather than guessing.

## 3. Variant diversity per family (computed from train.csv)

- **cipher** (1,576): all are general random monoalphabetic substitutions — **1,576 distinct alphabets, zero shift/affine/atbash ciphers**. Word order always preserved; answers lowercase English from a small Wonderland vocab (~80 words; traces enumerate a dictionary of them).
- **bit_manipulation** (1,602): **100% 8-bit → 8-bit**, 7–10 example pairs each. Rules are per-output-bit functions from {Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT} over 1–2 input bits (this is exactly the hypothesis menu the traces enumerate). ~1,580 distinct rule signatures — virtually every problem unique; a crude refit reproduces most bits but 907 problems have ≥1 bit outside my simplified search (their matcher is wider). **No other bit-widths exist.**
- **unit_conversion** (1,594): every problem is `y = k·x`, unit literal is always `"m"`, ~980 distinct k (range ≈0.1–2.2). **One structural variant; zero offsets, zero unit-name diversity.**
- **gravity** (1,597): every problem fits `d = ½gt²`, ~986 distinct g ∈ [4.91, 19.58]. One structural variant.
- **numeral** (1,576): **100% Arabic→Roman, query range 1–100.** One direction, one target system, tiny range. The single thinnest family.
- **equation_numeric** (732): visible digits, symbolic operators (341 distinct operator-symbol alphabets); hidden op drawn from ~30 candidates (concat, add, abs-diff, sub, mul, div, mod, digit-wise ops, cross-multiply, determinant…) composed with actions {identity, reversed operands, reversed result}.
- **cryptarithm** (823): fully symbol-encoded equations; each problem a unique symbol→digit alphabet (659+164 distinct, 11–12 symbols from a ~28-symbol pool); same hidden-op space.

**Takeaway:** parameter diversity (k, g, alphabets) is saturated; *structural* variant diversity is ~1 per family for gravity/unit/numeral/bit-width. That's the generalization gap the brief is after.

## 4. Reasoning trace style

All traces are **deterministic, template-generated procedural derivations** (produced by the missing `reasoners/` code) — mechanical and exhaustive rather than discursive. They DO show induction (hypothesis enumeration → elimination → apply winner), but in fixed scaffolds:

- **cipher** (`corpus/00189f6a`, 3,495 trained tokens): split words into chars (en-dash format matching the spelling drill), derive char→char mapping pair-by-pair across all examples, apply to query; unknown chars resolved by dictionary length/pattern matching ("Best match: 【book】").
- **bit_manipulation** (`corpus/00066667`, 6,673 tokens): transpose outputs into bit columns, test each output bit against the full op menu ("Identity absent, NOT NOT3, Constant absent, AND absent…"), select, apply. Median 6,735 trained tokens, p90 7,200 — **right at the inference budget edge** (generation.jsonl probing capped at 7,680 gen tokens). This is the truncation-risk family.
- **gravity** (`corpus/0040ff76`, 4,780 tokens): compute k per example via **digit-by-digit long division spelled out line-by-line** (~25 lines per division), take median k, long-multiply for query. ~90% of tokens are manual arithmetic, not induction.
- **unit_conversion** (2,864 tokens): identical long-division scaffold for the factor.
- **numeral** (74 tokens): one-liner — "This is an Arabic to Roman numeral conversion. Converting 100: 100 >= 100 -> C…". No induction at all (rule asserted, not inferred).
- **equation_numeric** (5,585 tokens): brute-force enumeration of ~30 ops × 4 action combos against examples, with all arithmetic written out; apply matching combo.
- **cryptarithm** (738 tokens): per-example operator/operand decomposition, concat/reverse-concat checks, op table, apply.

Style is uniform: opens "We need to…", ends `I will now return the answer in \boxed{}` → `\boxed{ans}` → `</think>` → `\boxed{ans}`.

## 5. Augmented entries: skill drills, NOT rule variants

The 8,463 non-train entries are **not new rule-puzzle instances**. They are low-level token-skill drills (100 rows each) teaching the tokenizer-hostile subskills the traces depend on:

- *spelling*: spell tokenizer words char-by-char in the en-dash format (`–c–a–t–`)
- *concatenation / splitting*: merge/split bracketed symbol groups (`【]】【}】` ↔ `【]}】`)
- *lstrip*: strip leading space inside 【】 brackets
- *matching*: bit-column matching sections lifted from bit_manipulation traces (find which op column matches an output column)

So the winning recipe = 6,355 procedural traces + 8,463 drills for the primitives those traces use. There is **no in-family variant augmentation at all** — lever #1 from the goal is genuinely unexploited.

## 6. Masking scheme

Prompt (incl. chat template and `<think>\n` opener): masked (weight 0). Everything the model must emit (reasoning, `</think>`, final `\boxed{}`, `<|im_end|>`): trained (weight 1). One masked→unmasked transition per entry; weights shifted by one position at loss time. Augmented drills place the completion *inside* the think block and train no `\boxed{}`.

## 7. Status tagging and wrong-answer training data

`problems.jsonl` statuses: rule_found 8,333 / hypothesis_formed 249 / rule_unknown 918. `investigations/` holds 240 hypothesis notes. All cipher/gravity/unit/numeral were solved; unsolved mass is cryptarithm (687 unknown+hyp) and bit_manipulation (238).

**267 of the 6,355 on-disk train-derived entries train on answers that do NOT match train.csv** (verified by detokenizing final `\boxed{}` and comparing with the grader's rules). These are the `_guess` categories' fallback traces (e.g. `01ef1e3e`: trains `\boxed{''`!}`, true answer `[](`). Deliberate "graceful fallback" teaching or accepted noise — either way, ~4.2% of family traces are answer-incorrect and we should decide whether to keep, fix, or drop them.

## 8. Grader quirks confirmed (`reasoning.py:compare_answer`)

Binary strings: strict exact match (leading zeros matter). Numerics: `rel_tol=1e-2, abs_tol=1e-5`. Everything else: case-insensitive exact string match. Answer extraction: last non-empty `\boxed{}`.

## Corrections to the task brief

1. `synthetic.jsonl` = interleaved masked/unmasked **segments**, not `tokens`+`mask` fields (loader reconstructs the mask).
2. Official train = **9,500** problems; 6,355 (not "~6,355 of 14,800 overlapping") have token files; the other 8,463 are skill drills, not augmented puzzle instances.
3. The ~8,400 augmented entries are **not** "same rules with new numbers" — they're a separate drill curriculum with 5 of their own categories.
4. Plain-text reasoning, raw parsed problems, and the trace-generator source are **absent** from the archive; traces must be recovered via detokenization (done) and generators rewritten from scratch (Phase 2).
5. `corpus.jsonl` index disagrees with `corpus/` on disk by 3,145 entries; trust the disk.

## Implications for Phase 2 (preview, not yet built)

- Variant robustness has maximal headroom in numeral (1 variant!), unit_conversion (scalar-only, "m"-only), gravity (one formula), bit_manipulation (8-bit only); cipher needs *structured* variants (shift/affine/keyword) plus the existing random ones.
- Traces should keep the proven procedural scaffold but (a) compress the arithmetic long-division blocks (the model surely doesn't need 25 lines per division), (b) add an explicit hypothesis-test-eliminate framing uniformly (numeral currently has none), (c) keep bit_manipulation well under ~7k tokens.
- The drill curriculum (spelling/splitting/matching) should be retained — it's likely a big part of the 0.85 — and extended to any new symbol sets we introduce.
- Held-out unseen-variant eval is straightforward: hold out entire (family, structural-variant) cells, e.g. specific bases, bit-widths, cipher types, unit systems.
