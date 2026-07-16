# equation_v2 — concise equation_numeric training corpus

Kills the truncation failure mode for the `equation_numeric` family: a base-model probe
showed it scoring **18.8% with 83.8% truncation** — the model rambles past the 7,680-token
budget and never boxes. The original corpus's equation traces enumerate all 32 operations ×
4 action combos with full arithmetic, running **~5,800 tokens (max ~6,700)**. equation_v2
traces reach the same verified answers in a fraction of the tokens:

**completion-token distribution: median 435, p90 501, max 780** (hard caps asserted:
median < 3,000, p90 < 3,000, max < 4,500 — the build fails otherwise).

Format is a byte-exact drop-in for the existing `train_sft.py` (reuses the validated bit_v2
pipeline structure: tokenizer gate, segment format, held-out split, manifest discipline).

## Run order (Kaggle, CPU-only generation, no downloads)

```bash
export NEMOTRON_INPUT=/kaggle/input/<nemotron-corpus-dataset>  # tokenizer.json, vocab.jsonl, corpus/, train.csv
export NEMOTRON_OUT=/kaggle/working/equation_v2_out

python verify_tokenizer.py    # MANDATORY GATE — 4x PASS or nothing else runs
python solve_rules.py         # optional: solver self-test (also prints reproduction rate)
python generate_problems.py   # rules + uniqueness + verified golds (~10s CPU)
python make_traces.py         # concise induction traces
python tokenize_corpus.py     # exact segment format + conciseness caps enforced
python build_split.py         # train/held-out + manifest + validation report
```

A completed, verified local run ships in `output/` (small artifacts + a corpus sample +
`corpus_full.tar.gz` containing all 3,650 tokenized entries).

## The family, and why only `deduce`

Each problem shows a few `AA<op>BB = R` lines (2-digit operands, leading zeros kept) and
asks for one query `AA<op>BB`. The operator is a **symbol** drawn from a ~25-symbol
alphabet, and in the real data **each operator symbol carries its own hidden rule**. There
are two sub-families in `train.csv`:

- **`equation_numeric_deduce`** (596 real): the query operator **also appears** in the
  examples, so its rule is deducible from the examples sharing it. The official corpus
  traces box the correct answer **540/540** of the present entries → reliably solvable.
- **`equation_numeric_guess`** (136 real): the query operator is **never** in the examples,
  so its rule is **not determinable** from the prompt. Even the official corpus traces box
  the wrong answer **118/136 (87%)** — they literally fall back to guessing "absolute
  difference". Per the discipline ("never guess; uniquely determined"), **equation_v2 builds
  the `deduce` family only**. The `guess` family is documented here and excluded.

## Rule space (reverse-engineered and verified against train.csv)

A rule is `(operation, operand_reverse, result_reverse)`:

- **operand_reverse** — reverse each operand's 2-char string (`06 -> 60`) before computing.
- **operation** — one of **32** candidates: `concatenation`, `reverse concatenation`,
  `addition`, `absolute difference`, `negated absolute difference`, `subtraction`,
  `reverse subtraction`, `multiplication`, `multiply±1`, `add±1`, `sub±1`, `max mod min`,
  `integer division`, `modulo`, `reverse division`, `reverse modulo`,
  `digit absolute diff`, `digit add mod10`, `digit sub mod10`, `cross multiply`,
  `cross multiply rev`, `digit multiply`, `digit multiply rev`, `digit sum diff/sum`,
  `digit product diff/sum`, `determinant`, `abs determinant`.
- **result_reverse** — reverse the result STRING (the sign moves to the end).
- **negative encoding** — if the final result is negative, the `-` is dropped and the
  **operator symbol** takes its place: at the front if the result was not reversed
  (`(17`), at the end if it was (`03!`). Verified against real golds.

`32 ops × 4 action combos = 128 candidate rules`, which collapse to **104 distinct
functional-equivalence classes** (e.g. `concatenation` ≡ `reverse concatenation` with
reversed operands AND result, since `rev(rev(b)+rev(a)) = a+b`). `solve_rules.py` precomputes
each rule's behaviour over all 100×100 operand pairs once, so uniqueness checks are O(1).

**Validation of the menu against real data:** for the 596 real `deduce` problems, the menu
contains a rule consistent with the query-operator group in **563/596 (94%)**; applying the
canonical-priority rule reproduces the exact gold in **~90%**. The unreproduced remainder are
(a) the `-` operator (its symbol collides with the negative sign, so its golds are ambiguous)
and (b) single-example query groups whose hidden rule is not recoverable. equation_v2
**excludes the `-` operator and the braces `{` `}`** (the latter would corrupt `\boxed{...}`
extraction of a negative gold), and generates only **uniquely-determined** groups.

## Generation discipline

- **Uniqueness** (`generate_problems.py` + `solve_rules.py`): each problem fixes a target
  rule and a query operator, then grows the query-operator example group — adding a
  disambiguating operand pair (one where the surviving consistent rules disagree) — until the
  consistent set is **functionally collapsed to a single equivalence class**. The gold is then
  invariant to the representative for ANY query (the same "gold-invariant" criterion bit_v2
  uses). Query groups are small: ~2–4 examples.
- **Distractors**: 0–2 extra operator symbols (their own rules, 1–2 examples each) are added
  to match the real multi-operator format; the model must learn to fit only the
  query-operator group. Total examples per problem: 2–9 (real is 3–5; ours skews slightly
  higher because uniqueness needs a larger query group than the real 1–2 — see Flags).
- **Golds**: every gold is produced by executing the solved rule and is asserted to reproduce
  every group example exactly. As a final independent check, all 3,650 problems are re-solved
  **from the prompt text alone**: 3,650/3,650 uniquely determined, 3,650/3,650 gold-matched.

## Cells and held-out split

| cell | operation pool | n | split |
|---|---|---|---|
| arithmetic | add / sub / rev-sub / abs-diff / neg-abs-diff / mul | 900 | train |
| concat | concatenation / reverse concatenation | 300 | train |
| modular | max-mod-min / int-div / mod / rev-div / rev-mod | 500 | train |
| offsets | mul±1 / add±1 / sub±1 | 500 | train |
| digitwise | digit abs-diff / add-mod10 / sub-mod10 / digit-mul(/rev) | 600 | train |
| digit_agg | cross-mul(/rev) / digit sum & product diff/sum | 600 | train |
| determinant | determinant / abs determinant | 250 | **HELD-OUT** |

Held-out = two entire operation TYPES never seen in training. Leakage is asserted on BOTH the
target operation AND the fitted representative operation: no train problem uses or fits to a
determinant op; every held-out problem does. The word "determinant" appears in **0** training
traces and all 250 held-out traces — a clean unseen-rule-type generalization test. Zero id
overlap; no collision with original corpus ids.

## Trace design (the point of this corpus)

(a) approach in 2 lines (per-operator fit; operator marks the sign of negatives) → (b)
examples re-listed once → (c) the query operator and its example group identified → (d) a
**couple** of rejected candidate operations, one compact computed line each (NOT the full
32-op enumeration that bloated the original) → (e) the winning rule, verified against every
group example on one line each → (f) the rule applied to the query, then the box, terminating
exactly like the original corpus (`The answer in \boxed{–} is \boxed{ANS}` +
`\n</think>\n\boxed{ANS}<|im_end|>`). Every number shown is recomputed and asserted true at
build time; the derived output is asserted equal to the solver-verified gold.

## Outputs (under `$NEMOTRON_OUT`)

`corpus/<id>/synthetic.jsonl` + `corpus.jsonl` (train, loader-compatible; category
`equation_numeric_deduce`), `heldout/` (corpus + `heldout_problems.jsonl` prompt+gold),
`heldout_ids.json`, `index_all.jsonl`, `manifest.json`, `validation_report.md`,
`tokenizer_gate.json`.

## Mandatory tokenizer gate

`verify_tokenizer.py` runs 4 checks against 3 REAL `equation_numeric_deduce` entries
(`01cd504a`, `026106f5`, `03f07b43`): (1) two-decoder round-trip; (2) byte-exact id match of
the rebuilt prompt AND re-encoded completion vs the originals; (3) special-token layout
(`<think>\n` masked, `</think>`/`\boxed` trained); (4) mask boundary at the end of
`<think>\n`. Generation scripts refuse to run until the gate writes `passed: true`.

## Flags

1. **`guess` sub-family excluded** — undeterminable by construction (see above). Only
   `deduce` is built. This is the honest reading of "uniquely determined / never guess".
2. **Operators excluded from generation**: `-` (sign collision) and `{` `}` (would corrupt
   `\boxed{...}`). 22 of the 25 real symbols remain; all encode unambiguously.
3. **Query-group size skews higher than real**: real `deduce` query groups are 1–2 examples
   (hence frequently ambiguous); uniqueness forces ours to ~2–4, raising total examples to
   2–9 vs the real 3–5. This is the uniqueness-vs-fidelity tradeoff resolved per spec; the
   distribution is in `manifest.json`. Prompt phrasing and operator alphabet are byte-exact.
4. **Menu coverage**: ~6% of real `deduce` golds use the `-` operator or rules the 32-op menu
   doesn't cover exactly; these are out of scope. The menu reproduces ~90% of real golds
   exactly and 94% have a consistent menu rule.
