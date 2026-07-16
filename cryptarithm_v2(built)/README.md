# cryptarithm_v2 — concise cryptarithm training corpus

Cryptarithm is the hardest family and the biggest headroom: the base model scores **0.0%
with 98.8% truncation** (mean ~7,610 gen tokens — it rambles to the budget and never boxes),
and the 0.85-winning corpus had almost no usable cryptarithm data (most were rule_unknown /
wrong-answer guesses). cryptarithm_v2 produces **code-verified, uniquely-determined,
extremely concise** traces that terminate:

**completion-token distribution: median 288, p90 311, max 515** (hard caps asserted:
median < 3,500, p90 < 3,500, max < 4,500 — the build fails otherwise).

Output format is a byte-exact drop-in for the existing combined training script: emits
`problems_all.jsonl` (`id, prompt, answer`) and `traces_all.jsonl` (`id, reasoning`) with the
same field names as bit_v2 / equation_v2, plus tokenized `corpus/<id>/synthetic.jsonl`.

## Run order (Kaggle, CPU-only generation)

```bash
export NEMOTRON_INPUT=/kaggle/input/<nemotron-corpus-dataset>
export NEMOTRON_OUT=/kaggle/working/cryptarithm_v2_out

python verify_tokenizer.py      # MANDATORY GATE — 4x PASS or nothing else runs
python generate_and_solve.py    # generate + solve + uniqueness-verify (golds code-computed)
python make_traces.py           # concise joint-induction traces
python tokenize_corpus.py       # exact segment format + conciseness caps enforced
python build_split.py           # train/held-out + manifest + validation report
```

A completed, verified local run ships in `output/` (artifacts + corpus sample +
`corpus_full.tar.gz`).

## What cryptarithm is (analysis of real train.csv)

Cryptarithm reuses the **exact equation_numeric prompt** ("In Alice's Wonderland, a secret
set of transformation rules is applied to equations…"). Every example LHS is 5 symbols
`AA op BB` (2-symbol operand, 1 operator symbol, 2-symbol operand); the result is a
variable-length symbol string. 3–5 examples; ~10 distinct symbols from a 26-char pool.
It is **equation_numeric with a per-problem symbol cipher**: the operator is decorative
(each operator carries its own hidden rule — in the official trace `+` and `*` both meant
*concatenation*), and the operands/results are written in a secret symbol→digit code.

`cryptarithm_deduce` (659): the query operator always appears in the examples → determinable.
`cryptarithm_guess` (164): the query operator never appears → **not determinable** (official
guess traces are 7% correct). We build **deduce only** and exclude guess.

**Two findings that shaped the design:**
1. Real deduce problems are **massively under-determined** — a single real problem admits
   59–114 distinct consistent golds. The "gold" is just the generator's secret, not
   inferable. Matching that ambiguity is impossible under the "never guess / unique"
   discipline, so we **generate our own uniquely-determined problems**.
2. The op-space splits into a **mapping-free positional regime** (concatenation /
   reverse-concatenation — solvable on symbols directly; the only regime the official traces
   ever solved, ~9% of real deduce) and an **arithmetic regime** (needs the digit cipher,
   ~90%). Rigorous arithmetic uniqueness over a ~10-symbol cipher (10! mappings) is
   computationally intractable, so the arithmetic cell uses a **small cipher** where a
   node-capped exhaustive uniqueness search is fast.

## Two regimes (both code-verified)

**Positional** (1,900 problems). The operation rearranges the operand symbols; the 8
operand/result-reversal variants collapse to **4 distinct functions**: `A|B`, `B|A`,
`revA|revB`, `revB|revA`. Uniqueness is checked within this 4-rule menu *and* against the
arithmetic menu (the latter is excluded for free on length for additive ops, since a 4-symbol
result can't come from a ≤3-digit sum/diff). Verifies in ~0.06 s/problem.

**Arithmetic** (290 problems). A small symbol→digit cipher (5–6 symbols) + an arithmetic op
(add / sub / reverse-sub / abs-diff / mod / reverse-mod / int-div, plus multiplication in the
held-out cell) + optional reverse-operands / reverse-result. Examples are added until a
**node-capped exhaustive search proves no alternative (op × action × injective cipher) over
the full menu reproduces all examples with a different query result**; anything not provably
unique within the node cap is **discarded** (golds are correct by construction, so discarding
only affects yield, never correctness).

## Generation discipline

- **Generate-then-verify**: a (rule [+ cipher]) is chosen first, examples + query are
  rendered, and the gold is computed by executing that rule. Every gold is therefore correct
  by construction and is re-asserted against every example at trace-build time.
- **Uniqueness**: every kept problem is proven uniquely determined (see above).
- **Independent re-verification** (from the prompt text alone): positional **1,900/1,900**
  unique with matching gold; arithmetic **290/290** unique, **zero conflicts** (no poison).

## Cells and held-out split

| cell | regime | n | split |
|---|---|---|---|
| pos_AB (A\|B) | positional | 650 | train |
| pos_BA (B\|A) | positional | 550 | train |
| pos_rArB (revA\|revB) | positional | 500 | train |
| pos_rBrA (revB\|revA) | positional | 200 | **HELD-OUT** |
| arith_main (add/sub/absdiff/mod/div…) | arithmetic | 220 | train |
| arith_mul (multiplication) | arithmetic | 70 | **HELD-OUT** |

Held-out tests unseen-rule generalization: `pos_rBrA` is the one arrangement that is never
the *answer* in training, and `arith_mul` (multiplication) is **never named or used in any
training trace** (a fully unseen operation). Zero id leakage, zero structural leakage, no
collision with original corpus ids. (Note: positional traces name all 4 arrangements as
candidates, so `rBrA` is *mentioned* as a rejected option in train — the holdout tests
producing it as the solution, which never occurs in training.)

## Trace design (the point of this corpus)

Positional: state that symbols are opaque tokens → list examples → identify the query
operator's group → test the 4 arrangements on the first example → confirm the winner on the
rest → apply to the query → box. Arithmetic: state that symbols are digits → give the deduced
symbol→digit code and operation → verify on every example (decode, compute, encode) → apply
to the query → box. Every value shown is recomputed and asserted true; the derived output is
asserted equal to the code-verified gold. Termination is exactly
`The answer in \boxed{–} is \boxed{ANS}` + `\n</think>\n\boxed{ANS}<|im_end|>`.

## Mandatory tokenizer gate

`verify_tokenizer.py` runs 4 checks against 3 REAL cryptarithm_deduce entries (`0133bcec`,
`02a04b59`, `02b8d816`): two-decoder round-trip; byte-exact id match of the rebuilt prompt
AND re-encoded completion; special-token layout (`<think>\n` masked, `</think>`/`\boxed`
trained); mask boundary at end of `<think>\n`. Generation refuses to run until it passes.

## Flags

1. **`guess` excluded** — undeterminable by construction (query operator unseen).
2. **Operators `{` `}` excluded** from the symbol pool (they would corrupt `\boxed{...}`); the
   other 24 real symbols are used (including `-`, which is a plain symbol here, not a sign).
3. **Arithmetic uses a small cipher (5–6 symbols)** so uniqueness is rigorously verifiable;
   real problems use ~10 symbols but are under-determined (unverifiable). Positional uses
   6–8. This favors correctness/rigor over surface realism — appropriate for a family at 0%.
4. **Arithmetic yield**: problems not provably unique within the node cap are discarded;
   golds are always correct by construction, so this only affects yield.
