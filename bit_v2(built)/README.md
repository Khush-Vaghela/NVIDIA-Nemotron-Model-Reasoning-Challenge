# bit_v2 — concise bit_manipulation training corpus

Kills the truncation failure mode: base-model probe showed bit_manipulation at
7.5% accuracy with 91% truncation; the original corpus's bit traces run ~6,700
tokens (p90 ~7,200) against a 7,680-token budget. bit_v2 traces reach the same
verified answers in a fraction of the tokens (hard caps: median < 3,500,
p90 < 4,000, max < 4,500 — asserted, build fails otherwise). Format is byte-exact
drop-in for the existing `train_sft.py` (reuses the validated numeral_v2 pipeline
structure: tokenizer gate, segment format, held-out split, manifest discipline).

## Run order (Kaggle, CPU-only generation, no downloads)

```bash
export NEMOTRON_INPUT=/kaggle/input/<nemotron-corpus-dataset>  # tokenizer.json, vocab.jsonl, corpus/, train.csv
export NEMOTRON_OUT=/kaggle/working/bit_v2_out

python verify_tokenizer.py    # MANDATORY GATE — 4x PASS or nothing else runs
python solve_rules.py         # optional: solver self-test
python generate_problems.py   # ~10 min CPU: rules + uniqueness + verified golds
python make_traces.py         # concise induction traces
python tokenize_corpus.py     # exact segment format + caps enforcement
python build_split.py         # train/held-out + manifest + validation report
```

## Rule space (verified against train.csv)

All 1,602 train bit problems match the byte-exact prompt template (asserted),
8-bit -> 8-bit, 7–10 examples. Each output bit j is one function of 1–2 input
bits from the menu {C0, C1, ID, NOT, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT}
(AND-NOT(a,b)=a&~b etc.; bits indexed 0–7 from the left). Solver findings on
real problems (informational): ~97.5% of output bits are satisfiable by this
menu; only ~2.5% need something wider (e.g. 3-bit majority). However ~75% of
real bits are AMBIGUOUS under the full menu given only their examples — the
winning solver used operand priors. bit_v2 problems are instead generated to be
UNIQUELY determined (see below), which is a strictly cleaner training signal.

## Generation discipline

- **Solver** (`solve_rules.py`): per output bit, all 250 menu candidates are
  precomputed as 256-entry truth tables; consistent set filtered against every
  example; bit solved iff all survivors share ONE truth table (gold invariant
  to representative). Solved rule re-applied to every example and asserted to
  reproduce it exactly.
- **Uniqueness**: problems start at 7 examples; while any bit is ambiguous, a
  disambiguating input (where surviving tables disagree) is appended, up to 12;
  otherwise the problem is resampled. **FLAG:** original problems have 7–10
  examples; reaching full-menu uniqueness pushes most of ours to 10–12 (the
  distribution is in the manifest). This is the uniqueness-vs-fidelity tradeoff
  resolved per spec ("add examples until unique").
- **Golds**: computed by executing the solved rule on the query; asserted to be
  exactly 8 binary chars (strict grader); solver rule asserted functionally
  identical to the sampled true rule on all 256 inputs.
- **Operands** are biased ~70% toward neighbor bits (offsets observed in
  solver-unique real rules); 30% arbitrary.

## Cells and held-out split

| cell | rule structure | n | split |
|---|---|---|---|
| basic_simple | only ID / NOT / constants | 800 | train |
| basic_mixed | + AND / OR / XOR | 2,000 | train |
| negated_ops | >=1 of AND-NOT / OR-NOT / XOR-NOT | 300 | **HELD-OUT** |

Held-out = an entire rule STRUCTURE never seen in training. Leakage asserts are
truth-table-based (airtight): every train problem's 8 fitted functions are
basic-op-expressible; every held-out problem has >=1 that is not. Zero id
overlap; no collision with original corpus ids.

## Trace design (the point of this corpus)

(a) approach in 2 lines → (b) example pairs re-listed once → (c) ONE line per
output bit naming the winning rule with all-example evidence
(`out0 = OR(in1, NOT(in7)): 0,0->1 1,0->1 ... - all match`) — no menu
enumeration — → (d) per-bit application to the query → (e) the original corpus's
exact termination (`The answer in \boxed{–} is \boxed{ANS}` +
`\n</think>\n\boxed{ANS}<|im_end|>`). Every evidence value in every trace is
re-executed and asserted true at trace-build time; the trace's derived output is
asserted equal to the solver-verified gold.

## Outputs (under `$NEMOTRON_OUT`)

`corpus/<id>/synthetic.jsonl` + `corpus.jsonl` (train, loader-compatible;
category `bit_manipulation`), `heldout/` (corpus + `heldout_problems.jsonl`
prompt+gold), `heldout_ids.json`, `manifest.json`, `validation_report.md`,
`tokenizer_gate.json`. A completed verified local run ships in `output/` in this
folder.

## Flags

1. Example-count distribution shifted to 10–12 vs original 7–10 (uniqueness
   requirement; see manifest).
2. ~2.5% of REAL bit problems use rules outside the 1–2-bit menu (3-bit ops);
   bit_v2 does not cover those (out of scope per the rule-space spec).
3. Train cells exclude negated ops entirely (they are the held-out structure),
   so the train corpus teaches the fit-and-apply METHOD on basic ops and the
   held-out set tests transfer to unseen op types.
