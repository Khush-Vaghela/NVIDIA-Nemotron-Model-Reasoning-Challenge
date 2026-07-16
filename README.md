# Concise, Code-Verified Synthetic Traces for the Nemotron-3-Nano Reasoning Challenge

## Summary

Our final adapter is a rank-32 LoRA over NVIDIA-Nemotron-3-Nano-30B-A3B trained on a
mixed corpus: four families sourced from the public reference corpus, and three families
(bit manipulation, equation, cryptarithm) regenerated from scratch as **concise,
code-verified reasoning traces**. The central finding driving our approach is that on the
hardest families, the base model fails almost entirely by **truncation** — it reasons
correctly but runs past the token budget and never emits a boxed answer — rather than by
faulty reasoning. Our concise traces target that failure directly.

## Diagnosis: the benchmark and the dominant failure mode

The benchmark consists of inductive rule-puzzles: each prompt presents input→output
examples, and the model must infer the hidden rule and apply it to a query, placing the
answer in `\boxed{}`. We grouped problems into families (bit manipulation, cipher, unit
conversion, gravity, numeral, and an "equations" family that includes both numeric-operator
equations and symbol-encoded cryptarithms).

We probed the base model on held-out slices of every family using the official grader. The
result was decisive: **accuracy tracks the inverse of the truncation rate almost exactly.**

| family | base accuracy | truncation rate | mean generated tokens |
|---|---|---|---|
| cryptarithm | 0.0% | 98.8% | 7,610 |
| bit manipulation | 7.5% | 91.2% | 7,352 |
| equation_numeric | 18.8% | 83.8% | 6,951 |
| cipher | 41.2% | 58.8% | 6,318 |
| unit conversion | 63.7% | 45.0% | 5,327 |
| gravity | 65.0% | 6.2% | 2,849 |
| numeral | 100.0% | 0.0% | 172 |

The model is not failing the hard families by reasoning incorrectly — it is rambling past
the 7,680-token budget and never boxing an answer. The lever, therefore, is not better
rule-coverage but **concise, bounded, terminating traces**.

## What we did NOT find useful

Before settling on conciseness, we tested a structural-variant hypothesis on the numeral
family: that augmenting with unseen rule structures (e.g. Roman→Arabic, other bases) would
improve generalization. A held-out probe refuted this for most cases — the base model
already generalizes to unseen numeral structures (Arabic→base-7: 100%, Roman→Arabic: 97%),
with one narrow exception (base-7→decimal induction: 18%). The takeaway: **structural-variant
augmentation is a narrow lever, not a broad one.** This redirected effort toward the
truncation problem, where the real headroom lay.

## Method: concise, code-verified synthetic traces

For each of the three high-headroom families (bit, equation, cryptarithm) we built a
generation pipeline with the following properties:

**Generate-then-verify.** Rather than solving each puzzle from scratch, we sample a ground-truth
rule first (a per-bit operation menu for bit; a composed numeric operation for equation; a
symbol mapping plus operation for cryptarithm), render the examples and query, and compute the
answer by executing the known rule. Every gold answer is verified by applying the solved rule
to all examples and asserting exact reproduction. This guarantees correctness — important
because the grader is exact and unforgiving, and the public reference corpus itself contained
a meaningful fraction of answer-incorrect traces.

**Uniqueness checking.** Especially for cryptarithm (which has two coupled unknowns — the
symbol mapping and the operation), we verify that the examples uniquely determine the rule:
no alternative rule in the search space reproduces all examples with a different query answer.
Non-unique problems are discarded.

**Concise traces.** The traces demonstrate the induction process (observe → hypothesize →
verify against examples → apply → box) but are deliberately compact, avoiding the exhaustive
enumeration that bloats the base model's output. The resulting completion-token distributions:

| family | base model | our traces (median / max) |
|---|---|---|
| cryptarithm | ~7,610 (98.8% truncated) | 288 / 515 |
| equation | ~5,800 (84% truncated) | 435 / 780 |
| bit manipulation | ~6,700 | ~1,000 / ~1,350 |

**Held-out structural validation.** Each family holds out entire rule-structure cells the
training set never contains (e.g. for cryptarithm, the multiplication operation and a specific
reversal permutation), with assertions against both id and structural leakage, so we can
measure generalization to unseen rule types rather than memorization.

**Format fidelity.** Every trace is tokenized into the exact segment format the model expects
(prompt masked through the `<think>` opener, completion supervised, terminating in
`</think>\n\boxed{ANSWER}<|im_end|>`), validated by a byte-exact tokenizer round-trip against
real reference-corpus entries before any generation.

## Training

The adapter is a rank-32 LoRA (alpha 32) targeting the attention, Mamba, and MoE-expert
projections (`q/k/v/o_proj`, `in/out_proj`, `up/down_proj`), trained with Unsloth's MoE-LoRA
support so the adapter is serveable through vLLM's LoRA path. Training uses pre-tokenized
sequences with prompt-masked labels, a single causal shift, bf16, one epoch, LR 2e-4 with a
cosine schedule. We supply our own concise traces for bit, equation, and cryptarithm, and use
the reference corpus for cipher, unit, gravity, and numeral.

## Results

The approach improved progressively as family coverage and data volume increased, with the
combined seven-family adapter (four reference families + our three concise families) reaching
our best score. The concise-trace families measurably contributed: adding our equation and
cryptarithm traces on top of the four reference families plus bit improved the score, on
exactly the families where the reference corpus was weakest (cryptarithm in particular was
largely unsolved in the reference corpus).

## Key lessons

- On hard reasoning families, **truncation can dominate over reasoning error** — diagnose the
  failure mode before optimizing rule coverage.
- **Code-verify every synthetic answer.** Exact graders punish any wrong gold, and unverified
  generation accumulates errors silently.
- **Hold out whole structural cells**, not random instances, to measure real generalization.
- **Match the training format to the scorer byte-for-byte** (prompt construction, special
  tokens, terminator), or the adapter silently underperforms.
- For MoE models, the LoRA must target experts through a serveable mechanism, or it loads in
  training but crashes at inference.

## Reproducibility

The generation pipelines (rule solver, problem generator, concise-trace generator, tokenizer
gate, held-out split, validation manifest) and the training/packaging notebook are provided.
Each family's corpus ships with a manifest reporting per-cell counts, token distributions, and
a verification summary (100% code-verified golds, 100% round-trip, zero leakage).

## References and acknowledgements

The four reference families (cipher, unit conversion, gravity, numeral) were sourced from the
public reference corpus snapshot, which we gratefully acknowledge:
[huikang-nemotron-repository-snapshot](https://www.kaggle.com/datasets/huikang/huikang-nemotron-repository-snapshot).
Our contribution is the concise, code-verified synthetic regeneration of the bit-manipulation,
equation, and cryptarithm families, plus the truncation-focused diagnosis and held-out
structural validation described above.
