# family_probe — per-family generalization probe (measurement only)

Measures where the BASE Nemotron-3-Nano-30B model has real headroom across the
puzzle families, BEFORE building any generators. No training corpus is produced.

## Run order

```bash
export NEMOTRON_INPUT=/kaggle/input/<nemotron-corpus-dataset>   # train.csv, problems.jsonl
export PROBE_OUT=/kaggle/working/family_probe_out
export NEMOTRON_MODEL=/kaggle/input/<nemotron-3-nano-30b-model-path>

python build_probe_set.py   # CPU — probe.jsonl + probe_counts.json
python run_probe.py         # GPU — vLLM, base model, official decoding params
python report.py            # CPU — table ranked weakest-first + weakness_map.md
```

## What it does

1. **build_probe_set.py** — classifies all 9,500 train problems into 7 families
   (cryptarithm/equation_numeric sub-categories merged, tracked in `sub_category`),
   samples ~80 per family **stratified by status** (`rule_found` /
   `hypothesis_formed` / `rule_unknown` where they exist: equal targets,
   shortfall redistributed proportionally). Deterministic (seed in file).
   Statuses only diverge for bit_manipulation, cryptarithm, equation_numeric —
   cipher/gravity/numeral/unit_conversion are 100% rule_found in problems.jsonl,
   so their slices are single-status by construction.
2. **run_probe.py** — base model (NO adapter) in vLLM, official scorer's exact
   prompt construction (suffix + `apply_chat_template(..., add_generation_prompt=True,
   enable_thinking=True)`), official decoding (temperature 0.0, top_p 1.0,
   max_tokens 7680, max_model_len 8192). Grades each output with the official
   `extract_final_answer` + `verify`.
3. **report.py** — per-family table ranked weakest-first with status breakdown,
   mean generated tokens, truncation rate (`finish_reason == "length"`); writes
   `weakness_map.md`.

## Why the status breakdown matters

`rule_unknown` problems are ones the 0.85 winner's solvers could not crack.
Base-model accuracy on those vs `rule_found` separates "model can't induce the
rule" (real headroom for rule-explicit synthetic traces) from "family already
saturated" (no headroom — don't build generators for it).

## Grader provenance (FLAG)

The official metric notebook is not retrievable offline. `official_metric.py`
copies `extract_answer`/`compare_answer` VERBATIM from the winning submission's
`reasoning.py` (attached corpus), whose docstrings state they match the official
`metric_reference.extract_final_answer`/`verify`, and which implement all
documented quirks: last-non-empty-`\boxed{}` extraction, binary strict, numeric
rel-tol 1e-2 (abs 1e-5 near zero), case-insensitive otherwise. A third-party
GitHub reimplementation was checked and REJECTED (it omits binary-strict and
case-insensitivity). If you have the actual metric notebook open on Kaggle,
diff it against `official_metric.py` before trusting absolute numbers.

## Outputs (under `$PROBE_OUT`)

- `probe.jsonl` — {id, family, sub_category, status, prompt, gold} (shuffled)
- `probe_counts.json` — sampled vs available per family x status
- `generations.jsonl` — per-problem result: extracted answer, correct,
  finish_reason, num_gen_tokens, output tail (set `SAVE_FULL_OUTPUT=1` for full text)
- `weakness_map.md` — the deliverable: ranked headroom map

A CPU-built copy of `probe.jsonl` + `probe_counts.json` ships in this folder
(`output/`); `generations.jsonl`/`weakness_map.md` require the GPU run.

## Notes / flags

- Probe problems are real train.csv problems (model may have indirect familiarity
  if a prior adapter was trained on them — irrelevant here since this probes the
  BASE model, but do NOT reuse this slice as eval after fine-tuning on train.csv;
  for post-training eval, exclude probe ids from the training mix).
- ~560 problems x up to 7,680 generated tokens ≈ 1–2 h on a single T4/L4-class
  GPU with max_num_seqs 64; faster on A100-class.
- `temperature=0.0` makes the run deterministic up to kernel nondeterminism.
