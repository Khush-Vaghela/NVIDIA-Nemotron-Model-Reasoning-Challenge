"""Run the family probe against the BASE model (no adapter) with vLLM. GPU script.

Prompt construction matches the official scorer exactly:
  user_content = prompt + "\\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`"
  tokenizer.apply_chat_template(messages, tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680), max_model_len=8192.

Grading uses the official extract_final_answer + verify (official_metric.py — see
provenance note there).

Output: OUT_DIR/generations.jsonl
  {id, family, sub_category, status, extracted, gold, correct,
   finish_reason, num_gen_tokens, output_tail}
(output_tail = last 400 chars, for debugging; full texts optionally kept with
SAVE_FULL_OUTPUT=1.)

Usage (Kaggle GPU notebook): python run_probe.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------- paths & knobs (Kaggle-parameterized) ----------------
MODEL_PATH = os.environ.get(
    "NEMOTRON_MODEL",
    "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
)
PROBE_PATH = Path(os.environ.get("PROBE_PATH", "/kaggle/working/family_probe_out/probe.jsonl"))
OUT_DIR = Path(os.environ.get("PROBE_OUT", "/kaggle/working/family_probe_out"))
SAVE_FULL_OUTPUT = os.environ.get("SAVE_FULL_OUTPUT", "0") == "1"

MAX_MODEL_LEN = 8192
MAX_TOKENS = 7680          # official inference cap
GPU_MEM_UTIL = 0.85
MAX_NUM_SEQS = 64

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    probe = [json.loads(l) for l in open(PROBE_PATH)]
    print(f"Loaded {len(probe)} probe problems from {PROBE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompts = []
    for p in probe:
        messages = [{"role": "user", "content": p["prompt"] + PROMPT_SUFFIX}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(text)

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_seqs=MAX_NUM_SEQS,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(prompts, sampling)

    from official_metric import extract_final_answer, verify

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_correct = 0
    with open(OUT_DIR / "generations.jsonl", "w") as f:
        for p, out in zip(probe, outputs):
            comp = out.outputs[0]
            text = comp.text
            extracted = extract_final_answer(text)
            correct = verify(p["gold"], extracted)
            n_correct += correct
            rec = {
                "id": p["id"],
                "family": p["family"],
                "sub_category": p["sub_category"],
                "status": p["status"],
                "extracted": extracted,
                "gold": p["gold"],
                "correct": bool(correct),
                "finish_reason": comp.finish_reason,
                "num_gen_tokens": len(comp.token_ids),
                "output_tail": text[-400:],
            }
            if SAVE_FULL_OUTPUT:
                rec["output"] = text
            f.write(json.dumps(rec) + "\n")

    print(f"Overall: {n_correct}/{len(probe)} = {n_correct / len(probe):.3f}")
    print(f"Wrote {OUT_DIR / 'generations.jsonl'} — now run report.py")


if __name__ == "__main__":
    main()
