"""Shared config, tokenizer wrapper, and cryptarithm machinery for cryptarithm_v2.

Paths (env-overridable, Kaggle defaults):
    NEMOTRON_INPUT -> folder with tokenizer.json, vocab.jsonl, corpus/, train.csv
    NEMOTRON_OUT   -> writable output folder

CRYPTARITHM = equation_numeric with a per-problem SYMBOL cipher.
Each example line is `AA<op>BB = R` where:
  * AA, BB are 2-SYMBOL operands and R is a variable-length SYMBOL result,
  * <op> is a decorative operator symbol (like equation_deduce: each operator carries its
    OWN hidden rule; the query operator always appears among the examples).
Symbols are drawn from a 22-char pool (the real 26-char pool minus '{' '}' which would
corrupt \boxed{...}, and we also avoid them as operators).

Two rule REGIMES (verified against real train.csv cryptarithm_deduce):
  1. POSITIONAL (mapping-free): the operation rearranges the operand SYMBOLS directly --
     concatenation / reverse-concatenation, each optionally reversing operands and/or the
     result. No digit decoding is needed; this is exactly what the few successful official
     traces did. Uniqueness is checked within the 8-rule positional menu (fast).
  2. ARITHMETIC (cipher): each symbol maps to a digit; the operation is add/sub/mul/... on
     the decoded 2-digit operands, the result re-encoded to symbols. Generated only with a
     SMALL cipher so a node-capped EXHAUSTIVE uniqueness search is tractable; any problem
     not PROVABLY unique within the node cap is discarded (golds are always correct by
     construction, so discarding only affects yield, never correctness).

The `guess` sub-family (query operator UNSEEN in the examples) is EXCLUDED: it is not
determinable from the prompt -- the official guess traces box the right answer only 7%.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

INPUT_DIR = Path(os.environ.get("NEMOTRON_INPUT", "/kaggle/input/nemotron-master"))
OUT_DIR = Path(os.environ.get("NEMOTRON_OUT", "/kaggle/working/cryptarithm_v2_out"))

TOKENIZER_PATH = INPUT_DIR / "tokenizer.json"
VOCAB_JSONL = INPUT_DIR / "vocab.jsonl"
REF_CORPUS_DIR = INPUT_DIR / "corpus"
TRAIN_CSV = INPUT_DIR / "train.csv"

SEED = 20260616
TOKEN_LIMIT = 8192
# HARD conciseness caps on the TRAINED completion (truncation is THE failure mode here):
TRACE_MAX_TOKENS = 4500
TRACE_P90_TARGET = 3500
TRACE_MEDIAN_TARGET = 3500

# ---- chat template (gate-verified against reference cryptarithm entries) ----
CHAT_PRE = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
CHAT_POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n"

# 3 REAL cryptarithm_deduce entries present in the reference corpus folder.
REF_CRYPT_IDS = ["0133bcec", "02a04b59", "02b8d816"]

# ---- original prompt template (byte-exact, asserted against train.csv) ----
PROMPT_INTRO = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to "
    "equations. Below are a few examples:\n"
)
PROMPT_QUERY = "\nNow, determine the result for: "

# 22-symbol pool: real 26-char pool minus '{' '}' (boxed safety) -- also used for operators.
SYMBOL_POOL = list("!\"#$%&'()*+-/:<>?@[\\]^`|")
assert len(SYMBOL_POOL) == 24 and "{" not in SYMBOL_POOL and "}" not in SYMBOL_POOL


def build_problem_prompt(examples: list[tuple[str, str, str, str]],
                         query: tuple[str, str, str]) -> str:
    """examples: list of (A, op, B, R); query: (A, op, B). A,B are 2-symbol, R is symbols."""
    body = "".join(f"{a}{op}{b} = {r}\n" for (a, op, b, r) in examples)
    qa, qop, qb = query
    return PROMPT_INTRO + body + PROMPT_QUERY + f"{qa}{qop}{qb}"


# ---------------- tokenizer ----------------
_tok = None


def get_tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]

        _tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    return _tok


def encode(text: str) -> list[int]:
    return get_tokenizer().encode(text, add_special_tokens=False).ids


def decode(ids: list[int]) -> str:
    return get_tokenizer().decode(ids, skip_special_tokens=False)


_BYTE_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")
_vocab: dict[int, str] | None = None


def vocab_decode(ids: list[int]) -> str:
    """Independent decoder from vocab.jsonl (mirrors the repo's generate_csv.py)."""
    global _vocab
    if _vocab is None:
        _vocab = {}
        with open(VOCAB_JSONL) as f:
            for line in f:
                e = json.loads(line)
                _vocab[e["token_id"]] = e["token"]
    parts: list[str] = []
    buf = bytearray()
    for t in ids:
        s = _vocab.get(t, f"<unk:{t}>")
        m = _BYTE_RE.findall(s)
        if m and _BYTE_RE.sub("", s) == "":
            for h in m:
                buf.append(int(h, 16))
            continue
        if buf:
            parts.append(buf.decode("utf-8", errors="replace"))
            buf.clear()
        parts.append(s)
    if buf:
        parts.append(buf.decode("utf-8", errors="replace"))
    return "".join(parts).replace("▁", " ")


def build_segments(tokens: list[int], mask: list[int]) -> list[dict]:
    """Interleaved masked/unmasked segments -- identical logic to corpus.py."""
    if not tokens:
        return []
    segments: list[dict] = []
    seg_start = 0
    current_type = "unmasked" if mask[0] == 1 else "masked"
    for i in range(1, len(tokens)):
        token_type = "unmasked" if mask[i] == 1 else "masked"
        if token_type != current_type:
            segments.append(
                {"type": current_type, "pos": seg_start, "tokens": tokens[seg_start:i]}
            )
            seg_start = i
            current_type = token_type
    segments.append(
        {"type": current_type, "pos": seg_start, "tokens": tokens[seg_start:]}
    )
    return segments


# ================= POSITIONAL (mapping-free) op menu =================
# Operation rearranges operand SYMBOLS directly. 2 base ops x reverse-operands x
# reverse-result = 8 rules. Operands are 2-symbol strings; result is a symbol string.

@dataclass(frozen=True)
class PosRule:
    op: str          # "concat" | "rconcat"
    opnd_rev: bool
    res_rev: bool

    def label(self) -> str:
        base = "concatenation" if self.op == "concat" else "reverse concatenation"
        acts = []
        if self.opnd_rev:
            acts.append("reversed operands")
        if self.res_rev:
            acts.append("reversed result")
        return base + ((" [" + ", ".join(acts) + "]") if acts else " [identity]")


def pos_render(rule: PosRule, A: str, B: str) -> str:
    a = A[::-1] if rule.opnd_rev else A
    b = B[::-1] if rule.opnd_rev else B
    r = (a + b) if rule.op == "concat" else (b + a)
    return r[::-1] if rule.res_rev else r


def pos_menu() -> list[PosRule]:
    out = []
    for op in ("concat", "rconcat"):
        for orv in (False, True):
            for rrv in (False, True):
                out.append(PosRule(op, orv, rrv))
    return out
