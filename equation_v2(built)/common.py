"""Shared config, tokenizer wrapper, and equation-rule machinery for equation_v2.

Paths (env-overridable, Kaggle defaults):
    NEMOTRON_INPUT -> folder with tokenizer.json, vocab.jsonl, corpus/, train.csv
    NEMOTRON_OUT   -> writable output folder

Rule space (reverse-engineered + verified against the real `equation_numeric_deduce`
family in train.csv -- reproduces ~94% of real golds EXACTLY, see README):

Each example line is `AA<op>BB = R` where AA, BB are 2-digit operands (00-99, leading
zeros kept) and <op> is a SYMBOLIC operator drawn from a symbol alphabet. In the `deduce`
family EACH operator symbol carries its OWN hidden rule; the query operator always appears
among the example operators, so the rule is deduced from the examples that share the query
operator (the other operators are distractors).

A rule = (operation, operand_reverse, result_reverse):
  * operand_reverse: reverse each operand's 2-char string (06 -> 60) before computing.
  * operation: one of 32 candidates (see MENU below): concat, add, abs-diff, sub, mul,
    div, mod, digit-wise ops, cross-multiply, determinant, +/-1 offsets, ...
  * result_reverse: reverse the result STRING (sign included, so it moves to the end).
NEGATIVE ENCODING (verified against real golds): if the final result is negative, the
'-' is dropped and the operator symbol takes its place -- at the front if the result was
not reversed (e.g. `(17`), at the end if it was reversed (`03!`). STR-type operations
(concatenation / digit-concat) are never negative.

The `guess` family (query operator UNSEEN in examples) is DELIBERATELY EXCLUDED: its
rule is not determinable from the prompt -- even the official corpus traces box the wrong
answer 87% of the time. Per the discipline ("never guess; uniquely determined"), we only
build `deduce`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

INPUT_DIR = Path(os.environ.get("NEMOTRON_INPUT", "/kaggle/input/nemotron-master"))
OUT_DIR = Path(os.environ.get("NEMOTRON_OUT", "/kaggle/working/equation_v2_out"))

TOKENIZER_PATH = INPUT_DIR / "tokenizer.json"
VOCAB_JSONL = INPUT_DIR / "vocab.jsonl"
REF_CORPUS_DIR = INPUT_DIR / "corpus"
TRAIN_CSV = INPUT_DIR / "train.csv"

SEED = 20260615
TOKEN_LIMIT = 8192
# HARD conciseness caps on the TRAINED completion (the whole point of equation_v2):
TRACE_MAX_TOKENS = 4500
TRACE_P90_TARGET = 3000
TRACE_MEDIAN_TARGET = 3000

# ---- chat template (gate-verified against reference deduce entries) ----
CHAT_PRE = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
CHAT_POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n"

# 3 REAL equation_numeric_deduce entries present in the reference corpus folder.
REF_EQ_IDS = ["01cd504a", "026106f5", "03f07b43"]

# ---- original prompt template (byte-exact, asserted against train.csv) ----
PROMPT_INTRO = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to "
    "equations. Below are a few examples:\n"
)
PROMPT_QUERY = "\nNow, determine the result for: "

# Operator alphabet observed in the real family. EXCLUDED from generation (flagged in
# README): '-' (its symbol collides with the negative sign, making golds ambiguous) and
# the braces '{' '}' (they would corrupt \boxed{...} extraction of a negative gold). The
# 22 remaining symbols all encode unambiguously inside \boxed{}.
OPERATOR_ALPHABET = list("!\"#$%&'()*+/:<>?@[\\]^`|")


def build_problem_prompt(examples: list[tuple[str, str, str, str]],
                         query: tuple[str, str, str]) -> str:
    """examples: list of (left, op, right, result); query: (left, op, right)."""
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


# ---------------- operation menu ----------------
def _digits(s: str) -> tuple[int, int]:
    return int(s[0]), int(s[1])


@dataclass(frozen=True)
class Op:
    name: str
    kind: str  # "INT" or "STR"
    rare: bool

    def raw(self, sa: str, sb: str):
        return _OP_FUNCS[self.name](sa, sb)


_OP_FUNCS = {
    "concatenation": lambda a, b: a + b,
    "reverse concatenation": lambda a, b: b + a,
    "addition": lambda a, b: int(a) + int(b),
    "absolute difference": lambda a, b: abs(int(a) - int(b)),
    "negated absolute difference": lambda a, b: -abs(int(a) - int(b)),
    "subtraction": lambda a, b: int(a) - int(b),
    "reverse subtraction": lambda a, b: int(b) - int(a),
    "multiplication": lambda a, b: int(a) * int(b),
    "multiply+1": lambda a, b: int(a) * int(b) + 1,
    "multiply-1": lambda a, b: int(a) * int(b) - 1,
    "add+1": lambda a, b: int(a) + int(b) + 1,
    "add-1": lambda a, b: int(a) + int(b) - 1,
    "sub+1": lambda a, b: int(a) - int(b) + 1,
    "sub-1": lambda a, b: int(a) - int(b) - 1,
    "max mod min": lambda a, b: (
        max(int(a), int(b)) % min(int(a), int(b)) if min(int(a), int(b)) != 0 else None
    ),
    "integer division": lambda a, b: (int(a) // int(b)) if int(b) != 0 else None,
    "modulo": lambda a, b: (int(a) % int(b)) if int(b) != 0 else None,
    "reverse division": lambda a, b: (int(b) // int(a)) if int(a) != 0 else None,
    "reverse modulo": lambda a, b: (int(b) % int(a)) if int(a) != 0 else None,
    "digit absolute diff": lambda a, b: str(abs(_digits(a)[0] - _digits(b)[0]))
    + str(abs(_digits(a)[1] - _digits(b)[1])),
    "digit add mod10": lambda a, b: str((_digits(a)[0] + _digits(b)[0]) % 10)
    + str((_digits(a)[1] + _digits(b)[1]) % 10),
    "digit sub mod10": lambda a, b: str((_digits(a)[0] - _digits(b)[0]) % 10)
    + str((_digits(a)[1] - _digits(b)[1]) % 10),
    "cross multiply": lambda a, b: _digits(a)[0] * _digits(b)[0]
    + _digits(a)[1] * _digits(b)[1],
    "cross multiply rev": lambda a, b: _digits(a)[0] * _digits(b)[1]
    + _digits(a)[1] * _digits(b)[0],
    "digit multiply": lambda a, b: str(_digits(a)[0] * _digits(b)[0])
    + str(_digits(a)[1] * _digits(b)[1]),
    "digit multiply rev": lambda a, b: str(_digits(a)[0] * _digits(b)[1])
    + str(_digits(a)[1] * _digits(b)[0]),
    "digit sum diff": lambda a, b: (_digits(a)[0] + _digits(a)[1])
    - (_digits(b)[0] + _digits(b)[1]),
    "digit sum sum": lambda a, b: (_digits(a)[0] + _digits(a)[1])
    + (_digits(b)[0] + _digits(b)[1]),
    "digit product diff": lambda a, b: _digits(a)[0] * _digits(a)[1]
    - _digits(b)[0] * _digits(b)[1],
    "digit product sum": lambda a, b: _digits(a)[0] * _digits(a)[1]
    + _digits(b)[0] * _digits(b)[1],
    "determinant": lambda a, b: _digits(a)[0] * _digits(b)[1]
    - _digits(a)[1] * _digits(b)[0],
    "abs determinant": lambda a, b: abs(
        _digits(a)[0] * _digits(b)[1] - _digits(a)[1] * _digits(b)[0]
    ),
}

_COMMON_NAMES = [
    "concatenation",
    "reverse concatenation",
    "addition",
    "absolute difference",
    "negated absolute difference",
    "subtraction",
    "reverse subtraction",
    "multiplication",
]
_RARE_NAMES = [n for n in _OP_FUNCS if n not in _COMMON_NAMES]
STR_OPS = {
    "concatenation",
    "reverse concatenation",
    "digit absolute diff",
    "digit add mod10",
    "digit sub mod10",
    "digit multiply",
    "digit multiply rev",
}

MENU: list[Op] = [
    Op(n, "STR" if n in STR_OPS else "INT", n in _RARE_NAMES)
    for n in (_COMMON_NAMES + _RARE_NAMES)
]
OP_BY_NAME = {op.name: op for op in MENU}
assert len(MENU) == 32


@dataclass(frozen=True)
class Rule:
    op: str
    opnd_rev: bool
    res_rev: bool

    def label(self) -> str:
        acts = []
        if self.opnd_rev:
            acts.append("reversed operands")
        if self.res_rev:
            acts.append("reversed result")
        acts_s = (" [" + ", ".join(acts) + "]") if acts else " [identity]"
        return f"{self.op}{acts_s}"


def reverse_string_result(s: str) -> str:
    """Reverse the full result string; a leading '-' moves to the end."""
    return s[::-1]


def render(rule: Rule, A: str, B: str, operator_symbol: str) -> str | None:
    """Apply a rule to operand strings A,B. Returns the gold STRING, or None if the
    operation is undefined (division/mod by zero)."""
    sa = A[::-1] if rule.opnd_rev else A
    sb = B[::-1] if rule.opnd_rev else B
    v = OP_BY_NAME[rule.op].raw(sa, sb)
    if v is None:
        return None
    s = str(v)
    if rule.res_rev:
        s = reverse_string_result(s)
    if "-" in s:  # negative encoding: operator symbol replaces the '-'
        s = s.replace("-", operator_symbol)
    return s


def all_rules() -> list[Rule]:
    """All 32 ops x 4 action combos = 128 candidate rules, in canonical priority order:
    common-before-rare, and for the action nesting the order observed in real traces."""
    rules = []
    actions = [(True, True), (False, False), (True, False), (False, True)]
    for rare in (False, True):
        for opnd_rev, res_rev in actions:
            for op in MENU:
                if op.rare == rare:
                    rules.append(Rule(op.name, opnd_rev, res_rev))
    return rules
