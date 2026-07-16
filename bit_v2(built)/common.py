"""Shared config, tokenizer wrapper, and bit-rule machinery for bit_v2.

Paths (env-overridable, Kaggle defaults):
    NEMOTRON_INPUT -> folder with tokenizer.json, vocab.jsonl, corpus/, train.csv
    NEMOTRON_OUT   -> writable output folder

Bit rule space (verified against all 1,602 train.csv bit problems: 8-bit -> 8-bit,
7-10 examples, byte-exact prompt template):
each OUTPUT bit j is one function of 1-2 INPUT bits from the menu
  {C0, C1, ID(a), NOT(a), AND(a,b), OR(a,b), XOR(a,b),
   AND-NOT(a,b)=a&~b, OR-NOT(a,b)=a|~b, XOR-NOT(a,b)=a^~b (=XNOR)}
Bits are indexed 0..7 LEFT to right (as displayed in the prompt strings).

Candidates are precomputed as 256-bit truth tables (int) over all 8-bit inputs;
"unique rule" = all menu candidates consistent with the examples share ONE truth
table (so the gold answer is invariant to representative choice).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

INPUT_DIR = Path(os.environ.get("NEMOTRON_INPUT", "/kaggle/input/nemotron-master"))
OUT_DIR = Path(os.environ.get("NEMOTRON_OUT", "/kaggle/working/bit_v2_out"))

TOKENIZER_PATH = INPUT_DIR / "tokenizer.json"
VOCAB_JSONL = INPUT_DIR / "vocab.jsonl"
REF_CORPUS_DIR = INPUT_DIR / "corpus"
TRAIN_CSV = INPUT_DIR / "train.csv"

SEED = 20260611
TOKEN_LIMIT = 8192
# HARD conciseness caps on the TRAINED completion (the whole point of bit_v2):
TRACE_MAX_TOKENS = 4500
TRACE_P90_TARGET = 4000
TRACE_MEDIAN_TARGET = 3500

N_BITS = 8

# ---- chat template (gate-verified against reference bit entries) ----
CHAT_PRE = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
CHAT_POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n"

REF_BIT_IDS = ["00066667", "0031df9c", "004ef7c7"]

# ---- original prompt template (byte-exact, asserted against train.csv) ----
PROMPT_INTRO = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
    "\n\nHere are some examples of input -> output:\n"
)
PROMPT_QUERY = "\nNow, determine the output for: "


def build_problem_prompt(examples: list[tuple[str, str]], query: str) -> str:
    body = "".join(f"{i} -> {o}\n" for i, o in examples)
    return PROMPT_INTRO + body + PROMPT_QUERY + query


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
    """Interleaved masked/unmasked segments — identical logic to corpus.py."""
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


# ---------------- bit-rule candidate space ----------------

@dataclass(frozen=True)
class Candidate:
    op: str                 # C0,C1,ID,NOT,AND,OR,XOR,ANDN,ORN,XORN
    a: int | None = None    # first operand (input bit index, 0 = leftmost)
    b: int | None = None    # second operand
    table: int = 0          # 256-bit truth table; bit x = output for input value x
    rank: int = 0           # simplicity rank for choosing the displayed representative

    def name(self) -> str:
        if self.op == "C0":
            return "constant 0"
        if self.op == "C1":
            return "constant 1"
        if self.op == "ID":
            return f"in{self.a}"
        if self.op == "NOT":
            return f"NOT(in{self.a})"
        if self.op in ("AND", "OR", "XOR"):
            return f"{self.op}(in{self.a}, in{self.b})"
        base = {"ANDN": "AND", "ORN": "OR", "XORN": "XOR"}[self.op]
        return f"{base}(in{self.a}, NOT(in{self.b}))"

    def eval_bits(self, bits: list[int]) -> int:
        if self.op == "C0":
            return 0
        if self.op == "C1":
            return 1
        a = bits[self.a]
        if self.op == "ID":
            return a
        if self.op == "NOT":
            return 1 - a
        b = bits[self.b]
        if self.op == "AND":
            return a & b
        if self.op == "OR":
            return a | b
        if self.op == "XOR":
            return a ^ b
        nb = 1 - b
        if self.op == "ANDN":
            return a & nb
        if self.op == "ORN":
            return a | nb
        if self.op == "XORN":
            return a ^ nb
        raise ValueError(self.op)


NEGATED_OPS = {"ANDN", "ORN", "XORN"}
BASIC_OPS = {"C0", "C1", "ID", "NOT", "AND", "OR", "XOR"}


def value_bits(x: int) -> list[int]:
    """8-bit value -> bit list, index 0 = LEFTMOST displayed char."""
    return [(x >> (N_BITS - 1 - i)) & 1 for i in range(N_BITS)]


def bits_to_str(bits: list[int]) -> str:
    return "".join(str(b) for b in bits)


def str_to_bits(s: str) -> list[int]:
    assert re.fullmatch(r"[01]{8}", s), s
    return [int(c) for c in s]


def _make_table(cand: Candidate) -> int:
    t = 0
    for x in range(256):
        if cand.eval_bits(value_bits(x)):
            t |= 1 << x
    return t


def all_candidates() -> list[Candidate]:
    """Full menu as truth-tabled candidates, simplicity-ranked."""
    cands: list[Candidate] = []

    def add(op, a=None, b=None, rank=0):
        c = Candidate(op=op, a=a, b=b, rank=rank)
        cands.append(Candidate(op=op, a=a, b=b, table=_make_table(c), rank=rank))

    add("C0", rank=0)
    add("C1", rank=0)
    for a in range(N_BITS):
        add("ID", a, rank=1)
        add("NOT", a, rank=2)
    for a in range(N_BITS):
        for b in range(a + 1, N_BITS):
            add("AND", a, b, rank=3)
            add("OR", a, b, rank=3)
            add("XOR", a, b, rank=3)
            add("XORN", a, b, rank=5)  # symmetric: a^~b == b^~a
    for a in range(N_BITS):
        for b in range(N_BITS):
            if a != b:
                add("ANDN", a, b, rank=4)
                add("ORN", a, b, rank=4)
    return cands


_ALL: list[Candidate] | None = None
_BASIC_TABLES: frozenset[int] | None = None


def menu() -> list[Candidate]:
    global _ALL
    if _ALL is None:
        _ALL = all_candidates()
    return _ALL


def basic_tables() -> frozenset[int]:
    """Truth tables reachable WITHOUT negated binary ops (held-out structure test)."""
    global _BASIC_TABLES
    if _BASIC_TABLES is None:
        _BASIC_TABLES = frozenset(c.table for c in menu() if c.op in BASIC_OPS)
    return _BASIC_TABLES


def apply_rule(rule: list[Candidate], in_str: str) -> str:
    bits = str_to_bits(in_str)
    out = [c.eval_bits(bits) for c in rule]
    s = bits_to_str(out)
    assert re.fullmatch(r"[01]{8}", s)
    return s
