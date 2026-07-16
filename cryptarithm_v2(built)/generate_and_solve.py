"""Generate-then-verify cryptarithm problems with code-verified, uniquely-determined golds.

Two regimes (see common.py):
  * POSITIONAL: pick one of the 4 distinct symbol-rearrangement functions
    {AB, BA, rev(A)rev(B), rev(B)rev(A)}, render examples + query directly on symbols.
  * ARITHMETIC: pick a small symbol->digit cipher (NS symbols) + an arithmetic op + action,
    render examples by decoding -> computing -> re-encoding.

UNIQUENESS (the core check): a problem is kept only if NO alternative rule -- across the
positional menu AND the arithmetic menu (all ops x actions x injective cipher) -- reproduces
every example yet yields a DIFFERENT query result. The search is node-capped and early-exits
on the first conflicting gold; if the cap is hit before uniqueness is proven, the problem is
DISCARDED (the gold is always correct by construction, so discarding only affects yield).
Examples are added one at a time until uniqueness is proven or the example cap is reached.

Output: OUT_DIR/problems_all.jsonl with fields {id, prompt, answer, ...metadata}.
Usage:  python generate_and_solve.py   (gate enforced)   [chunk env: CR_CELL/CR_START/CR_COUNT/CR_MERGE]
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter

from common import (
    OUT_DIR,
    SEED,
    SYMBOL_POOL,
    PosRule,
    build_problem_prompt,
    pos_render,
)

gate = OUT_DIR / "tokenizer_gate.json"
if not gate.exists() or not json.load(open(gate)).get("passed"):
    sys.exit("Tokenizer gate not passed. Run verify_tokenizer.py first.")

# ---- the 4 distinct positional functions (canonical representatives) ----
POS4 = {
    "AB": PosRule("concat", False, False),       # A . B
    "BA": PosRule("rconcat", False, False),      # B . A
    "rArB": PosRule("concat", True, False),      # rev(A) . rev(B)
    "rBrA": PosRule("rconcat", True, False),     # rev(B) . rev(A)
}
POS_RULES = list(POS4.values())

# ---- arithmetic op menu ----
ARITH_OPS = {
    "addition": lambda a, b: a + b,
    "subtraction": lambda a, b: a - b,
    "reverse subtraction": lambda a, b: b - a,
    "absolute difference": lambda a, b: abs(a - b),
    "multiplication": lambda a, b: a * b,
    "modulo": lambda a, b: (a % b) if b else None,
    "reverse modulo": lambda a, b: (b % a) if a else None,
    "integer division": lambda a, b: (a // b) if b else None,
}
ACTIONS = [(False, False), (False, True), (True, False), (True, True)]
NODECAP = 60000


def arith_render(op, orv, rrv, A, B, s2d, d2s):
    a = A[::-1] if orv else A
    b = B[::-1] if orv else B
    if a[0] not in s2d or a[1] not in s2d or b[0] not in s2d or b[1] not in s2d:
        return None
    ia = s2d[a[0]] * 10 + s2d[a[1]]
    ib = s2d[b[0]] * 10 + s2d[b[1]]
    v = ARITH_OPS[op](ia, ib)
    if v is None or v < 0:
        return None
    s = str(v)
    if rrv:
        s = s[::-1]
    if any(int(c) not in d2s for c in s):
        return None
    return "".join(d2s[int(c)] for c in s)


# ---------------- uniqueness verification ----------------
def _arith_has_conflict(grp, q, truegold, nodes):
    """True if some (arith op, action, injective map) reproduces grp but yields a query
    gold != truegold. Sets nodes[1]=True if node cap exceeded (=> caller discards)."""
    qa, qop, qb = q
    order = []
    for (a, b, r) in grp:
        for ch in a + b + r:
            if ch not in order:
                order.append(ch)
    for ch in qa + qb:
        if ch not in order:
            order.append(ch)
    syms = order
    n = len(syms)
    idx = {s: i for i, s in enumerate(syms)}
    if n > 10:
        nodes[1] = True
        return False
    for op in ARITH_OPS:
        f = ARITH_OPS[op]
        for orv, rrv in ACTIONS:
            ex = []
            for (a, b, r) in grp:
                a2 = a[::-1] if orv else a
                b2 = b[::-1] if orv else b
                need = set(idx[c] for c in a2 + b2 + r)
                ex.append((a2, b2, r, need, max(need)))
            assign = [-1] * n
            used = [False] * 10
            conflict = [False]

            def checkex(e):
                a2, b2, r, need, mx = e
                ia = assign[idx[a2[0]]] * 10 + assign[idx[a2[1]]]
                ib = assign[idx[b2[0]]] * 10 + assign[idx[b2[1]]]
                v = f(ia, ib)
                if v is None or v < 0:
                    return False
                s = str(v)
                if rrv:
                    s = s[::-1]
                if len(s) != len(r):
                    return False
                for ch, dg in zip(r, s):
                    if assign[idx[ch]] != int(dg):
                        return False
                return True

            def bt(i):
                if conflict[0]:
                    return
                nodes[0] += 1
                if nodes[0] > NODECAP:
                    nodes[1] = True
                    return "CAP"
                if i == n:
                    a2 = qa[::-1] if orv else qa
                    b2 = qb[::-1] if orv else qb
                    ia = assign[idx[a2[0]]] * 10 + assign[idx[a2[1]]]
                    ib = assign[idx[b2[0]]] * 10 + assign[idx[b2[1]]]
                    v = f(ia, ib)
                    if v is None or v < 0:
                        return
                    s = str(v)
                    if rrv:
                        s = s[::-1]
                    inv = {assign[j]: syms[j] for j in range(n)}
                    if any(int(c) not in inv for c in s):
                        return
                    g = "".join(inv[int(c)] for c in s)
                    if g != truegold:
                        conflict[0] = True
                    return
                for d in range(10):
                    if used[d]:
                        continue
                    assign[i] = d
                    used[d] = True
                    okp = True
                    for e in ex:
                        if e[4] == i and max(e[3]) <= i:
                            if not checkex(e):
                                okp = False
                                break
                    rv = bt(i + 1) if okp else None
                    used[d] = False
                    assign[i] = -1
                    if rv == "CAP":
                        return "CAP"
                    if conflict[0]:
                        return

            rv = bt(0)
            if rv == "CAP":
                return False
            if conflict[0]:
                return True
    return False


def _pos_conflict(grp, q, truegold):
    """True if some positional rule reproduces grp but yields a query gold != truegold."""
    qa, qop, qb = q
    for r in POS_RULES:
        if all(pos_render(r, a, b) == res for (a, b, res) in grp):
            if pos_render(r, qa, qb) != truegold:
                return True
    return False


def is_unique(grp, q, truegold):
    """Return True (proven unique), False (a conflicting alternative exists), or
    None (node cap hit -> discard)."""
    if _pos_conflict(grp, q, truegold):
        return False
    nodes = [0, False]
    conflict = _arith_has_conflict(grp, q, truegold, nodes)
    if conflict:
        return False
    if nodes[1]:
        return None
    return True


# ---------------- generation ----------------
def _rng(cell, seq):
    return random.Random(f"{SEED}:{cell}:{seq}")


def _rand_operand(rng, syms):
    return rng.choice(syms) + rng.choice(syms)


def gen_positional(rng, target: PosRule, sym_choices):
    """Build a positional problem group (query-operator examples) uniquely determined."""
    syms = sym_choices
    used = set()
    grp = []

    def rp():
        return _rand_operand(rng, syms), _rand_operand(rng, syms)

    # query
    qa, qb = rp()
    while True:
        a, b = rp()
        if (a, b) in used:
            continue
        used.add((a, b))
        grp.append((a, b, pos_render(target, a, b)))
        if len(grp) < 2:
            continue
        tg = pos_render(target, qa, qb)
        u = is_unique(grp, (qa, "#", qb), tg)
        if u is True:
            return grp, (qa, qb), tg
        if u is None or len(grp) >= 6:
            return None
    return None


def gen_arith(rng, op, sym_choices):
    orv, rrv = rng.choice(ACTIONS)
    ns = len(sym_choices)
    digs = rng.sample(range(10), ns)
    s2d = {s: digs[i] for i, s in enumerate(sym_choices)}
    d2s = {d: s for s, d in s2d.items()}
    syms = sym_choices

    def rp():
        return _rand_operand(rng, syms), _rand_operand(rng, syms)

    qa = qb = None
    for _ in range(60):
        a, b = rp()
        if arith_render(op, orv, rrv, a, b, s2d, d2s) is not None:
            qa, qb = a, b
            break
    if qa is None:
        return None
    tg = arith_render(op, orv, rrv, qa, qb, s2d, d2s)
    used = {(qa, qb)}
    grp = []
    while len(grp) < 10:
        a, b = rp()
        if (a, b) in used:
            continue
        r = arith_render(op, orv, rrv, a, b, s2d, d2s)
        if r is None:
            continue
        used.add((a, b))
        grp.append((a, b, r))
        if len(grp) < 3:
            continue
        u = is_unique(grp, (qa, "#", qb), tg)
        if u is True:
            return grp, (qa, qb), tg, {"op": op, "orv": orv, "rrv": rrv,
                                       "s2d": s2d}
        if u is None:
            return None
    return None


# distractor operators (positional rules) for realism in positional problems
def _distractors(rng, qsym, syms):
    k = rng.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
    out = []
    pool = [s for s in SYMBOL_POOL if s != qsym]
    rng.shuffle(pool)
    for i in range(k):
        if i >= len(pool):
            break
        osym = pool[i]
        rule = rng.choice(POS_RULES)
        m = rng.choice([1, 2])
        seen = set()
        for _ in range(40):
            a = _rand_operand(rng, syms)
            b = _rand_operand(rng, syms)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, osym, b, pos_render(rule, a, b)))
            if len([x for x in out if x[1] == osym]) >= m:
                break
    return out


def make_problem(spec, seq):
    rng = _rng(spec["cell"], seq)
    for _attempt in range(200):
        # cipher/operand symbol set
        if spec["regime"] == "pos":
            ns = rng.choice([6, 7, 8])
            syms = rng.sample(SYMBOL_POOL, ns)
            qsym = rng.choice([s for s in SYMBOL_POOL if s not in syms] or SYMBOL_POOL)
            target = POS4[spec["pos_class"]]
            built = gen_positional(rng, target, syms)
            if built is None:
                continue
            grp, (qa, qb), gold = built
            q_examples = [(a, qsym, b, r) for (a, b, r) in grp]
            distract = _distractors(rng, qsym, syms)
            examples = q_examples + distract
            rng.shuffle(examples)
            meta = dict(regime="pos", pos_class=spec["pos_class"],
                        rule_label=target.label())
        else:
            ns = rng.choice([5, 6])
            syms = rng.sample(SYMBOL_POOL, ns)
            qsym = rng.choice([s for s in SYMBOL_POOL if s not in syms] or SYMBOL_POOL)
            built = gen_arith(rng, spec["op"], syms)
            if built is None:
                continue
            grp, (qa, qb), gold, info = built
            q_examples = [(a, qsym, b, r) for (a, b, r) in grp]
            examples = list(q_examples)
            rng.shuffle(examples)
            meta = dict(regime="arith", op=info["op"], orv=info["orv"],
                        rrv=info["rrv"],
                        cipher={k: info["s2d"][k] for k in info["s2d"]},
                        rule_label=f"{info['op']}"
                        + (" [reversed operands]" if info["orv"] else "")
                        + (" [reversed result]" if info["rrv"] else ""))
        prompt = build_problem_prompt(examples, (qa, qsym, qb))
        pid = hashlib.sha256(f"crypt_v2:{spec['cell']}:{seq}".encode()).hexdigest()[:8]
        rec = dict(
            id=pid, _seq=seq, cell=spec["cell"], heldout=bool(spec.get("heldout")),
            op_symbol=qsym, query=[qa, qsym, qb], group=[list(g) for g in grp],
            examples=examples, n_examples=len(examples), n_group=len(grp),
            answer=gold, prompt=prompt, **meta,
        )
        return rec
    return None  # exhausted attempts -> caller skips (yield management)


# ---------------- cells ----------------
CELLS = [
    dict(cell="pos_AB", regime="pos", pos_class="AB", n=650),
    dict(cell="pos_BA", regime="pos", pos_class="BA", n=550),
    dict(cell="pos_rArB", regime="pos", pos_class="rArB", n=500),
    dict(cell="pos_rBrA", regime="pos", pos_class="rBrA", n=200, heldout=True),
    dict(cell="arith_main", regime="arith", op="MAIN", n=220),
    dict(cell="arith_mul", regime="arith", op="multiplication", n=70, heldout=True),
]
MAIN_OPS = ["addition", "subtraction", "reverse subtraction", "absolute difference",
            "modulo", "reverse modulo", "integer division"]


def _resolve_op(spec, seq):
    if spec["regime"] == "arith" and spec["op"] == "MAIN":
        return MAIN_OPS[seq % len(MAIN_OPS)]
    return spec.get("op")


def _emit_chunk(spec, start, count):
    parts = OUT_DIR / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    end = min(start + count, spec["n"])
    path = parts / f"{spec['cell']}_{start:05d}.jsonl"
    made = 0
    seq = start
    target = end - start
    # over-scan seqs to fill `target` accepted problems (arith discards some)
    scan = start
    with open(path, "w") as f:
        while made < target and scan < spec["n"] + 5000:
            s = dict(spec)
            s["op"] = _resolve_op(spec, scan)
            rec = make_problem(s, scan)
            scan += 1
            if rec is None:
                continue
            rec["_seq"] = start + made
            rec["id"] = hashlib.sha256(
                f"crypt_v2:{spec['cell']}:{start+made}".encode()).hexdigest()[:8]
            f.write(json.dumps(rec) + "\n")
            made += 1
    print(f"  part {path.name}: produced {made}/{target} (scanned {scan-start})")


def _merge():
    problems = []
    seen = set()
    for spec in CELLS:
        recs = {}
        for path in sorted((OUT_DIR / "parts").glob(f"{spec['cell']}_*.jsonl")):
            for line in open(path):
                p = json.loads(line)
                recs[p["_seq"]] = p
        got = sorted(recs)
        ho = " [HELD-OUT]" if spec.get("heldout") else ""
        excnt = Counter(p["n_examples"] for p in recs.values())
        print(f"  {spec['cell']:<11} got {len(got)}/{spec['n']}{ho} examples={dict(sorted(excnt.items()))}")
        for seq in got:
            p = recs[seq]
            if p["id"] in seen:
                continue
            seen.add(p["id"])
            problems.append(p)
    with open(OUT_DIR / "problems_all.jsonl", "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    nho = sum(p["heldout"] for p in problems)
    npos = sum(p["regime"] == "pos" for p in problems)
    print(f"\nWrote {len(problems)} problems ({len(problems)-nho} train, {nho} held-out; "
          f"{npos} positional, {len(problems)-npos} arithmetic)")
    print("All golds code-computed; all kept problems proven uniquely determined "
          "(no alternative positional/arithmetic rule fits with a different gold).")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CR_MERGE") == "1":
        _merge()
        return
    cell_env = os.environ.get("CR_CELL")
    if cell_env:
        spec = next(c for c in CELLS if c["cell"] == cell_env)
        _emit_chunk(spec, int(os.environ.get("CR_START", "0")),
                    int(os.environ.get("CR_COUNT", "100000")))
        return
    for spec in CELLS:
        _emit_chunk(spec, 0, spec["n"])
    _merge()


if __name__ == "__main__":
    main()
