# cryptarithm_v2 validation report

- family: cryptarithm_deduce (the `guess` sub-family is excluded -- query operator
  unseen in examples => not determinable; official guess traces are 7% correct).
- entries: 2190 (train 1920, held-out 270); regimes: {'arith': 290, 'pos': 1900}.
- two regimes: POSITIONAL (symbol rearrangement, mapping-free) and ARITHMETIC
  (small symbol->digit cipher + arithmetic op), all golds code-computed.
- EVERY kept problem proven uniquely determined: no alternative rule (positional
  menu + arithmetic ops x actions x injective cipher) reproduces all examples with
  a different query result (node-capped exhaustive search; non-unique discarded).
- COMPLETION token distribution: min 228, median 288, p90 311, max 515
  (caps: median<3500 p90<3500 max<4500; base model on this family ~7,610 tokens, 98.8% truncated).
- 100% byte-exact tokenize->detokenize round-trip (2 independent decoders).
- held-out: pos_rBrA (rev(B).rev(A) never an answer in train) + arith_mul
  (multiplication never named/used in any train trace); zero id/structural leakage.

| cell | regime | split | n | comp med | comp p90 | comp max |
|---|---|---|---|---|---|---|
| arith_main | arith | train | 220 | 287 | 390 | 515 |
| arith_mul | arith | heldout | 70 | 267 | 290 | 344 |
| pos_AB | pos | train | 650 | 284 | 303 | 327 |
| pos_BA | pos | train | 550 | 285 | 305 | 330 |
| pos_rArB | pos | train | 500 | 296 | 314 | 337 |
| pos_rBrA | pos | heldout | 200 | 296 | 315 | 334 |
