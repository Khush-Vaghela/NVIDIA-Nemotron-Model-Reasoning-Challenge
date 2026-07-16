# equation_v2 validation report

- family: equation_numeric_deduce (the `guess` sub-family is excluded -- its rule
  is not determinable from the prompt; even the official corpus traces miss it 87%).
- entries: 3650 (train 3400, held-out 250)
- held-out structure: the determinant / abs-determinant operations (entire cell;
  asserted absent from every train problem's target AND fitted representative op).
- COMPLETION token distribution: min 328, median 435, p90 502, max 780
  (caps: median<3000 p90<3000 max<4500; original corpus equation traces: median ~5,800 / max ~6,700)
- 100% of query-operator groups uniquely determine the gold (full 128-rule solver;
  gold invariant across the consistent set). 100% of golds produced by executing
  the solved rule, which reproduced every group example exactly.
- 100% byte-exact tokenize->detokenize round-trip (2 independent decoders).
- zero id leakage, zero structural leakage, no collision with original corpus.

| cell | split | n | comp med | comp p90 | comp max |
|---|---|---|---|---|---|
| arithmetic | train | 900 | 437 | 526 | 639 |
| concat | train | 300 | 449 | 481 | 504 |
| determinant | heldout | 250 | 437 | 509 | 606 |
| digit_agg | train | 600 | 423 | 481 | 682 |
| digitwise | train | 600 | 443 | 513 | 636 |
| modular | train | 500 | 439 | 524 | 780 |
| offsets | train | 500 | 420 | 468 | 514 |
