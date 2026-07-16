# bit_v2 validation report

- entries: 3100 (train 2800, held-out 300)
- held-out structure: rules using AND-NOT / OR-NOT / XOR-NOT (entire cell;
  truth-table-verified absent from every train problem)
- COMPLETION token distribution: min 771, median 1118, p90 1241, max 1334
  (caps: median<3500 p90<4000 max<4500; original corpus bit traces: median ~6,735 / p90 ~7,200)
- 100% rules uniquely determined by their examples (full-menu solver);
  100% golds produced by executing the solved rule, which reproduced every
  example pair exactly; gold asserted = 8 binary chars (strict grader).
- 100% byte-exact tokenize->detokenize round-trip (2 independent decoders).
- zero id leakage, zero structural leakage, no collision with original corpus.

| cell | split | n | comp med | comp p90 | comp max |
|---|---|---|---|---|---|
| basic_mixed | train | 2000 | 1148 | 1241 | 1316 |
| basic_simple | train | 800 | 961 | 1014 | 1050 |
| negated_ops | heldout | 300 | 1212 | 1293 | 1334 |
