# M4 integrated comparison

Selected on validation: **behavior**.

## Training ablation (validation only)

| scenario | behavior | distill | epoch | selection_score | recent_50 | recent_2048 | semantic_50 | semantic_128 | prune_50 | merge_50 | hybrid_50 | training_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_only | False | False | 1 | 0.72405 | 0.72449 | 0.72383 | 0.72433 | 0.72463 | 0.72316 | 0.72472 | 0.72317 | 57.53777 |
| behavior | True | False | 1 | 0.72883 | 0.7253 | 0.72735 | 0.72771 | 0.72975 | 0.73157 | 0.72873 | 0.73143 | 67.6359 |
| behavior_distill | True | True | 1 | 0.72883 | 0.7253 | 0.72735 | 0.72771 | 0.72975 | 0.73157 | 0.72873 | 0.73143 | 78.89472 |

## Index ablation

| encoding | bytes | prepare_seconds |
| --- | --- | --- |
| float32 | 181629650 | 13.32971 |
| int8 | 153780948 | 13.95468 |

## HTTP policy comparison (test assessment)

| encoding | policy | auc | gauc | mean_read_tokens | mean_linear_macs | p95_ms | throughput_rps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32 | full | 0.707887 | 0.579579 | 562.832223 | 3560792.811774 | 2.590379 | 427.040329 |
| float32 | fixed50 | 0.7071 | 0.57778 | 48.838575 | 328800.751046 | 1.547817 | 694.854041 |
| float32 | random50 | 0.707801 | 0.577036 | 48.838575 | 328800.751046 | 2.015409 | 649.792848 |
| float32 | semantic50 | 0.706242 | 0.57132 | 48.838575 | 328800.751046 | 1.481912 | 714.477478 |
| float32 | adaptive | 0.703016 | 0.5737 | 15.897909 | 121669.842293 | 1.569142 | 679.277218 |
| float32 | prune | 0.702028 | 0.569631 | 31.573354 | 253014.994888 | 1.838858 | 566.268694 |
| float32 | merge | 0.70393 | 0.572367 | 31.573354 | 124022.965143 | 1.866396 | 558.179743 |
| float32 | adaptive_exact | 0.703016 | 0.5737 | 15.897909 | 121669.842293 | 1.455263 | 709.407014 |
| float32 | hybrid | 0.702018 | 0.570448 | 31.573354 | 203805.54361 | 2.124067 | 502.73957 |
| float32 | adaptive_eager | 0.703016 | 0.5737 | 562.832223 | 121669.842293 | 1.833541 | 645.603743 |
| float32 | adaptive_pipeline | 0.703016 | 0.5737 | 15.897909 | 121669.842293 | 1.731796 | 615.626156 |
| int8 | full | 0.707887 | 0.579579 | 562.832223 | 3560792.811774 | 2.576619 | 448.601895 |
| int8 | fixed50 | 0.7071 | 0.57778 | 48.838575 | 328800.751046 | 1.598162 | 690.848171 |
| int8 | random50 | 0.707801 | 0.577036 | 48.838575 | 328800.751046 | 1.709325 | 630.45486 |
| int8 | semantic50 | 0.70652 | 0.570926 | 48.838575 | 328800.751046 | 1.605406 | 670.628534 |
| int8 | adaptive | 0.703052 | 0.572284 | 15.897909 | 121669.842293 | 1.572406 | 673.580144 |
| int8 | prune | 0.702325 | 0.570066 | 31.573354 | 253014.994888 | 2.058806 | 522.457104 |
| int8 | merge | 0.703946 | 0.572194 | 31.573354 | 124026.861658 | 2.028344 | 521.973073 |
| int8 | adaptive_exact | 0.703052 | 0.572284 | 15.897909 | 121669.842293 | 1.569325 | 675.983873 |
| int8 | hybrid | 0.702312 | 0.57082 | 31.573354 | 203802.621224 | 2.589694 | 463.236947 |
| int8 | adaptive_eager | 0.703052 | 0.572284 | 562.832223 | 121669.842293 | 1.847575 | 639.358327 |
| int8 | adaptive_pipeline | 0.703052 | 0.572284 | 15.897909 | 121669.842293 | 1.698008 | 624.300481 |

## Why these scenarios

Residual-only isolates the new model capacity; behavior adds co-occurrence vectors; behavior+distill adds a stopped-gradient full-history teacher. FP32 vs INT8 changes summary encoding and recalibrates the selector, so policy quality differences are joint effects. Full/fixed/random/semantic/adaptive/prune/merge/hybrid compare history operators with one checkpoint. Adaptive exact checks Bloom parity.

## Acceptance targets

| gate | met |
| --- | --- |
| full_test | True |
| mean_read_tokens_le_50 | True |
| point_auc_margin | True |
| noninferiority_ci | True |
| adaptive_auc_ge_fixed50 | False |
| p95_within_20pct_fixed50 | True |
| bloom_prediction_parity | True |

Execution PASS does not mean research targets passed. Timing is loopback single-request; linear MACs and logical bytes are partial cost accounting. Test has been previously inspected; confirmatory claims require a new independent holdout.

## Shared weights (matched architecture)

| training | k | auc | gauc | delta_auc_vs_specialist | ci95_vs_specialist |
| --- | --- | --- | --- | --- | --- |
| shared | 50 | 0.7071 | 0.57778 | 0.0015 | [-0.002191160283454477, 0.005781836558798064] |
| shared | 2048 | 0.70789 | 0.57958 | -0.00183 | [-0.0061938419110766476, 0.002223496413078534] |
| fixed_50 | 50 | 0.7056 | 0.57761 | 0.0 | [0.0, 0.0] |
| fixed_50 | 2048 | 0.70685 | 0.57962 | -0.00287 | [-0.006944336675611162, 0.0009056561133447385] |
| fixed_2048 | 50 | 0.69773 | 0.54635 | -0.00787 | [-0.013293311855105072, -0.002932966902374689] |
| fixed_2048 | 2048 | 0.70972 | 0.58323 | 0.0 | [0.0, 0.0] |

| strategy | checkpoints | checkpoint_bytes | tensor_bytes_if_all_resident | training_seconds |
| --- | --- | --- | --- | --- |
| independent | 2 | 487990584 | 487976232 | 91.3858 |
| full_only | 1 | 243995317 | 243988116 | 60.6375 |
| shared | 1 | 243995459 | 243988116 | 67.6359 |

Independent models specialize per budget; full-only tests truncation without variable-budget training; shared tests one checkpoint trained across budgets. Tensor footprint is not process RAM.

## Bloom utility (actual local storage)

| missing_fraction | mode | mean_exact_lookups | p95_local_ms | false_positives | filter_bytes | serial_rtt_2ms_mean_estimate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | indexed_exact | 1.0 | 0.07179 | 0 | 19366 | 2.06748 |
| 0.0 | indexed_bloom | 1.0 | 0.07321 | 0 | 19366 | 2.06915 |
| 0.0 | sharded_exact | 8.0 | 0.54054 | 0 | 19366 | 16.52377 |
| 0.0 | sharded_bloom | 1.02 | 0.08813 | 8 | 19366 | 2.11854 |
| 0.1 | indexed_exact | 1.0 | 0.08969 | 0 | 19366 | 2.07879 |
| 0.1 | indexed_bloom | 0.9 | 0.0908 | 0 | 19366 | 1.87218 |
| 0.1 | sharded_exact | 8.0 | 0.61729 | 0 | 19366 | 16.56197 |
| 0.1 | sharded_bloom | 0.93 | 0.08312 | 12 | 19366 | 1.93209 |
| 0.5 | indexed_exact | 1.0 | 0.07046 | 0 | 19366 | 2.06717 |
| 0.5 | indexed_bloom | 0.5025 | 0.07204 | 1 | 19366 | 1.04097 |
| 0.5 | sharded_exact | 8.0 | 0.53354 | 0 | 19366 | 16.52266 |
| 0.5 | sharded_bloom | 0.525 | 0.08286 | 10 | 19366 | 1.09548 |
| 0.9 | indexed_exact | 1.0 | 0.07046 | 0 | 19366 | 2.06683 |
| 0.9 | indexed_bloom | 0.1025 | 0.07122 | 1 | 19366 | 0.21394 |
| 0.9 | sharded_exact | 8.0 | 0.6359 | 0 | 19366 | 16.58205 |
| 0.9 | sharded_bloom | 0.115 | 0.081 | 6 | 19366 | 0.24718 |
| 1.0 | indexed_exact | 1.0 | 0.06925 | 0 | 19366 | 2.06484 |
| 1.0 | indexed_bloom | 0.0025 | 0.002 | 1 | 19366 | 0.00691 |
| 1.0 | sharded_exact | 8.0 | 0.53154 | 0 | 19366 | 16.51983 |
| 1.0 | sharded_bloom | 0.0125 | 0.00959 | 5 | 19366 | 0.03489 |

Known-shard routing and fan-out are different workloads. Remote RTT estimates assume serial lookups; they are sensitivity models, not measured cloud latency.

## Lazy retrieval and selector optimization

| policy | auc | mean_read_tokens | mean_k | mean_selector_ms | cpu_seconds | p95_ms |
| --- | --- | --- | --- | --- | --- | --- |
| adaptive | 0.703052 | 15.897909 | 15.897909 | 0.055217 | 6.194552 | 1.572406 |
| adaptive_eager | 0.703052 | 562.832223 | 15.897909 | 0.055358 | 6.731587 | 1.847575 |
| adaptive_pipeline | 0.703052 | 15.897909 | 15.897909 | 0.158412 | 6.935712 | 1.698008 |

Eager reads all tokens before selecting the same positions; pipeline uses the original sklearn selector instead of fused linear inference. Both must match adaptive predictions. CPU time includes client and server in one process. Peak RSS is process lifetime high-water mark, not per-policy memory.
