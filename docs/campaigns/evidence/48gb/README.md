# Selected 48 GB evidence reference

Selected evidence extracts supporting the [48 GB benchmark dossiers](../../48gb-v032.md).

[Diagnostic values](48gb-diagnostic-values.json) add the prefix-control matrices, comparator anomaly and Gemma results. Their DC identifiers are separate from the main C identifiers. Aggregate next-token agreement counts name their calculation; these are derived from recorded boolean outcomes, not new experiments.

[Selected values](48gb-selected-values.json) contain scalar outcomes and minimal configuration metadata. They exclude raw evaluation text, generated responses, logs, machine paths, GPU UUIDs, credentials and process state. Failed criteria are retained beside passing measurements. This extract does not provide a runnable harness or model weights.

| Dossier section | Claim references | Coverage |
|---|---|---|
| Native dense comparison | C001–C017 | Four configurations; retained second-pass greedy throughput, reasoning and tested target-only prefix parity |
| Native dense prefix failures | C018–C027 | W4A16 failure and native NVFP4 failed extended repeat |
| Dense YaRN exploration (Dossier 3) | C028–C040, C065–C066 | Near-max counts, cache state, missing expected strings, streamed TTFT; separate 512K prefix-ON parity failure |
| Retained native MoE | C041–C052 | Throughput, reasoning, parity, near-max counts/cache/time and comparator-specific KL |
| Filtered checkpoint comparison (Dossier 2) | C053–C064 | Retained-pass throughput and reasoning, including failed QUASAR gate |
| Occupied context and whole-request timing | C067–C072 | Native dense prompt/output/total; YaRN totals and 1M request duration |

Each claim names a source ID and JSON pointer. Source entries supply the original file's SHA256; the original bytes remain private. Configuration entries identify run keys and selected settings. Raw fields such as `draft_tokens` can remain populated with speculation disabled and are then inapplicable.

The `source_branch`, `source_commit`, and `source_sha256` fields record provenance for the private archive from which these scalar extracts were generated. Six native dense records (`Q01`–`Q04`, `F01`, `F02`) were extracted from a consolidated run archive whose historical per-run artifact paths were no longer present on disk.

Dossier 5 (prefix reproducibility) and Dossier 6 (Gemma diagnostics) use [`48gb-diagnostic-values.json`](48gb-diagnostic-values.json): `DC001`–`DC040` cover free-generation and fixed-continuation controls, `DC041`–`DC067` cover the nine-arm isolation matrix, `DC068`–`DC087` cover reference-comparator differences, `DC088`–`DC096` cover Gemma extended results, and `DC097`–`DC108` cover the 15-case chat comparison. [`Expanded dossier matrices`](48gb-dossier-matrix.json) map the remaining native dense and MoE sizing, perplexity, distribution, and timing fields used in the summary tables.
