# Dense Qwen: recorded check outcomes

This table covers the included initial 19–20 October dense-run JSON, not the later dense, MoE, or Ada campaigns. Recorded status labels below come from the artifact files and reflect the per-run harness gates rather than standing qualification (for example, the W4A16/DFlash2 FP8-KV configuration scored 188/200 despite a legacy PASS label; see the [24 GB dense dossiers](campaigns/24gb-v032.md) for numerical scores).

| Experiment | KV precision | PASS | FAILED | Other | Failed checks |
|---|---|---:|---:|---:|---|
| [W4A16 MTP-4: BF16 versus NVFP4 KV](../nvfp4/results/v032-sm120-controls-20261019-222235/result.json) | BF16 KV | 12 | 1 | 0 | 64K prefill |
| [W4A16 MTP-4: BF16 versus NVFP4 KV](../nvfp4/results/v032-sm120-controls-20261019-222235/result.json) | NVFP4 KV | 12 | 1 | 0 | 64K prefill |
| [DFlash2 with prefix caching: failed startup](../nvfp4/results/v032-sm120-dflash-prefix-cache-20261020-172508/result.json) | NVFP4 KV | 0 | 1 | 0 | Engine startup and admission |
| [DFlash2 with prefix caching: admission-only retry](../nvfp4/results/v032-sm120-dflash-prefix-cache-20261020-173050/result.json) | NVFP4 KV | 1 | 0 | 0 | — |
| [DFlash2 with prefix caching: 16K reuse and parity](../nvfp4/results/v032-sm120-dflash-prefix-cache16k-20261020-173648/result.json) | NVFP4 KV | 3 | 1 | 0 | Appended-turn cache reuse |
| [W4A16 DFlash2-7: KV precision comparison](../nvfp4/results/v032-sm120-dflash7-20261020-081728/result.json) | BF16 KV | 13 | 0 | 0 | — |
| [W4A16 DFlash2-7: KV precision comparison](../nvfp4/results/v032-sm120-dflash7-20261020-081728/result.json) | FP8 KV | 13 | 0 | 0 | — |
| [W4A16 DFlash2-7: KV precision comparison](../nvfp4/results/v032-sm120-dflash7-20261020-081728/result.json) | NVFP4 KV | 13 | 0 | 0 | — |
| [W4A16 MTP-4: FP8 KV](../nvfp4/results/v032-sm120-fp8-20261019-235735/result.json) | FP8 KV | 15 | 0 | 0 | — |
| [NVFP4 weights with DFlash2-7: KV precision comparison](../nvfp4/results/v032-sm120-nvfp4-dflash7-20261020-101407/result.json) | BF16 KV | 13 | 0 | 0 | — |
| [NVFP4 weights with DFlash2-7: KV precision comparison](../nvfp4/results/v032-sm120-nvfp4-dflash7-20261020-101407/result.json) | FP8 KV | 13 | 0 | 0 | — |
| [NVFP4 weights with DFlash2-7: KV precision comparison](../nvfp4/results/v032-sm120-nvfp4-dflash7-20261020-101407/result.json) | NVFP4 KV | 13 | 0 | 0 | — |
| [NVFP4 weights: target-only baseline](../nvfp4/results/v032-sm120-nvfp4-specoff-20261020-124254/result.json) | BF16 KV | 3 | 1 | 0 | GSM8K greedy (200 questions) |
| [NVFP4 weights: target-only baseline](../nvfp4/results/v032-sm120-nvfp4-specoff-20261020-124254/result.json) | FP8 KV | 3 | 1 | 0 | GSM8K greedy (200 questions) |
| [NVFP4 weights: target-only baseline](../nvfp4/results/v032-sm120-nvfp4-specoff-20261020-124254/result.json) | NVFP4 KV | 18 | 1 | 0 | Three-fact QA at 64K |
| [Target-only: prefix-cache correctness](../nvfp4/results/v032-sm120-prefix-cache-20261020-170341/result.json) | BF16 KV | 6 | 0 | 0 | — |
| [Target-only: prefix-cache correctness](../nvfp4/results/v032-sm120-prefix-cache-20261020-170341/result.json) | NVFP4 KV | 6 | 0 | 0 | — |
| [350K YaRN: initial reasoning gate](../nvfp4/results/v032-sm120-yarn350k-20261020-174357/result.json) | NVFP4 KV | 3 | 1 | 0 | GSM8K greedy (200 questions) |
