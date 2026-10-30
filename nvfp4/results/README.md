# Dense Qwen measurement files

These records support the [24 GB dense Qwen report](../../docs/campaigns/24gb-v032.md). Choose a comparison by its question rather than by the directory name. Each link opens its recorded JSON outcomes; directories retain the identifiers used when the experiments ran.

| Experiment | What the record helps assess | Results |
|---|---|---|
| W4A16 MTP-4: BF16 versus NVFP4 KV | Do smaller KV elements increase capacity without a large measured quality change? | [Result JSON](v032-sm120-controls-20261019-222235/result.json) |
| W4A16 MTP-4: FP8 KV | How does FP8 compare with the two control precisions? | [Result JSON](v032-sm120-fp8-20261019-235735/result.json) |
| W4A16 DFlash2-7: KV precision comparison | How do KV precision, speed, admission, and reasoning scores change with external drafting? | [Result JSON](v032-sm120-dflash7-20261020-081728/result.json) |
| NVFP4 weights with DFlash2-7: KV precision comparison | What changes when the target uses native NVFP4 weights? | [Result JSON](v032-sm120-nvfp4-dflash7-20261020-101407/result.json) |
| NVFP4 weights: target-only baseline | What speed and quality does the same weight preparation achieve without drafting? | [Result JSON](v032-sm120-nvfp4-specoff-20261020-124254/result.json) |
| Target-only: prefix-cache correctness | Does a repeated prefix reuse cache, preserve output, and invalidate correctly after mutation? | [Result JSON](v032-sm120-prefix-cache-20261020-170341/result.json) |
| DFlash2 with prefix caching: failed startup | Can the attempted memory configuration start? | [Result JSON](v032-sm120-dflash-prefix-cache-20261020-172508/result.json) |
| DFlash2 with prefix caching: admission-only retry | Does revised memory sizing start, and what does it admit? | [Result JSON](v032-sm120-dflash-prefix-cache-20261020-173050/result.json) |
| DFlash2 with prefix caching: 16K reuse and parity | Does the admitted configuration preserve cached prefixes and cold/warm output? | [Result JSON](v032-sm120-dflash-prefix-cache16k-20261020-173648/result.json) |
| 350K YaRN: initial reasoning gate | Does the initial 768-token reasoning allowance clear the gate before extended tests? | [Result JSON](v032-sm120-yarn350k-20261020-174357/result.json) |

A campaign-level `result.json` groups configurations; each precision subdirectory contains its own `result.json`. Records may also include frozen inputs, driver code, logs, and memory samples. Original PASS labels describe the recorded checks; numerical gate failures and cold/warm output mismatches are explained in the report and [check summary](../../docs/dense-checks.md).

The [later MTP/context tests](../../docs/campaigns/24gb-v032.md), [MoE comparisons](../../docs/campaigns/moe-v032.md), and [Ada tests](../../docs/campaigns/ada16-v032.md) have detailed reports but no raw run directories here.
