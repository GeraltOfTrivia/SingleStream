# SingleStream

Single-user, single-stream local LLM serving on workstation GPUs. These vLLM 0.32 experiments compare weight/KV precision, speculative decoding, prefix caching, and context extension with **one active request at a time**.

## Best measured configurations

PP is **cold prefill**, with the prompt length shown; cache hits are excluded. TG is **greedy output throughput**, including request time; decode-only rates are shown where available. Each column can describe a different configuration. Context entries distinguish tested occupation/retrieval from admission alone.

### RTX PRO 4000 Blackwell SFF — 24 GB, 70 W

| Model | Highest PP | Highest TG | Largest demonstrated context |
|---|---|---|---|
| **Qwen3.8-27B dense** | **3,120 tok/s at 1K** — NVFP4 weights/KV, target-only | **79.49 tok/s output / 81.23 decode** — W4A16 weights, NVFP4 KV, DFlash2-7 | **350K YaRN** — W4A16, NVFP4 KV, MTP-3; **349,760 occupied**, needle/QA probes pass |
| **Qwen3.6-35B-A3B MoE** | **5,802 tok/s at 1K** — AWQ weights, NVFP4 KV, MTP-3 | **202.04 tok/s output / 206.61 decode** — same configuration | **320K YaRN** — MTP-3 with prefix caching; **318,057 occupied**, needle/QA probes pass |
| **Gemma-4-26B-A4B MoE** | **5,244 tok/s at 16K** — NVFP4 weights/KV, target-only | **57.71 tok/s** — external assistant k=1; target-only is **57.50**, with little practical gain | **262K native** — target-only; **260,666 occupied**, needle/QA probes pass |

Detailed benchmarks: [dense Qwen dossiers](docs/campaigns/24gb-v032.md) · [Qwen/Gemma MoE dossiers](docs/campaigns/moe-v032.md).

### RTX 2000 Ada Generation — 16 GB, 70 W

| Model | Highest PP | Highest TG | Context result |
|---|---|---|---|
| **Qwen3.8-27B dense** | **672 tok/s at 1K** — W4A16, NVFP4 KV, MTP-3, embedding UVA offload | **33.92 tok/s** — same configuration | **41,216 admitted**; near-limit retrieval not evaluated |
| **Gemma-4 12B dense** | **1,514 tok/s at 1K** — QUASAR W4A16-G64, NVFP4 KV, target-only | **49.38 tok/s** — BF16 assistant k=3 speed screen; **49.29** in the extended run | **130,796 occupied** in the target-only ladder; **261K retrieval fails** despite 262K admission |

Detailed benchmarks: [Ada dossiers](docs/campaigns/ada16-v032.md).

### RTX PRO 5000 Blackwell — 48 GB, 300 W

| Model | Selected cold PP | Selected TG | Context result |
|---|---|---|---|
| **Qwen3.8-27B dense** | **11,798 tok/s at 1K** — NVFP4 weights, BF16 KV, target-only | **192.58 tok/s output / 197.63 decode** — W4A16 weights, NVFP4 KV, DFlash2-7, prefix OFF; qualification passed | **1M YaRN** — NVFP4 weights/KV, DFlash2-7, prefix OFF; **1,047,232 occupied**, expected retrieval/QA strings present; filtered exploration, not full qualification |
| **Qwen3.6-35B-A3B MoE** | **22,439 tok/s at 1K** — mixed FP8/NVFP4 weights, NVFP4 KV, MTP-3 | **336.49 tok/s output / 342.47 decode** — same configuration, prefix ON, GPU vision; **exact prefix parity fails** | **262K native** — **260,789 occupied**, expected retrieval/QA strings present; partially cached request |
| **Gemma-4-31B dense** | **8,908 tok/s at 1K** — NVFP4 weights, FP8 KV, assistant k=3 | **128.03 tok/s** — same configuration, prefix ON, GPU vision; diagnostic with **KL and prefix-parity failures** | **262K native** — **260,746 occupied**, expected retrieval/QA strings present; diagnostic, not qualification |

These are selected configurations, not an exhaustive speed ranking. The 1M dense probe had **2,515.29 s cold TTFT** and generated 182 tokens. MoE scored **192/200** using an FP8 distribution reference after an unresolved BF16-reference discrepancy; Gemma scored **196/200**, but neither extended configuration is a clean qualification.

Detailed benchmarks: [48 GB dossiers](docs/campaigns/48gb-v032.md).

The standing reasoning gate is 190/200 GSM8K. Qwen MoE scores 182–185/200 and Gemma-26B target-only scores 180/200 under relaxed campaign gates; their speed/context results are not standing-gate qualification. Dense Qwen and Ada reasoning scores, perplexity, and test-specific exceptions are in the dossiers.

## Main failures and blockers

- **DFlash2 + prefix caching on the native NVFP4 dense checkpoint:** 6,208-token LCM block alignment reduces the KV pool to **24,967 tokens** and admission to **24,832** on 24 GB SM120. The 16K test also fails cold/warm output parity. [Dense Dossier 3](docs/campaigns/24gb-v032.md#dossier-3-native-nvfp4-checkpoint-with-dflash2).
- **External DFlash2 beyond its native position range:** MoE YaRN runs fail at near-limit decode when the target is scaled but the drafter retains its native configuration. [MoE Dossier 3](docs/campaigns/moe-v032.md#dossier-3-moe-yarn-context-scaling--targetdrafter-context-mismatch).
- **FP8 KV kernel boundary on SM89:** Gemma-12B's head-dimension-512 prefill path cannot compile for Ada in the tested stack; NVFP4 KV boots. Its int4 assistant also has a projection-loader blocker. [Ada Dossier 1](docs/campaigns/ada16-v032.md#dossier-1-gemma-4-12b-quasar-w4a16-g64-first-qualification-target-only).
- **Gemma-12B near-limit retrieval and cache parity:** the 261K retrieval failure repeats with speculation and prefix caching disabled; the k3 assistant's 32K cache test has different cold/warm text. [Ada Dossier 2](docs/campaigns/ada16-v032.md#dossier-2-gemma-4-12b-drafter-arm-k1-to-k3-and-k3-extended-battery).
- **Dense NVFP4 target-only QA:** the PP-leading configuration fails the three-fact QA test at 64K. [Dense Dossier 3](docs/campaigns/24gb-v032.md#dossier-3-native-nvfp4-checkpoint-with-dflash2).
- **Gemma-26B assistant tradeoff:** negligible gain at k=1, slower at k=3, additional memory pressure, and a prompt-logprob discrepancy. [MoE Dossier 5](docs/campaigns/moe-v032.md#dossier-5-gemma-4-26b-a4b-initial-evaluation).

## Scope and contributions

- **Hardware & model evaluation campaigns ([`docs/campaigns/`](docs/campaigns/README.md)):** Empirical single-stream ($C=1$) benchmarks across 16 GB Ada (SM89), 24 GB Blackwell SFF (SM120, 70 W), and 48 GB Blackwell (SM120, 300 W) covering dense Qwen3.8-27B, Qwen3.6-35B-A3B MoE, and Gemma-4 (12B, 26B-A4B, 31B), including YaRN context scaling up to 350K–1M tokens and cross-backend comparisons (`llama.cpp`, `ExLlamaV3`).
- **Three-tier verification methodology ([`docs/methodology.md`](docs/methodology.md)):** Separates engine-admitted context, physical KV pool capacity, and verified occupied context across Smoke, Qualification, and Extended tiers (top-20 KL divergence, 200-question GSM8K, 48.4K-token 4-domain perplexity, 32K prefix-cache mutation/parity, and near-limit multi-depth NIAH and 6-fact QA), alongside root-cause isolation of prefix-cache parity, kernel-compilation, and target–drafter scaling failures.
- **vLLM 0.32 migration & patch curation ([`PATCHES.md`](PATCHES.md), [migration report](docs/campaigns/migration-v032.md)):** Audits the upstream [HyperQwen by syv-ai](https://github.com/syv-ai/HyperQwen) patch series against vLLM 0.32.0 and FlashInfer 0.7.0.post1, retiring 17 upstreamed or superseded v0.31 KV/speculation patches (including KVarN and custom split-KV attention in favor of native `--kv-cache-dtype nvfp4`) and porting the retained 37-patch series to vLLM 0.32 with `--fuzz 0` verification ([`patches/apply.sh`](patches/apply.sh)).

## Attribution and navigation

The serving launcher, benchmark helpers, and patch series originate in [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) and other credited upstream work; patch authorship and provenance are preserved in [`PATCHES.md`](PATCHES.md) and [`ATTRIBUTION.md`](ATTRIBUTION.md).

[Support matrix](docs/support-matrix.md) · [Methodology](docs/methodology.md) · [Patch series](PATCHES.md) · [Configuration notes](recipes/README.md) · [Attribution](ATTRIBUTION.md) · [Apache-2.0](LICENSE)
