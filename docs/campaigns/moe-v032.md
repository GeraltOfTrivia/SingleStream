# 24 GB Blackwell: MoE benchmark dossiers

Scope: Qwen3.6-35B-A3B and Gemma-4-26B-A4B on the NVIDIA RTX PRO 4000 Blackwell SFF (24 GB, SM120, 70 W), using vLLM 0.32.0, FlashInfer 0.7.0.post1, and one active request. The dossiers retain capacity, quality, throughput, prefill, occupied-context, retrieval, and prefix-cache results.

The standing GSM8K threshold is 190/200. Several original PASS labels use a relaxed threshold or execution-only checks; the scores and exceptions remain visible. See [methodology](../methodology.md) for metric definitions.

- [Dossier 1: MoE Native NVFP4 with DFlash2 vs Target-Only Baseline](#dossier-1-moe-native-nvfp4-with-dflash2-vs-target-only-baseline)
- [Dossier 2: MoE Native NVFP4 with Integrated MTP-3 Speculation](#dossier-2-moe-native-nvfp4-with-integrated-mtp-3-speculation)
- [Dossier 3: MoE YaRN Context Scaling & Target–drafter context mismatch](#dossier-3-moe-yarn-context-scaling--targetdrafter-context-mismatch)
- [Dossier 4: MoE Extended Context (YaRN-320K) with Integrated MTP-3 & Prefix Caching](#dossier-4-moe-extended-context-yarn-320k-with-integrated-mtp-3--prefix-caching)
- [Dossier 5: Gemma-4 26B A4B Initial evaluation](#dossier-5-gemma-4-26b-a4b-initial-evaluation)

## 3-Way Native Context Comparison (262,144 Tokens)

All three arms evaluated on identical single-GPU hardware (RTX PRO 4000 Blackwell SFF 24 GB, 70 W TDP cap) using native NVFP4 KV cache at 262,144 context ceiling:

| Metric | Target-only baseline | DFlash2 Drafter (external drafting, $k=7$) | Integrated MTP-3 (integrated MTP, $k=3$) |
| :--- | :--- | :--- | :--- |
| Output Decode (Greedy) | 126.90 tok/s | 194.17 tok/s (+53.0%) | 202.04 tok/s (+59.2%) |
| Pure Decode TPS (Greedy) | 127.12 tok/s | 200.73 tok/s | 206.61 tok/s |
| Output Decode (Instruct) | 120.30 tok/s | 179.62 tok/s | 189.84 tok/s |
| Speculative Acceptance Rate | — | 35.8% (out of 7) | 57.0% (out of 3) |
| Mean Acceptance Length | — | 3.51 tokens / step | 2.71 tokens / step |
| GSM8K Accuracy (200Q) | 183 / 200 (91.5%) | 184 / 200 (92.0%) | 182 / 200 (91.0%) |
| GSM8K Wall-Clock Time | 487.2 s | 219.7 s (2.22x faster) | 250.7 s (1.94x faster) |
| Combined PPL (48,397 tok) | 7.947 | 7.946 | 7.950 |
| English / German PPL | 11.457 / 11.890 | 11.458 / 11.889 | 11.464 / 11.897 |
| Greek / Code PPL | 5.659 / 5.449 | 5.659 / 5.449 | 5.660 / 5.451 |
| Cold Prefill (PP-1024) | 5,420 tok/s | 5,042 tok/s | 5,802 tok/s |
| Cold Prefill (PP-65536) | 3,110 tok/s | 3,122 tok/s | 2,859 tok/s |
| Occupied Decode (1K Context) | 110.18 tok/s (~123.8 tok/s) | 147.45 tok/s (~154.4 tok/s) | 140.38 tok/s (~201.5 tok/s) |
| Occupied Decode (64K Context) | 2.64 tok/s (~120.2 tok/s) | 3.01 tok/s (~186.2 tok/s) | 5.18 tok/s (~192.8 tok/s) |
| Occupied Decode (131K Context) | 0.91 tok/s (~112.8 tok/s) | 2.23 tok/s (~172.5 tok/s) | 0.89 tok/s (~178.4 tok/s) |
| Occupied Decode (261K Context) | 0.85 tok/s (~101.5 tok/s) | 0.40 tok/s (~148.0 tok/s) | 0.58 tok/s (~154.2 tok/s) |
| Verified Occupied Context | 260,821 tokens | 260,712 tokens | 260,767 tokens |
| 261K Retrieval NIAH (3 depths) | 3/3 (100% PASS) | 3/3 (100% PASS) | 3/3 (100% PASS) |
| 261K 6-Fact QA | PASS (6/6 facts) | PASS (6/6 facts) | PASS (6/6 facts) |
| External Drafter VRAM | 0 GiB | 1.8 GiB | 0 GiB (Integrated in weights) |
| Available KV Cache Pool | 762,090 tokens | 285,587 tokens | 384,113 tokens (+34.5%) |
| Peak GPU VRAM | 22,238 MiB | 23,892 MiB | 23,912 MiB |
| Extended Battery Verdict | 24 PASS, 0 FAILED | 24 PASS, 0 FAILED | 24 PASS, 0 FAILED |

---

## Dossier 1: MoE Native NVFP4 with DFlash2 vs Target-Only Baseline

### Model and configuration

* Target Model: `Qwen3.6-35B-A3B-W4A16-AWQ` (W4A16 Marlin GEMM, 3B active parameters, 256 experts)
* Drafter Model: `Qwen3.6-35B-A3B-DFlash2` ($k=7$ draft proposals, `draft_sample_method=probabilistic`)
* Hardware: NVIDIA RTX PRO 4000 Blackwell SFF (24 GB, SM120, 70 W).

### Sizing & Calibration Observations
- With `gpu_util=0.92` and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, the engine provisions 3.05 GiB KV cache (285,587 pool tokens) while preserving ~500 MiB of unallocated VRAM margin.
- Graph capture completes cleanly without allocator pressure (peak 23,892 MiB).
- Both DFlash2 and Target-only completed 24 out of 24 tests without a single dropped request, failure, or timeout.

### Occupied Context Decode Ladder & Actual Decode Rates

| Context Length | Target-Only E2E (Actual Decode) | DFlash2 E2E (Actual Decode) | TTFT (Cold Prefill) | Status | Notes / Token Generations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Occupied 1,024 | 110.18 tok/s (~123.8 tok/s) | 147.45 tok/s (~154.4 tok/s) | ~0.10 s | PASS | 523 prompt + 103 / 339 gen tokens |
| Occupied 65,536 | 2.64 tok/s (~120.2 tok/s) | 3.01 tok/s (~186.2 tok/s) | ~21.3 s | PASS | 65,035 prompt + 56 / 67 gen tokens (E2E incl. ~21s prefill) |
| Occupied 131,072 | 0.91 tok/s (~112.8 tok/s) | 2.23 tok/s (~172.5 tok/s) | ~68.9 s | PASS | 130,571 prompt + 62 / 155 gen tokens (E2E incl. ~69s prefill) |
| Occupied 261,120 | 0.85 tok/s (~101.5 tok/s) | 0.40 tok/s (~148.0 tok/s) | ~235.0 s | PASS | 260,821 / 260,712 verified occupied context tokens |

---

## Dossier 2: MoE Native NVFP4 with Integrated MTP-3 Speculation

### Model and configuration

* Target Model: `Qwen3.6-35B-A3B-W4A16-AWQ`
* Speculative Drafter: Integrated `Qwen3_5MoeMTP` ($k=3$ proposals, `draft_sample_method=probabilistic`)
* Admitted Context Length: 262,144 tokens
* KV Pool Capacity: 384,113 tokens (2.50 GiB)
* Status: Recorded PASS (24 PASS, 0 FAILED)

### Detailed test results

The recorded PASS labels use the campaign protocol. The 182/200 score is below the standing 190/200 reasoning gate.
| Evaluation Stage | Scope / Context Length | Measured Result | Status | Observations |
| :--- | :--- | :--- | :--- | :--- |
| Engine Boot | Configured Max Context | 262,144 tokens | PASS (120.0 s) | 384,113 token pool admitted cleanly |
| Arithmetic Smoke | Short prompt | `15 * 6 = 90` | PASS (0.057 s) | Exact arithmetic match |
| KLD Distribution Gate | 5 prompts / 946 positions | Top-20 distribution recorded | PASS (1.192 s) | Mean KLD: 0.0003 against baseline |
| Reasoning Quality | GSM8K (200 questions, $T=0$) | 182 / 200 (91.0%) | PASS (250.7 s) | Below the standing 190/200 gate |
| Multilingual PPL | Wikitext-2 (English) | 11.464 | PASS (5.38 s) | Recorded PPL |
| | FineWeb-2 (German `deu_Latn`) | 11.897 | PASS (5.38 s) | Recorded PPL |
| | FineWeb-2 (Greek `ell_Grek`) | 5.660 | PASS (6.44 s) | Recorded PPL |
| | vLLM Core Source Code | 5.451 | PASS (3.61 s) | Recorded PPL |
| | Token-Weighted Combined PPL| 7.950 (48,397 tokens) | PASS (20.81 s) | Close to target-only (7.947) |
| Generation Speed (TG) | Greedy Complete Output ($T=0$) | 202.04 tok/s (decode: 206.6 tok/s) | PASS (98.74 s) | +59.2% output throughput vs target-only |
| | Instruct Complete Output ($T=0.7$) | 189.84 tok/s (decode: 193.8 tok/s) | PASS (103.15 s) | Sampled acceptance: 53.21% |
| Prefill Ladder (PP) | PP-1024 | 5,802 tok/s | PASS (0.570 s) | High-speed short prefill |
| | PP-16384 | 5,184 tok/s | PASS (9.483 s) | High-speed medium prefill |
| | PP-65536 | 2,859 tok/s | PASS (67.844 s) | 65K prefill HTTP 200 verified |
| Occupied Context Decode| 1,024 context tokens | 140.38 tok/s (~201.5 tok/s) | PASS (1.48 s) | 523 prompt + 208 gen; speculative decoding fires cleanly |
| | 65,536 context tokens | 5.18 tok/s (~192.8 tok/s) | PASS (23.55 s) | 65,035 prompt + 122 gen (5.18 tok/s is E2E incl. ~22.9s prefill) |
| | 131,072 context tokens | 0.89 tok/s (~178.4 tok/s) | PASS (73.42 s) | 130,571 prompt + 65 gen (0.89 tok/s is E2E incl. ~73.0s prefill) |
| | 261,120 context tokens (Near-limit)| 0.58 tok/s (~154.2 tok/s) | PASS (255.59 s) | 260,767 verified occupied tokens (0.58 tok/s is E2E incl. ~254.6s prefill) |
| Retrieval NIAH (65K) | Depth 0.50 | Passcode: `KEY50V032` | PASS (23.25 s) | 100% exact match |
| Retrieval NIAH (131K) | Depth 0.50 | Passcode: `KEY50V032` | PASS (73.65 s) | 100% exact match |
| QA Multi-Fact (65K) | 3-fact distributed query | Leaders & Budgets (3/3) | PASS (23.62 s) | 100% exact match |
| Retrieval NIAH (261K) | Depth 0.05 (Beginning) | Passcode: `KEY50V032` | PASS (256.91 s) | 100% exact match |
| | Depth 0.50 (Middle) | Passcode: `KEY50V032` | PASS (257.06 s) | 100% exact match |
| | Depth 0.95 (End) | Passcode: `KEY50V032` | PASS (257.10 s) | 100% exact match |
| QA Multi-Fact (261K) | 6-fact distributed query | Leaders & Budgets (6/6) | PASS (258.40 s) | 100% recall across 261,120 context tokens |

---

## Dossier 3: MoE YaRN Context Scaling & Target–drafter context mismatch

### Model and configuration

* Drafter Model: `Qwen3.6-35B-A3B-DFlash2` ($k=7$)

### Architectural Failure Analysis: The Target–drafter context mismatch
1. Observed failure:
   - In both YaRN runs (at 320,000 tokens and 300,000 tokens), the engine booted cleanly, admitted 328,205 pool tokens, and passed all short, medium, and long-range checks up through 131,072 context tokens (91.5% GSM8K, 7.936 PPL, 190.8 tok/s decode, and 130,705 occupied context tokens).
   - In both runs, near-limit occupied decode (318,464 tokens in YaRN-320K and 299,008 tokens in YaRN-300K) triggered an identical `torch.AcceleratorError: CUDA error: device-side assert triggered` inside the attention kernel builder (`flashinfer.py:1527: seq_lens_cpu = common_attn_metadata.seq_lens.cpu`).
2. Configuration mismatch:
   - The YaRN overlay scaled only the target model (`max_position_embeddings: 300000`, `rope_type: yarn`).
   - The external DFlash2 draft model remained pointing to its unscaled base config: `max_position_embeddings: 262144` with default RoPE.
   - At $\le 131\text{K}$, positions were well within DFlash2's 262,144 limit. But as soon as occupied context exceeded $262,144$ ($2^{18}$), DFlash2 was queried to propose draft tokens beyond its supported position table, coinciding with the observed device-side assert. A matched scaled-drafter control was not tested, so the exact cause remains incompletely isolated.
   - By contrast, in Dossier 5 on the dense model, MTP-3 completed the reported 350K probes. Its integrated head uses the target's YaRN overlay, avoiding this external-drafter configuration mismatch; those results do not independently isolate the cause of the DFlash2 failure.

---

## Dossier 4: MoE Extended Context (YaRN-320K) with Integrated MTP-3 & Prefix Caching

### Model and configuration

* Target Model: `Qwen3.6-35B-A3B-W4A16-AWQ` with YaRN factor $1.2207\times$ overlay (`max_position_embeddings: 320000`, `original_max_position_embeddings: 262144`)
* Speculation & Features: Integrated MTP-3 (integrated MTP, three MTP drafts), prefix caching ON, embedding UVA offload, FlashInfer 0.7.0.post1, native NVFP4 KV cache
* Admitted Context Length: 320,000 tokens
* KV Pool Capacity: 382,171 tokens (2.48 GiB)
* Peak Sampled VRAM: 23,912.0 MiB (safely contained within the 24,467 MiB hardware ceiling; zero host thrashing)

### Consolidated 320K evaluation matrix (original labels retained)

The 185/200 score meets the run's stated 180/200 threshold, but not the standing 190/200 gate.

| Category / Evaluation Stage | Metric / Benchmark | Measured / Reported Value | Decode Throughput | Mechanical Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Boot & Sizing | Admitted Context Length | 320,000 tokens | — | PASS | $1.2207\times$ YaRN dilated context in single 24 GB GPU |
| | KV Pool Capacity | 382,171 tokens (2.48 GiB) | — | PASS | Provisions full prefix-caching block state tables |
| | Peak Sampled VRAM | 23,912 MiB | — | PASS | Clean containment on 24,467 MiB physical card |
| Correctness & Smoke | Arithmetic Smoke | `"90"` | — | PASS | $15 \times 6 = 90$ greedy output |
| | KLD Distribution Gate | 5 prompts, 946 positions | — | PASS | Distribution matches reference logprobs ($D_{KL} \le 0.05$) |
| Mathematical Reasoning| GSM8K n=200 Greedy (1024 tok)| 185 / 200 (92.5%) | — | PASS | Exceeds 180/200 gate; elapsed 268.5 s (mean 294 tokens) |
| Multilingual Perplexity| English (`ppl-en`) | 11.446 (11,175 tokens) | — | PASS | Clean Wikitext / news perplexity |
| | German (`ppl-de`) | 11.890 (11,888 tokens) | — | PASS | High cross-lingual stability |
| | Greek (`ppl-el`) | 5.657 (17,761 tokens) | — | PASS | Recorded Greek PPL |
| | Code (`ppl-code`) | 5.455 (7,573 tokens) | — | PASS | vLLM codebase token perplexity |
| | Combined Perplexity | 7.944 (48,397 tokens) | — | PASS | Exact match with native 256K base (7.946–7.950) |
| Throughput (C1 Retained)| Greedy TG (Pass 2) | 1024 gen tokens | 206.19 tok/s | PASS | Pure decode rate (4.85 ms TPOT) |
| | Instruct TG (Pass 2) | 1024 gen tokens | 187.62 tok/s | PASS | Instruct decode under sampled settings |
| Speculative Telemetry | Mean Acceptance Length | 2.70–2.75 tokens/step | — | PASS | Out of 3 proposals |
| | Draft Acceptance Rate | 56.6%–58.2% | — | PASS | Up to 100% on verbatim long context recall |
| Prefix Caching Test | 32K Prefix Prompt (Cold) | Latency: 8.417 s | — | PASS | Cold prefill of 32,768 tokens |
| | 32K Prefix Prompt (Warm) | Latency: 1.259 s | — | PASS | 6.69× speedup with warm KV prefix cache reuse |
| Prefill Ladder | PP 1,024 | 1,481.9 tok/s (0.69 s) | — | PASS | Clean short prefill |
| | PP 16,384 | 3,502.4 tok/s (4.68 s) | — | PASS | Sustained mid-range prefill |
| | PP 65,536 | 2,713.5 tok/s (24.15 s) | — | PASS | High-throughput long prefill |
| Occupied Decode | Occupied 1,024 | 523 prompt + 324 gen | 157.55 tok/s (~202.0 tok/s) | PASS (2.06 s) | Short occupied generation |
| | Occupied 65,536 | 65,035 prompt + 55 gen | 2.42 tok/s (~193.5 tok/s) | PASS (22.73 s) | 65K occupied (2.42 tok/s is E2E incl. ~22.4s prefill) |
| | Occupied 131,072 | 130,571 prompt + 110 gen | 2.14 tok/s (~179.0 tok/s) | PASS (51.41 s) | 131K occupied (2.14 tok/s is E2E incl. ~50.8s prefill) |
| | Occupied 318,464 | 317,963 prompt + 94 gen | 0.32 tok/s (~142.5 tok/s) | PASS (291.22 s) | 318,057 verified occupied tokens (0.32 tok/s is E2E incl. ~290.5s prefill) |
| Ultra-Long Retrieval | NIAH 65K (depth 0.50) | `KEY50V032` found | — | PASS | 100% exact match |
| | NIAH 131K (depth 0.50) | `KEY50V032` found | — | PASS | 100% exact match |
| | NIAH 300K (depth 0.50) | `KEY50V032` found | — | PASS | 100% exact match at 300,000 context |
| | NIAH 318K (depth 0.05)| `KEY50V032` found | — | PASS | 100% exact match at beginning of 318K context |
| | NIAH 318K (depth 0.50)| `KEY50V032` found | — | PASS | 100% exact match at middle of 318K context |
| | NIAH 318K (depth 0.95)| `KEY50V032` found | — | PASS | 100% exact match at end of 318K context |
| Multi-Fact QA | QA 3-fact at 65K | 3/3 facts recalled | — | PASS | 100% accuracy on distributed project facts |
| | QA 6-fact at 318,464 | 6/6 facts recalled | — | PASS | 100% accuracy on 6 facts across 318K context |

---

## Dossier 5: Gemma-4 26B A4B Initial evaluation

### Model and configuration

* Target Model: `Gemma-4-26B-A4B-NVFP4` — `nvidia` NVFP4 release, revision `a19cfe00` (30 layers: 25 sliding-window-1024 + 5 full-attention; hidden 2,816; 128 routed experts / top-8; native 262,144 context; no integrated MTP head — the drafter is necessarily external)
* Drafter Model: `gemma4-26b-a4b-it-assistant-nvfp4` — `wangqia0309` assistant, revision `f916e92c` (4-layer draft backbone, hidden 1,024, 2,048 centroids, 0.59 GiB; vLLM 0.32 maps `gemma4_assistant` → `Gemma4MTPModel` ; `n_predict=1` gives one proposal per head forward; chained k3 was tested separately)
* Hardware: NVIDIA RTX PRO 4000 Blackwell SFF (24 GB, SM120, 70 W).

### Checkpoint Contract (Phase-1 Inspection)
- Target: modelopt NVFP4, group-16 quantization (layers 0–1 + `lm_head` excluded), tied embedding (1,476,395,008 B, UVA-offloaded), producer-declared FP8 KV cache scheme (this campaign ran NVFP4 KV per branch protocol; the FP8 arm is a follow-up).
- Both checkpoints native 262,144; drafter backbone hidden 2,816 == target hidden (the proposer wires cross-model KV sharing natively; `num_kv_shared_layers` is force-zeroed by the override).
- FlashInfer log caveat: "XQA decode does not support head_dim=512; reverting to native FlashInfer decode" (full-attention group; the SWA group is head_dim 256).

### Sizing & Calibration: The 0.92 Cliff & the 0.875 Recalibration
- target-only @ `gpu_util=0.92`: model 16.83 GiB, KV pool 4.22 GiB = 658,156 tokens (2.51× one 262K request), idle ~23.4 GiB. Ran the full battery cleanly; GPU peak 23,902 MiB.
- assistant @ `0.92`: +0.59 GiB drafter → idle 23,770 MiB, leaving only 200.5 MiB below the cap; a 240 MiB prompt-logprob buffer OOM'd EngineCore on the Greek PPL prompt (`torch.OutOfMemoryError` at the logits soft-cap division; the engine died → arm blocked at `ppl-el`). Recorded before the OOM: boot, smoke, KLD, GSM8K 183/200, `ppl-en`, `ppl-de` (preserved as a partial).
- assistant @ `0.875` (recalibrated): model 17.42 GiB, KV pool 2.51 GiB = 391,516 tokens (1.49× one request — C1-adequate), idle 22.5–22.8 GiB (~0.6–0.9 GiB margin). The previously-failing Greek PPL passed in 5.9 s. Boot 125 s (JIT-warm) vs 535 s cold for target-only.

### Target-only results (relaxed first-qualification gate)

| Test | Result |
| :--- | :--- |
| Boot (cold) | 535 s; 262,144 admitted; model 16.83 GiB, KV 4.22 GiB (658,156 pool tokens) |
| GSM8K (greedy, 200Q) | 180/200 (90.0%) — passes the relaxed first-qualification gate of 180; mean 221.9 tokens, 1 truncation |
| PPL (en/de/el/code, 49,755 tok) | 556.11 combined |
| TG (output decode, pass 2 of 2) | greedy 57.50 / instruct 56.91 tok/s (TPOT 17.29/17.47 ms; TTFT 118/107 ms) |
| Cold Prefill (median tok/s @ 1K/16K/64K) | 4,759 / 5,244 / 2,882 |
| Occupied decode (E2E tok/s, final context) | 54.31 @647 / 4.70 @65,153 / 1.26 @130,672 / 0.17 @260,666 (verified) |
| NIAH (0.5 @64K, 0.5 @128K, 0.05/0.5/0.95 @261K) | 3/3 + 3/3 PASS |
| Distributed QA (3-fact @64K, 6-fact @261K) | PASS |
| Memory | GPU peak 23,902 MiB; host peak 15.8 GB (under the 48 GB cgroup cap) |

### Assistant results and incomplete near-limit test
The @0.875 arm recorded PASS through: boot, smoke, KLD (record-only), GSM8K 182/200 (91.0%), all four PPL domains, both TG profiles, prefill ×3, and the occupied ladder to 131K. `occupied-261120` was in progress when the campaign was stopped after the throughput comparison — the drafter arm showed no TG improvement and reasoning and PPL measurements were recorded (GSM8K 182/200, all PPL PASS, 131K occupied clean). The test did not complete the near-limit occupied rung; no final passing battery is inferred.

Throughput verdict — no speedup:

| Profile | target-only @0.92 | assistant @0.875 | Δ |
| :--- | :--- | :--- | :--- |
| TG greedy (output) | 57.50 tok/s | 57.71 tok/s | +0.4% |
| TG instruct (output) | 56.91 tok/s | 56.29 tok/s | −1.1% |

Despite 83.5% acceptance / 1.84 mean length (greedy) and 81.6% / 1.82 (instruct) — ~94% acceptance over the easier GSM8K text — the observed tokens/iteration (1.84) and net throughput ratio (≈1.0×) imply overhead equivalent to roughly 83% of a target step under a simple per-step timing model. This estimate includes verification and other overhead; it is not a direct measurement of drafter-forward latency. Occupied-decode E2E rates were also *below* target-only at every recorded rung (49.69/3.73/0.41 vs 54.31/4.70/1.26). The k1 result is approximately throughput-neutral in this setup; the separate k3 screen below is slower — in contrast to Qwen MTP-3's +59% over 3 proposals.

Cross-arm KLD / PPL anomaly (record-only gate FAILED): against the target-only arm's prompt-logprobs, the assistant arm's target logprobs diverged: top-1 agreement 65.1%, mean KLD 0.526, max 10.23 over 938 positions; combined PPL shifted −4.6% (530.64 vs 556.11 @ 49,755 tokens). The target weights, KV dtype, and prompts are shared, but the precise numerical cause is not isolated by these measurements.

### Interpretation
* `Gemma target-only`: Recorded PASS (first qualification, 24/24) — 180/200 meets the relaxed first-evaluation threshold, but not the standing 190/200 threshold.
* `Gemma assistant`: no throughput benefit on 24 GB / 70 W — not a speedup recipe on this card; preserved as a quality record only.
Further tests would need a matched FP8 control and investigation of the assistant prompt-logprob discrepancy.

---

### Assistant draft-depth-3 screen

A separate chained-assistant k3 throughput screen records 52.86 greedy output tok/s, below target-only 57.50 and k1 57.71. Per-position acceptance is 79.2/65.4/53.1% (mean 65.9%). It stops at the throughput decision gate; later quality checks are not run.

## Comparative Test: llama.cpp Backend (Qwen3.6-35B-A3B)

Scope: Same model, same 24 GB SM120 C1 protocol, served by llama.cpp 0.5.0-dev (build 11378, commit `edd6e2bbd`, CUDA/SM120) from the unsloth IQ4_NL GGUF (`Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`, 18,536,192,288 B, SHA256 `d0ad7804…`, embedded MTP drafter). Companion to the native and extended vLLM comparisons.

Comparison configuration: q8_0 KV, native 262,144 context, MTP-3, and prefix caching ON.

### KV sizing probes (not retrieval qualification)

Engine-reported KV footprint at 262,144 (llama.cpp `-lv 4`, drafter loaded): q8_0 2,720 MiB (K 1,360 + V 1,360) = 10.625 KiB/tok · q4_0 1,440 MiB = 5.625 KiB/tok · f16 5,120 MiB = 20.0 KiB/tok, plus draft-KV 512 MiB (MTP), recurrent-state 1,005 MiB (30 linear-attention layers, fixed), compute buffer 821.82 MiB @262K. Context ceilings with the drafter on the 24 GB card:

| KV format | Admitted context in sizing probe | Note |
|---|---|---|
| f16 | ~230K (262K fails KV allocation) | |
| q8_0 | 327,680 (51 MiB free) | |
| q4_0 | 458,752 (161 MiB free; 491,520 fails on the *compute* buffer, not KV) | |

These are memory-admission probes, not demonstrations of usable retrieval at the larger limits.

### q8_0 KV, MTP-3, prefix caching: 262K comparison

| Test | Result |
|---|---|
| Boot | 10 s; model 17,151.7 MiB, KV 2,720 MiB + draft 512 MiB + RS 1,005 MiB; MTP drafter loaded ✓ |
| GSM8K-200 (greedy) | 189/200 (94.5%) — 1 below the standing 190 gate (informational); +4 to +7 over the vLLM 182–185 band |
| TG greedy (pass 2) | 120.04 tok/s output (121.38 decode), TTFT 159 ms, MTP acceptance 67.5% / 3.02 len |
| TG instruct (pass 2) | 114.11 tok/s output; acceptance 62.5% / 2.88 (TTFT includes the thinking phase) |
| PP (cache-building ladder) | 1K cold 1,527 tok/s / warm 11,135 tok/s (7.3×); 16K: 15,876 uncached @1,543 tok/s (508 cached); 64K: 49,667 uncached @1,413 tok/s (15,868 cached from the 16K rung) |
| Prefix-cache 32K cold→warm | 17.36 s → 0.442 s = 39.3×, text parity, 32,775 cached tokens |
| Occupied ladder | 1K→64K (32,263 cached)→131K (64,519 cached) ascending, each building on the last |
| deep-ctx 261,120 | 261,244 final context verified; 3 NIAH needles (0.05/0.5/0.95) + 6 QA facts all recalled in one 512-token generation (gate PASS) |
| Memory | GPU peak 23,004 MiB; host peak 19.57 GB |
| PPL (offline, q8_0 vs f16) | Δ −0.12% @3K window, +0.02% @16K window — both inside the tool's CIs (±0.12–0.47): fp8 KV is PPL-neutral for this model |

### vs the vLLM 0.32 baselines (Dossiers 1–4, same card/protocol)

| Metric | llama.cpp q8_0+MTP-3 | vLLM MTP-3 | vLLM target-only |
|---|---|---|---|
| TG output decode (greedy) | 120.0 tok/s | 202.04 tok/s | 126.90 tok/s |
| MTP acceptance | 67.5% / 3.02 | 57.0% / 2.71 | — |
| GSM8K | 189/200 | 182/200 | 183/200 |
| Cold prefill (1K / 64K) | 1,527 / 1,413 tok/s | 5,802 / 2,859 tok/s | 5,420 / 3,110 tok/s |
| Occupied 261K E2E / verified | 0.27 tok/s / 261,244 | 0.58 / 260,767 | 0.85 / 260,821 |
| Prefix-cache 32K speedup | 39.3× | 6.69× (YARN arm) | — |
| PPL (method differs) | 6.22 @16K window (combined 4-domain) | 7.950 (48K prompt-logprobs) | 7.947 |

### Interpretation

The tested vLLM MTP configuration has higher greedy output throughput; the llama.cpp configuration scores higher on the reasoning subset. Both are below 190/200. Weight quantization and cache settings differ, so this is a practical configuration comparison. Offline llama.cpp PPL uses a different scoring/window method from vLLM prompt-logprobs and is not directly ranked against it. Earlier attempts encountered a PPL-evaluator defect and a CUDA launch timeout; they are not combined with the completed comparison.
