# v0.32 / 24 GB Blackwell MoE settings

The [MoE benchmark dossiers](../docs/campaigns/moe-v032.md) evaluate the configurations below on an RTX PRO 4000 Blackwell SFF (24 GB, 70 W). No configuration on this page clears the standing 190/200 GSM8K gate.

| Configuration | Key settings | Reasoning / result boundary |
|---|---|---|
| Qwen3.6-35B-A3B AWQ, target-only | NVFP4 KV, `GPU_UTIL=0.92`, native 262,144 limit, embedding UVA | 183/200 GSM8K; largest measured native KV pool |
| Same Qwen, integrated MTP-3 | Above settings, `SPEC=mtp`, 3 draft tokens, prefix OFF | 202.04 greedy output tok/s; 182/200, below standing gate |
| Same Qwen, DFlash2-7 | External matching drafter, 7 proposals, NVFP4 KV, native context | 194.17 greedy output tok/s; 184/200, below standing gate |
| Qwen YaRN-320K, MTP-3 | Model-specific factor-1.2207 overlay, `MAX_LEN=320000`, prefix ON | 318,057 occupied tokens; 185/200, below standing gate |
| Gemma-4-26B-A4B NVFP4, target-only | `GPU_UTIL=0.92`, native 262,144 limit, embedding UVA | 57.50 greedy output tok/s; 180/200, relaxed first-qualification gate |
| Gemma external assistant | Matching NVFP4 assistant; utilization lowered to 0.875 after OOM; k1 and separate k3 screen | Little benefit at k1, slower at k3; prompt-logprob anomaly and incomplete checks |

## Configuration notes

- **Model-specific invocation:** `single-user/start_qwen.sh` hardcodes Qwen parser flags; Gemma-4-26B-A4B requires a direct `vllm serve` invocation with model-appropriate arguments.
- **YaRN overlays:** Extended context requires a complete model-specific RoPE overlay passed via a supported runtime flag (`HF_CONFIG_PATH` is not read by `start_qwen.sh`).
- **Environment prerequisites:** See [local setup and reproduction](../docs/install.md).
