# Single-stream measurement methodology

All benchmarks in this repository target **single-stream ($C=1$) serving**: one active request/sequence at a time on a single workstation GPU, rather than concurrent batch throughput.

Each campaign separates four distinct dimensions that are often conflated in local-LLM benchmarks:

1. **Prefill vs. decode vs. end-to-end HTTP throughput:** Prompt-processing rate (PP), time to first token (TTFT), inter-token decode rate, and complete-request output throughput (`output_tokens / http_elapsed`) are reported separately. End-to-end output rate includes prefill time and is distinct from pure decode rate.
2. **Admitted context vs. KV pool capacity vs. occupied context:** Engine-admitted `max_model_len`, physical KV cache capacity in token equivalents, and actually tested occupied context (`prompt_tokens + completion_tokens`) are tracked independently. Booting at a context limit does not imply retrieval accuracy or prefix-cache correctness at that length.
3. **Numerical quality gates before speed rankings:** Configurations are evaluated against a top-20 logprob KL divergence check, a frozen 200-question GSM8K subset (standing gate $\ge 190/200$), and a 48,397-token four-domain perplexity suite before comparing throughput.
4. **Unmerged run records:** Failed, interrupted, or boot-blocked checks are retained alongside passing measurements rather than merged across separate configurations.

## Terminology

| Term | Meaning in these reports |
|---|---|
| C=1 | One active request/sequence at a time |
| W4A16 | Four-bit weights with 16-bit activations; any INT8 layer override is stated separately |
| KV cache | Stored attention keys and values; cache precision is distinct from weight precision |
| NVFP4 | NVIDIA's four-bit floating-point format with block scales, used here for weights or KV storage |
| MTP / draft depth k | Multi-token prediction; k is the number of proposed tokens per verification step |
| DFlash2 | An external speculative drafter, evaluated separately from integrated MTP |
| Output / decode tok/s | Complete-request output rate versus rate derived from inter-token timing |
| Admitted / occupied context | Engine-accepted limit versus actual prompt plus generated tokens in a test |

## Evaluation tiers

The three-tier protocol below defines the campaign standard. Earlier archived snapshots retain the test order, prompts, and thresholds used when they were executed.

| Stage | Smoke | Qualification | Extended |
|---|---|---|---|
| Boot and admission | Model/runtime identity, memory allocation, engine limit, and KV pool | Same | Same |
| Arithmetic smoke | `sqrt(144) == 12` | Same | Same |
| Generation throughput | Greedy, one pass, before slower quality tests | Greedy and instruct, two passes (second pass retained with speculative telemetry) | Same |
| Distribution gate | Five prompts, top-20 KL divergence $\le 0.05$, top-1 agreement $\ge 85\%$ | Same | Same |
| GSM8K | 20 questions, $\ge 18/20$ | 200 questions, standing gate $\ge 190/200$ | Same |
| Multilingual perplexity | Not run | 48,397 tokens across English, German, Greek, and Python code | Same |
| Prefill ladder | 1K cold/warm, then 16K | 1K, 16K, 64K ascending shared filler with cached-token telemetry | Same |
| Dedicated prefix cache | Not run | Rotated 32K filler: cold miss, warm hit, mutation invalidation, and $T=0$ output parity | Same |
| Occupied decode | Not run | 512-token generation budget at 1K, 64K, 128K (adapted to admitted capacity) | Same |
| Near-limit context | Not run | Not run | Combined three-depth NIAH needles (0.05, 0.50, 0.95) and six-fact QA with a 512-token generation budget |

## Interpretation and measurement boundaries

- **Status labels and relaxed gates:** Where an archived run recorded `PASS` under a campaign-specific relaxed threshold (such as 180/200 on initial MoE/Gemma runs) or an execution-only check, the dossiers report both the original artifact label and the standing 190/200 gate outcome (see [status vocabulary](support-matrix.md#status-vocabulary)).
- **Retained pass policy:** Two-pass generation tests report the declared retained pass (pass 2) for both complete-request output and decode rates, rather than cherry-picking the faster pass.
- **Speculative & power overhead:** A 70 W or 300 W GPU power cap describes the board power limit rather than whole-system wall power. Integrated MTP heads avoid a separate drafter weight allocation on disk/VRAM but still consume activation, recurrent-state, workspace, and CUDA-graph memory.
- **Scope of comparisons:** Rankings and perplexity deltas apply to the specific model preparation, runtime/patch stack, sampling profile, and frozen evaluation subset tested.
