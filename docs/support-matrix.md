# Hardware and result coverage

Results refer to the listed hardware, model preparation, runtime, and workload with one active request. GPU power limits are distinct from the campaigns' 48,000,000,000-byte host-memory containment.

| Hardware | Models / runtime | Coverage | Main limits |
|---|---|---|---|
| RTX PRO 4000 Blackwell SFF, 24 GB, SM120, 70 W | Qwen3.8-27B / vLLM 0.32 | [KV/speculation controls and later MTP/YaRN runs](campaigns/24gb-v032.md) | Initial controls contain failed checks. Later 350K/300K results span distinct configurations, response budgets, and diagnostic runs. |
| Same 24 GB Blackwell | Qwen3.6-35B-A3B / vLLM 0.32 | [Native target/DFlash/MTP and YaRN-320K](campaigns/moe-v032.md) | GSM8K 182–185/200 is below the standing 190/200 gate despite original PASS labels. |
| Same 24 GB Blackwell | Gemma-4-26B-A4B / vLLM 0.32 | [Target-only and assistant tests](campaigns/moe-v032.md#dossier-5-gemma-4-26b-a4b-initial-evaluation) | Target-only 180/200 meets its relaxed first-qualification gate, not the standing gate. Assistant has OOM/logprob anomalies and no measured speed advantage. |
| RTX 2000 Ada Generation, 16 GB, SM89, 70 W | Qwen3.8-27B / vLLM 0.32 | [MTP-3 qualification](campaigns/ada16-v032.md) | 193/200; 41,216 admitted tokens. Above-admission rungs are skipped; near-limit retrieval is not established by the cited Qwen summary. |
| Same 16 GB Ada | Gemma-4 12B / vLLM 0.32 | [Target/assistant and extended tests](campaigns/ada16-v032.md) | Reasoning passes; near-limit retrieval fails in both assistant and target-only controls. Assistant cache text parity fails; baseline KLD absent and PPL protocol differs between runs. |
| RTX PRO 5000 Blackwell, 48 GB, SM120, 300 W | vLLM 0.32 | [48 GB benchmarks](campaigns/48gb-v032.md) | Native dense qualification and extended context, filtered checkpoint comparisons, and MoE/Gemma diagnostics. Prefix-parity failures and reference anomalies remain; qualification limits are stated per dossier. |

## Status vocabulary

- **Measured:** a result record or campaign report documents the measurement.
- **Admitted:** the engine accepted a sequence limit; successful near-limit execution is a separate result.
- **Verified occupied tokens:** actual prompt plus generated tokens in the recorded test, distinct from its nominal rung label.
- **Recorded PASS:** the original label, which can conflict with a numerical gate or cover execution without output parity.
- **Qualification passed:** the specified run meets its stated gates, with any relaxed threshold or absent comparison noted alongside the standing 190/200 GSM8K gate.
- **Failed / skipped / not reached:** a check failed, exceeded admission, or was not executed; none is inferred to have passed.

Detailed JSON is available for selected dense v0.32 controls. Later dense/MoE and Ada records document measurements with settings and procedures; their raw run directories are not included. See [available records](evidence-policy.md).

## Runtime boundaries

Ampere (SM80/SM86) is outside the evaluated hardware scope. NVFP4 KV is the current cache-quantization path; TurboQuant is unsupported by the v0.32 configurations documented here.
