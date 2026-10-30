# Benchmark dossiers

| Hardware and models | Detailed report | Dossiers |
|---|---|---|
| 24 GB Blackwell — Qwen3.8-27B | [Dense Qwen](24gb-v032.md) | KV controls; W4A16 DFlash2; native NVFP4 DFlash2; integrated MTP; W4A16 YaRN-350K; NVFP4 YaRN-300K; other backends |
| 24 GB Blackwell — Qwen/Gemma MoE | [MoE benchmarks](moe-v032.md) | Native DFlash2 and target-only; integrated MTP; external-drafter context mismatch; YaRN-320K; Gemma target/assistant; llama.cpp comparison |
| 16 GB Ada — Qwen/Gemma dense | [Ada benchmarks](ada16-v032.md) | Gemma target-only and kernel boundary; assistant-depth and extended tests; Qwen qualification |
| 48 GB Blackwell | [48 GB benchmarks](48gb-v032.md) | Native dense; checkpoint comparison; YaRN-512K/1M; MoE extended; prefix controls; Gemma diagnostics |

Reports retain capacity, reasoning, perplexity, prefill, generation, occupied-context, retrieval/QA, and cache behavior where measured. See [methodology](../methodology.md) for definitions and the standing gates.
