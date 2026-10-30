# Local setup and reproduction

Use a dedicated Python environment, the exact model preparation and applicable vLLM patch stack, and settings from the [configuration notes](../recipes/README.md). The documented v0.32 experiments use vLLM 0.32.0, PyTorch 2.13.0, FlashInfer 0.7.0.post1, and CUDA 13 compiler components. See the [migration report](campaigns/migration-v032.md) for runtime and patch details, and [validation](validation.md) for repository check and reproduction status.

## Match the experiment configuration

1. **Model identity:** Pin model revisions and any preparation/overlay files ([`recipes/qwen-models.json`](../recipes/qwen-models.json)).
2. **Patch stack:** Apply the ordered series using `bash patches/apply.sh <vllm_dir>`. NVFP4 KV support is native to the v0.32 FlashInfer stack.
3. **Host and GPU containment:** Match GPU selection, memory utilization, prefill budget, speculation, and prefix caching. The campaigns run under a 48,000,000,000-byte systemd host-memory limit (`MemoryMax=48000000000`, `MemorySwapMax=0`, `KillMode=control-group`).
4. **Executable and arguments:** `single-user/start_qwen.sh` invokes `venv/bin/vllm` relative to the repository root and configures Qwen-specific parser arguments (`--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`); non-Qwen families require model-appropriate flags.

`single-user/start_qwen.sh` binds to `127.0.0.1` when `HOST` is unset, no API key is configured, and it runs outside a container. With an API key configured (`VLLM_API_KEY` or `api_key.txt`) or inside a container, an unset `HOST` binds to `0.0.0.0`; set `HOST=127.0.0.1` explicitly for loopback binding. Inspect inherited environment variables—especially `EXTRA_ARGS`, which expands last and overrides earlier flags.

This repository focuses on single-stream serving infrastructure and experiment configurations; interactive launchers, Docker distributions, and batch-serving workflows are out of scope.
