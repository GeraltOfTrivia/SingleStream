# Explicit single-stream serving

`start_qwen.sh` is the shell entry point used by the Qwen campaign drivers, adapted from upstream `syv-ai/HyperQwen`. It translates environment variables (`MODEL`, `CTX`, `SPEC`, `GPU_UTIL`, `MAX_LEN`, `MAX_SEQS`, `EMBED_UVA`, `PREFIX_CACHE`, `EXTRA_ARGS`) into a `vllm serve` invocation against `venv/bin/vllm`.

See [measured settings and command templates](../recipes/README.md) and [local setup and reproduction](../docs/install.md) for the exact environment exports used in the v0.32 single-stream ($C=1$, `MAX_SEQS=1`) benchmarks.
