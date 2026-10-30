# vLLM 0.32 / 24 GB Blackwell command draft

This MTP-4 template records the initial W4A16/NVFP4 KV control ([campaign report](../docs/campaigns/24gb-v032.md), [recorded check outcomes](../docs/dense-checks.md); its 64K-prefill check failed). Follow [local setup and reproduction](../docs/install.md) for environment, patch, and systemd containment prerequisites.

For the prepared W4A16-fast target ([`qwen-models.json`](qwen-models.json)) with MTP-4 and NVFP4 attention KV:

```bash
export CUDA_VISIBLE_DEVICES='<GPU_UUID>'
export MODEL='/path/to/prepared/W4A16-fast'
export MAX_JOBS=4 FLASHINFER_NVCC_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export CTX=long SPEC=mtp DRAFT_TOKENS=4 MAX_LEN=-1 GPU_UTIL=0.93 MAX_SEQS=1
export EMBED_UVA=1 VISION=0 SPEC_ATTN=0 PREFIX_CACHE=0 PREFILL_BUDGET=2048
export HOST=127.0.0.1 PORT=18031
export INT8_ACT='' PREFILL_ATTN=''
export EXTRA_ARGS='--attention-backend FLASHINFER --kv-cache-dtype nvfp4 --no-enable-prefix-caching'
bash single-user/start_qwen.sh
```

## Later measured configurations

The [dense Qwen dossiers](../docs/campaigns/24gb-v032.md) document separate preparations and overlays:

| Configuration | Measured settings and outcome |
|---|---|
| Native NVFP4-pre-final, MTP-3 | BF16 MTP tensors, 262,144 admission, 318,415 pool tokens; 47.73 output tok/s and 194/200 GSM8K |
| Native NVFP4 draft-depth sweep | k4 has the highest tested greedy output (49.31 tok/s); k3 has the highest tested instruct output (44.19 tok/s) |
| W4A16 YaRN | MTP-3, prefix ON, factor-2 overlay, INT8 MLP prefill; 351,024 admission and 349,760 occupied tokens; 193/200 with a 1,024-token response budget |
| Native NVFP4 YaRN | Dedicated ModelOpt-preserving overlay, MTP-3; 305,024 admission and 303,683 occupied tokens |

Each extended configuration requires its matching model preparation and RoPE overlay rather than only changing `MAX_LEN` in the initial-control template. The W4A16 YaRN dossier combines separate reasoning, QA-budget, and prefix-diagnostic runs.
