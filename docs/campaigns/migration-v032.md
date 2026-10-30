# The vLLM 0.32.0 Migration & Technical Report

**Scope:** Upgrading the HyperQwen single-user serving stack from vLLM 0.31.0 to vLLM 0.32.0 on a single 24 GB SM120 GPU (NVIDIA RTX PRO 4000 Blackwell SFF, 70 W), supporting Qwen3.8-27B (W4A16 AutoRound-fast), native NVFP4 KV cache, FP8 KV cache, and MTP / DFlash2 speculative decoding.
**Protocol Standard:** Owned systemd user service, `MemoryMax=48000000000` (48 GB), `MemorySwapMax=0`, `KillMode=control-group`, compiler concurrency 4 (`MAX_JOBS=4`), single GPU pinned by UUID (`<GPU_UUID>`).

---

## 1. Dependencies & Environment Isolation

* **Environment:** The experiments use a dedicated `venv-v032/`, selected through the repository's `venv` path.
* **Runtime Versions:**
  * `vllm==0.32.0` (upgraded from `0.31.0`)
  * `torch==2.13.0` (pinned)
  * `flashinfer-cubin==0.7.0.post1` with CUDA 13.0 runtime linker (`libcudart.so`) and system `nvcc` 13.0.88
  * `transformers<5.18`, `huggingface_hub>=1.32.0`, `xgrammar==0.2.7`
  * Python 3.14.4

---

## 2. Patch Series Audit & Retirements

This section records the migration-time inventory before later pruning. Its 50/46 counts and references to removed scripts and patches are historical, not the current installation inventory. The current ordered series contains 37 patches; [PATCHES.md](../../PATCHES.md) and `bash patches/apply.sh --list` describe the retained files.

The upstream vLLM 0.32.0 codebase was audited against the 50 patches in `patches/series` and the KVarN / NVFP4 series. Four patches were identified as fully obsolete due to upstream feature merges and were retired:

### Retired Patches

| Patch File | Upstream PR / Feature | Reason for Retirement |
|---|---|---|
| `patches/dflash2-lookup-drafting.patch` | PR #60452 | Merged into upstream `CandidateSampler` with native candidate extraction. |
| `patches/dflash2-ngram-chains.patch` | PR #60452 | Upstream natively supports multi-token speculative n-gram extraction. |
| `patches/dflash2-z-adaptive-emitted.patch` | PR #60452 | Upstream speculative engine natively manages dynamic emission lengths. |
| `patches/serve-404-served-names.patch` | PR #46963 | Upstream router cleanly handles 404 on unserved model names without hanging. |

### Ported Patches (46 remaining)

At migration time, all remaining 46 patches were reported rebased and verified clean with `--fuzz 0`:
* Line offsets and contextual anchors were adjusted to match vLLM 0.32's refactored modular structure.
* `patches/vllm-0.31.0-dflash-nvfp4.patch`: Updated attention backend signature to preserve `divisor_of: int | None = None`.
* `nvfp4/install_dflash_nvfp4_context_kv.py`: Retired unpack workarounds as vLLM 0.32 natively handles quantized linear context KV projections.
* `patches/series` was updated, and `python3 scripts/patches_md.py` regenerated `PATCHES.md` without discrepancies.

---

## 3. Blackwell SM120 Native NVFP4 KV Integration

In vLLM 0.31, NVFP4 KV cache required an external backport series (`nvfp4/install.sh`) and custom FlashInfer memory layout adapters. In vLLM 0.32:
* **Native Flag:** vLLM 0.32 introduces official `--kv-cache-dtype nvfp4` support.
* **Kernel & Memory Layouts:**
  Running `test_gpu.py` on the RTX PRO 4000 Blackwell SFF confirmed that all four memory layouts (`LBNHC`, `LBHNC`, `BLNHC`, `BLHNC`) pass with a relative L2 error of $\sim 0.095$ (below the test's stated acceptance threshold of $\le 0.12$; this is not a general FP4 error bound).
* **Token Embedding UVA:** Offloads 2.54 GB of token embeddings to pinned host memory via `EMBED_UVA=1`, ensuring the 27B model fits cleanly in 24 GB VRAM alongside 7.85 GiB of dedicated KV cache.

---

## 4. Methodology & Benchmark Suite Upgrades

The benchmarking harness (`nvfp4/run_v032_controls.py`) was significantly upgraded for the 0.32 campaign:

1. **Multilingual PPL Overhaul:**
   * Retired Danish Wikitext (`dan_Latn`).
   * Added FineWeb-2 German (`deu_Latn`, 40 chunks / 11,888 tokens) and Greek (`ell_Grek`, 40 chunks / 17,761 tokens) alongside Wikitext-2 English (11,175 tokens) and vLLM core Python code (7,573 tokens), bringing total evaluated text to 48,397 tokens.
2. **Fast Top-20 KL Divergence Gate (`kld-distribution`):**
   * Implemented a distribution fidelity gate that pre-fills 5 diverse snippets across English, German, Greek, and Code, collecting top-20 logprobs across 946 token positions.
   * Runs in $<3$ seconds and computes Top-1 token agreement, mean KL divergence ($D_{KL}$), and max divergence against the BF16 reference baseline.
3. **Formalized Sampling Profiles:**
   * `greedy`: $T=0.0, p=1.0, k=-1, \text{pp}=0.0, \text{rp}=1.0, \text{enable\_thinking}=\text{False}$ (used for PPL, prefill, and strict mathematical baseline checks).
   * `instruct`: $T=0.7, p=0.80, k=20, \text{pp}=1.5, \text{rp}=1.0, \text{enable\_thinking}=\text{False}$ (used for conversational tasks).
   * `thinking`: $T=1.0, p=0.95, k=20, \text{pp}=0.0, \text{rp}=1.0, \text{enable\_thinking}=\text{True}, \text{max\_tokens}=2048$ (used for native CoT reasoning).

---

## 5. Measured Acceptance & Performance Comparison

Comparing vLLM 0.31 against vLLM 0.32 on the W4A16 AutoRound checkpoint with MTP-4 speculation:

| Metric | vLLM 0.31 (measured) | vLLM 0.32 (measured) | Verdict |
|---|---|---|---|
| **Max Context Admitted (NVFP4)** | 262,144 tokens | **262,144 tokens** | Full native 256K context maintained |
| **KV Pool Tokens (7.85 GiB)** | 400,764 tokens | **400,764 tokens** | $3.5\times$ capacity vs BF16 |
| **TG-0 MTP Throughput (greedy)** | 58.2 tok/s | **57.56 tok/s** | Similar measured rates |
| **TG MTP Sampled Throughput** | 56.66 tok/s | **59.62 tok/s** | **+5.2% measured rate**; scheduler/graph contribution not isolated |
| **Prefill PP-1024** | 940 tok/s | **965 tok/s** | Faster short-context prefill |
| **Cold Compile Time (4 jobs)** | ~310 s | **~265 s** | About 15% lower recorded cold compile time |
| **Host Memory Containment** | ~24.5 GB peak | **23.78 GB peak** | Safely below 48 GB cgroup cap |
| **Memory Pressure Events** | 0 | **0** | Clean, zero OOMs or kernel kills |
