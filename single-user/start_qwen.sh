#!/bin/bash
# SingleStream: Single-User / Low Latency Serving Launch Script.
# Target architectures: NVIDIA Blackwell SM120 and Ada Lovelace SM89.
# Note: Ampere (SM80/SM86) is unsupported and not targeted. KVarN is retired.
#
# Single-user (C=1) serving with speculative decoding:
#  - SPEC=mtp: Multi-token-prediction head verification (up to 4 draft tokens).
#  - SPEC=dflash2: DFlash2 block drafter (7 drafts in one non-autoregressive pass).
#  - SPEC=off: Speculative decoding disabled.
#
# Context configurations:
#  - CTX=fast (default): bfloat16 KV via FlashAttention, 64k context.
#  - CTX=long: fp8 or nvfp4 KV via FlashInfer 0.7.0.post1, extended context capacity.
#
# Fast variant: MODEL defaults to models/Qwen3.8-27B-W4A16-AutoRound-fast when
# present, else models/Qwen3.8-27B-W4A16-AutoRound.
#
# max-num-seqs is 8 by default for low-latency single-user serving.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# flashinfer-cubin (the no-nvcc route, docs/install.md) must match flashinfer-python.
# This used to cover a pair PyPI could not match (cubin 0.6.13 against python
# 0.6.16.post3, #35); install.md now takes the cubin from flashinfer.ai at the
# same version, so the export is kept only for venvs built the old way.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# A dead engine leaves its OffloadingConnector region behind as
# /dev/shm/vllm_offload_*.mmap; the next boot then dies with OSError: Bad
# address in shared_offload_region.py, and under restart policies that loops
# (#33 found 70 ghosts after one host OOM). Unlink stale regions no live
# process maps. VLLM_OFFLOAD_KEEP_SHM=1 skips this (several engines sharing
# /dev/shm across namespaces, where the liveness scan cannot see the owner).
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[start_qwen] removing stale offload region $f"; rm -f "$f"; }
  done
fi
REPO="$(dirname "$DIR")"
cd "$REPO"
# vLLM 0.29 ships FlashInfer 0.6.18, whose JIT passes nvcc flags CUDA 12 does not know
# (--compress-mode=size): on a native install whose system nvcc is older than 13, the first
# boot that compiles a FlashInfer kernel (SPEC=mtp CTX=long's fp8 prefill, for one) dies
# with "nvcc fatal: Unknown option". FlashInfer takes CUDA_HOME before `which nvcc`, and pip
# already put a CUDA 13 toolchain in the venv, so point CUDA_HOME at it. Only when unset and
# the nvcc on PATH is older than 13 (or missing): the Docker image carries CUDA 13 and is
# untouched, and an explicit CUDA_HOME always wins.
if [ -z "${CUDA_HOME:-}" ]; then
  NVCC_MAJOR=$(nvcc --version 2>/dev/null | sed -nE 's/.*release ([0-9]+)\..*/\1/p')
  for CU13 in "$REPO"/venv/lib/python3*/site-packages/nvidia/cu13; do
    if [ -x "$CU13/bin/nvcc" ] && [ "${NVCC_MAJOR:-0}" -lt 13 ]; then
      export CUDA_HOME=$CU13
      echo "[start_qwen] nvcc on PATH is ${NVCC_MAJOR:-missing}, older than the CUDA 13 FlashInfer JIT needs: CUDA_HOME=$CUDA_HOME (set CUDA_HOME to override)"
    fi
    break
  done
fi

# Helper functions inlined for self-contained execution
resolve_vllm_key() {
  if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$REPO/api_key.txt" ]; then
    export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
  fi
}

resolve_bind_host() {
  if [ -n "${HOST:-}" ]; then
    BIND_HOST=$HOST
    if [ -z "${VLLM_API_KEY:-}" ]; then
      case "$HOST" in 127.*|localhost|::1) ;; *)
        echo "WARNING: no API key and HOST=$HOST: anything that can reach this port can use the server. Set VLLM_API_KEY or api_key.txt (openssl rand -hex 24)." >&2 ;;
      esac
    fi
  elif [ -n "${VLLM_API_KEY:-}" ]; then
    BIND_HOST=0.0.0.0
  elif [ -f /.dockerenv ]; then
    BIND_HOST=0.0.0.0
    echo "WARNING: no API key: this container listens on 0.0.0.0, so whatever the published port reaches is open. Set VLLM_API_KEY or publish the port on 127.0.0.1 only." >&2
  else
    BIND_HOST=127.0.0.1
    echo "no API key: binding 127.0.0.1 only. To serve other machines, set VLLM_API_KEY (openssl rand -hex 24 > api_key.txt) or HOST=0.0.0.0." >&2
  fi
}

qwen_int8_exports() {
  if [ -n "$1" ]; then
    export VLLM_MARLIN_INPUT_DTYPE=$1
    if [ -n "$2" ]; then export VLLM_MARLIN_INT8_INCLUDE_RE=$2; fi
  fi
}

qwen_exec() {
  if [ "${PRINT_ARGV:-0}" = 1 ]; then
    printf '%s\n' "$@"
    exit 0
  fi
  exec "$@"
}

QWEN_BOOL_KNOBS="VLLM_DFLASH2_CHAIN VLLM_DFLASH2_CHAIN_GREEDY_ONLY VLLM_DFLASH2_DRAFT_TOPK_TOPP \
  VLLM_DFLASH2_GRAPH_BOTH VLLM_DFLASH2_LOOKUP VLLM_DFLASH2_LOOKUP_ADAPTIVE VLLM_DRAFT_TOPK_TOPP \
  VLLM_EMBED_UVA VLLM_INT4_MQ_3D VLLM_INT4_MQ_3D_DEBUG VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS \
  VLLM_MARLIN_TUNE"

qwen_check_bool_knobs() {
  local k v bad=0
  for k in $QWEN_BOOL_KNOBS; do
    v=${!k-}
    case "$v" in
      ""|0|1) ;;
      *) echo "$k=$v: this knob is read as 1 = on and anything else = off, so set it to 0 or 1 (#310)" >&2; bad=1 ;;
    esac
  done
  [ "$bad" = 0 ] || { echo "refusing to boot with a boolean knob that is neither 0 nor 1" >&2; exit 1; }
}

qwen_kv_offload_args() {
  if [ -n "${KV_OFFLOAD_GB:-}" ] && [ "$KV_OFFLOAD_GB" != 0 ]; then
    case " ${EXTRA_ARGS:-} " in
      *"--kv-offloading-size"*|*"--kv-transfer-config"*)
        echo "KV_OFFLOAD_GB=$KV_OFFLOAD_GB ignored: EXTRA_ARGS already configures a KV connector" >&2 ;;
      *) EXTRA_ARGS="--kv-offloading-size $KV_OFFLOAD_GB ${EXTRA_ARGS:-}" ;;
    esac
  fi
}

qwen_kv_offload_shm_check() {
  local gib need avail
  gib=$(printf %s " ${EXTRA_ARGS:-}" | sed -En 's/.* --kv-offloading-size[= ]([0-9.]+).*/\1/p')
  [ -n "$gib" ] || return 0
  need=$(awk -v g="$gib" 'BEGIN { printf "%d", g * 1073741824 }')
  avail=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -n1 | tr -d ' ')
  if [ -n "$avail" ] && [ "$avail" -lt "$need" ]; then
    echo "WARNING: the KV offload tier needs ${gib} GiB in /dev/shm and $((avail / 1073741824)) GiB is free:" \
         "the boot will fail allocating it. Grow the tmpfs or lower the tier." >&2
  fi
}

resolve_effective_config() {
  [ -n "${REPO:-}" ] || { echo "resolve_config: refusing: REPO is unset" >&2; exit 1; }

  local ctx=${CTX:-} spec=${SPEC:-}
  ctx=${ctx:-fast}; spec=${spec:-mtp}
  case "$ctx" in fast|long) ;;
    *) echo "resolve_config: refusing: unknown CTX=$ctx (want fast|long)" >&2; exit 1 ;;
  esac
  case "$spec" in mtp|dflash2|off|none) ;;
    *) echo "resolve_config: refusing: unknown SPEC=$spec (want mtp|dflash2|off|none)" >&2; exit 1 ;;
  esac
  [ -z "${KV:-}" ] || echo "resolve_config: WARNING: KV=$KV is set but single-user mode ignores it" >&2

  local _shadow="--port | --host | --max-model-len | --max-num-seqs | --gpu-memory-utilization | --kv-cache-dtype | --attention-backend | --served-model-name | --api-server-count | --block-size | --mamba-ssm-cache-dtype | --tool-strict-level "
  local _f
  for _f in $_shadow; do
    case " ${EXTRA_ARGS:-} " in
      *" $_f"*|*" ${_f%= }="*) echo "resolve_config: WARNING: EXTRA_ARGS carries ${_f% } which overrides the launcher's own flag (EXTRA_ARGS expands last)" >&2 ;;
    esac
  done
  case " ${EXTRA_ARGS:-} " in
    *" --api-key "*|*" --api-key="*)
      echo "resolve_config: WARNING: EXTRA_ARGS appears to carry --api-key inline; prefer VLLM_API_KEY or api_key.txt (value not shown)" >&2 ;;
  esac

  local _model=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
  local _port=${PORT:-18031}
  local _key_status="unset"
  [ -n "${VLLM_API_KEY:-}" ] && _key_status="set(${#VLLM_API_KEY} chars)"
  local _keyfile="absent"
  [ -f "$REPO/api_key.txt" ] && _keyfile="present"
  {
    echo "[effective-config] MODE=single"
    echo "[effective-config] CTX=$ctx SPEC=$spec"
    echo "[effective-config] MODEL=$_model PORT=$_port"
    echo "[effective-config] MAX_LEN=${MAX_LEN:-<profile default>} MAX_SEQS=${MAX_SEQS:-<profile default>}"
    echo "[effective-config] VLLM_API_KEY=$_key_status api_key.txt=$_keyfile"
  } >&2
}

resolve_effective_config
qwen_check_bool_knobs
qwen_kv_offload_args
qwen_kv_offload_shm_check

source "$REPO/single-user/select_model.sh"
PORT=${PORT:-18031}
MAX_SEQS=${MAX_SEQS:-}
# Seconds between SSE ': keep-alive' comment lines on a streaming response, so
# an idle stream survives a proxy's idle timeout during a long prefill (Bifrost
# defaults to 120 s; 30 s clears it with a 4x margin). SSE_KEEP_ALIVE=0 passes
# the flag with the interval vLLM reads as off; SSE_KEEP_ALIVE= (empty) drops
# the flag entirely, which is what a vLLM tree WITHOUT patches/sse-keep-alive.patch
# applied needs — the flag does not exist there, and the deploy tree drifts.
SSE_KEEP_ALIVE=${SSE_KEEP_ALIVE-30}   # no colon: SSE_KEEP_ALIVE= keeps the empty value
# INT8_ACT=int8 turns on the W4A8 Marlin path (weights stay int4, activations
# quantized per token to int8, int8 tensor cores) for the layers INT8_LAYERS
# selects — the same knob batch mode ships on by default. At batch size 1 it
# does nothing for decode (memory-bound; C1 122±5 vs 121 tok/s over repeats,
# tok/step 3.2 either way) but prefill is compute-bound at every concurrency
# — a 4k prefill on this stack is 79% Marlin GEMM time with 15 ms of GPU idle
# (torch profile), so the GEMM dtype is the whole game. Measured on the
# seeded protocol (upstream bench/run_benchmarks.sh --prefill), dflash2 k=15, PC=1:
#
#   prefill tok/s        1k     4k     16k    51k
#   W4A16 (default)    1,437  1,494  1,410  1,200
#   INT8_LAYERS=mlp    1,638  1,696  1,587  1,320   (+14/+13/+13/+10%)
#   INT8_LAYERS=all*   1,845  1,937  1,791  1,423   (+29/+30/+27/+19%)
#
#   *all = "mlp|linear_attn|self_attn" — every linear except the int8-weight
#    lm_head/embed and the MTP module. The 51k row gains least because 16
#    full-attention layers grow quadratically (~40% of time at 51k) and FA2 at
#    head_dim 256 has no faster sm86 alternative (FlashInfer measured within
#    1.5%).
#
# Quality (upstream bench/quality_battery.py, this stack): all-linears = GSM8K 95.0%
# (fast-variant baseline 96.5), PPL 8.423 vs 8.095 (+4.1%, mostly prose;
# code ~flat); mlp = the batch-mode trade (+2.2% PPL, GSM8K 95.0). IFBench
# was flat on the batch int8 default. "mlp|linear_attn" (GDN-only middle
# point) crashes at first forward on this torch/vllm combo — an inductor
# codegen bug with the mixed set; use mlp or all.
INT8_ACT=${INT8_ACT-}
INT8_LAYERS=${INT8_LAYERS-mlp|linear_attn|self_attn}
qwen_int8_exports "$INT8_ACT" "$INT8_LAYERS"
# PREFILL_ATTN=int8: int8-QK Triton attention for the hd256 full-attention
# layers during prefill (retired patches/prefill-attn-int8.patch on v0.31;
# VLLM_PREFILL_ATTN remains registered in speed-knobs-envs.patch): 1.27-1.35x FA2 on
# the attention itself, worth up to ~+5% end-to-end at 51k on top of INT8_ACT
# (1,839/1,498 tok/s at 16k/51k with both on). It is a companion to INT8_ACT, not
# a standalone switch: on its own it moves prefill +0.3/+1.1/+3.3% at 4k/16k/51k
# here (1,167/1,126/1,012 against 1,164/1,114/980 stock, two interleaved arms
# reproducing to 0.1%), because without the int8 GEMMs attention is a smaller
# share of prefill. A WSL2 3090 measured it 1-6% *negative* standalone (#62), so
# the honest range for PREFILL_ATTN alone is "within a few percent either way" --
# set INT8_ACT with it or leave it off. Prefill-only; decode and the
# split-KV verify keep their existing paths. fp16 selects the same kernel
# without quantization (a debugging mode); empty keeps FA2.
PREFILL_ATTN=${PREFILL_ATTN-}
[ -n "$PREFILL_ATTN" ] && export VLLM_PREFILL_ATTN=$PREFILL_ATTN
# 0.93 here, NOT batch mode's 0.972: the DeltaNet workspace in the MTP decode
# path allocates beyond the startup memory profile (upstream docs/gotchas.md, gotcha 4).
GPU_UTIL=${GPU_UTIL:-0.93}
API_SERVERS=${API_SERVERS:-1}
# CTX=fast (default): bf16 KV via FlashAttention, ~64k context, 4 drafts (~+7%).
# CTX=long: fp8 or nvfp4 KV via FlashInfer, extended context capacity.
CTX=${CTX:-fast}
# SPEC=mtp (default): Qwen's own MTP head, k drafts chained.
# SPEC=dflash2: the DFlash2 block drafter.
# SPEC=off (or none): no speculative decoding.
SPEC=${SPEC:-mtp}
USER_MAX_LEN=${MAX_LEN:-}
# CTX validation lives in resolve_effective_config (called above), which refuses
# unknown values before anything boots.
if [ "$CTX" = "fast" ]; then
  MAX_LEN=${MAX_LEN:-65536}
  DRAFT_TOKENS=${DRAFT_TOKENS:-4}
  ATTN_ARGS="--attention-backend FLASH_ATTN --kv-cache-dtype bfloat16"
elif [ "$CTX" = "long" ]; then
  MAX_LEN=${MAX_LEN:-150000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--attention-backend FLASHINFER --kv-cache-dtype ${KV_DTYPE:-nvfp4}"
fi
if [ "$SPEC" = "dflash2" ] && [ "$CTX" != "fast" ] && [ "$CTX" != "long" ]; then
  echo "SPEC=dflash2 supports CTX=fast (bf16, 64k) and CTX=long; CTX=$CTX keeps SPEC=mtp" >&2
  SPEC=mtp
fi
if [ "$SPEC" = "dflash2" ]; then
  if [ -z "$DRAFT" ]; then
    for d in Qwen3.8-27B-DFlash2-W4A16 Qwen3.8-27B-DFlash2; do
      [ -f "$REPO/models/$d/model.safetensors" ] && DRAFT=$REPO/models/$d && break
    done
  fi
  [ -n "$DRAFT" ] || { echo "SPEC=dflash2 needs the drafter under models/ (e.g. models/Qwen3.8-27B-DFlash2-NVFP4-YourHighnessLA)" >&2; exit 1; }
  # Lookup-augmented drafting: when the model is reproducing something from its context,
  # draft from the context instead of from the drafter
  # (upstreamed at 0.32.0 from patches/dflash2-lookup-drafting.patch).
  export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
  # DFLASH_TOKENS is the *verify* block, which no longer has to equal the drafter's: the
  # DFlash2 checkpoint always proposes the 7 tokens it was trained for, and any position
  # past that is filled from the request's own context, costing the drafter nothing. The
  # block length is adaptive -- the long block is only scheduled while the lookup is
  # actually firing -- so ordinary steps still verify 8 tokens.
  #
  # DFLASH_TOKENS=15 is "reproduction mode": +50% where the model reproduces its context
  # (388 vs 259 tok/s reproducing a document verbatim) and +9% on the short-prompt C1 set,
  # against 3-20% on long-context work that mixes prose with quoting, 4 request slots
  # instead of 8 and 56k of context instead of 64k. Worth setting for a coding assistant
  # applying edits or a RAG front-end quoting sources; the default stays 7.
  DRAFT_TOKENS=${DFLASH_TOKENS:-7}
  # draft_sample_method is NOT optional here, and the default is the wrong one (#73).
  # The 0.27.1 fork allocated the draft-logits buffer unconditionally in its own
  # speculator; the 0.28 port inherits the upstream base class, which allocates it only
  # when the config asks. Without it the rejection test loses its denominator -- the
  # draft probability is pinned to 1 and acceptance is strictly stricter. Measured on the
  # reference 3090, CTX=fast k=15: 2.66 tok/step and 101.4 tok/s unset against 3.23 and
  # 121.7 set, with 3.19/120.5 on 0.27.1. The boot log says which you got: draft_logits.
  # The mtp branch below has always set it.
  # DRAFT_METHOD=dspark runs a DSpark drafter (RadixArk/Qwen3.8-27B-DSpark, seven drafts per
  # step like the shipped head) through the same profile; it needs
  # patches/dspark-draft-quant-config.patch and the architecture rename in the README.
  SPEC_CFG="{\"method\":\"${DRAFT_METHOD:-dflash}\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  # The ADAPTIVE verify length corrupts a prefix-cache hit under KVarN. When the block
  # alternates 8<->16 step to step and the request resumed from a cache hit, turn 2 over
  # the same document tracks the source for ~38 characters and then diverges -- turn 1 is
  # correct. Deterministic, at five of six prompt residues tested.
  #
  # It is the length CHANGING, not the lookup content, and not the KVarN kernels. Fresh
  # server per row, three requests each (one to arm the cache, two measured), sha-compared:
  #   baseline, adaptive 8<->16          self-hit 38/793 WRONG   12.71 tok/step
  #   VLLM_DFLASH2_LOOKUP_ADAPTIVE=0     self-hit 794/794 clean  14.71 tok/step
  #   PREFIX_CACHE=0, adaptive on        self-hit 794/794 clean  14.33
  #   KVARN_FUSED_VERIFY=0, adaptive on  self-hit 38/793 WRONG   12.71
  # and the constant-length setting is clean at every residue that broke (16/64/96/124),
  # cold and warm, byte-identical.
  #
  # Note what the earlier controls actually varied: DFLASH_TOKENS=7, LOOKUP=0 and SPEC=mtp
  # all make the block a CONSTANT length, so "clean" there was never evidence about the
  # block being short or the lookup being off. And a wrong draft cannot corrupt greedy
  # output at all -- rejection_sampler.py emits target_argmax whether it accepts or not --
  # so the draft content was never a candidate. The damage is on the target's forward.
  #
  # Pinning the length is not a sacrifice here: 14.71 tok/step is the FASTEST number in the
  # series, above the adaptive path's own 14.29 cold. Root cause still open; this is a
  # correct and fast setting, not a workaround with a cost.
  if [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ]; then
    # Adaptive block length means the worker tells the scheduler how many draft tokens to
    # put up for verification next step, and vLLM only feeds that back on the synchronous
    # scheduling path (async scheduling pads every decode step to num_speculative_tokens).
    # Measured cost of losing async scheduling at batch 1: under 1%.
    ASYNC_SCHED=${ASYNC_SCHED:-0}
  fi
  # Tensor parallelism changes two calibrations below, both measured in #40
  # (controlled 1-vs-2x3090 A/B on this harness):
  #
  # 1. The pinned KV_MEM is a 24-GiB-single-card constant and vLLM applies
  #    --kv-cache-memory PER WORKER, so at TP=2 the pin strands ~8 GiB per card:
  #    137,210 tokens of pool where GPU_UTIL sizing gets 302,223, with every
  #    decode delta inside run-to-run spread. Under TP>1, unless the user pinned
  #    one, size from GPU_UTIL instead. (The pin exists because the single-card
  #    transient margin is sharp -- gotcha 39; TP halves the per-card footprint
  #    and the same reporter's boxes boot clean unpinned.)
  # 2. DFLASH_TOKENS>7 relies on the lookup lane filling the verify tail; at
  #    TP=2 the one datapoint so far (#40) shows the tail accepting nothing and
  #    the wider block costing -27% at C1. Until that is diagnosed, warn.
  TP_SIZE=1
  case " ${EXTRA_ARGS:-} " in *"-tensor-parallel-size"*|*" -tp "*)
    TP_SIZE=$(printf %s " $EXTRA_ARGS" | sed -En "s/.* (--tensor-parallel-size[= ]|-tp )([0-9]+).*/\2/p")
    TP_SIZE=${TP_SIZE:-1}
  ;; esac
  if [ "$TP_SIZE" -gt 1 ]; then
    if [ -z "${KV_MEM+x}" ]; then
      echo "[start_qwen] tensor-parallel-size $TP_SIZE: skipping the single-card KV_MEM" \
           "pin, sizing the KV pool from GPU_UTIL=$GPU_UTIL (issue #40; export KV_MEM" \
           "to pin it, KV_MEM= for this behavior explicitly)."
      KV_MEM=
    fi
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      echo "[start_qwen] WARNING: DFLASH_TOKENS=$DRAFT_TOKENS at TP=$TP_SIZE lost 27% at" \
           "C1 in the one A/B so far (#40): the lookup-filled verify tail accepted" \
           "nothing there. Until diagnosed, DFLASH_TOKENS=7 is the measured setting" \
           "for TP>1." >&2
    fi
  fi
  # Memory: patches/hybrid-kv-groups-v2-cudagraph.patch stops the drafter's 5
  # sliding-window layers from padding the target's attention/GDN layers (78 instead of
  # 105 KB of pool per token), which is what makes 64k reachable here. The V2 runner's
  # profiled activation peak swings ~1 GiB between starts, so the pool is pinned by bytes
  # rather than by gpu-memory-utilization: 5.2 GiB -> 69,758 tokens = 1.06x at 64k,
  # leaving ~1.1 GiB for transients (the same margin MTP mode runs with). Soak-tested
  # with a 60k prompt, 4x16k concurrent and 8x4k generations. Lower it if you also run
  # something else on the card; KV_MEM= (empty) falls back to GPU_UTIL.
  #
  # A longer verify block costs pool twice: bigger CUDA graphs, and one aligned recurrent
  # state page per speculative block. That second term is what scales -- NOT the slot
  # count: 1 slot and 8 slots differ by about 8 MiB in total, so cutting MAX_SEQS buys no
  # context. Single-user mode keeps 4 slots when the block is long for the graphs.
  if [ "$CTX" = "long" ]; then
    # int8 KV: measured 136,429 tokens of pool at DFLASH_TOKENS=7 with prefix caching on
    # (138,696 without), against bf16's 69,758 in the same pinned 5.2 GiB. DFLASH_TOKENS>7
    # at this context is untested -- the graphs and the state pages both grow.
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-${USER_MAX_LEN:-131072}}
    KV_MEM=${KV_MEM-5583457484}
  elif [ "$DRAFT_TOKENS" -gt 7 ]; then
    # 4 slots and 56k instead of 8 and 64k: the aligned state pages and the bigger decode
    # graphs are what the long block costs, and this is where they still fit next to the
    # 5.2 GiB pool (57,669 tokens). DFLASH_TOKENS=7 gets 8 slots and 64k back.
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-${USER_MAX_LEN:-57344}}
    KV_MEM=${KV_MEM-5583457484}
    # Decode graphs are captured for both block lengths (the drafter's and the full verify
    # block), or the short step -- the common one -- runs piecewise and costs 8%. That is
    # 1.8 GiB of graphs instead of 1.45; 0.29 accounts for it in its own profiling.
  else
    MAX_LEN=${DFLASH_MAX_LEN:-${USER_MAX_LEN:-65536}}
    KV_MEM=${KV_MEM-5583457484}
  fi
  # The two memory budgets above (the pinned KV_MEM and GPU_UTIL; on 0.29 the runner's
  # CUDA-graph memory is profiled, not reserved by a knob) are sized for the shipped 1.2 GiB W4A16 head with about a
  # gigabyte to spare on a 24 GiB card, and none of them knows the drafter's size. A bf16
  # community drafter (3.5 to 4.0 GiB) does not fit them at any context: the floor is the
  # per-request state, not the token count, and a bigger drafter is charged twice, as
  # weights and as a larger per-request KV requirement (about 4.97 GiB per 65536-token
  # request beside the shipped head, derived from a served boot's pool of 68,605 tokens at
  # the 5.2 GiB pin; 7.05 GiB beside a 3.5 GiB drafter, logged at the refusal). None of
  # the engine's error messages names the pin, the reservation or DFLASH_MAX_LEN, so say
  # it here instead of after seven boots (#25, items 13 and 14). Measured to serve both
  # community drafters on 24 GiB: KV_MEM=3000000000 DFLASH_MAX_LEN=8192 (pinned, 4090), or
  # GPU_UTIL=0.97 KV_MEM= DFLASH_MAX_LEN=8192 (unpinned, 3090); the pinned profile costs the
  # shipped head nothing measurable at width 7. Warning, not a refusal: a bigger card has
  # room where 24 GiB does not.
  DRAFT_BYTES=$(du -sb "$DRAFT" 2>/dev/null | cut -f1)
  if [ -n "$DRAFT_BYTES" ] && [ "$DRAFT_BYTES" -gt 2147483648 ]; then
    DRAFT_GIB=$(( (DRAFT_BYTES + 536870912) / 1073741824 ))
    if [ -n "$KV_MEM" ] && [ -z "${DFLASH_MAX_LEN:-}" ] && [ -z "$USER_MAX_LEN" ]; then
      echo "[start_qwen] WARNING: the drafter at $DRAFT is about ${DRAFT_GIB} GiB of weights; the" \
           "memory defaults (KV_MEM=$KV_MEM pinned, GPU_UTIL=$GPU_UTIL," \
           "MAX_LEN=$MAX_LEN) are sized for the shipped 1.2 GiB head and a 24 GiB card, and a" \
           "drafter this size does not fit them at any context (the per-request state is the" \
           "floor, #25 item 13). Measured to serve a 4 GiB bf16 drafter on 24 GiB:" \
           "KV_MEM=3000000000 DFLASH_MAX_LEN=8192, or GPU_UTIL=0.97 KV_MEM= DFLASH_MAX_LEN=8192." \
           "On WSL2 an over-committed pin does not fail, it runs 3-6x slower (#25 item 14)." >&2
    else
      echo "[start_qwen] note: drafter $DRAFT is about ${DRAFT_GIB} GiB of weights, beside" \
           "KV_MEM=${KV_MEM:-unpinned} MAX_LEN=$MAX_LEN GPU_UTIL=$GPU_UTIL."
    fi
  fi
  MAX_SEQS=${MAX_SEQS:-8}
  # The V2 model runner captures decode graphs in multiples of k+1 tokens: cover MAX_SEQS
  # requests, but never ask for more than 64 query tokens' worth. Every default in this
  # script lands on 64 or below (8x8 at k=7, 4x16 at k=15), and the runner's graph memory
  # (profiled on 0.29) is sized for that. `DFLASH_TOKENS=15 MAX_SEQS=8` asks for 128, which boots and
  # then dies on the first concurrent batch -- torch.OutOfMemoryError inside the engine,
  # EngineDeadError, every request 500 while /health still answers. Past the cap the
  # bigger batches run piecewise instead of captured: slower, alive. Set CG explicitly to
  # override.
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1) > 64 ? 64 : MAX_SEQS * (DRAFT_TOKENS + 1)))}
  # Seats are admissions, not residency. Every RESIDENT request needs 1+k recurrent-state
  # slots out of the same pool before it stores one token of context: 0.88 GiB at
  # DFLASH_TOKENS=7 and 1.66 GiB at 15 (gotcha 33), i.e. ~0.098 GiB per (k+2), which the
  # ramp in upstream bench/conc_ladder.py confirms live (15.8% of the CTX=fast pool per request,
  # six resident, the seventh preempted). Print what the pool can actually hold so nobody
  # reads MAX_SEQS as a concurrency setting -- raising it past this queues, and once the
  # pool is full it preempts and recomputes.
  if [ -n "$KV_MEM" ]; then
    RESIDENT=$(( KV_MEM / ((DRAFT_TOKENS + 2) * 104988089) ))
    [ "$RESIDENT" -lt 1 ] && RESIDENT=1
    if [ "$MAX_SEQS" -gt "$RESIDENT" ]; then
      echo "[start_qwen] note: the pinned pool holds about $RESIDENT resident requests at" \
           "DFLASH_TOKENS=$DRAFT_TOKENS (state pages), fewer with long prompts;" \
           "MAX_SEQS=$MAX_SEQS admits more than that and the rest queue."
    fi
  fi
  KV_STOCK=5583457484
  if [ -n "$KV_MEM" ] && [ "$KV_MEM" -gt "$KV_STOCK" ]; then
    echo "[start_qwen] WARNING: KV_MEM=$KV_MEM is above the profile default $KV_STOCK." \
         "The extra pool is taken from the headroom prefill and the first concurrent" \
         "batch allocate from. Ladder 4k/16k TTFT against known-good rates before trusting it." >&2
  fi
elif [ "$SPEC" = "off" ] || [ "$SPEC" = "none" ]; then
  MAX_SEQS=${MAX_SEQS:-8}
  SPEC_CFG=""
  CG=${CG:-32}
elif [ "$SPEC" = "mtp" ]; then
  MAX_SEQS=${MAX_SEQS:-8}
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  CG=${CG:-32}
else
  echo "SPEC=$SPEC is not a mode: mtp (default), dflash2, off. Refusing rather than" >&2
  echo "silently running mtp -- an unrecognized SPEC in an A/B measures the wrong thing." >&2
  exit 1
fi
# The pin applies in every mode. The dflash2 branch above gives KV_MEM a default; under
# mtp and off it is only ever the user's own export (upstream docs/multi-gpu.md's "pin the KV
# pool"). It used to be passed inside the dflash2 branch alone, so SPEC=mtp and SPEC=off
# ignored an exported KV_MEM without a word (#68, #210).
[ -n "${KV_MEM:-}" ] && EXTRA_ARGS="--kv-cache-memory=$KV_MEM ${EXTRA_ARGS}"
SPEC_ARGS=()
[ -n "$SPEC_CFG" ] && SPEC_ARGS=(--speculative-config "$SPEC_CFG")

# PREFIX_CACHE=1: reuse the KV of a shared prompt prefix across requests, and resume the
# recurrent (GDN) state from the last cached block boundary instead of re-running the prompt.
# Turn-2+ of a chat with a 24k document goes from ~23 s to ~1 s; costs one extra state page
# per request (~16% of the KV pool). Hybrid models keep this opt-in upstream.
if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
  if [ -n "$SPEC_CFG" ]; then
    case " ${EXTRA_ARGS:-} " in
      *"--prefix-cache-retention-interval"*) ;;
      *)
        RETENTION=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL-${PREFIX_RETENTION:-None}}
        EXTRA_ARGS="--prefix-cache-retention-interval ${RETENTION:-None} ${EXTRA_ARGS}" ;;
    esac
    unset VLLM_PREFIX_CACHE_RETENTION_INTERVAL
  fi
fi
# An explicit CUDAGRAPH_MODE is honoured on every path, including dflash2, which
# otherwise takes vLLM's default. Without this branch there was no way to ask a
# dflash2 server for PIECEWISE, so the FULL-against-PIECEWISE comparison this repo
# quotes could not be re-measured after a75ee4b changed what FULL does.
if [ -n "${CUDAGRAPH_MODE:-}" ] && [ -z "${CG_MODE:-}" ]; then
  CG_MODE=",\"cudagraph_mode\":\"$CUDAGRAPH_MODE\""
fi

# ASYNC_SCHED=0 (set above for a long DFlash2 verify block) runs the scheduler
# synchronously, which is the only path on which vLLM lets the worker choose how many draft
# tokens to put up for verification. Note --async-scheduling is already the default in
# 0.28.0: --no-async-scheduling is what turns it off.
# Array, not $( ... || echo ... ): errexit-safe via the fallback, but the
# unquoted expansion word-splits; match METRICS_ARGS below.
ASYNC_ARGS=(--no-async-scheduling)
[ "${ASYNC_SCHED:-1}" = 1 ] && ASYNC_ARGS=(--async-scheduling)

# Tool / function calling. Without BOTH flags vLLM rejects any request carrying
# `tools` with tool_choice "auto": 400 '"auto" tool choice requires
# --enable-auto-tool-choice and --tool-call-parser to be set'. TOOLS=0 turns it off.
#
# qwen3_coder is a deliberate choice for this model, not a vLLM default and not a
# leftover -- do NOT "correct" it to hermes. The parser has to match the format the
# chat template asks the model for, and Qwen3.8's asks for XML --
# <tool_call><function=NAME><parameter=K>V</parameter> -- NOT the JSON body that
# hermes, the usual answer for a Qwen model, reads. Getting that wrong does not
# error: the call comes back as ordinary content and the client sees no tool_calls,
# which reads as the model being bad at tools rather than as a misconfigured server.
# The name is the call format, not the checkpoint -- nothing here is Qwen3-Coder.
# qwen3_coder, qwen3_xml and mimo are three names for one Qwen3EngineToolParser in
# 0.28.0, which is the tool-side adapter of the same parser engine that
# --reasoning-parser qwen3 already uses (vllm/parser/qwen3.py).
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
# Array, not $( [ ] && echo ): exits 1 when TOOLS is off (the shape #59 fixed)
# and word-splits $TOOL_PARSER; the array keeps the parser as one element.
TOOL_ARGS=()
# --tool-strict-level (new in vLLM 0.31): its default, auto, enforces a tool's argument schema only when the tool
# says strict: true, so a forced call (tool_choice=required) to a tool with no parameters invented arguments on
# 0.31 in 20 of 20 seeds where 0.30 sent {}; parameter holds every tool to its schema, as 0.30 did, and sent {}
# 20 of 20. TOOL_STRICT=auto restores upstream's default; a --tool-strict-level in EXTRA_ARGS wins.
[ "${TOOLS:-1}" = 1 ] && TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --tool-strict-level "${TOOL_STRICT:-parameter}")

# REQ_METRICS=1: per-request timing fields in every response plus usage on
# every request (--enable-per-request-metrics --enable-force-include-usage,
# issue #51; llama-swap reads them). Off by default only because the timing
# fields ride on the engine-stats path, so it cannot be paired with
# --disable-log-stats in EXTRA_ARGS. --enable-prompt-tokens-details is always
# on. The per-request spec-decode summary is in 0.29.0 and rides with REQ_METRICS=1.
# Array, not $( [ ] && echo ): that substitution exits 1 when the test is
# false, which kills a launcher running under `set -e` silently (#59).
METRICS_ARGS=()
if [ "${REQ_METRICS:-0}" = 1 ]; then
  # vLLM 0.29.0: per-request speculative-decoding acceptance metrics ride in the response under
  # metrics.speculative_decoding (n == 1 only; the field is experimental, shape as of v0.29.0). summary
  # is mean acceptance length, draft acceptance rate and the step histogram; REQ_METRICS_DETAILED=1
  # adds the ordered per-step accepted/proposed arrays, which upstream says is not free, so it is a
  # separate opt-in and off in every profile anyone benchmarks (#66, #75, gotcha 53).
  SPEC_METRICS=summary; [ "${REQ_METRICS_DETAILED:-0}" = 1 ] && SPEC_METRICS=detailed
  METRICS_ARGS=(--enable-per-request-metrics --enable-force-include-usage
                --per-request-spec-decode-metrics "$SPEC_METRICS")
fi

# Vision. --language-model-only drops the vision tower cleanly -- no weights loaded,
# 0.858 GiB on this checkpoint (gotcha 9) -- and stays the default. VISION=1 keeps
# the tower, for a client that sends images: screenshots into a coding assistant,
# captioning, document photos.
#
# Only --language-model-only needs a knob. It is hardcoded in the exec line below, so
# the alternative is countering it with --no-language-model-only from EXTRA_ARGS and
# depending on which flag argparse saw last -- which regresses silently: images are
# still accepted and still counted as prompt tokens, and the model answers from
# placeholder embeddings. The two flags VISION=1 adds have no such conflict and can
# be overridden from EXTRA_ARGS, which is expanded after them. The pixel cap is
# shipped rather than left to the processor default because vLLM profiles the encoder
# at the largest image it will accept, and that peak comes out of the KV pool:
# 2097152 px = 2048 image tokens.
if [ "${VISION:-0}" = 1 ]; then
  VISION_ARGS='--limit-mm-per-prompt {"image":{"count":1}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'
  # VISION_OFFLOAD keeps the tower's weights in pinned host RAM and copies each module to
  # the GPU for the duration of its own forward (patches/vision-tower-cpu-offload.patch).
  # It defaults ON, because on 24 GB SPEC=dflash2 + VISION=1 does not boot without it:
  # the tower is 0.85 GiB of the ~1.1 GiB transient margin the KV_MEM comment sizes, and
  # graph capture then dies allocating the split-KV verify buffer --
  #   torch.OutOfMemoryError: Tried to allocate 960.00 MiB ... 787.50 MiB is free
  #     (spec_decode_attn.py:184, self.part_o)
  # measured here, VISION=1 VISION_OFFLOAD=0 SPEC=dflash2, RTX 3090 at 250 W. With the
  # offload the same config comes up with the full 69,758-token pool and reads images.
  #
  # It is close to free: isolated-tower measurement at PCIe 4.0 x16, one 8192-patch image,
  # median of 10 forwards, 891.3 -> 9.0 MiB of resident weights and 1160.5 -> 308.2 MiB of
  # peak allocation for 296 -> 333 ms of encode, output bit-exact either way. Set
  # VISION_OFFLOAD=0 only on a card with room to spare, where 36 ms per image buys nothing.
  [ "${VISION_OFFLOAD:-1}" = 1 ] && export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
else
  VISION_ARGS="--language-model-only"
fi
# EMBED_UVA=1 keeps the quantized token embedding in pinned host RAM, read through a CUDA UVA view
# (patches/qwen3_5-embed-uva.patch, VLLM_EMBED_UVA). The freed VRAM goes to the KV pool: +15.7% tokens
# (204,336 to 236,479) at SPEC=mtp CTX=long on a 3090 at PCIe 4.0 x16, decode and prefill unchanged, for
# about 2.1 GiB of host RAM and 20 s of boot (#281). Off by default: a card with spare VRAM pays a PCIe
# read per decode step for nothing, and a narrow slot is unmeasured.
[ "${EMBED_UVA:-0}" = 1 ] && export VLLM_EMBED_UVA=1

# fp16 activations do not work with the speculative path, and the way you find that out
# is late and cryptic (#27): the split-KV verify kernel hardcodes tl.bfloat16 for the
# query cast and the dot accumulate (patches/spec-decode-attn.patch:217,236), so it
# fails to *compile* at first attention with "Both operands must be same dtype. Got bf16
# and fp16" -- a triton CompilationError that never names the dtype you set.
#
# Scope, stated honestly because an earlier version of this comment overreached: the
# BLOCKER I can point at is that kernel. @ahnguyen17 also hit a device-side assert in
# rejection_sample() with SPEC_ATTN=0, and I first wrote that up as a second bf16
# assumption -- it is not. rejection_sampler_utils.py contains zero bf16/fp16 literals;
# it promotes to tl.float32 on load and allocates its buffers float32/int64, so it is
# dtype-agnostic. That assert has some other cause (it was seen on a w8a16 GPTQ target
# whose GEMM path was already suspect, and a device-side assert reads like an
# out-of-bounds index, not a dtype mismatch), and it is not established.
#
# So this refuses the combination rather than claiming to enumerate why it breaks:
# fp16 + a speculator is untested here and its default path is bf16-only. SPEC=none is
# the escape hatch -- a limitation of these kernels, not of any checkpoint.
case " ${EXTRA_ARGS:-} " in
  *" --dtype float16 "*|*" --dtype=float16 "*|*" --dtype fp16 "*|*" --dtype=fp16 "*|*" --dtype half "*|*" --dtype=half "*)
    if [ "$SPEC" != "none" ] && [ "$SPEC" != "off" ]; then
      echo "--dtype float16 needs SPEC=off: this repo's Qwen speculative path is bf16-only (see upstream issue #27)." >&2
      exit 1
    fi ;;
esac

export PATH="$REPO/venv/bin:$PATH"
# expandable_segments needs CUDA VMM, which WSL2's paravirt driver rejects during
# Marlin repack. It is the single most reported failure on Windows (#2, #26) and it
# does not announce itself as an allocator problem -- the same VMM rejection surfaces
# as "CUDA driver error: device not ready", "CUDA driver error: out of memory" (on a
# card with 23 GiB free for a 16 GiB model) or a torch stable-ABI error out of
# aten::empty, with dmesg carrying "dxgkio_make_resident: Ioctl failed: -12". So
# detect WSL and default it off there rather than documenting a workaround: three
# separate reporters found the note in upstream docs/docker.md only after losing a day.
# Still overridable both ways, and untouched on native Linux, where gotcha 3 applies
# and turning it off costs you the top of the GPU_UTIL range.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
  ALLOC_DEFAULT=expandable_segments:False
  if [ "${SPEC:-mtp}" = dflash2 ] && [ -z "${VLLM_WSL2_ENABLE_PIN_MEMORY:-}" ]; then
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    echo "WSL detected + SPEC=dflash2: VLLM_WSL2_ENABLE_PIN_MEMORY=1 (V2 runner needs pinned UVA buffers here; set it explicitly to override)"
  fi
  [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && echo \
    "WSL detected: PYTORCH_CUDA_ALLOC_CONF=$ALLOC_DEFAULT (VMM breaks Marlin repack under the paravirt driver; set it explicitly to override)"
else
  ALLOC_DEFAULT=expandable_segments:True
fi
# The CPU offload tier (--kv-offloading-size in EXTRA_ARGS, or any --kv-transfer-config) is a KV connector, and
# vLLM 0.28 refuses every KV connector under expandable_segments:True unless the cumem allocator is on: the VMM
# allocator can move KV pages out from under the connector's pinned copies. On WSL2 the default above already
# avoids it; on native it is the default, so the tier could not boot with the launcher's defaults (#95).
case " ${EXTRA_ARGS:-} " in
  *"--kv-offloading-size"*|*"--kv-transfer-config"*)
    [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [ "$ALLOC_DEFAULT" = expandable_segments:True ] && echo "KV connector in EXTRA_ARGS: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (vLLM rejects the connector under VMM; set it explicitly to override)"
    ALLOC_DEFAULT=expandable_segments:False ;;
esac
# vLLM's custom all-reduce exports its graph buffers over CUDA IPC
# (`cudaIpcGetMemHandle`, csrc/custom_all_reduce.cuh:164) and an expandable
# (VMM) segment has no handle to export, so at TP>1 with CUDA graphs capture
# aborts with "Cuda error ... 'invalid argument'" and the worker dies before
# the server is up (#163, 2x3090 NVLink). --disable-custom-all-reduce also
# clears it by handing the collectives to NCCL, but that arm measured 6.4%
# slower at C1 on the reporting box, so default the allocator off and keep
# custom all-reduce. Skipped when the run already disables it or runs eager:
# neither captures a graph buffer to export.
case " ${EXTRA_ARGS:-} " in
  *"--disable-custom-all-reduce"*|*"--enforce-eager"*) ;;
  *"--tensor-parallel-size"*|*" -tp "*)
    ALLOC_TP=$(printf %s " ${EXTRA_ARGS:-}" | sed -En "s/.* (--tensor-parallel-size[= ]|-tp )([0-9]+).*/\2/p")
    if [ "${ALLOC_TP:-1}" -gt 1 ] 2>/dev/null; then
      if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [ "$ALLOC_DEFAULT" = expandable_segments:True ]; then
        echo "tensor-parallel-size $ALLOC_TP: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (custom all-reduce cannot export a VMM graph buffer over CUDA IPC, #163; set it explicitly to override)"
      fi
      ALLOC_DEFAULT=expandable_segments:False
    fi ;;
esac
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-$ALLOC_DEFAULT}
export VLLM_USE_FLASHINFER_SAMPLER=0

resolve_vllm_key
resolve_bind_host

qwen_exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host $BIND_HOST --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  ${VISION_ARGS} \
  $ATTN_ARGS \
  --mamba-ssm-cache-dtype float16 \
  "${ASYNC_ARGS[@]}" \
  --max-num-batched-tokens 2048 \
  "${SPEC_ARGS[@]}" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]${CG_MODE}}" \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  "${METRICS_ARGS[@]}" \
  "${TOOL_ARGS[@]}" \
  ${EXTRA_ARGS} \
  ${SSE_KEEP_ALIVE:+--sse-keep-alive-interval $SSE_KEEP_ALIVE}
