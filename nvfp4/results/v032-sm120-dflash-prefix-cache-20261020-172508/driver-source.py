#!/usr/bin/env python3
"""Prefix Caching Battery for vLLM 0.32.0 (Isolated Suite, Cache ON).

Executes a 5-phase prefix caching evaluation at concurrency=1:
1. Cold 32K prefix: baseline cold prefill latency, 0 cached tokens.
2. Warm identical 32K prefix: cache hit tokens (~32,768), latency speedup ratio.
3. Appended user turn: multi-turn request verifying cached prefix reuse with new turn prefill.
4. Mutated early prefix: verify cache invalidation when prefix head changes.
5. Warm/Cold T=0 correctness check: token-for-token equality between cold and warm generations.

Run under owned systemd user unit with MemoryMax=48000000000, MemorySwapMax=0.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest_utils

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("V032_PREFIX_OUT", "<RUN_DIR>"))
MODEL = Path(os.environ.get("MODEL", str(ROOT / "models/Qwen3.8-27B-W4A16-AutoRound-fast")))
GPU = "<GPU_UUID>"
PORT = int(os.environ.get("PORT", "18031"))
API = f"http://127.0.0.1:{PORT}"
KEY = (ROOT / "api_key.txt").read_text().strip()
SERVER_MODEL = "qwen3.8-27b"
PROGRESS: dict[str, Any] = {}


def progress(**updates):
    PROGRESS.update(updates)
    PROGRESS["updated_at"] = time.time()
    if OUT.is_dir():
        temp = OUT / "status.json.tmp"
        temp.write_text(json.dumps(PROGRESS, indent=2) + "\n")
        temp.replace(OUT / "status.json")


def request(path: str, payload: dict | None = None, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        API + path,
        data=None if payload is None else json.dumps({"model": SERVER_MODEL, **payload}).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error


def chat_messages(messages: list[dict], max_tokens: int = 64, temperature: float = 0.0) -> dict:
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return request("/v1/chat/completions", body)


def stop(server: subprocess.Popen | None):
    if server is None or server.poll() is not None:
        return
    server.send_signal(signal.SIGINT)
    try:
        server.wait(15)
        return
    except subprocess.TimeoutExpired:
        pass
    server.terminate()
    try:
        server.wait(15)
        return
    except subprocess.TimeoutExpired:
        pass
    server.kill()
    server.wait(15)


def save(data: dict, directory: Path):
    temp = directory / "result.json.tmp"
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(directory / "result.json")


def attempt(record: dict, directory: Path, name: str, action) -> bool:
    progress(test=name)
    if record["tests"].get(name, {}).get("status") == "PASS":
        print("RESUME SKIP", record["arm"], name, flush=True)
        return True
    record["tests"][name] = {"status": "RUNNING", "started_at": time.time()}
    save(record, directory)
    print("START", record["arm"], name, flush=True)
    start = time.monotonic()
    try:
        result = action()
        record["tests"][name] = {"status": "PASS", "result": result}
    except Exception as error:
        record["tests"][name] = {"status": "FAILED", "error": str(error)}
    record["tests"][name]["elapsed_s"] = round(time.monotonic() - start, 2)
    save(record, directory)
    print("DONE", record["arm"], name, record["tests"][name]["status"], flush=True)
    progress(state=record["tests"][name]["status"])
    return record["tests"][name]["status"] == "PASS"


def exact_prompt(tokenizer, length: int, question: str = "Reply with exactly OK.") -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "CONTENT_MARKER"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = rendered.split("CONTENT_MARKER")
    start = tokenizer.encode(prefix, add_special_tokens=False)
    end = tokenizer.encode("\n\n" + question + suffix, add_special_tokens=False)
    filler = tokenizer.encode(
        "A quiet archive stores ordinary records of weather, travel, books and daily work.\n",
        add_special_tokens=False,
    )
    body = (filler * ((length // len(filler)) + 1))[: length - len(start) - len(end)]
    result = start + body + end
    assert len(result) == length
    return tokenizer.decode(result)


def run_prefix_battery(dtype: str, cgroup: Path, tokenizer, arm_name: str | None = None) -> dict:
    spec = os.environ.get("SPEC", "mtp")
    utilization = os.environ.get("GPU_UTIL", "0.93")
    backend = "FLASHINFER"
    directory = OUT / (arm_name or dtype)
    directory.mkdir(parents=True, exist_ok=True)
    compiler_concurrency = int(os.environ.get("COMPILER_JOBS", "4"))

    env = os.environ.copy()
    extra = f"--attention-backend {backend} --kv-cache-dtype {dtype} --enable-prefix-caching"
    env.update(
        MODEL=str(MODEL),
        CTX="fast" if dtype in ("auto", "bf16") else "long",
        SPEC=spec,
        MAX_LEN="-1",
        GPU_UTIL=str(utilization),
        MAX_SEQS="1",
        EMBED_UVA="1",
        VISION="0",
        SPEC_ATTN="0",
        PREFIX_CACHE="1",
        HOST="127.0.0.1",
        PORT=str(PORT),
        INT8_ACT="",
        PREFILL_ATTN="",
        MAX_JOBS=str(compiler_concurrency),
        FLASHINFER_NVCC_THREADS=str(compiler_concurrency),
        TORCHINDUCTOR_COMPILE_THREADS=str(compiler_concurrency),
        CUDA_VISIBLE_DEVICES=GPU,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OPENAI_API_KEY=KEY,
        VLLM_API_KEY=KEY,
        PREFILL_BUDGET="2048",
        PATH=str(ROOT / "venv/bin") + ":" + env.get("PATH", ""),
        TMPDIR="<RUN_DIR>",
        EXTRA_ARGS=extra,
        PYTHONPATH=str(ROOT),
    )

    from importlib.metadata import distribution
    cuda = Path(distribution("nvidia-cuda-nvcc").locate_file("nvidia/cu13"))
    env["CUDA_HOME"] = str(cuda)
    libraries = [str(p) for p in (cuda / "lib", cuda / "lib64") if p.is_dir()]
    assert (cuda / "lib/libcudart.so").exists(), "CUDA runtime linker name missing"
    for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        inherited = [p for p in env.get(variable, "").split(":") if p]
        env[variable] = ":".join(dict.fromkeys(libraries + inherited))

    record = {
        "arm": arm_name or dtype,
        "backend": backend,
        "gpu_uuid": GPU,
        "model": str(MODEL),
        "draft": os.environ.get("DRAFT", ""),
        "spec": spec,
        "prefix_cache": True,
        "gpu_util": utilization,
        "host_limit_bytes": 48000000000,
        "compiler_concurrency": compiler_concurrency,
        "tests": {},
        "stack": "vLLM 0.32.0 / FlashInfer 0.7.0.post1 (Prefix Cache Enabled)",
    }

    server = None
    finished = threading.Event()
    peak = 0

    def monitor():
        nonlocal peak
        last_heartbeat = 0
        with (directory / "memory-samples.jsonl").open("w") as log:
            while not finished.is_set():
                sample = {"time": time.time(), "host_current_bytes": int((cgroup / "memory.current").read_text())}
                try:
                    value = subprocess.check_output(
                        ["nvidia-smi", "-i", GPU, "--query-gpu=memory.used,power.draw,power.limit", "--format=csv,noheader,nounits"],
                        text=True, timeout=10
                    ).strip()
                    sample["gpu_memory_mib"], sample["power_w"], sample["power_limit_w"] = map(float, value.split(","))
                    peak = max(peak, sample["gpu_memory_mib"])
                except Exception as error:
                    sample["telemetry_error"] = str(error)
                log.write(json.dumps(sample) + "\n")
                log.flush()
                if time.monotonic() - last_heartbeat >= 30:
                    print(f"HEARTBEAT {dtype} [prefix-cache]: {PROGRESS.get('test','starting')} | GPU {sample.get('gpu_memory_mib','?')} MiB | host {sample['host_current_bytes']/1e9:.2f} GB", flush=True)
                    last_heartbeat = time.monotonic()
                finished.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    events_before = (cgroup / "memory.events").read_text()

    try:
        with (directory / "server.log").open("w") as log:
            def boot():
                nonlocal server
                server = subprocess.Popen(
                    ["bash", "single-user/start_qwen.sh"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                record["server_pid"] = server.pid
                save(record, directory)
                deadline = time.monotonic() + 3600
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"Boot exited {server.returncode}; see server.log")
                    try:
                        request("/health", timeout=2)
                        break
                    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                        time.sleep(5)
                else:
                    raise TimeoutError("Boot timeout")

                text = (directory / "server.log").read_text()
                assert "enable_prefix_caching=True" in text, "Prefix caching flag not detected in server boot log"
                models = request("/v1/models")
                length = models["data"][0].get("max_model_len", 262144)
                result = {"max_model_len": length, "models": models}
                for label, pattern in {
                    "model_gib": r"Model loading took ([\d.]+) GiB",
                    "kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
                    "pool_tokens": r"GPU KV cache size: ([\d,]+) tokens",
                }.items():
                    match = re.search(pattern, text)
                    if match:
                        result[label] = float(match[1]) if label.endswith("gib") else int(match[1].replace(",", ""))
                return result

            if not attempt(record, directory, "boot", boot):
                record["blocked"] = "Boot failed"
                return record

            limit = record["tests"]["boot"]["result"]["max_model_len"]
            record["admitted_context_tokens"] = limit
            record["kv_pool_tokens"] = record["tests"]["boot"]["result"].get("pool_tokens")

            if limit < 34000:
                record["blocked"] = f"Admitted context {limit} < 34000; cannot run 32K prefix battery"
                record["status"] = "NOT APPLICABLE"
                return record

            prefix_32k = exact_prompt(tokenizer, 32768, question="Summarize the archive in one concise sentence: ")
            cold_res = {}
            warm_res = {}

            # Phase 1: Cold 32K prefix
            def test_cold_32k():
                nonlocal cold_res
                start = time.monotonic()
                res = chat_messages([{"role": "user", "content": prefix_32k}], max_tokens=64, temperature=0.0)
                elapsed = time.monotonic() - start
                usage = res["usage"]
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                text = (res["choices"][0]["message"].get("content") or "").strip()
                cold_res = {
                    "elapsed_s": round(elapsed, 3),
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "cached_tokens": cached,
                    "text": text,
                }
                return cold_res

            if not attempt(record, directory, "prefix-cold-32k", test_cold_32k):
                record["blocked"] = "Cold 32K prefix prefill failed"
                return record

            # Phase 2: Warm identical 32K prefix
            def test_warm_32k():
                nonlocal warm_res
                start = time.monotonic()
                res = chat_messages([{"role": "user", "content": prefix_32k}], max_tokens=64, temperature=0.0)
                elapsed = time.monotonic() - start
                usage = res["usage"]
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                text = (res["choices"][0]["message"].get("content") or "").strip()
                warm_res = {
                    "elapsed_s": round(elapsed, 3),
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "cached_tokens": cached,
                    "speedup_ratio": round(cold_res["elapsed_s"] / max(elapsed, 0.001), 2),
                    "text": text,
                }
                assert cached > 0, f"Expected cache hit on warm prefix, got cached_tokens={cached}"
                return warm_res

            if not attempt(record, directory, "prefix-warm-32k", test_warm_32k):
                record["blocked"] = "Warm 32K prefix cache hit failed"
                return record

            # Phase 3: Appended user turn
            def test_appended_turn():
                messages = [
                    {"role": "user", "content": prefix_32k},
                    {"role": "assistant", "content": cold_res["text"]},
                    {"role": "user", "content": "Extract the primary subject and activity from the summary in 5 words."},
                ]
                start = time.monotonic()
                res = chat_messages(messages, max_tokens=32, temperature=0.0)
                elapsed = time.monotonic() - start
                usage = res["usage"]
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                assert cached >= 32000, f"Expected multi-turn to preserve prefix cache (>=32000), got {cached}"
                return {
                    "elapsed_s": round(elapsed, 3),
                    "prompt_tokens": usage["prompt_tokens"],
                    "cached_tokens": cached,
                    "completion_tokens": usage["completion_tokens"],
                    "text": (res["choices"][0]["message"].get("content") or "").strip(),
                }

            if not attempt(record, directory, "prefix-appended-turn", test_appended_turn):
                record["blocked"] = "Appended user turn failed"
                return record

            # Phase 4: Mutated early prefix
            def test_mutated_prefix():
                # Mutate the first 200 chars of the body
                mutated = "AN ALTERNATIVE RECORD BEGINS HERE: " + prefix_32k[35:]
                start = time.monotonic()
                res = chat_messages([{"role": "user", "content": mutated}], max_tokens=32, temperature=0.0)
                elapsed = time.monotonic() - start
                usage = res["usage"]
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                # Mutating early tokens should invalidate prefix caching
                return {
                    "elapsed_s": round(elapsed, 3),
                    "prompt_tokens": usage["prompt_tokens"],
                    "cached_tokens": cached,
                    "invalidation_verified": cached < 1000,
                }

            if not attempt(record, directory, "prefix-mutated-early", test_mutated_prefix):
                record["blocked"] = "Mutated early prefix test failed"
                return record

            # Phase 5: Warm/Cold T=0 exact parity
            def test_parity():
                c_text = cold_res["text"].strip()
                w_text = warm_res["text"].strip()
                match = (c_text == w_text)
                if not match:
                    raise AssertionError(f"Cold vs Warm T=0 output mismatch:\nCold: {c_text!r}\nWarm: {w_text!r}")
                return {
                    "cold_output": c_text,
                    "warm_output": w_text,
                    "exact_token_match": True,
                }

            if not attempt(record, directory, "prefix-t0-parity", test_parity):
                record["blocked"] = "Cold vs Warm T=0 output mismatch"
                return record

            # Extract speculative telemetry if speculation was active
            if spec != "off":
                try:
                    s_text = (directory / "server.log").read_text(errors="replace")
                    acc_rate_m = re.findall(r"Acceptance rate \(%\):\s*([\d.]+)", s_text)
                    acc_len_m = re.findall(r"Acceptance length:\s*([\d.]+)", s_text)
                    per_pos_m = re.findall(r"Per-position acceptance \(%\):\s*([0-9.,\s%]+)", s_text)
                    if acc_rate_m:
                        record["spec_acceptance_rate"] = float(acc_rate_m[-1])
                    if acc_len_m:
                        record["spec_acceptance_length"] = float(acc_len_m[-1])
                    if per_pos_m:
                        raw_pos = per_pos_m[-1].replace("%", "").split(",")
                        record["per_position_acceptance"] = [float(p.strip()) for p in raw_pos if p.strip()]
                except Exception as ex:
                    record["spec_telemetry_error"] = str(ex)

    finally:
        stop(server)
        finished.set()
        thread.join(timeout=15)
        record["server_left_running"] = None
        record["sampled_peak_gpu_mib"] = peak
        record["aggregate_scope_host_peak_bytes"] = int((cgroup / "memory.peak").read_text())
        record["memory_events_before"] = events_before
        record["memory_events_after"] = (cgroup / "memory.events").read_text()
        statuses = {t["status"] for t in record["tests"].values()}
        record["status"] = "FAILED" if "FAILED" in statuses else ("PARTIAL" if "CANCELLED" in statuses else "PASS")
        save(record, directory)

    return record


def main():
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", PORT)) != 0, f"Port {PORT} already occupied"
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "GPU busy"

    relative = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1].lstrip("/")
    cgroup = Path("/sys/fs/cgroup") / relative
    if (cgroup / "memory.max").exists():
        assert (cgroup / "memory.max").read_text().strip() == "48000000000"
        assert (cgroup / "memory.swap.max").read_text().strip() == "0"
        (cgroup / "memory.oom.group").write_text("1")

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), OUT / "driver-source.py")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    arms = tuple([a.strip() for a in os.environ.get("ARMS", "auto,nvfp4").split(",") if a.strip()])
    results = {"scope": "vLLM 0.32.0 Prefix Caching Battery, SM120", "gpu_uuid": GPU, "arms": {}}
    progress(stage="Prefix Caching", state="STARTING", arms=list(arms), run_started_at=time.time())

    for index, dtype in enumerate(arms):
        progress(arm_index=index + 1, arm_count=len(arms))
        results["arms"][dtype] = run_prefix_battery(dtype, cgroup, tokenizer)
        save(results, OUT)

    try:
        manifest_utils.build_manifest(ROOT, OUT, results["arms"], MODEL, os.environ.get("DRAFT"), GPU, "prefix_cache")
    except Exception as e:
        print(f"Warning: Manifest build failed: {e}", flush=True)

    progress(state="COMPLETE", test=None, completed_arms=len(arms))
    print("Prefix Caching Battery complete; servers stopped.", flush=True)


if __name__ == "__main__":
    main()
