#!/usr/bin/env python3
"""SM120 C1 staged KV comparisons for vLLM 0.32.0.
Fresh arms, no retries, always stop servers.

Run in v032-kv-controls.scope with MemoryMax=48000000000, MemorySwapMax=0.
Artifacts and frozen quality inputs live under <RUN_DIR>
PPL evaluation uses English, German (deu_Latn), Greek (ell_Grek), and vLLM code.
Includes fast top-20 KLD distribution verification gate.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('V032_CONTROLS_OUT','<RUN_DIR>'))
MODEL = Path(os.environ.get('MODEL', str(ROOT / 'models/Qwen3.8-27B-W4A16-AutoRound-fast')))
GPU = '<GPU_UUID>'
PORT = int(os.environ.get('PORT', '18031'))
API = f'http://127.0.0.1:{PORT}'
KEY = (ROOT/'api_key.txt').read_text().strip()
SERVER_MODEL = 'qwen3.8-27b'
STAGE = int(os.environ.get('V032_STAGE','1'))
STAGE2 = ('nvfp4','turboquant_k8v4')
STAGE3 = ('fp8','nvfp4','turboquant_k8v4')
STAGE4 = ('fp8','nvfp4')
PROGRESS = {}


def progress(**updates):
    PROGRESS.update(updates)
    PROGRESS['updated_at'] = time.time()
    if OUT.is_dir():
        temp = OUT / 'status.json.tmp'
        temp.write_text(json.dumps(PROGRESS, indent=2) + '\n')
        temp.replace(OUT / 'status.json')


def request(path, payload=None, timeout=3600):
    req = urllib.request.Request(
        API + path,
        data=None if payload is None else json.dumps({'model': SERVER_MODEL, **payload}).encode(),
        headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from error


SAMPLING_PROFILES = {
    'greedy': {
        'temperature': 0.0,
        'top_p': 1.0,
        'top_k': -1,
        'min_p': 0.0,
        'presence_penalty': 0.0,
        'repetition_penalty': 1.0,
        'enable_thinking': False,
    },
    'instruct': {
        'temperature': 0.7,
        'top_p': 0.80,
        'top_k': 20,
        'min_p': 0.0,
        'presence_penalty': 1.5,
        'repetition_penalty': 1.0,
        'enable_thinking': False,
    },
    'thinking': {
        'temperature': 1.0,
        'top_p': 0.95,
        'top_k': 20,
        'min_p': 0.0,
        'presence_penalty': 0.0,
        'repetition_penalty': 1.0,
        'enable_thinking': True,
    }
}


def chat(prompt, max_tokens, profile='instruct', **overrides):
    cfg = dict(SAMPLING_PROFILES.get(profile, SAMPLING_PROFILES['instruct']))
    cfg.update(overrides)
    enable_thinking = cfg.pop('enable_thinking', False)
    params = {}
    if cfg.get('temperature') is not None:
        params['temperature'] = cfg['temperature']
    if cfg.get('top_p') is not None:
        params['top_p'] = cfg['top_p']
    if cfg.get('top_k', -1) > 0:
        params['top_k'] = cfg['top_k']
    if cfg.get('min_p', 0.0) > 0.0:
        params['min_p'] = cfg['min_p']
    if cfg.get('presence_penalty', 0.0) != 0.0:
        params['presence_penalty'] = cfg['presence_penalty']
    if cfg.get('repetition_penalty', 1.0) != 1.0:
        params['repetition_penalty'] = cfg['repetition_penalty']
    body = {
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'chat_template_kwargs': {'enable_thinking': enable_thinking},
        **params
    }
    return request('/v1/chat/completions', body)


def stop(server):
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


def save(data, directory):
    temp = directory / 'result.json.tmp'
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(directory / 'result.json')


def attempt(record, directory, name, action):
    progress(test=name)
    if record['tests'].get(name, {}).get('status') == 'PASS':
        print('RESUME SKIP', record['arm'], name, flush=True)
        return True
    record['tests'][name] = {'status': 'RUNNING', 'started_at': time.time()}
    save(record, directory)
    print('START', record['arm'], name, flush=True)
    start = time.monotonic()
    try:
        result = action()
        record['tests'][name] = {'status': 'PASS', 'result': result}
    except Exception as error:
        record['tests'][name] = {'status': 'FAILED', 'error': str(error)}
    record['tests'][name]['elapsed_s'] = round(time.monotonic() - start, 2)
    save(record, directory)
    print('DONE', record['arm'], name, record['tests'][name]['status'], flush=True)
    progress(state=record['tests'][name]['status'])
    return record['tests'][name]['status'] == 'PASS'


def prepare_quality():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    items = []
    
    # 1. English: Wikitext-2
    wiki_path = ROOT / 'bench/quality-data/wikitext-2-raw-v1-test.parquet'
    if not wiki_path.is_file():
        wiki_path = Path(hf_hub_download('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'))
    english = ''.join(pq.read_table(wiki_path).column('text').to_pylist())
    items.extend(('en', english[i:i+1200]) for i in range(0, min(len(english), 48000), 1200))
    
    # 2. German: FineWeb-2 deu_Latn
    de_path = ROOT / 'bench/quality-data/deu_Latn_test.parquet'
    if not de_path.is_file():
        de_path = Path(hf_hub_download('HuggingFaceFW/fineweb-2', 'data/deu_Latn/test/000_00000.parquet', repo_type='dataset'))
    german = pq.read_table(de_path, columns=['text']).column('text').to_pylist()
    random.Random(0).shuffle(german)
    items.extend(('de', text[:1200]) for text in [x for x in german if len(x) > 1500][:40])
    
    # 3. Greek: FineWeb-2 ell_Grek
    gr_path = ROOT / 'bench/quality-data/ell_Grek_test.parquet'
    if not gr_path.is_file():
        gr_path = Path(hf_hub_download('HuggingFaceFW/fineweb-2', 'data/ell_Grek/test/000_00000.parquet', repo_type='dataset'))
    greek = pq.read_table(gr_path, columns=['text']).column('text').to_pylist()
    random.Random(0).shuffle(greek)
    items.extend(('el', text[:1200]) for text in [x for x in greek if len(x) > 1500][:40])

    # 4. Code: vLLM core
    from importlib.metadata import distribution
    core = Path(distribution('vllm').locate_file('vllm/v1/core'))
    code = []
    for path in sorted(core.glob('*.py')):
        text = path.read_text()
        code.extend(('code', text[i:i+1200]) for i in range(0, min(len(text), 4800), 1200)
                    if len(text[i:i+1200]) > 800)
        if len(code) >= 40:
            break
    items.extend(code[:40])

    # 5. GSM8K
    gsm_path = ROOT / 'bench/quality-data/gsm8k/test-00000-of-00001.parquet'
    if not gsm_path.is_file():
        gsm_path = Path(hf_hub_download('openai/gsm8k', 'main/test-00000-of-00001.parquet', repo_type='dataset'))
    table = pq.read_table(gsm_path)
    gsm = list(zip(table.column('question').to_pylist(), table.column('answer').to_pylist()))[:200]
    assert len(gsm) == 200 and all(any(lang == x[0] for x in items) for lang in ('en', 'de', 'el', 'code'))

    paths = [wiki_path, de_path, gr_path, gsm_path]
    frozen = {
        'ppl': items,
        'gsm': gsm,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    }
    (OUT / 'quality-inputs.json').write_text(json.dumps(frozen, indent=2) + '\n')
    return frozen


def gsm(data, directory, profile='instruct'):
    correct = truncated = 0
    tokens = []
    max_tokens = 2048 if profile == 'thinking' else 768
    with (directory / f'gsm8k-{profile}-samples.jsonl').open('w') as log:
        for index, (question, answer) in enumerate(data['gsm']):
            progress(completed_samples=index, total_samples=len(data['gsm']))
            response = chat(question + "\n\nSolve step by step, then give the final answer as 'Final answer: <number>'.", max_tokens, profile=profile)
            choice = response['choices'][0]
            content = choice['message'].get('content') or ''
            reasoning = choice['message'].get('reasoning_content') or choice['message'].get('reasoning') or ''
            gold = answer.split('####')[-1].strip().replace(',', '')
            match = re.search(r'Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)', content) or re.search(r'Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)', reasoning)
            if match:
                pred = match.group(1).replace(',', '')
            else:
                fallback = re.findall(r'-?\d[\d,]*\.?\d*', content.replace('$', '')) or re.findall(r'-?\d[\d,]*\.?\d*', reasoning.replace('$', ''))
                pred = fallback[-1].replace(',', '') if fallback else None
            try:
                ok = abs(float(pred) - float(gold)) < 1e-6
            except (ValueError, TypeError):
                ok = False
            correct += ok
            truncated += choice['finish_reason'] == 'length'
            tokens.append(response['usage']['completion_tokens'])
            log.write(json.dumps({'index': index, 'gold': gold, 'prediction': pred, 'correct': ok,
                                  'finish_reason': choice['finish_reason'], 'response': response}) + '\n')
            log.flush()
            progress(completed_samples=index + 1, correct_answers=correct, truncated_answers=truncated)
            if (index + 1) % 20 == 0:
                print(f'  GSM8K [{profile}] {index+1}/200: {correct} correct, {truncated} truncated', flush=True)
    return {'n': 200, 'correct': correct, 'acc': correct / 200, 'truncations': truncated, 'mean_tokens': statistics.mean(tokens), 'profile': profile}


def ppl(data, directory, lang):
    total = count = 0
    with (directory / f'ppl-{lang}-samples.jsonl').open('w') as log:
        completed = 0
        samples = [text for label, text in data['ppl'] if label == lang]
        progress(total_samples=len(samples))
        for text in samples:
            response = request('/v1/completions', {'prompt': text, 'max_tokens': 1, 'temperature': 0, 'prompt_logprobs': 0})
            values = [next(iter(entry.values()))['logprob'] for entry in response['choices'][0]['prompt_logprobs'][1:] if entry]
            assert values, 'No scored prompt tokens'
            total += sum(values)
            count += len(values)
            log.write(json.dumps({'sum_logprob': sum(values), 'tokens': len(values)}) + '\n')
            log.flush()
            completed += 1
            progress(completed_samples=completed)
    return {'ppl': math.exp(-total / count), 'tokens': count, 'sum_logprob': total}


def kld(data, directory, baseline_ref_file=None):
    """Fast top-20 Kullback-Leibler divergence gate.
    
    Scores 5 multi-domain prompts with prompt_logprobs=20.
    Returns per-token top-20 distribution divergence against baseline.
    Completes in <5s.
    """
    records = []
    # Pick 5 diverse prompts (en, de, el, code)
    prompts = [text for _, text in data['ppl'][:5]]
    for idx, prompt in enumerate(prompts):
        response = request('/v1/completions', {'prompt': prompt[:800], 'max_tokens': 1, 'temperature': 0, 'prompt_logprobs': 20})
        token_entries = response['choices'][0]['prompt_logprobs'][1:]
        records.append({'prompt_idx': idx, 'entries': token_entries})

    out_file = directory / 'kld-logprobs.json'
    out_file.write_text(json.dumps(records, indent=2))

    summary = {'prompts_evaluated': len(prompts), 'total_positions': sum(len(r['entries']) for r in records)}
    if baseline_ref_file and Path(baseline_ref_file).is_file():
        ref_records = json.loads(Path(baseline_ref_file).read_text())
        kl_divs = []
        top1_matches = 0
        total_tokens = 0
        for cand, ref in zip(records, ref_records):
            for c_pos, r_pos in zip(cand['entries'], ref['entries']):
                if not c_pos or not r_pos:
                    continue
                # c_pos is dict: {token_str: {logprob: float}}
                c_probs = {t: math.exp(v['logprob']) for t, v in c_pos.items()}
                r_probs = {t: math.exp(v['logprob']) for t, v in r_pos.items()}
                # Top-1 match
                top_c = max(c_probs.items(), key=lambda x: x[1])[0]
                top_r = max(r_probs.items(), key=lambda x: x[1])[0]
                if top_c == top_r:
                    top1_matches += 1
                # Overlap KL: sum P_ref * log(P_ref / P_cand)
                common = set(r_probs) & set(c_probs)
                if common:
                    kl = sum(r_probs[t] * math.log(r_probs[t] / c_probs[t]) for t in common)
                    kl_divs.append(kl)
                total_tokens += 1
        summary['top1_agreement'] = top1_matches / total_tokens if total_tokens else 1.0
        summary['mean_kld'] = statistics.mean(kl_divs) if kl_divs else 0.0
        summary['max_kld'] = max(kl_divs) if kl_divs else 0.0
    return summary


def speed(directory, env, profile_name='instruct', *, iterations=(1, 2)):
    passes = []
    profile = SAMPLING_PROFILES.get(profile_name, SAMPLING_PROFILES['instruct'])
    assert iterations in ((1,), (1, 2)), 'Only explicit one/two-pass protocols supported'
    for iteration in iterations:
        progress(planned_pass=iteration, total_passes=len(iterations))
        print(f'  TG profile={profile_name}: planned pass {iteration}/{len(iterations)}', flush=True)
        log = directory / f'tg-{profile_name}-pass{iteration}.log'
        cmd = [
            str(ROOT / 'venv/bin/vllm'), 'bench', 'serve',
            '--host', '127.0.0.1', '--port', str(PORT),
            '--model', SERVER_MODEL, '--served-model-name', SERVER_MODEL, '--tokenizer', str(MODEL),
            '--dataset-name', 'custom', '--dataset-path', str(ROOT / 'bench/prompts_real.jsonl'),
            '--custom-output-len', '1024', '--num-prompts', '8', '--max-concurrency', '1',
            '--temperature', str(profile['temperature']),
            '--top-p', str(profile['top_p']),
        ]
        if profile.get('top_k', -1) > 0:
            cmd.extend(['--top-k', str(profile['top_k'])])
        if profile.get('presence_penalty', 0.0) != 0.0:
            cmd.extend(['--presence-penalty', str(profile['presence_penalty'])])
        if profile.get('repetition_penalty', 1.0) != 1.0:
            cmd.extend(['--repetition-penalty', str(profile['repetition_penalty'])])
        if profile.get('min_p', 0.0) > 0.0:
            cmd.extend(['--min-p', str(profile['min_p'])])
        with log.open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=2400)
        text = log.read_text()
        def number(label):
            match = re.search(re.escape(label) + r'\s*([\d.]+)', text)
            if not match:
                raise RuntimeError(f'Missing {label}; see {log}')
            return float(match[1])
        assert number('Successful requests:') == 8 and number('Failed requests:') == 0
        passes.append({
            'output_tps': number('Output token throughput (tok/s):'),
            'ttft_ms': number('Mean TTFT (ms):'),
            'tpot_ms': number('Mean TPOT (ms):')
        })
        passes[-1]['decode_tps'] = 1000 / passes[-1]['tpot_ms']
    return {'planned_passes': passes, 'retained_pass': iterations[-1], **passes[-1], 'profile': profile}


def exact_prompt(tokenizer, length, inserts=(), question='Reply with exactly OK.'):
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'CONTENT_MARKER'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prefix, suffix = rendered.split('CONTENT_MARKER')
    start = tokenizer.encode(prefix, add_special_tokens=False)
    end = tokenizer.encode('\n\n' + question + suffix, add_special_tokens=False)
    filler = tokenizer.encode('A quiet archive stores ordinary records of weather, travel, books and daily work.\n', add_special_tokens=False)
    body = (filler * ((length // len(filler)) + 1))[:length - len(start) - len(end)]
    for depth, text in inserts:
        ids = tokenizer.encode('\n' + text + '\n', add_special_tokens=False)
        offset = int((len(body) - len(ids)) * depth)
        body[offset:offset + len(ids)] = ids
    result = start + body + end
    assert len(result) == length
    return tokenizer.decode(result)


def context(tokenizer, length, inserts=(), question='Reply with exactly OK.', expected=('OK',)):
    prompt = exact_prompt(tokenizer, length, inserts, question)
    start = time.monotonic()
    response = chat(prompt, 64)
    choice = response['choices'][0]
    text = (choice['message']['content'] or '').strip()
    return {
        'elapsed_s': round(time.monotonic() - start, 2),
        'usage': response['usage'],
        'output': text,
        'pass': all(x.lower() in text.lower() for x in expected)
    }


def smoke():
    response = chat('What is the square root of 144? Answer with just the number.', 16)
    content = response['choices'][0]['message']['content'] or ''
    assert '12' in content, f'Smoke test unexpected output: {content}'
    return {'prompt': 'sqrt(144)', 'response': content.strip(), 'status': 'PASS'}


def run_arm(dtype, data, cgroup, tokenizer, arm_name=None):
    spec = os.environ.get('SPEC', 'mtp')
    utilization = os.environ.get('GPU_UTIL', '0.93')
    backend = 'FLASHINFER'
    directory = OUT / (arm_name or dtype)
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    compiler_concurrency = int(os.environ.get('COMPILER_JOBS', '4'))
    extra = f'--attention-backend {backend} --kv-cache-dtype {dtype} --no-enable-prefix-caching'
    env.update(
        MODEL=str(MODEL), CTX='fast' if dtype in ('auto', 'bf16') else 'long', SPEC=spec,
        MAX_LEN='-1', GPU_UTIL=str(utilization), MAX_SEQS='1',
        EMBED_UVA='1', VISION='0', SPEC_ATTN='0', PREFIX_CACHE='0', HOST='127.0.0.1', PORT=str(PORT),
        INT8_ACT='', PREFILL_ATTN='', MAX_JOBS=str(compiler_concurrency),
        FLASHINFER_NVCC_THREADS=str(compiler_concurrency),
        TORCHINDUCTOR_COMPILE_THREADS=str(compiler_concurrency),
        CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID', OPENAI_API_KEY=KEY,
        VLLM_API_KEY=KEY, PREFILL_BUDGET='2048',
        PATH=str(ROOT / 'venv/bin') + ':' + env.get('PATH', ''), TMPDIR='<RUN_DIR>',
        EXTRA_ARGS=extra, PYTHONPATH=str(ROOT)
    )
    from importlib.metadata import distribution
    cuda = Path(distribution('nvidia-cuda-nvcc').locate_file('nvidia/cu13'))
    env['CUDA_HOME'] = str(cuda)
    libraries = [str(p) for p in (cuda / 'lib', cuda / 'lib64') if p.is_dir()]
    assert (cuda / 'lib/libcudart.so').exists(), 'CUDA runtime linker name missing'
    for variable in ('LIBRARY_PATH', 'LD_LIBRARY_PATH'):
        inherited = [p for p in env.get(variable, '').split(':') if p]
        env[variable] = ':'.join(dict.fromkeys(libraries + inherited))

    record = {
        'arm': arm_name or dtype, 'stage': STAGE, 'backend': backend, 'gpu_uuid': GPU,
        'model': str(MODEL), 'draft': os.environ.get('DRAFT', ''), 'spec': spec, 'prefix_cache': False,
        'gpu_util': utilization, 'prefill_budget': 2048, 'max_seqs': 1,
        'host_limit_bytes': 48000000000, 'compiler_concurrency': compiler_concurrency,
        'tests': {}, 'stack': 'vLLM 0.32.0 / FlashInfer 0.7.0.post1'
    }
    if (directory / 'result.json').is_file():
        try:
            prior = json.loads((directory / 'result.json').read_text())
            record['tests'] = {k: v for k, v in prior.get('tests', {}).items() if k != 'boot'}
            if 'combined_ppl' in prior:
                record['combined_ppl'] = prior['combined_ppl']
        except Exception:
            pass
    server = None
    finished = threading.Event()
    peak = 0

    def monitor():
        nonlocal peak
        last_heartbeat = 0
        with (directory / 'memory-samples.jsonl').open('w') as log:
            while not finished.is_set():
                sample = {'time': time.time(), 'host_current_bytes': int((cgroup / 'memory.current').read_text())}
                try:
                    value = subprocess.check_output(
                        ['nvidia-smi', '-i', GPU, '--query-gpu=memory.used,power.draw,power.limit', '--format=csv,noheader,nounits'],
                        text=True, timeout=10
                    ).strip()
                    sample['gpu_memory_mib'], sample['power_w'], sample['power_limit_w'] = map(float, value.split(','))
                    peak = max(peak, sample['gpu_memory_mib'])
                except Exception as error:
                    sample['telemetry_error'] = str(error)
                log.write(json.dumps(sample) + '\n')
                log.flush()
                if time.monotonic() - last_heartbeat >= 30:
                    print(f"HEARTBEAT {dtype}: {PROGRESS.get('test','starting')} | GPU {sample.get('gpu_memory_mib','?')} MiB | host {sample['host_current_bytes']/1e9:.2f} GB", flush=True)
                    last_heartbeat = time.monotonic()
                finished.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    events_before = (cgroup / 'memory.events').read_text()
    try:
        with (directory / 'server.log').open('w') as log:
            def boot():
                nonlocal server
                server = subprocess.Popen(
                    ['bash', 'single-user/start_qwen.sh'], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                record['server_pid'] = server.pid
                save(record, directory)
                deadline = time.monotonic() + 3600
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f'Boot exited {server.returncode}; see server.log')
                    try:
                        request('/health', timeout=2)
                        break
                    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                        time.sleep(5)
                else:
                    raise TimeoutError('Boot timeout')
                text = (directory / 'server.log').read_text()
                models = request('/v1/models')
                length = models['data'][0].get('max_model_len', 262144)
                result = {'max_model_len': length, 'models': models}
                for label, pattern in {
                    'model_gib': r'Model loading took ([\d.]+) GiB',
                    'kv_gib': r'Available KV cache memory: ([\d.]+) GiB',
                    'pool_tokens': r'GPU KV cache size: ([\d,]+) tokens'
                }.items():
                    match = re.search(pattern, text)
                    if match:
                        result[label] = float(match[1]) if label.endswith('gib') else int(match[1].replace(',', ''))
                return result

            if not attempt(record, directory, 'boot', boot):
                record['blocked'] = 'Boot failed; no retries'
                return record
            limit = record['tests']['boot']['result']['max_model_len']

            selected = os.environ.get('TESTS')
            def should_run(test_name):
                if not selected:
                    return True
                parts = [p.strip() for p in selected.split(',') if p.strip()]
                return any(p == test_name or test_name.startswith(p) or p.startswith(test_name) for p in parts)

            if should_run('correctness-smoke'):
                if not attempt(record, directory, 'correctness-smoke', smoke):
                    record['blocked'] = 'Correctness gate failed'
                    return record

            # Fast KLD gate (~3s)
            if should_run('kld-distribution'):
                baseline_ref = os.environ.get('KLD_BASELINE')
                if not baseline_ref and (OUT / 'auto/kld-logprobs.json').is_file() and (arm_name or dtype) != 'auto':
                    baseline_ref = str(OUT / 'auto/kld-logprobs.json')
                attempt(record, directory, 'kld-distribution', lambda: kld(data, directory, baseline_ref_file=baseline_ref))

            raw_gsm_profiles = os.environ.get('GSM_PROFILES') or os.environ.get('GSM_PROFILE') or 'greedy'
            gsm_profiles = [p.strip() for p in raw_gsm_profiles.split(',') if p.strip()]
            for gsm_profile in gsm_profiles:
                test_key = f'gsm8k-200-{gsm_profile}'
                if should_run(test_key) or should_run('gsm8k'):
                    if not attempt(record, directory, test_key, lambda p=gsm_profile: gsm(data, directory, profile=p)):
                        record['blocked'] = f'Quality execution failed on {gsm_profile}'
                        return record

            # PPL on en, de, el, code
            if should_run('ppl'):
                for lang in ('en', 'de', 'el', 'code'):
                    if not attempt(record, directory, 'ppl-' + lang, lambda lang=lang: ppl(data, directory, lang)):
                        record['blocked'] = f'PPL {lang} gate failed'
                        return record
                scored = [record['tests']['ppl-' + lang]['result'] for lang in ('en', 'de', 'el', 'code')]
                n = sum(x['tokens'] for x in scored)
                record['combined_ppl'] = {'ppl': math.exp(-sum(x['sum_logprob'] for x in scored) / n), 'tokens': n}

            if should_run('tg'):
                tg_profiles = [p.strip() for p in os.environ.get('TG_PROFILES', 'greedy,instruct').split(',') if p.strip()]
                for profile_name in tg_profiles:
                    if not attempt(record, directory, f'tg-{profile_name}', lambda p=profile_name: speed(directory, env, profile_name=p)):
                        record['blocked'] = f'Speed execution failed on {profile_name}'
                        return record

            if should_run('pp'):
                for length in (1024, 16384, 65536):
                    name = f'pp-{length}'
                    if length + 1 > limit:
                        record['tests'][name] = {'status': 'NOT APPLICABLE', 'reason': 'Admitted capacity'}
                        continue
                    def pp(length=length):
                        trials = []
                        for _ in range(3):
                            start = time.monotonic()
                            response = request('/v1/completions', {'prompt': exact_prompt(tokenizer, length), 'temperature': 0, 'max_tokens': 1})
                            elapsed = time.monotonic() - start
                            actual_tokens = response['usage']['prompt_tokens']
                            assert abs(actual_tokens - length) <= 2, f"Expected {length} prompt tokens, got {actual_tokens}"
                            trials.append({'elapsed_s': elapsed, 'tps': actual_tokens / elapsed})
                        return {'trials': trials, 'median_tps': statistics.median(x['tps'] for x in trials)}
                    if not attempt(record, directory, name, pp):
                        return record

    finally:
        stop(server)
        finished.set()
        thread.join(timeout=15)
        record['server_left_running'] = None
        record['sampled_peak_gpu_mib'] = peak
        record['aggregate_scope_host_peak_bytes'] = int((cgroup / 'memory.peak').read_text())
        record['memory_events_before'] = events_before
        record['memory_events_after'] = (cgroup / 'memory.events').read_text()
        statuses = {t['status'] for t in record['tests'].values()}
        record['status'] = 'FAILED' if 'FAILED' in statuses else ('PARTIAL / CANCELLED TEST' if 'CANCELLED' in statuses else 'PASS')
        save(record, directory)
    return record


def main():
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', PORT)) != 0, f'Port {PORT} already occupied'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU busy'
    
    relative = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
    cgroup = Path('/sys/fs/cgroup') / relative
    if (cgroup / 'memory.max').exists():
        assert (cgroup / 'memory.max').read_text().strip() == '48000000000'
        assert (cgroup / 'memory.swap.max').read_text().strip() == '0'
        (cgroup / 'memory.oom.group').write_text('1')

    OUT.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(Path(__file__), OUT / 'driver-source.py')
    data = prepare_quality()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    arms = tuple([a.strip() for a in os.environ.get('ARMS', 'auto,nvfp4').split(',') if a.strip()])
    results = {'scope': 'vLLM 0.32.0 Stage 1 Upgrade Controls, SM120', 'gpu_uuid': GPU, 'arms': {}}
    progress(stage=STAGE, state='STARTING', arms=list(arms), run_started_at=time.time())
    for index, dtype in enumerate(arms):
        progress(arm_index=index + 1, arm_count=len(arms))
        results['arms'][dtype] = run_arm(dtype, data, cgroup, tokenizer)
        save(results, OUT)
    progress(state='COMPLETE', test=None, completed_arms=len(arms))
    print(f'Stage {STAGE} finished. Servers stopped; awaiting review.', flush=True)


if __name__ == '__main__':
    main()
