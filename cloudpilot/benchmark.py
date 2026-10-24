"""Bounded non-streaming OpenAI-compatible chat client. No automatic execution."""
import concurrent.futures
import hashlib
import ipaddress
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def endpoint_url(base_url, allow_remote=False):
    url = urllib.parse.urlsplit(base_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an http(s) base URL without credentials, query or fragment')
    try:
        local = ipaddress.ip_address(url.hostname).is_loopback
    except ValueError:
        local = url.hostname == 'localhost'
    if not local and not allow_remote:
        raise ValueError('Remote requests require explicit --allow-remote')
    if not local and url.scheme != 'https':
        raise ValueError('Remote requests require HTTPS; use an SSH tunnel for plain HTTP')
    return base_url.rstrip('/') + '/chat/completions'


def percentile(values, p):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)]


def summarize(rows, wall_seconds):
    ok = [r for r in rows if r['ok']]
    known = [r['output_tokens'] for r in ok if r['output_tokens'] is not None]
    all_usage = bool(ok) and len(known) == len(ok)
    return {'requests': len(rows), 'successful': len(ok), 'failed': len(rows)-len(ok),
            'wall_seconds': wall_seconds, 'p50_success_latency_s': percentile([r['latency_s'] for r in ok], .5),
            'p95_success_latency_s': percentile([r['latency_s'] for r in ok], .95),
            'known_success_output_tokens': sum(known), 'success_usage_complete': all_usage,
            'successful_output_tokens_per_wall_second': sum(known)/wall_seconds if all_usage and wall_seconds > 0 else None,
            'time_to_first_token': None,
            'limitations': ['Non-streaming: time to first token and inter-token latency are not measured.',
                           'Latency percentiles describe successful requests only; failures remain listed.',
                           'Output usage is reported by the server, not independently tokenized.',
                           'Throughput uses the complete batch wall window, including failures; it is not per-user decode speed.']}


def run(base_url, model, prompts, concurrency=1, max_tokens=32, timeout=20,
        execute=False, allow_remote=False, api_key_env=None):
    url = endpoint_url(base_url, allow_remote)
    if not isinstance(model, str) or not model.strip():
        raise ValueError('model must be nonempty')
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 100 or any(not isinstance(p, str) or not p.strip() or len(p)>10000 for p in prompts):
        raise ValueError('Use 1–100 nonempty prompts, each <= 10000 characters')
    for value, low, high, name in [(concurrency,1,8,'concurrency'),(max_tokens,1,1024,'max_tokens')]:
        if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high:
            raise ValueError(f'{name} outside allowed bounds')
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or not 0<timeout<=120:
        raise ValueError('timeout must be finite and in (0,120]')
    plan = {'mode': 'execute' if execute else 'dry_run', 'requests': len(prompts),
            'concurrency': concurrency, 'max_output_tokens_per_request': max_tokens,
            'model': model, 'prompt_set_sha256': hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()}
    if not execute:
        return plan
    token = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not token:
        raise ValueError('Requested API-key environment variable is missing')
    def one(index_prompt):
        index, prompt = index_prompt
        payload = {'model': model, 'messages':[{'role':'user','content':prompt}],
                   'max_tokens':max_tokens, 'temperature':0, 'stream':False}
        headers = {'Content-Type':'application/json'}
        if token:
            headers['Authorization'] = 'Bearer '+token
        req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
        start = time.perf_counter()
        row = {'request_id':index, 'ok':False, 'output_tokens':None}
        try:
            # No environment proxy; do not forward credentials across redirects.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            with opener.open(req, timeout=timeout) as response:
                raw = response.read(2_000_001)
                if len(raw)>2_000_000:
                    raise ValueError('Oversized response')
                data = json.loads(raw)
                content = data['choices'][0]['message']['content']
                if not isinstance(content,str):
                    raise ValueError('Expected textual chat completion')
                usage = data.get('usage') or {}
                count = usage.get('completion_tokens')
                if count is not None and (isinstance(count,bool) or not isinstance(count,int) or count<0):
                    raise ValueError('Invalid completion token count')
                row.update(ok=True, output_tokens=count, response_sha256=hashlib.sha256(content.encode()).hexdigest())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            # No server body, prompt, secret or URL goes into logs.
            row['error_type'] = type(exc).__name__
            if isinstance(exc,urllib.error.HTTPError):
                row['http_status'] = exc.code
        row['latency_s'] = time.perf_counter()-start
        return row
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows=list(pool.map(one, enumerate(prompts,1)))
    wall=time.perf_counter()-start
    return {'plan':plan, 'summary':summarize(rows,wall), 'requests':rows,
            'measurement_scope':'Client-side HTTP behavior only; GPU memory, task correctness and runtime compatibility not measured.'}
