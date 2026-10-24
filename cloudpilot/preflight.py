"""Validate supplied configuration; never infer hardware/model compatibility."""
import math
import re


def validate(config):
    errors = []
    def number(key, integer=False, zero=False):
        value = config.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < (0 if zero else 1e-12)
                or (integer and not isinstance(value, int))):
            errors.append(f'{key}: finite {"integer" if integer else "number"} required')
            return None
        return value
    for key in ('model_id', 'runtime_version', 'host_label'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            errors.append(f'{key}: nonempty string required')
    if config.get('runtime') not in ('vllm', 'sglang'):
        errors.append('runtime: choose vllm or sglang')
    if not re.fullmatch(r'[0-9a-f]{40}', str(config.get('model_revision', ''))):
        errors.append('model_revision: immutable 40-character commit required')
    values = {k: number(k, integer=k in ('gpu_count', 'tensor_parallel', 'context_tokens', 'concurrent_requests')) for k in
              ('gpu_count', 'tensor_parallel', 'context_tokens', 'concurrent_requests', 'gpu_memory_gib', 'gpu_memory_fraction', 'weights_total_gib')}
    for key in ('kv_cache_per_gpu_gib', 'overhead_per_gpu_gib'):
        values[key] = number(key, zero=True)
    if values['gpu_memory_fraction'] is not None and values['gpu_memory_fraction'] > 1:
        errors.append('gpu_memory_fraction: must be <= 1')
    if values['tensor_parallel'] and values['gpu_count'] and values['tensor_parallel'] > values['gpu_count']:
        errors.append('tensor_parallel exceeds GPUs on this one host')
    if errors:
        return {'status': 'invalid', 'errors': errors, 'deployment_verified': False}
    # Explicit user-supplied estimates, NOT an architecture/KV-cache inference engine.
    margin = (values['gpu_memory_gib'] * values['gpu_memory_fraction']
              - values['weights_total_gib'] / values['tensor_parallel']
              - values['kv_cache_per_gpu_gib'] - values['overhead_per_gpu_gib'])
    return {'status': 'estimate_exceeds_budget' if margin < 0 else 'needs_runtime_verification',
            'errors': [], 'estimated_margin_per_gpu_gib': round(margin, 6),
            'deployment_verified': False,
            'limitations': ['Weight sharding is an estimate; architecture, replication and kernel support are not verified.',
                           'KV-cache and overhead are user-supplied estimates for the intended context and concurrency.',
                           'A nonnegative margin does not prove successful loading or execution.']}
