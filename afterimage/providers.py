import os, json, math, hashlib, uuid, urllib.request, urllib.error
from pathlib import Path
from .common import canonical, check, pricing_cost

def make_payload(job, prompt, settings, file_ids):
    """Only prompt and ordered image references go to the model. No GT or dataset metadata."""
    if job['provider'] == 'openai':
        content = [{'type': 'input_text', 'text': prompt}]
        content += [{'type': 'input_image', 'file_id': f, 'detail': settings['image_detail']} for f in file_ids]
        return {'model': settings['id'], 'input': [{'role': 'user', 'content': content}],
                'reasoning': {'effort': settings['effort']}, 'max_output_tokens': settings['max_output_tokens'],
                'store': False, 'service_tier': 'default'}
    content = [{'type': 'text', 'text': prompt}]
    content += [{'type': 'image', 'source': {'type': 'file', 'file_id': f}} for f in file_ids]
    return {'model': settings['id'], 'messages': [{'role': 'user', 'content': content}],
            'thinking': {'type': settings['thinking']}, 'output_config': {'effort': settings['effort']},
            'max_tokens': settings['max_output_tokens']}

class Remote:
    """Network adapter is constructed only inside --execute after approval verification."""
    def __init__(self, providers=('openai','anthropic')):
        self.keys = {}
        for provider, env in [('openai', 'OPENAI_API_KEY'), ('anthropic', 'ANTHROPIC_API_KEY')]:
            if provider not in providers:
                continue
            value = os.environ.get(env)
            check(bool(value), f'{env} is not configured; credentials are never printed')
            self.keys[provider] = value
        self.calls = 0

    def request(self, provider, endpoint, data, content_type='application/json'):
        base = 'https://api.openai.com' if provider == 'openai' else 'https://api.anthropic.com'
        headers = {'Content-Type': content_type}
        if provider == 'openai':
            headers['Authorization'] = 'Bearer ' + self.keys[provider]
        else:
            headers.update({'x-api-key': self.keys[provider], 'anthropic-version': '2023-06-01'})
        request = urllib.request.Request(base + endpoint, data=data, headers=headers, method='POST')
        self.calls += 1
        # Never serialize HTTP exceptions or headers: they may contain sensitive request context.
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read())
                request_id = response.headers.get('x-request-id') or response.headers.get('request-id')
                return result, request_id
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Provider HTTP {error.code}; no automatic retry') from None
        except Exception:
            raise RuntimeError('Provider transport/parse failure; outcome may be unknown; no automatic retry') from None

    def upload(self, provider, path, sha):
        data = Path(path).read_bytes()
        check(hashlib.sha256(data).hexdigest() == sha, 'Input changed before upload')
        boundary = 'afterimage-' + uuid.uuid4().hex
        parts = []
        if provider == 'openai':
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nvision\r\n'.encode())
        parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{sha}.png"\r\nContent-Type: image/png\r\n\r\n'.encode(),
                  data, f'\r\n--{boundary}--\r\n'.encode()]
        result, _ = self.request(provider, '/v1/files', b''.join(parts), f'multipart/form-data; boundary={boundary}')
        return result['id']

    def count(self, provider, payload):
        if provider == 'openai':
            body = {k: payload[k] for k in ('model', 'input', 'reasoning')}
            endpoint = '/v1/responses/input_tokens'
        else:
            body = {k: payload[k] for k in ('model', 'messages', 'thinking')}
            endpoint = '/v1/messages/count_tokens'
        result, _ = self.request(provider, endpoint, canonical(body))
        check(type(result['input_tokens']) is int and result['input_tokens'] > 0, 'Invalid token count')
        return result['input_tokens']

    def infer(self, provider, payload):
        return self.request(provider, '/v1/responses' if provider == 'openai' else '/v1/messages', canonical(payload))

def parse_response(provider, response, task, rates):
    """Return charged usage even when output is invalid; billable reasoning is included."""
    u = response['usage']
    if provider == 'openai':
        inputs, outputs = u['input_tokens'], u['output_tokens']
        d = u.get('input_tokens_details', {})
        write, read = d.get('cache_write_tokens', 0), d.get('cached_tokens', 0)
        text = ''.join(c.get('text', '') for o in response.get('output', []) if o.get('type') == 'message'
                       for c in o.get('content', []) if c.get('type') == 'output_text')
        ended = response.get('status') == 'completed'
    else:
        write, read = u.get('cache_creation_input_tokens', 0), u.get('cache_read_input_tokens', 0)
        inputs, outputs = u['input_tokens'] + write + read, u['output_tokens']
        text = ''.join(b['text'] for b in response.get('content', []) if b.get('type') == 'text')
        ended = response.get('stop_reason') == 'end_turn'
    check(all(type(x) is int and x >= 0 for x in (inputs, outputs, write, read)), 'Invalid usage')
    cost = pricing_cost(provider, inputs, outputs, rates, write, read)
    reason = None
    description = None
    try:
        parsed = json.loads(text)
        check(isinstance(parsed, dict) and set(parsed) == {'description'}, 'Expected description-only JSON')
        description = parsed['description']
        check(isinstance(description, str) and description.strip(), 'Empty/non-string description')
        check(len(description.split()) <= (100 if task == 'GT1' else 180), 'Word limit exceeded')
    except (ValueError, TypeError):
        reason = 'invalid_json_or_output_contract'
    if not ended:
        reason = 'incomplete_or_nonstandard_stop'
    return {'status': 'completed' if reason is None else 'needs_review', 'review_reason': reason,
            'description_raw': text, 'description': description, 'usage': u, 'cost_usd': cost,
            'model_returned': response.get('model'), 'response_id': response.get('id')}

def reserve_cost(provider, count, settings):
    allowance = count + max(1024, math.ceil(count * .2))
    check(allowance + settings['max_output_tokens'] < (1050000 if provider == 'openai' else 1000000), 'Context reservation exceeds limit')
    return pricing_cost(provider, allowance, settings['max_output_tokens'], settings['usd_per_million'],
                        allowance if provider == 'openai' else 0)
