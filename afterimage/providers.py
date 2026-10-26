import os, json, math, hashlib, uuid, urllib.request, urllib.error, re, base64, copy
from pathlib import Path
from .common import canonical, check, pricing_cost

def make_payload(job, prompt, settings, file_ids):
    """Only prompt and ordered image references go to the model. No GT or dataset metadata."""
    properties=({'score':{'type':'integer'},'reason':{'type':'string'},'needs_human_review':{'type':'boolean'}}
                if job.get('kind')=='scoring' else {'description':{'type':'string'}})
    schema={'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    if job['provider'] == 'openai':
        content = [{'type': 'input_text', 'text': prompt}]
        content += [{'type': 'input_image', 'file_id': f, 'detail': settings['image_detail']} for f in file_ids]
        result={'model': settings['id'], 'input': [{'role': 'user', 'content': content}],
                'reasoning': {'effort': settings['effort']}, 'max_output_tokens': settings['max_output_tokens'],
                'store': False, 'service_tier': 'default'}
        if settings.get('structured_output'):result['text']={'format':{'type':'json_schema','name':'afterimage_output','schema':schema,'strict':True}}
        return result
    content = [{'type': 'text', 'text': prompt}]
    content += [{'type': 'image', 'source': {'type': 'file', 'file_id': f}} for f in file_ids]
    result={'model': settings['id'], 'messages': [{'role': 'user', 'content': content}],
            'thinking': {'type': settings['thinking']}, 'output_config': {'effort': settings['effort']},
            'max_tokens': settings['max_output_tokens']}
    if settings.get('structured_output'):result['output_config']['format']={'type':'json_schema','schema':schema}
    return result

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
        self.local_files = {}
        self.openai_file_mode = False
        self.count_payload_limit_bytes = 24_000_000

    def bind_file(self, provider, file_id, path):
        self.local_files[(provider,file_id)] = Path(path)

    def materialize(self, provider, payload):
        payload=copy.deepcopy(payload)
        if provider=='openai':
            for message in payload['input']:
                for block in message['content']:
                    ref=block.get('file_id','')
                    if block.get('type')=='input_image' and ref.startswith('local:'):
                        data=self.local_files[(provider,ref)].read_bytes()
                        block.pop('file_id')
                        block['image_url']='data:image/png;base64,'+base64.b64encode(data).decode('ascii')
        return payload

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
            # Keep only bounded, redacted diagnostic fields, never request headers/body.
            try:
                detail=json.loads(error.read(65536)).get('error', {})
                if not isinstance(detail,dict):detail={}
                message=str(detail.get('message',''))[:1500]
                for key in self.keys.values():message=message.replace(key,'[REDACTED]')
                message=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',message)
                info={'http_status':error.code,'endpoint':endpoint,'code':detail.get('code'),
                      'type':detail.get('type'),'param':detail.get('param'),'message':message}
            except Exception:
                info={'http_status':error.code,'endpoint':endpoint,'message':'Error detail unavailable'}
            raise RuntimeError('Provider rejected request; no automatic retry: '+json.dumps(info)) from None
        except Exception:
            raise RuntimeError('Provider transport/parse failure; outcome may be unknown; no automatic retry') from None

    def upload(self, provider, path, sha):
        data = Path(path).read_bytes()
        check(hashlib.sha256(data).hexdigest() == sha, 'Input changed before upload')
        # OpenAI accepts exact image bytes inline, as in the earlier pilot.
        # Store only a local reference in the ledger, never a duplicated data URL.
        if provider=='openai' and not self.openai_file_mode:return 'local:'+sha
        boundary = 'afterimage-' + uuid.uuid4().hex
        parts = []
        if provider == 'openai':
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'.encode())
        parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{sha}.png"\r\nContent-Type: image/png\r\n\r\n'.encode(),
                  data, f'\r\n--{boundary}--\r\n'.encode()]
        result, _ = self.request(provider, '/v1/files', b''.join(parts), f'multipart/form-data; boundary={boundary}')
        return result['id']

    def count(self, provider, payload):
        if provider == 'openai':
            body = {k: payload[k] for k in ('model', 'input', 'reasoning')}
            if 'text' in payload:body['text']=payload['text']
            endpoint = '/v1/responses/input_tokens'
        else:
            body = copy.deepcopy({k: payload[k] for k in ('model', 'messages', 'thinking')})
            if 'output_config' in payload:body['output_config']=copy.deepcopy(payload['output_config'])
            # Claude count_tokens rejects Files API references. Count the identical
            # PNG bytes inline; inference still uses the ordered uploaded file IDs.
            for message in body['messages']:
                for block in message['content']:
                    source=block.get('source',{})
                    if block.get('type')=='image' and source.get('type')=='file':
                        path=self.local_files[(provider,source['file_id'])]
                        block['source']={'type':'base64','media_type':'image/png',
                                         'data':base64.b64encode(path.read_bytes()).decode('ascii')}
            endpoint = '/v1/messages/count_tokens'
        bodies=[body]
        if provider=='anthropic' and len(canonical(body))>self.count_payload_limit_bytes:
            check(len(body['messages'])==1,'Chunk counter supports one stateless user message')
            content=body['messages'][0]['content'];prefix=[b for b in content if b['type']!='image'];images=[b for b in content if b['type']=='image']
            check(content==prefix+images,'Unexpected content order for chunk counter')
            groups=[];group=[];size=len(canonical(prefix))+16384
            for block in images:
                n=len(canonical(block));check(n+16384<self.count_payload_limit_bytes,'Single image exceeds count request budget')
                if group and (size+n>self.count_payload_limit_bytes or len(group)>=50):groups.append(group);group=[];size=len(canonical(prefix))+16384
                group.append(block);size+=n
            if group:groups.append(group)
            bodies=[]
            for group in groups:
                b={k:v for k,v in body.items() if k!='messages'}
                b['messages']=[{'role':'user','content':prefix+group}];bodies.append(b)
        total=0
        for body in bodies:
            result, _ = self.request(provider, endpoint, canonical(body))
            check(type(result['input_tokens']) is int and result['input_tokens'] > 0, 'Invalid token count')
            total+=result['input_tokens']
        self.last_count_metadata={'mode':'single_request' if len(bodies)==1 else 'sum_of_chunk_counts_estimate','count_requests':len(bodies),'note':'Chunk counts repeat prompt/schema overhead; used for budget reservation only. Inference sends all ordered images in one request. Actual usage comes from the generation response.'}
        return total

    def infer(self, provider, payload):
        return self.request(provider, '/v1/responses' if provider == 'openai' else '/v1/messages', canonical(payload))

def json_output(text):
    """Accept a single complete JSON fence, never free-form prefacing/commentary."""
    match=re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*',text,re.DOTALL|re.IGNORECASE)
    return (match.group(1),'single_json_fence_removed') if match else (text,None)

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
        normalized, normalization = json_output(text)
        parsed = json.loads(normalized)
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
            'model_returned': response.get('model'), 'response_id': response.get('id'),
            'format_normalization':json_output(text)[1], 'strict_json_format':json_output(text)[1] is None}

def reserve_cost(provider, count, settings):
    allowance = count + max(1024, math.ceil(count * .2))
    check(allowance + settings['max_output_tokens'] < (1050000 if provider == 'openai' else 1000000), 'Context reservation exceeds limit')
    return pricing_cost(provider, allowance, settings['max_output_tokens'], settings['usd_per_million'],
                        allowance if provider == 'openai' else 0)
