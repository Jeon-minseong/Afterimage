"""Frozen, budget-reserved Batch API submission and resumable result collection.

No image transformations. Each request remains independent, including repeats.
An ambiguous submission is never automatically retried. Polling does not submit.
"""
import base64, copy, datetime, json, os, re, time, urllib.request, urllib.error, uuid
from pathlib import Path
from .common import canonical, check, digest, fresh, read, safe, sha, write
from .experiment import load_plan, export_index, parse_score
from .providers import Remote, make_payload, parse_response, reserve_cost


class BatchRemote(Remote):
    def get_bytes(self, provider, endpoint):
        check(endpoint.startswith('/v1/'), 'Unexpected API route')
        base = 'https://api.openai.com' if provider == 'openai' else 'https://api.anthropic.com'
        headers = {'Authorization': 'Bearer '+self.keys[provider]} if provider == 'openai' else {'x-api-key': self.keys[provider], 'anthropic-version': '2023-06-01'}
        try:
            with urllib.request.urlopen(urllib.request.Request(base+endpoint, headers=headers), timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Batch read failed: HTTP {error.code}; no generation was submitted by this read') from None
        except Exception:
            raise RuntimeError('Batch read transport failed; safe to poll again, never resubmit blindly') from None

    def upload_batch(self, path):
        boundary = 'afterimage-'+uuid.uuid4().hex
        data = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{sha(path)}.jsonl"\r\nContent-Type: application/jsonl\r\n\r\n').encode()
        data += Path(path).read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
        return self.request('openai', '/v1/files', data, 'multipart/form-data; boundary='+boundary)[0]['id']


def inline_payload(job, prompt, settings, inputs):
    body = make_payload(job, prompt, settings, [x['sha256'] for x in job['images']])
    if job['provider'] == 'openai':
        # Batch routing determines execution tier. All model reasoning settings stay fixed.
        body.pop('service_tier', None)
        blocks = body['input'][0]['content']
    else:
        blocks = body['messages'][0]['content']
    for block, item in zip(blocks[1:], job['images']):
        path = safe(inputs, item['path']);check(sha(path)==item['sha256'], 'Image changed')
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        if job['provider'] == 'openai':
            block.pop('file_id');block['image_url']='data:image/png;base64,'+encoded
        else:
            block['source']={'type':'base64','media_type':'image/png','data':encoded}
    return body


def prepare(plan, inputs, out, provider, adapter=None, max_bytes=24_000_000, max_requests=64):
    plan, inputs = Path(plan), Path(inputs) if inputs else None
    record, config, jobs = load_plan(plan, inputs)
    jobs = [j for j in jobs if j['provider']==provider]
    check(jobs, 'No requests for provider')
    check(100_000 <= max_bytes <= 24_000_000 and 1 <= max_requests <= 1000, 'Unsafe chunk limits')
    out = fresh(out);remote = adapter or BatchRemote([provider]);cache={};parts=[];entries=[];lines=[];size=0
    def save_part():
        if not lines:return
        filename=f'part_{len(parts)+1:03d}.jsonl'
        (out/filename).write_bytes(b''.join(lines))
        parts.append({'file':filename,'sha256':sha(out/filename),'jobs':copy.deepcopy(entries),
                      'requests':len(entries),'reserved_usd':sum(x['reserved_usd'] for x in entries),'bytes':(out/filename).stat().st_size})
    for n,job in enumerate(jobs):
        settings=config['models'][provider] if job['kind']=='inference' else config['judge']['settings']
        prompt=safe(plan,job['prompt_path']).read_text(encoding='utf-8-sig')
        if job['kind']=='scoring':prompt+='\n\nREFERENCE AND CANDIDATE DATA:\n'+job['grading_data']
        body=inline_payload(job,prompt,settings,inputs)
        payload_hash=digest(body)
        if payload_hash not in cache:
            cache[payload_hash]=remote.count(provider,body)
        count=cache[payload_hash]
        custom_id='j'+digest({'plan':record['plan_sha256'],'job':job['job_id']})[:40]
        request=({'custom_id':custom_id,'method':'POST','url':'/v1/responses','body':body}
                 if provider=='openai' else {'custom_id':custom_id,'params':body})
        line=canonical(request)+b'\n'
        check(len(line)+1024<=max_bytes, 'One complete request exceeds chunk limit; never split inference images')
        if lines and (size+len(line)+1024>max_bytes or len(lines)>=max_requests):
            save_part();entries=[];lines=[];size=0
        entries.append({'job_id':job['job_id'],'custom_id':custom_id,'payload_sha256':payload_hash,'input_tokens_count':count,
                        'reserved_usd':reserve_cost(provider,count,settings)*.5})
        lines.append(line);size+=len(line)
        if (n+1)%32==0:print(json.dumps({'event':'batch_prepared','provider':provider,'requests':n+1,'total':len(jobs)}),flush=True)
    save_part()
    result={'schema':'afterimage.batch.v1','provider':provider,'plan_sha256':record['plan_sha256'],
            'parts':parts,'requests':len(jobs),'counted_unique_payloads':len(cache),'price_multiplier':.5,
            'reserved_usd':sum(x['reserved_usd'] for x in parts),'transport':'exact_inline_png_bytes',
            'source_files':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    result['pack_sha256']=digest(result);write(out/'pack.json',result)
    return {'provider':provider,'requests':len(jobs),'parts':len(parts),'reserved_usd':result['reserved_usd'],'pack_sha256':result['pack_sha256'],'generation_requests':0}


def load_pack(pack, approved=None):
    pack=Path(pack);record=read(pack/'pack.json')
    check(record['pack_sha256']==digest({k:v for k,v in record.items() if k!='pack_sha256'}),'Batch pack changed')
    if approved is not None:check(approved==record['pack_sha256'],'Exact pack approval required')
    for part in record['parts']:check(sha(safe(pack,part['file']))==part['sha256'],'Batch body changed')
    return record


def submit(pack, ledger_path, cap, approved, adapter=None, limit=None):
    record=load_pack(pack,approved);pack=Path(pack);provider=record['provider']
    check(type(cap) in (int,float) and 0<cap<=100,'Provider cap must be >0 and <=100')
    ledger_path=Path(ledger_path);lock=ledger_path.with_suffix('.lock')
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        ledger=read(ledger_path);remote=adapter or BatchRemote([provider]);submitted=[]
        allowed={'completed','reviewed_invalid','batch_pending','failed_no_charge'}
        check(all(x['status'] in allowed for x in ledger['requests'].values()),'Unresolved request blocks new submissions')
        for part in record['parts']:
            state_path=pack/(part['file']+'.state.json')
            if state_path.exists():
                state=read(state_path)
                check(state['status'] in ('submitted','collected'),'Ambiguous batch state; reconcile, never resubmit')
                continue
            if limit is not None and len(submitted)>=limit:break
            exposure=sum(x.get('cost_usd',x.get('reserved_usd',0)) for x in ledger['requests'].values() if x['provider']==provider)
            check(exposure+part['reserved_usd']<=cap,'Batch reservation exceeds cumulative provider budget')
            keys=[record['plan_sha256']+':'+x['job_id'] for x in part['jobs']]
            check(all(k not in ledger['requests'] for k in keys),'Job already reserved/submitted by another pack')
            file_id=remote.upload_batch(pack/part['file']) if provider=='openai' else None
            state={'status':'submitting_unknown','provider':provider,'pack_sha256':record['pack_sha256'],
                   'input_file_id':file_id,'part_sha256':part['sha256'],'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            for k,j in zip(keys,part['jobs']):
                ledger['requests'][k]={'status':'batch_submitting_unknown','provider':provider,'reserved_usd':j['reserved_usd'],
                                      'payload_sha256':j['payload_sha256'],'count_input_tokens':j['input_tokens_count']}
            write(ledger_path,ledger);write(state_path,state)
            if provider=='openai':
                payload={'input_file_id':file_id,'endpoint':'/v1/responses','completion_window':'24h',
                         'metadata':{'experiment':record['plan_sha256'][:32],'part':part['sha256'][:32]}}
                remote_state,_=remote.request(provider,'/v1/batches',canonical(payload))
            else:
                payload={'requests':[json.loads(x) for x in (pack/part['file']).read_text(encoding='utf-8').splitlines()]}
                remote_state,_=remote.request(provider,'/v1/messages/batches',canonical(payload))
            state.update(status='submitted',batch_id=remote_state['id'],remote=remote_state)
            write(state_path,state)
            for k in keys:ledger['requests'][k].update(status='batch_pending',batch_id=remote_state['id'])
            write(ledger_path,ledger);submitted.append(remote_state['id'])
            print(json.dumps({'event':'batch_submitted','provider':provider,'batch_id':remote_state['id'],'requests':part['requests']}),flush=True)
        return {'submitted_batches':len(submitted),'batch_ids':submitted}
    finally:lock.unlink()


def watch(packs, plan, inputs, out, ledger_path, interval=60, hours=25, adapter=None):
    """Collect already-submitted packs only. Never creates/retries generation."""
    check(30<=interval<=300 and 0<hours<=48,'Invalid watch interval/deadline')
    for pack in packs:
        record=load_pack(pack)
        for part in record['parts']:
            state_path=Path(pack)/(part['file']+'.state.json')
            check(state_path.exists() and read(state_path)['status'] in ('submitted','collected'), 'Submit/reconcile all parts before watching')
    deadline=time.monotonic()+hours*3600
    while True:
        updates=[collect(pack,plan,inputs,out,ledger_path,adapter=adapter) for pack in packs]
        all_terminal=all(read(Path(pack)/(part['file']+'.state.json'))['status']=='collected' for pack in packs for part in load_pack(pack)['parts'])
        index=read(Path(out)/'index.json')
        status={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'status':('completed' if index['complete'] else 'needs_review') if all_terminal else 'processing',
                'completed_requests':index['completed_requests'],'expected_requests':index['expected_requests'],'updates':updates}
        write(Path(out)/'watch_status.json',status);print(json.dumps(status),flush=True)
        if all_terminal:return status
        if time.monotonic()>=deadline:
            status['status']='watch_deadline_reached';write(Path(out)/'watch_status.json',status);return status
        time.sleep(interval)


def collect(pack, plan, inputs, out, ledger_path, adapter=None):
    pack=Path(pack);packed=load_pack(pack);provider=packed['provider']
    record,config,jobs=load_plan(plan,inputs)
    check(record['plan_sha256']==packed['plan_sha256'],'Wrong inference plan')
    by_id={j['job_id']:j for j in jobs};out=Path(out);ledger_path=Path(ledger_path)
    lock=ledger_path.with_suffix('.lock')
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        ledger=read(ledger_path);remote=adapter or BatchRemote([provider])
        if not (out/'run.json').exists():
            check(not out.exists(),'Output directory exists without run record');out.mkdir(parents=True)
            write(out/'run.json',{'plan_sha256':record['plan_sha256'],'kind':record['kind'],'expected_requests':len(jobs),'API_mode':'live_batch'})
        else:check(read(out/'run.json')['plan_sha256']==record['plan_sha256'],'Output belongs to another plan')
        pending=0;collected=0;issues=0
        for part in packed['parts']:
            state_path=pack/(part['file']+'.state.json')
            if not state_path.exists():continue
            state=read(state_path)
            check(state['status'] in ('submitted','collected'),'Unknown submission must be reconciled')
            if state['status']=='collected':continue
            batch_id=state['batch_id'];check(re.fullmatch(r'[A-Za-z0-9_-]+',batch_id),'Invalid batch ID')
            endpoint=('/v1/batches/' if provider=='openai' else '/v1/messages/batches/')+batch_id
            status=json.loads(remote.get_bytes(provider,endpoint));write(pack/(part['file']+'.latest.json'),status)
            terminal=(status.get('status') in ('completed','failed','expired','cancelled') if provider=='openai' else status.get('processing_status')=='ended')
            if not terminal:pending+=part['requests'];continue
            rows=[]
            if provider=='openai':
                for field in ['output_file_id','error_file_id']:
                    file_id=status.get(field)
                    if file_id:
                        check(re.fullmatch(r'[A-Za-z0-9_-]+',file_id),'Invalid result file ID')
                        data=remote.get_bytes(provider,'/v1/files/'+file_id+'/content')
                        raw_path=pack/(part['file']+'.'+field+'.jsonl')
                        if raw_path.exists():check(raw_path.read_bytes()==data,'Remote results changed')
                        else:raw_path.write_bytes(data)
                        rows.extend(json.loads(line) for line in data.splitlines() if line.strip())
            else:
                data=remote.get_bytes(provider,endpoint+'/results');raw_path=pack/(part['file']+'.results.jsonl')
                if raw_path.exists():check(raw_path.read_bytes()==data,'Remote results changed')
                else:raw_path.write_bytes(data)
                rows=[json.loads(line) for line in data.splitlines() if line.strip()]
            mapping={j['custom_id']:j for j in part['jobs']};seen=set()
            check(all(r['custom_id'] in mapping for r in rows),'Unknown custom ID in result')
            for row in rows:
                cid=row['custom_id'];check(cid not in seen,'Duplicate result custom ID');seen.add(cid)
                info=mapping[cid];job=by_id[info['job_id']];key=record['plan_sha256']+':'+job['job_id'];entry=ledger['requests'][key]
                target=out/'videos'/job['video_id']/(job['job_id']+'.json')
                if entry['status']=='completed':
                    check(sha(target)==entry['result_sha256'],'Completed result changed');continue
                response=row.get('response',{}) or {}
                succeeded=(response.get('status_code')==200 and not row.get('error') if provider=='openai' else row['result']['type']=='succeeded')
                if not succeeded:
                    entry['status']='batch_error_needs_review';issues+=1
                    if provider=='anthropic' and row['result']['type'] in ('errored','canceled','expired'):
                        entry.update(status='failed_no_charge',cost_usd=0)
                    write(ledger_path,ledger);continue
                raw=response['body'] if provider=='openai' else row['result']['message']
                rawfile=target.with_suffix('.raw.json');rawfile.parent.mkdir(parents=True,exist_ok=True)
                if rawfile.exists():check(read(rawfile)==raw,'Raw result differs on resume')
                else:write(rawfile,raw)
                settings=config['models'][provider] if job['kind']=='inference' else config['judge']['settings']
                rates={k:v*.5 for k,v in settings['usd_per_million'].items()}
                result=parse_response(provider,raw,job['task'],rates) if job['kind']=='inference' else parse_score(provider,raw,rates)
                if raw.get('model')!=settings['id'] or result['cost_usd']>entry['reserved_usd']:
                    result.update(status='needs_review',review_reason='model_or_reservation_mismatch')
                result.update(request=job,request_id=response.get('request_id') or raw.get('id'),raw_response_sha256=sha(rawfile),
                              generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),batch_id=batch_id,
                              batch_custom_id=cid,processing_mode='batch',pricing_multiplier=.5)
                if target.exists():check(read(target)==result or entry['status']=='completed','Existing result requires reconciliation')
                else:write(target,result)
                entry.update(status=result['status'],cost_usd=result['cost_usd'],result_path=str(target),result_sha256=sha(target))
                write(ledger_path,ledger);collected+=1;issues+=result['status']!='completed'
            missing=set(mapping)-seen
            if missing:
                issues+=len(missing)
                for cid in missing:
                    key=record['plan_sha256']+':'+mapping[cid]['job_id']
                    ledger['requests'][key]['status']='batch_missing_result_needs_review'
                write(ledger_path,ledger)
            state.update(status='collected',terminal_state=status,received_results=len(rows),missing_results=len(missing))
            write(state_path,state)
        export_index(out,jobs,record['kind'])
        return {'newly_collected':collected,'pending_requests':pending,'issues':issues,
                'complete':read(out/'index.json')['complete'],'completed_requests':read(out/'index.json')['completed_requests']}
    finally:lock.unlink()
