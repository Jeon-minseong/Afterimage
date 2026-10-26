"""Prepare immutable requests, generate per-video answers, score separately, summarize."""
import csv,datetime,json,os,shutil,statistics
from collections import defaultdict
from pathlib import Path
from .common import PACKAGE,CONDITIONS,check,read,write,sha,canonical,digest,safe,fresh,validate_inputs,pricing_cost
from .providers import Remote,make_payload,parse_response,reserve_cost

def finish_plan(out,config,jobs,kind,extra):
    write(out/'config.json',config)
    with (out/'requests.jsonl').open('wb') as f:
        for j in jobs:f.write(canonical(j)+b'\n')
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    record={'schema':'afterimage.plan.v1','kind':kind,'requests':len(jobs),'files':files,**extra}
    record['plan_sha256']=digest(record);write(out/'plan.json',record)
    return {'requests':len(jobs),'plan_sha256':record['plan_sha256'],'API_calls':0}

def prepare(inputs,out,config_path,repeats=None,ids=None):
    inputs=Path(inputs);config=read(config_path);m,validation=validate_inputs(inputs)
    repeat=repeats if repeats is not None else config['repeats'];check(repeat in (1,3),'Choose 1 or 3 repeats')
    check(tuple(config['conditions'])==CONDITIONS,'Only the four existing conditions are in scope')
    videos=[v for v in m['videos'] if not ids or v['video_id'] in ids]
    check(videos and (not ids or {v['video_id'] for v in videos}==set(ids)),'Unknown selected ID')
    out=fresh(out);(out/'prompts').mkdir();jobs=[]
    for name,p in config['prompts'].items():
        source=safe(PACKAGE,p['path']);check(sha(source)==p['sha256'],'Prompt hash mismatch')
        shutil.copy2(source,out/'prompts'/f'{name}.txt')
    for r in range(1,repeat+1):
        for v in videos:
            for task in (('GT1','GT2') if v['dataset']=='kisa' else ('GT1',)):
                for c in CONDITIONS:
                    for provider,s in config['models'].items():
                        pid=f'{task}_{"afterimage" if c=="afterimage" else "frames"}'
                        j={'kind':'inference','job_id':f'{provider}_{v["video_id"]}_{task}_{c}_r{r}',
                            'video_id':v['video_id'],'source_sha256':v['source_sha256'],'dataset':v['dataset'],
                            'provider':provider,'model':s['id'],'task':task,'condition':c,'repeat':r,
                            'images':v['conditions'][c],'prompt_path':f'prompts/{pid}.txt','prompt_sha256':sha(out/'prompts'/f'{pid}.txt')}
                        j['job_sha256']=digest(j);jobs.append(j)
    return finish_plan(out,config,jobs,'inference',{'input_manifest_sha256':sha(inputs/'manifest.json'),'input_validation':validation,'GT_sent':False})

def prepare_scores(answers,gt,out,config_path,allow_partial=False):
    answers=Path(answers);index=read(answers/'index.json');config=read(config_path)
    check(index['kind']=='inference','Expected inference results')
    check(allow_partial or index['complete'],'Answers incomplete; use --allow-partial explicitly for diagnostics')
    gtfile=read(gt);refs={(r['video_id'],r['task']):r['reference'] for r in gtfile['items']}
    check(len(refs)==len(gtfile['items']),'Duplicate GT references')
    out=fresh(out);(out/'prompts').mkdir();shutil.copy2(safe(PACKAGE,config['judge']['rubric']),out/'prompts/judge.txt')
    jobs=[]
    for item in index['items']:
        source=safe(answers,item['path']);check(sha(source)==item['sha256'],'Answer record hash mismatch')
        record=read(source);j=record['request'];check(record['status']=='completed','Cannot score invalid response')
        key=(j['video_id'],j['task']);check(key in refs,'GT missing: '+str(key))
        text=json.dumps({'task':j['task'],'reference':refs[key],'candidate':record['description']},ensure_ascii=False)
        job={'kind':'scoring','job_id':'judge_'+j['job_id'],'video_id':j['video_id'],'dataset':j['dataset'],
             'task':j['task'],'condition':j['condition'],'repeat':j['repeat'],
             'candidate_job_id':j['job_id'],'candidate_model':j['model'],'candidate_provider':j['provider'],
             'candidate_record_sha256':item['sha256'],'provider':config['judge']['provider'],
             'model':config['judge']['settings']['id'],'prompt_path':'prompts/judge.txt','prompt_sha256':sha(out/'prompts/judge.txt'),
             'grading_data':text,'images':[]}
        job['job_sha256']=digest(job);jobs.append(job)
    return finish_plan(out,config,jobs,'scoring',{'GT_sha256':sha(gt),'GT_status':gtfile['status'],
                       'answers_index_sha256':sha(answers/'index.json'),'rubric_status':'proposed_pending_final_discussion'})

def load_plan(path,inputs=None):
    path=Path(path);record=read(path/'plan.json')
    check(record['plan_sha256']==digest({k:v for k,v in record.items() if k!='plan_sha256'}),'Plan changed')
    for rel,h in record['files'].items():check(sha(safe(path,rel))==h,'Frozen plan file changed')
    jobs=[json.loads(l) for l in (path/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    check(len(jobs)==record['requests'] and len({j['job_id'] for j in jobs})==len(jobs),'Request count/uniqueness failed')
    for j in jobs:check(j['job_sha256']==digest({k:v for k,v in j.items() if k!='job_sha256'}),'Job changed')
    if record['kind']=='inference':
        check(inputs is not None,'Inference requires --inputs')
        check(sha(Path(inputs)/'manifest.json')==record['input_manifest_sha256'],'Input manifest changed')
        validate_inputs(inputs)
    return record,read(path/'config.json'),jobs

def parse_score(provider,raw,rates):
    parsed=parse_response(provider,raw,'GT2',rates)
    text=parsed['description_raw'];data=json.loads(text)
    check(set(data)=={'score','reason','needs_human_review'},'Score output fields differ')
    check(type(data['score']) is int and 0<=data['score']<=5 and isinstance(data['reason'],str) and type(data['needs_human_review']) is bool,'Invalid score')
    ended=raw.get('status')=='completed' if provider=='openai' else raw.get('stop_reason')=='end_turn'
    check(ended,'Incomplete scoring response')
    parsed.update(data);parsed['status']='completed';parsed.pop('description',None);parsed.pop('review_reason',None)
    return parsed

def execute(plan,inputs,out,ledger_path,caps,approved_sha=None,adapter=None):
    plan=Path(plan);out=Path(out).resolve();record,config,jobs=load_plan(plan,inputs)
    if approved_sha is None:return {'mode':'dry_run','requests':len(jobs),'plan_sha256':record['plan_sha256'],'API_calls':0}
    check(approved_sha==record['plan_sha256'],'Pass the exact reviewed --approve-plan hash')
    providers={j['provider'] for j in jobs}
    check(all(type(caps.get(p)) in (float,int) and 0<caps[p]<=100 for p in providers),'Explicit provider budget >0 and <=100 required')
    ledger_path=Path(ledger_path).resolve();ledger_path.parent.mkdir(parents=True,exist_ok=True)
    lock=ledger_path.with_suffix('.lock')
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        ledger=read(ledger_path) if ledger_path.exists() else {'requests':{},'uploads':{}}
        check(all(e['status']=='completed' for e in ledger['requests'].values()),'Unresolved ledger entry: reconcile before resuming')
        exposure=lambda p:sum(e.get('cost_usd',e.get('reserved_usd',0)) for e in ledger['requests'].values() if e['provider']==p)
        check(all(exposure(p)<=caps[p] for p in providers),'Previous cumulative cost exceeds cap')
        if (out/'run.json').exists():check(read(out/'run.json')['plan_sha256']==approved_sha,'Output belongs to another plan')
        else:
            check(not out.exists(),'Output exists without run manifest');out.mkdir(parents=True)
            write(out/'run.json',{'plan_sha256':approved_sha,'kind':record['kind'],'expected_requests':len(jobs),'API_mode':'live' if adapter is None else 'offline_fixture'})
        adapter=adapter or Remote(providers)
        for j in jobs:
            key=approved_sha+':'+j['job_id'];provider=j['provider'];s=config['models'][provider] if j['kind']=='inference' else config['judge']['settings']
            target=out/'videos'/j['video_id']/(j['job_id']+'.json')
            if key in ledger['requests']:
                e=ledger['requests'][key];check(e['result_path']==str(target) and sha(target)==e['result_sha256'],'Completed result missing/changed or output path differs');continue
            p=safe(plan,j['prompt_path']);check(sha(p)==j['prompt_sha256'],'Prompt changed')
            prompt=p.read_text(encoding='utf-8-sig');file_ids=[]
            for i in j['images']:
                image=safe(inputs,i['path']);check(sha(image)==i['sha256'],'Input changed')
                uploadkey=provider+':'+i['sha256']
                if uploadkey not in ledger['uploads']:
                    ledger['uploads'][uploadkey]=adapter.upload(provider,image,i['sha256']);write(ledger_path,ledger)
                file_ids.append(ledger['uploads'][uploadkey])
            if j['kind']=='scoring':prompt+='\n\nREFERENCE AND CANDIDATE DATA:\n'+j['grading_data']
            payload=make_payload(j,prompt,s,file_ids);count=adapter.count(provider,payload);reserve=reserve_cost(provider,count,s)
            if exposure(provider)+reserve>caps[provider]:
                export_index(out,jobs,record['kind']);return {'status':'budget_stopped','next_job':j['job_id'],'cumulative_usd':{p:exposure(p) for p in providers}}
            e={'status':'unknown','provider':provider,'reserved_usd':reserve,'job_sha256':j['job_sha256'],'payload_sha256':digest(payload),'count_input_tokens':count}
            ledger['requests'][key]=e;write(ledger_path,ledger)
            try:
                raw,rid=adapter.infer(provider,payload)
                rawfile=target.with_suffix('.raw.json');rawfile.parent.mkdir(parents=True,exist_ok=True)
                with rawfile.open('xb') as f:f.write(canonical(raw)+b'\n')
                result=parse_response(provider,raw,j['task'],s['usd_per_million']) if j['kind']=='inference' else parse_score(provider,raw,s['usd_per_million'])
                tier=raw.get('service_tier') if provider=='openai' else raw.get('usage',{}).get('service_tier')
                if raw.get('model')!=s['id'] or (tier is not None and tier!=('default' if provider=='openai' else 'standard')) or result['cost_usd']>reserve:
                    result.update(status='needs_review',review_reason='model_tier_or_reservation_mismatch')
                result.update(request=j,request_id=rid,raw_response_sha256=sha(rawfile),generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                write(target,result);e.update(status=result['status'],cost_usd=result['cost_usd'],result_path=str(target),result_sha256=sha(target));write(ledger_path,ledger)
                if result['status']!='completed':export_index(out,jobs,record['kind']);return {'status':'needs_review','job_id':j['job_id']}
            except Exception:
                write(ledger_path,ledger);raise RuntimeError('Request outcome/format unresolved; response retained if received. No automatic retry.') from None
        export_index(out,jobs,record['kind']);return {'status':'completed','requests':len(jobs),'cumulative_usd':{p:exposure(p) for p in providers}}
    finally:lock.unlink()

def export_index(out,jobs,kind):
    items=[]
    for j in jobs:
        p=Path(out)/'videos'/j['video_id']/(j['job_id']+'.json')
        if p.exists() and read(p)['status']=='completed':items.append({'video_id':j['video_id'],'job_id':j['job_id'],'path':p.relative_to(out).as_posix(),'sha256':sha(p)})
    write(Path(out)/'index.json',{'kind':kind,'expected_requests':len(jobs),'completed_requests':len(items),'complete':len(items)==len(jobs),'items':items})

def summarize(scores,out):
    scores=Path(scores);index=read(scores/'index.json');check(index['kind']=='scoring','Expected scoring results')
    out=fresh(out);rows=[];groups=defaultdict(list)
    for item in index['items']:
        p=safe(scores,item['path']);check(sha(p)==item['sha256'],'Score changed');r=read(p);j=r['request']
        row={k:j[k] for k in ['video_id','dataset','task','condition','repeat','candidate_model']};row.update(score=r['score'],needs_human_review=r['needs_human_review'],reason=r['reason'])
        rows.append(row);groups[(j['dataset'],j['task'],j['condition'],j['candidate_model'])].append(row)
    check(rows,'No completed scores')
    with (out/'individual_scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for key,g in groups.items():
        pervideo=defaultdict(list)
        for r in g:pervideo[r['video_id']].append(r['score'])
        means=[statistics.mean(v) for v in pervideo.values()]
        summary.append(dict(zip(['dataset','task','condition','model'],key),answers=len(g),videos=len(means),mean_score=statistics.mean(means),
            std_across_video_means=statistics.stdev(means) if len(means)>1 else None,needs_human_review=sum(r['needs_human_review'] for r in g)))
    write(out/'summary.json',{'complete':index['complete'],'aggregation':'Average repeats per video first; then average videos. GT1/GT2 and datasets separate. No inferential significance claim.',
                           'groups':summary,'source_index_sha256':sha(scores/'index.json')})
    return {'individual_scores':len(rows),'groups':len(summary),'complete':index['complete']}
