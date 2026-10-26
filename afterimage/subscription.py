"""Independent Codex ChatGPT-login trials. No paid API client or API key fallback.

CLI input/output token counts include the Codex harness and are NOT API-only
image tokens. Dollar cost is unknown, never imputed from API prices.
"""
import argparse,collections,csv,datetime,json,os,shutil,subprocess,tempfile,time,uuid
from pathlib import Path
from .common import read,write,sha,digest,safe,check
from .experiment import prepare,load_plan,export_index

DISABLED=['apps','plugins','hooks','memories','shell_tool','unified_exec','shell_snapshot',
          'browser_use','browser_use_external','computer_use','image_generation','multi_agent',
          'sleep_tool','view_image','skill_search','workspace_dependencies','code_mode_host','unbounded_connection_retries']
EXPECTED_WARNINGS=['Under-development features enabled: skip_host_skill_discovery.',
                   'Code Mode is unavailable because code-mode host is disabled.']
SCHEMA={'type':'object','properties':{'description':{'type':'string'}},'required':['description'],'additionalProperties':False}

def clean_env():
    env=os.environ.copy()
    for key in ('OPENAI_API_KEY','CODEX_API_KEY','ANTHROPIC_API_KEY','OPENAI_BASE_URL','CODEX_THREAD_ID','CODEX_TURN_ID'):
        env.pop(key,None)
    return env

def command(binary,job,settings,workspace,final):
    args=[binary,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
          '--sandbox','read-only','-m',job['model'],'-c','model_reasoning_effort='+json.dumps(settings['effort']),
          '-c','project_doc_max_bytes=0','-c','web_search="disabled"',
          '-c','model_provider="afterimage-subscription"',
          '-c','model_providers.afterimage-subscription.name="OpenAI subscription HTTP"',
          '-c','model_providers.afterimage-subscription.requires_openai_auth=true',
          '-c','model_providers.afterimage-subscription.wire_api="responses"',
          '-c','model_providers.afterimage-subscription.supports_websockets=false',
          '-c','model_providers.afterimage-subscription.stream_max_retries=0',
          '-c','model_providers.afterimage-subscription.request_max_retries=0','--json',
          '--output-schema','schema.json','-o',str(final)]
    for feature in DISABLED:args+=['--disable',feature]
    args+=['--enable','skip_host_skill_discovery']
    for i in range(len(job['images'])):args+=['-i',f'i{i:04}.png']
    return args+['-']

def parse_events(events,final,task):
    starts=[e['thread_id'] for e in events if e.get('type')=='thread.started']
    complete=[e for e in events if e.get('type')=='turn.completed']
    check(len(starts)==len(complete)==1,'Expected one fresh session and one completed turn')
    check(not any(e.get('type') in ('error','turn.failed') for e in events),'CLI failure; no automatic retry')
    messages=[]
    for e in events:
        if e.get('type') not in ('item.started','item.completed'):continue
        item=e['item'];kind=item['type']
        check(kind in ('agent_message','reasoning','error'),'Tool activity detected; exclude contaminated trial')
        if kind=='error':check(any(item.get('message','').startswith(w) for w in EXPECTED_WARNINGS),'Unexpected CLI warning/error')
        if kind=='agent_message' and e['type']=='item.completed':messages.append(item['text'])
    check(len(messages)==1 and messages[0].strip()==final.strip(),'Expected one final answer without commentary')
    value=json.loads(final)
    check(set(value)=={'description'} and isinstance(value['description'],str) and value['description'].strip(),'Invalid description JSON')
    check(len(value['description'].split())<=(100 if task=='GT1' else 180),'Description exceeds contract')
    usage=complete[0]['usage']
    check(all(isinstance(usage.get(k),int) and usage[k]>=0 for k in ('input_tokens','output_tokens')),'Missing CLI usage')
    return starts[0],value['description'],usage

def stage_inputs(job,inputs,workspace):
    ordered=[]
    for i,e in enumerate(job['images']):
        source=safe(inputs,e['path']);check(sha(source)==e['sha256'],'Input hash changed')
        target=workspace/f'i{i:04}.png'
        # Distinct slot names preserve repeated appearances even for identical bytes.
        shutil.copyfile(source,target)
        check(sha(target)==e['sha256'],'Staging changed image bytes')
        ordered.append({'slot':i,'filename':target.name,'sha256':e['sha256']})
    return ordered

def summarize(out,jobs):
    export_index(out,jobs,'inference')
    items=read(out/'index.json')['items'];rows=[]
    for item in items:
        a=read(out/item['path']);j=a['request'];u=a['usage']
        rows.append({'video_id':j['video_id'],'dataset':j['dataset'],'task':j['task'],'condition':j['condition'],
          'repeat':j['repeat'],'model_requested':j['model'],'execution_mode':a['processing_mode'],
          'image_appearances':len(j['images']),'input_tokens_including_harness':u['input_tokens'],
          'output_tokens_including_reasoning':u['output_tokens'],'cached_input_tokens':u.get('cached_input_tokens'),
          'reasoning_output_tokens':u.get('reasoning_output_tokens'),'cost_usd':None,'session_id':a['session_id']})
    if rows:
        with (out/'subscription_usage.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['dataset'],row['task'],row['condition'])].append(row)
    write(out/'subscription_summary.json',{'completed':len(rows),'expected':len(jobs),
        'usage_basis':'Codex CLI turn.completed; includes harness, not interchangeable with API payload usage',
        'api_key_auth_used':False,'dollar_cost_known':False,
        'groups':[{'dataset':k[0],'task':k[1],'condition':k[2],'n':len(v),
            'mean_input_tokens':sum(x['input_tokens_including_harness'] for x in v)/len(v),
            'mean_output_tokens':sum(x['output_tokens_including_reasoning'] for x in v)/len(v)} for k,v in groups.items()]})

def execute(plan,inputs,out,approved=None,limit=None,timeout=900,binary='codex'):
    plan=Path(plan).resolve();inputs=Path(inputs).resolve();out=Path(out).resolve()
    manifest,config,jobs=load_plan(plan,inputs)
    check(config.get('processing_mode')=='codex_chatgpt_subscription','Subscription-specific plan required')
    check(all(j['provider']=='openai' for j in jobs),'Only GPT jobs permitted')
    if approved is None:return {'mode':'dry_run','requests':len(jobs),'plan_sha256':manifest['plan_sha256']}
    check(approved==manifest['plan_sha256'],'Plan approval hash mismatch')
    executable=shutil.which(binary);check(executable is not None,'Codex CLI missing')
    env=clean_env()
    login=subprocess.run([executable,'login','status'],capture_output=True,text=True,encoding='utf-8',env=env,timeout=30)
    check(login.returncode==0 and 'Logged in using ChatGPT' in login.stdout+login.stderr,'ChatGPT subscription login required; never fall back to API')
    version=subprocess.check_output([executable,'--version'],env=env,text=True,encoding='utf-8').strip()
    out.mkdir(parents=True,exist_ok=True);lock=out/'runner.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        if (out/'run.json').exists():
            r=read(out/'run.json');check(r['plan_sha256']==approved and r['cli_version']==version and r['runner_sha256']==sha(Path(__file__)),'Run, CLI or source changed')
        else:write(out/'run.json',{'plan_sha256':approved,'kind':'inference','expected_requests':len(jobs),
                'processing_mode':'codex_chatgpt_subscription','cli_version':version,'runner_sha256':sha(Path(__file__)),
                'API_mode':False,'cost_usd':None,'schema':SCHEMA,'disabled_features':DISABLED,
                'limitations':['Codex system harness differs from API','No API output-token cap or image-detail setting equivalence claimed',
                    'Requested model retained; CLI does not return a model version in final events','Source PNG bytes/order verified; internal image preprocessing not audited',
                    'Subscription inference still uses provider servers, not local GPU inference']})
        completed=0;launched=0;seen=set()
        for j in jobs:
            if (out/'STOP_REQUESTED').exists():return {'status':'stopped','completed':completed}
            target=out/'videos'/j['video_id']/(j['job_id']+'.json')
            receipt=out/'receipts'/(j['job_id']+'.json')
            if target.exists():
                a=read(target);check(a['status']=='completed' and a['request']==j and a['plan_sha256']==approved,'Existing record invalid')
                check(receipt.exists() and read(receipt).get('result_sha256')==sha(target),'Result hash mismatch')
                check(a['session_id'] not in seen,'Independent sessions required');seen.add(a['session_id'])
                check(sha(out/a['raw_events_path'])==a['raw_events_sha256'],'CLI events changed')
                completed+=1;continue
            check(not receipt.exists(),'Prior attempt unresolved; no automatic retry')
            if limit is not None and launched>=limit:break
            prompt=safe(plan,j['prompt_path']);check(sha(prompt)==j['prompt_sha256'],'Prompt changed')
            workspace=Path(tempfile.gettempdir())/('afterimage_trial_'+uuid.uuid4().hex);workspace.mkdir()
            staged=stage_inputs(j,inputs,workspace);write(workspace/'schema.json',SCHEMA)
            rawdir=out/'raw';rawdir.mkdir(exist_ok=True)
            final=rawdir/(j['job_id']+'.final.json');eventsfile=rawdir/(j['job_id']+'.events.jsonl');errfile=rawdir/(j['job_id']+'.stderr.txt')
            args=command(executable,j,config['models']['openai'],workspace,final)
            attempt={'status':'running','job_sha256':j['job_sha256'],'plan_sha256':approved,'cli_version':version,
                'ordered_images':staged,'prompt_sha256':sha(prompt),'command':args,'working_directory':str(workspace),
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write(receipt,attempt)
            write(out/'STATUS.json',{'phase':'running','completed':completed,'expected':len(jobs),'current_job':j['job_id'],'pid':os.getpid(),
                'processing_mode':'codex_chatgpt_subscription','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            with eventsfile.open('xb') as stdout,errfile.open('xb') as stderr:
                p=subprocess.Popen(args,cwd=workspace,env=env,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr)
                try:p.communicate(prompt.read_bytes(),timeout=timeout)
                except subprocess.TimeoutExpired:
                    p.kill();p.communicate();attempt['status']='timeout_unknown';write(receipt,attempt)
                    raise RuntimeError('CLI timed out; attempt retained, no automatic retry')
            check(p.returncode==0,'CLI stopped (possibly quota/auth/runtime failure); no automatic retry')
            events=[json.loads(l) for l in eventsfile.read_text(encoding='utf-8').splitlines() if l.strip()]
            text=final.read_text(encoding='utf-8');sid,description,usage=parse_events(events,text,j['task'])
            check(sid not in seen,'Session unexpectedly reused');seen.add(sid)
            result={'status':'completed','description':description,'description_raw':text,'request':j,'plan_sha256':approved,
                'processing_mode':'codex_chatgpt_subscription','auth_mode':'ChatGPT login','session_id':sid,'model_requested':j['model'],
                'model_returned':None,'cli_version':version,'usage':usage,'usage_basis':'CLI turn.completed including harness',
                'cost_usd':None,'api_key_auth_used':False,'raw_events_path':eventsfile.relative_to(out).as_posix(),
                'raw_events_sha256':sha(eventsfile),'raw_response_sha256':sha(final),
                'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write(target,result);attempt.update(status='completed',result_sha256=sha(target),session_id=sid);write(receipt,attempt)
            # Remove only this trial's explicitly staged copies, never source inputs.
            for entry in staged:(workspace/entry['filename']).unlink()
            (workspace/'schema.json').unlink()
            if not any(workspace.iterdir()):workspace.rmdir()
            launched+=1;completed+=1
            if completed % 25 == 0:summarize(out,jobs)
            write(out/'STATUS.json',{'phase':'running','completed':completed,'expected':len(jobs),
                'last_completed_job':j['job_id'],'pid':os.getpid(),'processing_mode':'codex_chatgpt_subscription',
                'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            print(json.dumps({'completed':completed,'expected':len(jobs),'job_id':j['job_id'],'session_id':sid}),flush=True)
        summarize(out,jobs)
        state={'phase':'completed' if completed==len(jobs) else 'limited_run_finished','completed':completed,'expected':len(jobs)}
        write(out/'STATUS.json',state);return state
    except Exception as e:
        summarize(out,jobs)
        write(out/'STATUS.json',{'phase':'needs_attention','error':str(e)[:1500],'pid':os.getpid(),
              'no_automatic_retry':True,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
    finally:lock.unlink()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approve-plan');parser.add_argument('--limit',type=int);parser.add_argument('--timeout',type=int,default=900)
    a=parser.parse_args();check(not a.execute or a.approve_plan,'Execution requires exact --approve-plan')
    print(json.dumps(execute(a.plan,a.inputs,a.out,a.approve_plan if a.execute else None,a.limit,a.timeout)))
if __name__=='__main__':main()
