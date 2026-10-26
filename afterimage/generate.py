"""External video-directory entry point for the preserved C41 engine (Windows/CUDA)."""
import os,sys,shutil,subprocess,pickle
from pathlib import Path
from .common import PACKAGE,CONDITIONS,check,read,write,sha,fresh,validate_inputs

def import_fixed(project,out,ids=None):
    project=Path(project);index=read(project/'03_data/DATASET_INDEX.json')
    rows=[v for v in index if not ids or v['id'] in ids]
    check(rows and (not ids or {v['id'] for v in rows}==set(ids)),'Unknown video ID')
    out=fresh(out);videos=[]
    for v in rows:
        conditions={}
        for c,images in v['conditions'].items():
            conditions[c]=[]
            for i in images:
                source=project/i['path'];check(sha(source)==i['sha256'],'Source hash differs')
                rel='inputs/'+i['sha256']+'.png';dest=out/rel;dest.parent.mkdir(exist_ok=True)
                if not dest.exists():shutil.copy2(source,dest)
                conditions[c].append({'path':rel,'sha256':i['sha256'],'frame_id':i.get('frame_id')})
        videos.append({'video_id':v['id'],'dataset':v['dataset'],'source_sha256':v['source_sha256'],
            'source_relative_path':f'{v["dataset"]}/{v["id"]}.mp4','M':v['M'],'J':v['J'],'conditions':conditions})
    write(out/'manifest.json',{'schema':'afterimage.inputs.v1','origin':'import_of_frozen_inputs_not_regeneration','videos':videos})
    _,validation=validate_inputs(out);write(out/'validation.json',validation)
    return validation

def generate(videos,weights,out,ids=None,dataset='custom'):
    check(os.name=='nt','This release supports Windows/CUDA generation; API stages are platform independent')
    import cv2,numpy as np
    videos=Path(videos).resolve();weights=Path(weights).resolve()
    for rel,expected in read(PACKAGE/'configs/weights.json')['files'].items():check(sha(weights/rel)==expected,'Weight/config mismatch: '+rel)
    catalog=read(PACKAGE/'configs/dataset_catalog.json');byhash={r['source_sha256']:r for r in catalog}
    paths=sorted(p for p in videos.rglob('*') if p.suffix.lower() in ('.mp4','.avi','.mov','.mkv'))
    rows=[]
    for path in paths:
        # Standard catalog filenames permit inexpensive ID filtering before hashing.
        if ids and path.stem.startswith('v') and path.stem not in ids:continue
        h=sha(path);known=byhash.get(h)
        vid=known['video_id'] if known else 'v'+h[:12]
        if ids and vid not in ids:continue
        rows.append({'input_id':vid,'clip_id':vid,'dataset':known['dataset'] if known else dataset,'source':str(path),
            'source_sha256':h,'precursor_config':known['precursor_config'] if known else {**catalog[0]['precursor_config'],'resize':None},
            'source_relative_path':path.relative_to(videos).as_posix()})
    check(rows and len({r['input_id'] for r in rows})==len(rows),'No videos or duplicate video content')
    check(not ids or {r['input_id'] for r in rows}==set(ids),'Some requested IDs not found')
    out=fresh(out);run=out/'work';run.mkdir();engine=PACKAGE/'afterimage/engine'
    share=run/'연구_공유/20261017_CommonSplit_All'
    for sub in ['outputs','flow','episodes','tracks','detection']:(share/sub).mkdir(parents=True)
    for sub in ['only_split_ours_20261012/mask_cache','common_split_all_20261017/flow','common_split_all_20261017/render_work','common_split_all_20261017/detector_cache','package/random15_fixed6_laptop_20260918/work','main_experiment_native_20260928/native_signal_cache','overnight_20261016/yolo_settings']:(run/'local'/sub).mkdir(parents=True,exist_ok=True)
    for f in (engine/'local/only_split_ours_20261012').glob('*.py'):shutil.copy2(f,run/'local/only_split_ours_20261012'/f.name)
    env=os.environ.copy();env.update(AF_RUN_ROOT=str(run),AF_METHOD_ROOT=str(engine),AF_MODELS_ROOT=str(weights),
        HF_HOME=str(out/'runtime_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1')
    sys.path.insert(0,str(engine/'lib'));import flowsplit
    for r in rows:
        cfg=flowsplit.Config(**r['precursor_config']);cfg.resize=None;cfg.draw_time_bar=False
        meta=run/'metadata'/(r['input_id']+'.json');write(meta,{'config':r['precursor_config'],'segments':[]});r['metadata']=str(meta)
        sig=flowsplit.analyze(r['source'],cfg,print)
        with (run/'local/main_experiment_native_20260928/native_signal_cache'/(r['input_id']+'.pkl')).open('wb') as f:pickle.dump({'sig':sig},f)
    write(share/'INPUTS.json',rows);write(share/'flow/INPUTS.json',rows)
    write(out/'generation_protocol.json',{'mode':'cold','historical_caches_used':False,'video_ids':[r['input_id'] for r in rows],
          'source_code_sha256':{p.relative_to(engine).as_posix():sha(p) for p in engine.rglob('*.py')},'weights_sha256':read(PACKAGE/'configs/weights.json')['files']})
    for stage in ['detector','tracks','flow','episodes','generate']:
        print('STAGE',stage,flush=True)
        with (out/(stage+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(engine/'scripts'/f'common_full_{stage}_20261017.py')],cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    results=[]
    for r in rows:
        dest=share/'outputs'/r['input_id'];sel=read(dest/'SELECTION.json')['items'];selected=[fid for z in sel for fid in z['frame_ids']];M=len(selected);J=len(sel)
        cap=cv2.VideoCapture(r['source']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));check(n>0,'Empty video')
        uniform=lambda k:[int(x) for x in np.rint(np.linspace(0,n-1,k))] if k else []
        frame_ids={'uniform_observations':[0]+uniform(M)+[n-1],'selected_sequence':[0]+selected+[n-1],'uniform_budget':[0]+uniform(J)+[n-1]}
        wanted={x for seq in frame_ids.values() for x in seq};frames={}
        for i in range(max(wanted)+1):
            check(cap.grab(),'Video ended early')
            if i in wanted:
                ok,img=cap.retrieve();check(ok,'Frame decode failed')
                if r['dataset']=='ut_interaction':img=cv2.resize(img,(360,288),interpolation=cv2.INTER_AREA)
                ok,buf=cv2.imencode('.png',img);check(ok,'PNG encode failed');frames[i]=buf.tobytes()
        cap.release()
        def store(data,frame_id=None):
            import hashlib
            h=hashlib.sha256(data).hexdigest();rel='inputs/'+h+'.png';p=out/rel;p.parent.mkdir(exist_ok=True)
            if not p.exists():p.write_bytes(data)
            return {'path':rel,'sha256':h,'frame_id':frame_id}
        conditions={c:[store(frames[i],i) for i in indices] for c,indices in frame_ids.items()}
        conditions['afterimage']=[store((dest/name).read_bytes()) for name in ['state_start.png']+[f'afterimage_{i+1:03d}.png' for i in range(J)]+['state_end.png']]
        results.append({'video_id':r['input_id'],'dataset':r['dataset'],'source_sha256':r['source_sha256'],'source_relative_path':r['source_relative_path'],'M':M,'J':J,'conditions':conditions})
        target=out/'generation_records'/r['input_id'];target.mkdir(parents=True)
        for name in ['SELECTION.json','RECEIPT.json']:shutil.copy2(dest/name,target/name)
    write(out/'manifest.json',{'schema':'afterimage.inputs.v1','origin':'cold_generation','videos':results})
    _,validation=validate_inputs(out);write(out/'validation.json',validation);return validation
