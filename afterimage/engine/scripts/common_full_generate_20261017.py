import os
"""Flow-only representative-window prototype. Shared selection rules; no labels/GT."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
import sys,json,time,pickle,copy,hashlib,html,shutil,subprocess,msvcrt
from pathlib import Path
import cv2,numpy as np
from scipy.signal import find_peaks
ROOT=Path(os.environ['AF_RUN_ROOT']);LOCAL=(Path(os.environ['AF_RUN_ROOT'])/'local')
sys.path.insert(0,str(LOCAL/'only_split_ours_20261012'));import build as B
FS=B.FS;G=B.G
OUT=ROOT/'연구_공유/20261017_CommonSplit_All/outputs';OUT.mkdir(exist_ok=True)
WORK=LOCAL/'common_split_all_20261017/render_work';WORK.mkdir(exist_ok=True)
NAMES=['WZ_ira_bend','WZ_ira_jack','WZ_daria_pjump','WZ_daria_run','WZ_denis_wave2','UT_s01_hand_shake_01','UT_s01_hugging_01','UT_s01_kick_01','UT_s01_punching_01','UT_s01_push_01','KISA_싸움__C045204_003','KISA_방화__C050105_002','KISA_배회__C054300_008','KISA_불법주정차__C038200_008','KISA_쓰러짐__C050203_002']
ROWS=B.read(ROOT/'연구_공유/20261017_CommonSplit_All/INPUTS.json');B.write(OUT/'INPUTS.json',ROWS)
PARAM=dict(active_area=.0004,join_gap_sec=2.,min_span_sec=.3,peak_distance_sec=.8,peak_prominence_fraction=.15,anchor_lag_sec=.1,compact_lookback_sec=.6,travel_lookback_sec=1.5,compact_iou=.35,past_visible_fraction=.12,max_keys=3,duplicate_iou=.55,duplicate_direction_cos=.5,duplicate_reversal_delta=.25)
PARAM.update(episode_footprint_object_units=6.,episode_new_area_object_units=.5,episode_overlap=.3,episode_max_representatives=3,episode_gap_sec=3.,scale_rule="max(reference object area,current moving area)")
B.write(OUT/'PROTOCOL.json',dict(method='C35 common flow + C38 spatial episode retention + appearance diverse selection + geometry bounded travel + common ROI RMBG soft composition',hypothesis='Remove empty background cuts and spatially redundant repetitions while preserving distinct postures',limitations='Detector misses, repeated motion reduction, no VLM validation; cached runtime not cold runtime',gpu='0 sequential'))
B.write(OUT/'INPUTS.json',ROWS)
def rel(p,d):return Path(os.path.relpath(p,d)).as_posix()
def load(row):
 p=LOCAL/'common_split_all_20261017/flow'/(row['input_id']+'.pkl');data=pickle.load(p.open('rb'));sig=data['sig']
 m=B.read(row['metadata']);m.update(config=data['cfg'],analysis_stride=sig.stride,fps=sig.fps,segments=B.read(ROOT/'연구_공유/20261017_CommonSplit_All/episodes'/(row['input_id']+'.json'))['segments'])
 return m,sig,p

def select(sig):
 t0=time.perf_counter();fps=sig.fps;n=sig.n
 masks=np.stack([cv2.resize(FS.unpack(sig,t).astype(np.float32),(96,54),interpolation=cv2.INTER_AREA)>.2 for t in range(n)])
 area=np.asarray(getattr(sig,'motion_area_sus',sig.motion_area));cfg=FS.Config();cfg.active_area_frac=PARAM['active_area'];cfg.static_gap_sec=PARAM['join_gap_sec'];cfg.min_span_sec=PARAM['min_span_sec']
 spans=FS.active_spans(sig,cfg);flux=np.maximum(sig.flux,0).astype(float);flux=np.where(area>PARAM['active_area'],flux,0)
 smooth=FS._smooth(flux,max(1,round(.2*fps)));candidates=[]
 for si,(a,b) in enumerate(spans):
  vals=smooth[a:b+1];spread=float(np.percentile(vals,90)-np.percentile(vals,10));prom=max(1e-8,PARAM['peak_prominence_fraction']*spread)
  peaks,_=find_peaks(vals,prominence=prom,distance=max(1,round(PARAM['peak_distance_sec']*fps)))
  anchors=sorted(set([min(b,a+round(.6*fps)),b]+[min(b,a+int(x)+round(PARAM['anchor_lag_sec']*fps)) for x in peaks]))
  for e in anchors:
   if not masks[e].any() and max(area[max(a,e-round(.3*fps)):e+1],default=0)<=PARAM['active_area']:continue
   short=max(a,e-round(PARAM['compact_lookback_sec']*fps));u=masks[short:e+1].any(0);inter=(masks[short]&masks[e]).sum();iou=float(inter/max(1,(masks[short]|masks[e]).sum()))
   compact=iou>=PARAM['compact_iou'];s=short if compact else max(a,e-round(PARAM['travel_lookback_sec']*fps))
   # Bound travel window by newly swept area, relative to moving object area.
   scale=max(1,float(sig.obj_area)*96*54/(sig.aw*sig.ah),float(masks[e].sum()));u=masks[e].copy()
   for j in range(e-1,s-1,-1):
    u|=masks[j]
    if u.sum()/scale>3.0:s=j+1;break
   s=min(s,e);keys=[e];future=masks[e].copy();options=list(range(s,e))
   if options:
    # Prefer separated changes, not K mandatory poses. Visibility is a flow-mask proxy.
    score=[float((masks[j]&~future).sum()/max(1,masks[j].sum())) for j in options]
    earliest=next((j for j,v in zip(options,score) if v>=PARAM['past_visible_fraction']),None)
    if earliest is not None:
     keys.append(earliest);future|=masks[earliest]
     middle=[j for j in range(earliest+1,e) if min(j-earliest,e-j)>=max(1,round(.15*fps))]
     if middle:
      scores=[float((masks[j]&~future).sum()/max(1,masks[j].sum())) for j in middle];q=int(np.argmax(scores))
      if scores[q]>=PARAM['past_visible_fraction']:keys.append(middle[q])
   keys=sorted(keys);u=masks[s:e+1].any(0)
   vec=np.array([np.mean(sig.mean_vx[s:e+1]),np.mean(sig.mean_vy[s:e+1])]);rev=float(np.mean(sig.reversal[s:e+1]));quality=float(smooth[e])
   candidates.append(dict(span=si,start=s,end=e,keys=keys,compact=compact,union=u,direction=vec,reversal=rev,quality=quality,scale=scale,mandatory=e in [anchors[0],anchors[-1]],reason='compact local change' if compact else 'bounded travel change'))
 # Suppress genuinely similar spatial/directional candidates. Keep entry/end per activity span.
 accepted=[];rejected=[]
 for c in candidates:
  match=None
  if not c['mandatory']:
   for k,p in enumerate(accepted):
    if p['span']!=c['span']:continue
    iou=float((p['union']&c['union']).sum()/max(1,(p['union']|c['union']).sum()))
    norms=float(np.linalg.norm(p['direction'])*np.linalg.norm(c['direction']));cos=float(np.dot(p['direction'],c['direction'])/norms) if norms>1e-8 else 1.
    if iou>=PARAM['duplicate_iou'] and cos>=PARAM['duplicate_direction_cos'] and abs(p['reversal']-c['reversal'])<=PARAM['duplicate_reversal_delta']:match=k;break
  if match is None:accepted.append(c)
  else:rejected.append(dict(start=c['start'],end=c['end'],similar_to_end=accepted[match]['end'],reason='similar motion footprint, direction and reversal'))
 # A bounded spatial episode may contain many nearly stationary oscillations.
 # Preserve its first/strongest/last snippets, rather than one output per oscillation.
 episodes=[];scale=max(1,float(sig.obj_area)*96*54/(sig.aw*sig.ah))
 for c in accepted:
  if episodes:
   ep=episodes[-1];scale=max(ep['scale'],c['scale']);union=ep['union']|c['union'];overlap=(ep['union']&c['union']).sum()/max(1,c['union'].sum());novel=(c['union']&~ep['union']).sum()/scale
   if (c['start']-ep['items'][-1]['end'])/fps<=PARAM['episode_gap_sec'] and union.sum()/scale<=PARAM['episode_footprint_object_units'] and (overlap>=PARAM['episode_overlap'] or novel<=PARAM['episode_new_area_object_units']):
    ep['items'].append(c);ep['union']=union;ep['scale']=scale;continue
  episodes.append(dict(span=c['span'],union=c['union'].copy(),items=[c],scale=c['scale']))
 reduced=[]
 for ep in episodes:
  cs=ep['items'];keep={0,len(cs)-1,int(np.argmax([c['quality'] for c in cs]))}
  for j,c in enumerate(cs):
   if j in keep:reduced.append(c)
   else:rejected.append(dict(start=c['start'],end=c['end'],reason='same bounded spatial episode; retain first/strongest/last'))
 accepted=reduced;records=[]
 for i,c in enumerate(accepted):
  records.append(dict(index=i,start=c['start'],end=c['end'],keys=c['keys'],time_start=c['start']/fps,time_end=c['end']/fps,frame_ids=[k*sig.stride for k in c['keys']],compact=c['compact'],reason=c['reason'],mandatory=c['mandatory']))
 return records,dict(candidate_count=len(candidates),selected_count=len(records),rejected=rejected,planning_seconds=time.perf_counter()-t0)
def publish(done,current=''):
 B.write(OUT/'STATUS.json',dict(completed=len(done),total=len(ROWS),current=current,results=done))
 rows=''.join(f'<tr><td>{r["dataset"]}</td><td><a href="{r["id"]}/index.html">{html.escape(r["clip"])}</a></td><td>{r["old"]}</td><td>{r["afterimages"]}</td><td>{r["single_states"]}</td><td>{r["seconds"]:.1f}</td></tr>' for r in done)
 body=f'<p>{len(done)}/{len(ROWS)}편 완료 · 현재 {html.escape(current)}</p><p>통제 실험: 공통 약8fps flow → 객체 영역 제한 → 초 단위 분할 조건 → 좁은 공간의 반복 구간을 처음·중간·마지막으로 축약 → 외관 차이 기반 2–3프레임 → 객체 영역 RMBG 합성. 개발 후보이며 의미 보존을 검증해야 합니다.</p><p>원본 시작·끝 상태 2장을 포함합니다. 한 프레임만 남은 후보도 원본 상태 이미지로 구분합니다. 출력 수 감소는 의미 보존이나 VLM 성능 향상을 뜻하지 않습니다.</p><p>시간은 기존 flow·마스크 캐시를 사용하는 재생성 시간입니다. 최초 생성 속도가 아닙니다. 공통 flow 및 객체 검출 캐시를 재사용합니다. 한 가지 규칙을 적용하되 UT 결과의 표시 크기만 360×288로 맞춥니다.</p><table><tr><th>데이터세트</th><th>영상</th><th>기존 잔상</th><th>새 잔상</th><th>원본 상태</th><th>재생성 초</th></tr>{rows}</table>'
 (OUT/'index.html').write_text(B.page('C39 · 공통 전처리와 공간 구간 압축 후보',body),encoding='utf8')
"""Within-interval appearance diversity control. No semantic GT or per-clip rules."""
def select_appearance(row,m,sig):
 t0=time.perf_counter()
 segments=m['segments']
 wanted=sorted({int(k) for z in segments for k in np.rint(np.linspace(z['start'],z['end'],min(12,z['end']-z['start']+1))).astype(int)})
 cfg=FS.Config(**m['config']);frames,_=B.sparse_grab(row['source'],cfg,wanted,[])
 small={k:cv2.GaussianBlur(cv2.resize(frames[k],(96,72),interpolation=cv2.INTER_AREA),(3,3),0).astype(np.float32) for k in wanted}
 plans=[];audit=[]
 trackpath=ROOT/'연구_공유/20261017_CommonSplit_All/tracks'/(row['input_id']+'_tracks.json')
 tracks=B.read(trackpath)['tracks'] if trackpath.exists() else []
 for z in segments:
  ids=np.unique(np.rint(np.linspace(z['start'],z['end'],min(12,z['end']-z['start']+1))).astype(int)).tolist();d=np.zeros((len(ids),len(ids)),float)
  boxes=[o['xyxy'] for tr in tracks if tr['dynamic'] for o in tr['obs'] if z['start']/sig.fps-.15<=o['time']<=z['end']/sig.fps+.15]
  region=None
  if boxes:
   b=np.array(boxes);lo=b[:,:2].min(0);hi=b[:,2:].max(0);pad=(hi-lo)*.10;lo=np.maximum(0,lo-pad);hi=np.minimum(1,hi+pad);region=[*lo,*hi]
  local={}
  for k in ids:
   im=frames[k];h,w=im.shape[:2]
   if region is not None:
    x1,y1,x2,y2=region;crop=im[max(0,int(y1*h)):min(h,int(y2*h)+1),max(0,int(x1*w)):min(w,int(x2*w)+1)]
   else:crop=im
   local[k]=cv2.GaussianBlur(cv2.resize(crop,(96,72),interpolation=cv2.INTER_AREA),(3,3),0).astype(np.float32)
  for i in range(len(ids)):
   for j in range(i):
    delta=np.mean(np.abs(local[ids[i]]-local[ids[j]]),axis=2);delta=np.maximum(delta-np.median(delta),0)
    # Upper spatial quantile avoids dilution by a large static background.
    values=np.sort(delta.ravel());d[i,j]=d[j,i]=float(np.mean(values[-max(16,len(values)//20):]))
  keep=[len(ids)-1];mx=float(d.max())
  if len(ids)>1:
   j=int(np.argmax(d[:,-1]));keep.append(j if j!=keep[0] else 0)
  if len(ids)>2 and mx>0:
   gain=np.min(d[:,keep],axis=1);j=int(np.argmax(gain))
   if gain[j]>=.30*mx and j not in keep:keep.append(j)
  keys=sorted(set(ids[i] for i in keep));idx=len(plans)
  plans.append(dict(index=idx,start=z['start'],end=z['end'],keys=keys,time_start=z['start']/sig.fps,time_end=z['end']/sig.fps,frame_ids=[k*sig.stride for k in keys],compact=True,reason='common split; appearance-diverse keyframes',mandatory=False))
  audit.append(dict(index=idx,region=region,candidate_frame_ids=[k*sig.stride for k in ids],pairwise_max=mx,selected_frame_ids=[k*sig.stride for k in keys]))
 return plans,dict(candidate_count=len(plans),selected_count=len(plans),rejected=[],planning_seconds=time.perf_counter()-t0,selection_audit=audit,control='C35 common flow; C38 spatial episode subintervals; no historical boundaries; appearance-only selection')


"""C04: fixed coarse intervals; split travel into object-scale short sequences.
Largest coherent flow component is a motion proxy, NOT semantic object tracking.
Compact intervals fall back to C02 appearance diversity to preserve articulation.
"""
GEOMETRY_TRACKS=[]
def select_travel(row,m,sig):
 global GEOMETRY_TRACKS
 t0=time.perf_counter();base,_=select_appearance(row,m,sig)
 path=ROOT/'연구_공유/20261017_CommonSplit_All/tracks'/(row['input_id']+'_tracks.json')
 tracks=B.read(path)['tracks'];GEOMETRY_TRACKS=[t for t in tracks if t['dynamic']];plans=[];audits=[]
 for z in base:
  lo,hi=z['time_start'],z['time_end'];series=[]
  for track in GEOMETRY_TRACKS:
   obs=[o for o in track['obs'] if lo-.1<=o['time']<=hi+.1]
   if len(obs)<2:continue
   boxes=np.array([o['xyxy'] for o in obs])*np.array([sig.W,sig.H,sig.W,sig.H]);centers=(boxes[:,:2]+boxes[:,2:])/2;scales=np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1)
   net=float(np.linalg.norm(centers[-1]-centers[0])/max(.01,np.median(scales)))
   pathlen=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).sum());consistency=float(np.linalg.norm(centers[-1]-centers[0])/max(1e-8,pathlen))
   series.append((net,consistency,track,obs,centers,scales))
  travel=[v for v in series if v[0]>=1.0 and v[1]>=.60]
  if travel:
   _,_,track,obs,centers,scales=max(travel,key=lambda v:v[0]);chosen=[0];previous=0
   for j in range(1,len(obs)):
    gap=float(np.linalg.norm(centers[j]-centers[previous])/max(.01,min(scales[j],scales[previous])))
    if gap>=.50:chosen.append(j);previous=j
   if chosen[-1]!=len(obs)-1:
    final=len(obs)-1;tailgap=float(np.linalg.norm(centers[final]-centers[chosen[-1]])/max(1.,min(scales[final],scales[chosen[-1]])))
    if len(chosen)>1 and tailgap<.25:chosen[-1]=final
    else:chosen.append(final)
   keys=sorted(set(int(np.clip(round(obs[j]['time']*sig.fps),z['start'],z['end'])) for j in chosen))
   # Preserve coarse boundaries and appearance evidence outside reliable tracking.
   first,last=keys[0],keys[-1]
   fallback=[k for k in z['keys'] if k<first or k>last]
   keys=sorted(set(keys+fallback+[z['start'],z['end']]))
   for q in range(0,len(keys)-1,2):
    ks=keys[q:q+3]
    plans.append(dict(index=len(plans),start=ks[0],end=ks[-1],keys=ks,time_start=ks[0]/sig.fps,time_end=ks[-1]/sig.fps,frame_ids=[k*sig.stride for k in ks],compact=False,reason='same tracked object spatial travel budget',mandatory=False))
   audits.append(dict(start=z['start'],end=z['end'],branch='detector geometry travel',track=track['id'],keys=keys))
  else:
   # Compact articulation: keep the existing diverse pose selection and full interval.
   plans.append(dict(z,index=len(plans),reason='appearance within coarse interval; detector mask support'))
   audits.append(dict(start=z['start'],end=z['end'],branch='compact appearance' if series else 'unsupported retained',tracks=[v[2]['id'] for v in series]))
 return plans,dict(candidate_count=len(plans),selected_count=len(plans),rejected=[],planning_seconds=time.perf_counter()-t0,selection_audit=audits)

def main():
 if '--plan-only' in sys.argv:
  for row in ROWS:
   m,sig,_=load(row);plan,audit=select_travel(row,m,sig);B.write(OUT/(row['input_id']+'_PLAN.json'),dict(items=plan,audit=audit));print(row['clip_id'],len(m['segments']),'->',len(plan),'single',sum(len(s['keys'])==1 for s in plan),flush=True)
  return
 locks=[]
 for p in [LOCAL/'gpu0_main.lock',LOCAL/'package/random15_fixed6_laptop_20260918/work/gpu0.lock']:
  f=p.open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);locks.append(f)
 check=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);assert check.returncode==0 and int(check.stdout.strip().split(',')[-1])>5000
 import torch
 assert torch.__version__.startswith('2.6.0') and torch.version.cuda=='12.4' and torch.cuda.device_count()==1
 assert G.sha(G.MODEL/'model.safetensors')==G.WEIGHTS
 G.MK._RUNNER=B.Runner();runner=G.MK._RUNNER
 process_source=(LOCAL/'only_split_ours_20261012/frozen_process_video.py').read_text(encoding='utf8')
 process_source=process_source.replace('if not getattr(sig, "long_video", False) and cfg.mask_backend == "rmbg":', 'override = _mask_override(f,k,sig)\n            if override is not None:\n                m, rc = override\n            elif not getattr(sig, "long_video", False) and cfg.mask_backend == "rmbg":')
 exec(compile(process_source,'c17_mask_override_process.py','exec'),FS.__dict__)
 def mask_override(f,k,sig):
  when=k/sig.fps;h,w=f.shape[:2];boxes=[]
  for tr in GEOMETRY_TRACKS:
   ob=min(tr['obs'],key=lambda o:abs(o['time']-when))
   if abs(ob['time']-when)>.4:continue
   x1,y1,x2,y2=ob['xyxy'];px=(x2-x1)*.2;py=(y2-y1)*.2
   xa,ya=max(0,int((x1-px)*w)),max(0,int((y1-py)*h));xb,yb=min(w,int(np.ceil((x2+px)*w))),min(h,int(np.ceil((y2+py)*h)))
   if xb>xa and yb>ya:boxes.append((xa,ya,xb,yb))
  if not boxes:
   soft=G.MK.rmbg_soft(f)
   return soft,dict(backend='rmbg_soft_missing_box_fallback',core_px=int((soft>=.5).sum()),full_frame_accepted=1,roi_accepted=0)
  mask=np.zeros((h,w),np.float32)
  for xa,ya,xb,yb in boxes:
   soft=G.MK.rmbg_soft(f[ya:yb,xa:xb]);mask[ya:yb,xa:xb]=np.maximum(mask[ya:yb,xa:xb],soft)
  return mask,dict(backend='rmbg_soft_dynamic_boxes',boxes=boxes,core_px=int((mask>=.5).sum()),full_frame_accepted=0,roi_accepted=len(boxes))
 FS._mask_override=mask_override
 FS.grab_frames=B.sparse_grab;FS.plot_signals=lambda *a,**k:None
 def compose(fr,ms,cfg,times=None):
  base=fr[-1].astype(np.float32).copy();alphas=[.45] if len(fr)==2 else [.28,.58]
  for f,m,a in zip(fr[:-1],ms[:-1],alphas):
   al=m[...,None]*a;base=base*(1-al)+f.astype(np.float32)*al
  if len(fr)>1:
   al=ms[-1][...,None];base=base*(1-al)+fr[-1].astype(np.float32)*al
  return np.clip(base,0,255).astype(np.uint8)
 FS.compose_ref=compose;done=[]
 for row in ROWS:
  saved=OUT/row['input_id']/'RECEIPT.json'
  if saved.exists():
   receipt=B.read(saved);done.append(receipt['summary']);publish(done);continue
  publish(done,row['clip_id']);t0=time.perf_counter();m,sig,cache=load(row);plan,audit=select_travel(row,m,sig);key=row['input_id'];dest=OUT/key;dest.mkdir(exist_ok=True);work=WORK/key
  previous=ROOT/'연구_공유/20261016_Overnight_Research/C39_common_pipeline'/key
  prevplan=B.read(previous/'SELECTION.json')['items'] if (previous/'SELECTION.json').exists() else None
  signature=lambda xs:[(z['start'],z['end'],z['keys']) for z in xs]
  if prevplan is not None and signature(plan)==signature(prevplan) and B.read(previous/'RECEIPT.json')['source_sha256']==row['source_sha256']:
   shutil.copytree(previous,dest,dirs_exist_ok=True);receipt=B.read(dest/'RECEIPT.json');rec=dict(receipt['summary'],seconds=0.,reused_from=str(previous),reuse_planning_seconds=time.perf_counter()-t0);receipt['summary']=rec;B.write(dest/'RECEIPT.json',receipt);done.append(rec);publish(done);continue
  B.write(dest/'SELECTION.json',dict(items=plan,audit=audit))
  segs=[FS.Segment(idx=i,start=z['start'],end=z['end'],reason_end=z['reason'],poses_est=0,obj_area=float(sig.obj_area),evidence='strong',t_start=z['time_start'],t_end=(z['end']+1)/sig.fps) for i,z in enumerate(plan)]
  FS.analyze=lambda *a,**k:sig;FS.split_segments=lambda *a,**k:copy.deepcopy(segs);FS.select_keyframes=lambda sig,s,cfg:list(plan[s.idx]['keys'])
  bg=np.zeros((sig.H,sig.W,3),np.uint8);G.MK.flow_aware_background=lambda *a,**k:bg.copy()
  cfg=FS.Config(**m['config']);cfg.draw_time_bar=False;before_calls=runner.calls;before_hits=runner.hits
  result=FS.process_video(row['source'],str(work),cfg,lambda x:print(row['clip_id'],x,flush=True))
  items=[];cap=cv2.VideoCapture(row['source']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
  def output(im,name):
   if row['dataset']=='ut_interaction':im=cv2.resize(im,(360,288),interpolation=cv2.INTER_AREA)
   B.save(dest/name,im)
  for name,frame in [('state_start.png',0),('state_end.png',n-1)]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok;output(FS._resize_frame(im,cfg),name)
  cap.release()
  for s,z in zip(result['segments'],plan):
   name=f'afterimage_{s["idx"]+1:03d}.png';output(B.image(work/s['image']),name);sheet=f'frames_{s["idx"]+1:03d}.jpg';shutil.copy2(work/s['sheet'],dest/sheet)
   items.append(dict(z,image=name,sheet=sheet,mask_px=s['mask_px'],sha256=G.sha(dest/name),type='afterimage' if len(z['keys'])>1 else 'original_state'))
  elapsed=time.perf_counter()-t0
  rec=dict(id=key,clip=row['clip_id'],dataset=row['dataset'],old=len(m['segments']),afterimages=sum(x['type']=='afterimage' for x in items),single_states=2+sum(x['type']=='original_state' for x in items),seconds=elapsed,mask_calls=runner.calls-before_calls,mask_cache_hits=runner.hits-before_hits)
  B.write(dest/'RECEIPT.json',dict(summary=rec,source_sha256=G.sha(row['source']),flow_cache_sha256=G.sha(cache),items=items,full_generation=result))
  done.append(rec);publish(done)
 B.write(OUT/'COMPLETE.json',dict(results=done,code_sha256=G.sha(__file__),gpu=check.stdout,torch=torch.__version__,cuda=torch.version.cuda))
def render(row,m,items,rec,audit,n,fps):
 m=B.read(row['metadata'])
 dest=OUT/row['input_id'];video=ROOT/'연구_공유/20260928_Dataset_Native/videos'/row['input_id']/'source.mp4'
 if not video.exists():video=Path(row['source'])
 overview='<div class="gallery"><figure><img src="state_start.png"><figcaption>시작 원본</figcaption></figure>'+''.join(f'<a href="#s{i}"><img loading="lazy" src="{x["image"]}">{i+1} · {x["time_start"]:.1f}–{x["time_end"]:.1f}초</a>' for i,x in enumerate(items))+'<figure><img src="state_end.png"><figcaption>마지막 원본</figcaption></figure></div>'
 body=f'<a href="../index.html">전체 목록</a><p>기존 {rec["old"]}장 → 새 잔상 {rec["afterimages"]}장 + 원본 상태 {rec["single_states"]}장. 후보 {audit["candidate_count"]}개 중 {audit["selected_count"]}개 선택.</p><video id="v" controls preload="metadata" src="{rel(video,dest)}"></video><p>입력 분석 기록을 재사용한 재생성: {rec["seconds"]:.1f}초. 프레임 선택: {audit["planning_seconds"]:.3f}초. 새 마스크 추론 {rec["mask_calls"]}회, 캐시 적중 {rec["mask_cache_hits"]}회.</p><h2>새 결과 전체 순서</h2>'+overview
 for i,x in enumerate(items):
  related=[s for s in m['segments'] if s['t_start']<=x['time_end'] and s['t_end']>=x['time_start']]
  old=''.join(f'<figure><img loading="lazy" src="{rel(Path(row["metadata"]).parent/s["image"],dest)}"><figcaption>기존 {s["idx"]+1} · {s["t_start"]:.1f}–{s["t_end"]:.1f}초</figcaption></figure>' for s in related)
  if not old:old='<p>이 시간에 대응하는 기존 출력 없음</p>'
  typ='잔상' if x['type']=='afterimage' else '원본 상태(과거 변화 중첩을 생략)'
  body+=f'<section class="card" id="s{i}"><h2>{i+1} · {x["time_start"]:.2f}–{x["time_end"]:.2f}초 · {typ}</h2><button onclick="document.getElementById(\'v\').currentTime={x["time_start"]};document.getElementById(\'v\').scrollIntoView()">해당 원본 시점 보기</button><div class="pair"><div><h3>기존 구간</h3>{old}</div><div><h3>수정 후보</h3><a href="{x["image"]}" target="_blank"><img loading="lazy" src="{x["image"]}"></a><p>선택 frame ID: {x["frame_ids"]}<br>움직임 분류: {"같은 영역의 짧은 변화" if x["compact"] else "공간 이동 범위 제한"}<br>마스크 픽셀: {x["mask_px"]}</p><details><summary>실제 선택 프레임·마스크</summary><img loading="lazy" src="{x["sheet"]}"></details></div></div></section>'
 body+='<h2>해석할 때 주의할 점</h2><p>구간 전체를 균일하게 요약한 것이 아니라, 자동 선택한 짧은 변화의 시퀀스입니다. 중간에 선택하지 않은 시간대가 있습니다. flow peak는 행동의 의미나 자세 극점을 직접 인식한 결과가 아닙니다. 동일 장소의 반복을 억제하면서 중요한 차이가 빠질 가능성이 있으며, VLM 평가 전의 개발 후보입니다.</p>'
 page=B.page(row['clip_id'],body).replace('</style>','.gallery{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.gallery figure{margin:0}.gallery a{font-size:14px}button{padding:9px;margin:8px}@media(max-width:800px){.pair{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(2,1fr)}} </style>')
 (dest/'index.html').write_text(page,encoding='utf8')
if __name__=='__main__':main()



