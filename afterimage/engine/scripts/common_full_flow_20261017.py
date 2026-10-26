import os
from pathlib import Path
import json,pickle,sys,time,dataclasses,numpy as np,cv2,os
os.environ['OMP_NUM_THREADS']='1'
R=Path(os.environ['AF_RUN_ROOT']);L=(Path(os.environ['AF_RUN_ROOT'])/'local');sys.path.insert(0,str(L/'only_split_ours_20261012'));import build as B
S=R/'연구_공유/20261017_CommonSplit_All';O=S/'flow';O.mkdir(exist_ok=True);C=L/'common_split_all_20261017/flow';C.mkdir(exist_ok=True)
rows=B.read(S/'INPUTS.json');B.write(O/'INPUTS.json',rows)
import inspect
source=inspect.getsource(B.FS.split_segments)
source=source.replace('        start = a\n','        start = a\n        armed=True; quiet=0\n')
source=source.replace('            novel += float((m & ~painted).sum())', "            raw_flag=flag\n            if raw_flag in ('reversal','reentry'):\n                quiet=0\n                if not armed:flag=None\n            else:\n                quiet+=1\n                if quiet>=cfg.split_sustain:armed=True\n            novel += float((m & ~painted).sum())")
source=source.replace('            if reason is not None and t < b:', "            if reason is not None and t < b:\n                if reason in ('reversal','reentry'):armed=False;quiet=0")
exec(compile(source,'edge_trigger_split','exec'),B.FS.__dict__)

results=[]
for row in rows:
 key=row['input_id']
 if (O/(key+'.json')).exists() and (C/(key+'.pkl')).exists():
  prior=B.read(O/(key+'.json'));results.append(prior['summary']);continue
 cap=cv2.VideoCapture(row['source']);fps=cap.get(cv2.CAP_PROP_FPS);cap.release();cfg=B.FS.Config();cfg.analysis_stride=max(1,round(fps/8));cfg.long_video_sec=1e9;cfg.chronic_min_sec=1e9;cfg.max_seg_sec=1e9;cfg.resize=None;cfg.rev_hold=max(1,round(.12*fps/cfg.analysis_stride))
 B.write(O/'STATUS.json',dict(completed=len(results),total=len(rows),current=row['clip_id'],results=results));t0=time.perf_counter();original_refine=B.FS.refine;B.FS.refine=lambda *a,**k:None
 try:sig=B.FS.analyze(row['source'],cfg,lambda s:print(s,flush=True))
 finally:B.FS.refine=original_refine
 rawsec=time.perf_counter()-t0
 det=B.read(L/'common_split_all_20261017/detector_cache'/(key+'.json'));dets=det['records'];times=np.array([r['time'] for r in dets]);gated=0
 for t in range(sig.n):
  sec=t/sig.fps;j=int(np.argmin(abs(times-sec)));boxes=dets[j]['boxes'] if abs(times[j]-sec)<=.3 else []
  # Empty detection support is tested as zero motion, not full-frame fallback.
  support=np.zeros((sig.ah,sig.aw),bool)
  for b in boxes:
   x1,y1,x2,y2=b['xyxy'];px=(x2-x1)*.15;py=(y2-y1)*.15;x1=max(0,int((x1-px)*sig.aw));x2=min(sig.aw,int(np.ceil((x2+px)*sig.aw)));y1=max(0,int((y1-py)*sig.ah));y2=min(sig.ah,int(np.ceil((y2+py)*sig.ah)));support[y1:y2,x1:x2]=True
  m=B.FS.unpack(sig,t)&support;sig.moving_packed[t]=np.packbits(m,axis=1);gated+=1
 B.FS.refine(sig,cfg,lambda s:print(s,flush=True));sig.long_video=True;cfg.split_sustain=max(1,round(.08*sig.fps));active=np.asarray(getattr(sig,'motion_area_sus',sig.motion_area))>cfg.active_area_frac;segs=B.FS.split_segments(sig,cfg) if active.any() else []
 with (C/(key+'.pkl')).open('wb') as f:pickle.dump(dict(sig=sig,cfg=dataclasses.asdict(cfg),source_sha256=row['source_sha256'],note='raw flux/globalmean vectors not re-estimated after ROI mask gate; NOT used in current split/appearance selection'),f)
 info=dict(id=key,clip=row['clip_id'],dataset=row['dataset'],segments=len(segs),flow_seconds=rawsec,total_seconds=time.perf_counter()-t0,analysis_fps=sig.fps,gated_frames=gated,reasons={s.reason_end:sum(z.reason_end==s.reason_end for z in segs) for s in segs})
 B.write(O/(key+'.json'),dict(summary=info,segments=[dataclasses.asdict(s) for s in segs],config=dataclasses.asdict(cfg),cache=str(C/(key+'.pkl'))));results.append(info);print(row['clip_id'],len(segs),round(info['total_seconds'],2),flush=True)
B.write(O/'COMPLETE.json',results);B.write(O/'STATUS.json',dict(completed=len(results),total=len(rows),results=results))

