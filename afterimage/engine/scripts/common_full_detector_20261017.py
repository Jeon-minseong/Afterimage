import os
"""Low-rate detector geometry + cached flow support, GPU0 sequential only."""
import os,sys,time,json,pickle,hashlib,msvcrt,subprocess,traceback
from pathlib import Path
LOCAL=(Path(os.environ['AF_RUN_ROOT'])/'local');WORK=LOCAL/'overnight_20261016'
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(WORK/'yolo_settings'),YOLO_VERBOSE='false',OMP_NUM_THREADS='1')
sys.path.insert(0,str(Path(os.environ['AF_METHOD_ROOT'])/'vendor'));sys.path.insert(0,str(LOCAL/'only_split_ours_20261012'))
import build as B
import cv2,numpy as np,torch
from ultralytics import YOLO
ROOT=Path(os.environ['AF_RUN_ROOT']);OUT=ROOT/'연구_공유/20261017_CommonSplit_All/detection';OUT.mkdir(parents=True,exist_ok=True)
CACHE=LOCAL/'common_split_all_20261017/detector_cache';CACHE.mkdir(exist_ok=True)
rows=B.read(ROOT/'연구_공유/20261017_CommonSplit_All/INPUTS.json')
weights=Path(os.environ['AF_MODELS_ROOT'])/'yolo11n.pt';weights_hash=B.G.sha(weights)
locks=[]
for p in [LOCAL/'gpu0_main.lock',LOCAL/'package/random15_fixed6_laptop_20260918/work/gpu0.lock']:
 f=p.open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);locks.append(f)
check=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);assert check.returncode==0
assert torch.__version__.startswith('2.6.0') and torch.version.cuda=='12.4' and torch.cuda.device_count()==1
model=YOLO(str(weights));done=[]
for row in rows:
 key=row['input_id'];target=CACHE/(key+'.json')
 if target.exists():
  old=B.read(target)
  if old.get('source_sha256')==row['source_sha256'] and old.get('weights_sha256')==weights_hash and old.get('complete'):
   done.append(old['summary']);B.write(OUT/'STATUS.json',dict(completed=len(done),total=len(rows),current='cache reuse',results=done));continue
 B.write(OUT/'STATUS.json',dict(completed=len(done),total=len(rows),current=row['clip_id'],results=done))
 assert B.G.sha(row['source'])==row['source_sha256']
 flowpath=B.OLD/'native_signal_cache'/(key+'.pkl')
 if not flowpath.exists():flowpath=B.OLD/'flow_cache'/(key+'.pkl')
 sig=pickle.load(flowpath.open('rb'))['sig'];cap=cv2.VideoCapture(row['source']);fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=cap.get(cv2.CAP_PROP_FRAME_WIDTH);h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
 step=max(1,round(fps/4));records=[];infer=0.;t0=time.perf_counter()
 for i in range(n):
  ok=cap.grab()
  if not ok:break
  if i%step and i!=n-1:continue
  ok,im=cap.retrieve();assert ok
  torch.cuda.synchronize();ti=time.perf_counter();pred=model.predict(im,imgsz=640,conf=.35,iou=.5,device=0,half=False,verbose=False)[0];torch.cuda.synchronize();infer+=time.perf_counter()-ti
  fi=min(sig.n-1,round(i/sig.stride));flow=B.FS.unpack(sig,fi);boxes=[]
  for x1,y1,x2,y2,conf,cls in pred.boxes.data.cpu().numpy().tolist():
   xa,xb=sorted([int(np.clip(x1/w*sig.aw,0,sig.aw-1)),int(np.clip(np.ceil(x2/w*sig.aw),1,sig.aw))]);ya,yb=sorted([int(np.clip(y1/h*sig.ah,0,sig.ah-1)),int(np.clip(np.ceil(y2/h*sig.ah),1,sig.ah))])
   moving=float(flow[ya:yb,xa:xb].mean()) if xb>xa and yb>ya else 0
   boxes.append(dict(xyxy=[x1/w,y1/h,x2/w,y2/h],confidence=conf,cls=int(cls),flow_fraction=moving))
  records.append(dict(frame=i,time=i/fps,boxes=boxes))
 cap.release();summary=dict(id=key,clip=row['clip_id'],dataset=row['dataset'],frames=len(records),seconds=time.perf_counter()-t0,inference_seconds=infer,detected_frames=sum(bool(z['boxes']) for z in records))
 B.write(target,dict(complete=True,summary=summary,source_sha256=row['source_sha256'],weights_sha256=weights_hash,sampling_fps=fps/step,source_fps=fps,source_size=[w,h],flow_cache_sha256=B.G.sha(flowpath),records=records))
 done.append(summary);B.write(OUT/'STATUS.json',dict(completed=len(done),total=len(rows),current='',results=done));print(row['clip_id'],len(records),'frames',round(summary['seconds'],1),'sec',flush=True)
B.write(OUT/'COMPLETE.json',dict(results=done,model='YOLO11n',torch=torch.__version__,cuda=torch.version.cuda,weights_sha256=weights_hash,scope='Detection cache only. Not generated afterimages or VLM results.'))
