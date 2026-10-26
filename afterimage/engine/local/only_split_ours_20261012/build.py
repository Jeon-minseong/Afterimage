import os,sys,json,time,hashlib,inspect,copy,subprocess,msvcrt,traceback,html,types
from pathlib import Path
import numpy as np
import cv2
def unicode_imwrite(filename,img,params=None):
 p=Path(filename);p.parent.mkdir(parents=True,exist_ok=True)
 ok,encoded=cv2.imencode(p.suffix,img,params or [])
 if ok:encoded.tofile(str(p))
 return bool(ok)
cv2.imwrite=unicode_imwrite
R=Path(__file__).resolve().parent
ROOT=Path(os.environ['AF_METHOD_ROOT'])
OLD=R.parent/'main_experiment_native_20260928'
sys.path.insert(0,str(ROOT/'lib'))
import flowsplit as FS
import masks as MK
from rmbg_v2 import RMBG2Runner
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
G=types.SimpleNamespace(fs=FS,MK=MK,RMBG2Runner=RMBG2Runner,MODEL=Path(os.environ['AF_MODELS_ROOT'])/'rmbg2',WEIGHTS='566ed80c3d95f87ada6864d4cbe2290a1c5eb1c7bb0b123e984f60f76b02c3a7',sha=sha)
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf8')
def image(p):return cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_COLOR)
def save(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);cv2.imencode(p.suffix,a)[1].tofile(str(p))
def page(title,body):return '<!doctype html><meta charset="utf-8"><h1>'+html.escape(title)+'</h1>'+body
class Runner:
 def __init__(self):self.runner=None;self.calls=0;self.hits=0;self.seconds=0
 def predict(self,rgb):
  key=hashlib.sha256(rgb.tobytes()+str(rgb.shape).encode()+G.WEIGHTS.encode()+b'v3-normalized-fp32').hexdigest()
  p=R/'mask_cache'/(key+'.npy')
  for root in (R,OLD,R.parent/'main_experiment_20260927'):
   q=root/'mask_cache'/(key+'.npy')
   if q.exists():
    try:a=np.load(q,allow_pickle=False)
    except (ValueError,EOFError,OSError):
     if root!=R:raise
     continue
    assert a.shape==rgb.shape[:2] and np.isfinite(a).all()
    self.hits+=1;return a
  if self.runner is None:self.runner=G.RMBG2Runner(device='cuda:0',model_id=str(G.MODEL))
  t=time.time();a=self.runner.predict(rgb);self.calls+=1;self.seconds+=time.time()-t
  assert np.isfinite(a).all() and a.shape==rgb.shape[:2]
  p.parent.mkdir(exist_ok=True);tmp=p.with_suffix('.tmp.npy');np.save(tmp,a);os.replace(tmp,p);return a
def sparse_grab(path,cfg,wanted,bg_idx):
 # Decode only requested frames; never scan beyond the last requested source index.
 wanted=set(wanted);out={};cap=cv2.VideoCapture(path);assert cap.isOpened(),path
 if not wanted:cap.release();return out,[]
 by_source={int(k)*cfg.analysis_stride:int(k) for k in wanted}
 try:
  for src in range(max(by_source)+1):
   assert cap.grab(),f'Video ended before requested frame {src}'
   if src in by_source:
    ok,bgr=cap.retrieve();assert ok
    out[by_source[src]]=FS._resize_frame(bgr,cfg)
 finally:cap.release()
 assert set(out)==wanted
 return out,[]

