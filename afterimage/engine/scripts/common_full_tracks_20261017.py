import os
"""Deterministic short-gap association for geometry, not ground-truth identities."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(os.environ['AF_RUN_ROOT']);WORK=(Path(os.environ['AF_RUN_ROOT'])/'local/overnight_20261016')
OUT=ROOT/'연구_공유/20261017_CommonSplit_All/tracks';OUT.mkdir(exist_ok=True)
def geom(box):
 b=np.asarray(box,float);return (b[:2]+b[2:])/2,b[2:]-b[:2]
def iou(a,b):
 a=np.asarray(a);b=np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();return inter/max(1e-9,(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter)
def process(p):
 data=json.loads(p.read_text(encoding='utf8'));tracks=[]
 for rec in data['records']:
  active=[t for t in tracks if rec['time']-t['obs'][-1]['time']<=1.0];boxes=rec['boxes'];cost=np.full((len(active),len(boxes)),1e6)
  for i,t in enumerate(active):
   old=t['obs'][-1];c,s=geom(old['xyxy'])
   for j,b in enumerate(boxes):
    if b['cls']!=t['cls']:continue
    c2,s2=geom(b['xyxy']);distance=np.linalg.norm(c-c2)/max(.01,(np.linalg.norm(s)+np.linalg.norm(s2))/2)
    if distance<=1.:cost[i,j]=1-iou(old['xyxy'],b['xyxy'])+.5*distance
  assigned=set()
  if cost.size:
   ii,jj=linear_sum_assignment(cost)
   for i,j in zip(ii,jj):
    if cost[i,j]>=1e5:continue
    active[i]['obs'].append(dict(boxes[j],frame=rec['frame'],time=rec['time']));assigned.add(int(j))
  for j,b in enumerate(boxes):
   if j not in assigned:tracks.append(dict(id=len(tracks),cls=b['cls'],obs=[dict(b,frame=rec['frame'],time=rec['time'])]))
 for t in tracks:
  centers=np.array([geom(z['xyxy'])[0] for z in t['obs']]);sizes=np.array([geom(z['xyxy'])[1] for z in t['obs']]);scale=max(.01,float(np.median(np.linalg.norm(sizes,axis=1))))
  displacement=float(np.linalg.norm(np.percentile(centers,95,axis=0)-np.percentile(centers,5,axis=0))/scale)
  sizechange=float(np.max((np.percentile(sizes,95,axis=0)-np.percentile(sizes,5,axis=0))/np.maximum(np.median(sizes,axis=0),.01)))
  motion=np.array([z['flow_fraction'] for z in t['obs']]);duration=t['obs'][-1]['time']-t['obs'][0]['time']
  reliable=len(t['obs'])>=3 and duration>=.4
  moving=displacement>=.35 or sizechange>=.30 or int((motion>.10).sum())>=3
  t.update(reliable=bool(reliable),dynamic=bool(reliable and moving),displacement=displacement,size_change=sizechange,motion_p90=float(np.percentile(motion,90)),duration=duration)
 result=dict(summary=data['summary'],tracks=tracks,settings=dict(max_gap=1.,max_center_distance=1.,min_observations=3,min_duration=.4,displacement=.35,size_change=.30,motion_fraction=.1,motion_count=3),limitations='Association may swap identities or fragment; dynamic is a geometric heuristic, not action recognition.')
 (OUT/(p.stem+'_tracks.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(data['summary']['clip'],len(tracks),'tracks',sum(t['dynamic'] for t in tracks),'dynamic',flush=True)
 return result
if __name__=='__main__':
 results=[process(p) for p in sorted(((Path(os.environ['AF_RUN_ROOT'])/'local/common_split_all_20261017/detector_cache')).glob('*.json'))]
 (OUT/'TRACK_SUMMARY.json').write_text(json.dumps([dict(r['summary'],tracks=len(r['tracks']),dynamic=sum(t['dynamic'] for t in r['tracks'])) for r in results],ensure_ascii=False,indent=2),encoding='utf8')
