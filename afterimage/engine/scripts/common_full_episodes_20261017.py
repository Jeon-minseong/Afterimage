import os
"""Bounded spatial episodes; retain temporal representatives, never semantic labels."""
from pathlib import Path
import sys,json,numpy as np
R=Path(os.environ['AF_RUN_ROOT']);L=(Path(os.environ['AF_RUN_ROOT'])/'local');sys.path.insert(0,str(L/'only_split_ours_20261012'));import build as B
S=R/'연구_공유/20261017_CommonSplit_All';O=S/'episodes';O.mkdir(exist_ok=True);rows=B.read(S/'flow/INPUTS.json');B.write(O/'INPUTS.json',rows)
def footprint(tracks,a,b):
 ids=[];spread=0.
 for tr in tracks:
  if not tr['dynamic']:continue
  obs=[o for o in tr['obs'] if a<=o['time']<=b]
  if len(obs)<2:continue
  x=np.array([o['xyxy'] for o in obs]);center=(x[:,:2]+x[:,2:])/2;scale=np.median(np.linalg.norm(x[:,2:]-x[:,:2],axis=1));diam=np.linalg.norm(np.percentile(center,95,axis=0)-np.percentile(center,5,axis=0))/max(scale,1e-6);spread=max(spread,diam);ids.append(tr['id'])
 return set(ids),float(spread)
def continuous(tracks,ids,a,b):
 if b-a<=2:return True
 for tr in tracks:
  if tr['id'] not in ids:continue
  ts=[o['time'] for o in tr['obs'] if a-.3<=o['time']<=b+.3]
  if not ts or ts[0]>a+.4 or ts[-1]<b-.4 or max(np.diff(ts),default=0)>1.:return False
 return bool(ids)

def travelling(tracks,a,b):
 votes=[]
 for tr in tracks:
  if not tr['dynamic']:continue
  obs=[o for o in tr['obs'] if a<=o['time']<=b]
  if len(obs)<3:continue
  x=np.array([o['xyxy'] for o in obs]);c=(x[:,:2]+x[:,2:])/2;net=np.linalg.norm(c[-1]-c[0]);path=np.linalg.norm(np.diff(c,axis=0),axis=1).sum();scale=np.median(np.linalg.norm(x[:,2:]-x[:,:2],axis=1));votes.append(net/max(scale,1e-6)>.20 and net/max(path,1e-6)>.65)
 return bool(votes) and all(votes)

results=[]
for row in rows:
 key=row['input_id'];raw=B.read(S/'flow'/(key+'.json'));segs=raw['segments'];tracks=B.read(S/'tracks'/(key+'_tracks.json'))['tracks'];eps=[]
 for z in segs:
  if eps:
   ep=eps[-1];ids,diam=footprint(tracks,ep[0]['t_start'],z['t_end']);oldids,_=footprint(tracks,ep[0]['t_start'],ep[-1]['t_end']);newids,_=footprint(tracks,z['t_start'],z['t_end'])
   # Geometry anchored to the whole episode prevents small steps accumulating unnoticed.
   if ids and oldids==newids and diam<=.5 and continuous(tracks,ids,ep[-1]['t_end'],z['t_start']):
    ep.append(z);continue
  eps.append([z])
 selected=[];audit=[]
 for ep in eps:
  if len(ep)>1 and travelling(tracks,ep[0]['t_start'],ep[-1]['t_end']):
   z=dict(ep[0],end=ep[-1]['end'],t_end=ep[-1]['t_end'],reason_end='continuous_travel_merged',original_idx=ep[0]['idx']);selected.append(z);audit.append(dict(start=z['t_start'],end=z['t_end'],source_segments=[v['idx'] for v in ep],merged=True));continue
  if len(ep)<=3:chosen=list(range(len(ep)))
  else:
   mid=(ep[0]['t_start']+ep[-1]['t_end'])/2;chosen=sorted(set([0,min(range(1,len(ep)-1),key=lambda i:abs((ep[i]['t_start']+ep[i]['t_end'])/2-mid)),len(ep)-1]))
  audit.append(dict(start=ep[0]['t_start'],end=ep[-1]['t_end'],source_segments=[z['idx'] for z in ep],retained=[ep[i]['idx'] for i in chosen]))
  selected.extend(dict(ep[i],original_idx=ep[i]['idx']) for i in chosen)
 for i,z in enumerate(selected):z['idx']=i
 B.write(O/(key+'.json'),dict(segments=selected,episodes=audit,config=raw['config'],limitation='Repeated similar spatial episodes compressed to first/middle/last source subintervals. Duration and repetition count are not encoded. Geometry may miss pose-only semantic changes; must visually inspect.'))
 info=dict(id=key,clip=row['clip_id'],old=len(segs),new=len(selected),episodes=len(eps));results.append(info);print(row['clip_id'],len(segs),'->',len(selected),flush=True)
B.write(O/'COMPLETE.json',results)
