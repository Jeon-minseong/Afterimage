import hashlib,json,math
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent.parent
CONDITIONS=('uniform_observations','selected_sequence','afterimage','uniform_budget')
def check(ok,message):
    if not ok: raise ValueError(message)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(obj): return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def digest(obj):return hashlib.sha256(canonical(obj)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.tmp');t.write_bytes(canonical(obj)+b'\n');t.replace(path)
def safe(root,relative):
    r=Path(relative);p=(Path(root)/r).resolve()
    check(not r.is_absolute() and '..' not in r.parts and ':' not in str(relative) and p.is_relative_to(Path(root).resolve()),'Unsafe relative path')
    return p
def fresh(path):
    path=Path(path).resolve();check(not path.exists(),'Output exists; choose a new run directory');path.mkdir(parents=True);return path
def pricing_cost(provider,inputs,outputs,rates,cache_write=0,cache_read=0):
    check(inputs>=cache_write+cache_read>=0,'Invalid cache usage')
    long=provider=='openai' and inputs>272000
    return ((inputs-cache_write-cache_read)*rates['input']+cache_write*rates['cache_write']+cache_read*rates['cache_read'])*(2 if long else 1)/1e6+outputs*rates['output']*(1.5 if long else 1)/1e6
def validate_inputs(folder):
    folder=Path(folder);m=read(folder/'manifest.json');ids=set();checked={}
    for v in m['videos']:
        check(v['video_id'] not in ids,'Duplicate video ID');ids.add(v['video_id'])
        check(set(v['conditions'])==set(CONDITIONS),'Unexpected conditions')
        endpoints=None
        for c,images in v['conditions'].items():
            check(len(images)==(v['M'] if c in CONDITIONS[:2] else v['J'])+2,'M/J count differs')
            ep=[images[0]['sha256'],images[-1]['sha256']]
            if endpoints is None:endpoints=ep
            check(ep==endpoints,'Endpoint mismatch')
            for i in images:
                if i['path'] not in checked:checked[i['path']]=sha(safe(folder,i['path']))
                check(checked[i['path']]==i['sha256'],'Input hash mismatch')
    return m,{'videos':len(ids),'unique_images':len(checked),'image_appearances':sum(len(x) for v in m['videos'] for x in v['conditions'].values()),'status':'passed'}
