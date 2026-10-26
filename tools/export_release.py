"""Build a source-only archive with an explicit allowlist; never scan parent project secrets."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
top={'README.md','README.ko.md','THIRD_PARTY_NOTICES.md','.gitignore','.gitattributes','requirements-api.txt','requirements-generation.txt'}
folders={'afterimage','configs','gt','prompts','tests','docs','validation','tools','assets'}
files=[]
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT)
    if not p.is_file() or '__pycache__' in rel.parts or p.suffix=='.pyc':continue
    if len(rel.parts)==1 and p.name not in top:continue
    if len(rel.parts)>1 and rel.parts[0] not in folders:continue
    if any(x.lower().startswith(('.env','api_keys')) for x in rel.parts):raise ValueError('Secret file name in source allowlist')
    if p.suffix.lower() in ('.pt','.safetensors','.mp4','.avi','.pkl','.npy') and rel.as_posix() != 'assets/ut_push_demo_v1.mp4':raise ValueError('Non-source research asset in release')
    data=p.read_bytes()
    if re.search(rb'(?:sk-proj-|sk-ant-api)[A-Za-z0-9_-]{20,}',data):raise ValueError('Potential credential pattern in source')
    files.append(p)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=ROOT/'dist/afterimage_github_v1_source.zip')
target=parser.parse_args().out;target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():raise ValueError('Archive exists; preserve it and choose a new release version')
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
release=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
if release['files']!=manifest:raise ValueError('Release manifest is stale; review and refresh it before packaging')
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    z.write(ROOT/'RELEASE_MANIFEST.json','RELEASE_MANIFEST.json')
print(json.dumps({'files':len(files),'zip_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}))
