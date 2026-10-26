"""Verify distributed source bytes without credentials, a GPU or network access."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
for relative, expected in manifest['files'].items():
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Path outside repository')
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError('Missing or modified release file: ' + relative)
print(f"Verified {len(manifest['files'])} release files; no API calls.")
