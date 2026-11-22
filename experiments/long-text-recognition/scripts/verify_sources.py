from __future__ import annotations
import sys,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'src'/'SOURCE_SHA256.json').read_text())
for filename,expected in sorted(manifest.items()):
    p=root/'src'/filename
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=expected:raise AssertionError(f'Source changed: {filename}; expected {expected}, actual {got}')
    print('PASS',filename,got[:16])
print('SOURCE HASH CHECK PASS:',len(manifest),'modules')
