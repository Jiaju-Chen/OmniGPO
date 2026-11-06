"""Conservative release audit: source/config/docs only, no private lab assets."""
import ast
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
ignored={'.git','__pycache__','.pytest_cache','build','dist','.venv'}
forbidden_suffixes={'.pt','.pth','.bin','.safetensors','.parquet','.npz','.pem','.key','.log'}
patterns=[r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
    r'\bgh[pousr]_[A-Za-z0-9]{25,}\b',r'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
    r'/Users/MacBook/',r'/home/dataset-local/',r'/root/vepfs-data/',
    r'(?:61\.172\.170\.106|115\.190\.90\.101)',
    r'(?i)(?:api_key|password|access_token)\s*[:=]\s*[\"\x27][A-Za-z0-9_\-]{20,}']
files=[]
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT)
    if any(x in ignored or x.endswith('.egg-info') for x in rel.parts):continue
    assert not p.is_symlink(),f'Symlink not allowed: {rel}'
    if not p.is_file():continue
    assert p.suffix not in forbidden_suffixes and not p.name.startswith('.env'),str(rel)
    assert p.stat().st_size<1024**2,f'Unexpected large file: {rel}'
    text=p.read_text(encoding='utf-8')
    if p.name!='validate_release.py':
        assert not any(re.search(pattern,text) for pattern in patterns),f'Private material check failed: {rel}'
    if p.suffix=='.py':ast.parse(text,filename=str(rel))
    files.append(str(rel))
print(json.dumps(dict(files=len(files),bytes=sum((ROOT/p).stat().st_size for p in files),
    source_only=True,no_private_assets_detected=True,paths=sorted(files)),indent=2))
