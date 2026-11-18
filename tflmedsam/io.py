import csv
import hashlib
import json
import math
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path,value):
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v): return None
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        return v
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write_csv(path,rows,fields=None):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def safe_id(value):
    if not value or value in ('.','..') or any(x in value for x in '/\\:'):
        raise ValueError(f'Unsafe image identifier: {value!r}')
    return value
