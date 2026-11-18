"""Manifest validation and deterministic image/mask preprocessing."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from . import CLASSES
from .io import read_csv,safe_id,digest,write_csv,write_json

def resolve(manifest,value):
    p=Path(value)
    return p.resolve() if p.is_absolute() else (Path(manifest).resolve().parent/p).resolve()

def load_mask(path,size):
    with Image.open(path) as im:
        a=np.asarray(im)
        if a.ndim!=2: raise ValueError(f'Expected a single-channel lesion mask: {path}')
        binary=Image.fromarray(np.uint8(a>0)*255)
        return np.asarray(binary.resize((size,size),Image.Resampling.NEAREST))>0

def audit(manifest,out,size=1024,expected=None,allow_constructed=False):
    rows=read_csv(manifest)
    if not rows: raise ValueError('Manifest is empty')
    required={'dataset','image_id','split','image_path',*(c+'_path' for c in CLASSES)}
    if not required.issubset(rows[0]): raise ValueError(f'Missing columns: {required-set(rows[0])}')
    seen=set();hash_splits={};counts={};files=[];pixels=[];groups={};group_coverage={}
    for row in rows:
        dataset=safe_id(row['dataset']);sid=safe_id(row['image_id']);split=row['split']
        if split not in ('train','val','test'): raise ValueError(f'Invalid split: {split}')
        if not allow_constructed and sid.startswith(('SIM','REF','CHECK')):
            raise ValueError('Constructed identifiers are not accepted by the training-data audit')
        if (dataset,sid) in seen: raise ValueError(f'Duplicate identifier: {dataset}/{sid}')
        seen.add((dataset,sid));counts.setdefault(dataset,dict(train=0,val=0,test=0))[split]+=1
        group_coverage[dataset]=group_coverage.get(dataset,0)+bool(row.get('patient_id'))
        if row.get('patient_id'):
            key=(dataset,row['patient_id']);old=groups.setdefault(key,split)
            if old!=split: raise ValueError(f'Patient crosses partitions: {key}')
        im_path=resolve(manifest,row['image_path'])
        with Image.open(im_path) as image:
            dimensions=image.size
            # Canonical decoded RGB hash also detects re-encoded duplicates.
            import hashlib
            content=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        old=hash_splits.setdefault((dataset,content),split)
        if old!=split: raise ValueError(f'Decoded image content crosses partitions: {sid}')
        files.append(dict(dataset=dataset,image_id=sid,kind='image',sha256=digest(im_path)))
        stat=dict(dataset=dataset,image_id=sid,split=split,canvas_pixels=size*size)
        for c in CLASSES:
            value=row[c+'_path'].strip()
            if not value:
                raise ValueError(f'{sid}/{c}: absent mask must be resolved explicitly, not treated as negative')
            path=resolve(manifest,value)
            with Image.open(path) as m:
                if m.size!=dimensions: raise ValueError(f'Image/mask dimensions differ: {sid}/{c}')
            mask=load_mask(path,size);stat[c+'_positive_pixels']=int(mask.sum())
            files.append(dict(dataset=dataset,image_id=sid,kind=c,sha256=digest(path)))
        pixels.append(stat)
    if expected is not None and counts!=expected: raise ValueError(f'Partition counts differ: {counts} != {expected}')
    summary=[]
    for dataset in counts:
        r=[p for p in pixels if p['dataset']==dataset];den=sum(p['canvas_pixels'] for p in r)
        summary.append(dict(dataset=dataset,images=len(r),canvas_pixels=den,
          **{c+'_percent':100*sum(p[c+'_positive_pixels'] for p in r)/den for c in CLASSES}))
    write_csv(Path(out)/'input_hashes.csv',files);write_csv(Path(out)/'pixel_counts.csv',pixels)
    write_csv(Path(out)/'pixel_ratios.csv',summary)
    write_json(Path(out)/'audit.json',dict(manifest_sha256=digest(manifest),input_size=size,
      split_counts=counts,patient_ids_available=group_coverage,patient_grouping_verified=
      {d:group_coverage[d]==sum(counts[d].values()) for d in counts},
      ratio_denominator='pooled full resized canvases; all partitions; black background retained'))
    return rows

class ManifestDataset(Dataset):
    def __init__(self,manifest,split,size=1024,dataset=None):
        self.manifest=Path(manifest);self.size=size
        self.rows=[r for r in read_csv(manifest) if r['split']==split and (dataset is None or r['dataset']==dataset)]
        if not self.rows: raise ValueError(f'No cases for {dataset}/{split}')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(resolve(self.manifest,r['image_path'])) as im:
            x=np.asarray(im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)).copy()
        y=np.stack([load_mask(resolve(self.manifest,r[c+'_path']),self.size) for c in CLASSES])
        return torch.from_numpy(x).permute(2,0,1).float()/255,torch.from_numpy(y).float(),r['image_id']

class CheckDataset(Dataset):
    def __init__(self,split,size=64): self.split=split;self.size=size
    def __len__(self): return 3 if self.split=='train' else 1
    def __getitem__(self,i):
        seed=20261112+i+{'train':0,'val':100,'test':200}[self.split]
        rng=np.random.default_rng(seed);s=self.size;yy,xx=np.mgrid[:s,:s]
        masks=np.stack([(xx-(14+8*c))**2+(yy-(20+3*i))**2<(4+c)**2 for c in range(4)])
        x=np.clip(rng.normal(.5,.04,(s,s,3)),0,1).astype('float32')
        x[:,:,0]+=masks[0]*.2;x[:,:,1]+=masks[2]*.2
        return torch.from_numpy(x).permute(2,0,1),torch.from_numpy(masks).float(),f'CHECK_{self.split}_{i:03d}'
