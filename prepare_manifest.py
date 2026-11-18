"""Enumerate release image identifiers and resolve lesion masks."""
import argparse, csv, json, random, re
from pathlib import Path
from PIL import Image
import numpy as np

CLASSES=('MA','HE','EX','SE')
def split_name(path):
    value='/'.join(path.parts).lower()
    if re.search(r'test',value):return 'test'
    if re.search(r'valid|(?:^|/)val(?:/|$)',value):return 'val'
    if re.search(r'train',value):return 'train'
    return None
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['IDRiD','DDR'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--allow-missing-as-negative',action='store_true');a=p.parse_args()
    root=a.root.resolve();files=[f for f in root.rglob('*') if f.is_file()]
    images=sorted(f for f in files if f.suffix.lower() in ['.jpg','.jpeg'] and not any(x in str(f.relative_to(root)).lower() for x in ['groundtruth','annotation','mask']))
    if not images:raise ValueError('No JPEG images found; supply the segmentation release root')
    masks=[f for f in files if f.suffix.lower() in ['.tif','.tiff','.png']]
    index={}
    for f in masks:
        stem=f.stem;parts=[x.upper() for x in f.parts]
        for c in CLASSES:
            key=None
            if stem.upper().endswith('_'+c):key=stem[:-(len(c)+1)]
            elif c in parts:key=stem
            if key is not None:
                k=(split_name(f.relative_to(root)),key.casefold(),c)
                if k in index:raise ValueError(f'Duplicate mask key: {k}')
                index[k]=f
    train=[f for f in images if split_name(f.relative_to(root))=='train']
    selected=set()
    if a.dataset=='IDRiD':
        if len(train)!=54:raise ValueError(f'IDRiD development images: expected 54, found {len(train)}')
        shuffled=train.copy();random.Random(a.seed).shuffle(shuffled);selected=set(shuffled[:11])
    a.output.parent.mkdir(parents=True,exist_ok=True);rows=[]
    for f in images:
        original=split_name(f.relative_to(root));split='val' if f in selected else original
        if split is None:raise ValueError(f'Cannot determine release partition: {f}')
        row=dict(dataset=a.dataset,image_id=f.stem,split=split,patient_id='',eye_id='',image_path=str(f.resolve()))
        for c in CLASSES:
            found=index.get((original,f.stem.casefold(),c)) or index.get((None,f.stem.casefold(),c))
            if found is None:
                if not a.allow_missing_as_negative:raise FileNotFoundError(f'Unresolved {c} mask for {f.name}; review annotation metadata')
                with Image.open(f) as im:size=im.size
                found=a.output.parent/'resolved_empty_masks'/a.dataset/(f.stem+'_'+c+'.png');found.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.zeros((size[1],size[0]),np.uint8)).save(found)
            row[c+'_path']=str(found.resolve())
        rows.append(row)
    counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test']}
    expected={'IDRiD':{'train':43,'val':11,'test':27},'DDR':{'train':383,'val':149,'test':225}}[a.dataset]
    if counts!=expected:raise ValueError(f'Partition counts differ: {counts}; expected {expected}')
    with a.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    a.output.with_suffix('.json').write_text(json.dumps(dict(dataset=a.dataset,root=str(root),counts=counts,seed=a.seed,
        idrid_validation_selection='seeded new selection from development images' if a.dataset=='IDRiD' else None,
        missing_masks_explicitly_allowed=a.allow_missing_as_negative,patient_eye_source='not inferred'),indent=2),encoding='utf-8')
    print(a.output)
if __name__=='__main__':main()
