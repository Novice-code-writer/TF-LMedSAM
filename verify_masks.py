"""Recompute supplied mask metrics independently of stored score fields."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from tflmedsam.io import read_csv,write_json,digest
from tflmedsam.metrics import measure

def local_file(root,value):
    root=Path(root).resolve();path=(root/value).resolve()
    if not path.is_relative_to(root):raise ValueError('Record path leaves its directory')
    if not path.is_file():raise FileNotFoundError(path)
    return path

def verify(directory):
    directory=Path(directory);rows=read_csv(directory/'per_image_class.csv')
    errors=[];groups=defaultdict(list);image_scores=defaultdict(list);seen=set();hashes={};reference_cache={}
    def checked_hash(path):
        if path not in hashes:hashes[path]=digest(path)
        return hashes[path]
    for i,row in enumerate(rows):
        identity=tuple(row.get(k,'') for k in ['experiment','dataset','method','image_id','lesion'])
        if identity in seen:errors.append(dict(row=i+2,error='duplicate record',identity=identity))
        seen.add(identity)
        try:
            q=local_file(directory,row['prediction_path']);y=local_file(directory,row['reference_path'])
            if row.get('prediction_sha256') and checked_hash(q)!=row['prediction_sha256']:raise ValueError('Prediction hash differs')
            if row.get('reference_sha256') and checked_hash(y)!=row['reference_sha256']:raise ValueError('Reference hash differs')
            if row.get('state_path'):
                state=local_file(directory,row['state_path'])
                if checked_hash(state)!=row['state_sha256']:raise ValueError('State hash differs')
            with Image.open(q) as f:pred=np.asarray(f)>0
            if y not in reference_cache:
                if len(reference_cache)>=64:reference_cache.pop(next(iter(reference_cache)))
                with Image.open(y) as f:reference_cache[y]=np.asarray(f)>0
            truth=reference_cache[y]
            if pred.ndim!=2 or truth.ndim!=2:raise ValueError('Expected single-channel masks')
            actual=measure(pred,truth)
            for k in ['tp','fp','fn','tn','dice','iou','sensitivity','specificity']:
                if k not in row:continue
                saved=row[k].strip()
                if actual[k] is None:
                    if saved.lower() not in ('','none','null','nan'):raise ValueError(k+' must be undefined')
                elif not saved or abs(float(saved)-actual[k])>1e-10:raise ValueError(k+' differs')
            group=tuple(row.get(k,'') for k in ['experiment','dataset','method'])
            groups[group].append(dict(image_id=row['image_id'],lesion=row['lesion'],**actual))
            image_scores[(*group,row['image_id'])].append(actual['dice'])
        except (OSError,ValueError,KeyError) as exc:
            errors.append(dict(row=i+2,image_id=row.get('image_id'),lesion=row.get('lesion'),error=str(exc)))
        if (i+1)%2000==0:print(f'Checked {i+1}/{len(rows)} mask records',flush=True)
    from tflmedsam.metrics import summarize
    summaries=[dict(experiment=k[0],dataset=k[1],method=k[2],**summarize(v)) for k,v in sorted(groups.items())]
    distributions=[]
    for group in sorted(groups):
        values=[float(np.mean([v for v in vals if v is not None]))*100 for key,vals in image_scores.items() if key[:3]==group and any(v is not None for v in vals)]
        q=np.quantile(values,[.25,.5,.75]) if values else [None]*3
        distributions.append(dict(experiment=group[0],dataset=group[1],method=group[2],n=len(values),q1_percent=q[0],median_percent=q[1],q3_percent=q[2]))
    return dict(records=len(rows),errors=errors,passed=bool(rows) and not errors,summary=summaries,image_distributions=distributions,
        verification_scope='saved file hashes and pixel-level recomputation; source authenticity and neural inference are outside this check')

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--report',type=Path,help='Write outside the source directory when preserving supplied records')
    a=p.parse_args();result=verify(a.directory)
    write_json(a.report or a.directory/'mask_verification.json',result)
    if not result['passed']:raise ValueError(result['errors'][:10])
    print(f'Verified {result["records"]} prediction-mask pairs')
if __name__=='__main__':main()
