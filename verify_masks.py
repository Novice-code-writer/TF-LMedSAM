"""Recompute supplied mask metrics independently of stored score fields."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tflmedsam.io import read_csv,write_json,digest
from tflmedsam.metrics import measure

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    rows=read_csv(a.directory/'per_image_class.csv');errors=[]
    for row in rows:
        q=a.directory/row['prediction_path'];y=a.directory/row['reference_path']
        if digest(q)!=row['prediction_sha256']:errors.append([row['image_id'],'hash'])
        with Image.open(q) as f:pred=np.asarray(f)>0
        with Image.open(y) as f:truth=np.asarray(f)>0
        actual=measure(pred,truth)
        for k in ['tp','fp','fn','tn','dice','iou']:
            if abs(float(row[k])-actual[k])>1e-10:errors.append([row['image_id'],k])
    write_json(a.directory/'mask_verification.json',dict(records=len(rows),errors=errors,passed=not errors))
    if errors:raise ValueError(errors[:10])
    print(f'Verified {len(rows)} prediction masks')
if __name__=='__main__':main()
