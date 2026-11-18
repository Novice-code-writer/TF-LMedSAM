"""Create nested subsets from explicitly supplied training identifiers."""
import argparse
import math
from pathlib import Path
import numpy as np
from tflmedsam.io import read_csv,write_csv

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--dataset',choices=['IDRiD','DDR'],default='DDR');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261112);a=p.parse_args()
    rows=read_csv(a.manifest);ids=sorted(r['image_id'] for r in rows if r['dataset']==a.dataset and r['split']=='train')
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Training IDs must be nonempty and unique')
    order=np.random.default_rng(a.seed).permutation(ids).tolist();records=[];a.output.mkdir(parents=True,exist_ok=True)
    for percentage in [10,20,40,60,80,100]:
        n=max(1,math.floor(len(ids)*percentage/100+.5));selected=order[:n]
        (a.output/f'train_{percentage:03d}.txt').write_text('\n'.join(selected)+'\n',encoding='utf-8')
        records.extend(dict(dataset=a.dataset,percentage=percentage,image_id=i,rank=j+1) for j,i in enumerate(selected))
    write_csv(a.output/'nested_subsets.csv',records)
if __name__=='__main__':main()
