"""Generate binary-mask records constrained by aggregate overlap metrics."""
import argparse
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.special import expit,logit
from scipy.stats import norm
from tflmedsam import CLASSES
from tflmedsam.io import read_csv,write_csv,write_json,digest
from tflmedsam.metrics import measure

def seed(*values):return int(hashlib.sha256('|'.join(map(str,values)).encode()).hexdigest()[:8],16)

def fit(dice,iou,n,random_seed):
    d=dice/100;j=iou/100;z=norm.ppf((np.arange(n)+.5)/n)
    def residual(x):
        a=expit(x[0]+x[1]*z);return [a.mean()-d,(a/(2-a)).mean()-j]
    result=least_squares(residual,[logit(d),.7],bounds=([-20,0],[20,30]),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=1000)
    if max(abs(x) for x in residual(result.x))>1e-7:raise ValueError('Paired moments are incompatible')
    return np.random.default_rng(random_seed).permutation(expit(result.x[0]+result.x[1]*z))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Choose an empty output directory')
    a.output.mkdir(parents=True);root=Path(__file__).resolve().parent;jobs=[]
    for r in read_csv(root/'source_data/table2.csv'):
        for dataset in ['IDRiD','DDR']:
            jobs.append(dict(experiment='comparison',dataset=dataset,method=r['method'],dice=float(r[dataset+'_Dice_percent']),iou=float(r[dataset+'_IoU_percent'])))
    for r in read_csv(root/'source_data/table3.csv'):
        jobs.append(dict(experiment='ablation',dataset='IDRiD',method=r['configuration'],dice=float(r['Dice_percent']),iou=float(r['IoU_percent'])))
    for r in read_csv(root/'source_data/table4.csv'):
        for source,target in [('IDRiD','DDR'),('DDR','IDRiD')]:
            jobs.append(dict(experiment=f'cross_{source}_to_{target}',dataset=target,method=r['method'],dice=float(r[source+'_to_'+target+'_Dice_percent']),iou=None))
    records=[];summary=[];case_cache={}
    for job in jobs:
        n={'IDRiD':27,'DDR':225}[job['dataset']];d=job['dice']/100
        fit_iou=job['iou'] if job['iou'] is not None else 100*(d/(2-d)+.02*(d-d/(2-d)))
        for c,label in enumerate(CLASSES):
            fit_experiment='comparison' if job['experiment']=='ablation' and job['method']=='+ Uncertainty Loss (Full)' else job['experiment']
            fit_method='TF-LMedSAM' if job['experiment']=='ablation' and job['method']=='+ Uncertainty Loss (Full)' else job['method']
            scores=fit(job['dice'],fit_iou,n,seed(20261112,fit_experiment,job['dataset'],fit_method,label))
            if fit_method=='TF-LMedSAM':
                scores=np.roll(np.sort(scores),(c*n)//4)
            for i,score in enumerate(scores):
                sid=f'REF_{job["dataset"]}_{i+1:04d}';key=(job['dataset'],sid,label)
                if key not in case_cache:
                    rng=np.random.default_rng(seed(*key));yy,xx=np.mgrid[:128,:128]
                    order=np.argsort(((xx-64)**2+(yy-64)**2+rng.normal(0,20,(128,128))).ravel(),kind='stable')
                    truth=np.zeros(128*128,bool);truth[order[:2048]]=True
                    positives=order[:2048].copy();rng.shuffle(positives);negative=order[2048:]
                    folder=Path('cases')/job['dataset']/sid;(a.output/folder).mkdir(parents=True,exist_ok=True)
                    target_path=folder/(label+'_reference.png');input_path=folder/(label+'_input.png')
                    Image.fromarray(np.uint8(truth.reshape(128,128))*255).save(a.output/target_path)
                    gray=rng.normal(.5,.025,(128,128));gray[truth.reshape(128,128)]+=-.25 if label in ['MA','HE'] else .25
                    Image.fromarray(np.uint8(np.clip(gray,0,1)*255)).save(a.output/input_path)
                    case_cache[key]=(truth.reshape(128,128),positives,negative,target_path,input_path)
                truth,pos,neg,target_path,input_path=case_cache[key];tp=int(round(2048*score));fp=2048-tp
                pred=np.zeros(128*128,bool);pred[pos[:tp]]=True;pred[neg[:fp]]=True;pred=pred.reshape(128,128)
                tag=hashlib.sha256(job['method'].encode()).hexdigest()[:12]
                pred_path=Path('predictions')/job['experiment']/job['dataset']/tag/sid/(label+'.png')
                (a.output/pred_path).parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.uint8(pred)*255).save(a.output/pred_path)
                records.append(dict(experiment=job['experiment'],dataset=job['dataset'],method=job['method'],image_id=sid,lesion=label,
                  **measure(pred,truth),record_origin='constructed_metric_fixture',optimization_executed=False,
                  reference_path=target_path.as_posix(),input_path=input_path.as_posix(),prediction_path=pred_path.as_posix(),prediction_sha256=digest(a.output/pred_path)))
        subset=[r for r in records if all(r[k]==job[k] for k in ['experiment','dataset','method'])]
        dmean=float(np.mean([r['dice'] for r in subset])*100);jmean=float(np.mean([r['iou'] for r in subset])*100)
        if abs(dmean-job['dice'])>.01 or (job['iou'] is not None and abs(jmean-job['iou'])>.01):raise ValueError('Rasterization error exceeds 0.01 percentage points')
        summary.append(dict(experiment=job['experiment'],dataset=job['dataset'],method=job['method'],reported_dice_percent=job['dice'],
           reported_iou_percent=job['iou'],constructed_dice_percent=dmean,constructed_iou_percent=jmean,records=len(subset),record_origin='constructed_metric_fixture'))
        print(job['experiment'],job['dataset'],job['method'],round(dmean,4),flush=True)
    write_csv(a.output/'per_image_class.csv',records);write_csv(a.output/'aggregate.csv',summary)
    write_json(a.output/'origin.json',dict(record_origin='constructed_metric_fixture',network_inference_executed=False,
      official_image_identifiers=False,input_size=128,reference_positive_pixels=2048,metric_scope='positive lesion-centered regions',
      table1_prevalence_recomputed=False,conditions=len(jobs),mask_records=len(records)))
if __name__=='__main__':main()
