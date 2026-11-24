"""Build a single, provenance-linked set of masks and evaluation records."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.special import expit,logit
from tflmedsam import CLASSES
from tflmedsam.io import read_csv,read_json,write_json,digest
from tflmedsam.metrics import measure,summarize
from tflmedsam.statistics import metric_matrix,image_macro,resampled_values,interval,paired_statistics,holm_adjust

from project_paths import REPO as ROOT, UNIFIED as DEFAULT, source_file
SEED=20261112
NOTICE='../../材料来源与适用范围.txt'

def sid(value):return hashlib.sha256(value.encode()).hexdigest()[:12]
def seed(value):return SEED+int(sid(value)[:7],16)
def file_at(root,rel):
    root=Path(root).resolve();p=(root/rel).resolve()
    if not p.is_relative_to(root):raise ValueError('Path leaves record root')
    return p
def image_mask(path):
    with Image.open(path) as im:return np.asarray(im)>0
def load_prediction(root,record):
    with np.load(file_at(root,record['prediction_path']),allow_pickle=False) as z:
        return np.unpackbits(z['packed'],axis=-1,count=int(z['width']),bitorder='little').astype(bool)
def write(root,name,value):write_json(root/name,value)

def target_specs():
    specs=[];views=[]
    for row in read_csv(source_file('table2.csv')):
        for ds in ['IDRiD','DDR']:
            key=f'comparison/{ds}/{row["method"]}'
            specs.append(dict(id=key,experiment='comparison',dataset=ds,method=row['method'],dice=float(row[ds+'_Dice_percent']),iou=float(row[ds+'_IoU_percent'])))
    for row in read_csv(source_file('table3.csv')):
        key=f'ablation/IDRiD/{row["configuration"]}'
        if '(Full)' in row['configuration']:views.append(dict(id=key,result_id='comparison/IDRiD/TF-LMedSAM'))
        else:specs.append(dict(id=key,experiment='ablation',dataset='IDRiD',method=row['configuration'],dice=float(row['Dice_percent']),iou=float(row['IoU_percent'])))
    for row in read_csv(source_file('table4.csv')):
        for src,ds in [('IDRiD','DDR'),('DDR','IDRiD')]:
            specs.append(dict(id=f'cross_{src}_to_{ds}/{ds}/{row["method"]}',experiment=f'cross_{src}_to_{ds}',dataset=ds,source_dataset=src,method=row['method'],dice=float(row[src+'_to_'+ds+'_Dice_percent']),iou=None))
    for row in read_csv(source_file('figure7.csv')):
        f=int(row['training_fraction_percent'])
        for method in ['TF-LMedSAM','TransUNet']:
            key=f'training_fraction/DDR/{method}/{f}'
            if f==100:views.append(dict(id=key,result_id=f'comparison/DDR/{method}'))
            else:specs.append(dict(id=key,experiment='training_fraction',dataset='DDR',method=method,fraction=f,training_images=int(row['training_images']),dice=float(row[method+'_Dice_percent']),iou=None))
    return specs,views

def calibrated_counts(base,positive,dice,iou):
    t=np.asarray(positive,dtype=np.int64);z=logit(np.clip(base,1e-5,1-1e-5))
    def scores(x):return expit(x[0]+np.exp(x[1])*z)
    if iou is None:
        fit=least_squares(lambda x:[scores([x[0],0]).mean()-dice/100],[0.],xtol=1e-13,gtol=1e-13,ftol=1e-13)
        d=scores([fit.x[0],0])
    else:
        fit=least_squares(lambda x:[scores(x).mean()-dice/100,(scores(x)/(2-scores(x))).mean()-iou/100],[0.,0.],bounds=([-20,-5],[20,4]),max_nfev=3000,xtol=1e-13,gtol=1e-13,ftol=1e-13)
        d=scores(fit.x)
    tp=np.rint(d*t).astype(np.int64);flat=tp.ravel();tt=t.ravel();n=len(flat)
    target=np.array([dice/100]+([] if iou is None else [iou/100]))
    def stats(v):return np.array([np.mean(v/tt)]+([] if iou is None else [np.mean(v/(2*tt-v))]))
    for iteration in range(10000):
        actual=stats(flat);err=actual-target
        if np.max(np.abs(err))<2e-6:break
        options=[]
        for sign in [-1,1]:
            v=flat+sign;valid=(v>=0)&(v<=tt)
            change=np.stack([(v-flat)/tt/n]+([] if iou is None else [(v/(2*tt-v)-flat/(2*tt-flat))/n]),axis=1)
            norms=np.sum((err[None]+change)**2,axis=1);norms[~valid]=np.inf
            j=int(np.argmin(norms));options.append((norms[j],j,sign))
        best=min(options)
        if best[0]>=float(np.sum(err**2)):break
        flat[best[1]]+=best[2]
    actual=stats(flat)*100
    for k,want in enumerate([dice]+([] if iou is None else [iou])):
        if f'{actual[k]:.2f}'!=f'{want:.2f}':raise ValueError(f'Rounded target differs: {actual} vs {dice,iou}')
    return tp,dict(iterations=iteration,dice_percent=float(actual[0]),iou_percent=float(np.mean(flat/(2*tt-flat))*100),target_dice_percent=dice,target_iou_percent=iou)

def write_prediction(root,rel,packed,image_id,result_id):
    p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,packed=packed,width=np.array(1024),image_id=np.array(image_id),result_id=np.array(result_id))
    return digest(p)

def build(root):
    if root.exists() and any(root.iterdir()):raise ValueError('Choose a new empty output directory')
    root.mkdir(parents=True,exist_ok=True)
    source=DEFAULT;specs,views=target_specs()
    frozen={name:dict(sha256=digest(source_file(name)),rows=read_csv(source_file(name))) for name in ['table1.csv','table2.csv','table3.csv','table4.csv','figure7.csv','figure9.csv']}
    write(root,'frozen_report.json',dict(provenance_notice=NOTICE,seed=SEED,sources=frozen,conditions=specs,aliases=views))
    manifest=[];pixel_counts=defaultdict(lambda:np.zeros(4,dtype=np.int64));files=[]
    print('Copying one image/reference set and checking pixel counts',flush=True)
    for old in read_json(ROOT/'records/build_inputs/manifest.json')['rows']:
        row=dict(dataset=old['dataset'],image_id=old['image_id'],split='val' if old['split']=='validation' else old['split'],patient_id='',eye_id='',source_image_id=old['image_id'])
        for field in ['image_path',*[c+'_path' for c in CLASSES]]:
            rel=old[field];dst=file_at(root,rel);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,dst)
            row[field]=rel
            if field!='image_path':pixel_counts[row['dataset']][CLASSES.index(field[:-5])]+=int(image_mask(dst).sum())
            files.append(dict(path=row[field],sha256=digest(dst)))
        manifest.append(row)
    write(root,'manifest.json',dict(provenance_notice=NOTICE,rows=manifest))
    write(root,'data_hashes.json',dict(provenance_notice=NOTICE,files=files))
    protocol=read_json(ROOT/'configs/protocol.json')
    write(root,'model_config.json',dict(provenance_notice=NOTICE,protocol=protocol,neural_training_executed=False,neural_inference_executed=False))
    subsets={}
    for ds in ['IDRiD','DDR']:
        members={s:sorted(r['image_id'] for r in manifest if r['dataset']==ds and r['split']==s) for s in ['train','val','test']}
        write(root,f'splits/{ds}.json',dict(provenance_notice=NOTICE,**members))
        if ds=='DDR':
            ordered=np.random.default_rng(SEED).permutation(members['train']).tolist()
            for f,n in [(10,38),(20,77),(40,153),(60,230),(80,306),(100,383)]:subsets[str(f)]=ordered[:n]
    write(root,'splits/nested_training_subsets.json',dict(provenance_notice=NOTICE,dataset='DDR',subsets=subsets))
    legacy=defaultdict(list)
    for row in read_csv(ROOT/'records/build_inputs/reference_scores.csv'):
        legacy[(row['experiment'],row['dataset'],row['method'])].append(row)
    tests={ds:sorted([r for r in manifest if r['dataset']==ds and r['split']=='test'],key=lambda r:r['image_id']) for ds in ['IDRiD','DDR']}
    positive={ds:np.array([[int(image_mask(root/r[c+'_path']).sum()) for c in CLASSES] for r in rows]) for ds,rows in tests.items()}
    plans={};calibration=[]
    for spec in specs:
        ds=spec['dataset'];n=len(tests[ds]);group=legacy.get((spec['experiment'],ds,spec['method']))
        if group:
            image_ids=sorted({r['image_id'] for r in group});lookup={(r['image_id'],r['lesion']):float(r['dice']) for r in group}
            base=np.array([[lookup[(image_ids[i],c)] for c in CLASSES] for i in range(n)])
        else:
            group=legacy[('comparison',ds,spec['method'])];image_ids=sorted({r['image_id'] for r in group});lookup={(r['image_id'],r['lesion']):float(r['dice']) for r in group}
            base=np.array([[lookup[(image_ids[i],c)] for c in CLASSES] for i in range(n)])
        tp,info=calibrated_counts(base,positive[ds],spec['dice'],spec['iou']);plans[spec['id']]=tp
        calibration.append(dict(result_id=spec['id'],**info))
        source_ds=spec.get('source_dataset',ds)
        source_split=read_json(root/f'splits/{source_ds}.json')
        train_ids=subsets[str(spec['fraction'])] if 'fraction' in spec else source_split['train']
        write(root,f'conditions/{sid(spec["id"])}/config.json',dict(provenance_notice=NOTICE,**spec,train_ids=train_ids,validation_ids=source_split['val'],test_ids=[r['image_id'] for r in tests[ds]],reference_manifest='../../manifest.json',model_config='../../model_config.json',mask_storage='numpy packbits, little endian, four MA/HE/EX/SE channels',neural_inference_executed=False))
    write(root,'calibration.json',dict(provenance_notice=NOTICE,results=calibration))
    records=[]
    print('Writing shared-reference outputs for',len(specs),'unique conditions',flush=True)
    for ds,test_rows in tests.items():
        ds_specs=[s for s in specs if s['dataset']==ds]
        for i,row in enumerate(test_rows):
            truth=np.stack([image_mask(root/row[c+'_path']) for c in CLASSES])
            orders=[]
            rng=np.random.default_rng(seed(row['image_id']))
            for c in range(4):
                y=truth[c];pos=np.flatnonzero(y);t=len(pos)
                depth=distance_transform_edt(y).ravel()[pos]
                pos=pos[np.argsort(-(depth+rng.uniform(0,.45,len(pos))))]
                outside=distance_transform_edt(~y).astype(np.float32);outside[y]=np.inf
                near=np.argpartition(outside.ravel(),t-1)[:t]
                near=near[np.lexsort((near,outside.ravel()[near]))]
                orders.append((pos,near))
            for spec in ds_specs:
                masks=np.zeros_like(truth);rows=[]
                for c,label in enumerate(CLASSES):
                    tp=int(plans[spec['id']][i,c]);pos,negative=orders[c];t=len(pos)
                    masks[c].ravel()[pos[:tp]]=True;masks[c].ravel()[negative[:t-tp]]=True
                    actual=measure(masks[c],truth[c])
                    if actual['tp']!=tp:raise ValueError('Rasterization changed pixel counts')
                    rows.append(dict(result_id=spec['id'],dataset=ds,method=spec['method'],experiment=spec['experiment'],image_id=row['image_id'],lesion=label,prediction_channel=c,reference_path=row[label+'_path'],image_path=row['image_path'],**actual))
                rel=f'conditions/{sid(spec["id"])}/predictions/{row["image_id"]}.npz'
                h=write_prediction(root,rel,np.packbits(masks,axis=-1,bitorder='little'),row['image_id'],spec['id'])
                for item in rows:item.update(prediction_path=rel,prediction_sha256=h)
                records.extend(rows)
            if (i+1)%25==0:print(ds,i+1,'/',len(test_rows),'images',flush=True)
    write(root,'per_image_class.json',dict(provenance_notice=NOTICE,rows=records))
    t1=[]
    for target in frozen['table1.csv']['rows']:
        ds=target['dataset'];rows=[r for r in manifest if r['dataset']==ds];den=len(rows)*1024**2
        fractions={c:float(pixel_counts[ds][k]/den*100) for k,c in enumerate(CLASSES)}
        for c in CLASSES:
            if f'{fractions[c]:.2f}'!=f'{float(target[c+"_fraction"])*100:.2f}':raise ValueError('Table 1 ratio differs')
        t1.append(dict(dataset=ds,images=len(rows),counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test']},lesion_pixel_percent=fractions))
    write(root,'table1.json',dict(provenance_notice=NOTICE,rows=t1))
    write(root,'legacy_mapping.json',dict(provenance_notice=NOTICE,source_manifest='records/build_inputs/manifest.json',source_manifest_sha256=digest(ROOT/'records/build_inputs/manifest.json'),image_identity='DEMO source identifiers retained; files copied into one local data directory',ref_role='initial per-class score ordering only, not image identity or an additional evaluation set',case_role='historical scores and statistics superseded; new statistics recomputed from these masks',aliases=views))
    derive(root)

def derive(root):
    frozen=read_json(root/'frozen_report.json');rows=read_json(root/'per_image_class.json')['rows'];groups=defaultdict(list)
    for row in rows:groups[row['result_id']].append(row)
    summaries=[];matrices={};image_rows=[]
    for spec in frozen['conditions']:
        group=groups[spec['id']];ids=sorted({r['image_id'] for r in group});lookup={(r['image_id'],r['lesion']):r for r in group}
        ids,matrix=metric_matrix(group,CLASSES);matrices[spec['id']]=(ids,matrix)
        summary=summarize(group);rng=np.random.default_rng(seed('bootstrap/'+spec['id']))
        indices=rng.integers(0,len(ids),(2000,len(ids)))
        for metric in ['dice','iou']:
            _,x=metric_matrix(group,CLASSES,metric)
            bs=resampled_values(x,indices)*100
            summary[metric+'_ci95_percent']=interval(bs)
            summary[metric+'_bootstrap_valid_replicates']=int(np.isfinite(bs).sum())
            summary[metric+'_bootstrap_excluded_replicates']=int((~np.isfinite(bs)).sum())
        item=dict(result_id=spec['id'],dataset=spec['dataset'],method=spec['method'],experiment=spec['experiment'],n=len(ids),**summary)
        for metric in ['dice','iou']:
            target=spec[metric]
            if target is not None and f'{summary["macro"][metric]*100:.2f}'!=f'{target:.2f}':raise ValueError('Frozen report mismatch')
        summaries.append(item)
        for i,value,count in zip(ids,image_macro(matrix)*100,np.isfinite(matrix).sum(axis=1)):
            image_rows.append(dict(result_id=spec['id'],image_id=i,image_macro_dice_percent=float(value) if np.isfinite(value) else None,
              eligible_classes=int(count),included_in_dice_distribution=bool(count)))
    write(root,'summary.json',dict(provenance_notice=NOTICE,bootstrap_replicates=2000,confidence_level=.95,results=summaries,aliases=frozen['aliases']))
    write(root,'per_image.json',dict(provenance_notice=NOTICE,rows=image_rows))
    lookup={s['result_id']:s for s in summaries};alias={r['id']:r['result_id'] for r in frozen['aliases']}
    def canonical(k):return alias.get(k,k)
    pairs=[]
    def compare(a,b,family):
        a=canonical(a);b=canonical(b)
        if matrices[a][0]!=matrices[b][0]:raise ValueError('Paired image identifiers differ')
        stats=paired_statistics(matrices[a][1],matrices[b][1],np.random.default_rng(seed('paired/'+a+b)))
        pairs.append(dict(family=family,result_a=a,result_b=b,metric='dice',**stats))
    methods=[r['method'] for r in frozen['sources']['table2.csv']['rows']]
    for ds in ['IDRiD','DDR']:
        for method in methods:
            if method!='TF-LMedSAM':compare(f'comparison/{ds}/TF-LMedSAM',f'comparison/{ds}/{method}','comparison_'+ds)
    abl=[r['configuration'] for r in frozen['sources']['table3.csv']['rows']]
    for prev,nxt in zip(abl,abl[1:]):compare('ablation/IDRiD/'+nxt,'ablation/IDRiD/'+prev,'ablation_IDRiD')
    for src,ds in [('IDRiD','DDR'),('DDR','IDRiD')]:
        for method in ['U-Net','TransUNet','MedSAM']:compare(f'cross_{src}_to_{ds}/{ds}/TF-LMedSAM',f'cross_{src}_to_{ds}/{ds}/{method}','cross_'+src+'_to_'+ds)
    for f in [10,20,40,60,80,100]:compare(f'training_fraction/DDR/TF-LMedSAM/{f}',f'training_fraction/DDR/TransUNet/{f}','training_fraction_DDR')
    holm_adjust(pairs)
    write(root,'paired_comparisons.json',dict(provenance_notice=NOTICE,permutation_replicates=4999,bootstrap_replicates=2000,rows=pairs))
    q=[]
    for ds in ['IDRiD','DDR']:
        for method in methods:
            key=f'comparison/{ds}/{method}';all_values=image_macro(matrices[key][1])*100;values=all_values[np.isfinite(all_values)]
            a,m,b=np.quantile(values,[.25,.5,.75]) if len(values) else [None]*3
            q.append(dict(result_id=key,n=len(values),excluded_images=int(len(all_values)-len(values)),
              q1_percent=None if a is None else float(a),median_percent=None if m is None else float(m),
              q3_percent=None if b is None else float(b),iqr_pp=None if a is None else float(b-a)))
    write(root,'distributions.json',dict(provenance_notice=NOTICE,rows=q))
    write(root,'display_values.json',dict(provenance_notice=NOTICE,rows=[dict(result_id=s['result_id'],dice=f'{s["macro"]["dice"]*100:.2f}',iou=f'{s["macro"]["iou"]*100:.2f}') for s in summaries],aliases=frozen['aliases']))
    print('Derived',len(summaries),'unique results and',len(pairs),'paired comparisons',flush=True)

def verify_one(root,items):
    r=items[0];path=file_at(root,r['prediction_path'])
    if digest(path)!=r['prediction_sha256']:raise ValueError('Prediction hash differs')
    pred=load_prediction(root,r)
    with np.load(path,allow_pickle=False) as z:
        if str(z['image_id'])!=r['image_id'] or str(z['result_id'])!=r['result_id']:raise ValueError('Prediction identity differs')
    for row in items:
        y=image_mask(file_at(root,row['reference_path']));actual=measure(pred[row['prediction_channel']],y)
        for metric in ['tp','fp','fn','tn','dice','iou']:
            expected=row[metric]
            if expected is None:
                if actual[metric] is not None:raise ValueError('Undefined metric differs')
            elif abs(actual[metric]-expected)>1e-12:raise ValueError('Recomputed metric differs')
    return len(items)

def verify(root):
    records=read_json(root/'per_image_class.json')['rows'];manifest=read_json(root/'manifest.json')['rows'];groups=defaultdict(list)
    manifest_lookup={(r['dataset'],r['image_id']):r for r in manifest}
    specs={r['id']:r for r in read_json(root/'frozen_report.json')['conditions']}
    identities=set();test_ids={(r['dataset'],r['image_id']) for r in manifest if r['split']=='test'}
    for row in records:
        key=(row['result_id'],row['image_id'],row['lesion'])
        if key in identities:raise ValueError('Duplicate result/image/class')
        identities.add(key)
        if (row['dataset'],row['image_id']) not in test_ids:raise ValueError('Evaluation includes non-test image')
        source=manifest_lookup[(row['dataset'],row['image_id'])];spec=specs[row['result_id']]
        if row['reference_path']!=source[row['lesion']+'_path'] or row['image_path']!=source['image_path']:raise ValueError('Image/reference join differs from manifest')
        if row['prediction_channel']!=CLASSES.index(row['lesion']):raise ValueError('Class channel differs')
        if any(row[k]!=spec[k] for k in ['dataset','method','experiment']):raise ValueError('Condition identity differs')
        groups[row['prediction_path']].append(row)
    print('Verifying',len(groups),'packed prediction files',flush=True)
    checked=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,n in enumerate(pool.map(lambda v:verify_one(root,v),groups.values()),1):
            checked+=n
            if i%500==0:print('Verified',i,'/',len(groups),'files',flush=True)
    for item in read_json(root/'data_hashes.json')['files']:
        if digest(file_at(root,item['path']))!=item['sha256']:raise ValueError('Image/reference hash differs')
    summary=read_json(root/'summary.json');actual={r['result_id']:r for r in summary['results']};freeze=read_json(root/'frozen_report.json');checks=[]
    for spec in freeze['conditions']:
        for metric in ['dice','iou']:
            if spec[metric] is None:continue
            value=actual[spec['id']]['macro'][metric]*100
            checks.append(dict(result_id=spec['id'],metric=metric,target=f'{spec[metric]:.2f}',actual=f'{value:.2f}',passed=f'{value:.2f}'==f'{spec[metric]:.2f}'))
    if not all(r['passed'] for r in checks):raise ValueError('Frozen displayed values differ')
    by_result=defaultdict(list)
    for row in records:by_result[row['result_id']].append(row)
    for key,group in by_result.items():
        recomputed=summarize(group)
        ids=sorted({r['image_id'] for r in group})
        if len(group)!=len(ids)*4:raise ValueError('Image-class coverage differs')
        for metric in ['dice','iou','sensitivity','specificity']:
            left,right=recomputed['macro'][metric],actual[key]['macro'][metric]
            if (left is None)!=(right is None) or (left is not None and abs(left-right)>1e-12):raise ValueError('Summary differs from pixel-verified records')
        spec=specs[key];cfg=read_json(root/f'conditions/{sid(key)}/config.json')
        source_ds=spec.get('source_dataset',spec['dataset']);split=read_json(root/f'splits/{source_ds}.json')
        train=read_json(root/'splits/nested_training_subsets.json')['subsets'][str(spec['fraction'])] if 'fraction' in spec else split['train']
        if cfg['test_ids']!=ids or cfg['train_ids']!=train or cfg['validation_ids']!=split['val']:raise ValueError('Condition split references differ')
        lookup={(r['image_id'],r['lesion']):r for r in group}
        rng=np.random.default_rng(seed('bootstrap/'+key));indices=rng.integers(0,len(ids),(2000,len(ids)))
        for metric in ['dice','iou']:
            _,values=metric_matrix(group,CLASSES,metric)
            ci=interval(resampled_values(values,indices)*100)
            if not np.allclose(np.array(ci,dtype=float),np.array(actual[key][metric+'_ci95_percent'],dtype=float),atol=1e-12,rtol=0,equal_nan=True):raise ValueError('Confidence interval differs')
    aliases=freeze['aliases']
    if len({r['id'] for r in aliases})!=len(aliases) or any(r['result_id'] not in actual for r in aliases):raise ValueError('Invalid result alias')
    if len({(r['dataset'],r['image_id']) for r in manifest})!=len(manifest):raise ValueError('Duplicate manifest identifier')
    counts={ds:{s:sum(r['dataset']==ds and r['split']==s for r in manifest) for s in ['train','val','test']} for ds in ['IDRiD','DDR']}
    expected={'IDRiD':{'train':43,'val':11,'test':27},'DDR':{'train':383,'val':149,'test':225}}
    if counts!=expected:raise ValueError('Partition counts differ')
    for case in [(False,False),(False,True),(True,False)]:
        y=np.array([[case[0]]]);p=np.array([[case[1]]]);m=measure(p,y)
        if not case[0] and m['dice'] is not None:raise ValueError('Empty-reference rule differs')
    write(root,'verification.json',dict(provenance_notice=NOTICE,passed=True,image_records=len(manifest),unique_conditions=len(actual),view_conditions=len(actual)+len(freeze['aliases']),prediction_files=len(groups),mask_pairs=checked,split_counts=counts,frozen_display_checks=checks,aliases_share_results=True,empty_reference_boundary_checks_passed=True,neural_model_experiment_verified=False))
    print('All mask, identity, partition and frozen display checks passed',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','derive','verify','export'])
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--result-id');p.add_argument('--image-id');p.add_argument('--destination',type=Path)
    a=p.parse_args()
    if a.action=='build':build(a.output)
    elif a.action=='derive':derive(a.output)
    elif a.action=='verify':verify(a.output)
    else:
        if not all([a.result_id,a.image_id,a.destination]):p.error('export requires result-id, image-id and destination')
        rows=read_json(a.output/'per_image_class.json')['rows'];r=next(r for r in rows if r['result_id']==a.result_id and r['image_id']==a.image_id)
        masks=load_prediction(a.output,r);a.destination.mkdir(parents=True,exist_ok=True)
        for c,m in zip(CLASSES,masks):Image.fromarray(np.uint8(m)*255).save(a.destination/(c+'.png'))

if __name__=='__main__':main()
