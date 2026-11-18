import numpy as np
from . import CLASSES

def measure(prediction,target):
    p=np.asarray(prediction,bool);y=np.asarray(target,bool)
    if p.shape!=y.shape: raise ValueError('Prediction and reference shapes differ')
    tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum());tn=int((~p&~y).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
      dice=2*tp/(2*tp+fp+fn) if tp+fn else None,
      iou=tp/(tp+fp+fn) if tp+fn else None,
      sensitivity=tp/(tp+fn) if tp+fn else None,
      specificity=tn/(tn+fp) if tn+fp else None,
      reference_positive=bool(tp+fn),negative_image_false_positive=bool(not(tp+fn) and fp))

def summarize(rows):
    per_class=[]
    for c in CLASSES:
        group=[r for r in rows if r['lesion']==c];negative=[r for r in group if not r['reference_positive']]
        row=dict(lesion=c,images=len(group),positive_images=sum(r['reference_positive'] for r in group),negative_images=len(negative))
        for k in ['dice','iou','sensitivity','specificity']:
            v=[r[k] for r in group if r[k] is not None];row[k]=float(np.mean(v)) if v else None
        row['negative_image_false_positive_rate']=sum(r['negative_image_false_positive'] for r in negative)/len(negative) if negative else None
        for k in ['tp','fp','fn','tn']:row[k]=sum(r[k] for r in group)
        per_class.append(row)
    macro={}
    for k in ['dice','iou','sensitivity','specificity']:
        v=[r[k] for r in per_class if r[k] is not None];macro[k]=float(np.mean(v)) if v else None
    return dict(macro=macro,per_class=per_class)
