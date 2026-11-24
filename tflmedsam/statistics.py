"""Class-balanced statistics retaining image pairing and undefined overlaps."""
import numpy as np


def metric_matrix(rows, classes, metric='dice'):
    ids=sorted({r['image_id'] for r in rows})
    lookup={(r['image_id'],r['lesion']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate image/class metric')
    if len(rows)!=len(ids)*len(classes):raise ValueError('Incomplete image/class metric coverage')
    return ids,np.array([[lookup[(i,c)][metric] for c in classes] for i in ids],dtype=float)


def image_macro(matrix):
    matrix=np.asarray(matrix,dtype=float)
    counts=np.isfinite(matrix).sum(axis=1)
    return np.divide(np.nansum(matrix,axis=1),counts,out=np.full(len(matrix),np.nan),where=counts>0)


def class_balanced(matrix):
    matrix=np.asarray(matrix,dtype=float)
    if np.isfinite(matrix).all():return float(matrix.mean())
    counts=np.isfinite(matrix).sum(axis=0)
    means=np.divide(np.nansum(matrix,axis=0),counts,out=np.full(matrix.shape[1],np.nan),where=counts>0)
    return float(np.nanmean(means)) if (counts>0).any() else None


def resampled_values(matrix,indices):
    matrix=np.asarray(matrix,dtype=float)
    if np.isfinite(matrix).all():return matrix[indices].mean((1,2))
    required=np.isfinite(matrix).any(axis=0)
    if not required.any():return np.full(len(indices),np.nan)
    selected=matrix[indices][:,:,required]
    counts=np.isfinite(selected).sum(axis=1)
    means=np.divide(np.nansum(selected,axis=1),counts,out=np.full(counts.shape,np.nan),where=counts>0)
    # A replicate must represent every class included in the original estimand.
    values=means.mean(axis=1)
    values[(counts==0).any(axis=1)]=np.nan
    return values


def interval(values):
    values=np.asarray(values,dtype=float);valid=values[np.isfinite(values)]
    return [float(x) for x in np.quantile(valid,[.025,.975])] if len(valid) else [None,None]


def paired_statistics(a,b,rng,bootstrap_replicates=2000,permutation_replicates=4999):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or not np.array_equal(np.isfinite(a),np.isfinite(b)):
        raise ValueError('Paired methods must share image/class eligibility')
    d=a-b;delta=class_balanced(d)
    if delta is None:raise ValueError('No eligible paired overlaps')
    idx=rng.integers(0,len(d),(bootstrap_replicates,len(d)))
    bs=resampled_values(d,idx)*100
    if np.isfinite(d).all():
        per_image=d.mean(1)
        null=rng.choice([-1,1],size=(permutation_replicates,len(d)))@per_image/len(d)
    else:
        counts=np.isfinite(d).sum(axis=0);keep=counts>0
        contributions=np.nansum(d[:,keep]/counts[keep],axis=1)/keep.sum()
        null=rng.choice([-1,1],size=(permutation_replicates,len(d)))@contributions
    p=(int((np.abs(null)>=abs(delta)-1e-14).sum())+1)/(permutation_replicates+1)
    return dict(difference_pp=delta*100,ci95_pp=interval(bs),p_raw=p,
      bootstrap_valid_replicates=int(np.isfinite(bs).sum()),bootstrap_excluded_replicates=int((~np.isfinite(bs)).sum()),
      paired_images=len(d),eligible_paired_images=int(np.isfinite(d).any(axis=1).sum()),
      per_class_eligible_counts=np.isfinite(d).sum(axis=0).tolist())


def holm_adjust(rows):
    for family in {r['family'] for r in rows}:
        group=sorted([r for r in rows if r['family']==family],key=lambda r:r['p_raw']);previous=0
        for i,row in enumerate(group):
            previous=max(previous,min(1,row['p_raw']*(len(group)-i)))
            row['p_holm']=previous
    return rows
