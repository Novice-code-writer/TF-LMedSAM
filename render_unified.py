"""Render figures only from the unified record family."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tflmedsam import CLASSES
from tflmedsam.io import read_json,write_json,digest
from unified_records import DEFAULT,load_prediction,sid
from project_paths import FIGURES

COLORS=['#7797B7','#99B08C','#D4AA7A','#A797BC','#70A2A5','#C87871']
METHODS=['U-Net','Att-UNet','TransUNet','Swin-Unet','MedSAM (Full)','TF-LMedSAM']
def closed(ax):
    ax.grid(False);ax.minorticks_off();ax.tick_params(direction='in',top=True,right=True)
    for spine in ax.spines.values():spine.set_visible(True)
def save(fig,out,n):
    for ext in ['png','svg','pdf']:fig.savefig(out/f'Figure {n}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.12,facecolor='white')
    fig.savefig(out/f'Figure {n}_preview.png',dpi=120,bbox_inches='tight',pad_inches=.12,facecolor='white');plt.close(fig)
def clean(ax,img,title=None,cmap=None):
    ax.imshow(img,cmap=cmap);ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    if title:ax.set_title(title,fontsize=11,pad=8)
def overlay(image,mask,color):
    image=np.asarray(image,float)/255
    result=image.copy();result[mask]=.35*image[mask]+.65*np.array(color)
    return result
def render(root=DEFAULT):
    root=Path(root);out=FIGURES if root.resolve()==DEFAULT.resolve() else root/'figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.labelsize':14,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':1.1})
    manifest=read_json(root/'manifest.json')['rows'];byid={r['image_id']:r for r in manifest}
    records=read_json(root/'per_image_class.json')['rows'];lookup={(r['result_id'],r['image_id']):r for r in records}
    image_rows=read_json(root/'per_image.json')['rows'];summary={r['result_id']:r for r in read_json(root/'summary.json')['results']}
    aliases={r['id']:r['result_id'] for r in read_json(root/'summary.json')['aliases']}
    sources={}
    def pixels(i):
        row=byid[i]
        with Image.open(root/row['image_path']) as im:img=np.asarray(im.convert('RGB'))
        truth=np.stack([np.asarray(Image.open(root/row[c+'_path']))>0 for c in CLASSES])
        return img,truth
    def pred(k,i):return load_prediction(root,lookup[(k,i)])
    full=[r for r in image_rows if r['result_id']=='comparison/IDRiD/TF-LMedSAM' and r['image_macro_dice_percent'] is not None]
    ordered=sorted(full,key=lambda r:r['image_macro_dice_percent'])
    ids=[ordered[round((len(ordered)-1)*q)]['image_id'] for q in [.25,.5,.75]]
    # Figure 3 shares the exact image/reference files used by the metrics.
    fig,axes=plt.subplots(3,6,figsize=(12,6.5),layout='constrained')
    for j,i in enumerate(ids):
        img,y=pixels(i)
        values=[img,img,*y]
        for k,v in enumerate(values):clean(axes[j,k],v,['Input RGB','Resized 1024 × 1024','MA mask','HE mask','EX mask','SE mask'][k] if j==0 else None,'gray' if k>=2 else None)
        axes[j,0].set_ylabel(chr(65+j),weight='bold')
    save(fig,out,3);sources['Figure 3']=dict(image_ids=ids,manifest='manifest.json',reference_fields=[c+'_path' for c in CLASSES])
    # Figure 4 uses all 27 valid image-level scores.
    values=[[r['image_macro_dice_percent'] for r in image_rows if r['result_id']==f'comparison/IDRiD/{m}' and r['image_macro_dice_percent'] is not None] for m in METHODS]
    eligible_ids=[{r['image_id'] for r in image_rows if r['result_id']==f'comparison/IDRiD/{m}' and r['image_macro_dice_percent'] is not None} for m in METHODS]
    if not values[0] or not all(ids==eligible_ids[0] for ids in eligible_ids):raise ValueError('Figure 4 requires identical eligible image identifiers')
    fig,ax=plt.subplots(figsize=(9.4,5.5))
    bp=ax.boxplot(values,widths=.56,patch_artist=True,whis=1.5,medianprops=dict(color='#26313B',linewidth=1.8),boxprops=dict(edgecolor='#34414D',linewidth=1.1),whiskerprops=dict(color='#34414D',linewidth=1.1),capprops=dict(color='#34414D',linewidth=1.1),flierprops=dict(marker='o',markersize=4.3,markerfacecolor='white',markeredgecolor='#34414D'))
    for patch,color in zip(bp['boxes'],COLORS):patch.set_facecolor(color)
    ax.set_xticks(np.arange(1,7),['U-Net','Attention\nU-Net','TransUNet','Swin-Unet','MedSAM\n(full)','TF-LMedSAM'])
    ax.set(ylim=(0,100),ylabel='Image-level macro Dice (%)');ax.yaxis.set_major_locator(MultipleLocator(20));closed(ax)
    ax.text(.985,.965,f'n = {len(values[0])} per method',transform=ax.transAxes,ha='right',va='top',fontsize=11)
    fig.subplots_adjust(left=.1,right=.98,bottom=.17,top=.97);save(fig,out,4)
    sources['Figure 4']=dict(records='per_image.json',statistics='distributions.json',result_ids=[f'comparison/IDRiD/{m}' for m in METHODS],n=len(values[0]),eligible_image_ids=sorted(eligible_ids[0]))
    # Figure 5: select by image-score rank, then select connected components.
    fig,axes=plt.subplots(2,6,figsize=(12,4.4),layout='constrained');regions=[]
    for j,(i,c) in enumerate([(ids[0],0),(ids[2],1)]):
        img,y=pixels(i);components,n=label(y[c]);sizes=np.bincount(components.ravel());sizes[0]=0
        positive=np.flatnonzero(sizes)
        component=positive[np.argmin(sizes[positive])] if j==0 else int(np.argmax(sizes))
        yy,xx=np.nonzero(components==component);cx=int(np.mean(xx));cy=int(np.mean(yy));half=max(48,int(max(np.ptp(xx),np.ptp(yy))*.8)+12)
        x0=max(0,cx-half);x1=min(1024,cx+half);y0=max(0,cy-half);y1=min(1024,cy+half)
        vals=[img[y0:y1,x0:x1],y[c,y0:y1,x0:x1]]+[pred(f'comparison/IDRiD/{m}',i)[c,y0:y1,x0:x1] for m in ['U-Net','TransUNet','MedSAM (Full)','TF-LMedSAM']]
        for k,v in enumerate(vals):clean(axes[j,k],v,['Input region','Reference','U-Net','TransUNet','MedSAM','TF-LMedSAM'][k] if j==0 else None,'gray' if k else None)
        if j==0:axes[j,0].annotate('',xy=(cx-x0,cy-y0),xytext=(min(x1-x0-4,cx-x0+half*.7),cy-y0),arrowprops=dict(arrowstyle='->',color='#D74438',lw=1.4))
        axes[j,0].set_ylabel(chr(65+j)+'  '+CLASSES[c],weight='bold')
        regions.append(dict(image_id=i,lesion=CLASSES[c],crop_xyxy=[x0,y0,x1,y1],selection='image-score quartile rank; component size rule'))
    save(fig,out,5);sources['Figure 5']=dict(regions=regions,records='per_image_class.json')
    # Figure 7: canonical aliases guarantee identical full-data endpoints and CIs.
    fractions=[10,20,40,60,80,100];fig,ax=plt.subplots(figsize=(8.4,5.4));points=[]
    for method,color,marker in [('TF-LMedSAM','#BD4542','o'),('TransUNet','#346F9D','^')]:
        y=[];lo=[];hi=[]
        for f in fractions:
            view=f'training_fraction/DDR/{method}/{f}';key=aliases.get(view,view);s=summary[key];v=s['macro']['dice']*100
            y.append(v);lo.append(v-s['dice_ci95_percent'][0]);hi.append(s['dice_ci95_percent'][1]-v)
            points.append(dict(method=method,fraction=f,result_id=key,dice_percent=v,ci95_percent=s['dice_ci95_percent']))
        ax.errorbar(fractions,y,yerr=[lo,hi],color=color,marker=marker,markersize=6,capsize=3,linewidth=1.4,label=method)
    ax.set(xlabel='DDR training data (%)',ylabel='Mean Dice (%)',xlim=(0,105),ylim=(25,80));ax.set_xticks([0,20,40,60,80,100]);ax.yaxis.set_major_locator(MultipleLocator(10));ax.legend(frameon=False,loc='lower right');closed(ax);fig.tight_layout();save(fig,out,7)
    sources['Figure 7']=dict(points=points,records='summary.json',training_subsets='splits/nested_training_subsets.json',error_bars='95% percentile bootstrap confidence intervals')
    # Figure 8 uses only transfer methods with actual saved outputs in this set.
    cross_id='cross_DDR_to_IDRiD/IDRiD/TF-LMedSAM';cross_rows=sorted([r for r in image_rows if r['result_id']==cross_id and r['image_macro_dice_percent'] is not None],key=lambda r:r['image_macro_dice_percent'])
    cross_ids=[cross_rows[round((len(cross_rows)-1)*q)]['image_id'] for q in [.25,.75]]
    fig,axes=plt.subplots(2,4,figsize=(10.5,6),layout='constrained')
    for j,i in enumerate(cross_ids):
        img,y=pixels(i);truth=y.any(0);trans=pred('cross_DDR_to_IDRiD/IDRiD/TransUNet',i).any(0);full=pred(cross_id,i).any(0)
        values=[img,overlay(img,truth,[.15,.85,.2]),overlay(img,trans,[1,.15,.1]),overlay(img,full,[1,.15,.1])]
        for k,v in enumerate(values):clean(axes[j,k],v,['Input image','Reference (green)','TransUNet (red)','TF-LMedSAM (red)'][k] if j==0 else None)
        axes[j,0].set_ylabel(chr(65+j),weight='bold')
    save(fig,out,8);sources['Figure 8']=dict(image_ids=cross_ids,source_dataset='DDR',target_dataset='IDRiD',crop_xyxy=[0,0,1024,1024],results=['cross_DDR_to_IDRiD/IDRiD/TransUNet',cross_id],selection='nearest image-score ranks to the 25th and 75th percentiles',records='per_image_class.json')
    # Figure 9 keeps published template FPS values separate from mask-derived Dice.
    fps=read_json(root/'frozen_report.json')['sources']['figure9.csv']['rows'];fig,ax=plt.subplots(figsize=(8,5.3));points=[]
    offsets=[(-8,12),(10,-18),(10,-15),(10,9)]
    for j,r in enumerate(fps):
        value=summary['comparison/IDRiD/'+r['method']]['macro']['dice']*100
        ci=summary['comparison/IDRiD/'+r['method']]['dice_ci95_percent']
        ax.errorbar(float(r['FPS']),value,yerr=np.array([[value-ci[0]],[ci[1]-value]]),fmt=['s','^','D','o'][j],markersize=8,color=COLORS[[0,2,4,5][j]],markeredgecolor='#34414D',markeredgewidth=.8,capsize=4,elinewidth=1.2)
        ax.annotate(r['method'],(float(r['FPS']),value),xytext=offsets[j],textcoords='offset points',ha='right' if j==0 else 'left',fontsize=11)
        points.append(dict(method=r['method'],dice_percent=value,dice_ci95_percent=ci,fps=float(r['FPS'])))
    ax.set(xlabel='Inference speed (frames/s)',ylabel='Mean Dice on IDRiD (%)',xlim=(10,50),ylim=(60,84));ax.xaxis.set_major_locator(MultipleLocator(10));ax.yaxis.set_major_locator(MultipleLocator(4));closed(ax);fig.tight_layout();save(fig,out,9)
    sources['Figure 9']=dict(points=points,dice_source='summary.json',fps_source='frozen_report.json',error_bars='vertical 95% image-level bootstrap intervals for Dice',device_timing_verified=False)
    write_json(root/'figure_sources.json',dict(provenance_notice='../../材料来源与适用范围.txt',figures=sources,output_files=[dict(path=str(p.relative_to(DEFAULT.parents[1])) if p.is_relative_to(DEFAULT.parents[1]) else str(p),sha256=digest(p)) for p in sorted(out.glob('Figure *.png')) if '_preview' not in p.name]))
    print('Rendered figures 3, 4, 5, 7, 8 and 9:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args();render(a.root)
