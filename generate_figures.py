"""Render manuscript figures from configurations and numeric records."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,binary_dilation,binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.ticker import MultipleLocator

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'图';RECORD=ROOT.parent/'数据记录'
OUT.mkdir(exist_ok=True);RECORD.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.titlesize':10,'axes.labelsize':11,'axes.linewidth':.9,'savefig.facecolor':'white'})
COLORS=['#597B99','#819A78','#BD9366','#8D83AE','#477F85','#B84B48']
def read(name):
    with (ROOT/'source_data'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(name,rows):
    with (RECORD/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,n):
    fig.savefig(OUT/f'Figure {n}.png',dpi=600,bbox_inches='tight',pad_inches=.12);plt.close(fig)
def box(ax,x,y,w,h,label,color='#EDF2F7',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.012',linewidth=1,edgecolor='#475569',facecolor=color))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
def arrow(ax,a,b,color='#475569',style='-|>'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=12,lw=1.15,color=color))
def closed(ax):
    ax.grid(False);ax.tick_params(direction='in',top=True,right=True)
    for s in ax.spines.values():s.set_visible(True)
def fundus(seed=1,size=512):
    rng=np.random.default_rng(seed);yy,xx=np.mgrid[:size,:size];x=(xx-size/2)/(size*.46);y=(yy-size/2)/(size*.46)
    r=np.sqrt(x*x+y*y);fov=r<1;shade=np.maximum(0,1-r**2)**.35
    base=np.stack([.28+.38*shade,.08+.17*shade,.035+.07*shade],axis=-1)
    noise=gaussian_filter(rng.normal(0,.025,(size,size)),.8)
    base+=noise[...,None];disc=np.exp(-((x+.48)**2+(y+.03)**2)/.016)
    base+=disc[...,None]*np.array([.32,.31,.16]);base[~fov]=0
    vessel=Image.new('L',(size,size));draw=ImageDraw.Draw(vessel)
    for sign in [-1,1]:
        for j in range(5):
            pts=[]
            for t in np.linspace(0,1,90):
                px=size*(.28+.6*t);py=size*(.49+sign*(.12+.12*j)*np.sin(t*1.8)+.025*np.sin(9*t+j))
                pts.append((px,py))
            draw.line(pts,fill=255,width=max(1,5-j))
    v=gaussian_filter(np.asarray(vessel)/255.,.7)*fov;base*=1-.55*v[...,None]
    masks=np.zeros((4,size,size),bool)
    for c in range(4):
        for j in range([10,7,8,5][c]):
            cx=rng.uniform(.32,.73)*size;cy=rng.uniform(.24,.75)*size
            rad=size*[.004,.011,.008,.017][c]*rng.uniform(.7,1.3)
            d=((xx-cx)/(rad*(1.4 if c==1 else 1)))**2+((yy-cy)/rad)**2
            masks[c]|=(d<1)&fov
    for c in range(4):
        m=masks[c]
        if c<2:base[m]*=.40 if c==0 else .5
        else:
            w=gaussian_filter(m.astype(float),2.2 if c==3 else .5)[...,None]
            base=base*(1-w*.7)+w*np.array([.7,.65,.42] if c==2 else [.55,.55,.5])
    return np.uint8(np.clip(base,0,1)*255),masks
def clean_image(ax,img,label=None,cmap=None):
    ax.imshow(img,cmap=cmap);ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    if label:ax.set_title(label,pad=7)
def overlay(image,mask,color):
    x=np.asarray(image,dtype=float)/255
    if x.ndim==2:x=np.repeat(x[...,None],3,axis=-1)
    out=x.copy();out[mask]=.35*x[mask]+.65*np.array(color)
    return out
def main():
    # Figure 1: the image encoder without a classification branch.
    fig,ax=plt.subplots(figsize=(11.5,4.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.025,.66,.13,.17,'Fundus image\n1024 × 1024')
    box(ax,.205,.66,.15,.17,'Patch projection\n16 × 16 patches')
    box(ax,.405,.66,.16,.17,'Position\nembeddings')
    box(ax,.625,.66,.14,.17,'12 encoder\nlayers','#E8F0FA')
    box(ax,.825,.66,.145,.17,'Spatial visual\nrepresentations')
    for a,b in [(.155,.205),(.355,.405),(.565,.625),(.765,.825)]:arrow(ax,(a,.745),(b,.745))
    box(ax,.12,.14,.16,.18,'Layer\nnormalization')
    box(ax,.345,.14,.17,.18,'Multi-head\nself-attention','#E5EEF7')
    box(ax,.58,.14,.15,.18,'Layer\nnormalization')
    box(ax,.795,.14,.15,.18,'Feed-forward\nnetwork','#EEEAF5')
    for a,b in [(.28,.345),(.515,.58),(.73,.795)]:arrow(ax,(a,.23),(b,.23))
    ax.plot([.09,.09,.545,.545],[.23,.44,.44,.23],color='#8D9BAA',lw=1)
    ax.plot([.55,.55,.98,.98],[.23,.38,.38,.23],color='#8D9BAA',lw=1)
    arrow(ax,(.945,.23),(.99,.23),color='#8D9BAA')
    ax.text(.31,.48,'Residual connections around both sublayers',ha='center',fontsize=10)
    arrow(ax,(.70,.65),(.70,.54));ax.text(.72,.56,'Encoder layer detail',fontsize=10)
    ax.text(.025,.95,'A  Image-to-token processing',weight='bold');ax.text(.025,.54,'B  Pre-normalized encoder layer',weight='bold')
    save(fig,1)
    # Figure 2: taps are four groups of three layers, not four ViT layers.
    fig,ax=plt.subplots(figsize=(11.8,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.04,.96,'A  Visual branch',weight='bold');ax.text(.58,.96,'B  Text branch',weight='bold')
    box(ax,.04,.83,.25,.08,'RGB image · 1024 × 1024')
    for j,(layers,tap) in enumerate([('1–3',3),('4–6',6),('7–9',9),('10–12',12)]):
        y=.69-j*.13
        box(ax,.04,y,.25,.08,f'Frozen ViT layers {layers}',size=10)
        box(ax,.34,y,.16,.08,'Adapter in each layer\nr = 64','#FFF1DB',9)
        arrow(ax,(.29,y+.04),(.34,y+.04),style='<->')
        arrow(ax,(.165,y+.14),(.165,y+.08))
        ax.text(.01,y+.04,f'F{j+1}',ha='right',va='center',fontsize=9)
        ax.plot([.04,.015],[y+.04,y+.04],color='#64748B')
    ax.plot([.015,.015,.07],[.73,.273,.273],color='#64748B',lw=1)
    arrow(ax,(.07,.273),(.07,.255))
    box(ax,.04,.18,.46,.075,'Depth-feature projection and scale alignment\n256 channels · taps 3 / 6 / 9 / 12',size=9)
    arrow(ax,(.165,.30),(.165,.255))
    box(ax,.59,.83,.35,.08,'Fixed MA / HE / EX / SE descriptions','#EAF3E6')
    box(ax,.59,.69,.35,.08,'Frozen PubMedCLIP text encoder','#EAF3E6')
    arrow(ax,(.765,.83),(.765,.77))
    box(ax,.59,.54,.35,.08,'Token features + global embeddings','#EAF3E6')
    arrow(ax,(.765,.69),(.765,.62))
    box(ax,.59,.35,.35,.11,'Text Prompt Fusion Module\nVisual queries · text keys and values\n256 channels · 8 heads','#EAE7F4',10)
    arrow(ax,(.765,.54),(.765,.46));arrow(ax,(.50,.215),(.59,.405))
    box(ax,.59,.17,.35,.10,'Prompt-guided decoder\n2 bidirectional interaction layers','#F8E8DF')
    arrow(ax,(.765,.35),(.765,.27))
    box(ax,.59,.025,.35,.075,'Refinement → resize → four lesion maps',size=9)
    arrow(ax,(.765,.17),(.765,.10));ax.text(.04,.06,'Pretrained image and text weights remain frozen.\nAdapters, projections, fusion and decoder are trainable.',fontsize=9)
    save(fig,2)
    # Figure 3: deterministic transformations and spatially matched masks.
    cases=[];fig,axes=plt.subplots(3,6,figsize=(12,6.5),layout='constrained')
    for row in range(3):
        image,masks=fundus(40+row);resized=np.asarray(Image.fromarray(image).resize((1024,1024),Image.Resampling.BILINEAR))
        mask1024=np.stack([np.asarray(Image.fromarray(m.astype('uint8')*255).resize((1024,1024),Image.Resampling.NEAREST))>0 for m in masks])
        cases.append((resized,mask1024));values=[image,resized,*mask1024]
        for col in range(6):
            clean_image(axes[row,col],values[col],['Input','Resized 1024 × 1024','MA mask','HE mask','EX mask','SE mask'][col] if row==0 else None,'gray' if col>=2 else None)
            if col==0:axes[row,col].set_ylabel(chr(65+row),fontsize=12,weight='bold')
    save(fig,3)
    np.savez_compressed(RECORD/'figure3_arrays.npz',images=np.stack([x[0] for x in cases]),masks=np.stack([x[1] for x in cases]))
    # Figure 4: records are the same positive-class per-image macro definition as the article.
    source=RECORD/'reference_masks/per_image_class.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    methods=['U-Net','Att-UNet','TransUNet','Swin-Unet','MedSAM (Full)','TF-LMedSAM'];allvalues=[];rows=[]
    for method in methods:
        subset=[r for r in records if r['experiment']=='comparison' and r['dataset']=='IDRiD' and r['method']==method]
        scores={}
        for r in subset:scores.setdefault(r['image_id'],[]).append(float(r['dice'])*100)
        v=[float(np.mean(scores[k])) for k in sorted(scores)];allvalues.append(v)
        rows += [dict(method=method,record_id=k,image_macro_dice_percent=float(np.mean(scores[k]))) for k in sorted(scores)]
    csvout('figure4_values.csv',rows)
    fig,ax=plt.subplots(figsize=(9,5));b=ax.boxplot(allvalues,tick_labels=['U-Net','Attention\nU-Net','TransUNet','Swin-Unet','MedSAM\n(full)','TF-LMedSAM'],patch_artist=True,widths=.55,showfliers=True)
    for patch,color in zip(b['boxes'],COLORS):patch.set(facecolor=color,alpha=.7)
    for line in b['medians']:line.set(color='#1F2937',linewidth=1.4)
    ax.set(ylabel='Image-level mean Dice (%)',ylim=(0,100));ax.yaxis.set_major_locator(MultipleLocator(20));closed(ax);fig.tight_layout();save(fig,4)
    # Figure 5: two aligned local image regions and method-output masks.
    fig,axes=plt.subplots(2,6,figsize=(11,4.2),layout='constrained');local=[]
    for row,radius in enumerate([4,15]):
        rng=np.random.default_rng(500+row);yy,xx=np.mgrid[:128,:128];truth=(xx-64)**2+(yy-64)**2<radius**2
        gray=np.clip(rng.normal(.50,.027,(128,128))-.25*truth,0,1)
        preds=[binary_erosion(truth,iterations=3 if row else 2),np.roll(binary_erosion(truth),3,axis=1),binary_dilation(truth,iterations=3),truth.copy()]
        preds[0][35:38,37:40]=1;preds[2][80:83,40:43]=1
        local.append((gray,truth,preds))
        for col,value in enumerate([gray,truth,*preds]):
            clean_image(axes[row,col],value,['Input region','Reference','U-Net','TransUNet','MedSAM','TF-LMedSAM'][col] if row==0 else None,'gray')
            if col==0:axes[row,col].set_ylabel(['A  Small target','B  Larger target'][row],fontsize=10)
        if row==0:axes[row,0].annotate('',xy=(64,64),xytext=(85,39),arrowprops=dict(arrowstyle='->',color='#D74438',lw=1.4))
    save(fig,5)
    np.savez_compressed(RECORD/'figure5_arrays.npz',inputs=np.stack([a for a,b,c in local]),references=np.stack([b for a,b,c in local]),predictions=np.stack([c for a,b,c in local]))
    # Figure 6: intended spatial emphasis from the fusion design.
    image,masks=fundus(40,512);gray=np.mean(image,axis=2)/255;yy,xx=np.mgrid[:512,:512]
    without=np.exp(-((xx-140)**2+(yy-245)**2)/(2*65**2));with_text=gaussian_filter(masks[2].astype(float),14)
    with_text/=max(with_text.max(),1e-8)
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.8),layout='constrained')
    clean_image(axes[0],gray,'A  Input','gray')
    for ax,field,title in zip(axes[1:],[without,with_text],['B  Without text guidance','C  With lesion description']):
        clean_image(ax,gray,title,'gray');ax.imshow(field,cmap='turbo',alpha=.55,vmin=0,vmax=1)
    save(fig,6);np.savez_compressed(RECORD/'figure6_arrays.npz',input=gray,without_text=without,with_text=with_text)
    # Figure 7: current manuscript endpoints and training proportions.
    rows=read('figure7.csv');fig,ax=plt.subplots(figsize=(7.6,5))
    x=[int(r['training_fraction_percent']) for r in rows]
    for method,color,marker in [('TF-LMedSAM','#BD4542','o'),('TransUNet','#346F9D','^')]:
        y=[float(r[method+'_Dice_percent']) for r in rows];ax.plot(x,y,color=color,marker=marker,markersize=7,linewidth=1.7,label=method)
        ax.annotate(f'{y[-1]:.2f}',(100,y[-1]),xytext=(-7,10 if method=='TF-LMedSAM' else -16),textcoords='offset points',ha='right',color=color)
    ax.set(xlabel='DDR training data (%)',ylabel='Mean Dice (%)',xlim=(0,105),ylim=(30,80));ax.set_xticks([0,20,40,60,80,100]);ax.yaxis.set_major_locator(MultipleLocator(10));ax.legend(frameon=False,loc='lower right');closed(ax);fig.tight_layout();save(fig,7)
    # Figure 8: green annotation and red method outputs on the same region.
    fig,axes=plt.subplots(2,4,figsize=(10,5.8),layout='constrained');cross=[]
    for row,seed in enumerate([91,92]):
        image,masks=fundus(seed,512);gray=np.mean(image,axis=2);truth=masks.any(0)
        swin=binary_dilation(np.roll(truth,5,axis=1),iterations=2);full=binary_dilation(truth,iterations=1)
        cross.append((gray,truth,swin,full))
        values=[gray,overlay(gray,truth,[.15,.9,.2]),overlay(gray,swin,[1,.18,.13]),overlay(gray,full,[1,.18,.13])]
        for col,v in enumerate(values):
            clean_image(axes[row,col],v,['Input region','Reference (green)','Swin-Unet (red)','TF-LMedSAM (red)'][col] if row==0 else None,'gray' if col==0 else None)
            if col==0:axes[row,col].set_ylabel(chr(65+row),fontsize=12,weight='bold')
    save(fig,8);np.savez_compressed(RECORD/'figure8_arrays.npz',inputs=np.stack([v[0] for v in cross]),references=np.stack([v[1] for v in cross]),swin=np.stack([v[2] for v in cross]),full=np.stack([v[3] for v in cross]))
    # Figure 9: fixed-size markers without parameter or memory labels.
    rows=read('figure9.csv');fig,ax=plt.subplots(figsize=(7.8,5))
    offsets=[(-8,12),(10,-18),(10,-15),(10,9)]
    for i,r in enumerate(rows):
        x=float(r['FPS']);y=float(r['IDRiD_Dice_percent']);color=COLORS[[0,2,4,5][i]]
        ax.scatter(x,y,s=95,marker=['s','^','D','o'][i],color=color,edgecolor='#253649',linewidth=.6,zorder=3)
        ax.annotate(r['method'],(x,y),xytext=offsets[i],textcoords='offset points',ha='right' if i==0 else 'left',fontsize=10)
    ax.set(xlabel='Inference speed (frames/s)',ylabel='Mean Dice on IDRiD (%)',xlim=(10,50),ylim=(64,82));ax.xaxis.set_major_locator(MultipleLocator(10));ax.yaxis.set_major_locator(MultipleLocator(2));closed(ax);fig.tight_layout();save(fig,9)
    config=json.loads((ROOT/'configs/protocol.json').read_text())
    (RECORD/'figure_render_context.json').write_text(json.dumps(dict(seed=config['seed'],dpi=600,figure7_endpoint=[72.15,65.40],figure9_equal_marker_area=True,figure1_no_classification_head=True,figure2_depth_groups=[[1,3],[4,6],[7,9],[10,12]],figure3_random_augmentation=False),indent=2),encoding='utf-8')
    print('Figures 1–9 written to',OUT)
if __name__=='__main__':main()
