"""Training, checkpoint selection, and mask-level evaluation."""
import argparse
import json
import logging
import platform
import random
import sys
import shutil
from importlib import metadata
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,Subset,Dataset
from . import CLASSES
from .io import read_json,write_json,write_csv,digest,safe_id
from .data import ManifestDataset,audit
from .model import TF_LMedSAM,CheckEncoder,FrozenLinearSegmenter,loss_function
from .metrics import measure,summarize

class CheckData(Dataset):
    def __init__(self,split,size=64):self.split=split;self.size=size
    def __len__(self):return 3 if self.split=='train' else 1
    def __getitem__(self,index):
        seed=20261112+index+{'train':0,'val':100,'test':200}[self.split]
        g=np.random.default_rng(seed);h=self.size;yy,xx=np.mgrid[:h,:h]
        masks=np.stack([((xx-(15+7*c))**2+(yy-(18+5*c))**2)<(4+c)**2 for c in range(4)])
        image=g.uniform(.15,.3,(3,h,h)).astype(np.float32);image+=masks.any(0)[None]*.3
        return torch.tensor(image),torch.tensor(masks,dtype=torch.float32),f'CHECK_{self.split}_{index:03d}'

def text_features(path,device):
    with np.load(path,allow_pickle=False) as z:
        if z['classes'].tolist()!=list(CLASSES):raise ValueError('Text-feature class order differs')
        values=tuple(torch.from_numpy(z[k].copy()).to(device) for k in ['token_features','global_features','token_valid'])
    tokens,global_values,valid=values
    if tokens.ndim!=3 or tokens.shape[0]!=4 or global_values.shape!=(4,tokens.shape[-1]) or valid.shape!=tokens.shape[:2]:
        raise ValueError('Invalid text-feature shapes')
    if not valid.any(1).all():raise ValueError('Each prompt must contain valid tokens')
    return tokens.float(),global_values.float(),valid.bool()

def build(config,mode,text_width,checkpoint=None,variant='full'):
    if mode=='smoke':encoder=CheckEncoder();c,r,h,l=32,8,4,2
    else:
        from segment_anything import sam_model_registry
        encoder=sam_model_registry['vit_b'](checkpoint=str(checkpoint)).image_encoder
        if encoder.pos_embed.shape[-1]!=768:raise ValueError('ViT-B hidden width must be 768')
        c=config['model']['fusion_channels'];r=config['model']['adapter_rank'];h=config['model']['attention_heads'];l=config['model']['decoder_layers']
    if variant in ['baseline','adapter']:
        return FrozenLinearSegmenter(encoder,rank=r,use_adapter=variant=='adapter')
    return TF_LMedSAM(encoder,text_width,c,r,h,l,use_adapter=True,use_text=True)

def evaluate(model,loader,text,device,out=None):
    model.eval();rows=[]
    with torch.no_grad():
        for x,y,ids in loader:
            pred=(torch.sigmoid(model(x.to(device),*text))>=.5).cpu().numpy()
            for b,sid in enumerate(ids):
                safe_id(sid)
                for c,label in enumerate(CLASSES):
                    row=dict(image_id=sid,lesion=label,**measure(pred[b,c],y[b,c].numpy()))
                    if out:
                        file=Path(out)/'predictions'/sid/(label+'.png');file.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(np.uint8(pred[b,c])*255).save(file)
                        row.update(prediction_path=file.relative_to(out).as_posix(),prediction_sha256=digest(file))
                    rows.append(row)
    result=summarize(rows)
    if out:write_csv(Path(out)/'per_image_class.csv',rows);write_json(Path(out)/'metrics.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/protocol.json'))
    p.add_argument('--mode',choices=['smoke','train'],required=True);p.add_argument('--manifest',type=Path)
    p.add_argument('--dataset',choices=['IDRiD','DDR']);p.add_argument('--backbone-checkpoint',type=Path)
    p.add_argument('--text-features',type=Path);p.add_argument('--train-ids',type=Path)
    p.add_argument('--variant',choices=['baseline','adapter','text','full'],default='full')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='auto');p.add_argument('--epochs',type=int)
    a=p.parse_args();cfg=read_json(a.config);seed=cfg['seed']
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Choose an empty run directory')
    if a.mode=='train' and not all([a.manifest,a.dataset,a.backbone_checkpoint,a.text_features]):
        p.error('train requires manifest, dataset, backbone-checkpoint and text-features')
    for path in [a.manifest,a.backbone_checkpoint,a.text_features,a.train_ids]:
        if path is not None and not path.is_file():raise FileNotFoundError(path)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    a.output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('train');logger.setLevel(logging.INFO)
    handler=logging.FileHandler(a.output/'train.log',encoding='utf-8');handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler);logger.addHandler(logging.StreamHandler())
    if a.mode=='smoke':
        size=64;batch=2;accum=2;epochs=a.epochs or 2;every=1
        g=torch.Generator().manual_seed(seed)
        text=(torch.randn(4,8,24,generator=g,device='cpu').to(device),torch.randn(4,24,generator=g).to(device),torch.ones(4,8,dtype=torch.bool,device=device))
        datasets={s:CheckData(s) for s in ['train','val','test']};origin='procedural_pipeline_check'
        np.savez_compressed(a.output/'text_features.npz',classes=np.array(CLASSES),
           token_features=text[0].cpu().numpy(),global_features=text[1].cpu().numpy(),token_valid=text[2].cpu().numpy())
    else:
        size=cfg['preprocessing']['input_size'];batch=cfg['training']['batch_size'];accum=cfg['training']['accumulation_steps']
        epochs=a.epochs or cfg['training']['epochs'];every=cfg['training']['validate_every']
        audit(a.manifest,a.output/'data_audit',size,{a.dataset:cfg['datasets'][a.dataset]})
        datasets={s:ManifestDataset(a.manifest,s,size,a.dataset) for s in ['train','val','test']}
        if a.train_ids:
            chosen=set(a.train_ids.read_text(encoding='utf-8-sig').splitlines());train=datasets['train']
            indices=[i for i,r in enumerate(train.rows) if r['image_id'] in chosen]
            if not chosen or len(indices)!=len(chosen):raise ValueError('Training subset contains unknown IDs')
            datasets['train']=Subset(train,indices)
        text=text_features(a.text_features,device);origin='manifest_training'
        if not a.text_features.with_suffix('.provenance.json').is_file():raise ValueError('Exported text features require a provenance sidecar')
        shutil.copy2(a.manifest,a.output/'source_manifest.csv')
        shutil.copy2(a.text_features.with_suffix('.provenance.json'),a.output/'text_encoder_provenance.json')
        if a.train_ids:shutil.copy2(a.train_ids,a.output/'selected_training_ids.txt')
        text_record=read_json(a.text_features.with_suffix('.provenance.json'))
        if text_record.get('feature_sha256')!=digest(a.text_features):raise ValueError('Text-feature provenance hash differs')
        expected_prompts=read_json(Path(__file__).resolve().parents[1]/'configs/prompts.json')
        if text_record.get('prompts')!=expected_prompts:raise ValueError('Cached prompt descriptions differ from the configured four descriptions')
        if not text_record.get('model_id') or not text_record.get('source_url'):raise ValueError('Text encoder identity and source URL are required')
    if epochs<1:raise ValueError('epochs must be positive')
    model=build(cfg,a.mode,text[0].shape[-1],a.backbone_checkpoint,a.variant).to(device)
    gen=torch.Generator().manual_seed(seed)
    loaders={s:DataLoader(d,batch_size=batch if s=='train' else 1,shuffle=s=='train',num_workers=0,generator=gen) for s,d in datasets.items()}
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=cfg['training']['min_lr'])
    context=dict(mode=a.mode,record_origin=origin,python=sys.version,platform=platform.platform(),torch=str(torch.__version__),
        cuda_runtime=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        seed=seed,input_size=size,batch_size=batch,accumulation_steps=accum,epochs=epochs,variant=a.variant,
        split_counts={s:len(d) for s,d in datasets.items()},text_width=text[0].shape[-1],
        manifest_path=str(a.manifest.resolve()) if a.manifest else None,
        backbone_checkpoint_sha256=digest(a.backbone_checkpoint) if a.backbone_checkpoint else None,
        text_features_sha256=digest(a.text_features) if a.text_features else digest(a.output/'text_features.npz'),
        manifest_sha256=digest(a.manifest) if a.manifest else None,
        train_ids_sha256=digest(a.train_ids) if a.train_ids else None,
        config=cfg,code_sha256={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')})
    context['dependencies']={}
    for name in ['numpy','scipy','Pillow','torch','torchvision','matplotlib','transformers','segment-anything']:
        try:context['dependencies'][name]=metadata.version(name)
        except metadata.PackageNotFoundError:context['dependencies'][name]=None
    try:context['backbone_install_provenance']=json.loads(metadata.distribution('segment-anything').read_text('direct_url.json') or 'null')
    except metadata.PackageNotFoundError:context['backbone_install_provenance']=None
    snapshot=a.output/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,snapshot/file.name)
    write_json(a.output/'run_context.json',context)
    groups={}
    for name,param in model.named_parameters():
        key='adapters' if '.adapter.' in name else name.split('.')[0]
        item=groups.setdefault(key,dict(total=0,trainable=0));item['total']+=param.numel();item['trainable']+=param.numel() if param.requires_grad else 0
    write_json(a.output/'parameter_counts.json',groups)
    best=-1.;history=[];loss_options=dict(cfg['loss'])
    if a.variant!='full':loss_options['beta']=0.
    logger.info('mode=%s device=%s split_counts=%s',a.mode,device,context['split_counts'])
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad();values=[]
        batches=len(loaders['train'])
        for i,(x,y,_) in enumerate(loaders['train']):
            loss=loss_function(model(x.to(device),*text),y.to(device),**loss_options)
            actual=min(accum,batches-(i//accum)*accum)
            (loss/actual).backward();values.append(float(loss.detach()))
            if (i+1)%accum==0 or i+1==batches:
                if any(not torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None):raise ValueError('Nonfinite gradients')
                opt.step();opt.zero_grad()
        row=dict(epoch=epoch,train_loss=float(np.mean(values)),learning_rate=opt.param_groups[0]['lr'],val_dice=None,val_iou=None)
        improved=False
        if epoch%every==0 or epoch==epochs:
            val=evaluate(model,loaders['val'],text,device)['macro'];row.update(val_dice=val['dice'],val_iou=val['iou'])
            if val['dice'] is None:raise ValueError('Validation requires at least one positive class')
            if val['dice']>best:best=val['dice'];improved=True
        scheduler.step();history.append(row);write_csv(a.output/'history.csv',history)
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best_val_dice=best,context=context)
        torch.save(state,a.output/'last_model.pth')
        if improved:torch.save(state,a.output/'best_model.pth')
        logger.info('epoch=%d loss=%.6f val_dice=%s val_iou=%s',epoch,row['train_loss'],row['val_dice'],row['val_iou'])
    state=torch.load(a.output/'best_model.pth',map_location=device,weights_only=True);model.load_state_dict(state['model'])
    result=evaluate(model,loaders['test'],text,device,a.output)
    write_json(a.output/'checkpoint_test_link.json',dict(checkpoint='best_model.pth',checkpoint_sha256=digest(a.output/'best_model.pth'),
      selected_epoch=state['epoch'],record_origin=origin,manifest_sha256=context['manifest_sha256'],text_features_sha256=context['text_features_sha256'],metrics=result))
    logger.info('test=%s',result['macro']);print(a.output)

if __name__=='__main__':main()
