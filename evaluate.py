import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tflmedsam.engine import build,text_features,evaluate
from tflmedsam.data import ManifestDataset
from tflmedsam.io import digest,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--backbone-checkpoint',type=Path,required=True);p.add_argument('--text-features',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--dataset',choices=['IDRiD','DDR'],required=True)
    p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Choose an empty output directory')
    device=torch.device(a.device);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);c=state['context']
    if c['mode']!='train':raise ValueError('Use a manifest-trained checkpoint for dataset evaluation')
    if digest(a.text_features)!=c['text_features_sha256']:raise ValueError('Text features differ from training')
    model=build(c['config'],'train',c['text_width'],a.backbone_checkpoint,c['variant']).to(device)
    model.load_state_dict(state['model'])
    ds=ManifestDataset(a.manifest,a.split,c['input_size'],a.dataset)
    result=evaluate(model,DataLoader(ds,batch_size=1),text_features(a.text_features,device),device,a.output)
    write_json(a.output/'evaluation_context.json',dict(checkpoint_sha256=digest(a.checkpoint),manifest_sha256=digest(a.manifest),
       source_manifest_sha256=c['manifest_sha256'],target_dataset=a.dataset,split=a.split,metrics=result))
if __name__=='__main__':main()
