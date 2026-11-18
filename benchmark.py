"""Measure a manifest-trained checkpoint; no reference FPS is substituted."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from tflmedsam.io import write_json,digest
from tflmedsam.engine import build,text_features

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--backbone-checkpoint',type=Path,required=True);p.add_argument('--text-features',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();device=torch.device(a.device);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);c=state['context']
    if c['mode']!='train':raise ValueError('Profile requires a manifest-trained checkpoint')
    if digest(a.text_features)!=c['text_features_sha256']:raise ValueError('Text features differ from training')
    model=build(c['config'],'train',c['text_width'],a.backbone_checkpoint,c['variant']).to(device).eval()
    text=text_features(a.text_features,device);x=torch.zeros(1,3,c['input_size'],c['input_size'],device=device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    options=c['config']['profiling']
    with torch.no_grad():
        for _ in range(options['warmup']):model(x,*text)
        sync()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for _ in range(options['iterations']):
            sync();start=time.perf_counter();model(x,*text);sync();times.append(time.perf_counter()-start)
    result=dict(device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
      input_size=c['input_size'],batch_size=1,frames_per_second=len(times)/sum(times),median_latency_ms=float(np.median(times)*1000),
      p95_latency_ms=float(np.percentile(times,95)*1000),total_parameters=sum(p.numel() for p in model.parameters()),
      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),checkpoint_sha256=digest(a.checkpoint),
      inference_peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
      inference_peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None,
      training_memory_measured=False)
    write_json(a.output,result);print(a.output)
if __name__=='__main__':main()
