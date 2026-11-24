"""Measure a manifest-trained checkpoint; no reference FPS is substituted."""
import argparse
import time
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from tflmedsam.io import write_json,write_csv,digest
from tflmedsam.timing import summarize_timing
from tflmedsam.engine import build,text_features

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--backbone-checkpoint',type=Path,required=True);p.add_argument('--text-features',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--method-label',required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--interop-threads',type=int,default=1)
    a=p.parse_args();device=torch.device(a.device);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);c=state['context']
    if a.repeats<1:raise ValueError('repeats must be positive')
    if min(a.threads,a.interop_threads)<1:raise ValueError('Thread counts must be positive')
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(a.interop_threads)
    if c['mode']!='train':raise ValueError('Profile requires a manifest-trained checkpoint')
    expected_labels={'baseline':'Baseline (Frozen MedSAM)','adapter':'MedSAM Adapter','text':'TF-LMedSAM (Text)','full':'TF-LMedSAM'}
    if a.method_label!=expected_labels.get(c['variant']):
        raise ValueError('Method label does not match this checkpoint architecture; U-Net, TransUNet and fully fine-tuned MedSAM require their own implementations')
    if c['input_size']!=1024:raise ValueError('Figure 9 protocol requires 1024 x 1024 inputs')
    if digest(a.backbone_checkpoint)!=c['backbone_checkpoint_sha256']:raise ValueError('Backbone checkpoint differs from training')
    if digest(a.text_features)!=c['text_features_sha256']:raise ValueError('Text features differ from training')
    model=build(c['config'],'train',c['text_width'],a.backbone_checkpoint,c['variant'])
    model.load_state_dict(state['model'],strict=True)
    model=model.to(device=device,dtype=torch.float32).eval()
    text=text_features(a.text_features,device);x=torch.zeros(1,3,c['input_size'],c['input_size'],device=device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    options=c['config']['profiling']
    samples=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for run in range(a.repeats):
            for _ in range(options['warmup']):model(x,*text)
            sync()
            for iteration in range(options['iterations']):
                sync();start=time.perf_counter();model(x,*text);sync();duration=time.perf_counter()-start
                samples.append(dict(run_id=run+1,iteration=iteration+1,batch_size=1,duration_seconds=duration))
    sample_file=a.output.with_name(a.output.stem+'_samples.csv')
    write_csv(sample_file,samples)
    result=dict(record_type='measured_benchmark',method=a.method_label,device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
      input_size=c['input_size'],batch_size=1,**summarize_timing(samples),total_parameters=sum(p.numel() for p in model.parameters()),
      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),checkpoint_sha256=digest(a.checkpoint),
      backbone_checkpoint_sha256=digest(a.backbone_checkpoint),text_features_sha256=digest(a.text_features),
      architecture_variant=c['variant'],training_mode=c['mode'],model_state_loaded_strictly=True,
      inference_peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
      inference_peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None,
      training_memory_measured=False,raw_samples_path=sample_file.name,raw_samples_sha256=digest(sample_file),
      environment=dict(platform=platform.platform(),processor=platform.processor(),python=sys.version,torch=torch.__version__,
        cuda_build_version=torch.version.cuda,cuda_used=device.type=='cuda',torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads()),
      timing_protocol=dict(warmup_per_run=options['warmup'],iterations_per_run=options['iterations'],
        repeats=a.repeats,timer='time.perf_counter',input='fixed zero tensor',precision='float32',
        scope='model forward with four lesion maps; cached text features',excludes=['disk access','preprocessing','initial text encoding']))
    write_json(a.output,result);print(a.output)
if __name__=='__main__':main()
