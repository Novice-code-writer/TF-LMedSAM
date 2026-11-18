"""Verify a pipeline-check checkpoint and exported prediction masks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tflmedsam.engine import build,text_features,CheckData
from tflmedsam.io import read_csv,read_json,write_json,digest
from tflmedsam.metrics import measure
from tflmedsam import CLASSES

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    link=read_json(a.directory/'checkpoint_test_link.json')
    if digest(a.directory/link['checkpoint'])!=link['checkpoint_sha256']:raise ValueError('Checkpoint hash differs')
    state=torch.load(a.directory/link['checkpoint'],map_location='cpu',weights_only=True);c=state['context']
    if c['mode']!='smoke':raise ValueError('This checker is for the procedural pipeline-check path')
    model=build(c['config'],'smoke',c['text_width'],variant=c['variant']);model.load_state_dict(state['model']);model.eval()
    x,y,sid=CheckData('test')[0];text=text_features(a.directory/'text_features.npz',torch.device('cpu'))
    with torch.no_grad():pred=(torch.sigmoid(model(x[None],*text))[0]>=.5).numpy()
    rows=read_csv(a.directory/'per_image_class.csv')
    for k,label in enumerate(CLASSES):
        row=next(r for r in rows if r['image_id']==sid and r['lesion']==label)
        with Image.open(a.directory/row['prediction_path']) as f:saved=np.asarray(f)>0
        if not np.array_equal(pred[k],saved):raise ValueError('Checkpoint does not reproduce the saved masks')
        actual=measure(saved,y[k].numpy())
        for metric in ['tp','fp','fn','tn','dice','iou']:
            if abs(actual[metric]-float(row[metric]))>1e-10:raise ValueError('Prediction metric differs')
    write_json(a.directory/'verification.json',dict(checkpoint_hash_verified=True,predictions_reproduced=True,
       pixel_metrics_recomputed=True,selected_epoch=state['epoch'],input_size=c['input_size'],mode=c['mode']))
    print('Checkpoint, masks and pixel metrics verified:',a.directory)
if __name__=='__main__':
    torch.set_num_threads(2);main()
