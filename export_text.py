"""Export frozen local CLIP-compatible text features with file provenance."""
import argparse
from pathlib import Path
import numpy as np
import torch
from tflmedsam import CLASSES
from tflmedsam.io import read_json,write_json,digest

def main():
    from transformers import CLIPTokenizer,CLIPTextModel
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--model-id',required=True);p.add_argument('--source-url',required=True)
    p.add_argument('--max-length',type=int,default=77)
    p.add_argument('--prompts',type=Path,default=Path('configs/prompts.json'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prompts=read_json(a.prompts)
    tokenizer=CLIPTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=CLIPTextModel.from_pretrained(a.model_dir,local_files_only=True).float().eval()
    for q in model.parameters():q.requires_grad_(False)
    encoded=tokenizer([prompts[c] for c in CLASSES],padding='max_length',max_length=a.max_length,truncation=True,return_tensors='pt')
    with torch.no_grad():features=model(**encoded)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,classes=np.array(CLASSES),token_features=features.last_hidden_state.numpy(),
      global_features=features.pooler_output.numpy(),token_valid=encoded['attention_mask'].numpy().astype(bool))
    files=[dict(file=str(f.relative_to(a.model_dir)),sha256=digest(f)) for f in sorted(a.model_dir.rglob('*')) if f.is_file()]
    write_json(a.output.with_suffix('.provenance.json'),dict(model_id=a.model_id,source_url=a.source_url,feature_sha256=digest(a.output),
      classes=CLASSES,prompts=prompts,files=files,torch=str(torch.__version__),tokenizer_class=type(tokenizer).__name__,max_length=a.max_length,text_hidden_size=int(features.last_hidden_state.shape[-1]),features_sha256=digest(a.output)))
    print(a.output)
if __name__=='__main__':main()
