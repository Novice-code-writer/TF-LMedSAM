import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tflmedsam.model import Adapter,AdaptedFFN,CheckEncoder,TF_LMedSAM,loss_function
from tflmedsam.metrics import measure,summarize
from tflmedsam.data import audit
from tflmedsam.io import write_csv

class Contract(unittest.TestCase):
    def test_overlap_and_empty_reference(self):
        result=measure(np.array([[1,1],[0,0]]),np.array([[1,0],[1,0]]))
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['tn']),(1,1,1,1))
        self.assertEqual(result['dice'],.5);self.assertAlmostEqual(result['iou'],1/3)
        empty=measure(np.ones((2,2)),np.zeros((2,2)))
        self.assertIsNone(empty['dice']);self.assertTrue(empty['negative_image_false_positive'])

    def test_zero_adapter_and_frozen_encoder(self):
        torch.manual_seed(3);ffn=torch.nn.Linear(32,32);adapted=AdaptedFFN(copy.deepcopy(ffn),32,8);x=torch.randn(2,4,32)
        torch.testing.assert_close(ffn(x),adapted(x))
        model=TF_LMedSAM(CheckEncoder(),24,channels=32,rank=8,heads=4,layers=2)
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if n.startswith(('patch_embed','pos_embed','blocks')) and '.adapter.' not in n))
        y=model(torch.rand(1,3,64,64),torch.randn(4,8,24),torch.randn(4,24),torch.ones(4,8,dtype=torch.bool))
        self.assertEqual(tuple(y.shape),(1,4,64,64))
        loss=loss_function(y,torch.randint(0,2,y.shape).float());loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.adapter.up.' in n))

    def test_decoder_coarse_then_refine(self):
        model=TF_LMedSAM(CheckEncoder(),24,32,8,4,2);seen=[]
        hook=model.refinement.register_forward_pre_hook(lambda m,x:seen.append(tuple(x[0].shape)))
        model(torch.rand(1,3,64,64),torch.randn(4,8,24),torch.randn(4,24),torch.ones(4,8,dtype=torch.bool));hook.remove()
        self.assertEqual(seen,[(1,33,8,8)]*4)

    def test_cross_split_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.fromarray(np.zeros((8,8,3),np.uint8)).save(root/'image.png')
            Image.fromarray(np.zeros((8,8),np.uint8)).save(root/'mask.png')
            rows=[dict(dataset='IDRiD',image_id='case'+str(i),split=s,patient_id='',image_path='image.png',**{c+'_path':'mask.png' for c in ['MA','HE','EX','SE']}) for i,s in enumerate(['train','test'])]
            write_csv(root/'manifest.csv',rows)
            with self.assertRaises(ValueError):audit(root/'manifest.csv',root/'audit',size=8)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
