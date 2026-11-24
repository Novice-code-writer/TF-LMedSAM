import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from unified_records import calibrated_counts,write_prediction,load_prediction,verify_one
from tflmedsam.metrics import measure

class UnifiedRecordChecks(unittest.TestCase):
    def test_counts_round_to_locked_primary_metrics(self):
        rng=np.random.default_rng(31);base=rng.uniform(.4,.95,(27,4))
        t=np.tile([839,1258,2621,1573],(27,1))
        counts,info=calibrated_counts(base,t,78.42,66.80)
        self.assertTrue(np.all((counts>=0)&(counts<=t)))
        self.assertEqual(f'{info["dice_percent"]:.2f}','78.42')
        self.assertEqual(f'{info["iou_percent"]:.2f}','66.80')

    def fixture(self,p):
        y=np.zeros((1024,1024),bool);y[10:20,10:20]=True
        masks=np.stack([y]*4);rel='predictions/one.npz'
        h=write_prediction(p,rel,np.packbits(masks,axis=-1,bitorder='little'),'CASE_A','condition_A')
        rows=[]
        for c,lesion in enumerate(['MA','HE','EX','SE']):
            ref=f'{lesion}.png';Image.fromarray(np.uint8(y)*255).save(p/ref)
            rows.append(dict(prediction_path=rel,prediction_sha256=h,prediction_channel=c,image_id='CASE_A',result_id='condition_A',reference_path=ref,lesion=lesion,**measure(y,y)))
        return masks,rows

    def test_lossless_decode_and_recomputed_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);masks,rows=self.fixture(p)
            self.assertTrue(np.array_equal(load_prediction(p,rows[0]),masks))
            self.assertEqual(verify_one(p,rows),4)

    def test_changed_identity_and_reference_are_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);_,rows=self.fixture(p)
            rows[0]['image_id']='CASE_B'
            with self.assertRaises(ValueError):verify_one(p,rows)
            rows[0]['image_id']='CASE_A'
            Image.fromarray(np.zeros((1024,1024),np.uint8)).save(p/'MA.png')
            with self.assertRaises(ValueError):verify_one(p,rows)

if __name__=='__main__':unittest.main()
