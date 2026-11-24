import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from tflmedsam.io import write_csv,digest
from tflmedsam.metrics import measure
from verify_masks import verify,local_file

class MaskRecordVerification(unittest.TestCase):
    def make_records(self,p,empty=False,fp=False):
        target=np.zeros((8,8),bool)
        if not empty:target[2:5,2:5]=True
        pred=target.copy()
        if fp:pred[0,0]=True
        for name,value in [('reference.png',target),('prediction.png',pred)]:Image.fromarray(np.uint8(value)*255).save(p/name)
        row=dict(image_id='CASE_01',lesion='MA',reference_path='reference.png',prediction_path='prediction.png',prediction_sha256=digest(p/'prediction.png'),**measure(pred,target))
        write_csv(p/'per_image_class.csv',[row]);return row

    def test_positive_recomputation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.make_records(p);r=verify(p)
            self.assertTrue(r['passed']);self.assertEqual(r['summary'][0]['macro']['dice'],1.)

    def test_empty_reference_with_and_without_false_positive(self):
        for fp in [False,True]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);self.make_records(p,empty=True,fp=fp);r=verify(p)
                self.assertTrue(r['passed']);self.assertIsNone(r['summary'][0]['macro']['dice'])
                self.assertEqual(r['summary'][0]['per_class'][0]['negative_image_false_positive_rate'],float(fp))

    def test_changed_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.make_records(p);Image.fromarray(np.zeros((8,8),np.uint8)).save(p/'prediction.png')
            self.assertFalse(verify(p)['passed'])

    def test_duplicate_identifier_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);row=self.make_records(p);write_csv(p/'per_image_class.csv',[row,row])
            self.assertFalse(verify(p)['passed'])

    def test_path_must_stay_in_record_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):local_file(Path(d),'../outside.png')

if __name__=='__main__':unittest.main()
