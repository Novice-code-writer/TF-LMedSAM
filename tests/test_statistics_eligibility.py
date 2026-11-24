import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tflmedsam import CLASSES
from tflmedsam.io import read_json,write_json
from tflmedsam.metrics import measure
from tflmedsam.statistics import metric_matrix,image_macro,class_balanced,resampled_values,paired_statistics,holm_adjust
from unified_records import derive,DEFAULT


class EligibilityChecks(unittest.TestCase):
    def test_empty_reference_with_and_without_false_positives(self):
        y=np.zeros((4,4),bool);p=y.copy()
        self.assertIsNone(measure(p,y)['dice'])
        p[0,0]=True;m=measure(p,y)
        self.assertIsNone(m['dice']);self.assertEqual(m['fp'],1)
        self.assertTrue(m['negative_image_false_positive'])
        y[0,0]=True;self.assertEqual(measure(np.zeros_like(y),y)['dice'],0)

    def test_image_and_class_balanced_estimands_differ_with_missing_classes(self):
        x=np.array([[1,np.nan],[.5,.2],[np.nan,.8],[np.nan,np.nan]])
        self.assertTrue(np.allclose(image_macro(x)[:3],[1,.35,.8]))
        self.assertTrue(np.isnan(image_macro(x)[3]))
        self.assertAlmostEqual(class_balanced(x),.625)
        self.assertNotAlmostEqual(np.nanmean(image_macro(x)),class_balanced(x))

    def test_bootstrap_requires_original_class_coverage(self):
        x=np.array([[1,np.nan],[.5,.2],[np.nan,.8]])
        values=resampled_values(x,np.array([[0,0,0],[0,1,2]]))
        self.assertTrue(np.isnan(values[0]));self.assertAlmostEqual(values[1],.625)
        self.assertIsNone(class_balanced(np.full((2,4),np.nan)))

    def test_pairing_eligibility_and_all_zero_difference(self):
        x=np.array([[1,np.nan],[.5,.2],[np.nan,.8],[np.nan,np.nan]])
        r=paired_statistics(x,x,np.random.default_rng(4))
        self.assertEqual(r['p_raw'],1);self.assertEqual(r['ci95_pp'],[0,0])
        self.assertEqual(r['eligible_paired_images'],3)
        y=x.copy();y[0,1]=0
        with self.assertRaises(ValueError):paired_statistics(x,y,np.random.default_rng(4))

    def test_holm_family_isolation(self):
        r=holm_adjust([dict(family='a',p_raw=.01),dict(family='a',p_raw=.02),dict(family='b',p_raw=.03)])
        self.assertEqual([v['p_holm'] for v in r],[.02,.02,.03])

    def test_derive_end_to_end_with_partial_and_entirely_empty_images(self):
        freeze=copy.deepcopy(read_json(DEFAULT/'frozen_report.json'))
        rows=[]
        for spec in freeze['conditions']:
            spec['dice']=None;spec['iou']=None
            for i in range(4):
                for c,lesion in enumerate(CLASSES):
                    y=np.zeros((4,4),bool);p=y.copy()
                    if i<3 and c<2 and not(i==0 and c==1):y[0,0]=True;p[0,0]=True
                    if i==3:p[0,1]=True
                    rows.append(dict(result_id=spec['id'],image_id=f'{spec["dataset"]}_{i}',lesion=lesion,**measure(p,y)))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write_json(root/'frozen_report.json',freeze);write_json(root/'per_image_class.json',{'rows':rows})
            derive(root)
            image_rows=read_json(root/'per_image.json')['rows']
            self.assertEqual(sum(r['image_macro_dice_percent'] is None for r in image_rows),len(freeze['conditions']))
            self.assertTrue(all(r['n']==3 and r['excluded_images']==1 for r in read_json(root/'distributions.json')['rows']))
            self.assertTrue(all(r['p_raw']==1 for r in read_json(root/'paired_comparisons.json')['rows']))

if __name__=='__main__':unittest.main()
