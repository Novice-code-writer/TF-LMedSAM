import unittest
from tflmedsam.timing import summarize_timing


class TimingChecks(unittest.TestCase):
    def test_weighted_throughput_and_run_intervals(self):
        rows=[dict(run_id=1,iteration=1,batch_size=1,duration_seconds=.1),dict(run_id=2,iteration=1,batch_size=1,duration_seconds=.2)]
        r=summarize_timing(rows)
        self.assertAlmostEqual(r['frames_per_second'],2/.3)
        self.assertEqual(r['fps_ci95'],[5,10])
        self.assertEqual(r['benchmark_runs'],2)

    def test_invalid_duplicate_or_absent_measurements(self):
        row=dict(run_id=1,iteration=1,batch_size=1,duration_seconds=.1)
        with self.assertRaises(ValueError):summarize_timing([])
        with self.assertRaises(ValueError):summarize_timing([row,row])
        with self.assertRaises(ValueError):summarize_timing([{**row,'duration_seconds':0}])
        self.assertIsNone(summarize_timing([row])['fps_ci95'])

if __name__=='__main__':unittest.main()
