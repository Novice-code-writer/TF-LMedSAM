"""Generate explicitly non-measured timing examples for the reporting pipeline."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from tflmedsam.timing import summarize_timing

from project_paths import REPO as BASE, UNIFIED as ROOT
OUT=ROOT/'comment4/reference_timing'

def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    targets=json.loads((ROOT/'frozen_report.json').read_text(encoding='utf-8'))['sources']['figure9.csv']['rows']
    summaries=[]
    for index,target in enumerate(targets):
        method=target['method'];fps=float(target['FPS']);rng=np.random.default_rng(20261112+index)
        # Positive variation is an example construction rule, not an estimate of this CPU.
        durations=np.exp(rng.normal(0,.025, (5,100))+rng.normal(0,.01,(5,1)))
        durations*=500/fps/durations.sum()
        samples=[dict(run_id=r+1,iteration=i+1,batch_size=1,duration_seconds=float(durations[r,i]))
                 for r in range(5) for i in range(100)]
        prefix=f'method_{index+1}'
        rows=[dict(record_type='reference_timing_example',method=method,**row) for row in samples]
        raw=OUT/f'{prefix}_samples.csv'
        with raw.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        stats=summarize_timing(samples)
        assert abs(stats['frames_per_second']-fps)<1e-10
        result=dict(record_type='reference_timing_example',method=method,actual_model_executed=False,
            measured_on_current_host=False,device=None,hardware_attribution=None,
            reference_fps=fps,construction_seed=20261112+index,
            construction='positive lognormal examples normalized to the frozen reference FPS',
            example_summary=stats,
            interval_scope='variation within generated examples; not measured hardware uncertainty',
            raw_samples_path=raw.name,raw_samples_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            intended_protocol='../benchmark_protocol.json')
        dump(OUT/f'{prefix}_summary.json',result)
        summaries.append(dict(method=method,reference_fps=fps,rows=len(samples),summary=f'{prefix}_summary.json',
            record_type='reference_timing_example',measured_on_current_host=False))
    dump(OUT/'index.json',dict(record_type='reference_timing_example',models_executed=0,
        methods=summaries,total_example_rows=2000,actual_measured_fps_available=False))
    # Test the provenance boundary with the actual audit entry point.
    from audit_timing import audit
    rejected=0
    for i in range(1,5):
        try:audit(ROOT,[OUT/f'method_{i}_summary.json'])
        except ValueError as exc:
            assert 'Only measured benchmark records' in str(exc);rejected+=1
        else:raise AssertionError('Example accepted as measured evidence')
    dump(OUT/'verification.json',dict(passed=True,example_rows=2000,methods=4,
        fps_arithmetic_checks=4,reference_records_rejected_by_measured_audit=rejected,
        actual_models_executed=0))
    print(json.dumps(dict(example_methods=4,example_rows=2000,measurement_audit_rejections=rejected,
        actual_models_executed=0,output=str(OUT)),ensure_ascii=False))

if __name__=='__main__':main()
