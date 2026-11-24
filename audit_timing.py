"""Audit measured benchmark outputs against raw samples and Figure 9 coordinates."""
import argparse
from pathlib import Path
from tflmedsam.io import read_json,read_csv,write_json,digest
from tflmedsam.timing import summarize_timing
from unified_records import DEFAULT


def audit(root,paths):
    frozen=read_json(root/'frozen_report.json')['sources']['figure9.csv']['rows']
    supplied={}
    for path in paths:
        result=read_json(path);method=result['method']
        if result.get('record_type')!='measured_benchmark':raise ValueError('Only measured benchmark records can support a timing audit')
        if method in supplied:raise ValueError('Duplicate benchmark method')
        raw=(path.parent/result['raw_samples_path']).resolve()
        if not raw.is_relative_to(path.parent.resolve()):raise ValueError('Timing sample path leaves benchmark directory')
        if digest(raw)!=result['raw_samples_sha256']:raise ValueError('Timing samples hash differs')
        actual=summarize_timing(read_csv(raw))
        if abs(actual['frames_per_second']-result['frames_per_second'])>1e-10:raise ValueError('Timing summary differs from raw measurements')
        supplied[method]=dict(path=str(path.resolve()),summary=actual,device=result['device'],environment=result['environment'])
    rows=[]
    for target in frozen:
        method=target['method'];record=supplied.get(method)
        rows.append(dict(method=method,reported_fps=float(target['FPS']),measurement_supplied=record is not None,
          rounded_fps_matches=None if record is None else f'{record["summary"]["frames_per_second"]:.2f}'==f'{float(target["FPS"]):.2f}',measurement=record))
    report=dict(rows=rows,all_methods_have_measurements=all(r['measurement_supplied'] for r in rows),
      comparison_rule='Match method, device, precision, dimensions, timing scope and software before pooling measurements.')
    write_json(root/'comment3/timing_audit.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--measurement',type=Path,action='append',default=[])
    a=p.parse_args();audit(a.root,a.measurement)
