"""Validate measured iteration records and summarize independent benchmark runs."""
import numpy as np


def summarize_timing(rows,seed=20261112,replicates=2000):
    if not rows:raise ValueError('No timing samples supplied')
    keys=set();runs={}
    for row in rows:
        key=(int(row['run_id']),int(row['iteration']))
        seconds=float(row['duration_seconds']);images=int(row['batch_size'])
        if key in keys:raise ValueError('Duplicate timing iteration')
        if not np.isfinite(seconds) or seconds<=0 or images<1:raise ValueError('Invalid timing duration or batch size')
        keys.add(key);runs.setdefault(key[0],[]).append((seconds,images))
    totals=np.array([[sum(v[0] for v in group),sum(v[1] for v in group)] for _,group in sorted(runs.items())])
    elapsed=totals[:,0].sum();images=totals[:,1].sum();latencies=np.array([float(r['duration_seconds']) for r in rows])*1000
    result=dict(frames_per_second=float(images/elapsed),elapsed_seconds=float(elapsed),processed_images=int(images),
      measured_iterations=len(rows),benchmark_runs=len(runs),median_latency_ms=float(np.median(latencies)),p95_latency_ms=float(np.percentile(latencies,95)),
      fps_ci95=None,interval_resampling_unit='benchmark run',bootstrap_replicates=replicates)
    if len(runs)>=2:
        indices=np.random.default_rng(seed).integers(0,len(runs),(replicates,len(runs)))
        sums=totals[indices].sum(axis=1);fps=sums[:,1]/sums[:,0]
        result['fps_ci95']=[float(x) for x in np.quantile(fps,[.025,.975])]
    return result
