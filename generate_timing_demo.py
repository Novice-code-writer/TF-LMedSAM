"""Create synthetic timing fixtures; never executes or measures neural models."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = [('U-Net', 45.0), ('TransUNet', 20.0), ('MedSAM (Full)', 15.0), ('TF-LMedSAM', 28.0)]
SEED = 20261127


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'records'
    output.mkdir(exist_ok=True)
    device = output / 'device_scenario.json'
    save(device, dict(record_type='synthetic_device_scenario', measured=False,
         cpu_label='Intel Core i5-4590 @ 3.30 GHz', execution_device_label='cpu',
         memory_capacity_label_GB=16, batch_size=1, input_shape=[1, 3, 1024, 1024],
         precision='float32', threads=4, interop_threads=1,
         note='Illustrative protocol labels, not a device performance measurement.',
         peak_memory_bytes=None, measured_power_watts=None, actual_model_executed=False))
    summaries=[]
    for index,(method,fps) in enumerate(METHODS,1):
        folder=output / f'method_{index}'
        folder.mkdir(exist_ok=True)
        state=folder / 'scenario_state.json'
        save(state, dict(record_type='synthetic_timing_generator_state', method=method,
             random_seed=SEED+index, target_fps=fps, repeats=5, samples_per_repeat=100,
             distribution='lognormal durations rescaled to target FPS',
             neural_checkpoint=False, neural_weights_available=False,
             actual_model_executed=False))
        rng=random.Random(SEED+index)
        raw=[math.exp(rng.gauss(0,0.025)) for _ in range(500)]
        factor=(500/fps)/math.fsum(raw)
        rows=[dict(record_type='synthetic_timing_sample', actual_model_executed=False,
             measurement_observed=False, method=method, run_id=i//100+1,
             iteration=i%100+1, batch_size=1, duration_seconds=v*factor,
             scenario_state_sha256=digest(state), device_scenario_sha256=digest(device))
             for i,v in enumerate(raw)]
        samples=folder / 'synthetic_execution.jsonl'
        samples.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        times=sorted(row['duration_seconds'] for row in rows)
        summary=dict(record_type='synthetic_benchmark_summary', method=method,
             actual_model_executed=False, measured_on_device=False,
             reference_fps=fps, computed_fps=500/math.fsum(times),
             latency_mean_ms=1000*math.fsum(times)/500,
             latency_median_ms=1000*(times[249]+times[250])/2,
             latency_p95_ms=1000*times[math.ceil(.95*500)-1],
             warmup_executed=False, intended_warmup_iterations=30,
             scenario_state_file=str(state.relative_to(ROOT)),
             scenario_state_sha256=digest(state),
             neural_checkpoint_path=None, neural_checkpoint_sha256=None,
             samples_file=str(samples.relative_to(ROOT)), samples_sha256=digest(samples),
             device_scenario_file=str(device.relative_to(ROOT)), device_scenario_sha256=digest(device))
        save(folder/'summary.json',summary)
        summaries.append(summary)
    save(output/'summary.json',dict(record_type='synthetic_timing_fixture_package',
         actual_models_executed=0, observed_device_measurements=0,
         original_checkpoints_recovered=0, methods=summaries))
    checks=[]
    for summary in summaries:
        samples=ROOT/summary['samples_file'];state=ROOT/summary['scenario_state_file']
        rows=[json.loads(line) for line in samples.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==500
        assert all(row['record_type']=='synthetic_timing_sample' and not row['actual_model_executed'] for row in rows)
        assert digest(samples)==summary['samples_sha256'] and digest(state)==summary['scenario_state_sha256']
        assert all(row['scenario_state_sha256']==digest(state) and row['device_scenario_sha256']==digest(device) for row in rows)
        fps=500/math.fsum(row['duration_seconds'] for row in rows)
        assert abs(fps-summary['reference_fps'])<1e-10
        checks.append(dict(method=summary['method'], sample_rows=len(rows), fps=fps,
                           state_hash_verified=True, sample_hash_verified=True))
    save(output/'verification.json',dict(passed=True, checks=checks,
         verification_scope='Generated fixture consistency only; not model or hardware verification.',
         actual_models_executed=0, observed_device_measurements=0))
    print(json.dumps(dict(methods=4, synthetic_samples=2000, verified=True,
                         actual_models_executed=0, output=str(ROOT)),ensure_ascii=False))


if __name__=='__main__':
    main()
