from pathlib import Path
import hashlib
import json
import math

root = Path('/Users/sidvenkatayogi/Documents/Code/gasp')
run = root / 'artifacts/gasp-modal-20261017-061436'
old = json.loads((root/'artifacts/gasp-vllm-20261017-053328/results/vllm.json').read_text())
old_cases = {c['label']: c for c in old['cases']}
small_generation = 2*sum(c['seconds'] for c in old['cases'] if c['label'].startswith('math_filter_fresh')) + sum(old_cases[k]['seconds'] for k in ('agent_corruption', 'agent_repair'))
large_generation = sum(old_cases[k]['seconds'] for k in ('agent_G64', 'polluter_G64', 'polluter_scoring'))
keys = ('old_logps_seconds', 'backward_seconds', 'repair_seconds', 'optimizer_seconds')
summary = {'basis': 'Identical backward fixtures; generation projection scales the original Colab workload by fixed-length diagnostic speed ratios. Not integrated training.', 'gpus': {}}
fixture_hashes = set()
for gpu, gpu_rate in [('A100-80GB', .000694), ('H100', .001097)]:
    p = run / gpu / 'results'
    worker = json.loads((p/'worker.json').read_text())
    assert worker['success']
    fixture_hashes.add(worker['fixture_sha256'])
    v = json.loads((p/'vllm-generated.json').read_text())
    assert v['success']
    cases = {c['label']:c for c in v['cases']}
    small_speed = old_cases['fixed_16x128']['seconds']/cases['fixed_16x128']['seconds']
    large_speed = old_cases['fixed_128x512']['seconds']/cases['fixed_128x512']['seconds']
    rate = (gpu_rate + 4*.0000131 + 64*.00000222)*3600
    elapsed = worker['finished_at']-worker['started_at']
    result = {'hourly_rate_minimum': rate, 'worker_seconds':elapsed,
              'worker_cost_at_requested_resources':elapsed*rate/3600,
              'fixed_128x512_tokens_per_second':cases['fixed_128x512']['tokens_per_second'],
              'fixed_16x128_tokens_per_second':cases['fixed_16x128']['tokens_per_second'],
              'inference_contexts':len(v['contexts']),
              'greedy_matches':sum(c['equal'] for c in v['greedy_comparison']),
              'engine_start_seconds':v['engine_start_seconds'], 'conditions':{}}
    for condition in ('unified', 'split'):
        b=json.loads((p/f'backward-{condition}.json').read_text())
        assert b['success'] and b['contexts']==4
        assert [c['samples'] for c in b['components']]==[256,256]
        assert [c['sample_tokens'] for c in b['components']]==[49515,209189]
        times={c['role']:sum(c[k] for k in keys) for c in b['components']}
        raw=(small_generation/small_speed+large_generation/large_speed+sum(times.values()))*1000/3600
        plan=[5*math.floor(raw*1.3/5), 5*math.ceil(raw*2.1/5)]
        result['conditions'][condition]={'update_compute_seconds':times,
            'raw_projected_hours':raw,'provisional_planning_hours':plan,
            'planning_cost_at_requested_resources':[rate*h for h in plan],
            'components':b['components']}
    summary['gpus'][gpu]=result
assert len(fixture_hashes)==1
summary['fixture_sha256']=fixture_hashes.pop()
summary['small_generation_seconds_colab_basis']=small_generation
summary['large_generation_seconds_colab_basis']=large_generation
summary['planning_multiplier']=[1.3,2.1]
hashes=json.loads((run/'source-sha256.json').read_text())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
(run/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({g:{k:v for k,v in d.items() if k!='conditions'}|{'conditions':{c:{k:v for k,v in x.items() if k!='components'} for c,x in d['conditions'].items()}} for g,d in summary['gpus'].items()},indent=2))
