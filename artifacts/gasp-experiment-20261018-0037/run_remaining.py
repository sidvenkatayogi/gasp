"""Run the two approved training arms sequentially, then both CPU analyses.

Run only after the shared baseline has completed and returned its budget ledger.
No automatic retries, configuration changes, or budget increases are performed.
"""
from datetime import datetime,timezone
import json,os
from pathlib import Path
import subprocess,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_modal_experiment import local_spend


def main():
    manifest=json.loads((HERE/'experiment.json').read_text())
    plan=json.loads((HERE/'results/paired-plan.json').read_text())
    assert plan['endpoint']==70 and plan['config']['role_steps']==35
    assert (HERE/'results/baseline/eval/step-000000/metrics.json').exists()
    state_path=HERE/'pipeline-status.json'
    def status(**values):
        record={'updated_at':datetime.now(timezone.utc).isoformat(),**values}
        state_path.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    for position,condition in enumerate(('unified','split')):
        complete=HERE/'results/train'/condition/'completion.json'
        if complete.exists():
            saved=json.loads(complete.read_text())
            assert saved['complete'] and saved['step']==70
            continue
        if (HERE/'results/train'/condition/'run.json').exists():
            raise RuntimeError('Existing partial initialization/training needs inspection before retry: '+condition)
        remaining=manifest['total_gpu_seconds']-local_spend(HERE,manifest)
        # Preserve 180s for the uncertain old disconnect cleanup tail. Every
        # worker separately charges 60s for lifecycle beyond its measured work.
        stage_count=2-position
        limit=min(10500 if stage_count==2 else 12600,
                  int((remaining-180)/stage_count)-60)
        if limit<600:
            raise RuntimeError('Insufficient total budget for a bounded training stage')
        command=[sys.executable,str(HERE/'run_detached.py'),'--directory',str(HERE),
            '--stage','train','--condition',condition,'--role-steps','35',
            '--max-seconds',str(limit)]
        status(stage='train',condition=condition,hard_timeout_seconds=limit,
               remaining_gpu_seconds_before_reservation=remaining,
               endpoint=70,source_sha256=manifest['source_sha256'])
        log=HERE/(condition+'-training-console.log')
        with log.open('a') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode or not complete.exists():
            status(stage='stopped_for_inspection',condition=condition,returncode=result.returncode,
                   log=str(log),note='No next GPU stage launched. Preserve checkpoints and inspect the attempt report.')
            raise SystemExit(1)
        saved=json.loads(complete.read_text())
        if not saved.get('complete') or saved['step']!=70:
            raise RuntimeError('Condition did not complete the frozen endpoint: '+condition)
    status(stage='cpu_analysis')
    comparison=HERE/'comparison'
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
    with (HERE/'analysis-console.log').open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_exploratory.py'),
            str(HERE/'results/baseline/eval/step-000000'),
            str(HERE/'results/train/unified/eval/step-000070'),
            str(HERE/'results/train/split/eval/step-000070'),str(comparison)],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(HERE/'analyze_quality_sensitivity.py'),str(comparison),
            str(HERE/'results/corpus-review.json'),str(HERE/'results/prepare/eval.jsonl.manifest.json')],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    status(stage='paired_endpoints_and_analyses_complete',comparison=str(comparison),
           charged_gpu_hours=local_spend(HERE,manifest)/3600,
           note='Inspect learning-signal and sampled training traces before the scientific interpretation.')

if __name__=='__main__':main()
