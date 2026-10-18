"""Bounded preprocessing correction; reuses the frozen worker's budget/lifecycle.
No training algorithm or frozen source file is changed. No API secret is uploaded.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import zipfile



def redraw(stage, condition, root, inputs, max_seconds, commit, **unused):
    import time
    import torch
    from gasp.config import Config
    from gasp.data import read_cache, write_json, cut_window, sha256
    from gasp.models import load_model, Rollouts, model_digest
    from gasp.prompts import Prompts
    from gasp.verification import correct, reasoning
    started = time.monotonic()
    policy = json.loads(Path('/redraw-policy.json').read_text())
    rows, manifest = read_cache(root / 'prepare/eval-contexts.jsonl', 'eval_contexts')
    if manifest['sha256'] != policy['original_contexts_sha256']:
        raise ValueError('Correction policy belongs to a different original cache')
    if (root / 'paired-plan.json').exists():
        raise ValueError('Source correction must precede frozen baseline and training')
    out = root / 'trace-redraw' / ('seed-' + str(policy['seed']))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'policy.json', policy)
    cfg = Config(**manifest['provenance']['config']).with_changes(seed=policy['seed'])
    model, tok = load_model(cfg)
    digest = model_digest(model)
    if digest != manifest['provenance']['initial_model_sha256']:
        raise ValueError('Initial checkpoint differs')
    prompts, rollouts = Prompts(tok, cfg), Rollouts(tok, cfg)
    lookup = {r['question_id']: r for r in rows}
    selected = [lookup[q] for q in policy['question_ids']]
    with (out / 'candidates.jsonl').open('x') as stream:
        for candidate_index in range(policy['candidates_per_question']):
            if time.monotonic() - started > max_seconds - 90:
                raise TimeoutError('Bounded trace draw exceeded soft deadline')
            groups = rollouts.generate_many(model,
                [prompts.agent(r['question']) for r in selected], 1,
                cfg.completion_tokens, 'semantic_trace_redraw', temperature=.7)
            if len(groups) != len(selected) or any(len(g) != 1 for g in groups):
                raise ValueError('Incorrect sample count')
            for row, group in zip(selected, groups):
                sample = group[0]
                trace = reasoning(sample.text)
                candidate = {'question_id': row['question_id'], 'question': row['question'],
                    'answer': row['answer'], 'candidate_index': candidate_index,
                    'text': sample.text, 'numeric_correct': correct(sample.text, row['answer'], cfg.verifier),
                    'reasoning_nonempty': bool(trace), 'generated_tokens': len(sample.tokens),
                    'cuts': [{'alpha': a, 'prefix': cut_window(trace, tok, a, cfg.window_cap)[0],
                              'clean': cut_window(trace, tok, a, cfg.window_cap)[1]}
                             for a in (0., .25, .5, .75)] if trace else []}
                stream.write(json.dumps(candidate) + '\n')
                print(json.dumps({k: candidate[k] for k in ('question_id', 'candidate_index', 'numeric_correct', 'generated_tokens')}), flush=True)
            stream.flush()
            commit()
    result = {'complete': True, 'initial_model_sha256': digest, 'counts': rollouts.counts,
        'seconds': time.monotonic()-started, 'gpu': torch.cuda.get_device_name(),
        'candidate_sha256': sha256(out / 'candidates.jsonl'), 'semantic_review': 'pending'}
    write_json(out / 'result.json', result)
    commit()
    return result


def redraw_worker(*args):
    # The original worker remains responsible for timeout, durable ledger and logs.
    # Only its stage dispatch is replaced for this preprocessing-only operation.
    root = Path('/work/gasp')
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    from run_modal_experiment import safe_extract, worker
    with zipfile.ZipFile('/source.zip') as archive:
        safe_extract(archive, root)
    import experiment_worker
    experiment_worker.run = redraw
    return worker(*args)


def main():
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[1]
    import modal
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_modal_experiment import claim_local_lock, local_spend, verify_launch_sources, safe_extract
    lock = claim_local_lock(HERE)
    manifest = json.loads((HERE / 'experiment.json').read_text())
    archive = HERE / 'source.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Frozen archive changed')
    verify_launch_sources(ROOT, archive)
    limit = 600
    if local_spend(HERE, manifest) + limit + 60 > manifest['total_gpu_seconds']:
        raise ValueError('Insufficient remaining budget')
    cfg = json.loads((ROOT / 'data/full-v4/resolved_config.json').read_text())
    download = ('from huggingface_hub import snapshot_download; '
        f"snapshot_download({cfg['model']!r}, revision={cfg['revision']!r}, "
        "allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])")
    image = (modal.Image.debian_slim(python_version='3.12').apt_install('build-essential')
        .pip_install_from_requirements(str(ROOT / 'requirements-colab.txt'))
        .pip_install('vllm==0.10.1.1')
        .env({'HF_HOME': '/opt/hf-cache', 'HF_HUB_DISABLE_TELEMETRY': '1'})
        .run_commands('python -c ' + shlex.quote(download))
        .add_local_file(ROOT / 'scripts/run_modal_benchmark.py', '/root/run_modal_benchmark.py')
        .add_local_file(ROOT / 'scripts/run_modal_experiment.py', '/root/run_modal_experiment.py')
        .add_local_file(archive, '/source.zip')
        .add_local_file(Path(__file__), '/root/redraw_traces.py')
        .add_local_file(HERE / 'trace-redraw-policy.json', '/redraw-policy.json'))
    volume = modal.Volume.from_name(manifest['volume'])
    app = modal.App('gasp-source-trace-correction')
    remote = app.function(image=image, gpu='A100-80GB', cpu=4, memory=(65536, 131072),
        volumes={'/persist': volume}, timeout=limit, startup_timeout=900, retries=0,
        min_containers=0, max_containers=1, scaledown_window=2,
        single_use_containers=True, include_source=False)(redraw_worker)
    attempt = 'trace-redraw-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    launch = {'attempt_id': attempt, 'stage': 'trace-redraw', 'condition': None,
        'max_seconds': limit, 'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / (attempt + '.launch.json')).write_text(json.dumps(launch, indent=2))
    with modal.enable_output(), app.run(detach=True):
        data = remote.remote('trace-redraw', None, manifest['run_id'], attempt,
            manifest['source_sha256'], limit, 35, False, manifest['prior_gpu_seconds'],
            manifest['total_gpu_seconds'], manifest['volume'])
        (HERE / (attempt + '.zip')).write_bytes(data)
        with zipfile.ZipFile(io.BytesIO(data)) as result:
            safe_extract(result, HERE / 'results')
        report = json.loads((HERE / 'results/attempts' / attempt / 'worker.json').read_text())
        print(json.dumps(report, indent=2), flush=True)
        if not report['success']:
            raise RuntimeError('Trace redraw failed; inspect durable report')
    lock.close()


if __name__ == '__main__':
    main()
