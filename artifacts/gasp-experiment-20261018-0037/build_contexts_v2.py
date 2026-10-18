"""Publish a new local cache version from explicitly reviewed raw trace draws."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from gasp.data import read_cache, write_cache, sha256, assert_disjoint
from gasp.prompts import POLLUTER

rows, manifest = read_cache(HERE / 'results/prepare/eval-contexts.jsonl', 'eval_contexts')
review = json.loads((HERE / 'replacement-trace-review.json').read_text())
expected = json.loads((HERE / 'source-trace-audit-summary.json').read_text())
assert manifest['sha256'] == expected['input_contexts_sha256']
assert {r['question_id'] for r in review} == set(expected['replacement_question_ids'])
assert len(review) == 12 and all(r['verdict'] == 'pass' for r in review)
lookup = {}
for verdict in review:
    seed = verdict['seed']
    path = HERE / 'results/trace-redraw' / f'seed-{seed}' / 'candidates.jsonl'
    candidates = [json.loads(l) for l in path.read_text().splitlines()]
    candidates = [r for r in candidates if r['question_id'] == verdict['question_id']
                  and r['candidate_index'] == verdict['candidate_index']]
    assert len(candidates) == 1
    sample = candidates[0]
    assert sample['numeric_correct'] and sample['reasoning_nonempty']
    assert verdict['candidate_file_sha256'] == sha256(path)
    lookup[verdict['question_id']] = (sample, verdict)
updated = deepcopy(rows)
for row in updated:
    if row['question_id'] not in lookup:
        continue
    sample, verdict = lookup[row['question_id']]
    assert (sample['question'], sample['answer']) == (row['question'], row['answer'])
    cut, = [c for c in sample['cuts'] if c['alpha'] == row['alpha']]
    row.update(prefix=cut['prefix'], clean=cut['clean'], clean_trace=sample['text'],
        original_clean_attempts=row['clean_attempts'], clean_attempts=sample['candidate_index'] + 1,
        trace_redraw_seed=verdict['seed'], trace_redraw_candidate=sample['candidate_index'])
for old, new in zip(rows, updated):
    for key in ('id','question_id','question','answer','split','alpha','filter_scores','initial_clean_solved'):
        assert old[key] == new[key]
    if old['question_id'] not in lookup:
        assert old == new
assert len(updated) == 400 and len({r['question_id'] for r in updated}) == 100
train, _ = read_cache(HERE / 'results/prepare/train.jsonl', 'train')
assert_disjoint(train, updated)
provenance = deepcopy(manifest['provenance'])
provenance['protocol'] += '+semantic-trace-correction-v2'
provenance['subset_selection'] += '; same question subset; semantic clean-trace redraw for12 questions before any model scores'
provenance['trace_correction'] = {'original_contexts_sha256': manifest['sha256'],
    'review_sha256': sha256(HERE / 'replacement-trace-review.json'),
    'review': review, 'source_full_audit_sha256': sha256(HERE / 'source-trace-audit.jsonl'),
    'draws': {str(seed): json.loads((HERE / 'results/trace-redraw' / f'seed-{seed}' / 'result.json').read_text())
              for seed in {v['seed'] for v in review}},
    'paper_deviation': 'Additional semantic acceptance enforces the intended sound clean continuation; not a claimed exact author filtering procedure. Online training is unchanged.'}
version = HERE / 'results/prepare-v2/eval-contexts.jsonl'
result = write_cache(version, updated, provenance)
read_cache(version, 'eval_contexts')
response_dir = HERE / 'corpus-v2/held-out-corruptions.jsonl.responses'
response_dir.mkdir(parents=True, exist_ok=False)
settings = {'model':'gpt-6.1-sol','reasoning':{'effort':'low'},'max_output_tokens':4096,'store':False}
job = {'contexts_sha256':result['sha256'],'settings':settings,
       'polluter_prompt_sha256':hashlib.sha256(POLLUTER.encode()).hexdigest()}
(response_dir / 'job.json').write_text(json.dumps(job, indent=2)+'\n')
reuse = []
invalid = set(json.loads((HERE / 'corpus-v2-audit-policy.json').read_text())['initial_invalid_indices'])
for index, row in enumerate(updated):
    previous = HERE / 'corpus/held-out-corruptions.jsonl.responses' / f'{index:06d}.json'
    if not previous.exists() or index in invalid:
        continue
    saved = json.loads(previous.read_text())
    payload = settings | {'input': [{'role':'system','content':'You are a helpful assistant.'},
        {'role':'user','content':POLLUTER.format(q=row['question'],prefix=row['prefix'],continuation=row['clean'])}]}
    if saved['context_id'] == row['id'] and saved['request'] == payload:
        shutil.copy2(previous, response_dir / previous.name)
        reuse.append({'index':index,'id':row['id'],'original':str(previous),'response_sha256':sha256(previous)})
(HERE / 'corpus-v2/response-reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
(HERE / 'context-v2-validation.json').write_text(json.dumps({'verdict':'pass',
    'original_sha256':manifest['sha256'],'v2_sha256':result['sha256'],'questions':100,
    'contexts':400,'replaced_questions':12,'unchanged_questions':88,
    'train_disjoint':True,'unchanged_ids_order_questions_answers_alphas_and_K4_evidence':True,
    'reused_api_responses':len(reuse)},indent=2)+'\n')
print(json.dumps({'contexts':result['sha256'],'reused_responses':len(reuse)}))
