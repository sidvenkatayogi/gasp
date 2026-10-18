"""Required predeclared strict-corpus sensitivity, using saved paired predictions.

Reconstructs the full-set estimates and intervals first and checks them against
our tested primary analysis before applying the fixed quality exclusions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS = ('corrupted_accuracy', 'clean_selected_subset_accuracy')
PAIRS = (('split','unified'), ('unified','initial'), ('split','initial'))


def calculate(records, samples, seed):
    arrays = {name: {key: np.array([r[name][key] for r in records]) for key in METRICS}
              for name in ('initial','unified','split')}
    indices = np.random.default_rng(seed).integers(0, len(records), size=(samples,len(records)))
    scores = {name:{key:float(a.mean()) for key,a in values.items()} for name,values in arrays.items()}
    differences = {}
    for left,right in PAIRS:
        differences[left+'_minus_'+right] = {}
        for key in METRICS:
            delta = arrays[left][key]-arrays[right][key]
            differences[left+'_minus_'+right][key] = {'estimate':float(delta.mean()),
                'ci95':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()}
    return {'scores':scores,'paired_differences':differences}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison',type=Path)
    parser.add_argument('review',type=Path)
    parser.add_argument('eval_manifest',type=Path)
    args=parser.parse_args()
    review=json.loads(args.review.read_text())
    manifest=json.loads(args.eval_manifest.read_text())
    primary=json.loads((args.comparison/'summary.json').read_text())
    records=list(map(json.loads,(args.comparison/'question_scores.jsonl').read_text().splitlines()))
    assert manifest['provenance']['input_sha256']==review['corpus_sha256']
    assert primary['identity']['eval_sha256']==manifest['sha256']
    samples,seed=primary['bootstrap']['samples'],primary['bootstrap']['seed']
    full=calculate(records,samples,seed)
    for pair,values in full['paired_differences'].items():
        for metric,values in values.items():
            expected=primary['paired_differences'][pair][metric]
            assert np.allclose([values['estimate'],*values['ci95']],
                               [expected['estimate'],*expected['ci95']],rtol=0,atol=1e-12)
    policy=review['required_quality_sensitivity']
    excluded=set(policy['excluded_question_ids'])
    assert excluded <= {r['question_id'] for r in records}
    retained=[r for r in records if r['question_id'] not in excluded]
    assert len(retained)==policy['questions'] and 4*len(retained)==policy['corruptions']
    result=calculate(retained,samples,seed) | {'analysis':'predeclared_quality_sensitivity',
        'n_questions':len(retained),'n_corruptions':4*len(retained),
        'excluded_question_ids':sorted(excluded),'bootstrap':primary['bootstrap'],
        'training_seed':primary['training_seed'],'full_set_reconstruction_matches_primary':True,
        'corpus_sha256':review['corpus_sha256'],
        'frozen_review_sha256':hashlib.sha256(args.review.read_bytes()).hexdigest(),
        'interpretation':'Exclusions use pre-outcome corpus-quality findings. Report alongside the primary400-row result regardless of direction. Unmodified prefixes may still contain answers; one training seed does not establish reproducibility.'}
    path=args.comparison/'quality-sensitivity.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
