"""Persist human/assistant adjudications; never infer semantic acceptance."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent

def record(entries):
    path=P/'corpus-v2/semantic-review.jsonl'
    existing={r['index']:r for r in map(json.loads,path.read_text().splitlines())}
    contexts=list(map(json.loads,(P/'results/prepare-v2/eval-contexts.jsonl').read_text().splitlines()))
    for key,value in entries.items():
        i=int(key)
        assert i not in existing, f'Already reviewed: {i}'
        value={'mathematical_error':value} if isinstance(value,str) else value
        response=P/'corpus-v2/held-out-corruptions.jsonl.responses'/f'{i:06d}.json'
        existing[i]={'index':i,'id':contexts[i]['id'],'status':'pass',
            'new_final_answer_leak':False,'meta_instruction':False,**value,
            'response_sha256':hashlib.sha256(response.read_bytes()).hexdigest()}
    path.write_text(''.join(json.dumps(existing[i])+'\n' for i in sorted(existing)))
    print('Reviewed',len(existing),'of400')

if __name__=='__main__':record(json.load(sys.stdin))
