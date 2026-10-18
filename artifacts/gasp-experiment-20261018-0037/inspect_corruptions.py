"""Review every changed span against the already audited clean source."""
import argparse,difflib,json,re,sys
from decimal import Decimal
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]/'src'))
from gasp.eval_corpus import response_pollution
parser=argparse.ArgumentParser();parser.add_argument('start',type=int);parser.add_argument('end',type=int);parser.add_argument('--full',action='store_true');parser.add_argument('--corpus-directory',default='corpus-v2');a=parser.parse_args()
rows=[json.loads(l) for l in (P/'results/prepare-v2/eval-contexts.jsonl').read_text().splitlines()]
last=None
for i in range(a.start,min(a.end,len(rows))):
 f=P/a.corpus_directory/'held-out-corruptions.jsonl.responses'/f'{i:06d}.json'
 if not f.exists(): print('PENDING',i);break
 row=rows[i];saved=json.loads(f.read_text());polluted,_=response_pollution(saved['response'],'gpt-6.1-sol')
 if last!=row['question_id']:print(json.dumps({'Q':row['question_id'],'question':row['question'],'gold':row['answer']}));last=row['question_id']
 clean=row['clean'].split();new=polluted.split();diff=[]
 for tag,l,r,x,y in difflib.SequenceMatcher(a=clean,b=new,autojunk=False).get_opcodes():
  if tag!='equal':diff.append({'before':' '.join(clean[max(0,l-6):l]),'old':' '.join(clean[l:r]),'new':' '.join(new[x:y]),'after':' '.join(clean[r:r+6])})
 out={'i':i,'alpha':row['alpha'],'changes':diff}
 values = [Decimal(t.replace(',', '')) for t in re.findall(r'(?<!\w)-?\d[\d,]*(?:\.\d+)?', polluted)]
 if a.full or Decimal(row['answer'].replace(',', '')) in values:out['polluted']=polluted
 print(json.dumps(out))
