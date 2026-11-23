import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
row = next(r for r in data['grounded']['rows'] if r['id']=='eval-rewrite' and r['variant']=='rewrite')
print('original:',row['question'])
print('actual rewrite:',row['plan']['calls'][0]['raw'])
print('actual search:',row['plan']['searches'][0]['query'])
assert row['plan']['calls'][0]['raw'].strip()==row['plan']['searches'][0]['query']
print('raw answer:',row['answer_call']['raw'])
