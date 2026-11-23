import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
row = next(r for r in data['grounded']['rows'] if r['id']=='eval-hop' and r['variant']=='multi-hop')
for step in row['plan']['trace']:
    print('evidence:',step.get('evidence_ids'),'next query:',step.get('next_query'))
print('actual searches:',len(row['plan']['searches']))
print('stop:',row['plan']['stop_reason'])
assert len(row['plan']['searches'])<=3
