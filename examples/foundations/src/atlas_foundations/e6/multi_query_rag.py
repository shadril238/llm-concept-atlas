import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
row = next(r for r in data['grounded']['rows'] if r['id']=='eval-compare' and r['variant']=='multi-query')
scores = {}
for search in row['plan']['searches']:
    print('actual query:',search['query'])
    for rank,hit in enumerate(search['hits'],1):scores[hit['id']]=scores.get(hit['id'],0)+1/(60+rank)
ranking = sorted(scores,key=lambda id:(-scores[id],id))
assert ranking==[h['id'] for h in row['plan']['hits']]
print('fused first IDs:',ranking[:4])
