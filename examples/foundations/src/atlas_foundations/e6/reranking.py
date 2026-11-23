import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
for method in ['dense','reranked']:
    row = next(r for r in data['retrieval']['rows'] if r['id']=='eval-hop' and r['method']==method)
    print(method,[(h['id'],round(h['score'],3)) for h in row['hits'][:3]])
print('Pair scores are raw logits; only six dense candidate chunks are reranked.')
