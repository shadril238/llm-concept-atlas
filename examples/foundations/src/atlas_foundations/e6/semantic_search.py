import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
query = next(q for q in data['retrieval']['queries'] if q['id']=='eval-access')
print(query['text'])
for method in ['lexical','dense']:
    row = next(r for r in data['retrieval']['rows'] if r['id']==query['id'] and r['method']==method)
    print(method, row['document_ranking'][:3])
