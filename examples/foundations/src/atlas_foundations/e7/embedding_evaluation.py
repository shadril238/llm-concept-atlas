import json
from pathlib import Path
data = json.loads(Path('src/data/training-results.json').read_text())['embedding']
for row in data['conditions']:
    predictions = row['retrieval']['rows']
    mrr = sum(1 / p['rank'] for p in predictions) / len(predictions)
    assert abs(mrr - row['retrieval']['mrr']) < 1e-12
    print(row['kind'], 'test queries:', len(predictions), 'MRR:', round(mrr, 4))
print('scope: actual recorded task; complete MTEB/STSB not executed')
