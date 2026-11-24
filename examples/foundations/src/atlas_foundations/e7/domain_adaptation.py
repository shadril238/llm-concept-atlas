import json
from pathlib import Path
data = json.loads(Path('src/data/training-results.json').read_text())['denoising']
print('baseline test MRR:', round(data['baseline']['retrieval']['mrr'], 4))
print('TSDAE-only test MRR:', round(data['tsdae']['retrieval']['mrr'], 4))
for row in data['adaptation']:
    print(row['condition'], 'test MRR:', round(row['retrieval']['mrr'], 4))
print('reconstruction-loss / retrieval disagreement intervals:', len(data['counterexamples']))
