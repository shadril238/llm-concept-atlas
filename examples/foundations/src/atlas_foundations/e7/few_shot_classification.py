import json
from pathlib import Path
from statistics import mean, pstdev
data = json.loads(Path('src/data/training-results.json').read_text())['classification']
for method in ['setfit', 'frozen-sentence-logistic']:
    scores = [r['test']['macro_f1'] for r in data['runs'] if r['method'] == method]
    print(method, 'selections:', len(scores), 'mean:', round(mean(scores), 4), 'population SD:', round(pstdev(scores), 4))
print('Three fixed selections describe variability; they are not a confidence interval.')
