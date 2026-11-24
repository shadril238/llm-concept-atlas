import json
from pathlib import Path
data = json.loads(Path('src/data/training-results.json').read_text())['augmentation']
print('silver candidates:', len(data['silver']))
print('teacher diagnostic disagreements:', sum(r['disagreement'] for r in data['diagnostics']))
for row in data['students']:
    print(row['condition'], 'test MSE:', round(row['similarity']['mse'], 4))
print('Silver scores are continuous model predictions, not gold targets.')
