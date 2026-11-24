import json
from pathlib import Path
data = json.loads(Path('src/data/training-results.json').read_text())['classification']
for row in data['runs']:
    if row['method'] == 'bert-full':
        print('seed:', row['seed'], 'train reviews:', len(row['train_ids']))
        print('selected step:', row['selected_step'], 'test macro-F1:', round(row['test']['macro_f1'], 4))
print('Each reported model used validation-only checkpoint selection.')
