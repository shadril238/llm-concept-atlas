import json
from pathlib import Path
data = json.loads(Path('src/data/training-results.json').read_text())['embedding']
for row in data['conditions']:
    training = row['training']
    assert training['reload_verified'] and training['updated']
    changed = sum(group['changed_tensors'] for group in training['parameters'].values())
    print(row['kind'], 'selected step:', training['selected_step'], 'changed tensors:', changed)
    print('selection:', training['selection'])
print('Rerun embedding-training for new optimization; this replays recorded evidence.')
