import json
from pathlib import Path
data = json.loads(Path('src/data/training-results.json').read_text())['classification']
row = next(r for r in data['runs'] if r['seed'] == 11 and r['method'] == 'bert-partial')
for name, group in row['parameters'].items():
    if group['trainable'] == 0:
        assert group['changed_tensors'] == 0
    print(name, 'trainable:', group['trainable'], 'changed tensors:', group['changed_tensors'])
print('exact frozen-weight check:', row['frozen_verified'])
