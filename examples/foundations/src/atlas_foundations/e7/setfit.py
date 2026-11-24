import json
from pathlib import Path
data = json.loads(Path('src/data/training-results.json').read_text())['classification']
for row in data['runs']:
    if row['method'] == 'setfit':
        assert row['initial_sha256'] != row['updated_sha256']
        assert row['first_query_gradient_norm'] > 0
        print('seed:', row['seed'], 'unique labeled reviews:', len(row['train_ids']))
        print('saved body/head reload:', row['reload_verified'])
print('Pair adaptation and classifier fitting are separate verified stages.')
